"""VRAM budgeting shared by both reranker profiles.

Pure byte arithmetic with no torch, FastAPI or model imports, so the fast
test suite can exercise it without a CUDA build. The guarded-call helper
added in Task 4 lives here too and takes its torch module by parameter.
"""

_DTYPE_SIZES = {"fp16": 2, "bf16": 2, "fp32": 4}

# Unknown precision must over-estimate: a too-large byte count yields a
# smaller batch, which is safe. A too-small one yields an OOM.
_DEFAULT_DTYPE_BYTES = 4


def dtype_size_bytes(precision: str) -> int:
    """Bytes per element for a serving precision name."""
    return _DTYPE_SIZES.get(precision, _DEFAULT_DTYPE_BYTES)


def logits_bytes_per_token(vocab_size: int, dtype_bytes: int) -> int:
    """Bytes the LM head writes per sequence position.

    This is the dominant term in the reranker forward: the head projects to
    the full vocabulary for every position it is asked to compute.
    """
    return int(vocab_size) * int(dtype_bytes)


def usable_headroom_bytes(
    device_free_bytes: int, reserved_bytes: int, allocated_bytes: int
) -> int:
    """Bytes a new allocation can actually use.

    Device-free alone is misleading: the caching allocator may already have
    reserved most of the device, reporting zero free while still holding a
    large unused pool. Usable headroom is that unused pool plus whatever the
    device has left unmapped.
    """
    pool_free = max(0, int(reserved_bytes) - int(allocated_bytes))
    return pool_free + max(0, int(device_free_bytes))


def safe_max_batch_tokens(
    headroom_bytes: int,
    reserve_bytes: int,
    bytes_per_token: int,
    configured_cap: int,
) -> int:
    """Largest per-batch token budget that fits, never above the configured cap.

    Acts only as a ceiling: it lowers a configured value, never raises one.
    """
    if bytes_per_token <= 0:
        return configured_cap
    usable = int(headroom_bytes) - int(reserve_bytes)
    if usable <= 0:
        return 1
    derived = usable // int(bytes_per_token)
    return max(1, min(int(configured_cap), int(derived)))


# Substrings identifying an allocation failure that does not arrive as
# torch.OutOfMemoryError. Under expandable_segments the failure surfaces
# from cuMemCreate as an internal assert instead - an open upstream bug
# (pytorch/pytorch#166234). Matching is deliberately narrow; the negative
# test in the suite is what keeps it from swallowing unrelated errors.
_ALLOCATOR_EXHAUSTION_SIGNATURES = (
    "cuda out of memory",
    "!handles_.at(i)",
    "cuda driver error: device not ready",
    "cudacachingallocator.cpp",
)


class VramExhausted(RuntimeError):
    """A batch could not be allocated on the device.

    Carries the numbers a caller needs to react - split the batch, or reject
    the request with an actionable message - without holding any reference
    to the original exception or the tensors its frames captured.
    """

    def __init__(
        self,
        *,
        batch_items: int,
        batch_tokens: int,
        estimated_bytes: int,
        original: str,
    ):
        """Record the batch that failed to fit and the driver's own message.

        Args:
            batch_items: Number of items in the batch that could not be allocated.
            batch_tokens: Total token count across those items.
            estimated_bytes: Pre-flight estimate of the allocation, in bytes.
            original: The driver/allocator message, already reduced to a string
                so no traceback frame (and no tensor it references) is retained.
        """
        super().__init__(
            f"VRAM exhausted for a batch of {batch_items} item(s), "
            f"{batch_tokens} token(s), ~{estimated_bytes} byte(s): {original}"
        )
        self.batch_items = batch_items
        self.batch_tokens = batch_tokens
        self.estimated_bytes = estimated_bytes
        self.original = original


def is_vram_exhaustion(exc: BaseException, torch_module=None) -> bool:
    """True when ``exc`` represents a device allocation failure."""
    oom_type = getattr(torch_module, "OutOfMemoryError", None)
    if isinstance(oom_type, type) and isinstance(exc, oom_type):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return any(sig in message for sig in _ALLOCATOR_EXHAUSTION_SIGNATURES)


def release_cuda_cache(torch_module) -> None:
    """Return unused cached blocks to the driver. Failure path only."""
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return
    try:
        cuda.empty_cache()
    except Exception:  # pragma: no cover - never mask the original failure
        pass


def run_guarded(
    operation,
    *,
    batch_items: int,
    batch_tokens: int,
    bytes_per_token: int,
    torch_module,
):
    """Run ``operation`` and convert device exhaustion into ``VramExhausted``.

    ``operation`` must build its device tensors in its own frame. That frame
    is only reachable through the original exception's traceback, which this
    function deliberately drops, so the tensors become collectable as soon as
    the failure is raised.

    The raise happens *after* the except block so ``sys.exc_info()`` is
    already cleared and no ``__context__`` chain is attached. ``raise ...
    from None`` does not achieve this: it suppresses the display of the
    context but still populates ``__context__``, keeping the original
    traceback - and the batch - alive.
    """
    failure = None
    try:
        return operation()
    except BaseException as exc:
        if not is_vram_exhaustion(exc, torch_module):
            raise
        failure = VramExhausted(
            batch_items=batch_items,
            batch_tokens=batch_tokens,
            estimated_bytes=int(batch_tokens) * int(bytes_per_token),
            original=f"{type(exc).__name__}: {exc}",
        )

    release_cuda_cache(torch_module)
    raise failure
