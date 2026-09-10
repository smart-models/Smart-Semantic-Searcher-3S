"""Tests for the embedder and reranker wrappers, and the shared test harness.

Also the home of ``load_server`` and ``install_support_fakes``, which every
other module here uses to import ``smart-embedder.py`` with fakes standing
in for the model libraries.
"""

import asyncio
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SERVER_PATH = Path(__file__).resolve().parents[1] / "smart-embedder.py"
if str(SERVER_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SERVER_PATH.parent))

import basic_autotune as autotune  # noqa: E402
from model_revisions import pinned_model_id  # noqa: E402


class FakeInnerBgeModel:
    """Placeholder for the inner torch module a BGE-M3 wrapper exposes."""

    pass


class FakeBGEM3FlagModel:
    """BGE-M3 stand-in recording its construction and every encode call."""

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.model = FakeInnerBgeModel()
        self.tokenizer = FakeTokenizer()
        self.calls = []

    def encode(self, *args, **kwargs):
        if getattr(self.model, "raise_on_encode", False):
            raise RuntimeError("compiled forward failed")
        self.calls.append((args, kwargs))
        return {"dense_vecs": []}


class FakeFlagReranker:
    """BGE reranker stand-in recording the pairs it was asked to score."""

    calls = []

    def __init__(self, model_name, use_fp16=False):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.tokenizer = FakeTokenizer()

    def compute_score(self, pairs, normalize=False, **kwargs):
        self.calls.append((pairs, normalize, kwargs))
        return [0.25 for _ in pairs]


class FakeTokenizer:
    """Tokenizer stand-in with predictable ids, lengths and padding."""

    def __init__(self):
        self.deprecation_warnings = {}
        self.encoded = []
        self.decoded = []
        self.calls = []
        self.pad_token = None
        self.eos_token = "<eos>"
        self.bos_token_id = 0
        self.eos_token_id = 2

    def convert_tokens_to_ids(self, token):
        return {"no": 0, "yes": 1}[token]

    def encode(self, text, add_special_tokens=False):
        if text.startswith("tok:"):
            count = int(text.split(":", 1)[1])
            return list(range(count))
        return [10, 11]

    def decode(self, token_ids, skip_special_tokens=True):
        self.decoded.append(list(token_ids))
        return "tok:" + str(len(token_ids))

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    def __call__(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        self.encoded.extend(pairs)
        return {
            "input_ids": [[21, 22] for _ in pairs],
            "attention_mask": [[1, 1] for _ in pairs],
        }

    def pad(self, inputs, **kwargs):
        if self.pad_token is None:
            raise ValueError(
                "Asking to pad but the tokenizer does not have a padding token"
            )
        return {"input_ids": FakeTensor(inputs["input_ids"])}


class IndexTaggingQwenTokenizer(FakeTokenizer):
    """Encodes each pair's passage index into its input_ids.

    Tests that fake ``_forward_qwen_chunk`` need scores that differ per item
    to catch a bug that mis-assigns a score to the wrong passage (e.g. a
    swapped split/requeue order). A uniform stand-in score like ``0.5``
    would pass just as happily against such a bug, so this tokenizer makes
    each item's own content recoverable from its ``input_ids``.
    """

    def __call__(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        self.encoded.extend(pairs)
        input_ids = []
        for pair in pairs:
            document = pair.split("<Document>: ", 1)[1]
            index = int(document[1:])
            input_ids.append([1000 + index])
        return {"input_ids": input_ids}


def _qwen_pair_index(item):
    """Recover the passage index embedded by IndexTaggingQwenTokenizer."""
    return next(token for token in item if token >= 1000) - 1000


class FakeAutoTokenizer:
    """Stand-in for transformers' AutoTokenizer factory."""

    @classmethod
    def from_pretrained(cls, model_name, padding_side="left", **kwargs):
        return FakeTokenizer()


class FakeCausalLM:
    """Causal LM stand-in returning fixed logits for reranker scoring."""

    def __init__(self):
        self.config = types.SimpleNamespace(use_cache=True)
        self.calls = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **inputs):
        self.calls.append(dict(inputs))
        return types.SimpleNamespace(logits=FakeLogits())


class FakeAutoModelForCausalLM:
    """Stand-in for transformers' AutoModelForCausalLM factory."""

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return FakeCausalLM()


class FakeEmbeddingModel:
    """Embedding model stand-in returning a fixed hidden state."""

    def __init__(self, dense_size=1024):
        self.dense_size = dense_size
        self.last_dense_tensor = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **inputs):
        self.last_dense_tensor = FakeDenseTensor(self.dense_size)
        return types.SimpleNamespace(
            last_hidden_state=FakeEmbeddingTensor(self.last_dense_tensor)
        )


class FakeAutoModel:
    """Stand-in for transformers' AutoModel factory."""

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        if os.getenv("FAKE_QWEN_DENSE_LOAD_ERROR") == "1":
            raise RuntimeError("qwen dense load failed")
        dense_size = int(os.getenv("FAKE_QWEN_DENSE_DIM", "1024"))
        return FakeEmbeddingModel(dense_size)


class FakeTensor:
    """Minimal tensor stand-in that survives a device move."""

    def __init__(self, value):
        self.value = value

    def to(self, device):
        return self


class FakeEmbeddingTensor:
    """Hidden-state stand-in yielding its dense tensor on any slice."""

    shape = (1, 1, 1024)

    def __init__(self, dense_tensor):
        self.dense_tensor = dense_tensor

    def __getitem__(self, key):
        return self.dense_tensor


class FakeDenseTensor:
    """Dense tensor stand-in convertible to plain nested lists."""

    def __init__(self, dense_size=1024):
        self.dense_size = dense_size
        self.shape = (1, dense_size)
        self.normalized = False

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return [[1.0] * self.dense_size]


class FakeLogits:
    """Logits stand-in that survives any slicing unchanged."""

    def __getitem__(self, key):
        return self


class FakeCudaEvent:
    """Stand-in for ``torch.cuda.Event``, used by the device-phase timers."""

    def __init__(self, enable_timing=True):
        self.enable_timing = enable_timing

    def record(self):
        return None

    def synchronize(self):
        return None

    def elapsed_time(self, other):
        return 0.0


class FakeCudaDeviceProperties:
    """Device-properties stand-in reporting a fixed 8 GiB of VRAM."""

    total_memory = 8 * 1024**3


class FakeTorchModule(types.ModuleType):
    """Stand-in torch module exposing only what the server imports."""

    float16 = "float16"
    float32 = "float32"
    Tensor = FakeTensor

    def __init__(self, cuda_available=False):
        super().__init__("torch")
        self.cuda_available = cuda_available
        self.memory_allocated_value = 123
        self.memory_reserved_value = 456
        self.mem_get_info_value = (789, 1000)
        self.cuda = types.SimpleNamespace(
            is_available=lambda: self.cuda_available,
            device_count=lambda: 0,
            memory_allocated=lambda: self.memory_allocated_value,
            memory_reserved=lambda: self.memory_reserved_value,
            max_memory_allocated=lambda: self.memory_allocated_value,
            max_memory_reserved=lambda: self.memory_reserved_value,
            reset_peak_memory_stats=lambda: None,
            mem_get_info=lambda: self.mem_get_info_value,
            get_device_name=lambda index=0: "Fake GPU",
            get_device_properties=lambda index=0: FakeCudaDeviceProperties(),
            Event=lambda enable_timing=True: FakeCudaEvent(enable_timing),
        )
        self.nn = types.SimpleNamespace(
            functional=types.SimpleNamespace(
                log_softmax=self._log_softmax,
                normalize=self._normalize,
            )
        )

    class _NoGrad:
        """No-op context manager standing in for torch.no_grad."""

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def no_grad(self):
        return self._NoGrad()

    def inference_mode(self):
        return self._NoGrad()

    def stack(self, values, dim=1):
        return FakeProbability()

    def set_num_threads(self, value):
        self.num_threads = value

    def _log_softmax(self, values, dim=1):
        return FakeProbability()

    def _normalize(self, value, p=2, dim=1):
        value.normalized = True
        return value


class FakeProbability:
    """Probability tensor stand-in convertible to a plain score list."""

    def __getitem__(self, key):
        return self

    def exp(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [0.75]


class FakeFastAPI:
    """FastAPI stand-in recording the routes and middleware registered."""

    def __init__(self, *args, **kwargs):
        self.title = kwargs.get("title")
        self.version = kwargs.get("version")
        self.lifespan = kwargs.get("lifespan")
        self.docs_url = kwargs.get("docs_url", "/docs")
        self.redoc_url = kwargs.get("redoc_url", "/redoc")
        self.openapi_url = kwargs.get("openapi_url", "/openapi.json")
        self.get_calls = []
        self.post_calls = []
        self.added_middleware = []

    def get(self, *args, **kwargs):
        self.get_calls.append((args, kwargs))
        return self._decorator

    def post(self, *args, **kwargs):
        self.post_calls.append((args, kwargs))
        return self._decorator

    def middleware(self, *args, **kwargs):
        return self._decorator

    def add_middleware(self, middleware_class, **options):
        # Pure ASGI layers are registered this way rather than through the
        # `middleware` decorator: the body-size cap has to wrap `receive`.
        self.added_middleware.append((middleware_class, options))

    @staticmethod
    def _decorator(func):
        return func


class FakeHTTPException(Exception):
    """Stand-in for FastAPI's HTTPException, carrying status and detail."""

    def __init__(self, status_code=None, detail=None, headers=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.headers = headers


class FakeHTTPBearer:
    """Stand-in for FastAPI's HTTPBearer security scheme."""

    def __init__(self, *args, **kwargs):
        pass


class FakeBaseModel:
    """Stand-in base model accepting and exposing arbitrary fields."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeMetric:
    """Prometheus metric stand-in recording every observation made on it."""

    instances = []

    def __init__(self):
        self.last_info = None
        self.last_set = None
        self.label_calls = []
        self.increment_calls = []
        FakeMetric.instances.append(self)

    def labels(self, *args, **kwargs):
        self.label_calls.append((args, kwargs))
        return self

    def inc(self, *args, **kwargs):
        self.increment_calls.append(args[0] if args else 1)
        return None

    def dec(self, *args, **kwargs):
        return None

    def set(self, *args, **kwargs):
        self.last_set = args[0] if args else None
        return None

    def observe(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        self.last_info = args[0] if args else kwargs
        return None

    def time(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def identity_decorator(*args, **kwargs):
    def decorate(func):
        return func

    return decorate


class FakeORJSONResponse:
    """Response stand-in rendering payloads without orjson installed."""

    media_type = "application/json"

    def __init__(
        self,
        content=None,
        status_code=200,
        headers=None,
        media_type=None,
        background=None,
    ):
        self.content = content
        self.status_code = status_code
        self.headers = headers
        self.media_type = media_type or self.media_type
        self.background = background

    def render(self, content):
        return b"{}"


def install_support_fakes():
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.ndarray = list
    fake_numpy.array = lambda value: value
    fake_numpy.maximum = lambda value, minimum: value
    fake_numpy.linalg = types.SimpleNamespace(norm=lambda *args, **kwargs: 1)

    def fake_concatenate(arrays, axis=0):
        if axis != 0:
            raise NotImplementedError("fake concatenate only supports axis=0")
        combined = []
        for array in arrays:
            combined.extend(array)
        return combined

    fake_numpy.concatenate = fake_concatenate

    fake_fastapi = types.ModuleType("fastapi")
    fake_fastapi.Depends = lambda dependency=None: dependency
    fake_fastapi.FastAPI = FakeFastAPI
    fake_fastapi.HTTPException = FakeHTTPException
    fake_fastapi.Request = object

    fake_responses = types.ModuleType("fastapi.responses")
    fake_responses.JSONResponse = object
    fake_responses.Response = object
    fake_responses.ORJSONResponse = FakeORJSONResponse

    fake_security = types.ModuleType("fastapi.security")
    fake_security.HTTPAuthorizationCredentials = object
    fake_security.HTTPBearer = FakeHTTPBearer

    fake_prometheus = types.ModuleType("prometheus_client")
    fake_prometheus.CONTENT_TYPE_LATEST = "text/plain"
    fake_prometheus.Counter = lambda *args, **kwargs: FakeMetric()
    fake_prometheus.Gauge = lambda *args, **kwargs: FakeMetric()
    fake_prometheus.Histogram = lambda *args, **kwargs: FakeMetric()
    fake_prometheus.Info = lambda *args, **kwargs: FakeMetric()
    fake_prometheus.generate_latest = lambda: b""

    fake_pydantic = types.ModuleType("pydantic")
    fake_pydantic.BaseModel = FakeBaseModel
    fake_pydantic.Field = lambda default=None, **kwargs: (
        kwargs["default_factory"]() if "default_factory" in kwargs else default
    )
    fake_pydantic.field_validator = identity_decorator
    fake_pydantic.model_validator = identity_decorator

    fake_starlette_status = types.ModuleType("starlette.status")
    fake_starlette_status.HTTP_504_GATEWAY_TIMEOUT = 504

    fake_huggingface_hub = types.ModuleType("huggingface_hub")

    def missing_snapshot(*args, **kwargs):
        if kwargs.get("local_files_only"):
            raise OSError("not cached in unit test")
        return f"/downloaded/{kwargs['repo_id']}/{kwargs['revision']}"

    fake_huggingface_hub.snapshot_download = missing_snapshot

    return {
        "numpy": fake_numpy,
        "fastapi": fake_fastapi,
        "fastapi.responses": fake_responses,
        "fastapi.security": fake_security,
        "prometheus_client": fake_prometheus,
        "pydantic": fake_pydantic,
        "starlette.status": fake_starlette_status,
        "huggingface_hub": fake_huggingface_hub,
    }


def load_server(env=None, *, cuda_available=False, stub_model_resolution=True):
    module_name = "smart_embedder_under_test"
    sys.modules.pop(module_name, None)
    FakeMetric.instances = []

    fake_flag_embedding = types.ModuleType("FlagEmbedding")
    fake_flag_embedding.BGEM3FlagModel = FakeBGEM3FlagModel
    fake_flag_embedding.FlagReranker = FakeFlagReranker

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: types.SimpleNamespace(
            multilingual_encoder_checkpoint="BAAI/bge-m3-unsupervised",
            lsr_encoder_checkpoint="naver/splade-v3",
        )
    )
    fake_transformers.AutoModel = FakeAutoModel
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM

    modules = {
        **install_support_fakes(),
        "FlagEmbedding": fake_flag_embedding,
        "transformers": fake_transformers,
        "torch": FakeTorchModule(cuda_available=cuda_available),
    }

    test_env = {
        "SMART_EMBEDDER_ALLOW_UNAUTHENTICATED_LOCAL": "true",
        **(env or {}),
    }
    with (
        patch.dict(sys.modules, modules),
        patch.dict(os.environ, test_env, clear=False),
    ):
        spec = importlib.util.spec_from_file_location(module_name, SERVER_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        if stub_model_resolution:
            module._cache_first_model_path = lambda model_name: (
                f"/unit-test-cache/{model_name.replace('/', '--')}"
            )
        return module


class TrackingBgeBackend:
    """BGE backend stand-in recording each embed call and its flags."""

    def __init__(self):
        self.calls = []

    def embed(
        self,
        sentences,
        return_dense=True,
        return_sparse=True,
        return_colbert=True,
        normalize_dense=False,
        deadline=None,
    ):
        self.calls.append(
            {
                "sentences": sentences,
                "return_dense": return_dense,
                "return_sparse": return_sparse,
                "return_colbert": return_colbert,
                "normalize_dense": normalize_dense,
            }
        )
        result = {}
        if return_dense:
            result["dense_vecs"] = [[3.0, 4.0] for _ in sentences]
        if return_sparse:
            result["lexical_weights"] = [{"42": 0.5} for _ in sentences]
        if return_colbert:
            result["colbert_vecs"] = [[[0.1, 0.2]] for _ in sentences]
        return result


class TrackingQwenDenseBackend:
    """Qwen dense backend stand-in recording each embed call."""

    def __init__(self):
        self.calls = []

    def embed_dense(self, sentences, normalize_dense=False, deadline=None):
        self.calls.append({"sentences": sentences, "normalize_dense": normalize_dense})
        return {"dense_vecs": [[9.0] * 1024 for _ in sentences]}


class FailingQwenDenseBackend:
    """Qwen dense backend stand-in that always raises."""

    def embed_dense(self, sentences, normalize_dense=False, deadline=None):
        raise RuntimeError("qwen dense failed")


def build_embedding_service(module, dense_model_name):
    service = module.EmbeddingService.__new__(module.EmbeddingService)
    service.profile = (
        module.EMBEDDER_PROFILE_ADVANCED
        if dense_model_name == module.QWEN_DENSE_EMBEDDING_MODEL
        else module.EMBEDDER_PROFILE_BASIC
    )
    service.bge_backend = TrackingBgeBackend()
    service.qwen_dense_backend = (
        TrackingQwenDenseBackend()
        if dense_model_name == module.QWEN_DENSE_EMBEDDING_MODEL
        else None
    )
    service.milco_sparse_backend = None
    service.bge_model_name = module.BGE_EMBEDDING_MODEL
    service.dense_model_name = dense_model_name
    service.sparse_model_name = module.BGE_EMBEDDING_MODEL
    service.colbert_model_name = module.BGE_EMBEDDING_MODEL
    return service


class RerankerWrapperTests(unittest.TestCase):
    """Auth, model loading, length-aware batching and token accounting."""

    def test_api_token_is_required_by_default(self):
        with self.assertRaisesRegex(ValueError, "SMART_EMBEDDER_API_TOKENS"):
            load_server(
                {
                    "API_TOKENS": "",
                    "SMART_EMBEDDER_ALLOW_UNAUTHENTICATED_LOCAL": "false",
                }
            )

    def test_explicit_local_authentication_opt_out_is_accepted(self):
        module = load_server(
            {
                "API_TOKENS": "",
                "SMART_EMBEDDER_ALLOW_UNAUTHENTICATED_LOCAL": "true",
            }
        )

        self.assertTrue(module.ALLOW_UNAUTHENTICATED_LOCAL)

    def test_offline_runtime_rejects_a_missing_pinned_snapshot(self):
        module = load_server(stub_model_resolution=False)
        fake_hub = install_support_fakes()["huggingface_hub"]

        with (
            patch.dict(sys.modules, {"huggingface_hub": fake_hub}),
            patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}),
            self.assertRaisesRegex(RuntimeError, "preparation service"),
        ):
            module._cache_first_model_path(module.BGE_EMBEDDING_MODEL)

    def test_bge_embedding_loads_the_resolved_local_snapshot(self):
        module = load_server()

        with patch.object(
            module, "_cache_first_model_path", return_value="/cache/bge-snapshot"
        ):
            backend = module.BgeM3EmbeddingBackend(module.BGE_EMBEDDING_MODEL)

        self.assertEqual(backend.model.init_args[0], "/cache/bge-snapshot")

    def test_length_aware_batching_restores_all_outputs_to_api_order(self):
        module = load_server()
        service = build_embedding_service(module, module.BGE_EMBEDDING_MODEL)

        class OrderAwareBackend:
            """Backend stand-in recording the order sentences reached it in."""

            def __init__(self):
                self.sentences = []

            def embed(self, sentences, **_kwargs):
                self.sentences = list(sentences)
                return {
                    "dense_vecs": [[text] for text in sentences],
                    "lexical_weights": [{"text": text} for text in sentences],
                    "colbert_vecs": [[[text]] for text in sentences],
                }

        backend = OrderAwareBackend()
        service.bge_backend = backend
        sentences = ["the longest sentence", "x", "medium"]

        result = service.embed(sentences)

        self.assertEqual(backend.sentences, ["x", "medium", "the longest sentence"])
        self.assertEqual(result["dense_vecs"], [[text] for text in sentences])
        self.assertEqual(
            result["lexical_weights"], [{"text": text} for text in sentences]
        )
        self.assertEqual(result["colbert_vecs"], [[[text]] for text in sentences])

    def test_length_aware_batching_can_be_disabled(self):
        module = load_server({"EMBED_LENGTH_AWARE_BATCHING": "false"})
        service = build_embedding_service(module, module.BGE_EMBEDDING_MODEL)
        sentences = ["long sentence", "x"]

        service.embed(sentences)

        self.assertEqual(service.bge_backend.calls[0]["sentences"], sentences)

    def test_length_aware_order_is_stable_for_equal_lengths(self):
        module = load_server()

        order = module.length_aware_sentence_order(["bbb", "a", "ccc", "dd"])

        self.assertEqual(order, [1, 3, 0, 2])

    def test_embedding_token_batch_records_useful_and_padded_tokens(self):
        module = load_server()

        module.record_embedding_token_batch(
            {"attention_mask": [[1, 1, 0], [1, 0, 0]]}, backend="bge"
        )

        self.assertEqual(module.embedding_input_tokens.increment_calls[-1], 3)
        self.assertEqual(module.embedding_padded_tokens.increment_calls[-1], 6)
        self.assertEqual(module.embedding_tokenized_batches.increment_calls[-1], 1)
        self.assertEqual(
            module.embedding_input_tokens.label_calls[-1], ((), {"backend": "bge"})
        )

    def test_embedding_token_batch_accepts_onnx_numpy_attention_mask(self):
        module = load_server()

        module.record_embedding_token_batch(
            {"attention_mask": np.asarray([[1, 1, 0], [1, 0, 0]])},
            backend="bge",
        )

        self.assertEqual(module.embedding_input_tokens.increment_calls[-1], 3)
        self.assertEqual(module.embedding_padded_tokens.increment_calls[-1], 6)

    def test_advanced_token_batch_records_backend_specific_occupancy(self):
        module = load_server()

        module.record_advanced_token_batch(
            {"attention_mask": [[1, 1, 0, 0], [1, 1, 1, 0]]},
            backend="qwen_dense",
        )

        self.assertEqual(module.advanced_backend_input_tokens.increment_calls[-1], 5)
        self.assertEqual(module.advanced_backend_padded_tokens.increment_calls[-1], 8)
        self.assertEqual(
            module.advanced_backend_input_tokens.label_calls[-1],
            ((), {"backend": "qwen_dense"}),
        )

    def test_instrumented_tokenizer_delegates_and_records_existing_batch(self):
        module = load_server()
        tokenizer = FakeTokenizer()
        wrapped = module.InstrumentedEmbeddingTokenizer(tokenizer, backend="bge")

        with patch.object(module, "record_embedding_token_batch") as recorder:
            result = wrapped(["alpha"], padding=True)

        recorder.assert_called_once_with(result, "bge")
        self.assertEqual(wrapped.eos_token, tokenizer.eos_token)

    def test_qwen_dense_only_request_calls_only_qwen_backend(self):
        module = load_server()
        service = build_embedding_service(module, module.QWEN_DENSE_EMBEDDING_MODEL)
        result = service.embed(
            ["alpha"],
            return_dense=True,
            return_sparse=False,
            return_colbert=False,
            normalize_dense=True,
        )
        self.assertEqual(result["dense_vecs"][0], [9.0] * 1024)
        self.assertEqual(service.bge_backend.calls, [])
        self.assertEqual(
            service.qwen_dense_backend.calls,
            [{"sentences": ["alpha"], "normalize_dense": True}],
        )

    def test_qwen_sparse_colbert_request_calls_only_bge_backend(self):
        module = load_server()
        service = build_embedding_service(module, module.QWEN_DENSE_EMBEDDING_MODEL)
        result = service.embed(
            ["alpha"],
            return_dense=False,
            return_sparse=True,
            return_colbert=True,
            normalize_dense=False,
        )
        self.assertNotIn("dense_vecs", result)
        self.assertEqual(result["lexical_weights"], [{"42": 0.5}])
        self.assertEqual(result["colbert_vecs"], [[[0.1, 0.2]]])
        self.assertEqual(service.qwen_dense_backend.calls, [])
        self.assertEqual(len(service.bge_backend.calls), 1)
        self.assertFalse(service.bge_backend.calls[0]["return_dense"])

    def test_qwen_hybrid_request_merges_dense_with_bge_sparse_and_colbert(self):
        module = load_server()
        service = build_embedding_service(module, module.QWEN_DENSE_EMBEDDING_MODEL)
        result = service.embed(
            ["alpha"],
            return_dense=True,
            return_sparse=True,
            return_colbert=True,
            normalize_dense=False,
        )
        self.assertEqual(result["dense_vecs"][0], [9.0] * 1024)
        self.assertEqual(result["lexical_weights"], [{"42": 0.5}])
        self.assertEqual(result["colbert_vecs"], [[[0.1, 0.2]]])
        self.assertEqual(len(service.qwen_dense_backend.calls), 1)
        self.assertEqual(len(service.bge_backend.calls), 1)
        self.assertFalse(service.bge_backend.calls[0]["return_dense"])

    def test_bge_dense_request_uses_bge_for_all_requested_outputs(self):
        module = load_server()
        service = build_embedding_service(module, module.BGE_EMBEDDING_MODEL)
        result = service.embed(
            ["alpha"],
            return_dense=True,
            return_sparse=True,
            return_colbert=True,
            normalize_dense=True,
        )
        self.assertEqual(result["dense_vecs"], [[3.0, 4.0]])
        self.assertEqual(result["lexical_weights"], [{"42": 0.5}])
        self.assertEqual(result["colbert_vecs"], [[[0.1, 0.2]]])
        self.assertIsNone(service.qwen_dense_backend)
        self.assertEqual(len(service.bge_backend.calls), 1)
        self.assertTrue(service.bge_backend.calls[0]["return_dense"])
        self.assertTrue(service.bge_backend.calls[0]["normalize_dense"])

    def test_bge_reranker_prefers_cached_snapshot_path(self):
        module = load_server()

        with patch.object(
            module,
            "_cache_first_model_path",
            return_value="/cache/bge-reranker-v2-m3",
        ):
            wrapper = module.RerankerWrapper(module.BGE_RERANKER_MODEL)

        self.assertEqual(wrapper.model.model_name, "/cache/bge-reranker-v2-m3")

    def test_max_length_configuration_is_split_by_backend(self):
        module = load_server(
            {
                "BGE_EMBED_MAX_LENGTH": "3072",
                "QWEN_EMBED_MAX_LENGTH": "12000",
                "BGE_RERANK_MAX_LENGTH": "1024",
                "QWEN_RERANK_MAX_LENGTH": "16000",
            }
        )

        self.assertEqual(module.BGE_EMBED_MAX_LENGTH, 3072)
        self.assertEqual(module.QWEN_EMBED_MAX_LENGTH, 12000)
        self.assertEqual(module.BGE_RERANK_MAX_LENGTH, 1024)
        self.assertEqual(module.QWEN_RERANK_MAX_LENGTH, 16000)

    def test_default_max_lengths_match_documented_model_caps(self):
        module = load_server()

        self.assertEqual(module.BGE_EMBED_MAX_LENGTH, 8192)
        self.assertEqual(module.QWEN_EMBED_MAX_LENGTH, 32768)
        self.assertEqual(module.BGE_RERANK_MAX_LENGTH, 8192)
        self.assertEqual(module.QWEN_RERANK_MAX_LENGTH, 32768)

    def test_bge_embed_batch_override_is_isolated_from_other_backends(self):
        module = load_server({"BGE_EMBED_BATCH_SIZE": "7"})
        backend = module.BgeM3EmbeddingBackend(module.BGE_EMBEDDING_MODEL)

        backend.embed(
            ["alpha", "beta"],
            return_dense=True,
            return_sparse=True,
            return_colbert=False,
        )

        _, kwargs = backend.model.calls[-1]
        self.assertEqual(module.batch_size, 7)
        self.assertEqual(kwargs["batch_size"], 7)
        self.assertEqual(module.QWEN_EMBED_BATCH_SIZE, 1)
        self.assertEqual(module.MILCO_BATCH_SIZE, 1)
        self.assertEqual(module.BGE_RERANK_BATCH_SIZE, 1)

    def test_valid_autotune_profile_sets_threads_lanes_and_batch_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile_path = Path(temporary) / "profile.json"
            detected_cpu = {
                "architecture": "x86_64",
                "model": "Unit Test CPU",
                "logical_cpu_count": 4,
                "physical_core_count": 4,
                "affinity_cpu_count": 4,
                "affinity_cpu_ids": list(range(4)),
                "effective_cpu_count": 4,
                "cpu_quota": 4.0,
                "cpuset": list(range(4)),
                "instruction_sets": ["avx2"],
                "numa_node_count": 1,
            }
            with patch.object(autotune, "detect_cpu", return_value=detected_cpu):
                fingerprint = autotune.build_fingerprint(
                    torch_module=FakeTorchModule(),
                    app_version="1.2.0",
                    model_revision=pinned_model_id("BAAI/bge-m3"),
                    artifact_digest=autotune.source_artifact_digest(),
                )
                config = autotune.AutotuneConfig.from_env(
                    {"SMART_EMBEDDER_AUTOTUNE_PROFILE": str(profile_path)}
                )
                profile = autotune.make_profile(
                    fingerprint=fingerprint,
                    config=config,
                    selected={
                        "runtime": "torch",
                        "precision": "fp32",
                        "threads": 2,
                        "inference_concurrency": 2,
                        "batch_policy": [
                            {"max_tokens": 128, "batch_size": 4},
                            {"max_tokens": 8192, "batch_size": 1},
                        ],
                    },
                    results=[],
                )
                autotune.atomic_write_profile(profile_path, profile)

                module = load_server(
                    {
                        "SMART_EMBEDDER_AUTOTUNE_PROFILE": str(profile_path),
                        "EMBEDDER_NUM_THREADS": "",
                        "BGE_EMBED_BATCH_SIZE": "",
                        "BGE_EMBED_INFERENCE_CONCURRENCY": "",
                    }
                )

        self.assertEqual(module.AUTOTUNE_RUNTIME_PROFILE.state, "active")
        self.assertEqual(module.torch.num_threads, 2)
        self.assertEqual(module.BGE_EMBED_INFERENCE_CONCURRENCY, 2)
        self.assertEqual(module.batch_size, 4)
        self.assertEqual(module.bge_batch_size_for_token_length(512), 1)

    def test_explicit_runtime_overrides_beat_valid_autotune_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile_path = Path(temporary) / "profile.json"
            fingerprint = autotune.build_fingerprint(
                torch_module=FakeTorchModule(),
                app_version="1.2.0",
                model_revision=pinned_model_id("BAAI/bge-m3"),
                artifact_digest=autotune.source_artifact_digest(),
            )
            config = autotune.AutotuneConfig.from_env(
                {"SMART_EMBEDDER_AUTOTUNE_PROFILE": str(profile_path)}
            )
            profile = autotune.make_profile(
                fingerprint=fingerprint,
                config=config,
                selected={
                    "threads": 2,
                    "inference_concurrency": 2,
                    "batch_policy": [{"max_tokens": 8192, "batch_size": 8}],
                },
                results=[],
            )
            autotune.atomic_write_profile(profile_path, profile)

            module = load_server(
                {
                    "SMART_EMBEDDER_AUTOTUNE_PROFILE": str(profile_path),
                    "EMBEDDER_NUM_THREADS": "1",
                    "BGE_EMBED_BATCH_SIZE": "3",
                    "BGE_EMBED_INFERENCE_CONCURRENCY": "1",
                }
            )

        self.assertEqual(module.torch.num_threads, 1)
        self.assertEqual(module.batch_size, 3)
        self.assertEqual(module.BGE_EMBED_INFERENCE_CONCURRENCY, 1)
        self.assertEqual(module.BGE_EMBED_BATCH_POLICY[-1]["batch_size"], 3)

    def test_valid_autotune_profile_activates_measured_torch_compile(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile_path = Path(temporary) / "profile.json"
            fingerprint = autotune.build_fingerprint(
                torch_module=FakeTorchModule(),
                app_version="1.2.0",
                model_revision=pinned_model_id("BAAI/bge-m3"),
                artifact_digest=autotune.source_artifact_digest(),
            )
            config = autotune.AutotuneConfig.from_env(
                {
                    "SMART_EMBEDDER_AUTOTUNE_PROFILE": str(profile_path),
                    "BGE_TORCH_COMPILE": "auto",
                }
            )
            profile = autotune.make_profile(
                fingerprint=fingerprint,
                config=config,
                selected={
                    "runtime": "torch",
                    "precision": "fp16",
                    "torch_compile": True,
                    "torch_compile_reason": "measured-material-gain",
                    "inference_concurrency": 1,
                    "batch_policy": [{"max_tokens": 8192, "batch_size": 1}],
                },
                results=[],
            )
            autotune.atomic_write_profile(profile_path, profile)

            module = load_server(
                {
                    "SMART_EMBEDDER_AUTOTUNE_PROFILE": str(profile_path),
                    "BGE_TORCH_COMPILE": "auto",
                }
            )

        self.assertEqual(module.AUTOTUNE_RUNTIME_PROFILE.state, "active")
        self.assertTrue(module.BGE_TORCH_COMPILE_ENABLED)
        self.assertEqual(
            module.AUTOTUNE_SELECTED["torch_compile_reason"],
            "measured-material-gain",
        )

    def test_bge_backend_applies_length_aware_policy_per_group(self):
        module = load_server()
        backend = module.BgeM3EmbeddingBackend(module.BGE_EMBEDDING_MODEL)
        module.BGE_EMBED_BATCH_POLICY = [
            {"max_tokens": 128, "batch_size": 8},
            {"max_tokens": 8192, "batch_size": 1},
        ]
        backend.model.tokenizer.token_lengths = lambda sentences, limit: [64, 700]

        backend.embed(["short", "long"])

        self.assertEqual(len(backend.model.calls), 2)
        self.assertEqual(backend.model.calls[0][1]["batch_size"], 8)
        self.assertEqual(backend.model.calls[1][1]["batch_size"], 1)

    def test_bge_backend_splits_cpu_work_across_calibrated_lanes(self):
        module = load_server()
        with patch.object(module, "BGE_EMBED_INFERENCE_CONCURRENCY", 2):
            backend = module.BgeM3EmbeddingBackend(module.BGE_EMBEDDING_MODEL)
            backend.model.tokenizer.token_lengths = lambda sentences, limit: (
                [64] * len(sentences)
            )
            backend.embed(["a", "b", "c", "d"])
            backend.close()

        self.assertEqual(len(backend.model.calls), 2)
        self.assertEqual([len(call[0][0]) for call in backend.model.calls], [2, 2])

    def test_long_sequence_bge_batch_tiers_scale_by_vram(self):
        module = load_server()

        self.assertEqual(module._long_sequence_bge_batch_size(3.0), 2)
        self.assertEqual(module._long_sequence_bge_batch_size(4.0), 4)
        self.assertEqual(module._long_sequence_bge_batch_size(7.99), 8)
        self.assertEqual(module._long_sequence_bge_batch_size(15.99), 16)
        self.assertEqual(module._long_sequence_bge_batch_size(23.99), 32)

    def test_bge_embedding_backend_uses_bge_embed_max_length(self):
        module = load_server({"BGE_EMBED_MAX_LENGTH": "6"})
        backend = module.BgeM3EmbeddingBackend(module.BGE_EMBEDDING_MODEL)

        backend.embed(
            ["alpha"],
            return_dense=True,
            return_sparse=True,
            return_colbert=True,
            normalize_dense=False,
        )

        _, kwargs = backend.model.calls[-1]
        self.assertEqual(kwargs["max_length"], 6)

    def test_env_bool_empty_string_uses_default_without_warning(self):
        module = load_server()

        with (
            patch.dict(os.environ, {"BGE_TORCH_COMPILE": ""}),
            self.assertNoLogs(level="WARNING"),
        ):
            self.assertTrue(module._env_bool("BGE_TORCH_COMPILE", True))
            self.assertFalse(module._env_bool("BGE_TORCH_COMPILE", False))

    def test_swagger_exposure_defaults_to_disabled(self):
        module = load_server({"EXPOSE_SMART_EMBEDDER_SWAGGER": ""})

        self.assertIsNone(module.app.docs_url)
        self.assertIsNone(module.app.redoc_url)
        self.assertIsNone(module.app.openapi_url)

    def test_swagger_exposure_accepts_false_like_values(self):
        for value in ("false", "FALSE", "no", "0", "off"):
            with self.subTest(value=value):
                module = load_server({"EXPOSE_SMART_EMBEDDER_SWAGGER": value})
                self.assertIsNone(module.app.docs_url)
                self.assertIsNone(module.app.redoc_url)
                self.assertIsNone(module.app.openapi_url)

    def test_swagger_exposure_accepts_true_like_values(self):
        for value in ("true", "TRUE", "yes", "1", "on"):
            with self.subTest(value=value):
                module = load_server({"EXPOSE_SMART_EMBEDDER_SWAGGER": value})
                self.assertEqual(module.app.docs_url, "/docs")
                self.assertEqual(module.app.redoc_url, "/redoc")
                self.assertEqual(module.app.openapi_url, "/openapi.json")

    def test_invalid_swagger_exposure_value_uses_disabled_default(self):
        module = load_server({"EXPOSE_SMART_EMBEDDER_SWAGGER": "invalid"})

        self.assertIsNone(module.app.docs_url)
        self.assertIsNone(module.app.redoc_url)
        self.assertIsNone(module.app.openapi_url)

    def test_swagger_switch_does_not_change_business_route_decorators(self):
        module = load_server({"EXPOSE_SMART_EMBEDDER_SWAGGER": "false"})

        get_paths = [call[0][0] for call in module.app.get_calls]
        post_paths = [call[0][0] for call in module.app.post_calls]
        self.assertEqual(get_paths, ["/health", "/models", "/stats", "/metrics"])
        self.assertEqual(post_paths, ["/embeddings/", "/rerank"])

    def test_bge_embedding_backend_compiles_inner_model_when_available(self):
        module = load_server({"BGE_TORCH_COMPILE": "1"})
        compile_calls = []
        compiled_model = object()

        def fake_compile(model, **kwargs):
            compile_calls.append((model, kwargs))
            return compiled_model

        module.torch.compile = fake_compile

        backend = module.BgeM3EmbeddingBackend(
            module.BGE_EMBEDDING_MODEL,
            device="cuda",
        )

        self.assertTrue(compile_calls)
        target, kwargs = compile_calls[-1]
        self.assertIsInstance(target, FakeInnerBgeModel)
        self.assertIs(backend.model.model, compiled_model)
        self.assertTrue(kwargs["dynamic"])
        self.assertEqual(kwargs["mode"], "default")
        self.assertNotIn("dynamic_shapes", kwargs)
        self.assertTrue(backend.torch_compile_active)

    def test_bge_embedding_backend_uses_compile_mode_override(self):
        module = load_server(
            {
                "BGE_TORCH_COMPILE": "1",
                "BGE_TORCH_COMPILE_MODE": "max-autotune-no-cudagraphs",
            }
        )
        compile_calls = []
        compiled_model = object()

        def fake_compile(model, **kwargs):
            compile_calls.append((model, kwargs))
            return compiled_model

        module.torch.compile = fake_compile

        backend = module.BgeM3EmbeddingBackend(
            module.BGE_EMBEDDING_MODEL,
            device="cuda",
        )

        self.assertTrue(compile_calls)
        _, kwargs = compile_calls[-1]
        self.assertIs(backend.model.model, compiled_model)
        self.assertEqual(kwargs["mode"], "max-autotune-no-cudagraphs")

    def test_bge_embedding_backend_skips_compile_when_disabled_on_cuda(self):
        module = load_server({"BGE_TORCH_COMPILE": "0"})
        compile_calls = []

        def fake_compile(model, **kwargs):
            compile_calls.append((model, kwargs))
            return object()

        module.torch.compile = fake_compile

        with self.assertLogs(level="INFO") as logs:
            backend = module.BgeM3EmbeddingBackend(
                module.BGE_EMBEDDING_MODEL,
                device="cuda",
            )

        self.assertEqual(compile_calls, [])
        self.assertIsInstance(backend.model.model, FakeInnerBgeModel)
        self.assertEqual(len(backend.model.calls), 2)
        _, full_kwargs = backend.model.calls[0]
        _, dense_kwargs = backend.model.calls[1]
        self.assertTrue(full_kwargs["return_dense"])
        self.assertTrue(full_kwargs["return_sparse"])
        self.assertTrue(full_kwargs["return_colbert_vecs"])
        self.assertTrue(dense_kwargs["return_dense"])
        self.assertFalse(dense_kwargs["return_sparse"])
        self.assertFalse(dense_kwargs["return_colbert_vecs"])
        self.assertEqual(len(backend.model.calls[0][0][0]), module.batch_size)
        self.assertEqual(full_kwargs["batch_size"], module.batch_size)
        self.assertEqual(dense_kwargs["batch_size"], module.batch_size)
        self.assertFalse(backend.torch_compile_active)
        self.assertIn("effective setting disabled", "\n".join(logs.output))

    def test_bge_embedding_backend_keeps_original_model_when_compile_fails(self):
        module = load_server({"BGE_TORCH_COMPILE": "1"})

        def failing_compile(model, **kwargs):
            raise RuntimeError("compile unsupported")

        module.torch.compile = failing_compile

        with self.assertLogs(level="WARNING") as logs:
            backend = module.BgeM3EmbeddingBackend(module.BGE_EMBEDDING_MODEL)

        self.assertIsInstance(backend.model.model, FakeInnerBgeModel)
        self.assertIn("torch.compile disabled", "\n".join(logs.output))

    def test_bge_embedding_backend_restores_original_when_compiled_warmup_fails(self):
        module = load_server({"BGE_TORCH_COMPILE": "1"})
        compiled_model = types.SimpleNamespace(raise_on_encode=True)

        def fake_compile(model, **kwargs):
            return compiled_model

        module.torch.compile = fake_compile

        with self.assertLogs(level="WARNING") as logs:
            backend = module.BgeM3EmbeddingBackend(
                module.BGE_EMBEDDING_MODEL,
                device="cuda",
            )

        self.assertIsInstance(backend.model.model, FakeInnerBgeModel)
        self.assertFalse(backend.torch_compile_active)
        self.assertEqual(len(backend.model.calls), 2)
        self.assertIn("warm-up failed", "\n".join(logs.output))

    def test_qwen_dense_backend_uses_qwen_embed_max_length(self):
        module = load_server(
            {
                "EMBEDDER_PROFILE": "advanced",
                "QWEN_EMBED_MAX_LENGTH": "9",
            }
        )
        backend = module.embedding_service.qwen_dense_backend

        backend.embed_dense(["alpha"], normalize_dense=False)

        _, kwargs = backend.tokenizer.calls[-1]
        self.assertEqual(kwargs["max_length"], 9)

    def test_embedding_warnings_use_active_backend_specific_limits(self):
        module = load_server(
            {
                "EMBEDDER_PROFILE": "advanced",
                "BGE_EMBED_MAX_LENGTH": "4",
                "QWEN_EMBED_MAX_LENGTH": "8",
            }
        )
        request = module.EmbedRequest(
            sentences=["tok:6"],
            return_dense=True,
            return_sparse=False,
            return_colbert=False,
            normalize_dense=False,
            sparse_as_indices=False,
        )

        warnings = module.collect_embedding_truncation_warnings(request)

        self.assertEqual(warnings, [])

    def test_advanced_sparse_request_emits_no_truncation_warnings(self):
        module = load_server(
            {
                "EMBEDDER_PROFILE": "advanced",
                "BGE_EMBED_MAX_LENGTH": "4",
                "QWEN_EMBED_MAX_LENGTH": "8",
            }
        )
        request = module.EmbedRequest(
            sentences=["tok:6"],
            return_dense=False,
            return_sparse=True,
            return_colbert=False,
            normalize_dense=False,
            sparse_as_indices=False,
        )
        # MILCO truncates internally and BGE is not loaded, so no warnings.
        self.assertEqual(module.collect_embedding_truncation_warnings(request), [])

    def test_qwen_hybrid_backend_failure_raises_without_partial_response(self):
        module = load_server()
        service = build_embedding_service(module, module.QWEN_DENSE_EMBEDDING_MODEL)
        service.qwen_dense_backend = FailingQwenDenseBackend()
        with self.assertRaisesRegex(RuntimeError, "qwen dense failed"):
            service.embed(
                ["alpha"],
                return_dense=True,
                return_sparse=True,
                return_colbert=True,
                normalize_dense=False,
            )

    def test_bge_backend_uses_flag_reranker_compute_score(self):
        module = load_server()
        wrapper = module.RerankerWrapper(module.BGE_RERANKER_MODEL)
        scores = wrapper.score([["query", "passage"]], normalize=True)
        self.assertEqual(scores, [0.25])
        pairs, normalize, kwargs = wrapper.model.calls[-1]
        self.assertEqual(pairs, [["query", "passage"]])
        self.assertTrue(normalize)
        self.assertEqual(kwargs["max_length"], module.BGE_RERANK_MAX_LENGTH)

    def test_bge_reranker_restores_transformers_five_pair_builder(self):
        module = load_server()
        wrapper = module.RerankerWrapper(module.BGE_RERANKER_MODEL)

        prepared = wrapper.model.tokenizer.prepare_for_model(
            [10, 11],
            [20, 21, 22],
            truncation="only_second",
            max_length=7,
            padding=False,
        )

        self.assertEqual(prepared["input_ids"], [0, 10, 11, 2, 2, 20, 2])
        self.assertEqual(prepared["attention_mask"], [1] * 7)

    def test_bge_rerank_prepare_pairs_truncates_passage_first(self):
        module = load_server({"BGE_RERANK_MAX_LENGTH": "8"})
        wrapper = module.RerankerWrapper(module.BGE_RERANKER_MODEL)

        pairs, warnings = wrapper.prepare_pairs_for_scoring(
            query="tok:3",
            passages=["tok:10"],
        )

        self.assertEqual(pairs, [["tok:3", "tok:2"]])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].code, "passage_truncated")
        self.assertEqual(warnings[0].target.field, "passages")
        self.assertEqual(warnings[0].target.index, 0)
        self.assertEqual(warnings[0].details.original_tokens, 10)
        self.assertEqual(warnings[0].details.max_tokens, 2)

    def test_bge_rerank_prepare_pairs_truncates_query_when_query_exceeds_budget(self):
        module = load_server({"BGE_RERANK_MAX_LENGTH": "5"})
        wrapper = module.RerankerWrapper(module.BGE_RERANKER_MODEL)

        pairs, warnings = wrapper.prepare_pairs_for_scoring(
            query="tok:10",
            passages=["tok:4"],
        )

        self.assertEqual(pairs, [["tok:2", "tok:0"]])
        self.assertEqual(
            [warning.code for warning in warnings],
            [
                "query_truncated",
                "passage_truncated",
            ],
        )

    def test_score_bge_converts_allocator_failure_to_vram_exhausted(self):
        module = load_server()
        reranker = module.reranker

        def explode(pairs, normalize=False, **kwargs):
            raise RuntimeError("CUDA out of memory")

        reranker.model.compute_score = explode
        with self.assertRaises(module.vram_budget.VramExhausted) as ctx:
            reranker.score([["q", "p"]], False)
        self.assertIsNone(ctx.exception.__context__)

    def test_qwen_rerank_prepare_pairs_reserves_prompt_tokens_once(self):
        module = load_server(
            {
                "QWEN_RERANK_MAX_LENGTH": "10",
            }
        )
        wrapper = module.RerankerWrapper(module.QWEN_RERANKER_MODEL)

        class CountingQwenTokenizer(FakeTokenizer):
            """Tokenizer counting the query and document fields of a Qwen prompt."""

            @staticmethod
            def _field_count(text, marker):
                value = text.split(marker, 1)[1].split("\n", 1)[0].strip()
                return int(value.split(":", 1)[1]) if value else 0

            def encode(self, text, add_special_tokens=False, verbose=False):
                _ = verbose
                if "<Query>:" in text and "<Document>:" in text:
                    return list(
                        range(
                            2
                            + self._field_count(text, "<Query>:")
                            + self._field_count(text, "<Document>:")
                        )
                    )
                return super().encode(text, add_special_tokens=add_special_tokens)

        wrapper.tokenizer = CountingQwenTokenizer()

        pairs, warnings = wrapper.prepare_pairs_for_scoring(
            query="tok:3",
            passages=["tok:10"],
        )

        self.assertEqual(pairs, [["tok:3", "tok:1"]])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].code, "passage_truncated")
        self.assertEqual(warnings[0].details.max_tokens, 1)
        self.assertEqual(warnings[0].details.original_tokens, 10)

    def test_qwen_rerank_endpoint_retruncates_decode_drift_instead_of_500(self):
        module = load_server(
            {
                "QWEN_RERANK_MAX_LENGTH": "12",
            }
        )

        class DriftQwenTokenizer(FakeTokenizer):
            """Tokenizer whose decoded length drifts from the encoded one."""

            def __init__(self):
                super().__init__()
                self.pad_token = self.eos_token

            def _text_token_count(self, text):
                if text.startswith("tok:"):
                    return int(text.split(":", 1)[1])
                if text.startswith("drift:"):
                    return int(text.split(":", 1)[1]) * 3
                if not text:
                    return 0
                return 1

            def _field_text(self, text, marker):
                value = text.split(marker, 1)[1]
                return value.split("\n", 1)[0].strip()

            def encode(self, text, add_special_tokens=False):
                if "<Query>:" in text and "<Document>:" in text:
                    query = self._field_text(text, "<Query>:")
                    document = self._field_text(text, "<Document>:")
                    count = (
                        2
                        + self._text_token_count(query)
                        + self._text_token_count(document)
                    )
                    return list(range(count))
                return list(range(self._text_token_count(text) or 2))

            def decode(self, token_ids, skip_special_tokens=True):
                self.decoded.append(list(token_ids))
                return "drift:" + str(len(token_ids))

            def __call__(self, pairs, **kwargs):
                self.calls.append((pairs, kwargs))
                self.encoded.extend(pairs)
                return {
                    "input_ids": [
                        self.encode(pair, add_special_tokens=False) for pair in pairs
                    ]
                }

        wrapper = module.RerankerWrapper(module.QWEN_RERANKER_MODEL)
        wrapper.tokenizer = DriftQwenTokenizer()
        module.reranker = wrapper
        request = module.RerankRequest(
            query="tok:3",
            passages=["tok:10"],
            normalize=False,
        )

        response = asyncio.run(module.rerank(request))

        self.assertEqual(response.results[0].score, 0.75)
        self.assertEqual(response.results[0].passage, "drift:1")
        self.assertEqual(
            [warning.code for warning in response.warnings], ["passage_truncated"]
        )

    def test_qwen_backend_uses_yes_no_probability_scorer(self):
        module = load_server()
        wrapper = module.RerankerWrapper(module.QWEN_RERANKER_MODEL)
        scores = wrapper.score([["query", "passage"]], normalize=False)
        self.assertEqual(scores, [0.75])
        self.assertFalse(wrapper.model.config.use_cache)
        self.assertEqual(wrapper.tokenizer.pad_token, wrapper.tokenizer.eos_token)
        self.assertIn("<Query>: query", wrapper.tokenizer.encoded[0])
        self.assertIn("<Document>: passage", wrapper.tokenizer.encoded[0])

    def test_score_qwen_requests_only_the_last_position_logits(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        reranker = module.reranker
        reranker.score([["query", "passage one"], ["query", "passage two"]], False)

        self.assertTrue(reranker.model.calls, "model was never called")
        for call in reranker.model.calls:
            self.assertEqual(
                call.get("logits_to_keep"),
                1,
                "forward must not materialise logits for every position",
            )
            self.assertFalse(call.get("use_cache", True))

    def test_score_qwen_converts_allocator_failure_to_vram_exhausted(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        reranker = module.reranker

        def explode(chunk):
            raise RuntimeError(
                "!handles_.at(i) INTERNAL ASSERT FAILED at "
                '"/pytorch/c10/cuda/CUDACachingAllocator.cpp":467'
            )

        # Patch the instance, never FakeCausalLM's class: that class is shared
        # by every load_server() call and deleting an attribute off it breaks
        # the rest of the suite.
        reranker._forward_qwen_chunk = explode
        with self.assertRaises(module.vram_budget.VramExhausted) as ctx:
            reranker.score([["q", "p"]], False)
        self.assertEqual(ctx.exception.batch_items, 1)
        self.assertIsNone(ctx.exception.__context__)

    def test_halving_converges_and_preserves_order(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        reranker = module.reranker
        reranker.tokenizer = IndexTaggingQwenTokenizer()
        attempted = []

        def fake_forward(chunk):
            attempted.append(len(chunk["input_ids"]))
            if len(chunk["input_ids"]) > 2:
                raise RuntimeError(
                    "!handles_.at(i) INTERNAL ASSERT FAILED at "
                    '"/pytorch/c10/cuda/CUDACachingAllocator.cpp":467'
                )
            return [float(_qwen_pair_index(item)) for item in chunk["input_ids"]]

        reranker._forward_qwen_chunk = fake_forward
        scores = reranker.score([["q", f"p{i}"] for i in range(8)], False)

        # Positional equality, not just length/uniformity: a bug that
        # swaps which half of a split chunk gets which scores (e.g. head
        # and tail requeued in the wrong order) would still pass a
        # length-and-uniform-value check but fails this one.
        self.assertEqual(scores, [float(i) for i in range(8)])
        # A real constraint, not the trivially-true "no attempt exceeds the
        # configured batch size": after the first failure at the full
        # size, the ceiling is halved, so nothing after it may be attempted
        # above half the original chunk size.
        self.assertEqual(attempted[0], 8)
        self.assertLessEqual(max(attempted[1:]), attempted[0] // 2)
        self.assertIn(2, attempted)

    def test_halving_is_sticky_for_the_rest_of_the_request(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        reranker = module.reranker
        reranker.tokenizer = IndexTaggingQwenTokenizer()
        attempted = []

        def fake_forward(chunk):
            size = len(chunk["input_ids"])
            attempted.append(size)
            # Every batch above the eventual floor keeps failing here, so
            # this models a cascade (16 -> 8 -> 4 -> 2) rather than one
            # failure followed by a return to full size: once the ceiling
            # drops, later chunks are pre-split below it before a forward
            # is ever attempted again, so nothing above size 2 can recur
            # within this request regardless of whether it would "succeed".
            if size > 2:
                raise RuntimeError("CUDA out of memory")
            return [float(_qwen_pair_index(item)) for item in chunk["input_ids"]]

        reranker._forward_qwen_chunk = fake_forward
        scores = reranker.score([["q", f"p{i}"] for i in range(16)], False)

        self.assertEqual(scores, [float(i) for i in range(16)])
        first_success = next(i for i, s in enumerate(attempted) if s <= 2)
        self.assertTrue(
            all(size <= 2 for size in attempted[first_success:]),
            f"chunk size grew back after a failure: {attempted}",
        )

        # Design §6: the halving ceiling is local to _score_qwen and must
        # reset on the next request rather than persist on the reranker
        # instance. A second call - even one that never fails - must
        # attempt its first chunk at the full configured batch size again.
        second_attempted = []

        def fake_forward_no_failure(chunk):
            second_attempted.append(len(chunk["input_ids"]))
            return [float(_qwen_pair_index(item)) for item in chunk["input_ids"]]

        reranker._forward_qwen_chunk = fake_forward_no_failure
        batch_size = module.QWEN_RERANK_BATCH_SIZE
        second_scores = reranker.score(
            [["q", f"p{i}"] for i in range(batch_size)], False
        )

        self.assertEqual(second_scores, [float(i) for i in range(batch_size)])
        self.assertEqual(second_attempted[0], batch_size)

    def test_single_item_failure_surfaces_as_vram_exhausted(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        reranker = module.reranker

        def always_fail(chunk):
            raise RuntimeError("CUDA out of memory")

        reranker._forward_qwen_chunk = always_fail
        with self.assertRaises(module.vram_budget.VramExhausted) as ctx:
            reranker.score([["q", "p"]], False)
        self.assertEqual(ctx.exception.batch_items, 1)

    def test_capacity_error_is_413_and_not_retryable(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        failure = module.vram_budget.VramExhausted(
            batch_items=1,
            batch_tokens=8100,
            estimated_bytes=2_457_037_800,
            original="OutOfMemoryError: CUDA out of memory",
        )
        module._STARTUP_HEADROOM_BYTES = 2_287_634_944

        exc = module._rerank_capacity_response(failure, passage_count=200)

        self.assertEqual(exc.status_code, 413)
        self.assertEqual(exc.detail["error"], "rerank_capacity_exceeded")
        self.assertEqual(exc.detail["passages"], 200)
        self.assertEqual(exc.detail["batch_tokens"], 8100)

    def test_transient_pressure_is_503_with_retry_after(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        # A single-item batch that WOULD fit against the startup baseline:
        # only the current moment is bad.
        failure = module.vram_budget.VramExhausted(
            batch_items=1,
            batch_tokens=100,
            estimated_bytes=30_333_800,
            original="OutOfMemoryError: CUDA out of memory",
        )
        module._STARTUP_HEADROOM_BYTES = 2_287_634_944

        exc = module._rerank_capacity_response(failure, passage_count=20)

        self.assertEqual(exc.status_code, 503)
        self.assertEqual(exc.detail["error"], "rerank_vram_pressure")

    def test_unknown_cost_at_batch_size_one_is_413_not_retryable(self):
        # The BGE path reports estimated_bytes=0 because it has no per-token
        # cost model. Unknown must not be read as "fits", or every BGE
        # capacity failure would be retried three times by the middleware.
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        failure = module.vram_budget.VramExhausted(
            batch_items=1,
            batch_tokens=8192,
            estimated_bytes=0,
            original="OutOfMemoryError: CUDA out of memory",
        )
        module._STARTUP_HEADROOM_BYTES = 2_287_634_944

        exc = module._rerank_capacity_response(failure, passage_count=1)

        self.assertEqual(exc.status_code, 413)

    def test_unknown_cost_above_batch_size_one_stays_retryable(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        failure = module.vram_budget.VramExhausted(
            batch_items=8,
            batch_tokens=65536,
            estimated_bytes=0,
            original="OutOfMemoryError: CUDA out of memory",
        )
        module._STARTUP_HEADROOM_BYTES = 2_287_634_944

        exc = module._rerank_capacity_response(failure, passage_count=8)

        self.assertEqual(exc.status_code, 503)

    def test_unknown_cost_503_omits_fabricated_fields(self):
        # estimated_bytes == 0 is the BGE path's honest "no cost model"
        # signal, not a real zero. Fabricating suggested_max_passages from
        # it previously divided the startup headroom (bytes) by a zero
        # per-item cost and reported that byte count as a passage limit.
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        failure = module.vram_budget.VramExhausted(
            batch_items=8,
            batch_tokens=65536,
            estimated_bytes=0,
            original="OutOfMemoryError: CUDA out of memory",
        )
        module._STARTUP_HEADROOM_BYTES = 2_287_634_944

        exc = module._rerank_capacity_response(failure, passage_count=8)

        self.assertEqual(exc.status_code, 503)
        self.assertNotIn("suggested_max_passages", exc.detail)
        self.assertNotIn("effective_max_batch_tokens", exc.detail)
        self.assertNotIn("passages per request", exc.detail["message"])

    def test_multi_item_failure_suggests_a_smaller_passage_count(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        failure = module.vram_budget.VramExhausted(
            batch_items=8,
            batch_tokens=4240,
            estimated_bytes=1_286_153_120,
            original="OutOfMemoryError: CUDA out of memory",
        )
        module._STARTUP_HEADROOM_BYTES = 2_287_634_944

        exc = module._rerank_capacity_response(failure, passage_count=200)

        self.assertEqual(exc.status_code, 503)
        self.assertGreaterEqual(exc.detail["suggested_max_passages"], 1)
        self.assertIn("effective_max_batch_tokens", exc.detail)
        self.assertGreater(exc.detail["effective_max_batch_tokens"], 0)

    def test_qwen_normalize_is_api_compatible_noop(self):
        module = load_server()
        wrapper = module.RerankerWrapper(module.QWEN_RERANKER_MODEL)
        self.assertEqual(
            wrapper.score([["query", "passage"]], normalize=True),
            wrapper.score([["query", "passage"]], normalize=False),
        )

    def test_rerank_endpoint_returns_truncation_warnings(self):
        module = load_server({"BGE_EMBED_MAX_LENGTH": "8"})

        class FakeReranker:
            """Reranker stand-in returning fixed pairs and scores."""

            model_name = module.BGE_RERANKER_MODEL
            executor = module.GPU_EXECUTOR

            def prepare_pairs_for_scoring(self, query, passages):
                return (
                    [[query, "tok:2"]],
                    [
                        module.ApiWarning(
                            code="passage_truncated",
                            message=(
                                "Rerank passage was truncated to the model token limit."
                            ),
                            target=module.WarningTarget(
                                field="passages",
                                index=0,
                                pointer="/passages/0",
                            ),
                            details=module.WarningDetails(
                                model=module.BGE_RERANKER_MODEL,
                                max_tokens=2,
                                original_tokens=10,
                                truncated_tokens=8,
                            ),
                        )
                    ],
                )

            def score(self, pairs, normalize, deadline=None):
                return [0.25]

        module.reranker = FakeReranker()
        request = module.RerankRequest(
            query="tok:3",
            passages=["tok:10"],
            normalize=False,
        )

        response = asyncio.run(module.rerank(request))

        self.assertEqual(len(response.warnings), 1)
        self.assertEqual(response.warnings[0].code, "passage_truncated")
        self.assertEqual(response.results[0].passage, "tok:2")

    def test_health_exposes_selected_models(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        health = asyncio.run(module.health_check())
        self.assertEqual(module.APP_VERSION, "1.2.0")
        self.assertEqual(module.app.version, "1.2.0")
        self.assertEqual(health["version"], "1.2.0")
        self.assertEqual(
            health["dense_embedding_model"], module.QWEN_DENSE_EMBEDDING_MODEL
        )
        self.assertIsNone(health["model"])
        self.assertEqual(health["reranker_model"], module.QWEN_RERANKER_MODEL)
        self.assertEqual(health["profile"], "advanced")
        self.assertEqual(health["sparse_model_name"], module.MILCO_SPARSE_MODEL)
        self.assertIsNone(health["colbert_model_name"])
        self.assertEqual(health["bge_embed_max_length"], module.BGE_EMBED_MAX_LENGTH)
        self.assertEqual(health["qwen_embed_max_length"], module.QWEN_EMBED_MAX_LENGTH)
        self.assertTrue(health["milco_source_view"])
        self.assertEqual(health["milco_top_k"], module.MILCO_TOP_K)
        self.assertEqual(health["milco_max_length"], module.MILCO_MAX_LENGTH)
        self.assertEqual(health["milco_batch_size"], module.MILCO_BATCH_SIZE)
        self.assertEqual(
            health["milco_max_batch_tokens"], module.MILCO_MAX_BATCH_TOKENS
        )
        self.assertEqual(
            health["milco_projection_chunk_tokens"],
            module.MILCO_PROJECTION_CHUNK_TOKENS,
        )
        self.assertEqual(health["bge_rerank_max_length"], module.BGE_RERANK_MAX_LENGTH)
        self.assertEqual(
            health["qwen_rerank_max_length"], module.QWEN_RERANK_MAX_LENGTH
        )
        self.assertEqual(
            health["reranker_content_budget"],
            module.reranker._rerank_content_budget(),
        )
        self.assertEqual(
            health["reranker_fixed_overhead_tokens"],
            module.QWEN_RERANK_MAX_LENGTH - module.reranker._rerank_content_budget(),
        )
        self.assertEqual(
            health["reranker_instruction"], module.QWEN_RERANKER_INSTRUCTION
        )
        self.assertTrue(health["advanced_config"]["active"])
        self.assertEqual(health["advanced_config"]["schedule"], "sequential")
        self.assertTrue(health["advanced_config"]["milco_source_view"])
        self.assertTrue(health["length_aware_batching"])
        self.assertEqual(health["bge_embed_batch_size"], module.batch_size)
        self.assertFalse(health["autotune"]["torch_compile"])
        self.assertEqual(health["autotune"]["torch_compile_policy"], "auto")
        self.assertEqual(health["autotune"]["torch_compile_mode"], "default")

    def test_server_info_exposes_dense_embedding_model(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        self.assertEqual(
            module.server_info.last_info["dense_embedding_model"],
            module.QWEN_DENSE_EMBEDDING_MODEL,
        )
        self.assertEqual(module.server_info.last_info["version"], module.APP_VERSION)
        self.assertEqual(module.server_info.last_info["length_aware_batching"], "True")
        self.assertEqual(
            module.server_info.last_info["bge_embed_batch_size"],
            str(module.batch_size),
        )
        self.assertEqual(module.advanced_config_info.last_info["active"], "True")
        self.assertEqual(
            module.advanced_config_info.last_info["milco_source_view"], "True"
        )

    def test_gpu_memory_snapshot_updates_global_and_legacy_metrics(self):
        module = load_server()
        module.has_cuda = True
        module.torch.cuda.memory_allocated_value = 123
        module.torch.cuda.memory_reserved_value = 456
        module.torch.cuda.mem_get_info_value = (789, 1000)

        module.record_gpu_memory_metrics()

        self.assertEqual(module.gpu_memory_allocated.last_set, 123)
        self.assertEqual(module.gpu_memory_reserved.last_set, 456)
        self.assertEqual(module.process_gpu_memory_allocated.last_set, 123)
        self.assertEqual(module.process_gpu_memory_reserved.last_set, 456)
        self.assertEqual(module.process_gpu_memory_free.last_set, 789)
        self.assertEqual(module.process_gpu_memory_total.last_set, 1000)

    def test_calibrated_policy_lowers_but_never_raises_the_configured_cap(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        # Both are mandatory: the function early-returns without CUDA, and
        # again when vocab_size is 0. FakeCausalLM.config carries only
        # use_cache, and FakeTorchModule.cuda_available defaults to False.
        module.has_cuda = True
        module.reranker.model.config.vocab_size = 151669
        module.torch.cuda.memory_allocated_value = 4_769_281_536
        module.torch.cuda.memory_reserved_value = 7_056_916_480
        module.torch.cuda.mem_get_info_value = (0, 8_585_216_000)

        policy = module._calibrated_rerank_token_policy()

        configured = {
            entry["max_tokens"]: entry["max_batch_tokens"]
            for entry in module.QWEN_RERANK_MAX_BATCH_TOKENS_POLICY
        }
        for entry in policy:
            self.assertLessEqual(
                entry["max_batch_tokens"], configured[entry["max_tokens"]]
            )
        # 2.2 GB headroom minus reserve, over 303338 bytes/token, is far below
        # the configured 8192 - so the clamp must actually bite here.
        self.assertLess(max(e["max_batch_tokens"] for e in policy), 8192)

    def test_calibration_is_a_noop_without_cuda(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        module.has_cuda = False

        policy = module._calibrated_rerank_token_policy()

        self.assertEqual(policy, module.QWEN_RERANK_MAX_BATCH_TOKENS_POLICY)

    def test_advanced_profile_imports_under_cuda_without_nameerror(self):
        """Loading the advanced profile under CUDA raises no NameError.

        RerankerWrapper._init_qwen scores a warm-up pair during construction,
        which runs while CUDA is reported available - before the calibrated
        token-budget globals below it in the module are assigned. Loading the
        advanced profile with the fake CUDA path enabled reproduces that
        ordering exactly as it happens at real server startup.
        """
        module = load_server({"EMBEDDER_PROFILE": "advanced"}, cuda_available=True)

        self.assertTrue(hasattr(module, "QWEN_RERANK_EFFECTIVE_MAX_BATCH_TOKENS"))
        self.assertIsInstance(module.QWEN_RERANK_EFFECTIVE_MAX_BATCH_TOKENS, int)

    def test_embeddings_response_model_declares_backend_metadata(self):
        import typing

        module = load_server()
        fields = module.EmbeddingsListResponse.__annotations__
        self.assertIs(fields["model_name"], str)
        self.assertIs(fields["dense_model_name"], str)
        self.assertIs(fields["sparse_model_name"], str)
        # colbert is None in the advanced profile, so the field is Optional.
        self.assertIn(type(None), typing.get_args(fields["colbert_model_name"]))

    def test_api_warning_models_capture_target_and_details(self):
        module = load_server()

        warning = module.ApiWarning(
            code="input_truncated",
            message="Input text was truncated to the model token limit.",
            target=module.WarningTarget(
                field="sentences",
                index=1,
                pointer="/sentences/1",
            ),
            details=module.WarningDetails(
                model=module.BGE_EMBEDDING_MODEL,
                max_tokens=module.BGE_EMBED_MAX_LENGTH,
                original_tokens=module.BGE_EMBED_MAX_LENGTH + 5,
                truncated_tokens=5,
            ),
        )

        self.assertEqual(warning.code, "input_truncated")
        self.assertEqual(warning.severity, "warning")
        self.assertEqual(warning.target.field, "sentences")
        self.assertEqual(warning.target.index, 1)
        self.assertEqual(warning.details.truncation_side, "end")

    def test_token_count_suppresses_transformers_length_warning_when_supported(self):
        module = load_server()

        class VerboseAwareTokenizer:
            """Tokenizer recording the verbose flag each encode was given."""

            def __init__(self):
                self.verbose_values = []

            def encode(self, text, add_special_tokens=False, verbose=True):
                self.verbose_values.append(verbose)
                return [1, 2, 3]

        tokenizer = VerboseAwareTokenizer()

        self.assertEqual(module._count_tokens(tokenizer, "long input"), 3)
        self.assertEqual(tokenizer.verbose_values, [False])

    def test_embedding_truncation_warnings_are_request_local(self):
        module = load_server({"BGE_EMBED_MAX_LENGTH": "4"})
        request = module.EmbedRequest(
            sentences=["tok:3", "tok:7"],
            return_dense=True,
            return_sparse=False,
            return_colbert=False,
            normalize_dense=False,
            sparse_as_indices=False,
        )

        warnings = module.collect_embedding_truncation_warnings(request)

        self.assertEqual(len(warnings), 1)
        warning = warnings[0]
        self.assertEqual(warning.code, "input_truncated")
        self.assertEqual(warning.target.field, "sentences")
        self.assertEqual(warning.target.index, 1)
        self.assertEqual(warning.target.pointer, "/sentences/1")
        self.assertEqual(warning.details.max_tokens, 4)
        self.assertEqual(warning.details.original_tokens, 7)
        self.assertEqual(warning.details.truncated_tokens, 3)

    def test_embeddings_endpoint_uses_bge_model_name_by_default(self):
        module = load_server()

        async def fake_process_request(request, *, lease=None):
            return {"dense_vecs": [[1.0]]}

        module.processor.process_request = fake_process_request
        request = module.EmbedRequest(
            sentences=["alpha"],
            return_dense=True,
            return_sparse=False,
            return_colbert=False,
            normalize_dense=False,
            sparse_as_indices=False,
        )

        response = asyncio.run(module.get_embeddings(request))
        self.assertEqual(response.model_name, module.BGE_EMBEDDING_MODEL)
        self.assertEqual(response.dense_model_name, module.BGE_EMBEDDING_MODEL)

    def test_embeddings_endpoint_returns_truncation_warnings(self):
        module = load_server({"BGE_EMBED_MAX_LENGTH": "4"})

        async def fake_process_request(request, *, lease=None):
            return {
                "dense_vecs": [[1.0], [2.0]],
                "_warnings": module.collect_embedding_truncation_warnings(request),
            }

        module.processor.process_request = fake_process_request
        request = module.EmbedRequest(
            sentences=["tok:2", "tok:6"],
            return_dense=True,
            return_sparse=False,
            return_colbert=False,
            normalize_dense=False,
            sparse_as_indices=False,
        )

        response = asyncio.run(module.get_embeddings(request))

        self.assertEqual(len(response.warnings), 1)
        self.assertEqual(response.warnings[0].code, "input_truncated")
        self.assertEqual(response.warnings[0].target.index, 1)
        self.assertEqual(response.warnings[0].details.truncated_tokens, 2)
        self.assertEqual(len(response.data), 2)

    def test_qwen_dense_backend_returns_1024_dimension_vectors(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        backend = module.embedding_service.qwen_dense_backend
        result = backend.embed_dense(["alpha"], normalize_dense=False)
        self.assertEqual(len(result["dense_vecs"][0]), 1024)

    def test_qwen_dense_backend_applies_normalization_when_requested(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        backend = module.embedding_service.qwen_dense_backend
        result = backend.embed_dense(["alpha"], normalize_dense=True)
        self.assertEqual(len(result["dense_vecs"][0]), 1024)
        self.assertTrue(backend.model.last_dense_tensor.normalized)

    def test_qwen_dense_backend_wrong_dimension_raises_runtime_error(self):
        module = load_server(
            {
                "EMBEDDER_PROFILE": "advanced",
                "FAKE_QWEN_DENSE_DIM": "768",
            }
        )
        backend = module.embedding_service.qwen_dense_backend
        with self.assertRaisesRegex(RuntimeError, "expected 1024"):
            backend.embed_dense(["alpha"], normalize_dense=False)

    def test_qwen_dense_load_failure_fails_server_startup(self):
        with self.assertRaisesRegex(RuntimeError, "qwen dense load failed"):
            load_server(
                {
                    "EMBEDDER_PROFILE": "advanced",
                    "FAKE_QWEN_DENSE_LOAD_ERROR": "1",
                }
            )

    def test_capacity_counters_record_increments_and_labels(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})

        # FakeMetric (tests/test_reranker_wrapper.py:406) records calls; it has
        # no .value, and labels() returns self rather than a per-label child.
        module.rerank_batch_halvings.inc()
        module.rerank_batch_halvings.inc()
        self.assertEqual(module.rerank_batch_halvings.increment_calls, [1, 1])

        module.rerank_capacity_rejections.labels(reason="oversized").inc()
        module.rerank_capacity_rejections.labels(reason="pressure").inc()
        self.assertEqual(
            module.rerank_capacity_rejections.label_calls,
            [((), {"reason": "oversized"}), ((), {"reason": "pressure"})],
        )
        self.assertEqual(module.rerank_capacity_rejections.increment_calls, [1, 1])

    def test_rejection_counter_is_incremented_by_the_capacity_response(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        module._STARTUP_HEADROOM_BYTES = 2_287_634_944
        failure = module.vram_budget.VramExhausted(
            batch_items=1,
            batch_tokens=8100,
            estimated_bytes=2_457_037_800,
            original="OutOfMemoryError: CUDA out of memory",
        )

        module._rerank_capacity_response(failure, passage_count=200)

        self.assertIn(
            ((), {"reason": "oversized"}),
            module.rerank_capacity_rejections.label_calls,
        )
        self.assertEqual(module.rerank_capacity_rejections.increment_calls, [1])


class RerankOffEventLoopTests(unittest.TestCase):
    """Pair preparation runs off the event loop, not inside it."""

    def test_rerank_prepares_pairs_off_event_loop(self):
        import threading

        module = load_server()
        main_thread = threading.get_ident()
        seen_threads = []
        original_prepare = module.reranker.prepare_pairs_for_scoring

        def tracking_prepare(query, passages):
            seen_threads.append(threading.get_ident())
            return original_prepare(query, passages)

        module.reranker.prepare_pairs_for_scoring = tracking_prepare
        request = module.RerankRequest(query="q", passages=["a", "b"])

        asyncio.run(module.rerank(request))

        # prepare tokenizes with the reranker's tokenizer; if it runs on the
        # event-loop thread while a previous request scores on the executor
        # worker, the shared fast tokenizer raises "Already borrowed". Running
        # prepare inside the single-worker executor serializes it with scoring.
        self.assertTrue(seen_threads)
        self.assertTrue(
            all(t != main_thread for t in seen_threads),
            "rerank must prepare/tokenize off the event-loop thread",
        )


if __name__ == "__main__":
    unittest.main()
