"""ADVANCED memory-bounded helpers for the MILCO learned-sparse runtime.

``build_dual_view_sparse()`` preserves pivot and LexEcho indices, while
``memory_bounded_milco_forward()`` chunks exact vocabulary projection.
"""

from __future__ import annotations

from typing import Any

import torch


def build_dual_view_sparse(
    pivot_view: torch.Tensor,
    source_ids: torch.Tensor,
    source_weights: torch.Tensor,
    *,
    english_vocab_size: int,
    multilingual_vocab_size: int,
) -> torch.Tensor:
    """Build MILCO's pivot + LexEcho COO tensor without changing its indices."""
    batch_size = source_ids.size(0)
    vocab_size = english_vocab_size + multilingual_vocab_size

    pivot_rows, pivot_cols = pivot_view.nonzero(as_tuple=True)
    pivot_values = pivot_view[pivot_rows, pivot_cols].contiguous()

    source_rows = torch.arange(
        batch_size,
        device=source_ids.device,
    ).repeat_interleave(source_ids.size(1))
    source_cols = source_ids.reshape(-1)
    source_values = source_weights.reshape(-1)
    nonzero = source_values > 0
    source_rows = source_rows[nonzero]
    source_cols = source_cols[nonzero] + english_vocab_size
    source_values = source_values[nonzero]

    indices = torch.stack(
        [
            torch.cat([pivot_rows, source_rows]),
            torch.cat([pivot_cols, source_cols]),
        ],
        dim=0,
    )
    values = torch.cat([pivot_values, source_values])
    return torch.sparse_coo_tensor(
        indices,
        values,
        (batch_size, vocab_size),
        device=pivot_view.device,
    ).coalesce()


def memory_bounded_milco_forward(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    projection_chunk_tokens: int,
    source_view: bool,
) -> torch.Tensor:
    """Run MILCO with sequence-chunked vocabulary projection.

    MILCO max-pools vocabulary logits over the sequence dimension. Max is
    associative, so projecting bounded sequence chunks and reducing their
    maxima is numerically equivalent to materialising ``[batch, seq, vocab]``.
    """
    if projection_chunk_tokens < 1:
        raise ValueError("projection_chunk_tokens must be positive")

    hidden_states_m = model.m_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state
    hidden_states_e = model.activation(model.projector(hidden_states_m))

    pivot_max = None
    sequence_length = hidden_states_e.size(1)
    for start in range(0, sequence_length, projection_chunk_tokens):
        end = min(sequence_length, start + projection_chunk_tokens)
        logits = model.mlm_head(hidden_states_e[:, start:end, :])
        logits = logits * attention_mask[:, start:end].unsqueeze(-1)
        chunk_max = logits.max(dim=1).values
        pivot_max = (
            chunk_max if pivot_max is None else torch.maximum(pivot_max, chunk_max)
        )

    if pivot_max is None:
        raise ValueError("MILCO received an empty token sequence")
    pivot_view = torch.log1p(torch.relu(pivot_max))

    if not source_view:
        return pivot_view.to_sparse().coalesce()

    source_scores = torch.relu(model.echo(hidden_states_m).squeeze(-1))
    source_weights = source_scores * attention_mask
    return build_dual_view_sparse(
        pivot_view,
        input_ids,
        source_weights,
        english_vocab_size=int(model.config.en_vocab_size),
        multilingual_vocab_size=int(model.config.m_vocab_size),
    )
