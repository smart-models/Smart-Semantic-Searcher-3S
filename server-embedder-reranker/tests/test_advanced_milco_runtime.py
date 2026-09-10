"""Tests for the MILCO learned-sparse runtime used by the advanced profile.

Tiny stand-in ``torch`` modules replace the real encoder and head so the
runtime's pooling, masking and output shaping can be checked directly.
"""

import importlib
import math
import sys
import types
import unittest

import torch


def load_runtime():
    if "advanced_milco_runtime" in sys.modules:
        return importlib.reload(sys.modules["advanced_milco_runtime"])
    return importlib.import_module("advanced_milco_runtime")


class TinyEncoder(torch.nn.Module):
    """Two-layer stand-in for the MILCO text encoder."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return types.SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TrackingHead(torch.nn.Module):
    """Sparse head stand-in recording the inputs it was called with."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5)
        self.sequence_lengths = []

    def forward(self, hidden_states):
        self.sequence_lengths.append(hidden_states.size(1))
        return self.linear(hidden_states)


class TinyMilco(torch.nn.Module):
    """Minimal MILCO model pairing TinyEncoder with TrackingHead."""

    def __init__(self):
        super().__init__()
        self.m_model = TinyEncoder()
        self.projector = torch.nn.Linear(4, 4)
        self.activation = torch.nn.Tanh()
        self.mlm_head = TrackingHead()
        self.echo = torch.nn.Linear(4, 1)
        self.config = types.SimpleNamespace(en_vocab_size=5, m_vocab_size=16)


class MilcoRuntimeTests(unittest.TestCase):
    """Chunked projection must match the equivalent full projection."""

    def setUp(self):
        torch.manual_seed(7)
        self.runtime = load_runtime()
        self.model = TinyMilco().eval()
        self.input_ids = torch.tensor(
            [[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]],
            dtype=torch.long,
        )
        self.attention_mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]],
            dtype=torch.long,
        )

    def reference_dense(self, source_view):
        hidden_m = self.model.m_model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
        ).last_hidden_state
        hidden_e = self.model.activation(self.model.projector(hidden_m))
        logits = self.model.mlm_head.linear(hidden_e)
        logits = logits * self.attention_mask.unsqueeze(-1)
        pivot = torch.log1p(torch.relu(logits.max(dim=1).values))
        if not source_view:
            return pivot

        source = torch.relu(self.model.echo(hidden_m).squeeze(-1))
        source = source * self.attention_mask
        expected = torch.zeros(2, 21)
        expected[:, :5] = pivot
        for row in range(self.input_ids.size(0)):
            for column in range(self.input_ids.size(1)):
                weight = source[row, column]
                if weight > 0:
                    expected[row, 5 + self.input_ids[row, column]] += weight
        return expected

    def test_chunked_pivot_matches_full_projection(self):
        expected = self.reference_dense(source_view=False)

        actual = self.runtime.memory_bounded_milco_forward(
            self.model,
            self.input_ids,
            self.attention_mask,
            projection_chunk_tokens=2,
            source_view=False,
        ).to_dense()

        torch.testing.assert_close(actual, expected)
        self.assertEqual(self.model.mlm_head.sequence_lengths, [2, 2, 1])

    def test_chunked_lexecho_matches_full_dual_view(self):
        expected = self.reference_dense(source_view=True)

        actual = self.runtime.memory_bounded_milco_forward(
            self.model,
            self.input_ids,
            self.attention_mask,
            projection_chunk_tokens=2,
            source_view=True,
        ).to_dense()

        torch.testing.assert_close(actual, expected)
        self.assertEqual(
            len(self.model.mlm_head.sequence_lengths),
            math.ceil(self.input_ids.size(1) / 2),
        )

    def test_rejects_non_positive_projection_chunk(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self.runtime.memory_bounded_milco_forward(
                self.model,
                self.input_ids,
                self.attention_mask,
                projection_chunk_tokens=0,
                source_view=True,
            )


if __name__ == "__main__":
    unittest.main()
