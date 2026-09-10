"""Tests for the basic autotuner's calibration token counter.

A word-splitting stand-in replaces the real tokenizer, so the counter's
accounting is checked independently of any vocabulary.
"""

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from basic_autotune_tokens import CalibrationTokenCounter  # noqa: E402


class WordTokenizer:
    """Whitespace tokenizer stand-in, so counts do not depend on a vocabulary."""

    def __call__(self, texts, **kwargs):
        return {"input_ids": [[0] + text.split() + [1] for text in texts]}


class CalibrationTokenCounterTests(unittest.TestCase):
    """The counter builds exact uniform and mixed-length workloads."""

    def test_generates_exact_uniform_and_mixed_lengths(self):
        counter = CalibrationTokenCounter(WordTokenizer(), "test", 64)

        uniform = counter.texts_at_token_length(3, 16)
        mixed = counter.texts_at_token_pattern(4, [8, 16])

        self.assertEqual([counter.count([text]) for text in uniform], [16, 16, 16])
        self.assertEqual([counter.count([text]) for text in mixed], [8, 16, 8, 16])


if __name__ == "__main__":
    unittest.main()
