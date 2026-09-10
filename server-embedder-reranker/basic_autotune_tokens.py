"""Deterministic exact-token inputs used by BASIC hardware calibration.

``CalibrationTokenCounter`` synthesizes reproducible uniform and mixed-length
workloads against the active tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CALIBRATION_TEXT_SEEDS = (
    (
        "Modern retrieval systems transform multilingual documents into dense and "
        "sparse vectors while preserving semantic relationships across languages, "
        "domains, and document lengths. Efficient batching reduces padding overhead, "
        "keeps accelerators busy, and improves throughput without changing the "
        "embedding model or its numerical behavior."
    ),
    (
        "Un motore di ricerca semantica prepara i documenti, calcola rappresentazioni "
        "vettoriali e combina segnali lessicali con similarità dense. Una pipeline "
        "efficiente misura separatamente tokenizzazione, inferenza, trasferimenti di "
        "memoria e serializzazione, mantenendo latenza e affidabilità sotto controllo."
    ),
    (
        "Reliable performance engineering uses deterministic inputs, repeatable warmup "
        "cycles, exact token counts, and explicit error reporting. Measurements cover "
        "short queries, medium passages, and long documents so that optimization "
        "decisions reflect both interactive latency and sustained service throughput."
    ),
    (
        "Vector databases support semantic search by indexing numerical "
        "representations of text. Production systems also need backpressure, bounded "
        "queues, health "
        "checks, authentication, and observable resource usage because raw model speed "
        "alone does not guarantee stable end-to-end application performance."
    ),
)


@dataclass(frozen=True)
class CalibrationTokenCounter:
    """Generate deterministic texts with exact model-token lengths."""

    tokenizer: Any
    model_name: str
    max_length: int

    def count(self, texts: list[str]) -> int:
        """Count post-truncation tokens across a text batch.

        Args:
            texts: Input texts measured without padding.

        Returns:
            Total token count including model special tokens.
        """
        if not texts:
            return 0
        encoded = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        return sum(len(input_ids) for input_ids in encoded["input_ids"])

    def texts_at_token_length(self, size: int, target_tokens: int) -> list[str]:
        """Build deterministic texts of one exact token length.

        Args:
            size: Number of texts to generate.
            target_tokens: Exact per-text token length.

        Returns:
            ``size`` texts that reproduce the requested model-token length.

        Raises:
            ValueError: If the length is out of range or cannot be synthesized.
        """
        if target_tokens < 1 or target_tokens > self.max_length:
            raise ValueError(
                f"target token length must be between 1 and {self.max_length}"
            )

        fillers = (
            "benchmark ",
            "performance ",
            "data ",
            "test ",
            "vector ",
            "semantic ",
            "retrieval ",
            "embedding ",
        )
        texts = []
        for index in range(size):
            seed_words = CALIBRATION_TEXT_SEEDS[
                index % len(CALIBRATION_TEXT_SEEDS)
            ].split()
            copies = (target_tokens * 2 // len(seed_words)) + 2
            word_stream = seed_words * copies

            low = 0
            high = len(word_stream)
            base_text = ""
            while low <= high:
                word_count = (low + high) // 2
                text = " ".join(word_stream[:word_count]) + " "
                actual_tokens = self.count([text])
                if actual_tokens <= target_tokens:
                    base_text = text
                    low = word_count + 1
                else:
                    high = word_count - 1

            exact_text = base_text if self.count([base_text]) == target_tokens else ""
            for filler_offset in range(len(fillers)):
                if exact_text:
                    break
                filler = fillers[(index + filler_offset) % len(fillers)]
                low = 0
                high = target_tokens
                while low <= high:
                    repeats = (low + high) // 2
                    text = base_text + filler * repeats
                    actual_tokens = self.count([text])
                    if actual_tokens == target_tokens:
                        exact_text = text
                        break
                    if actual_tokens < target_tokens:
                        low = repeats + 1
                    else:
                        high = repeats - 1

            if not exact_text:
                raise ValueError(
                    f"could not synthesize an exact {target_tokens}-token input "
                    f"with tokenizer '{self.model_name}'"
                )
            texts.append(exact_text)
        return texts

    def texts_at_token_pattern(self, size: int, target_tokens: list[int]) -> list[str]:
        """Cycle exact token lengths across a deterministic text batch.

        Args:
            size: Number of texts to generate.
            target_tokens: Non-empty sequence of exact per-text lengths.

        Returns:
            Texts whose lengths repeat the supplied pattern.

        Raises:
            ValueError: If ``target_tokens`` is empty.
        """
        if not target_tokens:
            raise ValueError("mixed token workloads require at least one length")
        generated = {
            target: self.texts_at_token_length(size, target)
            for target in dict.fromkeys(target_tokens)
        }
        return [
            generated[target_tokens[index % len(target_tokens)]][index]
            for index in range(size)
        ]
