from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence


class TextAugmentationBank:
    def __init__(
        self,
        num_positive: int,
        num_negative: int,
        json_path: str | None = None,
        seed: int = 42,
    ) -> None:
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.seed = seed
        self.records: Dict[str, Dict] = {}
        if json_path and Path(json_path).is_file():
            with Path(json_path).open("r", encoding="utf-8") as handle:
                self.records = json.load(handle)
        self.corpus: List[str] = []

    def register_corpus(self, texts: Sequence[str]) -> None:
        normalized = []
        for text in texts:
            text = " ".join(text.lower().split())
            if text:
                normalized.append(text)
        self.corpus = sorted(set(normalized))

    def _rng(self, sample_id: str) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}:{sample_id}".encode("utf-8")).hexdigest()
        return random.Random(int(digest[:16], 16))

    def _fallback_negatives(self, sample_id: str, original_text: str) -> List[str]:
        if not self.corpus:
            return ["background object"] * self.num_negative
        pool = [text for text in self.corpus if text != original_text]
        if not pool:
            return ["background object"] * self.num_negative
        rng = self._rng(sample_id)
        negatives = []
        while len(negatives) < self.num_negative:
            negatives.append(rng.choice(pool))
        return negatives

    def get(self, sample_id: str, original_text: str) -> Dict[str, List]:
        original_text = " ".join(original_text.lower().split())
        record = self.records.get(sample_id, {})

        positives = [original_text]
        confidences = [1.0]
        for text, score in zip(
            record.get("positive_texts", []),
            record.get("positive_confidences", record.get("positive_scores", [])),
        ):
            text = " ".join(str(text).lower().split())
            if text and text != original_text:
                positives.append(text)
                confidences.append(float(score))

        while len(positives) < self.num_positive:
            positives.append(original_text)
            confidences.append(1.0)
        positives = positives[: self.num_positive]
        confidences = confidences[: self.num_positive]

        negatives = [" ".join(str(text).lower().split()) for text in record.get("negative_texts", [])]
        negatives = [text for text in negatives if text and text != original_text]
        if len(negatives) < self.num_negative:
            negatives.extend(self._fallback_negatives(sample_id, original_text))
        negatives = negatives[: self.num_negative]

        return {
            "original_text": original_text,
            "positive_texts": positives,
            "positive_confidences": confidences,
            "negative_texts": negatives,
        }
