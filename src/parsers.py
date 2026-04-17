from __future__ import annotations

import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


class RuleBasedParser:
    def __init__(self) -> None:
        self.object_aliases: list[tuple[str, str]] = []
        self.phrase_to_action = [
            ("pick up", "PickupObject"),
            ("grab", "PickupObject"),
            ("take", "PickupObject"),
            ("put", "PutObject"),
            ("place", "PutObject"),
            ("drop", "PutObject"),
            ("open", "OpenObject"),
            ("close", "CloseObject"),
            ("turn on", "ToggleObjectOn"),
            ("switch on", "ToggleObjectOn"),
            ("turn off", "ToggleObjectOff"),
            ("switch off", "ToggleObjectOff"),
            ("clean", "CleanObject"),
            ("rinse", "CleanObject"),
            ("wash", "CleanObject"),
            ("heat", "HeatObject"),
            ("warm", "HeatObject"),
            ("microwave", "HeatObject"),
            ("cool", "CoolObject"),
            ("chill", "CoolObject"),
            ("slice", "SliceObject"),
            ("cut", "SliceObject"),
            ("examine", "ExamineObject"),
            ("look at", "ExamineObject"),
        ]

    def fit(self, records: list[dict]) -> "RuleBasedParser":
        alias_counts: Counter[str] = Counter()
        for record in records:
            for step in record["target_actions"]:
                for alias in self._expand_object_aliases(step["object"]):
                    alias_counts[alias] += 1
        ranked_aliases = sorted(alias_counts.items(), key=lambda item: (-len(item[0]), -item[1], item[0]))
        self.object_aliases = [(alias, alias) for alias, _ in ranked_aliases if alias]
        return self

    def predict(self, instruction: str) -> list[dict[str, str]]:
        segments = self._split_instruction(instruction)
        steps = [self._parse_segment(segment) for segment in segments]
        return [step for step in steps if step["action"]]

    def _split_instruction(self, instruction: str) -> list[str]:
        text = normalize_text(instruction)
        parts = re.split(r"\b(?:then|after that|next|and then)\b|[.;]", text)
        cleaned = [part.strip(" ,") for part in parts if part.strip(" ,")]
        return cleaned or [text]

    def _parse_segment(self, segment: str) -> dict[str, str]:
        action = self._match_action(segment)
        objects = self._match_objects(segment)
        return {
            "action": action,
            "object": " -> ".join(objects),
        }

    def _match_action(self, segment: str) -> str:
        for phrase, action in self.phrase_to_action:
            if phrase in segment:
                return action
        if segment.startswith("go to ") or segment.startswith("walk to "):
            return "GotoLocation"
        return "UnknownAction"

    def _match_objects(self, segment: str) -> list[str]:
        matches = []
        for alias, canonical in self.object_aliases:
            if alias and re.search(rf"\b{re.escape(alias)}\b", segment):
                if canonical not in matches:
                    matches.append(canonical)
        if matches:
            return matches[:2]
        tokens = re.findall(r"[a-z]+", segment)
        content_tokens = [token for token in tokens if token not in _STOPWORDS]
        return content_tokens[-1:] if content_tokens else [""]

    @staticmethod
    def _expand_object_aliases(object_string: str) -> set[str]:
        aliases = set()
        for chunk in object_string.split("->"):
            normalized = normalize_text(chunk)
            normalized = normalized.replace("_", " ")
            if normalized:
                aliases.add(normalized)
        return aliases


class RetrievalParser:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        self.train_vectors = None
        self.train_records: list[dict] = []

    def fit(self, records: list[dict]) -> "RetrievalParser":
        self.train_records = list(records)
        instructions = [record["instruction"] for record in records]
        self.train_vectors = self.vectorizer.fit_transform(instructions)
        return self

    def predict(self, instruction: str, instruction_type: str) -> list[dict[str, str]]:
        if self.train_vectors is None or not self.train_records:
            raise ValueError("RetrievalParser must be fitted before prediction.")

        query_vector = self.vectorizer.transform([instruction])
        similarities = cosine_similarity(query_vector, self.train_vectors).ravel()

        ranked_indices = np.argsort(similarities)[::-1]
        for index in ranked_indices:
            candidate = self.train_records[int(index)]
            if candidate["instruction_type"] == instruction_type:
                return candidate["target_actions"]
        return self.train_records[int(ranked_indices[0])]["target_actions"]


_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "from",
    "in",
    "into",
    "it",
    "of",
    "on",
    "the",
    "to",
    "up",
    "with",
}

