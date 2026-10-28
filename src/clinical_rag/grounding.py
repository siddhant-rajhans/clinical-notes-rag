from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from clinical_rag.config import RAGConfig

logger = logging.getLogger(__name__)

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_ABBREVIATIONS = frozenset(
    "dr mr mrs ms jr sr vs etc approx pt mg ml dl kg lb oz hr min sec "
    "vol no dept div est govt inc ltd corp assn natl intl".split()
)


@dataclass
class GroundingResult:
    """Entailment score and per-sentence breakdown from the NLI grounding check."""
    score: float
    is_grounded: bool
    per_sentence_scores: list[float]


class GroundingChecker:
    """NLI cross-encoder that scores whether an answer is entailed by retrieved context."""

    _DEFAULT_ENTAILMENT_IDX = 1

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self._model: CrossEncoder | None = None
        self._entailment_idx: int | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            logger.info("Loading grounding model: %s", self.config.grounding_model)
            self._model = CrossEncoder(self.config.grounding_model)
            self._entailment_idx = self._detect_entailment_idx(self._model)
        return self._model

    @property
    def entailment_idx(self) -> int:
        if self._entailment_idx is None:
            self._entailment_idx = self._detect_entailment_idx(self.model)
        return self._entailment_idx

    @staticmethod
    def _detect_entailment_idx(model: CrossEncoder) -> int:
        """Read the model's label config to find the entailment class index."""
        try:
            cfg = model.model.config
            if hasattr(cfg, "label2id"):
                for label, idx in cfg.label2id.items():
                    if label.lower() == "entailment":
                        return int(idx)
            if hasattr(cfg, "id2label"):
                for idx, label in cfg.id2label.items():
                    if label.lower() == "entailment":
                        return int(idx)
        except Exception:
            pass
        logger.warning(
            "Could not detect entailment index from model config; falling back to %d",
            GroundingChecker._DEFAULT_ENTAILMENT_IDX,
        )
        return GroundingChecker._DEFAULT_ENTAILMENT_IDX

    def check(
        self,
        answer: str,
        context_docs: list[Document],
    ) -> GroundingResult:
        """Score how well the answer is grounded in the context."""
        context = "\n\n".join(doc.page_content for doc in context_docs)
        sentences = _split_sentences(answer)

        if not sentences:
            return GroundingResult(score=0.0, is_grounded=False, per_sentence_scores=[])

        pairs = [(context, sent) for sent in sentences]
        logits = self.model.predict(pairs, apply_softmax=True)

        per_sentence: list[float] = []
        for row in logits:
            entailment_prob = float(row[self.entailment_idx])
            per_sentence.append(entailment_prob)

        score = sum(per_sentence) / len(per_sentence)
        is_grounded = score >= self.config.grounding_threshold

        logger.info(
            "Grounding: score=%.3f, grounded=%s (%d/%d sentences above threshold)",
            score,
            is_grounded,
            sum(1 for s in per_sentence if s >= self.config.grounding_threshold),
            len(per_sentence),
        )

        return GroundingResult(
            score=score,
            is_grounded=is_grounded,
            per_sentence_scores=per_sentence,
        )


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling common clinical abbreviations."""
    parts = _SENTENCE_END_RE.split(text.strip())

    merged: list[str] = []
    for part in parts:
        last_word = part.rstrip(".!?").rsplit(None, 1)[-1].lower() if part.rstrip(".!?") else ""
        if merged and last_word in _ABBREVIATIONS:
            merged[-1] = merged[-1] + " " + part
        elif merged and len(part.strip()) <= 10:
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)

    return [s.strip() for s in merged if len(s.strip()) > 10]
