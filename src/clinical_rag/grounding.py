from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from clinical_rag.config import RAGConfig

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    score: float
    is_grounded: bool
    per_sentence_scores: list[float]


class GroundingChecker:
    """NLI-based grounding checker.

    Verifies whether a generated answer is entailed by the retrieved context
    by scoring each answer sentence against the concatenated context with a
    natural language inference cross-encoder.

    Labels (nli-deberta-v3-base): 0 = contradiction, 1 = entailment, 2 = neutral.
    """

    ENTAILMENT_IDX = 1

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            logger.info("Loading grounding model: %s", self.config.grounding_model)
            self._model = CrossEncoder(self.config.grounding_model)
        return self._model

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
            entailment_prob = float(row[self.ENTAILMENT_IDX])
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
    """Simple sentence splitter for clinical text."""
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]
