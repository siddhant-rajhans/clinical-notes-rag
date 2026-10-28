from unittest.mock import MagicMock, patch

import numpy as np
from langchain_core.documents import Document

from clinical_rag.config import RAGConfig
from clinical_rag.grounding import GroundingChecker, _split_sentences


class TestSplitSentences:
    def test_splits_basic_sentences(self):
        text = "Patient has diabetes. Blood pressure is 140/90. No allergies reported."
        sentences = _split_sentences(text)
        assert len(sentences) == 3

    def test_filters_short_fragments(self):
        text = "OK. Yes. The patient was discharged in stable condition."
        sentences = _split_sentences(text)
        assert len(sentences) == 1

    def test_empty_text(self):
        assert _split_sentences("") == []


class TestGroundingChecker:
    @patch("clinical_rag.grounding.CrossEncoder")
    def test_grounded_answer(self, mock_ce_cls):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([
            [0.05, 0.10, 0.85],
            [0.03, 0.12, 0.85],
        ])
        mock_ce_cls.return_value = mock_model

        config = RAGConfig(grounding_threshold=0.6)
        checker = GroundingChecker(config)
        checker._model = mock_model

        docs = [Document(page_content="The patient has Type 2 diabetes managed with metformin.")]
        result = checker.check(
            "The patient has diabetes. They take metformin for glucose control.",
            docs,
        )

        assert result.is_grounded is True
        assert result.score > 0.6
        assert len(result.per_sentence_scores) == 2

    @patch("clinical_rag.grounding.CrossEncoder")
    def test_ungrounded_answer(self, mock_ce_cls):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([
            [0.60, 0.30, 0.10],
            [0.50, 0.40, 0.10],
        ])
        mock_ce_cls.return_value = mock_model

        config = RAGConfig(grounding_threshold=0.6)
        checker = GroundingChecker(config)
        checker._model = mock_model

        docs = [Document(page_content="Patient presented with cough.")]
        result = checker.check(
            "The patient has cancer. They need surgery immediately.",
            docs,
        )

        assert result.is_grounded is False
        assert result.score < 0.6

    @patch("clinical_rag.grounding.CrossEncoder")
    def test_empty_answer(self, mock_ce_cls):
        config = RAGConfig(grounding_threshold=0.6)
        checker = GroundingChecker(config)
        checker._model = MagicMock()

        docs = [Document(page_content="Some context.")]
        result = checker.check("", docs)

        assert result.is_grounded is False
        assert result.score == 0.0
