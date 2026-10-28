from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from ragas import evaluate as ragas_evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from clinical_rag.config import RAGConfig
from clinical_rag.embeddings import ClinicalEmbedder
from eval.dataset import EvalQuestion

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    question_id: str
    question: str
    answer: str
    expected_answer: str
    faithfulness: float
    answer_correctness: float
    context_precision: float
    context_recall: float
    grounding_score: float | None
    is_grounded: bool | None
    retrieved_note_ids: list[str]
    expected_note_ids: list[str]
    recall_at_k: float


class RAGEvaluator:
    """Evaluates RAG pipeline using RAGAS metrics."""

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()

    def evaluate_single(
        self,
        question: EvalQuestion,
        answer: str,
        retrieved_docs: list[Document],
        grounding_score: float | None = None,
        is_grounded: bool | None = None,
    ) -> EvalResult:
        """Evaluate a single QA result against ground truth."""
        retrieved_ids = [
            doc.metadata.get("note_id", "") for doc in retrieved_docs
        ]
        expected_ids = question.relevant_note_ids

        hit_count = sum(1 for rid in retrieved_ids if rid in expected_ids)
        recall_at_k = hit_count / len(expected_ids) if expected_ids else 0.0

        contexts = [doc.page_content for doc in retrieved_docs]

        sample = SingleTurnSample(
            user_input=question.question,
            response=answer,
            reference=question.expected_answer,
            retrieved_contexts=contexts,
        )

        eval_dataset = EvaluationDataset(samples=[sample])

        eval_llm = LangchainLLMWrapper(ChatOpenAI(model=self.config.llm_model, temperature=0))

        embedder = ClinicalEmbedder(self.config)

        class _EmbeddingAdapter:
            def __init__(self, clinical_embedder: ClinicalEmbedder):
                self._embedder = clinical_embedder

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return self._embedder.embed_texts(texts, show_progress=False).tolist()

            def embed_query(self, text: str) -> list[float]:
                return self._embedder.embed_query(text).tolist()

        eval_embeddings = LangchainEmbeddingsWrapper(_EmbeddingAdapter(embedder))

        metrics = [
            Faithfulness(llm=eval_llm),
            AnswerCorrectness(llm=eval_llm, embeddings=eval_embeddings),
            ContextPrecision(llm=eval_llm),
            ContextRecall(llm=eval_llm),
        ]

        results = ragas_evaluate(dataset=eval_dataset, metrics=metrics)
        scores = results.to_pandas().iloc[0]

        return EvalResult(
            question_id=question.question_id,
            question=question.question,
            answer=answer,
            expected_answer=question.expected_answer,
            faithfulness=float(scores.get("faithfulness", 0.0)),
            answer_correctness=float(scores.get("answer_correctness", 0.0)),
            context_precision=float(scores.get("context_precision", 0.0)),
            context_recall=float(scores.get("context_recall", 0.0)),
            grounding_score=grounding_score,
            is_grounded=is_grounded,
            retrieved_note_ids=retrieved_ids,
            expected_note_ids=expected_ids,
            recall_at_k=recall_at_k,
        )
