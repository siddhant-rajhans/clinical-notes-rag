"""Run the full evaluation suite over the clinical notes RAG pipeline."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from clinical_rag.config import RAGConfig

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--index-dir", type=click.Path(exists=True), default="data/index")
@click.option("--eval-data", type=click.Path(exists=True), default="data/eval_questions.json")
@click.option("--output", type=click.Path(), default="eval_results.json")
@click.option("--model", default="gpt-4o-mini")
@click.option("--grounding-threshold", type=float, default=0.6)
@click.option("--max-questions", type=int, default=None, help="Limit questions for quick runs")
def run_eval(
    index_dir: str,
    eval_data: str,
    output: str,
    model: str,
    grounding_threshold: float,
    max_questions: int | None,
) -> None:
    """Run RAGAS evaluation over labeled QA pairs."""
    from clinical_rag.graph import build_rag_graph
    from clinical_rag.index import FAISSIndex
    from clinical_rag.retrieval.retriever import TwoStageRetriever
    from eval.dataset import load_eval_dataset
    from eval.evaluate import RAGEvaluator

    config = RAGConfig(
        llm_model=model,
        grounding_threshold=grounding_threshold,
    )

    click.echo("Loading index...")
    index = FAISSIndex.load(Path(index_dir), config)
    retriever = TwoStageRetriever(index, config)
    graph = build_rag_graph(retriever, config)
    evaluator = RAGEvaluator(config)

    click.echo("Loading eval dataset...")
    questions = load_eval_dataset(Path(eval_data))
    if max_questions:
        questions = questions[:max_questions]
    click.echo(f"Evaluating {len(questions)} questions")

    results = []
    grounded_count = 0
    ungrounded_refused_count = 0
    total_recall = 0.0

    for i, q in enumerate(questions):
        click.echo(f"\n[{i+1}/{len(questions)}] {q.question[:60]}...")

        state = graph.invoke({
            "question": q.question,
            "documents": [],
            "relevant_documents": [],
            "answer": "",
            "grounding": None,
            "messages": [],
        })

        grounding = state.get("grounding")
        g_score = grounding.score if grounding else None
        g_grounded = grounding.is_grounded if grounding else None

        if g_grounded:
            grounded_count += 1
        elif g_grounded is False:
            ungrounded_refused_count += 1

        eval_result = evaluator.evaluate_single(
            question=q,
            answer=state["answer"],
            retrieved_docs=state.get("relevant_documents", []),
            grounding_score=g_score,
            is_grounded=g_grounded,
        )
        total_recall += eval_result.recall_at_k

        results.append({
            "question_id": eval_result.question_id,
            "question": eval_result.question,
            "answer": eval_result.answer,
            "faithfulness": eval_result.faithfulness,
            "answer_correctness": eval_result.answer_correctness,
            "context_precision": eval_result.context_precision,
            "context_recall": eval_result.context_recall,
            "grounding_score": eval_result.grounding_score,
            "is_grounded": eval_result.is_grounded,
            "recall_at_k": eval_result.recall_at_k,
        })

    n = len(results)
    avg_faithfulness = sum(r["faithfulness"] for r in results) / n if n else 0
    avg_correctness = sum(r["answer_correctness"] for r in results) / n if n else 0
    avg_precision = sum(r["context_precision"] for r in results) / n if n else 0
    avg_recall_at_k = total_recall / n if n else 0

    summary = {
        "total_questions": n,
        "avg_faithfulness": round(avg_faithfulness, 3),
        "avg_answer_correctness": round(avg_correctness, 3),
        "avg_context_precision": round(avg_precision, 3),
        "avg_recall_at_10": round(avg_recall_at_k, 3),
        "grounded_answers": grounded_count,
        "refused_ungrounded": ungrounded_refused_count,
        "grounding_refusal_rate": round(ungrounded_refused_count / n, 3) if n else 0,
    }

    output_data = {"summary": summary, "results": results}
    Path(output).write_text(json.dumps(output_data, indent=2))

    click.echo("\n" + "=" * 60)
    click.echo("EVALUATION RESULTS")
    click.echo("=" * 60)
    for key, val in summary.items():
        click.echo(f"  {key}: {val}")
    click.echo(f"\nResults saved to {output}")


if __name__ == "__main__":
    run_eval()
