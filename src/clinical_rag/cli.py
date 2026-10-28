from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from clinical_rag.config import RAGConfig

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")


@click.group()
@click.option("--verbose", "-v", is_flag=True)
def cli(verbose: bool) -> None:
    """Clinical Notes RAG - question answering over clinical notes."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option("--data-dir", type=click.Path(exists=True), required=True)
@click.option("--index-dir", type=click.Path(), default="data/index")
@click.option("--chunk-size", type=int, default=512)
@click.option("--chunk-overlap", type=int, default=64)
@click.option("--use-ivf", is_flag=True, help="Use IVF index for large collections")
def ingest(
    data_dir: str,
    index_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    use_ivf: bool,
) -> None:
    """Ingest clinical notes and build the FAISS index."""
    from clinical_rag.index import FAISSIndex
    from clinical_rag.ingest.chunker import chunk_documents
    from clinical_rag.ingest.loader import load_all

    config = RAGConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_ivf=use_ivf,
        index_dir=Path(index_dir),
        data_dir=Path(data_dir),
    )

    click.echo(f"Loading documents from {data_dir}...")
    documents = load_all(Path(data_dir))
    click.echo(f"Loaded {len(documents)} documents")

    click.echo("Chunking documents...")
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    click.echo(f"Created {len(chunks)} chunks")

    click.echo("Building FAISS index...")
    index = FAISSIndex.build(chunks, config)
    index.save(Path(index_dir))
    click.echo(f"Index saved to {index_dir} ({index.index.ntotal} vectors)")


@cli.command()
@click.argument("question")
@click.option("--index-dir", type=click.Path(exists=True), default="data/index")
@click.option("--model", default="gpt-4o-mini")
@click.option("--grounding-threshold", type=float, default=0.6)
@click.option("--json-output", is_flag=True, help="Output as JSON")
def query(
    question: str,
    index_dir: str,
    model: str,
    grounding_threshold: float,
    json_output: bool,
) -> None:
    """Ask a question over the indexed clinical notes."""
    from clinical_rag.graph import build_rag_graph
    from clinical_rag.index import FAISSIndex
    from clinical_rag.retrieval.retriever import TwoStageRetriever

    config = RAGConfig(
        llm_model=model,
        grounding_threshold=grounding_threshold,
        index_dir=Path(index_dir),
    )

    index = FAISSIndex.load(Path(index_dir), config)
    retriever = TwoStageRetriever(index, config)
    graph = build_rag_graph(retriever, config)

    result = graph.invoke({
        "question": question,
        "documents": [],
        "relevant_documents": [],
        "answer": "",
        "grounding": None,
        "messages": [],
    })

    if json_output:
        output = {
            "question": question,
            "answer": result["answer"],
            "grounding_score": result["grounding"].score if result.get("grounding") else None,
            "is_grounded": result["grounding"].is_grounded if result.get("grounding") else None,
            "sources": [
                {
                    "note_id": doc.metadata.get("note_id"),
                    "section": doc.metadata.get("section"),
                    "specialty": doc.metadata.get("specialty"),
                }
                for doc in result.get("relevant_documents", [])
            ],
        }
        click.echo(json.dumps(output, indent=2))
    else:
        click.echo(f"\nQuestion: {question}\n")
        click.echo(f"Answer: {result['answer']}\n")
        if grounding := result.get("grounding"):
            click.echo(f"Grounding score: {grounding.score:.3f}")
            click.echo(f"Grounded: {grounding.is_grounded}")
        click.echo("\nSources:")
        for doc in result.get("relevant_documents", []):
            note_id = doc.metadata.get("note_id", "unknown")
            section = doc.metadata.get("section", "")
            click.echo(f"  - {note_id} ({section})")
