from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document

from clinical_rag.config import RAGConfig
from clinical_rag.embeddings import ClinicalEmbedder

logger = logging.getLogger(__name__)

DOCUMENTS_FILE = "documents.json"


class FAISSIndex:
    """FAISS vector index paired with a JSON document store."""

    def __init__(
        self,
        index: faiss.Index,
        documents: list[Document],
        embedder: ClinicalEmbedder,
    ):
        self.index = index
        self.documents = documents
        self.embedder = embedder

    @classmethod
    def build(
        cls,
        documents: list[Document],
        config: RAGConfig | None = None,
    ) -> FAISSIndex:
        """Build a FAISS index from a list of chunked documents."""
        config = config or RAGConfig()
        embedder = ClinicalEmbedder(config)

        logger.info("Embedding %d document chunks", len(documents))
        embeddings = embedder.embed_documents(documents)

        dim = embeddings.shape[1]

        if config.use_ivf and len(documents) > config.ivf_nlist * 40:
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, config.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            index.nprobe = config.ivf_nprobe
            logger.info("Built IVF index with nlist=%d", config.ivf_nlist)
        else:
            index = faiss.IndexFlatIP(dim)
            logger.info("Built flat inner-product index")

        index.add(embeddings)
        logger.info("Indexed %d vectors (dim=%d)", index.ntotal, dim)

        return cls(index=index, documents=documents, embedder=embedder)

    def search(self, query: str, top_k: int = 20) -> list[tuple[Document, float]]:
        """Search the index for the top_k most similar documents."""
        query_vec = self.embedder.embed_query(query).reshape(1, -1)
        scores, indices = self.index.search(query_vec, top_k)

        results: list[tuple[Document, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self.documents[idx]
            results.append((doc, float(score)))

        return results

    def save(self, directory: Path) -> None:
        """Persist the index and document metadata to disk."""
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / "index.faiss"))

        serialized = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in self.documents
        ]
        with open(directory / DOCUMENTS_FILE, "w") as f:
            json.dump(serialized, f)

        logger.info("Saved index (%d vectors) to %s", self.index.ntotal, directory)

    @classmethod
    def load(cls, directory: Path, config: RAGConfig | None = None) -> FAISSIndex:
        """Load a persisted index from disk."""
        config = config or RAGConfig()

        index_path = directory / "index.faiss"
        docs_path = directory / DOCUMENTS_FILE

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Run `clinical-rag ingest` to build the index first."
            )
        if not docs_path.exists():
            # Fall back to legacy pickle if JSON doesn't exist yet
            legacy_path = directory / "documents.pkl"
            if legacy_path.exists():
                import pickle
                logger.warning("Loading legacy pickle store; re-save to migrate to JSON")
                with open(legacy_path, "rb") as f:
                    documents = pickle.load(f)  # noqa: S301
            else:
                raise FileNotFoundError(
                    f"Document store not found at {docs_path}. "
                    "Run `clinical-rag ingest` to build the index first."
                )
        else:
            with open(docs_path) as f:
                raw = json.load(f)
            documents = [
                Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
                for d in raw
            ]

        index = faiss.read_index(str(index_path))
        embedder = ClinicalEmbedder(config)
        logger.info("Loaded index with %d vectors from %s", index.ntotal, directory)
        return cls(index=index, documents=documents, embedder=embedder)
