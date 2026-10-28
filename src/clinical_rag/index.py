from __future__ import annotations

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document

from clinical_rag.config import RAGConfig
from clinical_rag.embeddings import ClinicalEmbedder

logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS index with document metadata store."""

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
        """Persist the index and metadata to disk."""
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / "index.faiss"))

        with open(directory / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        logger.info("Saved index to %s", directory)

    @classmethod
    def load(cls, directory: Path, config: RAGConfig | None = None) -> FAISSIndex:
        """Load a persisted index."""
        config = config or RAGConfig()

        index = faiss.read_index(str(directory / "index.faiss"))

        with open(directory / "documents.pkl", "rb") as f:
            documents = pickle.load(f)  # noqa: S301

        embedder = ClinicalEmbedder(config)
        logger.info(
            "Loaded index with %d vectors from %s", index.ntotal, directory
        )
        return cls(index=index, documents=documents, embedder=embedder)
