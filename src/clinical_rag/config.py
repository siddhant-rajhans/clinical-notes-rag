from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RAGConfig:
    """Configuration for the clinical RAG pipeline."""

    embedding_model: str = "emilyalsentzer/Bio_ClinicalBERT"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    grounding_model: str = "cross-encoder/nli-deberta-v3-base"
    llm_model: str = "gpt-4o-mini"

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    grounding_threshold: float = 0.6

    embedding_batch_size: int = 64
    embedding_dim: int = 768
    use_ivf: bool = False
    ivf_nlist: int = 100
    ivf_nprobe: int = 10

    index_dir: Path = field(default_factory=lambda: Path("data/index"))
    data_dir: Path = field(default_factory=lambda: Path("data"))

    refusal_message: str = (
        "I cannot answer this question with sufficient confidence "
        "based on the available clinical notes."
    )

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if not 0 <= self.grounding_threshold <= 1:
            raise ValueError(
                f"grounding_threshold must be in [0, 1], got {self.grounding_threshold}"
            )
        if self.top_k_retrieval <= 0:
            raise ValueError(f"top_k_retrieval must be positive, got {self.top_k_retrieval}")
        if self.top_k_rerank <= 0:
            raise ValueError(f"top_k_rerank must be positive, got {self.top_k_rerank}")
