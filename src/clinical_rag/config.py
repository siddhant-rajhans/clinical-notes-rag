from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RAGConfig:
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
