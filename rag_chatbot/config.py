"""RAG settings; model generation capabilities remain in chatbot.config."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(
        os.getenv("RAG_DATA_DIR", str(Path(__file__).resolve().parent / "data"))
    ).resolve())
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
    qdrant_api_key: str | None = field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    qdrant_local_path: str | None = field(default_factory=lambda: os.getenv("QDRANT_LOCAL_PATH"))
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    vision_model: str = "gpt-4.1-mini"
    chunk_size: int = 600
    chunk_overlap: int = 80
    candidate_count: int = 20
    evidence_count: int = 8
    evidence_token_budget: int = 6000
    # Workspace identity isolates Qdrant collections without changing extraction
    # fingerprints. Empty keeps direct, standalone LibraryStore use compatible.
    namespace: str = ""

    def fingerprint(self):
        return f"v1:{self.embedding_model}:{self.embedding_dimensions}:{self.chunk_size}:{self.chunk_overlap}"
