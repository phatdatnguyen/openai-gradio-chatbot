"""Application-owned source and retrieval records."""

from dataclasses import asdict, dataclass


@dataclass
class SourceElement:
    text: str
    locator: str
    kind: str = "text"
    image_path: str | None = None


@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    locator: str
    kind: str = "text"
    image_path: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Evidence:
    chunk: Chunk
    filename: str
    score: float
    source_path: str

