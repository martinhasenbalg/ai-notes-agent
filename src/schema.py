"""
schema.py — Modelos de datos del agente de notas IA
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


def _stable_id(source_path: str, suffix: str = "") -> str:
    """
    ID determinista basado en la ruta del archivo.
    Re-indexar el mismo archivo siempre produce el mismo ID
    → el upsert actualiza en lugar de duplicar.
    """
    raw = f"{source_path}{suffix}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


@dataclass
class NoteMetadata:
    source_path: str
    source_type: str
    title: Optional[str] = None
    date: Optional[datetime] = None
    tags: list[str] = field(default_factory=list)
    raw_frontmatter: dict = field(default_factory=dict)


@dataclass
class Note:
    content: str = ""
    metadata: NoteMetadata = field(default_factory=lambda: NoteMetadata(
        source_path="", source_type="unknown"
    ))
    ingested_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def id(self) -> str:
        """ID estable basado en el path del archivo fuente."""
        return _stable_id(self.metadata.source_path)

    def __repr__(self) -> str:
        title = self.metadata.title or "(sin título)"
        return f"Note(id={self.id[:8]}…, title='{title}', chars={len(self.content)})"


@dataclass
class Chunk:
    note_id: str = ""
    content: str = ""
    chunk_index: int = 0
    char_start: int = 0
    char_end: int = 0
    metadata: NoteMetadata = field(default_factory=lambda: NoteMetadata(
        source_path="", source_type="unknown"
    ))

    @property
    def id(self) -> str:
        """ID estable: path del archivo + índice del chunk."""
        return _stable_id(self.metadata.source_path, f":chunk:{self.chunk_index}")

    def __repr__(self) -> str:
        return (
            f"Chunk(id={self.id[:8]}…, note={self.note_id[:8]}…, "
            f"index={self.chunk_index}, chars={len(self.content)})"
        )


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    embedding: list[float]
    model: str = "voyage-3-lite"
    embedded_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def dimensions(self) -> int:
        return len(self.embedding)
