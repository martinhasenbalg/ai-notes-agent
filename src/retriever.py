"""
retriever.py — Motor de búsqueda semántica (Fase 2)

Pipeline completo:
  query (texto) → embedding → top-K chunks → reranking → contexto

Características:
  - Búsqueda semántica con Voyage AI
  - Filtros por fecha, tag y fuente
  - Deduplicación por nota (evita repetir la misma nota N veces)
  - Construcción de contexto estructurado para el LLM
  - Métricas de relevancia con umbral configurable
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.db.vector_store import LocalVectorStore
from src.embeddings.embedder import Embedder


# ── Configuración por defecto ─────────────────────────────────────────────────

DEFAULT_TOP_K       = 5      # chunks a recuperar
DEFAULT_MIN_SCORE   = 0.3    # umbral mínimo de similitud (con embeddings reales)
MAX_CONTEXT_CHARS   = 6000   # tope para el contexto que se pasa al LLM


# ── Modelos de resultado ──────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """Un chunk recuperado con su score de relevancia."""
    id: str
    note_id: str
    content: str
    title: str
    source_path: str
    source_type: str
    date: Optional[str]
    tags: list[str]
    score: float

    def short_source(self) -> str:
        """Nombre del archivo fuente sin ruta completa."""
        if self.source_path.startswith("notion://"):
            return f"Notion:{self.source_path.split('/')[-1][:8]}"
        from pathlib import Path
        return Path(self.source_path).name if self.source_path else "desconocido"


@dataclass
class RetrievalResult:
    """Resultado completo de una búsqueda."""
    query: str
    chunks: list[RetrievedChunk]
    context: str                      # texto listo para inyectar en el prompt
    elapsed_ms: float
    filters_applied: dict = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return len(self.chunks) > 0

    @property
    def top_score(self) -> float:
        return self.chunks[0].score if self.chunks else 0.0

    @property
    def unique_notes(self) -> list[str]:
        seen, titles = set(), []
        for c in self.chunks:
            if c.note_id not in seen:
                seen.add(c.note_id)
                titles.append(c.title or c.short_source())
        return titles

    def summary(self) -> str:
        if not self.found:
            return "No se encontraron resultados relevantes."
        lines = [
            f"🔍 '{self.query}'",
            f"   {len(self.chunks)} chunks de {len(self.unique_notes)} nota(s) — {self.elapsed_ms:.0f}ms",
        ]
        for c in self.chunks:
            bar = "█" * int(c.score * 20) + "░" * (20 - int(c.score * 20))
            lines.append(f"   [{c.score:.3f}] {bar}  {c.title or c.short_source()}")
        return "\n".join(lines)


# ── Retriever ─────────────────────────────────────────────────────────────────

class Retriever:
    """
    Motor de búsqueda semántica sobre el índice de notas.

    Uso básico:
        retriever = Retriever()
        result = retriever.search("cómo funciona RAG")
        print(result.context)   # → listo para el LLM

    Con filtros:
        result = retriever.search(
            "técnicas de prompting",
            tags=["llm"],
            after=datetime(2024, 11, 1),
            source_type="markdown",
        )
    """

    def __init__(
        self,
        db_path: str = "notes_index.db",
        embedder: Optional[Embedder] = None,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        simulate: bool = False,
    ):
        self.db_path   = db_path
        self.top_k     = top_k
        self.min_score = min_score
        self.embedder  = embedder or Embedder(simulate=simulate)
        self._simulate = simulate

    # ── Búsqueda principal ────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        tags: Optional[list[str]] = None,
        source_type: Optional[str] = None,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
        deduplicate_notes: bool = True,
    ) -> RetrievalResult:
        """
        Busca chunks relevantes para la query.

        Args:
            query:             Pregunta o tema en lenguaje natural
            top_k:             Número máximo de chunks a devolver
            min_score:         Umbral mínimo de similitud coseno (0-1)
            tags:              Filtrar por tags (OR — cualquiera de ellos)
            source_type:       "markdown" | "text" | "notion"
            after:             Solo notas con fecha posterior a este datetime
            before:            Solo notas con fecha anterior a este datetime
            deduplicate_notes: Si True, máximo 2 chunks por nota
        """
        import time
        t0 = time.time()

        k       = top_k    or self.top_k
        min_sc  = min_score or self.min_score

        # 1. Vectorizar la query
        query_vec = self.embedder.embed_query(query)

        # 2. Recuperar del vector store
        store = LocalVectorStore(self.db_path)
        raw = store.search(
            query_embedding=query_vec,
            top_k=k * 4,          # recupera más para poder filtrar
            source_type=source_type,
            tags_filter=tags,
        )
        store.close()

        # 3. Convertir a RetrievedChunk
        chunks = [
            RetrievedChunk(
                id=r["id"],
                note_id=r["note_id"],
                content=r["content"],
                title=r.get("title") or "",
                source_path=r.get("source_path") or "",
                source_type=r.get("source_type") or "",
                date=r.get("date"),
                tags=r.get("tags") or [],
                score=r["score"],
            )
            for r in raw
        ]

        # 4. Filtros post-retrieval
        if after or before:
            chunks = self._filter_by_date(chunks, after, before)

        # 5. Umbral de score (con embeddings reales; en modo simulado se ignora)
        if not self._simulate:
            chunks = [c for c in chunks if c.score >= min_sc]

        # 6. Deduplicación por nota (máx 2 chunks por nota)
        if deduplicate_notes:
            chunks = self._deduplicate(chunks, max_per_note=2)

        # 7. Recortar al top_k final
        chunks = chunks[:k]

        # 8. Construir contexto para el LLM
        context = self._build_context(query, chunks)

        elapsed = (time.time() - t0) * 1000
        filters = {}
        if tags:         filters["tags"]        = tags
        if source_type:  filters["source_type"] = source_type
        if after:        filters["after"]        = after.isoformat()
        if before:       filters["before"]       = before.isoformat()

        return RetrievalResult(
            query=query,
            chunks=chunks,
            context=context,
            elapsed_ms=elapsed,
            filters_applied=filters,
        )

    # ── Filtros ───────────────────────────────────────────────────────────────

    def _filter_by_date(
        self,
        chunks: list[RetrievedChunk],
        after: Optional[datetime],
        before: Optional[datetime],
    ) -> list[RetrievedChunk]:
        result = []
        for c in chunks:
            if not c.date:
                result.append(c)   # sin fecha → no excluir
                continue
            try:
                note_date = datetime.fromisoformat(c.date.replace("Z", ""))
                if after  and note_date < after:  continue
                if before and note_date > before: continue
                result.append(c)
            except ValueError:
                result.append(c)
        return result

    def _deduplicate(
        self,
        chunks: list[RetrievedChunk],
        max_per_note: int = 2,
    ) -> list[RetrievedChunk]:
        counts: dict[str, int] = {}
        result = []
        for c in chunks:
            n = counts.get(c.note_id, 0)
            if n < max_per_note:
                result.append(c)
                counts[c.note_id] = n + 1
        return result

    # ── Construcción de contexto ──────────────────────────────────────────────

    def _build_context(self, query: str, chunks: list[RetrievedChunk]) -> str:
        """
        Construye un bloque de texto estructurado para inyectar en el prompt del LLM.

        Formato:
            ## Contexto recuperado para: "<query>"

            ### [1] Título de la nota
            Fuente: archivo.md | Fecha: 2024-11-15 | Tags: rag, llm
            Relevancia: 0.87

            <contenido del chunk>

            ---
            ...
        """
        if not chunks:
            return f'No se encontró contexto relevante para: "{query}"'

        lines = [
            f'## Contexto recuperado para: "{query}"\n',
            f"_(Fuentes: {len(chunks)} fragmentos de {len(set(c.note_id for c in chunks))} nota(s))_\n",
        ]

        total_chars = 0
        for i, chunk in enumerate(chunks, 1):
            # Cabecera del chunk
            meta_parts = []
            if chunk.source_path:
                meta_parts.append(f"Fuente: {chunk.short_source()}")
            if chunk.date:
                meta_parts.append(f"Fecha: {chunk.date[:10]}")
            if chunk.tags:
                meta_parts.append(f"Tags: {', '.join(chunk.tags)}")
            meta_parts.append(f"Relevancia: {chunk.score:.2f}")

            header = f"### [{i}] {chunk.title or chunk.short_source()}"
            meta   = " | ".join(meta_parts)

            # Contenido (sin el prefijo [Nota: ...] que añadió el chunker)
            content = chunk.content
            if content.startswith("[Nota:"):
                content = content.split("]\n\n", 1)[-1]

            # Truncar si el contexto total supera el límite
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining <= 0:
                lines.append(f"\n_(se omitieron {len(chunks) - i + 1} fragmentos por límite de contexto)_")
                break
            if len(content) > remaining:
                content = content[:remaining] + "…"

            lines.extend([header, meta, "", content, "\n---"])
            total_chars += len(content)

        return "\n".join(lines)

    # ── Utilidades ────────────────────────────────────────────────────────────

    def search_multi(self, queries: list[str], **kwargs) -> list[RetrievalResult]:
        """Ejecuta múltiples búsquedas y devuelve sus resultados."""
        return [self.search(q, **kwargs) for q in queries]
