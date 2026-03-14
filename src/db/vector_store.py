"""
db/vector_store.py — Almacenamiento de embeddings

Implementación en dos modos:
  - LocalVectorStore: SQLite + JSON (desarrollo, sin dependencias externas)
  - PgVectorStore:    PostgreSQL + pgvector (producción)

Ambos exponen la misma interfaz para que el indexer y el retriever
sean agnósticos al backend de almacenamiento.
"""

import json
import math
import sqlite3
from pathlib import Path
from typing import Optional

from src.schema import Chunk, EmbeddedChunk


# ── Similitud coseno ─────────────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x**2 for x in a))
    norm_b = math.sqrt(sum(x**2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── LocalVectorStore ─────────────────────────────────────────────────────────

class LocalVectorStore:
    """
    Vector store local usando SQLite.
    
    Ideal para desarrollo y MVPs con <10k chunks.
    La búsqueda es fuerza bruta O(n) — suficiente para notas personales.

    Schema:
        chunks(id, note_id, content, chunk_index, char_start, char_end,
               source_path, source_type, title, date, tags,
               embedding_json, model, embedded_at)
    """

    def __init__(self, db_path: str = "notes_index.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id           TEXT PRIMARY KEY,
                note_id      TEXT NOT NULL,
                content      TEXT NOT NULL,
                chunk_index  INTEGER,
                char_start   INTEGER,
                char_end     INTEGER,
                source_path  TEXT,
                source_type  TEXT,
                title        TEXT,
                date         TEXT,
                tags         TEXT,
                embedding    TEXT NOT NULL,
                model        TEXT,
                embedded_at  TEXT
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_note_id ON chunks(note_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_source_type ON chunks(source_type)"
        )
        self.conn.commit()

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """Inserta o reemplaza chunks. Devuelve el número insertado."""
        rows = []
        for ec in embedded_chunks:
            c = ec.chunk
            rows.append((
                c.id,
                c.note_id,
                c.content,
                c.chunk_index,
                c.char_start,
                c.char_end,
                c.metadata.source_path,
                c.metadata.source_type,
                c.metadata.title,
                c.metadata.date.isoformat() if c.metadata.date else None,
                json.dumps(c.metadata.tags),
                json.dumps(ec.embedding),
                ec.model,
                ec.embedded_at.isoformat(),
            ))

        self.conn.executemany("""
            INSERT OR REPLACE INTO chunks
            (id, note_id, content, chunk_index, char_start, char_end,
             source_path, source_type, title, date, tags,
             embedding, model, embedded_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)
        self.conn.commit()
        return len(rows)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        source_type: Optional[str] = None,
        tags_filter: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Búsqueda semántica por similitud coseno.
        
        Devuelve lista de dicts con chunk info + score, ordenada por relevancia.
        """
        # Carga todos los chunks (filtrando por source_type si se especifica)
        query = "SELECT * FROM chunks"
        params: list = []
        if source_type:
            query += " WHERE source_type = ?"
            params.append(source_type)

        cursor = self.conn.execute(query, params)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]

        results = []
        for row in rows:
            r = dict(zip(cols, row))
            embedding = json.loads(r["embedding"])
            score = _cosine_similarity(query_embedding, embedding)

            # Filtro por tags (post-retrieval, barato)
            if tags_filter:
                chunk_tags = json.loads(r.get("tags", "[]"))
                if not any(t in chunk_tags for t in tags_filter):
                    continue

            results.append({
                "id": r["id"],
                "note_id": r["note_id"],
                "content": r["content"],
                "chunk_index": r["chunk_index"],
                "title": r["title"],
                "source_path": r["source_path"],
                "source_type": r["source_type"],
                "date": r["date"],
                "tags": json.loads(r.get("tags", "[]")),
                "score": round(score, 4),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def count(self) -> dict:
        """Estadísticas del índice."""
        total = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        notes = self.conn.execute("SELECT COUNT(DISTINCT note_id) FROM chunks").fetchone()[0]
        by_type = self.conn.execute(
            "SELECT source_type, COUNT(*) FROM chunks GROUP BY source_type"
        ).fetchall()
        return {
            "total_chunks": total,
            "total_notes": notes,
            "by_source_type": dict(by_type),
        }

    def clear(self):
        """Elimina todos los chunks del índice."""
        self.conn.execute("DELETE FROM chunks")
        self.conn.commit()

    def close(self):
        self.conn.close()


# ── PgVectorStore (interfaz para producción) ─────────────────────────────────

class PgVectorStore:
    """
    Vector store usando PostgreSQL + pgvector.
    
    Requiere:
        pip install psycopg2-binary pgvector
        CREATE EXTENSION vector;  -- en tu base de datos
    
    Uso:
        store = PgVectorStore("postgresql://user:pass@host/db")
    """

    def __init__(self, connection_string: str, dimensions: int = 1536):
        self.connection_string = connection_string
        self.dimensions = dimensions
        self._conn = None
        self._init_connection()

    def _init_connection(self):
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
            self._conn = psycopg2.connect(self.connection_string)
            register_vector(self._conn)
            self._create_tables()
        except ImportError:
            raise ImportError(
                "Instala psycopg2-binary y pgvector:\n"
                "  pip install psycopg2-binary pgvector"
            )

    def _create_tables(self):
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id           TEXT PRIMARY KEY,
                    note_id      TEXT NOT NULL,
                    content      TEXT NOT NULL,
                    chunk_index  INTEGER,
                    source_path  TEXT,
                    source_type  TEXT,
                    title        TEXT,
                    date         TIMESTAMPTZ,
                    tags         TEXT[],
                    embedding    vector({self.dimensions}),
                    model        TEXT,
                    embedded_at  TIMESTAMPTZ
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
                "ON chunks USING ivfflat (embedding vector_cosine_ops)"
            )
        self._conn.commit()

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        import numpy as np
        with self._conn.cursor() as cur:
            for ec in embedded_chunks:
                c = ec.chunk
                cur.execute("""
                    INSERT INTO chunks
                    (id, note_id, content, chunk_index, source_path, source_type,
                     title, date, tags, embedding, model, embedded_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET
                        content=EXCLUDED.content,
                        embedding=EXCLUDED.embedding,
                        embedded_at=EXCLUDED.embedded_at
                """, (
                    c.id, c.note_id, c.content, c.chunk_index,
                    c.metadata.source_path, c.metadata.source_type,
                    c.metadata.title,
                    c.metadata.date,
                    c.metadata.tags,
                    np.array(ec.embedding),
                    ec.model,
                    ec.embedded_at,
                ))
        self._conn.commit()
        return len(embedded_chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        source_type: Optional[str] = None,
    ) -> list[dict]:
        import numpy as np
        vec = np.array(query_embedding)
        where = "WHERE source_type = %s" if source_type else ""
        params = [vec, top_k]
        if source_type:
            params.insert(1, source_type)

        with self._conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, note_id, content, title, source_path, source_type,
                       1 - (embedding <=> %s::vector) as score
                FROM chunks
                {where}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, [vec] + ([source_type] if source_type else []) + [vec, top_k])
            rows = cur.fetchall()

        return [
            {
                "id": r[0], "note_id": r[1], "content": r[2],
                "title": r[3], "source_path": r[4], "source_type": r[5],
                "score": round(float(r[6]), 4),
            }
            for r in rows
        ]

    def count(self) -> dict:
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*), COUNT(DISTINCT note_id) FROM chunks")
            total, notes = cur.fetchone()
        return {"total_chunks": total, "total_notes": notes}
