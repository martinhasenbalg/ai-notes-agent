"""
tests/test_phase1.py — Suite de tests para la Fase 1

Cubre: parser, chunker, embedder (simulado) y vector store.

Ejecutar:
  python -m pytest tests/test_phase1.py -v
  python tests/test_phase1.py          # sin pytest
"""

import sys
import math
from pathlib import Path
from datetime import datetime
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import Note, Chunk, NoteMetadata, EmbeddedChunk
from src.parsers.markdown_parser import MarkdownParser
from src.parsers.chunker import Chunker
from src.embeddings.embedder import Embedder
from src.db.vector_store import LocalVectorStore, _cosine_similarity as vs_cosine


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_MARKDOWN = """\
---
title: Nota de prueba
date: 2024-11-15
tags: [test, python, llm]
---

# Nota de prueba

Este es el primer párrafo de la nota de prueba.
Tiene varias líneas y habla sobre LLMs.

## Sección importante

Aquí hay información muy relevante sobre RAG y embeddings.
Los embeddings son representaciones vectoriales del texto.

## Conclusión

La conclusión es que los embeddings son útiles para búsqueda semántica.
"""

SAMPLE_TEXT = """\
Vector Databases Comparison

Pinecone es un servicio gestionado de base de datos vectorial.
Ofrece excelente rendimiento pero tiene coste elevado.

pgvector es una extensión de PostgreSQL para búsqueda vectorial.
Ideal para proyectos que ya usan Postgres.
"""


# ── Tests del Parser ──────────────────────────────────────────────────────────

def test_parse_markdown_with_frontmatter():
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
        f.write(SAMPLE_MARKDOWN)
        fpath = Path(f.name)

    try:
        parser = MarkdownParser()
        note = parser.parse(fpath)

        assert note.metadata.title == "Nota de prueba", f"Título esperado 'Nota de prueba', got '{note.metadata.title}'"
        assert note.metadata.date is not None, "Fecha debería estar presente"
        assert note.metadata.date.year == 2024
        assert "test" in note.metadata.tags, "Tag 'test' debería estar presente"
        assert "python" in note.metadata.tags
        assert "llm" in note.metadata.tags
        assert len(note.content) > 50, "Contenido muy corto"
        assert "---" not in note.content, "Frontmatter no debería aparecer en el contenido"
        print("  ✓ test_parse_markdown_with_frontmatter")
    finally:
        fpath.unlink()


def test_parse_markdown_infers_title():
    md = "# Mi Título Inferido\n\nContenido de la nota."
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
        f.write(md)
        fpath = Path(f.name)
    try:
        parser = MarkdownParser()
        note = parser.parse(fpath)
        assert note.metadata.title == "Mi Título Inferido"
        print("  ✓ test_parse_markdown_infers_title")
    finally:
        fpath.unlink()


def test_parse_text_file():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
        f.write(SAMPLE_TEXT)
        fpath = Path(f.name)
    try:
        parser = MarkdownParser()
        note = parser.parse(fpath)
        assert note.metadata.source_type == "text"
        assert "Vector Databases" in note.metadata.title
        assert len(note.content) > 20
        print("  ✓ test_parse_text_file")
    finally:
        fpath.unlink()


def test_parse_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        (path / "note1.md").write_text(SAMPLE_MARKDOWN, encoding="utf-8")
        (path / "note2.txt").write_text(SAMPLE_TEXT, encoding="utf-8")
        (path / "ignored.py").write_text("print('hello')", encoding="utf-8")

        parser = MarkdownParser()
        notes = parser.parse_directory(path, recursive=False)
        assert len(notes) == 2, f"Esperadas 2 notas, got {len(notes)}"
        print("  ✓ test_parse_directory")


# ── Tests del Chunker ──────────────────────────────────────────────────────────

def test_chunker_basic():
    metadata = NoteMetadata(source_path="/test.md", source_type="markdown", title="Test")
    note = Note(
        content="Párrafo uno.\n\nPárrafo dos.\n\nPárrafo tres.\n\nPárrafo cuatro.",
        metadata=metadata,
    )
    chunker = Chunker(max_chars=50, overlap_chars=10)
    chunks = chunker.chunk(note)
    assert len(chunks) >= 1, "Debería haber al menos 1 chunk"
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.note_id == note.id
        assert len(c.content) > 0
    print("  ✓ test_chunker_basic")


def test_chunker_includes_title_context():
    metadata = NoteMetadata(source_path="/test.md", source_type="markdown", title="Mi Nota")
    note = Note(content="Contenido relevante aquí.", metadata=metadata)
    chunker = Chunker()
    chunks = chunker.chunk(note)
    assert any("Mi Nota" in c.content for c in chunks), "El título debería aparecer en el chunk"
    print("  ✓ test_chunker_includes_title_context")


def test_chunker_many():
    notes = []
    for i in range(5):
        meta = NoteMetadata(source_path=f"/note{i}.md", source_type="markdown", title=f"Nota {i}")
        notes.append(Note(content=f"Contenido de la nota {i}. " * 10, metadata=meta))
    chunker = Chunker()
    all_chunks = chunker.chunk_many(notes)
    assert len(all_chunks) >= 5
    print("  ✓ test_chunker_many")


# ── Tests del Embedder ─────────────────────────────────────────────────────────

def test_embedder_simulate():
    meta = NoteMetadata(source_path="/test.md", source_type="markdown")
    chunk = Chunk(note_id="note-1", content="Texto de prueba para embedding", metadata=meta)
    embedder = Embedder(simulate=True)
    assert embedder.dims == 512  # voyage-3-lite
    results = embedder.embed([chunk], verbose=False)
    assert len(results) == 1
    assert len(results[0].embedding) == 512  # voyage-3-lite dims
    # Verifica que el vector está normalizado (norma ≈ 1)
    norm = math.sqrt(sum(x**2 for x in results[0].embedding))
    assert abs(norm - 1.0) < 1e-5
    print("  ✓ test_embedder_simulate")


def test_embedder_deterministic():
    """El mismo texto debe producir el mismo embedding en modo simulado."""
    meta = NoteMetadata(source_path="/test.md", source_type="markdown")
    chunk = Chunk(note_id="note-1", content="Texto determinista", metadata=meta)
    embedder = Embedder(simulate=True)
    r1 = embedder.embed([chunk], verbose=False)[0].embedding
    r2 = embedder.embed([chunk], verbose=False)[0].embedding
    assert r1 == r2, "Embeddings simulados deben ser deterministas"
    print("  ✓ test_embedder_deterministic")


def test_embedder_different_texts_differ():
    """Textos distintos deben producir embeddings distintos."""
    meta = NoteMetadata(source_path="/test.md", source_type="markdown")
    c1 = Chunk(note_id="n1", content="RAG retrieval augmented", metadata=meta)
    c2 = Chunk(note_id="n2", content="Transformer attention mechanism", metadata=meta)
    embedder = Embedder(simulate=True)
    assert embedder.dims == 512  # voyage-3-lite
    results = embedder.embed([c1, c2], verbose=False)
    sim = vs_cosine(results[0].embedding, results[1].embedding)
    assert sim < 0.99, "Textos diferentes deberían tener embeddings distintos"
    print("  ✓ test_embedder_different_texts_differ")


def test_embedder_estimate():
    meta = NoteMetadata(source_path="/test.md", source_type="markdown")
    chunks = [Chunk(note_id="n", content="a" * 400, metadata=meta) for _ in range(10)]
    embedder = Embedder(simulate=True)
    est = embedder.estimate(chunks)
    assert est["chunks"] == 10
    assert est["estimated_tokens"] > 0
    print("  ✓ test_embedder_estimate")


# ── Tests del Vector Store ─────────────────────────────────────────────────────

def test_vector_store_upsert_and_search():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = LocalVectorStore(db_path)
        meta = NoteMetadata(source_path="/test.md", source_type="markdown", title="Test Note")

        embedder = Embedder(simulate=True)
        chunks = [
            Chunk(note_id="note-1", content=f"Chunk de prueba número {i}", metadata=meta)
            for i in range(5)
        ]
        embedded = embedder.embed(chunks, verbose=False)
        saved = store.upsert(embedded)
        assert saved == 5

        # Búsqueda con la query del primer chunk
        query_vec = embedded[0].embedding
        results = store.search(query_vec, top_k=3)
        assert len(results) <= 3
        assert results[0]["score"] >= results[-1]["score"], "Resultados deben estar ordenados por score"
        assert results[0]["score"] > 0.5, "El primer resultado debería ser muy similar"

        stats = store.count()
        assert stats["total_chunks"] == 5
        assert stats["total_notes"] == 1

        store.close()
        print("  ✓ test_vector_store_upsert_and_search")
    finally:
        os.unlink(db_path)


def test_vector_store_upsert_idempotent():
    """Indexar las mismas notas dos veces no debe duplicar chunks."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = LocalVectorStore(db_path)
        meta = NoteMetadata(source_path="/test.md", source_type="markdown")
        embedder = Embedder(simulate=True)
        chunk = Chunk(note_id="note-1", content="Texto único", metadata=meta)
        embedded = embedder.embed([chunk], verbose=False)

        store.upsert(embedded)
        store.upsert(embedded)  # segunda vez

        stats = store.count()
        assert stats["total_chunks"] == 1, "No debería duplicar con el mismo ID"
        store.close()
        print("  ✓ test_vector_store_upsert_idempotent")
    finally:
        os.unlink(db_path)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all_tests():
    tests = [
        test_parse_markdown_with_frontmatter,
        test_parse_markdown_infers_title,
        test_parse_text_file,
        test_parse_directory,
        test_chunker_basic,
        test_chunker_includes_title_context,
        test_chunker_many,
        test_embedder_simulate,
        test_embedder_deterministic,
        test_embedder_different_texts_differ,
        test_embedder_estimate,
        test_vector_store_upsert_and_search,
        test_vector_store_upsert_idempotent,
    ]

    groups = {
        "Parser": tests[0:4],
        "Chunker": tests[4:7],
        "Embedder": tests[7:11],
        "Vector Store": tests[11:],
    }

    passed = failed = 0
    for group, group_tests in groups.items():
        print(f"\n📦 {group}")
        for test in group_tests:
            try:
                test()
                passed += 1
            except Exception as e:
                print(f"  ✗ {test.__name__}: {e}")
                failed += 1

    print(f"\n{'='*40}")
    print(f"✅ {passed} passed   ❌ {failed} failed")
    print(f"{'='*40}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
