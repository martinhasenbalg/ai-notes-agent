"""
tests/test_phase2.py — Tests del motor de búsqueda (Fase 2)

Ejecutar:
    python tests/test_phase2.py
    python -m pytest tests/test_phase2.py -v
"""

import sys
import os
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import Note, Chunk, NoteMetadata
from src.embeddings.embedder import Embedder
from src.db.vector_store import LocalVectorStore
from src.retriever import Retriever, RetrievedChunk, RetrievalResult
from src.parsers.markdown_parser import MarkdownParser
from src.parsers.chunker import Chunker


# ── Fixture: índice de prueba con notas temáticamente distintas ───────────────

NOTES_CONTENT = {
    "rag.md": (
        "---\ntitle: RAG y recuperación semántica\ndate: 2024-11-10\ntags: [rag, llm, retrieval]\n---\n\n"
        "# RAG y recuperación semántica\n\n"
        "RAG significa Retrieval-Augmented Generation. Es una arquitectura que combina "
        "búsqueda semántica con generación de texto. Los embeddings permiten encontrar "
        "documentos relevantes para una query. El retriever recupera los chunks más "
        "similares y los pasa al LLM como contexto."
    ),
    "prompting.md": (
        "---\ntitle: Técnicas de prompting\ndate: 2024-12-01\ntags: [llm, prompting]\n---\n\n"
        "# Técnicas de prompting\n\n"
        "Chain of Thought mejora el razonamiento paso a paso. "
        "Few-shot learning da ejemplos del formato esperado al modelo. "
        "El system prompt define el rol y las restricciones del asistente."
    ),
    "vectordb.txt": (
        "Comparativa de bases de datos vectoriales\n\n"
        "Pinecone es managed y muy rápido. pgvector es la extensión de PostgreSQL "
        "para búsqueda vectorial. Qdrant es open source y muy eficiente. "
        "Para MVP se recomienda pgvector si ya usas Postgres."
    ),
}


def _build_test_index(db_path: str, notes_dir: Path):
    """Construye un índice de prueba con notas conocidas."""
    for filename, content in NOTES_CONTENT.items():
        (notes_dir / filename).write_text(content, encoding="utf-8")

    parser   = MarkdownParser()
    chunker  = Chunker(max_chars=600, overlap_chars=100)
    embedder = Embedder(simulate=True)
    store    = LocalVectorStore(db_path)

    notes    = parser.parse_directory(notes_dir, recursive=False)
    chunks   = chunker.chunk_many(notes)
    embedded = embedder.embed(chunks, verbose=False)
    store.upsert(embedded)
    store.close()
    return len(notes), len(chunks)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_retriever_returns_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        notes_dir = Path(tmpdir) / "notes"
        notes_dir.mkdir()
        db_path = str(Path(tmpdir) / "test.db")

        _build_test_index(db_path, notes_dir)

        retriever = Retriever(db_path=db_path, simulate=True)
        result = retriever.search("RAG retrieval embeddings")

        assert isinstance(result, RetrievalResult)
        assert result.query == "RAG retrieval embeddings"
        assert result.found
        assert len(result.chunks) > 0
        assert result.elapsed_ms > 0
        print("  ✓ test_retriever_returns_results")


def test_retriever_context_is_structured():
    with tempfile.TemporaryDirectory() as tmpdir:
        notes_dir = Path(tmpdir) / "notes"
        notes_dir.mkdir()
        db_path = str(Path(tmpdir) / "test.db")
        _build_test_index(db_path, notes_dir)

        retriever = Retriever(db_path=db_path, simulate=True)
        result = retriever.search("bases de datos vectoriales")

        # El contexto debe tener la estructura esperada
        assert "## Contexto recuperado" in result.context
        assert "Relevancia:" in result.context
        assert result.query in result.context
        print("  ✓ test_retriever_context_is_structured")


def test_retriever_top_k_respected():
    with tempfile.TemporaryDirectory() as tmpdir:
        notes_dir = Path(tmpdir) / "notes"
        notes_dir.mkdir()
        db_path = str(Path(tmpdir) / "test.db")
        _build_test_index(db_path, notes_dir)

        retriever = Retriever(db_path=db_path, simulate=True)
        result = retriever.search("llm", top_k=2)

        assert len(result.chunks) <= 2
        print("  ✓ test_retriever_top_k_respected")


def test_retriever_filter_by_source_type():
    with tempfile.TemporaryDirectory() as tmpdir:
        notes_dir = Path(tmpdir) / "notes"
        notes_dir.mkdir()
        db_path = str(Path(tmpdir) / "test.db")
        _build_test_index(db_path, notes_dir)

        retriever = Retriever(db_path=db_path, simulate=True)
        result = retriever.search("información general", source_type="text")

        # Solo debe devolver chunks de archivos .txt
        for chunk in result.chunks:
            assert chunk.source_type == "text", f"Esperado 'text', got '{chunk.source_type}'"
        print("  ✓ test_retriever_filter_by_source_type")


def test_retriever_filter_by_tags():
    with tempfile.TemporaryDirectory() as tmpdir:
        notes_dir = Path(tmpdir) / "notes"
        notes_dir.mkdir()
        db_path = str(Path(tmpdir) / "test.db")
        _build_test_index(db_path, notes_dir)

        retriever = Retriever(db_path=db_path, simulate=True)
        result = retriever.search("técnicas avanzadas", tags=["prompting"])

        for chunk in result.chunks:
            assert "prompting" in chunk.tags, f"Tag 'prompting' no encontrado en {chunk.tags}"
        print("  ✓ test_retriever_filter_by_tags")


def test_retriever_filter_by_date_after():
    with tempfile.TemporaryDirectory() as tmpdir:
        notes_dir = Path(tmpdir) / "notes"
        notes_dir.mkdir()
        db_path = str(Path(tmpdir) / "test.db")
        _build_test_index(db_path, notes_dir)

        retriever = Retriever(db_path=db_path, simulate=True)
        # Solo notas después de nov 2024 → excluye rag.md (2024-11-10) no, pero después del 15
        cutoff = datetime(2024, 11, 15)
        result = retriever.search("llm embeddings", after=cutoff)

        for chunk in result.chunks:
            if chunk.date:
                note_date = datetime.fromisoformat(chunk.date[:10])
                assert note_date >= cutoff, f"Fecha {chunk.date} anterior al cutoff"
        print("  ✓ test_retriever_filter_by_date_after")


def test_retriever_deduplication():
    with tempfile.TemporaryDirectory() as tmpdir:
        notes_dir = Path(tmpdir) / "notes"
        notes_dir.mkdir()
        db_path = str(Path(tmpdir) / "test.db")
        _build_test_index(db_path, notes_dir)

        retriever = Retriever(db_path=db_path, simulate=True)
        result = retriever.search("llm", top_k=10, deduplicate_notes=True)

        # Con dedup, máx 2 chunks por nota
        from collections import Counter
        counts = Counter(c.note_id for c in result.chunks)
        for note_id, count in counts.items():
            assert count <= 2, f"Nota {note_id[:8]} aparece {count} veces (máx 2)"
        print("  ✓ test_retriever_deduplication")


def test_retriever_empty_index():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "empty.db")
        retriever = Retriever(db_path=db_path, simulate=True)
        result = retriever.search("cualquier query")

        assert not result.found
        assert "No se encontró" in result.context
        print("  ✓ test_retriever_empty_index")


def test_retrieval_result_unique_notes():
    with tempfile.TemporaryDirectory() as tmpdir:
        notes_dir = Path(tmpdir) / "notes"
        notes_dir.mkdir()
        db_path = str(Path(tmpdir) / "test.db")
        _build_test_index(db_path, notes_dir)

        retriever = Retriever(db_path=db_path, simulate=True)
        result = retriever.search("llm modelos de lenguaje", top_k=6)

        unique = result.unique_notes
        assert len(unique) <= len(result.chunks)
        assert len(unique) == len(set(c.note_id for c in result.chunks))
        print("  ✓ test_retrieval_result_unique_notes")


def test_retriever_search_multi():
    with tempfile.TemporaryDirectory() as tmpdir:
        notes_dir = Path(tmpdir) / "notes"
        notes_dir.mkdir()
        db_path = str(Path(tmpdir) / "test.db")
        _build_test_index(db_path, notes_dir)

        retriever = Retriever(db_path=db_path, simulate=True)
        queries = ["RAG", "prompting", "bases de datos"]
        results = retriever.search_multi(queries)

        assert len(results) == 3
        for r, q in zip(results, queries):
            assert r.query == q
        print("  ✓ test_retriever_search_multi")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all_tests():
    tests = [
        test_retriever_returns_results,
        test_retriever_context_is_structured,
        test_retriever_top_k_respected,
        test_retriever_filter_by_source_type,
        test_retriever_filter_by_tags,
        test_retriever_filter_by_date_after,
        test_retriever_deduplication,
        test_retriever_empty_index,
        test_retrieval_result_unique_notes,
        test_retriever_search_multi,
    ]

    print("\n📦 Retriever (Fase 2)")
    passed = failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"✅ {passed} passed   ❌ {failed} failed")
    print(f"{'='*40}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
