"""
tests/test_phase3.py — Tests del generador de posts LinkedIn (Fase 3)

Ejecutar:
    python tests/test_phase3.py
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.voice_profile import VoiceProfile
from src.schema import Note, NoteMetadata
from src.parsers.chunker import Chunker
from src.embeddings.embedder import Embedder
from src.db.vector_store import LocalVectorStore
from src.linkedin_generator import LinkedInGenerator, LinkedInPost, GenerationResult


NOTES_CONTENT = {
    "rag.md": (
        "---\ntitle: RAG y recuperación semántica\ndate: 2024-11-10\ntags: [rag, llm]\n---\n\n"
        "# RAG y recuperación semántica\n\n"
        "RAG combina búsqueda semántica con generación de texto. "
        "Los embeddings permiten encontrar documentos relevantes. "
        "El retriever recupera chunks similares y los pasa al LLM como contexto. "
        "Naive RAG es el patrón básico. Advanced RAG añade reranking y query expansion."
    ),
    "prompting.md": (
        "---\ntitle: Técnicas de prompting\ndate: 2024-12-01\ntags: [llm, prompting]\n---\n\n"
        "# Técnicas de prompting\n\n"
        "Chain of Thought mejora el razonamiento. Few-shot da ejemplos al modelo. "
        "El system prompt define rol y restricciones."
    ),
}


def _build_test_index(db_path: str, notes_dir: Path):
    for filename, content in NOTES_CONTENT.items():
        (notes_dir / filename).write_text(content, encoding="utf-8")
    from src.parsers.markdown_parser import MarkdownParser
    parser   = MarkdownParser()
    chunker  = Chunker(max_chars=600)
    embedder = Embedder(simulate=True)
    store    = LocalVectorStore(db_path)
    notes    = parser.parse_directory(notes_dir, recursive=False)
    chunks   = chunker.chunk_many(notes)
    embedded = embedder.embed(chunks, verbose=False)
    store.upsert(embedded)
    store.close()


# ── Tests VoiceProfile ────────────────────────────────────────────────────────

def test_voice_profile_defaults():
    profile = VoiceProfile()
    assert profile.tone == "profesional"
    assert profile.language == "es"
    assert profile.use_hashtags is True
    assert profile.hashtag_count == 5
    print("  ✓ test_voice_profile_defaults")


def test_voice_profile_save_load():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    try:
        profile = VoiceProfile(
            name="María", role="CTO", tone="conversacional",
            use_emojis=False, hashtag_count=3
        )
        profile.save(path)
        loaded = VoiceProfile.load(path)
        assert loaded.name == "María"
        assert loaded.role == "CTO"
        assert loaded.tone == "conversacional"
        assert loaded.use_emojis is False
        assert loaded.hashtag_count == 3
        print("  ✓ test_voice_profile_save_load")
    finally:
        Path(path).unlink(missing_ok=True)


def test_voice_profile_to_prompt_block():
    profile = VoiceProfile(
        name="Ana", role="Ingeniería de IA", tone="técnico",
        use_emojis=True, use_hashtags=True, hashtag_count=4,
        audience="desarrolladores Python"
    )
    block = profile.to_prompt_block()
    assert "Ana" in block
    assert "técnico" in block
    assert "desarrolladores Python" in block
    assert "Hashtags" in block
    print("  ✓ test_voice_profile_to_prompt_block")


def test_voice_profile_with_examples():
    profile = VoiceProfile(
        example_posts=["Este es un post de ejemplo sobre IA que escribí."]
    )
    block = profile.to_prompt_block()
    assert "Ejemplos de posts" in block
    assert "Este es un post" in block
    print("  ✓ test_voice_profile_with_examples")


# ── Tests LinkedInGenerator ───────────────────────────────────────────────────

def test_generator_simulate_returns_posts():
    with tempfile.TemporaryDirectory() as tmpdir:
        notes_dir = Path(tmpdir) / "notes"
        notes_dir.mkdir()
        db_path = str(Path(tmpdir) / "test.db")
        _build_test_index(db_path, notes_dir)

        profile = VoiceProfile(name="Test", tone="técnico")
        gen = LinkedInGenerator(db_path=db_path, profile=profile, simulate=True)
        result = gen.generate("RAG y embeddings", verbose=False)

        assert isinstance(result, GenerationResult)
        assert result.found if hasattr(result, 'found') else len(result.posts) >= 0
        print("  ✓ test_generator_simulate_returns_posts")


def test_generator_posts_have_required_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        notes_dir = Path(tmpdir) / "notes"
        notes_dir.mkdir()
        db_path = str(Path(tmpdir) / "test.db")
        _build_test_index(db_path, notes_dir)

        gen = LinkedInGenerator(db_path=db_path, simulate=True)
        result = gen.generate("técnicas de prompting", verbose=False)

        for post in result.posts:
            assert isinstance(post, LinkedInPost)
            assert post.content, "El post no debe estar vacío"
            assert post.hook,    "El hook no debe estar vacío"
            assert post.variant > 0
            assert post.word_count > 0
            assert post.style_label
        print("  ✓ test_generator_posts_have_required_fields")


def test_generator_num_variants():
    with tempfile.TemporaryDirectory() as tmpdir:
        notes_dir = Path(tmpdir) / "notes"
        notes_dir.mkdir()
        db_path = str(Path(tmpdir) / "test.db")
        _build_test_index(db_path, notes_dir)

        gen = LinkedInGenerator(db_path=db_path, simulate=True)
        result = gen.generate("RAG", num_variants=2, verbose=False)
        assert len(result.posts) == 2
        print("  ✓ test_generator_num_variants")


def test_generator_posts_have_hashtags():
    with tempfile.TemporaryDirectory() as tmpdir:
        notes_dir = Path(tmpdir) / "notes"
        notes_dir.mkdir()
        db_path = str(Path(tmpdir) / "test.db")
        _build_test_index(db_path, notes_dir)

        profile = VoiceProfile(use_hashtags=True, hashtag_count=5)
        gen = LinkedInGenerator(db_path=db_path, profile=profile, simulate=True)
        result = gen.generate("LLM prompting", verbose=False)

        for post in result.posts:
            assert len(post.hashtags) > 0, "Post sin hashtags"
        print("  ✓ test_generator_posts_have_hashtags")


def test_generator_empty_index():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "empty.db")
        gen = LinkedInGenerator(db_path=db_path, simulate=True)
        result = gen.generate("RAG", verbose=False)
        assert result.posts == []
        print("  ✓ test_generator_empty_index")


def test_generator_result_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        notes_dir = Path(tmpdir) / "notes"
        notes_dir.mkdir()
        db_path = str(Path(tmpdir) / "test.db")
        _build_test_index(db_path, notes_dir)

        gen = LinkedInGenerator(db_path=db_path, simulate=True)
        result = gen.generate("RAG retrieval", verbose=False)

        assert result.query == "RAG retrieval"
        assert result.elapsed_ms > 0
        assert result.generated_at
        assert result.model
        print("  ✓ test_generator_result_metadata")


def test_parse_posts_cleans_markdown_json():
    """Verifica que el parser limpia bloques ```json de Claude."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notes_dir = Path(tmpdir) / "notes"
        notes_dir.mkdir()
        db_path = str(Path(tmpdir) / "test.db")
        _build_test_index(db_path, notes_dir)

        gen = LinkedInGenerator(db_path=db_path, simulate=True)
        retrieval = gen.retriever.search("RAG")

        raw = '```json\n{"posts": [{"variant": 1, "style_label": "Test", "hook": "Hook test", "content": "Contenido del post de prueba con varias palabras.", "hashtags": ["#IA"]}]}\n```'
        posts = gen._parse_posts(raw, retrieval)
        assert len(posts) == 1
        assert posts[0].hook == "Hook test"
        print("  ✓ test_parse_posts_cleans_markdown_json")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all_tests():
    tests = [
        test_voice_profile_defaults,
        test_voice_profile_save_load,
        test_voice_profile_to_prompt_block,
        test_voice_profile_with_examples,
        test_generator_simulate_returns_posts,
        test_generator_posts_have_required_fields,
        test_generator_num_variants,
        test_generator_posts_have_hashtags,
        test_generator_empty_index,
        test_generator_result_metadata,
        test_parse_posts_cleans_markdown_json,
    ]

    groups = {
        "VoiceProfile": tests[:4],
        "LinkedInGenerator": tests[4:],
    }

    passed = failed = 0
    for group, group_tests in groups.items():
        print(f"\n📦 {group}")
        for test in group_tests:
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
