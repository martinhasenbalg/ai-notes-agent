"""
indexer.py — CLI principal para indexar notas

Orquesta el pipeline completo:
  Archivos/Notion → Parser → Chunker → Embedder → VectorStore

Uso:
  # Indexar carpeta local
  python indexer.py index ./mis-notas

  # Indexar con modo simulado (sin API key)
  python indexer.py index ./mis-notas --simulate

  # Ver estadísticas del índice
  python indexer.py stats

  # Búsqueda rápida de prueba
  python indexer.py search "RAG retrieval augmented generation"

  # Indexar desde Notion
  python indexer.py notion <database-id>

  # Limpiar el índice
  python indexer.py clear
"""

import sys
import os
import time
from pathlib import Path

# Añade el directorio raíz al path para imports relativos
sys.path.insert(0, str(Path(__file__).parent))

# Carga variables de entorno desde .env (si existe)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # sin dotenv, usa variables del sistema directamente

from src.parsers.markdown_parser import MarkdownParser
from src.parsers.chunker import Chunker
from src.embeddings.embedder import Embedder
from src.db.vector_store import LocalVectorStore


# ── Configuración ─────────────────────────────────────────────────────────

DB_PATH = "notes_index.db"
DEFAULT_TOP_K = 5


# ── Pipeline ──────────────────────────────────────────────────────────────

def run_index_pipeline(
    directory: str,
    simulate: bool = False,
    recursive: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Ejecuta el pipeline completo de indexación sobre un directorio.
    
    Retorna estadísticas del proceso.
    """
    start_time = time.time()
    path = Path(directory)

    if not path.exists():
        print(f"❌ Directorio no encontrado: {directory}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"🗂️  AI Notes Indexer — Fase 1")
    print(f"{'='*50}")
    print(f"Directorio: {path.resolve()}")
    print(f"Modo:       {'simulado (sin API)' if simulate else 'OpenAI API'}")
    print()

    # 1. Parseo
    print("── 1/4  Parseando archivos ──")
    parser = MarkdownParser()
    notes = parser.parse_directory(path, recursive=recursive)

    if not notes:
        print("⚠️  No se encontraron archivos .md/.txt. Verifica el directorio.")
        return {}

    if verbose:
        print("\nNotas encontradas:")
        for note in notes:
            date_str = note.metadata.date.strftime("%Y-%m-%d") if note.metadata.date else "sin fecha"
            tags_str = ", ".join(note.metadata.tags) if note.metadata.tags else "sin tags"
            print(f"  • {note.metadata.title:<40} | {date_str} | [{tags_str}]")

    # 2. Chunking
    print(f"\n── 2/4  Dividiendo en chunks ──")
    chunker = Chunker()
    chunks = chunker.chunk_many(notes)
    avg_chunks = len(chunks) / len(notes) if notes else 0
    print(f"✅ {len(chunks)} chunks generados ({avg_chunks:.1f} promedio por nota)")

    # 3. Embeddings
    print(f"\n── 3/4  Generando embeddings ──")
    embedder = Embedder(simulate=simulate)
    embedded_chunks = embedder.embed(chunks, verbose=True)

    # 4. Almacenamiento
    print(f"── 4/4  Guardando en vector store ──")
    store = LocalVectorStore(DB_PATH)
    saved = store.upsert(embedded_chunks)
    stats = store.count()
    store.close()

    elapsed = time.time() - start_time
    print(f"✅ {saved} chunks guardados en {DB_PATH}")
    print(f"\n{'='*50}")
    print(f"✨ Indexación completa en {elapsed:.1f}s")
    print(f"   Notas:  {stats['total_notes']}")
    print(f"   Chunks: {stats['total_chunks']}")
    print(f"   DB:     {DB_PATH}")
    print(f"{'='*50}\n")

    return stats


def run_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    simulate: bool = False,
    tags: list = None,
    source_type: str = None,
    show_context: bool = False,
):
    """Búsqueda semántica con pipeline RAG completo."""
    from src.retriever import Retriever

    if not Path(DB_PATH).exists():
        print("⚠️  No hay índice. Ejecuta primero: python indexer.py index <directorio>")
        return

    retriever = Retriever(db_path=DB_PATH, top_k=top_k, simulate=simulate)
    result = retriever.search(query, tags=tags, source_type=source_type)

    print(result.summary())

    if show_context:
        print("\n" + "="*50)
        print("CONTEXTO PARA EL LLM:")
        print("="*50)
        print(result.context)


def run_stats():
    """Muestra estadísticas del índice."""
    if not Path(DB_PATH).exists():
        print("⚠️  No hay índice. Ejecuta primero: python indexer.py index <directorio>")
        return
    store = LocalVectorStore(DB_PATH)
    stats = store.count()
    store.close()
    print(f"\n📊 Estado del índice ({DB_PATH})")
    print(f"   Notas indexadas:  {stats['total_notes']}")
    print(f"   Chunks totales:   {stats['total_chunks']}")
    if stats.get("by_source_type"):
        print(f"   Por tipo de fuente:")
        for stype, count in stats["by_source_type"].items():
            print(f"     {stype:<12} {count} chunks")
    print()


def run_clear():
    """Limpia el índice."""
    if not Path(DB_PATH).exists():
        print("No hay índice que limpiar.")
        return
    confirm = input("¿Eliminar todos los chunks del índice? [s/N] ").strip().lower()
    if confirm == "s":
        store = LocalVectorStore(DB_PATH)
        store.clear()
        store.close()
        print("✅ Índice limpiado.")
    else:
        print("Cancelado.")


# ── Entry point ───────────────────────────────────────────────────────────


def run_generate(query: str, simulate: bool = False, save: bool = False):
    """Genera posts de LinkedIn desde las notas indexadas."""
    from src.linkedin_generator import LinkedInGenerator
    from src.voice_profile import VoiceProfile
    from pathlib import Path as P

    if not P(DB_PATH).exists():
        print("⚠️  No hay índice. Ejecuta primero: python indexer.py index <directorio>")
        return

    profile = VoiceProfile.load()
    gen = LinkedInGenerator(db_path=DB_PATH, profile=profile, simulate=simulate)
    result = gen.generate(query)

    if not result.posts:
        return

    result.print_all()

    if save:
        import json, datetime
        fname = f"posts_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(fname, "w", encoding="utf-8") as f:
            data = {
                "query": result.query,
                "generated_at": result.generated_at,
                "posts": [
                    {"variant": p.variant, "style": p.style_label,
                     "content": p.content, "word_count": p.word_count}
                    for p in result.posts
                ]
            }
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Posts guardados en {fname}")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]
    args = sys.argv[2:]
    simulate = "--simulate" in args

    if command == "index":
        if not args:
            print("Uso: python indexer.py index <directorio> [--simulate]")
            sys.exit(1)
        directory = args[0]
        run_index_pipeline(directory, simulate=simulate)

    elif command == "search":
        if not args:
            print("Uso: python indexer.py search <query> [--simulate]")
            sys.exit(1)
        query = args[0]
        show_ctx = "--context" in args
        run_search(query, simulate=simulate, show_context=show_ctx)

    elif command == "stats":
        run_stats()

    elif command == "clear":
        run_clear()

    elif command == "notion":
        if not args:
            print("Uso: python indexer.py notion <database-id> [--simulate]")
            sys.exit(1)
        from src.connectors.notion_connector import NotionConnector
        connector = NotionConnector()
        database_id = args[0]
        notes = connector.fetch_database(database_id)
        if notes:
            chunker = Chunker()
            chunks = chunker.chunk_many(notes)
            embedder = Embedder(simulate=simulate)
            embedded = embedder.embed(chunks)
            store = LocalVectorStore(DB_PATH)
            store.upsert(embedded)
            print(f"✅ {len(notes)} notas de Notion indexadas.")
            store.close()

    elif command == "generate":
        if not args:
            print("Uso: python indexer.py generate <tema> [--simulate] [--save]")
            sys.exit(1)
        query = args[0]
        save  = "--save" in args
        run_generate(query, simulate=simulate, save=save)

    elif command == "profile":
        from src.voice_profile import VoiceProfile
        VoiceProfile.from_onboarding()

    else:
        print(f"Comando desconocido: {command}")
        print("Comandos disponibles: index, search, generate, profile, stats, clear, notion")
        sys.exit(1)


if __name__ == "__main__":
    main()
