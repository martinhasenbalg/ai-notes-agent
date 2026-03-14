"""
api.py — Servidor Flask para la interfaz web del AI Notes Agent

Endpoints:
  GET  /api/status          — estado del índice y conexiones
  POST /api/index           — indexar un directorio
  POST /api/search          — búsqueda semántica
  POST /api/generate        — generar posts LinkedIn
  GET  /api/profile         — obtener perfil de voz
  POST /api/profile         — guardar perfil de voz
  GET  /api/posts           — listar posts guardados
  GET  /api/posts/<fname>   — obtener posts de un archivo

Uso:
  python api.py
  → Abre http://localhost:5000
"""

import os
import sys
import json
import glob
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

# Cargar .env si existe
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__, static_folder="web", static_url_path="")
CORS(app)

# En Railway usa /data (volumen persistente); en local usa el directorio actual
DATA_DIR = Path(os.getenv("DATA_DIR", "."))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH          = str(DATA_DIR / "notes_index.db")
PROFILE_PATH     = str(DATA_DIR / "voice_profile.json")
SEEN_ARTICLES    = str(DATA_DIR / ".seen_articles.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_embedder(simulate=False):
    from src.embeddings.embedder import Embedder
    return Embedder(simulate=simulate)

def _get_store():
    from src.db.vector_store import LocalVectorStore
    return LocalVectorStore(DB_PATH)

def _get_retriever(simulate=False):
    from src.retriever import Retriever
    return Retriever(db_path=DB_PATH, simulate=simulate)

def _get_generator(simulate=False):
    from src.linkedin_generator import LinkedInGenerator
    from src.voice_profile import VoiceProfile
    profile = VoiceProfile.load(PROFILE_PATH)
    return LinkedInGenerator(db_path=DB_PATH, profile=profile, simulate=simulate)

def _simulate_mode():
    """Usa modo simulado si no hay keys configuradas."""
    return not os.getenv("VOYAGE_API_KEY")


# ── Rutas de la SPA ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("web", "index.html")


# ── API: Status ───────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    status = {
        "voyage_key":    bool(os.getenv("VOYAGE_API_KEY")),
        "anthropic_key": bool(os.getenv("ANTHROPIC_API_KEY")),
        "index_exists":  Path(DB_PATH).exists(),
        "index_stats":   {},
        "profile_exists": Path("voice_profile.json").exists(),
    }
    if status["index_exists"]:
        try:
            store = _get_store()
            status["index_stats"] = store.count()
            store.close()
        except Exception as e:
            status["index_error"] = str(e)
    return jsonify(status)


# ── API: Indexar ──────────────────────────────────────────────────────────────

@app.route("/api/index", methods=["POST"])
def api_index():
    data = request.json or {}
    directory = data.get("directory", "notes_sample")
    simulate  = data.get("simulate", _simulate_mode())

    if not Path(directory).exists():
        return jsonify({"error": f"Directorio no encontrado: {directory}"}), 400

    try:
        from src.parsers.markdown_parser import MarkdownParser
        from src.parsers.chunker import Chunker
        from src.embeddings.embedder import Embedder
        from src.db.vector_store import LocalVectorStore

        parser   = MarkdownParser()
        chunker  = Chunker()
        embedder = Embedder(simulate=simulate)
        store    = LocalVectorStore(DB_PATH)

        notes    = parser.parse_directory(Path(directory))
        chunks   = chunker.chunk_many(notes)
        embedded = embedder.embed(chunks, verbose=False)
        saved    = store.upsert(embedded)
        stats    = store.count()
        store.close()

        return jsonify({
            "success": True,
            "notes_parsed": len(notes),
            "chunks_saved": saved,
            "stats": stats,
            "notes": [
                {
                    "title": n.metadata.title,
                    "date":  n.metadata.date.isoformat() if n.metadata.date else None,
                    "tags":  n.metadata.tags,
                    "source_type": n.metadata.source_type,
                }
                for n in notes
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Búsqueda ─────────────────────────────────────────────────────────────

@app.route("/api/search", methods=["POST"])
def api_search():
    data     = request.json or {}
    query    = data.get("query", "").strip()
    top_k    = int(data.get("top_k", 5))
    tags     = data.get("tags") or None
    simulate = data.get("simulate", _simulate_mode())

    if not query:
        return jsonify({"error": "Query vacía"}), 400
    if not Path(DB_PATH).exists():
        return jsonify({"error": "No hay índice. Indexa primero tus notas."}), 400

    try:
        retriever = _get_retriever(simulate=simulate)
        result    = retriever.search(query, top_k=top_k, tags=tags)

        return jsonify({
            "query":      result.query,
            "found":      result.found,
            "elapsed_ms": result.elapsed_ms,
            "chunks": [
                {
                    "title":       c.title,
                    "content":     c.content[:400],
                    "score":       c.score,
                    "source_type": c.source_type,
                    "date":        c.date,
                    "tags":        c.tags,
                    "source_path": Path(c.source_path).name if c.source_path else "",
                }
                for c in result.chunks
            ],
            "context": result.context,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Generar posts ────────────────────────────────────────────────────────

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data         = request.json or {}
    query        = data.get("query", "").strip()
    num_variants = int(data.get("num_variants", 3))
    top_k        = int(data.get("top_k", 4))
    simulate     = data.get("simulate", not bool(os.getenv("ANTHROPIC_API_KEY")))

    if not query:
        return jsonify({"error": "Query vacía"}), 400
    if not Path(DB_PATH).exists():
        return jsonify({"error": "No hay índice. Indexa primero tus notas."}), 400

    try:
        gen    = _get_generator(simulate=simulate)
        result = gen.generate(query, top_k=top_k, num_variants=num_variants, verbose=False)

        posts_data = [
            {
                "variant":     p.variant,
                "style_label": p.style_label,
                "hook":        p.hook,
                "content":     p.content,
                "hashtags":    p.hashtags,
                "word_count":  p.word_count,
                "sources":     p.sources,
            }
            for p in result.posts
        ]

        # Guardar automáticamente
        saved_file = None
        if posts_data:
            fname = f"posts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump({
                    "query": result.query,
                    "generated_at": result.generated_at,
                    "model": result.model,
                    "posts": posts_data,
                }, f, ensure_ascii=False, indent=2)
            saved_file = fname

        return jsonify({
            "success":    True,
            "query":      result.query,
            "model":      result.model,
            "elapsed_ms": result.elapsed_ms,
            "posts":      posts_data,
            "sources":    result.retrieval.unique_notes,
            "saved_file": saved_file,
            "simulated":  simulate,
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "detail": traceback.format_exc()}), 500


# ── API: Perfil de voz ────────────────────────────────────────────────────────

@app.route("/api/profile", methods=["GET"])
def api_get_profile():
    from src.voice_profile import VoiceProfile
    profile = VoiceProfile.load(PROFILE_PATH)
    from dataclasses import asdict
    return jsonify(asdict(profile))


@app.route("/api/profile", methods=["POST"])
def api_save_profile():
    from src.voice_profile import VoiceProfile
    data = request.json or {}
    try:
        profile = VoiceProfile(**{
            k: v for k, v in data.items()
            if k in VoiceProfile.__dataclass_fields__
        })
        profile.save(PROFILE_PATH)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Posts guardados ──────────────────────────────────────────────────────

@app.route("/api/posts")
def api_list_posts():
    files = sorted(glob.glob("posts_*.json"), reverse=True)
    result = []
    for f in files[:20]:
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            result.append({
                "filename":     f,
                "query":        data.get("query", ""),
                "generated_at": data.get("generated_at", ""),
                "model":        data.get("model", ""),
                "num_posts":    len(data.get("posts", [])),
            })
        except Exception:
            pass
    return jsonify(result)


@app.route("/api/posts/<filename>")
def api_get_posts(filename):
    try:
        with open(filename, encoding="utf-8") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"error": "Archivo no encontrado"}), 404


# ── Main ──────────────────────────────────────────────────────────────────────


# ── API: News Scraper ─────────────────────────────────────────────────────────

@app.route("/api/scrape", methods=["POST"])
def api_scrape():
    data         = request.json or {}
    filter_model = data.get("model")       # None = todos
    summarize    = data.get("summarize", True)
    max_articles = int(data.get("max", 5))

    try:
        from news_scraper import run_scraper
        saved = run_scraper(
            seen_file=SEEN_ARTICLES,
            filter_model = filter_model,
            summarize    = summarize and bool(os.getenv("ANTHROPIC_API_KEY")),
            max_articles = max_articles,
            auto_index   = True,
        )
        store = _get_store()
        stats = store.count()
        store.close()
        return jsonify({
            "success":       True,
            "articles_saved": len(saved),
            "files":         [str(p.name) for p in saved],
            "index_stats":   stats,
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "detail": traceback.format_exc()}), 500

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 AI Notes Agent — Interfaz Web")
    print("="*50)
    print(f"   VOYAGE_API_KEY:    {'✅' if os.getenv('VOYAGE_API_KEY') else '⚠️  no encontrada'}")
    print(f"   ANTHROPIC_API_KEY: {'✅' if os.getenv('ANTHROPIC_API_KEY') else '⚠️  no encontrada'}")
    print(f"   Índice:            {'✅ ' + DB_PATH if Path(DB_PATH).exists() else '⚠️  no existe'}")
    print(f"\n   Abre en el navegador: http://localhost:5000")
    print("="*50 + "\n")
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, port=port, host="0.0.0.0")
