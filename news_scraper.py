"""
news_scraper.py — Scraper de noticias de IA → Notas Markdown

Fuentes RSS gratuitas y legales:
  - The Verge (AI)
  - VentureBeat (AI)
  - Ars Technica
  - MIT Technology Review
  - Google DeepMind Blog
  - Anthropic Blog
  - OpenAI Blog
  - TechCrunch (AI)
  - Hugging Face Blog

Flujo:
  1. Lee los RSS feeds
  2. Filtra artículos sobre los modelos que te interesan
  3. Claude resume cada artículo en una nota estructurada
  4. Guarda como .md con frontmatter en la carpeta de notas
  5. Indexa automáticamente en el vector store

Uso:
  # Scraping manual
  python news_scraper.py

  # Solo un modelo
  python news_scraper.py --model gemini

  # Sin resumir con Claude (solo guarda el texto raw)
  python news_scraper.py --no-summarize

  # Programar diariamente en Windows Task Scheduler:
  # Acción: python C:\\ruta\\ai-notes-agent-v4\\news_scraper.py
"""

import os
import sys
import re
import json
import time
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# ── Dependencias ──────────────────────────────────────────────────────────────
try:
    import feedparser
except ImportError:
    print("Instala feedparser: pip install feedparser")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Instala requests: pip install requests")
    sys.exit(1)

# Cargar .env
sys.path.insert(0, str(Path(__file__).parent))
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Configuración ─────────────────────────────────────────────────────────────

NOTES_DIR     = Path("notes_ai_news")     # carpeta donde se guardan las notas
SEEN_FILE     = Path(".seen_articles.json") # evita duplicados entre ejecuciones
MAX_ARTICLES  = 5                          # máximo por ejecución (evita costes altos)
REQUEST_DELAY = 1.5                        # segundos entre requests


# ── RSS Feeds ─────────────────────────────────────────────────────────────────

RSS_FEEDS = [
    {
        "name":   "The Verge — AI",
        "url":    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
        "lang":   "en",
    },
    {
        "name":   "VentureBeat — AI",
        "url":    "https://venturebeat.com/category/ai/feed/",
        "lang":   "en",
    },
    {
        "name":   "Ars Technica — Tech",
        "url":    "https://feeds.arstechnica.com/arstechnica/technology-lab",
        "lang":   "en",
    },
    {
        "name":   "MIT Technology Review",
        "url":    "https://www.technologyreview.com/feed/",
        "lang":   "en",
    },
    {
        "name":   "TechCrunch — AI",
        "url":    "https://techcrunch.com/category/artificial-intelligence/feed/",
        "lang":   "en",
    },
    {
        "name":   "Google DeepMind Blog",
        "url":    "https://deepmind.google/blog/rss.xml",
        "lang":   "en",
    },
    {
        "name":   "Anthropic Blog",
        "url":    "https://www.anthropic.com/rss.xml",
        "lang":   "en",
    },
    {
        "name":   "OpenAI Blog",
        "url":    "https://openai.com/news/rss.xml",
        "lang":   "en",
    },
    {
        "name":   "Hugging Face Blog",
        "url":    "https://huggingface.co/blog/feed.xml",
        "lang":   "en",
    },
]


# ── Keywords de modelos y temas de interés ────────────────────────────────────

MODEL_KEYWORDS = {
    "claude":    ["claude", "anthropic", "claude 3", "claude sonnet", "claude opus", "claude haiku"],
    "chatgpt":   ["chatgpt", "chat gpt", "openai", "gpt-4", "gpt-4o", "gpt-5", "o1", "o3"],
    "gemini":    ["gemini", "google ai", "deepmind", "bard", "gemini pro", "gemini ultra", "gemini flash"],
    "llama":     ["llama", "meta ai", "llama 3", "llama 4"],
    "mistral":   ["mistral", "mixtral"],
    "grok":      ["grok", "xai", "x.ai"],
    "general":   [
        "large language model", "llm", "foundation model",
        "ai model", "modelo de ia", "inteligencia artificial",
        "machine learning", "deep learning", "multimodal",
        "rag", "retrieval", "fine-tuning", "prompt engineering",
        "ai agent", "agentic", "reasoning model",
    ],
}

ALL_KEYWORDS = [kw for kws in MODEL_KEYWORDS.values() for kw in kws]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _article_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _load_seen() -> set:
    if SEEN_FILE.exists():
        return set(json.loads(SEEN_FILE.read_text()))
    return set()


def _save_seen(seen: set):
    SEEN_FILE.write_text(json.dumps(list(seen)))


def _safe_filename(title: str, date: str) -> str:
    clean = re.sub(r'[^\w\s-]', '', title.lower())
    clean = re.sub(r'[\s_]+', '-', clean).strip('-')[:60]
    return f"{date}_{clean}.md"


def _matches_keywords(text: str, filter_model: Optional[str] = None) -> tuple[bool, list[str]]:
    """Devuelve (matches, modelos_detectados)."""
    lower = text.lower()
    if filter_model:
        kws = MODEL_KEYWORDS.get(filter_model, [])
        matched = [k for k in kws if k in lower]
        return bool(matched), [filter_model] if matched else []

    matched_models = []
    for model, kws in MODEL_KEYWORDS.items():
        if any(k in lower for k in kws):
            matched_models.append(model)
    return bool(matched_models), matched_models


def _strip_html(text: str) -> str:
    """Elimina tags HTML básicos."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── RSS Fetcher ───────────────────────────────────────────────────────────────

def fetch_feed(feed_config: dict) -> list[dict]:
    """Lee un RSS feed y devuelve artículos como dicts."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; AINotesAgent/1.0)',
        'Accept': 'application/rss+xml, application/xml, text/xml',
    }
    try:
        resp = requests.get(
            feed_config["url"],
            headers=headers,
            timeout=15,
            allow_redirects=True,
        )
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except Exception as e:
        print(f"   ⚠️  {feed_config['name']}: {e}")
        return []

    articles = []
    for entry in feed.entries[:20]:
        # Extrae texto del summary/content
        summary = ""
        if hasattr(entry, "content"):
            summary = _strip_html(entry.content[0].get("value", ""))
        elif hasattr(entry, "summary"):
            summary = _strip_html(entry.summary)

        # Extrae fecha
        date_str = datetime.now().strftime("%Y-%m-%d")
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                date_str = dt.strftime("%Y-%m-%d")
            except Exception:
                pass

        articles.append({
            "id":      _article_id(entry.get("link", entry.get("id", ""))),
            "title":   entry.get("title", "Sin título"),
            "url":     entry.get("link", ""),
            "summary": summary[:3000],
            "date":    date_str,
            "source":  feed_config["name"],
        })
    return articles


# ── Claude Summarizer ─────────────────────────────────────────────────────────

SUMMARY_SYSTEM = """\
Eres un asistente que convierte artículos de noticias sobre IA en notas de conocimiento
estructuradas en español. Tu objetivo es extraer las ideas clave y presentarlas de forma
clara y reutilizable para generar contenido en LinkedIn.

IMPORTANTE:
- Responde SOLO con el contenido de la nota, sin explicaciones adicionales
- Escribe en español, aunque el artículo esté en inglés
- Sé concreto: hechos, capacidades, comparaciones, fechas de lanzamiento
- Máximo 400 palabras
- Usa secciones con ## para organizar
"""

SUMMARY_PROMPT = """\
Convierte este artículo en una nota de conocimiento estructurada:

TÍTULO: {title}
FUENTE: {source}
FECHA: {date}
URL: {url}

CONTENIDO:
{summary}

---

Estructura tu nota así:
## Qué es / Qué ocurrió
(1-2 frases resumen)

## Puntos clave
- punto 1
- punto 2
- punto 3

## Por qué importa
(1 párrafo sobre el impacto o relevancia)

## Para recordar
(1 frase memorable o dato concreto)
"""


def summarize_with_claude(article: dict) -> Optional[str]:
    """Usa Claude para resumir el artículo en español."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=SUMMARY_SYSTEM,
            messages=[{
                "role": "user",
                "content": SUMMARY_PROMPT.format(**article)
            }]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"   ⚠️  Claude error: {e}")
        return None


# ── Note Writer ───────────────────────────────────────────────────────────────

def save_note(article: dict, content: str, models: list[str]) -> Path:
    """Guarda el artículo como nota Markdown con frontmatter."""
    NOTES_DIR.mkdir(exist_ok=True)

    filename = _safe_filename(article["title"], article["date"])
    filepath = NOTES_DIR / filename

    # Tags: modelos detectados + fuente normalizada
    tags = list(set(models + ["ia-noticias", "news"]))

    frontmatter = f"""---
title: "{article['title'].replace('"', "'")}"
date: {article['date']}
source: {article['source']}
url: {article['url']}
tags: [{', '.join(tags)}]
scraped_at: {datetime.now().isoformat()}
---

"""
    filepath.write_text(frontmatter + content, encoding="utf-8")
    return filepath


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_scraper(
    filter_model: Optional[str] = None,
    summarize: bool = True,
    max_articles: int = MAX_ARTICLES,
    auto_index: bool = True,
    notes_dir: Optional[Path] = None,
    seen_file: Optional[str] = None,
):
    global NOTES_DIR, SEEN_FILE
    if notes_dir: NOTES_DIR = Path(notes_dir)
    if seen_file: SEEN_FILE = Path(seen_file)

    print(f"\n{'='*55}")
    print(f"🗞️  AI News Scraper")
    print(f"{'='*55}")
    print(f"   Modelos: {filter_model or 'todos'}")
    print(f"   Resumir con Claude: {'sí' if summarize and os.getenv('ANTHROPIC_API_KEY') else 'no (sin ANTHROPIC_API_KEY)'}")
    print(f"   Carpeta destino: {NOTES_DIR.resolve()}\n")

    seen      = _load_seen()
    saved     = []
    checked   = 0

    for feed_cfg in RSS_FEEDS:
        print(f"📡 {feed_cfg['name']}…", end=" ", flush=True)
        articles = fetch_feed(feed_cfg)
        new = [a for a in articles if a["id"] not in seen]
        print(f"{len(new)} nuevos de {len(articles)}")
        time.sleep(REQUEST_DELAY)

        for article in new:
            if len(saved) >= max_articles:
                break

            # Filtrar por keywords
            text = f"{article['title']} {article['summary']}"
            matches, models = _matches_keywords(text, filter_model)
            if not matches:
                continue

            checked += 1
            print(f"\n   → {article['title'][:65]}…")
            print(f"     Modelos: {', '.join(models)} | Fuente: {article['source']}")

            # Resumir con Claude o usar summary raw
            if summarize and os.getenv("ANTHROPIC_API_KEY"):
                print(f"     ✍️  Resumiendo con Claude…", end=" ", flush=True)
                content = summarize_with_claude(article)
                if content:
                    print("✓")
                else:
                    content = f"## {article['title']}\n\n{article['summary']}"
                    print("(fallback a texto raw)")
            else:
                content = (
                    f"## {article['title']}\n\n"
                    f"{article['summary']}\n\n"
                    f"Fuente: {article['url']}"
                )

            # Guardar nota
            filepath = save_note(article, content, models)
            seen.add(article["id"])
            saved.append(filepath)
            print(f"     💾 Guardado: {filepath.name}")

        if len(saved) >= max_articles:
            print(f"\n   (límite de {max_articles} artículos alcanzado)")
            break

    _save_seen(seen)

    print(f"\n{'='*55}")
    print(f"✅ {len(saved)} notas guardadas en {NOTES_DIR}/")

    # Auto-indexar
    if saved and auto_index:
        print(f"\n🗂️  Indexando notas nuevas…")
        try:
            from src.parsers.markdown_parser import MarkdownParser
            from src.parsers.chunker import Chunker
            from src.embeddings.embedder import Embedder
            from src.db.vector_store import LocalVectorStore

            simulate = not bool(os.getenv("VOYAGE_API_KEY"))
            parser   = MarkdownParser()
            chunker  = Chunker()
            embedder = Embedder(simulate=simulate)
            store    = LocalVectorStore("notes_index.db")

            notes    = parser.parse_directory(NOTES_DIR)
            chunks   = chunker.chunk_many(notes)
            embedded = embedder.embed(chunks, verbose=False)
            store.upsert(embedded)
            stats = store.count()
            store.close()

            print(f"✅ Indexadas. Total en el índice: {stats['total_notes']} notas / {stats['total_chunks']} chunks")
        except Exception as e:
            print(f"⚠️  Error al indexar: {e}")
            print(f"   Puedes indexar manualmente: python indexer.py index {NOTES_DIR}")

    print(f"\n💡 Genera un post sobre las noticias de hoy:")
    print(f"   python indexer.py generate \"novedades en modelos de IA\"")
    print(f"{'='*55}\n")

    return saved


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI News Scraper")
    parser.add_argument("--model",
        choices=list(MODEL_KEYWORDS.keys()),
        help="Filtrar por modelo específico (claude, chatgpt, gemini, llama, mistral, grok, general)")
    parser.add_argument("--no-summarize", action="store_true",
        help="No usar Claude para resumir (más rápido, sin coste)")
    parser.add_argument("--max", type=int, default=MAX_ARTICLES,
        help=f"Máximo de artículos a guardar (default: {MAX_ARTICLES})")
    parser.add_argument("--no-index", action="store_true",
        help="No indexar automáticamente después de scraping")
    args = parser.parse_args()

    run_scraper(
        filter_model = args.model,
        summarize    = not args.no_summarize,
        max_articles = args.max,
        auto_index   = not args.no_index,
    )
