"""
embeddings/embedder.py — Generación de embeddings

Stack Anthropic-nativo:
  - LLM generación  → Claude (anthropic SDK)      ← Fase 3
  - Embeddings       → Voyage AI (partner oficial de Anthropic)
                       voyage-3-lite: 512 dims, muy rápido y barato
                       voyage-3:      1024 dims, mejor calidad

Anthropic NO tiene API propia de embeddings — Voyage AI es su partner
recomendado: https://docs.anthropic.com/en/docs/build-with-claude/embeddings

Variables de entorno necesarias:
  VOYAGE_API_KEY   → https://dash.voyageai.com  (gratis hasta 50M tokens/mes)

Modo simulado (--simulate):
  Genera vectores deterministas sin API key, útil para desarrollo local.
"""

import os
import time
import random
import math
from typing import Optional

from src.schema import Chunk, EmbeddedChunk


# ── Modelos disponibles ──────────────────────────────────────────────────────
MODELS = {
    "voyage-3-lite": {"dims": 512,  "ctx": 32000, "cost_per_1m": 0.02},
    "voyage-3":      {"dims": 1024, "ctx": 32000, "cost_per_1m": 0.06},
    "voyage-3-large":{"dims": 2048, "ctx": 32000, "cost_per_1m": 0.18},
}

DEFAULT_MODEL   = "voyage-3-lite"   # rápido, gratuito generoso, suficiente para notas
BATCH_SIZE      = 128               # máximo por request Voyage AI
MAX_RETRIES     = 3
BASE_DELAY      = 1.0
CHARS_PER_TOKEN = 4


# ── Helpers ──────────────────────────────────────────────────────────────────

def _estimate_tokens(texts: list[str]) -> int:
    return sum(len(t) for t in texts) // CHARS_PER_TOKEN


def _fake_embedding(text: str, dimensions: int) -> list[float]:
    """Embedding aleatorio normalizado y reproducible (modo simulado)."""
    random.seed(hash(text) % (2**32))
    raw = [random.gauss(0, 1) for _ in range(dimensions)]
    norm = math.sqrt(sum(x**2 for x in raw))
    return [x / norm for x in raw] if norm > 0 else raw


# ── Embedder ─────────────────────────────────────────────────────────────────

class Embedder:
    """
    Genera embeddings con Voyage AI (partner oficial de Anthropic).

    Args:
        api_key:  Voyage AI key. Si None, usa VOYAGE_API_KEY del entorno.
        model:    "voyage-3-lite" (default) | "voyage-3" | "voyage-3-large"
        simulate: True = vectores aleatorios, sin llamadas a API.

    Obtener API key gratis:
        https://dash.voyageai.com  (50M tokens/mes gratuitos)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        simulate: bool = False,
    ):
        self.model    = model
        self.simulate = simulate
        self.dims     = MODELS.get(model, MODELS[DEFAULT_MODEL])["dims"]
        self._client  = None

        if not simulate:
            self._init_client(api_key)

    def _init_client(self, api_key: Optional[str]):
        try:
            import voyageai
            key = api_key or os.getenv("VOYAGE_API_KEY")
            if not key:
                print(
                    "⚠️  VOYAGE_API_KEY no encontrada.\n"
                    "   Consíguela gratis en https://dash.voyageai.com\n"
                    "   Activando modo simulado."
                )
                self.simulate = True
                return
            self._client = voyageai.Client(api_key=key)
            print(f"✅ Voyage AI conectado — modelo: {self.model} ({self.dims} dims)")
        except ImportError:
            print("⚠️  voyageai no instalado: pip install voyageai")
            self.simulate = True

    def estimate(self, chunks: list[Chunk]) -> dict:
        texts  = [c.content for c in chunks]
        tokens = _estimate_tokens(texts)
        cpm    = MODELS.get(self.model, {}).get("cost_per_1m", 0.02)
        return {
            "chunks":             len(chunks),
            "estimated_tokens":   tokens,
            "estimated_cost_usd": round((tokens / 1_000_000) * cpm, 6),
            "model":              self.model,
            "dimensions":         self.dims,
        }

    def embed(self, chunks: list[Chunk], verbose: bool = True) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        if verbose:
            est  = self.estimate(chunks)
            mode = "simulado" if self.simulate else f"Voyage AI / {self.model}"
            print(f"\n📐 Embeddings — {mode}")
            print(f"   Chunks: {est['chunks']}  |  Tokens: ~{est['estimated_tokens']:,}  |  Dims: {est['dimensions']}")
            if not self.simulate:
                print(f"   Coste estimado: ~${est['estimated_cost_usd']}")

        results: list[EmbeddedChunk] = []
        n_batches = math.ceil(len(chunks) / BATCH_SIZE)

        for i in range(n_batches):
            batch  = chunks[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            if verbose:
                print(f"   Batch {i+1}/{n_batches} ({len(batch)} chunks)…", end=" ", flush=True)
            vectors = self._embed_with_retry([c.content for c in batch])
            for chunk, vector in zip(batch, vectors):
                results.append(EmbeddedChunk(chunk=chunk, embedding=vector, model=self.model))
            if verbose:
                print("✓")

        if verbose:
            print(f"\n✅ {len(results)} embeddings listos\n")
        return results

    def embed_query(self, query: str) -> list[float]:
        """Vectoriza una query de búsqueda (string único)."""
        if self.simulate:
            return _fake_embedding(query, self.dims)
        return self._embed_with_retry([query], input_type="query")[0]

    def _embed_with_retry(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        if self.simulate:
            return [_fake_embedding(t, self.dims) for t in texts]

        for attempt in range(MAX_RETRIES):
            try:
                result = self._client.embed(texts, model=self.model, input_type=input_type)
                embs = result.embeddings
                if embs and hasattr(embs[0], "embedding"):
                    return [e.embedding for e in embs]
                return embs
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                print(f"\n   ⚠️  Reintentando ({attempt+1}/{MAX_RETRIES}) en {delay:.1f}s — {e}")
                time.sleep(delay)
        return []
