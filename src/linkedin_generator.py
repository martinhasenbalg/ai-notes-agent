"""
linkedin_generator.py — Generador de posts LinkedIn con Claude (Fase 3)

Pipeline completo:
  query → Retriever → contexto → Claude → 3 variantes de post

Cada post incluye:
  - Hook impactante (primera línea que para el scroll)
  - Desarrollo con las ideas clave de las notas
  - CTA adaptado al estilo del autor
  - Hashtags relevantes

Uso:
  from src.linkedin_generator import LinkedInGenerator
  gen = LinkedInGenerator()
  result = gen.generate("RAG y retrieval semántico")
  print(result.posts[0].content)
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.retriever import Retriever, RetrievalResult
from src.voice_profile import VoiceProfile


# ── Configuración ─────────────────────────────────────────────────────────────

CLAUDE_MODEL    = "claude-sonnet-4-6"
MAX_TOKENS      = 2000
NUM_VARIANTS    = 3

LENGTH_WORDS = {
    "corto":  150,
    "medio":  250,
    "largo":  400,
}


# ── Modelos de datos ──────────────────────────────────────────────────────────

@dataclass
class LinkedInPost:
    """Un post de LinkedIn generado."""
    variant: int                    # 1, 2 o 3
    content: str                    # texto completo listo para copiar
    hook: str                       # primera línea (para preview)
    hashtags: list[str]
    word_count: int
    style_label: str                # "Directo", "Narrativo", "Con datos"
    sources: list[str]              # títulos de las notas fuente

    def preview(self) -> str:
        return (
            f"── Variante {self.variant}: {self.style_label} "
            f"({self.word_count} palabras) ──\n"
            f"Hook: {self.hook[:80]}{'…' if len(self.hook) > 80 else ''}\n"
        )


@dataclass
class GenerationResult:
    """Resultado completo de una generación."""
    query: str
    posts: list[LinkedInPost]
    retrieval: RetrievalResult
    model: str
    elapsed_ms: float
    generated_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    @property
    def best(self) -> LinkedInPost:
        return self.posts[0] if self.posts else None

    def print_all(self):
        print(f"\n{'='*60}")
        print(f"📝 Posts generados para: \"{self.query}\"")
        print(f"   Fuentes: {', '.join(r.title for r in self.retrieval.chunks[:3] if r.title)}")
        print(f"   Modelo:  {self.model} | Tiempo: {self.elapsed_ms:.0f}ms")
        print(f"{'='*60}\n")
        for post in self.posts:
            print(post.preview())
            print(post.content)
            print()


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Eres un experto en contenido para LinkedIn especializado en tecnología e inteligencia artificial.
Tu tarea es generar posts de LinkedIn auténticos y de alto impacto basados en notas personales.

REGLAS FUNDAMENTALES:
1. Usa SOLO la información del contexto proporcionado — no inventes datos ni estadísticas
2. Cada variante debe tener un enfoque distinto (ver estilos)
3. El hook (primera línea) es crítico: debe parar el scroll en 3 segundos
4. Escribe en el idioma y con la voz del autor según su perfil
5. Responde ÚNICAMENTE con JSON válido, sin texto adicional antes ni después

ESTRUCTURA DEL JSON DE RESPUESTA:
{
  "posts": [
    {
      "variant": 1,
      "style_label": "Directo y provocador",
      "hook": "primera línea del post",
      "content": "texto completo del post con saltos de línea reales",
      "hashtags": ["hashtag1", "hashtag2", "hashtag3", "hashtag4", "hashtag5"]
    },
    ...
  ]
}

ESTILOS DE VARIANTES:
- Variante 1 — "Directo": afirmación contundente + puntos clave numerados + CTA
- Variante 2 — "Narrativo": empieza con una historia o situación + desarrollo + reflexión
- Variante 3 — "Con datos/comparativa": estructura comparativa o con insights específicos

HOOKS QUE FUNCIONAN EN LINKEDIN:
- Pregunta que crea curiosidad: "¿Sabías que el 80% de los proyectos RAG fallan por esto?"
- Afirmación contraintuitiva: "Naive RAG es casi siempre un error. Te explico por qué."
- Número específico: "3 patrones de RAG que todo ingeniero de IA debería conocer"
- Experiencia personal: "Llevo 6 meses probando arquitecturas RAG. Esto es lo que aprendí."
"""


def _build_user_prompt(
    query: str,
    context: str,
    profile: VoiceProfile,
    num_variants: int = NUM_VARIANTS,
) -> str:
    word_target = LENGTH_WORDS.get(profile.post_length, 250)

    return f"""
{profile.to_prompt_block()}

---

{context}

---

INSTRUCCIÓN:
Genera {num_variants} variantes de post de LinkedIn sobre el tema: "{query}"

Requisitos de formato:
- Longitud: ~{word_target} palabras por post
- Emojis: {'sí, úsalos estratégicamente' if profile.use_emojis else 'no usar emojis'}
- Hashtags: {'sí, ' + str(profile.hashtag_count) + ' hashtags temáticos al final' if profile.use_hashtags else 'no usar hashtags'}
- CTA al final: {profile.cta_style}
- Idioma: {'español' if profile.language == 'es' else 'inglés'}
- Primera persona: {'sí ("Hoy aprendí...", "En mi experiencia...")' if profile.first_person else 'no'}

Devuelve SOLO el JSON, sin markdown ni explicaciones.
"""


# ── Generator ─────────────────────────────────────────────────────────────────

class LinkedInGenerator:
    """
    Genera posts de LinkedIn combinando RAG + Claude.

    Uso:
        gen = LinkedInGenerator()
        result = gen.generate("técnicas avanzadas de RAG")
        result.print_all()

    Con perfil personalizado:
        from src.voice_profile import VoiceProfile
        profile = VoiceProfile.load()
        gen = LinkedInGenerator(profile=profile)
        result = gen.generate("vector databases comparativa")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        db_path: str = "notes_index.db",
        profile: Optional[VoiceProfile] = None,
        simulate: bool = False,
    ):
        self.simulate = simulate
        self.profile  = profile or VoiceProfile.load()
        self.retriever = Retriever(db_path=db_path, simulate=simulate)
        self._client  = None

        if not simulate:
            self._init_claude(api_key)

    def _init_claude(self, api_key: Optional[str]):
        try:
            import anthropic
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                print(
                    "⚠️  ANTHROPIC_API_KEY no encontrada.\n"
                    "   Consíguela en https://console.anthropic.com/settings/keys\n"
                    "   Activando modo simulado."
                )
                self.simulate = True
                return
            self._client = anthropic.Anthropic(api_key=key)
            print(f"✅ Claude conectado — modelo: {CLAUDE_MODEL}")
        except ImportError:
            print("⚠️  anthropic no instalado: pip install anthropic")
            self.simulate = True

    # ── Generación principal ──────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        top_k: int = 4,
        tags: Optional[list[str]] = None,
        source_type: Optional[str] = None,
        num_variants: int = NUM_VARIANTS,
        verbose: bool = True,
    ) -> GenerationResult:
        """
        Genera posts de LinkedIn sobre un tema buscando en tus notas.

        Args:
            query:        Tema o pregunta en lenguaje natural
            top_k:        Número de chunks a recuperar como contexto
            tags:         Filtrar notas por tags
            source_type:  Filtrar por tipo de fuente
            num_variants: Número de variantes a generar (1-3)
            verbose:      Mostrar progreso
        """
        t0 = time.time()

        if verbose:
            print(f"\n🔎 Buscando contexto para: \"{query}\"")

        # 1. Recuperar contexto con RAG
        retrieval = self.retriever.search(
            query, top_k=top_k, tags=tags, source_type=source_type
        )

        if not retrieval.found:
            print("⚠️  No se encontró contexto relevante. Indexa más notas.")
            return GenerationResult(
                query=query, posts=[], retrieval=retrieval,
                model=CLAUDE_MODEL, elapsed_ms=0
            )

        if verbose:
            print(f"   ✓ {len(retrieval.chunks)} fragmentos recuperados de: "
                  f"{', '.join(retrieval.unique_notes)}")
            print(f"\n✍️  Generando {num_variants} variantes con Claude…")

        # 2. Construir prompt
        user_prompt = _build_user_prompt(
            query, retrieval.context, self.profile, num_variants
        )

        # 3. Llamar a Claude (o simular)
        raw_json = self._call_claude(user_prompt) if not self.simulate \
                   else self._simulate_response(query, retrieval, num_variants)

        # 4. Parsear respuesta
        posts = self._parse_posts(raw_json, retrieval)

        elapsed = (time.time() - t0) * 1000

        if verbose:
            print(f"   ✓ {len(posts)} posts generados en {elapsed:.0f}ms\n")

        return GenerationResult(
            query=query,
            posts=posts,
            retrieval=retrieval,
            model=CLAUDE_MODEL if not self.simulate else "simulado",
            elapsed_ms=elapsed,
        )

    # ── Claude API ────────────────────────────────────────────────────────────

    def _call_claude(self, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    def _simulate_response(
        self,
        query: str,
        retrieval: RetrievalResult,
        num_variants: int,
    ) -> str:
        """Genera posts de ejemplo en modo simulado (sin API key)."""
        source_titles = ", ".join(retrieval.unique_notes[:2])
        posts = []
        styles = [
            ("Directo y provocador",
             f"La mayoría lo hace mal. Así se hace bien: {query}.",
             f"La mayoría implementa {query} sin entender los fundamentos.\n\n"
             f"Después de revisar mis notas sobre {source_titles}, esto es lo que realmente importa:\n\n"
             f"→ El retrieval de calidad vale más que el modelo más potente\n"
             f"→ Los chunks mal diseñados arruinan cualquier arquitectura\n"
             f"→ Sin métricas, no sabes si funciona o simplemente parece que funciona\n\n"
             f"¿Cuál de estos errores has cometido tú?"),
            ("Narrativo",
             f"Hace unos meses empecé a explorar {query} en serio.",
             f"Hace unos meses empecé a explorar {query} en serio.\n\n"
             f"Lo que encontré en mis notas cambió cómo pienso sobre el tema.\n\n"
             f"La clave no está en la tecnología. Está en entender el problema antes de elegir la solución.\n\n"
             f"Mis aprendizajes de {source_titles}:\n\n"
             f"1. Empieza simple. Naive RAG resuelve el 70% de los casos.\n"
             f"2. Mide antes de optimizar. No añadas complejidad sin datos.\n"
             f"3. El chunking es más arte que ciencia.\n\n"
             f"¿Qué añadirías tú a esta lista?"),
            ("Con datos/comparativa",
             f"3 enfoques para {query}. Solo uno vale para producción.",
             f"3 enfoques para {query}. Solo uno vale para producción.\n\n"
             f"Según mis notas de {source_titles}:\n\n"
             f"❌ Naive RAG — rápido de implementar, pobre en precisión\n"
             f"⚠️  Advanced RAG — mejor recall, más complejo de mantener\n"
             f"✅ Modular RAG — flexible, escalable, production-ready\n\n"
             f"La diferencia real está en cómo gestionas el contexto que le pasas al LLM.\n\n"
             f"Comparte si te ha resultado útil. 👇"),
        ]
        for i, (label, hook, content) in enumerate(styles[:num_variants], 1):
            posts.append({
                "variant": i,
                "style_label": label,
                "hook": hook,
                "content": content,
                "hashtags": ["#InteligenciaArtificial", "#LLM", "#RAG", "#MachineLearning", "#IA"],
            })
        return json.dumps({"posts": posts})

    # ── Parseo ────────────────────────────────────────────────────────────────

    def _parse_posts(
        self,
        raw_json: str,
        retrieval: RetrievalResult,
    ) -> list[LinkedInPost]:
        # Limpia posibles bloques markdown que Claude añada
        clean = raw_json.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1]
            if clean.endswith("```"):
                clean = clean.rsplit("```", 1)[0]

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            print(f"⚠️  Error parseando JSON de Claude: {e}")
            print(f"   Raw: {clean[:200]}")
            return []

        source_titles = [c.title for c in retrieval.chunks if c.title]
        posts = []

        for p in data.get("posts", []):
            content = p.get("content", "")
            hashtags = p.get("hashtags", [])

            # Si los hashtags no están en el contenido, añádelos al final
            if hashtags and not any(h in content for h in hashtags[:1]):
                content = content.rstrip() + "\n\n" + " ".join(hashtags)

            posts.append(LinkedInPost(
                variant    = p.get("variant", len(posts) + 1),
                content    = content,
                hook       = p.get("hook", content.split("\n")[0][:100]),
                hashtags   = hashtags,
                word_count = len(content.split()),
                style_label= p.get("style_label", f"Variante {len(posts)+1}"),
                sources    = source_titles,
            ))

        return posts
