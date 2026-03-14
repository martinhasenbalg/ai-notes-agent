"""
voice_profile.py — Perfil de voz personal del usuario

Captura el tono, estilo y preferencias para que Claude genere posts
que suenen como tú, no como un post genérico de LinkedIn.

Se guarda en voice_profile.json y se carga automáticamente.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


PROFILE_PATH = "voice_profile.json"


@dataclass
class VoiceProfile:
    """
    Perfil de voz del autor para guiar la generación de posts.
    """
    # Identidad
    name: str = ""
    role: str = ""                    # "Ingeniero de IA", "CTO", "Consultor"
    industry: str = ""                # "tecnología", "startups", "educación"

    # Tono y estilo
    tone: str = "profesional"         # profesional | conversacional | técnico | inspiracional
    technicality: str = "medio"       # básico | medio | avanzado
    language: str = "es"              # es | en

    # Preferencias de formato
    use_emojis: bool = True
    use_hashtags: bool = True
    hashtag_count: int = 5
    post_length: str = "medio"        # corto (~150 palabras) | medio (~250) | largo (~400)

    # Voz narrativa
    first_person: bool = True         # "Hoy aprendí..." vs "Se ha descubierto..."
    storytelling: bool = True         # incluir experiencias personales
    cta_style: str = "pregunta"       # pregunta | invitación | reflexión | ninguno

    # Ejemplos de posts propios (few-shot para Claude)
    example_posts: list[str] = field(default_factory=list)

    # Temas recurrentes / audiencia
    audience: str = ""                # "profesionales de IA", "emprendedores tech"
    avoid_topics: list[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        """Convierte el perfil en instrucciones para el system prompt de Claude."""
        lines = ["## Perfil de voz del autor\n"]

        if self.name:
            lines.append(f"**Autor:** {self.name}")
        if self.role:
            lines.append(f"**Rol:** {self.role}")
        if self.industry:
            lines.append(f"**Industria:** {self.industry}")
        if self.audience:
            lines.append(f"**Audiencia objetivo:** {self.audience}")

        lines.append(f"\n**Tono:** {self.tone}")
        lines.append(f"**Nivel técnico:** {self.technicality}")
        lines.append(f"**Longitud de post:** {self.post_length}")
        lines.append(f"**Perspectiva:** {'primera persona' if self.first_person else 'tercera persona'}")
        lines.append(f"**Storytelling:** {'sí, incluir experiencias personales' if self.storytelling else 'no'}")
        lines.append(f"**Emojis:** {'sí' if self.use_emojis else 'no'}")
        lines.append(f"**Hashtags:** {'sí, ' + str(self.hashtag_count) + ' al final' if self.use_hashtags else 'no'}")
        lines.append(f"**CTA:** {self.cta_style}")

        if self.avoid_topics:
            lines.append(f"**Evitar:** {', '.join(self.avoid_topics)}")

        if self.example_posts:
            lines.append("\n**Ejemplos de posts del autor (imitar este estilo):**")
            for i, ex in enumerate(self.example_posts[:2], 1):
                lines.append(f"\nEjemplo {i}:\n```\n{ex}\n```")

        return "\n".join(lines)

    def save(self, path: str = PROFILE_PATH):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        print(f"✅ Perfil guardado en {path}")

    @classmethod
    def load(cls, path: str = PROFILE_PATH) -> "VoiceProfile":
        if not Path(path).exists():
            return cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_onboarding(cls) -> "VoiceProfile":
        """Onboarding interactivo por consola (3 minutos)."""
        print("\n" + "="*50)
        print("✍️  Configura tu perfil de voz para LinkedIn")
        print("="*50)
        print("(Pulsa Enter para usar el valor por defecto)\n")

        def ask(prompt: str, default: str = "") -> str:
            val = input(f"{prompt} [{default}]: ").strip()
            return val if val else default

        def ask_bool(prompt: str, default: bool = True) -> bool:
            default_str = "S/n" if default else "s/N"
            val = input(f"{prompt} [{default_str}]: ").strip().lower()
            if not val:
                return default
            return val in ("s", "si", "sí", "y", "yes")

        profile = cls()
        profile.name       = ask("Tu nombre")
        profile.role       = ask("Tu rol profesional", "Profesional de IA")
        profile.industry   = ask("Tu industria", "tecnología")
        profile.audience   = ask("Tu audiencia en LinkedIn", "profesionales de tecnología e IA")

        print("\n── Estilo de escritura ──")
        tone_opts = "profesional / conversacional / técnico / inspiracional"
        profile.tone         = ask(f"Tono ({tone_opts})", "profesional")
        tech_opts = "básico / medio / avanzado"
        profile.technicality = ask(f"Nivel técnico ({tech_opts})", "medio")
        length_opts = "corto / medio / largo"
        profile.post_length  = ask(f"Longitud de post ({length_opts})", "medio")

        print("\n── Formato ──")
        profile.use_emojis   = ask_bool("¿Usar emojis?", True)
        profile.use_hashtags = ask_bool("¿Usar hashtags?", True)
        if profile.use_hashtags:
            n = ask("¿Cuántos hashtags?", "5")
            profile.hashtag_count = int(n) if n.isdigit() else 5

        cta_opts = "pregunta / invitación / reflexión / ninguno"
        profile.cta_style = ask(f"Tipo de CTA ({cta_opts})", "pregunta")

        print("\n── Opcional: pega un ejemplo de tu estilo ──")
        print("(Pega un post tuyo que te guste, o pulsa Enter para saltar)")
        example = input("> ").strip()
        if example:
            profile.example_posts = [example]

        profile.save()
        print(f"\n✅ Perfil creado para {profile.name or 'el autor'}.")
        return profile
