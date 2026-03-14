"""
parsers/markdown_parser.py — Parser de archivos .md y .txt

Extrae contenido limpio + metadatos (título, fecha, tags) de:
  - Markdown con frontmatter YAML (estilo Obsidian/Jekyll)
  - Markdown sin frontmatter (infiere título del primer H1)
  - Texto plano .txt (infiere título de la primera línea)
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml

from src.schema import Note, NoteMetadata


# ── Helpers de inferencia ────────────────────────────────────────────────────

def _extract_frontmatter(text: str) -> tuple[dict, str]:
    """
    Separa el bloque frontmatter YAML del cuerpo de la nota.
    Devuelve (frontmatter_dict, body_sin_frontmatter).
    """
    pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    match = pattern.match(text)
    if not match:
        return {}, text
    try:
        fm = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        fm = {}
    body = text[match.end():]
    return fm, body


def _infer_title_from_markdown(content: str) -> Optional[str]:
    """Extrae el primer encabezado H1 como título."""
    match = re.search(r"^#{1}\s+(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else None


def _infer_title_from_text(content: str) -> Optional[str]:
    """Usa la primera línea no vacía como título en archivos .txt."""
    for line in content.splitlines():
        line = line.strip()
        if line:
            # Trunca si es demasiado larga (probablemente es párrafo, no título)
            return line[:80] if len(line) <= 80 else None
    return None


def _parse_date(value) -> Optional[datetime]:
    """Intenta convertir distintos formatos de fecha a datetime."""
    if isinstance(value, datetime):
        return value
    # PyYAML parsea fechas ISO como datetime.date, no datetime.datetime
    try:
        from datetime import date as date_type
        if isinstance(value, date_type):
            return datetime(value.year, value.month, value.day)
    except Exception:
        pass
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%B %d, %Y"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


def _parse_tags(value) -> list[str]:
    """Normaliza el campo tags: puede ser lista, string separado por comas, etc."""
    if isinstance(value, list):
        return [str(t).strip().lower() for t in value]
    if isinstance(value, str):
        return [t.strip().lower() for t in value.split(",") if t.strip()]
    return []


def _clean_markdown(content: str) -> str:
    """Limpia marcado Markdown para dejar texto más legible para el embedding."""
    # Elimina bloques de código (pero deja una marca)
    content = re.sub(r"```[\s\S]*?```", "[código]", content)
    # Elimina imágenes
    content = re.sub(r"!\[.*?\]\(.*?\)", "", content)
    # Convierte links en solo el texto
    content = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", content)
    # Elimina negrita/cursiva
    content = re.sub(r"\*{1,3}([^\*]+)\*{1,3}", r"\1", content)
    content = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", content)
    # Elimina líneas de encabezado (mantiene el texto)
    content = re.sub(r"^#{1,6}\s+", "", content, flags=re.MULTILINE)
    # Elimina líneas horizontales
    content = re.sub(r"^[-*_]{3,}$", "", content, flags=re.MULTILINE)
    # Normaliza espacios múltiples y líneas vacías
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


# ── Parser principal ─────────────────────────────────────────────────────────

class MarkdownParser:
    """
    Parser para archivos .md y .txt.

    Uso:
        parser = MarkdownParser()
        note = parser.parse(Path("mi-nota.md"))
    """

    def parse(self, path: Path) -> Note:
        """Parsea un archivo y devuelve un objeto Note."""
        raw = path.read_text(encoding="utf-8", errors="replace")
        suffix = path.suffix.lower()

        if suffix in (".md", ".markdown"):
            return self._parse_markdown(path, raw)
        elif suffix == ".txt":
            return self._parse_text(path, raw)
        else:
            raise ValueError(f"Formato no soportado: {suffix}")

    def _parse_markdown(self, path: Path, raw: str) -> Note:
        frontmatter, body = _extract_frontmatter(raw)
        clean_body = _clean_markdown(body)

        # Metadatos: frontmatter tiene prioridad, luego inferencia
        title = (
            frontmatter.get("title")
            or _infer_title_from_markdown(body)
            or path.stem.replace("-", " ").replace("_", " ").title()
        )
        date = _parse_date(frontmatter.get("date"))
        tags = _parse_tags(frontmatter.get("tags", []))

        metadata = NoteMetadata(
            source_path=str(path.resolve()),
            source_type="markdown",
            title=title,
            date=date,
            tags=tags,
            raw_frontmatter=frontmatter,
        )
        return Note(content=clean_body, metadata=metadata)

    def _parse_text(self, path: Path, raw: str) -> Note:
        clean = raw.strip()
        title = (
            _infer_title_from_text(clean)
            or path.stem.replace("-", " ").replace("_", " ").title()
        )
        metadata = NoteMetadata(
            source_path=str(path.resolve()),
            source_type="text",
            title=title,
            date=None,
            tags=[],
        )
        return Note(content=clean, metadata=metadata)

    def parse_directory(
        self,
        directory: Path,
        recursive: bool = True,
        extensions: tuple[str, ...] = (".md", ".markdown", ".txt"),
    ) -> list[Note]:
        """
        Parsea todos los archivos compatibles en un directorio.
        Retorna lista de Notes, saltando archivos que fallen.
        """
        glob = "**/*" if recursive else "*"
        files = [f for f in directory.glob(glob) if f.suffix.lower() in extensions]
        files.sort()

        notes, errors = [], []
        for f in files:
            try:
                notes.append(self.parse(f))
            except Exception as e:
                errors.append((f, str(e)))

        if errors:
            print(f"⚠️  {len(errors)} archivo(s) con error:")
            for f, err in errors:
                print(f"   {f.name}: {err}")

        print(f"✅ {len(notes)} notas parseadas desde {directory}")
        return notes
