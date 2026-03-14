"""
connectors/notion_connector.py — Conector para Notion API

Lee páginas y bases de datos de Notion y las convierte en Notes.

Requisitos:
  pip install notion-client
  NOTION_API_KEY en variables de entorno

Uso:
  connector = NotionConnector()
  notes = connector.fetch_database("tu-database-id")
"""

import os
from typing import Optional
from src.schema import Note, NoteMetadata
from datetime import datetime


def _parse_rich_text(rich_text: list) -> str:
    """Extrae texto plano de un array de rich_text de Notion."""
    return "".join(block.get("plain_text", "") for block in rich_text)


def _parse_property(prop: dict) -> Optional[str]:
    """Extrae el valor de una propiedad Notion según su tipo."""
    ptype = prop.get("type")
    if ptype == "title":
        return _parse_rich_text(prop.get("title", []))
    if ptype == "rich_text":
        return _parse_rich_text(prop.get("rich_text", []))
    if ptype == "select":
        sel = prop.get("select")
        return sel.get("name") if sel else None
    if ptype == "multi_select":
        return ", ".join(s["name"] for s in prop.get("multi_select", []))
    if ptype == "date":
        d = prop.get("date")
        return d.get("start") if d else None
    if ptype == "checkbox":
        return str(prop.get("checkbox", False))
    if ptype == "number":
        return str(prop.get("number", ""))
    if ptype == "url":
        return prop.get("url", "")
    return None


def _blocks_to_markdown(blocks: list) -> str:
    """Convierte bloques de Notion a texto Markdown aproximado."""
    lines = []
    for block in blocks:
        btype = block.get("type", "")
        content = block.get(btype, {})
        rich = content.get("rich_text", [])
        text = _parse_rich_text(rich)

        if btype == "heading_1":
            lines.append(f"# {text}")
        elif btype == "heading_2":
            lines.append(f"## {text}")
        elif btype == "heading_3":
            lines.append(f"### {text}")
        elif btype == "paragraph":
            lines.append(text)
        elif btype == "bulleted_list_item":
            lines.append(f"- {text}")
        elif btype == "numbered_list_item":
            lines.append(f"1. {text}")
        elif btype == "quote":
            lines.append(f"> {text}")
        elif btype == "code":
            lang = content.get("language", "")
            lines.append(f"```{lang}\n{text}\n```")
        elif btype == "divider":
            lines.append("---")
        elif btype == "callout":
            lines.append(f"💡 {text}")
        else:
            if text:
                lines.append(text)

    return "\n\n".join(line for line in lines if line.strip())


class NotionConnector:
    """
    Conecta con Notion API para leer páginas y bases de datos.

    Args:
        api_key: Token de integración de Notion.
                 Si None, usa NOTION_API_KEY del entorno.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._client = None
        key = api_key or os.getenv("NOTION_API_KEY")
        if not key:
            print("⚠️  NOTION_API_KEY no encontrada. El conector Notion está desactivado.")
            return
        self._init_client(key)

    def _init_client(self, key: str):
        try:
            from notion_client import Client
            self._client = Client(auth=key)
            print("✅ Notion API conectada")
        except ImportError:
            print("⚠️  notion-client no instalado. Ejecuta: pip install notion-client")

    def _is_available(self) -> bool:
        return self._client is not None

    def fetch_page(self, page_id: str) -> Optional[Note]:
        """Descarga una página de Notion y la convierte en Note."""
        if not self._is_available():
            return None
        try:
            page = self._client.pages.retrieve(page_id=page_id)
            blocks_resp = self._client.blocks.children.list(block_id=page_id)
            blocks = blocks_resp.get("results", [])

            # Extrae título de las propiedades
            props = page.get("properties", {})
            title = None
            for prop in props.values():
                if prop.get("type") == "title":
                    title = _parse_rich_text(prop.get("title", []))
                    break

            # Extrae fecha de creación/última edición
            date_str = page.get("created_time", "")
            date = None
            if date_str:
                try:
                    date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except ValueError:
                    pass

            content = _blocks_to_markdown(blocks)

            metadata = NoteMetadata(
                source_path=f"notion://{page_id}",
                source_type="notion",
                title=title,
                date=date,
                tags=[],
                raw_frontmatter={"notion_id": page_id},
            )
            return Note(content=content, metadata=metadata)

        except Exception as e:
            print(f"❌ Error leyendo página {page_id}: {e}")
            return None

    def fetch_database(
        self,
        database_id: str,
        filter_params: Optional[dict] = None,
        max_pages: int = 100,
    ) -> list[Note]:
        """
        Descarga todas las páginas de una base de datos Notion.

        Args:
            database_id:   ID de la base de datos
            filter_params: Filtros opcionales (formato Notion API)
            max_pages:     Límite máximo de páginas a descargar
        """
        if not self._is_available():
            return []

        notes: list[Note] = []
        has_more = True
        start_cursor = None
        fetched = 0

        print(f"📚 Descargando Notion database {database_id[:8]}…")

        while has_more and fetched < max_pages:
            query_params: dict = {"database_id": database_id, "page_size": 100}
            if filter_params:
                query_params["filter"] = filter_params
            if start_cursor:
                query_params["start_cursor"] = start_cursor

            try:
                response = self._client.databases.query(**query_params)
            except Exception as e:
                print(f"❌ Error consultando database: {e}")
                break

            for page in response.get("results", []):
                if fetched >= max_pages:
                    break
                note = self.fetch_page(page["id"])
                if note and note.content.strip():
                    notes.append(note)
                fetched += 1

            has_more = response.get("has_more", False)
            start_cursor = response.get("next_cursor")

        print(f"✅ {len(notes)} páginas descargadas desde Notion")
        return notes
