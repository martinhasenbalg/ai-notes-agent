"""
parsers/chunker.py — Estrategia de chunking para notas

Divide una Note en Chunks solapados para maximizar el recall semántico.

Estrategia: chunking por párrafos con ventana deslizante.
  1. Divide el contenido en párrafos naturales
  2. Agrupa párrafos en chunks de ~MAX_TOKENS tokens
  3. Añade solapamiento de ~OVERLAP_TOKENS tokens entre chunks consecutivos

Por qué párrafos y no tokens fijos:
  - Respetar límites semánticos naturales mejora la coherencia de cada chunk
  - Evita cortar frases a la mitad
  - Funciona bien para notas de formato mixto (listas + prosa)
"""

from src.schema import Chunk, Note


# Configuración de chunking (ajusta según tus notas)
MAX_CHARS = 800       # ~200 tokens con tokenización típica
OVERLAP_CHARS = 150   # ~40 tokens de solapamiento


def _split_paragraphs(text: str) -> list[str]:
    """Divide texto en párrafos, filtrando vacíos."""
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]


def _build_chunks(paragraphs: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """
    Construye ventanas de texto de ~max_chars caracteres con solapamiento.
    
    Algoritmo:
    - Acumula párrafos hasta llegar al límite
    - Al crear un nuevo chunk, retrocede overlap_chars para solapar contexto
    """
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        # Si el párrafo solo ya supera el límite, lo añadimos como chunk propio
        if para_len > max_chars:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts, current_len = [], 0
            # Dividir párrafo largo en subchunks de max_chars
            for i in range(0, para_len, max_chars - overlap_chars):
                sub = para[i:i + max_chars]
                if sub.strip():
                    chunks.append(sub)
            continue

        if current_len + para_len > max_chars and current_parts:
            chunks.append("\n\n".join(current_parts))
            # Solapamiento: retener los últimos párrafos que quepan en overlap_chars
            overlap_parts: list[str] = []
            overlap_len = 0
            for p in reversed(current_parts):
                if overlap_len + len(p) <= overlap_chars:
                    overlap_parts.insert(0, p)
                    overlap_len += len(p)
                else:
                    break
            current_parts = overlap_parts
            current_len = overlap_len

        current_parts.append(para)
        current_len += para_len

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


class Chunker:
    """
    Divide Notes en Chunks listos para vectorizar.

    Uso:
        chunker = Chunker()
        chunks = chunker.chunk(note)
    """

    def __init__(self, max_chars: int = MAX_CHARS, overlap_chars: int = OVERLAP_CHARS):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, note: Note) -> list[Chunk]:
        """Divide una nota en chunks solapados."""
        paragraphs = _split_paragraphs(note.content)
        raw_chunks = _build_chunks(paragraphs, self.max_chars, self.overlap_chars)

        chunks: list[Chunk] = []
        char_cursor = 0

        for idx, raw in enumerate(raw_chunks):
            # Calcula posición aproximada en el texto original
            start = note.content.find(raw[:40], char_cursor)
            if start == -1:
                start = char_cursor
            end = start + len(raw)
            char_cursor = max(char_cursor, start)

            # Prepende título de la nota para dar contexto al embedding
            contextual_content = (
                f"[Nota: {note.metadata.title}]\n\n{raw}"
                if note.metadata.title
                else raw
            )

            chunks.append(Chunk(
                note_id=note.id,
                content=contextual_content,
                chunk_index=idx,
                char_start=start,
                char_end=end,
                metadata=note.metadata,
            ))

        return chunks

    def chunk_many(self, notes: list[Note]) -> list[Chunk]:
        """Chunktea una lista de notas."""
        all_chunks: list[Chunk] = []
        for note in notes:
            all_chunks.extend(self.chunk(note))
        return all_chunks
