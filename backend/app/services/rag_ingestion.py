import json
from pathlib import Path

from app.schemas.rag import RagDocumentChunk, RagSourceDocument


def load_seed_documents(seed_documents_path: str) -> list[RagSourceDocument]:
    path = Path(seed_documents_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path

    if not path.exists():
        raise FileNotFoundError(f"RAG seed document file was not found at {path}.")

    raw_documents = json.loads(path.read_text(encoding="utf-8"))
    return [RagSourceDocument.model_validate(item) for item in raw_documents]


def chunk_source_documents(
    documents: list[RagSourceDocument],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[RagDocumentChunk]:
    chunks: list[RagDocumentChunk] = []
    for document in documents:
        chunks.extend(
            _chunk_source_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks


def _chunk_source_document(
    document: RagSourceDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[RagDocumentChunk]:
    normalized_text = _normalize_text(document.content)
    if len(normalized_text) <= chunk_size:
        return [
            RagDocumentChunk(
                destination_name=document.destination_name,
                travel_style=document.travel_style,
                source_type=document.source_type,
                source_title=document.source_title,
                source_url=document.source_url,
                chunk_index=0,
                content=normalized_text,
            )
        ]

    paragraphs = [part.strip() for part in normalized_text.split("\n\n") if part.strip()]
    if not paragraphs:
        paragraphs = [normalized_text]

    chunks: list[RagDocumentChunk] = []
    current_text = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current_text else f"{current_text}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current_text = candidate
            continue

        if current_text:
            chunks.append(
                _build_chunk(
                    document=document,
                    chunk_index=len(chunks),
                    content=current_text,
                )
            )
            overlap_text = current_text[-chunk_overlap:].strip() if chunk_overlap > 0 else ""
            current_text = (
                f"{overlap_text}\n\n{paragraph}".strip()
                if overlap_text
                else paragraph
            )
        else:
            slices = _slice_long_text(paragraph, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for slice_text in slices[:-1]:
                chunks.append(
                    _build_chunk(
                        document=document,
                        chunk_index=len(chunks),
                        content=slice_text,
                    )
                )
            current_text = slices[-1]

    if current_text:
        chunks.append(
            _build_chunk(
                document=document,
                chunk_index=len(chunks),
                content=current_text,
            )
        )

    return chunks


def _build_chunk(
    *,
    document: RagSourceDocument,
    chunk_index: int,
    content: str,
) -> RagDocumentChunk:
    return RagDocumentChunk(
        destination_name=document.destination_name,
        travel_style=document.travel_style,
        source_type=document.source_type,
        source_title=document.source_title,
        source_url=document.source_url,
        chunk_index=chunk_index,
        content=content.strip(),
    )


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    cleaned_lines: list[str] = []
    blank_streak = 0
    for line in lines:
        if not line:
            blank_streak += 1
            if blank_streak <= 1:
                cleaned_lines.append("")
            continue
        blank_streak = 0
        cleaned_lines.append(" ".join(line.split()))
    return "\n".join(cleaned_lines).strip()


def _slice_long_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    step = max(chunk_size - chunk_overlap, 1)
    slices: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        slices.append(text[start:end].strip())
        if end >= len(text):
            break
        start += step
    return slices
