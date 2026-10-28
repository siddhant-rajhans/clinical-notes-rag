from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document

CLINICAL_SECTION_PATTERN = re.compile(
    r"(?:^|\n\n?)"
    r"((?:CHIEF COMPLAINT|HISTORY OF PRESENT ILLNESS|PAST MEDICAL HISTORY|"
    r"PAST SURGICAL HISTORY|MEDICATIONS|ALLERGIES|SOCIAL HISTORY|"
    r"FAMILY HISTORY|REVIEW OF SYSTEMS|PHYSICAL EXAMINATION|"
    r"EXAMINATION|LABORATORY|LAB DATA|IMAGING|RADIOLOGY|"
    r"ASSESSMENT AND PLAN|ASSESSMENT|PLAN|IMPRESSION|"
    r"DIAGNOSIS|DIAGNOSES|PROCEDURE|PROCEDURES|"
    r"OPERATIVE FINDINGS|PREOPERATIVE DIAGNOSIS|POSTOPERATIVE DIAGNOSIS|"
    r"INDICATIONS?|TECHNIQUE|FINDINGS|RECOMMENDATIONS?|"
    r"DISPOSITION|DISCHARGE INSTRUCTIONS|FOLLOW[- ]UP)"
    r"[:\s]*)",
    re.IGNORECASE,
)


@dataclass
class ChunkConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_length: int = 30


def chunk_clinical_document(
    doc: Document,
    config: ChunkConfig | None = None,
) -> list[Document]:
    """Split a clinical note into chunks aware of section boundaries.

    First splits on clinical section headers (HISTORY, EXAMINATION, etc.),
    then applies recursive character splitting within each section.
    """
    config = config or ChunkConfig()
    sections = _split_into_sections(doc.page_content)
    chunks: list[Document] = []

    for section_name, section_text in sections:
        if len(section_text.strip()) < config.min_chunk_length:
            continue

        sub_chunks = _recursive_split(section_text, config.chunk_size, config.chunk_overlap)

        for i, text in enumerate(sub_chunks):
            if len(text.strip()) < config.min_chunk_length:
                continue
            metadata = {
                **doc.metadata,
                "section": section_name,
                "chunk_index": i,
            }
            chunks.append(Document(page_content=text.strip(), metadata=metadata))

    if not chunks and len(doc.page_content.strip()) >= config.min_chunk_length:
        sub_chunks = _recursive_split(doc.page_content, config.chunk_size, config.chunk_overlap)
        for i, text in enumerate(sub_chunks):
            if len(text.strip()) < config.min_chunk_length:
                continue
            metadata = {**doc.metadata, "section": "FULL_NOTE", "chunk_index": i}
            chunks.append(Document(page_content=text.strip(), metadata=metadata))

    return chunks


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    matches = list(CLINICAL_SECTION_PATTERN.finditer(text))

    if not matches:
        return [("FULL_NOTE", text)]

    sections: list[tuple[str, str]] = []

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("PREAMBLE", preamble))

    for i, match in enumerate(matches):
        header = match.group(1).strip().rstrip(":")
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((header.upper(), body))

    return sections


def _recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", ". ", ", ", " "]
    return _split_with_separators(text, separators, chunk_size, overlap)


def _split_with_separators(
    text: str, separators: list[str], chunk_size: int, overlap: int
) -> list[str]:
    if not separators or len(text) <= chunk_size:
        return [text] if text else []

    sep = separators[0]
    parts = text.split(sep)

    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part

        if len(candidate) > chunk_size and current:
            chunks.append(current)
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + sep + part
            else:
                current = part
        else:
            current = candidate

    if current:
        if len(current) > chunk_size and len(separators) > 1:
            chunks.extend(
                _split_with_separators(current, separators[1:], chunk_size, overlap)
            )
        else:
            chunks.append(current)

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """Chunk a list of clinical documents."""
    config = ChunkConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: list[Document] = []
    for doc in documents:
        all_chunks.extend(chunk_clinical_document(doc, config))
    return all_chunks
