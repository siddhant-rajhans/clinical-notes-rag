from langchain_core.documents import Document

from clinical_rag.ingest.chunker import (
    ChunkConfig,
    chunk_clinical_document,
    chunk_documents,
)


def _make_doc(text: str, **meta: str) -> Document:
    return Document(page_content=text, metadata={"note_id": "test-1", **meta})


class TestClinicalSectionSplitting:
    def test_splits_on_section_headers(self):
        text = (
            "Patient presents with knee pain.\n\n"
            "HISTORY OF PRESENT ILLNESS:\n"
            "The patient is a 55-year-old male with progressive right knee pain "
            "for the past 3 months, worse with weight bearing and stairs.\n\n"
            "PHYSICAL EXAMINATION:\n"
            "Right knee with mild effusion, tenderness along medial joint line, "
            "positive McMurray test. ROM 0 to 120 degrees.\n\n"
            "ASSESSMENT AND PLAN:\n"
            "Degenerative meniscus tear, right knee. Recommend MRI and "
            "conservative management with physical therapy."
        )
        doc = _make_doc(text)
        chunks = chunk_clinical_document(doc, ChunkConfig(chunk_size=2000))

        sections = {c.metadata["section"] for c in chunks}
        assert "HISTORY OF PRESENT ILLNESS" in sections
        assert "PHYSICAL EXAMINATION" in sections
        assert "ASSESSMENT AND PLAN" in sections

    def test_preserves_metadata(self):
        text = "ASSESSMENT:\nDiabetes mellitus type 2, well controlled on metformin."
        doc = _make_doc(text, specialty="Endocrinology")
        chunks = chunk_clinical_document(doc)

        for chunk in chunks:
            assert chunk.metadata["note_id"] == "test-1"
            assert chunk.metadata["specialty"] == "Endocrinology"
            assert "section" in chunk.metadata

    def test_handles_no_sections(self):
        text = "This is a simple clinical note without any section headers. " * 5
        doc = _make_doc(text)
        chunks = chunk_clinical_document(doc)

        assert len(chunks) >= 1
        assert all(c.metadata["section"] == "FULL_NOTE" for c in chunks)

    def test_respects_chunk_size(self):
        text = "HISTORY:\n" + ("Long clinical text. " * 200)
        doc = _make_doc(text)
        config = ChunkConfig(chunk_size=256, chunk_overlap=32)
        chunks = chunk_clinical_document(doc, config)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 512

    def test_skips_short_sections(self):
        text = "PLAN:\nOK.\n\nASSESSMENT:\nDetailed assessment of the patient condition and prognosis."
        doc = _make_doc(text)
        config = ChunkConfig(min_chunk_length=10)
        chunks = chunk_clinical_document(doc, config)

        sections = [c.metadata["section"] for c in chunks]
        assert "ASSESSMENT" in sections


class TestChunkDocuments:
    def test_chunks_multiple_documents(self):
        docs = [
            _make_doc("HISTORY:\nPatient has diabetes and hypertension on medication.", note_id="a"),
            _make_doc("EXAMINATION:\nBP 140/90, heart regular rhythm, lungs clear.", note_id="b"),
        ]
        chunks = chunk_documents(docs, chunk_size=1000)
        note_ids = {c.metadata["note_id"] for c in chunks}
        assert "a" in note_ids
        assert "b" in note_ids
