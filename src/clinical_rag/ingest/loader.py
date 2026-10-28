from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_mtsamples(csv_path: Path) -> list[Document]:
    """Load MTSamples dataset from CSV.

    Expected columns: sample_id, medical_specialty, sample_name,
    transcription, description, keywords.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["transcription"])

    documents: list[Document] = []
    for _, row in df.iterrows():
        text = str(row["transcription"]).strip()
        if len(text) < 50:
            continue

        metadata = {
            "source": "mtsamples",
            "note_id": f"mt-{row.get('sample_id', row.name)}",
            "specialty": str(row.get("medical_specialty", "Unknown")).strip(),
            "title": str(row.get("sample_name", "")).strip(),
            "description": str(row.get("description", "")).strip(),
        }
        documents.append(Document(page_content=text, metadata=metadata))

    logger.info("Loaded %d documents from MTSamples", len(documents))
    return documents


def load_synthea_fhir(bundle_dir: Path) -> list[Document]:
    """Load clinical notes from Synthea FHIR JSON bundles.

    Parses DocumentReference and DiagnosticReport resources that contain
    free-text clinical narratives.
    """
    documents: list[Document] = []
    bundle_files = list(bundle_dir.glob("*.json"))

    for fpath in bundle_files:
        try:
            bundle = json.loads(fpath.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Skipping invalid FHIR bundle: %s", fpath.name)
            continue

        entries = bundle.get("entry", [])
        patient_name = _extract_patient_name(entries)

        for entry in entries:
            resource = entry.get("resource", {})
            rtype = resource.get("resourceType", "")

            if rtype == "DocumentReference":
                doc = _parse_document_reference(resource, patient_name, fpath.stem)
                if doc:
                    documents.append(doc)
            elif rtype == "DiagnosticReport":
                doc = _parse_diagnostic_report(resource, patient_name, fpath.stem)
                if doc:
                    documents.append(doc)

    logger.info("Loaded %d documents from Synthea FHIR bundles", len(documents))
    return documents


def _extract_patient_name(entries: list[dict]) -> str:
    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Patient":
            names = resource.get("name", [{}])
            given = " ".join(names[0].get("given", []))
            family = names[0].get("family", "")
            return f"{given} {family}".strip()
    return "Unknown"


def _parse_document_reference(
    resource: dict, patient_name: str, bundle_id: str
) -> Document | None:
    content_list = resource.get("content", [])
    if not content_list:
        return None

    attachment = content_list[0].get("attachment", {})
    text = attachment.get("data", "")
    if not text or len(text) < 50:
        return None

    category = "Unknown"
    categories = resource.get("category", [])
    if categories:
        codings = categories[0].get("coding", [{}])
        category = codings[0].get("display", "Unknown")

    return Document(
        page_content=text,
        metadata={
            "source": "synthea",
            "note_id": f"syn-{resource.get('id', bundle_id)}",
            "specialty": category,
            "patient": patient_name,
            "date": resource.get("date", ""),
        },
    )


def _parse_diagnostic_report(
    resource: dict, patient_name: str, bundle_id: str
) -> Document | None:
    text_parts: list[str] = []

    presented = resource.get("presentedForm", [])
    for form in presented:
        if data := form.get("data", ""):
            text_parts.append(data)

    conclusion = resource.get("conclusion", "")
    if conclusion:
        text_parts.append(conclusion)

    text = "\n\n".join(text_parts).strip()
    if len(text) < 50:
        return None

    category = "Diagnostic Report"
    categories = resource.get("category", [])
    if categories:
        codings = categories[0].get("coding", [{}])
        category = codings[0].get("display", category)

    return Document(
        page_content=text,
        metadata={
            "source": "synthea",
            "note_id": f"syn-{resource.get('id', bundle_id)}",
            "specialty": category,
            "patient": patient_name,
            "date": resource.get("effectiveDateTime", ""),
        },
    )


def load_all(data_dir: Path) -> list[Document]:
    """Load documents from all available sources under *data_dir*."""
    documents: list[Document] = []

    mtsamples_csv = data_dir / "mtsamples" / "mtsamples.csv"
    if mtsamples_csv.exists():
        documents.extend(load_mtsamples(mtsamples_csv))

    synthea_dir = data_dir / "synthea"
    if synthea_dir.is_dir():
        documents.extend(load_synthea_fhir(synthea_dir))

    if not documents:
        logger.warning("No documents found under %s", data_dir)

    return documents
