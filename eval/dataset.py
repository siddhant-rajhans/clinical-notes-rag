from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalQuestion:
    question_id: str
    question: str
    expected_answer: str
    relevant_note_ids: list[str]
    specialty: str
    difficulty: str


def load_eval_dataset(path: Path) -> list[EvalQuestion]:
    """Load labeled evaluation questions from JSON."""
    data = json.loads(path.read_text())
    return [
        EvalQuestion(
            question_id=item["question_id"],
            question=item["question"],
            expected_answer=item["expected_answer"],
            relevant_note_ids=item["relevant_note_ids"],
            specialty=item.get("specialty", "General"),
            difficulty=item.get("difficulty", "medium"),
        )
        for item in data["questions"]
    ]
