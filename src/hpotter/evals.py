from pydantic import model_validator, BaseModel
from typing import Literal
import json
from pathlib import Path


class EvalCase(BaseModel):
    question: str
    expected_keywords: list[str] = []
    category: Literal["extractable", "multi_hop", "unanswerable"]
    note: str = ""

    @model_validator(mode="after")
    def check_category(self):
        if self.category in ("extractable", "multi_hop"):
            if not self.expected_keywords:
                raise ValueError(
                    f"{self.category} cannot be empty: {self.expected_keywords}"
                )
        elif self.category == "unanswerable" and self.expected_keywords:
            raise ValueError(f"{self.category} should not have any keywords")
        return self


def load_cases(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EvalCase(**item) for item in raw]
