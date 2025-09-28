from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class LearningParameters:
    ddd: float = 1.0