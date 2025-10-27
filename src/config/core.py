from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class CoreSettings:
    period_len: int = 10
    periods: int = 10
    start_year: int = 2020