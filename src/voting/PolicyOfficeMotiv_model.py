from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math

@dataclass(slots=True)
class OfficePolicyParams:
    nu: float