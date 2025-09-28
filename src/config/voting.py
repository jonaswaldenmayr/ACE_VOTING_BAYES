from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class VotingParameters:
    num_voters: int = 1000
    mean_sensitivity: float = 0.5
    std_sensitivity: float = 0.2
    economic_weight: float = 0.4
    damage_range: tuple[float, float] = (0.2, 1.0)
    noise_scale: float = 0.8
    tax_scale: float = 10.0