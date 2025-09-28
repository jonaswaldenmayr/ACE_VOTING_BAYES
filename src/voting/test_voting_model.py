from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from src.config.voting import VotingParameters


@dataclass(slots=True)
class ElectionResult:
    tax_per_year: float
    green_votes: int
    brown_votes: int


class VotingModel:
    """Very small voting block.

    Interface mirrors your earlier usage: initialize with (E_opti, E_bau), and call
    `run_election(damage)` to get a tuple (tax, green_votes, brown_votes).

    Here, `tax` actually represents the *chosen emissions per YEAR*. ACEModel will
    multiply by `dt` internally to get the per-period stock addition.
    """

    def __init__(self, E_opti: float, E_bau: float, p: VotingParameters):
        self.E_opti = float(E_opti)
        self.E_bau = float(E_bau)
        self.p = p

    def run_election(self, damages_t: float) -> tuple[float, int, int]:
        # Map damages in [0,1] to weight in [0,1]
        w = float(np.clip(damages_t, 0.0, 1.0))

        # Weighted average between BAU (brown) and optimal (green)
        tax = (1.0 - w) * self.E_bau + w * self.E_opti

        # Toy vote shares proportional to w (you can replace with your full model)
        green_votes = int(round(self.p.num_voters * w))
        brown_votes = self.p.num_voters - green_votes

        return tax, green_votes, brown_votes