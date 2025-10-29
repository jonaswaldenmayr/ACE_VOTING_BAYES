from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass(slots=True)
class PVMParams:
    nu: float 
    beta: float
    delta: float
    kappa: float

    xi: float

    qH: float
    qL: float
    a_H: float          # uniform half-widths a_j for η_j ~ U[-a_j, a_j]
    a_L: float
    num_voters: int = 1000


@dataclass(slots=True)
class Group: 
    name: str
    share: float
    xi_hat: float
    a_width: float


class OfficeMotivPVM:
    """
    Probabilistic voting with emissions E as the platform, office motivated parties, uniform taste shocks, forward-looking welfare via φ_M, and closed-form E*.
    """

    def __init__(self, p: PVMParams) -> None:
        self.p = p
        total = p.qH + p.qL
        qH = p.qH / total
        qL = p.qL / total


        # self.groups = [
        #     Group("Green", qH, xi_H, p.a_H),
        #     Group("Brown", qL, xi_L, p.a_L),
        # ]

    def phi_M(self, xi_hat_j: float) -> float:
        # φ_{M,j} = - ξ̂_j / ((1 - β(1-δ)) (1 - βκ))
        denom = (1.0 - self.p.beta * (1 - self.p.delta)) * (1 - self.p.beta * self.p.kappa)
        return - xi_hat_j / denom

    def E_star(
        self,
        xi_H: float,
        xi_L: float,
        E_before: float | None = None,
        pol_slack: float | None = None,
    ) -> float:
        # Base (unconstrained) choice
        denom = self.p.qH * self.phi_M(xi_H) + self.p.qL * self.phi_M(xi_L)
        E_raw = -self.p.nu / denom

        # No political slackness constraint OR no previous E: just clamp at zero
        if pol_slack is None or E_before is None:
            return 0.0 if E_raw < 0.0 else E_raw

        # With slackness: clamp to allowed band around E_before, then clamp at zero
        lower = max(0.0, (1.0 - pol_slack) * E_before)
        upper = (1.0 + pol_slack) * E_before
        E_clamped = min(max(E_raw, lower), upper)

        return 0.0 if E_clamped < 0.0 else E_clamped



def build_pvm_params( cfg) -> PVMParams:
    """
    ace: ACEParameters
    voting: VotingParameters 
    """
    return PVMParams(
        nu = cfg.nu,
        beta = cfg.beta,
        delta = cfg.delta,
        kappa = cfg.kappa,
        xi = cfg.xi,
        qH = cfg.qH, qL = cfg.qL,
        a_H = cfg.a_H, a_L = cfg.a_L,
        num_voters = cfg.num_voters,
    )

@dataclass
class VotingOutcome:
    t: int
    E_star: float
    xi_H: float
    xi_L: float
    vote_share_G: float
    vote_share_B: float