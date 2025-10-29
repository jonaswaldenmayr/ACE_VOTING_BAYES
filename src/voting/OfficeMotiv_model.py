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
    xi_mult_H: float
    xi_mult_L: float
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

        xi_H = p.xi * p.xi_mult_H
        xi_L = p.xi * p.xi_mult_L

        self.groups = [
            Group("Green", qH, xi_H, p.a_H),
            Group("Brown", qL, xi_L, p.a_L),
        ]

    def phi_M(self, xi_hat_j: float) -> float:
        # φ_{M,j} = - ξ̂_j / ((1 - β(1-δ)) (1 - βκ))
        denom = (1.0 - self.p.beta * (1 - self.p.delta)) * (1 - self.p.beta * self.p.kappa)
        return - xi_hat_j / denom

    def E_star(
        self,
        xi_H: float,
        xi_L: float,
        E_before: float| None = None,                         # E in t-1
        pol_slackness: float | None = None,     # slackness in the political system permits only ±20% changes per period
    ) -> float:

        if pol_slackness is None:
            denom = self.p.qH * self.phi_M(xi_H) + self.p.qL * self.phi_M(xi_L)
            E_star = -self.p.nu / denom
            if E_star < 0:
                return 0.0

        if E_before is None:
            denom = self.p.qH * self.phi_M(xi_H) + self.p.qL * self.phi_M(xi_L)
            E_star = -self.p.nu / denom
        else:
            denom = self.p.qH * self.phi_M(xi_H) + self.p.qL * self.phi_M(xi_L)
            E_raw = -self.p.nu / denom

            lowerBound = max(0.0, (1.0 - pol_slackness) * E_before)
            upperBound = (1.0 + pol_slackness) * E_before
            E_star = min(max(E_raw, lowerBound), upperBound)


        

        if E_star < 0:
            return 0.0

        return E_star




def build_pvm_params(ace, voting) -> PVMParams:
    """
    ace: ACEParameters
    voting: VotingParameters 
    """
    return PVMParams(
        nu = ace.nu,
        beta = ace.beta,
        delta = ace.delta,
        kappa = ace.kappa,
        xi = ace.xi,
        qH = voting.qH, qL = voting.qL,
        xi_mult_H = voting.xi_mult_H, xi_mult_L = voting.xi_mult_L,
        a_H = voting.a_H, a_L = voting.a_L,
        num_voters = voting.num_voters,
    )

@dataclass
class VotingOutcome:
    t: int
    E_star: float
    xi_H: float
    xi_L: float
    vote_share_G: float
    vote_share_B: float