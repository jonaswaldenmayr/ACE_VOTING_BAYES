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

    qG: float
    qB: float
    xi_mult_G: float
    xi_mult_B: float
    a_G: float          # uniform half-widths a_j for η_j ~ U[-a_j, a_j]
    a_B: float
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
        total = p.qG + p.qB
        qG = p.qG / total
        qB = p.qB / total

        xi_G = p.xi * p.xi_mult_G
        xi_B = p.xi * p.xi_mult_B

        self.groups = [
            Group("Green", qG, xi_G, p.a_G),
            Group("Brown", qB, xi_B, p.a_B),
        ]

    def phi_M(self, xi_hat_j: float) -> float:
        # φ_{M,j} = - ξ̂_j / ((1 - β(1-δ)) (1 - βκ))
        denom = (1.0 - self.p.beta * (1 - self.p.delta)) * (1 - self.p.beta * self.p.kappa)
        return - xi_hat_j / denom

    def E_star(self, xi_G: float, xi_B: float) -> float:
        denom = self.p.qG * self.phi_M(xi_G) + self.p.qB * self.phi_M(xi_B)
        E_star = -self.p.nu / denom

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
        qG = voting.qG, qB = voting.qB,
        xi_mult_G = voting.xi_mult_G, xi_mult_B = voting.xi_mult_B,
        a_G = voting.a_G, a_B = voting.a_B,
        num_voters = voting.num_voters,
    )