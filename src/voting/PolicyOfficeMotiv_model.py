from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math

@dataclass(slots=True)
class OfficePolicyParams:
    nu: float
    m_G: float
    m_B: float

    beta: float
    delta: float
    kappa: float

    a_H: float          # uniform h
    a_L: float

    num_voters: int
    qH: float
    qL: float

    pol_slack: float

def build_pvm_params(cfg) -> OfficePolicyParams:
    return OfficePolicyParams(
        nu=cfg.nu,
        m_G = cfg.m_G,
        m_B = cfg.m_B,

        beta = cfg.beta,
        delta = cfg.delta,
        kappa = cfg.kappa,

        a_H = cfg.a_H,
        a_L= cfg.a_L,

        qH = cfg.qH,
        qL = cfg.qL,
        num_voters= cfg.num_voters,
        pol_slack = cfg.pol_slack,
    )


class OfficePolicyMotivPVM:
    """
    Office & Policy-motivated parties with uniform taste shocks.
    """

    def __init__(self, params: OfficePolicyParams):
        self.p = params

    def policy_and_election(
        self,
        xi_H: float,
        xi_L: float,

        E_BAU: float,

    ) -> Tuple[float]:
        """
        Returns:
          (E_winner, P_G, P_B, E_G, E_B)
        """

        # φ_{M,j} for each group from ξ̂_j
        phi_M_H = self._phi_M(xi_H)
        phi_M_L =  self._phi_M(xi_L)
        
        sum_phi_M = self.p.qH * phi_M_H + self.p.qL * phi_M_L
        

        # Pure-office (both parties m≈1 ⇒ platform convergence)
        if math.isclose(self.p.m_G, 1.0, rel_tol=0, abs_tol=1e-12) and math.isclose(self.p.m_B, 1.0, rel_tol=0, abs_tol=1e-12):
            E_star = self._pure_office_E_star(sum_phi_M, self.p.pol_slack, E_BAU)
            E_G = max(E_star, self.p.E_floor)
            E_B = max(E_star, self.p.E_floor)
            P_G = 0.5
            P_B = 0.5
            E_star = E_star
            return (E_star, P_G, P_B, E_G, E_B)

        
        # Office & Policy Motivation (platforms diverge)
        





    def _pure_office_E_star(self, sum_phi_M: float, pol_slack: float, E_BAU: float) -> float:
        """
        Analogue to the earlier closed form E* = -nu / (sum_j q_j φ_{M,j})
        """
        denom = sum_phi_M
        E_raw = -self.p.nu / denom

        if pol_slack is None or E_BAU is None:
            return 0.0 if E_raw < 0.0 else E_raw
    
        # With slackness: clamp to allowed band around E_before, then clamp at zero
        lower = max(0.0, (1.0 - pol_slack) * E_BAU)
        upper = (1.0 + pol_slack) * E_BAU
        E_clamped = min(max(E_raw, lower), upper)
    
        return 0.0 if E_clamped < 0.0 else E_clamped

    def _phi_M(self, xi_hat_j: float) -> float:
        # φ_{M,j} = - ξ̂_j / ((1 - β(1-δ)) (1 - βκ))
        beta, delta, kappa = self.p.beta, self.p.delta, self.p.kappa
        denom = (1.0 - beta * (1.0 - delta)) * (1.0 - beta * kappa)
        return - xi_hat_j / denom