from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math

from src.helpers import pol_system_cal

@dataclass(slots=True)
class OfficePolicyParams:
    nu: float
    m_G: float
    m_B: float

    beta: float
    delta: float
    kappa: float

    a_unified: float
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

        a_unified= cfg.a_unified,
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
        t: int,
        xi_H: float,
        xi_L: float,

        E_BAU: float,
        E_SCC: float | None = None,

    ) -> Tuple[float]:
        """
        Returns:
          (E_winner, V_G, V_B, E_G, E_B)
        """

        xi_phy = 0.0002046
        xi_old = 0.00008
        scale = xi_old/xi_phy



        E_floor = 1e-12

        E_BAU = E_BAU *10 # scaled to decade

        E_SCC = self._E_SCC_level(xi_H, xi_L, t)

        cl = pol_system_cal(t)


        print(f"E_SCC: ", E_SCC)
        print(f"E_bau: ", E_BAU)

        political_pref_scaling = 28

        # φ_{M,j} for each group from ξ̂_j
        phi_M_H = self._phi_M(xi_H,1)*political_pref_scaling
        phi_M_L =  self._phi_M(xi_L,1)*political_pref_scaling
        
        sum_phi_M = self.p.qH * phi_M_H + self.p.qL * phi_M_L
        # print("#########")
        # print(sum_phi_M)
        # print("#########")


        a_eff = self.p.a_unified
        c = 1.0 / (2.0 * a_eff)
        

        # Pure-office (both parties m≈1 ⇒ platform convergence)
        if math.isclose(self.p.m_G, 1.0, rel_tol=0, abs_tol=1e-12) and math.isclose(self.p.m_B, 1.0, rel_tol=0, abs_tol=1e-12):
            E_star = self._pure_office_E_star(sum_phi_M, self.p.pol_slack, E_BAU)
            E_G = max(E_star, E_floor)
            E_B = max(E_star, E_floor)
            V_G = 0.5
            V_B = 0.5
            E_star = E_star
            return (E_star, V_G, V_B, E_G, E_B)

        
        # Office & Policy Motivation (platforms diverge)
        E_G = self._solve_green_max(self.p.m_G, c, sum_phi_M, E_SCC)
        E_B = self._solve_brown_max(self.p.m_B, c, sum_phi_M, E_BAU)
    

        # Compute expected vote share using uniform shocks:
        # Δw_j = ν[log(E_G) - log(E_B)] + φ_{M,j}(E_G - E_B)
        
        ddd = self.p.nu * (math.log(E_G) - math.log(E_B))
        ppp = (phi_M_L*1) * (E_G - E_B)
        print(f"ddd" ,ddd)
        print(f"ppp",ppp)



        welf_diff_H = self.p.nu * (math.log(E_G) - math.log(E_B)) + (phi_M_H*1) * (E_G - E_B)
        welf_diff_L = self.p.nu * (math.log(E_G) - math.log(E_B)) + (phi_M_L*1) * (E_G - E_B)
        print(f"welf_diff_H", welf_diff_H)
        print(f"welf_diff_L", welf_diff_L)
        # F(z) = (z + a)/(2a) for μ=0, clipped to [0,1]
        F_H = (welf_diff_H + a_eff) / (2.0 * a_eff)
        F_L = (welf_diff_L + a_eff) / (2.0 * a_eff)
        # V_G(E_G, E_B) = sum_j q_j F(Δw_j) with F(z) = (z - (mu-a)) / (2a), clipped to [0,1].
        V_G = self.p.qH * self._clip(F_H) + self.p.qL * self._clip(F_L)
        V_B = 1 - V_G
        print(f"Vote share Green:", V_G)
        print(f"Vote share Brown:", V_B)
        # input("Press Enter to continue...")



        # Winner by expected plurality 
        if V_G > 0.5:
            E_star = E_G
        else:
            E_star = E_B

        self.last_E_G, self.last_E_B = E_G, E_B
        self.last_V_G, self.last_V_B = V_G, V_B
        self.last_E_star = E_star

        return E_star, V_G, V_B, E_G, E_B, E_SCC



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

    def _phi_M(self, xi_hat_j: float, cl:float) -> float:
        # φ_{M,j} = - ξ̂_j / ((1 - β(1-δ)) (1 - βκ))
        xi_hat_j = xi_hat_j *0.15
        beta, delta, kappa = self.p.beta, self.p.delta, self.p.kappa
        denom = (1.0 - beta * (1.0 - delta)) * (1.0 - beta * kappa)
        print(f"phi",(- xi_hat_j  / denom) * cl)
        return (- xi_hat_j  / denom) * cl

    def _solve_green_max(self, m: float, c: float, sum_phi_M: float, E_SCC: float) -> float:
        """
        0 = 2(1-m) E_G^2 - [2(1-m)E_SCC - m c bar_phi_M] E_G - m c nu
        """
        #c = c * 5000
        A = 2.0 * (1.0 - m)
        B = - (2.0 * (1.0 - m) * E_SCC - m * c * sum_phi_M)
        C = - m * c * self.p.nu    

        disc = B * B - 4.0 * A * C
        root = (-B + math.sqrt(disc)) / (2.0 * A)

        return max(root, 0.0)

    def _solve_brown_max(self, m: float, c: float, sum_phi_M: float, E_BAU: float) -> float:
        """
        0 = 2(1-m) E_B^2 - [2(1-m)E_BAU - m c bar_phi_M] E_B + m c nu
        """
        #c = c * 5000
        A = 2.0 * (1.0 - m)
        B = - (2.0 * (1.0 - m) * E_BAU - m * c * sum_phi_M)
        C = + m * c * self.p.nu

        disc = B * B - 4.0 * A * C
        root = (-B + math.sqrt(disc)) / (2.0 * A)
        return max(root, 0.0)

    def _E_SCC_level(self, xi_H:float, xi_L:float, t:int) -> float:
        xi_combined = xi_H*self.p.qH + xi_L * self.p.qL
        level_scale = 0.02                                                       # 2020 calibration
        red_scale = (t*0.005)
        E_SCC = ((self.p.nu / xi_combined)*(1/self.p.beta)-(1-self.p.delta)) * level_scale - red_scale
        return E_SCC

    @staticmethod
    def _clip(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    

