# from __future__ import annotations
# from dataclasses import dataclass
# import numpy as np
# from typing import Callable

# from src.config.voting import VotingParameters


# @dataclass(slots=True)
# class Group:
#     name: str
#     share: float                                                # q_j
#     xi: float                                                   # perceived damage weight ξ_j
#     taste_scale: float    # s_j for logistic taste shocks

# class TestVotingModel:
#     """
#     Reduced-form probabilistic voting with CLOSED-FORM τ*.
#     """

#     def __init__(
#         self,
#         E_bar: float,                                           # E before taxation
#         kappa_E: float,
#         nu: float, 
#         beta: float,
#         delta: float,
#         xi_physical: float,
#         p: VotingParameters,
#     ):
#         self.E_bar = float(E_bar)
#         self.kappa_E = float(kappa_E)
#         self.nu = float(nu)
#         self.beta = float(beta)
#         self.delta = float(delta)
#         self.xi_physical = float(xi_physical)                   # if crashes because of xi=0, use: float(max(xi_struct, 1e-12))
#         self.p = p

#         # shares
#         total_q = (p.qG + p.qB)
#         qG = p.qG / total_q
#         qB = p.qB / total_q

#         # perceived damages from multiplyers (to be updated later)
#         xi_G = p.xi_mult_G * self.xi_physical
#         xi_B = p.xi_mult_B * self.xi_physical

#         self.groups = [
#             Group(name="Green", share=qG, xi=xi_G, taste_scale=p.taste_scale_G),
#             Group(name="Brown", share=qB, xi=xi_B, taste_scale=p.taste_scale_B),
#         ]

#         def f_eta_zero(s: float) -> float:
#             return 1.0 / (4.0 * s)

#         self._f0_G = f_eta_zero(self.groups[0].taste_scale) if self.groups[0].name == "Green" else print(IndexError)        
#         self._f0_B = f_eta_zero(self.groups[1].taste_scale) if self.groups[1].name == "Brown" else print(IndexError)

#         self.W = sum(g.share * f_eta_zero(g.taste_scale) for g in self.groups)
#         self.X = sum(g.share * f_eta_zero(g.taste_scale) * g.xi for g in self.groups)

#         # discounted stock multiplier
#         self.stock_mult = 1.0 / (1.0 - self.beta * (1.0 - self.delta))

#     def tau_star(self) -> float:
#         """
#         Closed form τ*
#         """
#         num = (self.E_bar * self.stock_mult) * self.X
#         den = self.nu * self.W
#         return (1 / self.kappa_E) * np.log(num / den)

#     def _vote_shares_at_tau(self, tau: float) -> tuple[int, int]:
#         """
#         Informative split using a tiny platform wedge:
#           τ_G = τ + ε, τ_B = τ - ε  ⇒ ΔU ≈ 2ε * [ν dlogE/dτ - ξ_j dM*/dτ], logit choice.
#         """
#         eps = 1e-4
#         n = self.p.num_voters
#         # dlogE/dτ = -κ_E (const), E(τ)=E^- e^{-κ_E τ}
#         dlogE = -self.kappa_E
#         E_tau = self.E_bar * np.exp(-self.kappa_E * tau)
#         dMstar = self.stock_mult * E_tau * dlogE  # = stock_mult * E * (-κ_E)
#         green_votes = 0.0
#         for g in self.groups:
#             slope = self.nu * dlogE - g.xi * dMstar
#             dU = 2.0 * eps * slope
#             s = g.taste_scale
#             p_green = 1.0 / (1.0 + np.exp(-(dU / s)))  # logit CDF
#             green_votes += g.share * n * p_green
#         gv = int(round(green_votes))
#         bv = n - gv
#         return gv, bv
        
#     def run_election(self, t: float = 0) -> tuple[float, int, int]:
#         tau = self.tau_star()
#         gv, bv = self._vote_shares_at_tau(tau)
#         return tau, gv, bv
