from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict
import numpy as np

from src.config.core import CoreSettings
from src.config.ace import ACEParameters
#from src.learning.belief_updating import BeliefUpdating


@dataclass(slots=True)
class ShadowValues:
    phi_k: float
    phi_M: float
    phi_M_BAU: float

@dataclass(slots=True)
class Controls:
    x_opti: float
    E_opti: float


class ACEModel_RF:
    """
    Reduced Form ACE model

     Conventions
    -----------
    - One model step is `dt = core.period_len` years (10y)
    - Policy chooses `E_year` (per-year emissions) each period. -> make that decade!!!
    - Internally we convert to per-period where needed:
        * Output Y_t is a *flow over the period*: Y_t = A_t K_t^κ E_year^ν * dt
        * Carbon stock: M_{t+1} = (1-δ)^{dt} M_t + E_year * dt
        * Log-capital: k_{t+1} = ln A_t + κ k_t + ν ln E_year + ln dt - ξ M_t + ln(1-x)
    - Arrays:
        * Stocks A, M, k are length T+1 (store t=0 initial and terminal t=T)
        * Flows Y, C, E, D, K are length T
    """

    def __init__(self, p: ACEParameters, core: CoreSettings):
        self.p = p
        self.core = core
        self.dt = int(core.period_len)                                                      # years per period
        self.T = int(core.periods)                                                          # periods

        # Series
        self.Y = np.zeros(self.T)
        self.K = np.zeros(self.T)
        self.A = np.zeros(self.T + 1)
        self.C = np.zeros(self.T)
        self.k = np.zeros(self.T + 1)            # log capital
        self.D = np.zeros(self.T)
        self.M = np.zeros(self.T + 1)
        self.E = np.zeros(self.T)

        # Init state at t=0
        self.M[0] = p.M_init
        self.A[0] = p.A_init
        self.K[0] = p.K_init
        self.k[0] = np.log(self.K[0])
        self.D[0] = 1.0 - np.exp(-p.xi * self.M[0])
        self.Y[0] = p.Y_init

        # Precompute shadows / controls -> reduced form
        self.shadows = self._compute_shadow_values()
        self.controls = self._compute_controls(self.shadows)

    def _compute_shadow_values(self) -> ShadowValues:
        p = self.p
        phi_k = p.kappa / (1.0 - (p.beta * p.kappa))
        phi_M = -(p.xi * (1.0 + (p.beta * phi_k))) / (1.0 - (p.beta * (1.0 - p.delta)))
        phi_M_BAU = -(p.nu * (1.0 + p.beta * phi_k)) / (p.beta * p.E_bau)                   # ass. 35 Gtco2/y (CHECK!) -> -0.00318 as BAU shadow value of c
        return ShadowValues(phi_k=phi_k, phi_M=phi_M, phi_M_BAU=phi_M_BAU)

    def _compute_controls(self, sv: ShadowValues) -> Controls:
        p = self.p
        x_opti = 1.0 / (1.0 + p.beta * sv.phi_k)
        E_opti = (p.nu * (1.0 + p.beta * sv.phi_k)) / (p.beta * (-sv.phi_M))
        return Controls(x_opti=x_opti, E_opti=E_opti)

    def _output(self, A: float, K: float, E: float) -> float:
        return A * (K ** self.p.kappa) * (E ** self.p.nu)

    def step(self, t: int, tax: float) -> dict:
        p, dt = self.p, self.dt
        x = self.controls.x_opti

        # tax into emissions / year
        E_year = float(p.E_bau * np.exp(-p.kappa * tax))
        E_step = E_year * dt

        D_t = 1.0 - np.exp(-p.xi * self.M[t])
        K_t = np.exp(self.k[t])
        Y_t = self._output(self.A[t], K_t, E_year) * dt
        C_t = x * Y_t * (1.0 - D_t)

        # eq of motions
        self.k[t + 1] = (
            np.log(self.A[t])
            + p.kappa * self.k[t]
            + p.nu * np.log(max(E_year, 1e-12))
            + np.log(dt)
            - p.xi * self.M[t]
            + np.log(1.0 - x)
        )
        self.M[t + 1] = (1.0 - p.delta) ** dt * self.M[t] + E_step
        self.A[t + 1] = self.A[t] * (1.0 + p.tech_improvement_rate) ** dt

        # Save
        self.E[t] = E_year             # store per-year policy for readability
        self.D[t] = D_t
        self.Y[t] = Y_t
        self.C[t] = C_t
        self.K[t] = K_t

        print(f"Y",Y_t)
        print(f"C",C_t)
        print(f"K",K_t)
        print(f"D",D_t)

        return {"Y": Y_t, "C": C_t, "K": K_t, "E_year": E_year, "D": D_t, "A": self.A[t], "M": self.M[t]}

    def simulate(self, policy_fn):
        for t in range(self.T):
            tax = policy_fn(self.D[t], t)
            self.step(t, tax=tax)
        return self
