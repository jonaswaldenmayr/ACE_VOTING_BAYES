from __future__ import annotations
from dataclasses import dataclass
import numpy as np



@dataclass(slots=True)
class GroupBeliefUpdating:
    mu: np.ndarray              # [mu_G, mu_B]
    var: np.ndarray             # [var_G, var_B]  (this is σ_{ξ,t}^2 per group)
    sigma_epsilon: float        # σ_ε (damage shock var)


    # ALL ONLY BOILER PLATE

    @classmethod
    def config(cls, updating_cfg):
        mu_G0 = 0
        mu_B0 = 0
            # ALL ONLY BOILER PLATE


        sigma_G0 = updating_cfg.xi_sigma_G
        sigma_B0 = updating_cfg.xi_sigma_B
        return cls(
            mu=np.array([mu_G0, mu_B0], dtype=float),
            var=np.array([1, 1], dtype=float),
            sigma_epsilon=np.array(1, dtype=float),
    # ALL ONLY BOILER PLATE


        )

    def update(self, M_t: float, D_t: float):
        print("hello")