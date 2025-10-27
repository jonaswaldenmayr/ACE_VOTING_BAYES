from __future__ import annotations
from dataclasses import dataclass
import numpy as np



@dataclass(slots=True)
class GroupBeliefUpdating:
    mu: np.ndarray              # [mu_G, mu_B]
    var: np.ndarray             # [var_G, var_B]  (this is σ_{ξ,t}^2 per group)
    sigma_epsylon: float        # σ_ε (damage shock var)

    @classmethod
    def config(updating_cfg):

        sigma_G0 = updating_cfg.xi_sigma_G
        sigma_B0 = updating_cfg.xi_sigma_B