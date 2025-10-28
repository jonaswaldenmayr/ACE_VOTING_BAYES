from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class BeliefParameters:

    # Initial prior means (mu_{ξ,0})
    xi_mu_G0: float = 0.2
    xi_mu_B0: float = 0.6

    # Initial prior standard deviations (σ_{ξ,0})
    xi_sigma_G: float = 0.05        # Green group: open to new info
    xi_sigma_B: float = 0.05

    # Damage shock noise (σ_ε)
    sigma_epsilon_G: float = 0.03
    sigma_epsilon_B: float = 0.03     