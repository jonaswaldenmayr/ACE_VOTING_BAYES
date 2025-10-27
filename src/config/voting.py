from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class VotingParameters:
    """
    Reduced-form probabilistic voting config.

    Groups:
      - B (Brown): low concern  (ξ_B = ξ_mult_B * ξ_struct)
      - G (Green): high concern (ξ_G = ξ_mult_G * ξ_struct)

    Taste shocks:
      η_i = ε_{iG} - ε_{iB} ~ Logistic(0, s_j) at j ∈ {B,G}.
      Swing weight enters via density at zero: f_η,j(0) = 1 / (4 s_j).

    Notes:
      - num_voters only affects the integer rounding of reported votes.
      - Closed-form τ* does NOT solver, ect
    """
    num_voters: int = 1000                              #doesn't actually affect

    # Group shares
    q_G: float = 0.5
    q_B: float = 0.5

    ### UNDER CONSTRUCTION ###
    # Perceived damage multiplier (remove once learning is implemented!!!)
    xi_mult_G: float = 1.3
    xi_mult_B: float = 0.7
    
    # Logit taste-noise scales s_j; density at zero is f_j(0)=1/(4 s_j)
    taste_scale_G: float = 1.0
    taste_scale_B: float = 1.0

