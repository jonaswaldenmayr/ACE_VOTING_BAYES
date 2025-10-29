from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class VotingParameters:
    """
    
    """
    num_voters: int = 1000                    
    # Group shares
    qH: float = 0.5
    qL: float = 0.5

    #### Office Motivated Model ########################################
    # Perceived damage multiplier (remove once learning is implemented!!!)
    xi_mult_H: float = 1.3
    xi_mult_L: float = 0.7
    
    # UNIFORM taste noise
    a_G: float = 1.0
    a_B: float = 1.0



    #### Office & Policy Motivated Model ###############################


