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

    
    # UNIFORM taste noise
    a_G: float = 1.0
    a_B: float = 1.0



    #### Office & Policy Motivated Model ###############################


