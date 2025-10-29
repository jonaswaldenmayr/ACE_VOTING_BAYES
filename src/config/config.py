from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class Parameters:
    ########################################################################################################
    ######## CORE ###########################################################################################
    period_len: int = 10
    periods: int = 3
    start_year: int = 2020
    ########################################################################################################
    ######## ACE ###########################################################################################
    # Economy
    Y_init: float = 130.0                                           #in Trillion USD
    A_init: float = 16.09
    K_init: float = 130.0 * 3
    nu: float = 0.07                                                # fossil-energy dependency of production
    kappa: float = 0.30                                             # capital elasticity
    beta: float = 0.84                                              # discount factor
    tech_improvement_rate: float = 0.018  # per year
    prtp = 0.013896                                                 #pure rate of time preference -> currently not used
    @property
    def E_bau(self) -> float:
        return self.BAU_E_CO2_init *(self.molar_mass_CO2/self.molar_mass_C)
    # Climate
    xi: float = 0.00008817                                          #0.0002046
    delta: float = 0.01                                             # carbon decay rate
    ppm_value: float = 427.0
    ppm_to_GtC: float = 2.31
    BAU_E_CO2_init: float = 35.0
    molar_mass_CO2: float = 44.01
    molar_mass_C: float = 12.01
    kappa_E: float = 0.35
    @property
    def M_init(self) -> float:
        return self.ppm_value * self.ppm_to_GtC
    ########################################################################################################
    ######## VOTING #########################################################################################
    num_voters: int = 1000                    
    # Group shares
    qG: float = 0.5
    qB: float = 0.5
    #### Office Motivated Model ########################################
    # Perceived damage multiplier (remove once learning is implemented!!!)
    xi_mult_G: float = 1.3
    xi_mult_B: float = 0.7
    # UNIFORM taste noise
    a_G: float = 1.0
    a_B: float = 1.0
    ########################################################################################################
    ######## BAYES UPDATING ##################################################################################
    # Initial prior means (mu_{ξ,0})
    xi_mu_G0: float = 3.5
    xi_mu_B0: float = 2.5

    # Initial prior standard deviations (σ_{ξ,0})
    xi_sigma_G: float = 0.7        # Green group: open to new info
    xi_sigma_B: float = 0.7

    # Damage shock noise (σ_ε)
    sigma_epsilon_G: float = 0.03
    sigma_epsilon_B: float = 0.03     

def all_config() -> Parameters:
    return Parameters()