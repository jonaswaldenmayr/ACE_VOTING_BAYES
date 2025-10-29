from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class Parameters:
    ########################################################################################################
    ######## CORE ###########################################################################################
    period_len: int = 10
    periods: int = 30
    start_year: int = 2020
    ########################################################################################################
    ######## ACE ###########################################################################################
    # Economy
    Y_init: float = 130.0                                           #in Trillion USD
    A_init: float = 16.09
    K_init: float = 130.0 * 3
    E_before: float = 20                                             # E level in t-1   
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
    @property
    def M_init(self) -> float:
        return self.ppm_value * self.ppm_to_GtC
    ########################################################################################################
    ######## VOTING #########################################################################################
    num_voters: int = 1000    
    pol_slackness: float = 0.2                                      # slackness in the political system permits only ±20% changes per period                
    # Group shares
    qH: float = 0.5
    qL: float = 0.5
    #### Office Motivated Model ########################################
    # Perceived damage multiplier (remove once learning is implemented!!!)
    xi_mult_H: float = 1.3 * xi
    xi_mult_L: float = 0.7 * xi
    # UNIFORM taste noise
    a_H: float = 1.0
    a_L: float = 1.0
    ########################################################################################################
    ######## BAYES UPDATING ##################################################################################
    # Initial prior means (mu_{ξ,0})
    xi_mu_H0: float = 0.1
    xi_mu_L0: float = 0.05

    # Initial prior standard deviations (σ_{ξ,0})
    xi_sigma_H = 0.9 * xi                   # Green group: open to new info
    xi_sigma_L = 0.9 * xi        

    # Damage shock noise (σ_ε)
    sigma_epsilon_H: float = 0.3
    sigma_epsilon_L: float = 0.5    

def all_config() -> Parameters:
    return Parameters()