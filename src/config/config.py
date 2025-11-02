

from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class Parameters:
    ########################################################################################################
    ######## CORE ###########################################################################################
    period_len: int =   10
    periods: int =      30
    start_year: int =   2020
    ########################################################################################################
    ######## ACE ###########################################################################################
    # Economy
    Y_init: float = 130.0                                                       #in Trillion USD
    A_init: float = 16.09
    K_init: float = 130.0
    nu: float = 0.3                                                             # fossil-energy dependency of production
    kappa: float = 0.3                                                         # capital elasticity
    beta: float = 0.84                                                          # discount factor
    tech_improvement_rate: float = 0.014  # per year
    @property
    def E_bau(self) -> float:                                                   # in Gt/C
        return self.BAU_E_CO2_init * (self.molar_mass_C / self.molar_mass_CO2)
    # Climate
    xi: float = 0.0002046                                                      #0.0002046, 00008817
    delta: float = 0.03                                                         # carbon decay rate
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
    num_voters: int = 10000    
    qH: float = 0.20                                                             # Group shares
    qL: float = 0.80
    #### Office Motivated Model ########################################
    # UNIFORM taste noise
    # a_H: float = 1.0
    # a_L: float = 1.0
    pol_slack: float = 0.2                                                      # slackness in the political system permits only ±20% changes per period                
    ########################################################################################################
    #### Office Motivated Model ########################################
    m_G: float = 0.5                                                             # office VS policy weight -> m=1 is purely office motivated
    m_B: float = 0.8
    a_unified: float = 0.1                                                     # for uniform distr
    a_H: float = 1                                                 
    a_L: float = 1

    ########################################################################################################
    ######## BAYES UPDATING ##################################################################################
    # Initial prior means (mu_{ξ,0})
    xi_mu_H0: float = 0.00028
    xi_mu_L0: float = 0.00010

    # Initial prior standard deviations (σ_{ξ,0})
    xi_sigma_H = 28000 * xi                   # Green group: open to new info
    xi_sigma_L = 28000 * xi        

    # Damage shock noise (σ_ε)
    sigma_epsilon_H: float = 40000
    sigma_epsilon_L: float = 40000

def all_config() -> Parameters:
    return Parameters()
