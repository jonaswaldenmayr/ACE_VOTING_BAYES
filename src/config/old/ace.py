from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class ACEParameters:
    # Economy
    Y_init: float = 130.0                                           #in Trillion USD
    A_init: float = 16.09
    K_init: float = 130.0 * 3
    # Technology / preferences
    nu: float = 0.07                                                # fossil-energy dependency of production
    kappa: float = 0.30                                             # capital elasticity
    beta: float = 0.84                                              # discount factor
    tech_improvement_rate: float = 0.018  # per year
    prtp = 0.013896                                                 #pure rate of time preference -> currently not used
    # Climate
    xi: float = 0.0002046
    delta: float = 0.01                                             # carbon decay rate
    ppm_value: float = 427.0
    ppm_to_GtC: float = 2.31
    BAU_E_CO2_init: float = 35.0
    molar_mass_CO2: float = 44.01
    molar_mass_C: float = 12.01

    # tax→emissions elasticity in E(τ)=E^- * exp(-kappa_E * τ)
    kappa_E: float = 0.35


    @property
    def M_init(self) -> float:
        return self.ppm_value * self.ppm_to_GtC

    @property
    def E_bau(self) -> float:
        return self.BAU_E_CO2_init *(self.molar_mass_CO2/self.molar_mass_C)

    @property
    def E_bar(self) -> float:
        return self.E_bau