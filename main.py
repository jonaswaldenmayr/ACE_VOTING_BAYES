from __future__ import annotations
import numpy as np

from src.config.app import all_config
from src.voting.OfficeMotiv_model import OfficeMotivPVM, build_pvm_params
from src.voting.simple_test_model import TestVotingModel
from src.ACE.ACE_reduced import ACEModel_RF
from src.plotting import plot_time_series
from src.updating.belief_updating import GroupBeliefUpdating



def main():
    cfg = all_config()

    print("Parameters - core:", cfg.core)
    print("Parameters – ACE :", cfg.ace)
    print("Parameters – vote:", cfg.voting)
    print("Parameters – updating:", cfg.updating)


    #--- ACE -------------------------------------------------------
    ACE = ACEModel_RF(cfg.ace, cfg.core)

    #----- Voting Model --------------------------------------------
    # voting = TestVotingModel(
    #         E_bar=cfg.ace.E_bar,     # this is the E baseline level for the period
    #         kappa_E=cfg.ace.kappa_E,
    #         nu=cfg.ace.nu,
    #         beta=cfg.ace.beta,
    #         delta=cfg.ace.delta,
    #         xi_physical=cfg.ace.xi,
    #         p=cfg.voting,
    #     )
    pvm_params = build_pvm_params(cfg.ace, cfg.voting)  
    voting = OfficeMotivPVM(pvm_params)

    #----- Set Start Beliefs ---------------------------------------
    beliefs = GroupBeliefUpdating.config(cfg.updating)

    #----- Policy fn running the voting model & belief updates --------
    def policy_fn(dmg: float, t) -> float: 

        # Observe current ACE state
        M_t, D_t = ACE.M[t], ACE.D[t]

        #### Update Beliefs
        # 1) Use LAST period's xi (the current prior) for the election
        xi_G, xi_B = beliefs.current_xi()
        # 2) Update NOW with THIS period's (M_t, D_t) for next time's prior
        beliefs.update(M_t=M_t, D_t=D_t)

        # Run election
        E_star = voting.E_star(xi_G, xi_B) 

        #tau, gv, bv = voting.run_election(dmg) # HAND OVER XI'S!!!!!!!!!!!!!!!!!!!!!


        print(f"[Election t={t}] E*={E_star:.3f} | ξ̂_G={xi_G:.3f} | ξ̂_B={xi_B:.3f} | Vote Share: 50 / 50")
        return E_star  
    
    #----- Simulate ---------------------------------------------------
    ACE.simulate(policy_fn)

    #----- Plotting ---------------------------------------------------
    years = np.arange(cfg.core.start_year, cfg.core.start_year + cfg.core.period_len * ACE.T, cfg.core.period_len)
    plot_time_series(ACE.Y, x=years, title="GDP Y", xlabel="Year", ylabel="GDP")
    plot_time_series(ACE.K, x=years, title="Capital K", xlabel="Year", ylabel="Capital")
    plot_time_series(ACE.D, x=years, title="Damages D", xlabel="Year", ylabel="Damage")
    plot_time_series(ACE.M[1:], x=years, title="Atmospheric Carbon M", xlabel="Year", ylabel="Carbon (GtC)")
    plot_time_series(ACE.E, x=years, title="Emissions policy (per year)", xlabel="Year", ylabel="E (per year)")





if __name__ == "__main__":
    main()