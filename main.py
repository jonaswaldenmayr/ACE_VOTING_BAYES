from __future__ import annotations
import numpy as np

from src.voting.OfficeMotiv_model import OfficeMotivPVM, build_pvm_params, VotingOutcome
from src.voting.PolicyOfficeMotiv_model import OfficePolicyMotivPVM, build_pvm_params
from src.ACE.ACE_reduced import ACEModel_RF
from src.plotting import plot_time_series, log_election, plot_election_series
from src.updating.belief_updating import GroupBeliefUpdating
from src.config.config import Parameters, all_config




def main():
    cfg = all_config()


    #--- ACE -------------------------------------------------------
    ACE = ACEModel_RF(cfg, cfg)   
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
    pvm_params = build_pvm_params(cfg)  
    voting = OfficePolicyMotivPVM(pvm_params)
    elections = []

    e_prev = cfg.E_bau


    #----- Set Start Beliefs ---------------------------------------
    beliefs = GroupBeliefUpdating.config(cfg)

    #----- Policy fn running the voting model & belief updates --------
    def policy_fn(D_t: float, t) -> float: 
        nonlocal e_prev

        # Observe current ACE state
        M_t, D_t = ACE.M[t], ACE.D[t]

        #### Update Beliefs
        # 1) Use LAST period's xi (the current prior) for the election
        xi_H, xi_L = beliefs.current_xi()
        print(f"Current xi's:", xi_H, xi_L)
        # 2) Update NOW with THIS period's (M_t, D_t) for next time's prior
        beliefs.update(M_t=M_t, D_t=D_t)

        # Run election
        E_star, P_G, P_B, E_G, E_B = voting.policy_and_election(
            xi_H,
            xi_L,
            e_prev
        )
        log_election(elections, t, E_star, xi_H, xi_L, vote_share=0.5)  # adjust vote_share when you have it


        #tau, gv, bv = voting.run_election(dmg) # HAND OVER XI'S!!!!!!!!!!!!!!!!!!!!!


        print(f"[Election t={t}] E*={E_star:.3f} | ξ̂_G={xi_H:.3f} | ξ̂_B={xi_L:.3f} | Vote Share: 50 / 50")
        e_prev = E_star
        return E_star  
    
    #----- Simulate ---------------------------------------------------
    ACE.simulate(policy_fn)

    #----- Plotting ---------------------------------------------------
    years = np.arange(cfg.start_year, cfg.start_year + cfg.period_len * ACE.T, cfg.period_len)
    plot_election_series(elections, years)
    plot_time_series(ACE.Y, x=years, title="GDP Y", xlabel="Year", ylabel="GDP")
    plot_time_series(ACE.K, x=years, title="Capital K", xlabel="Year", ylabel="Capital")
    plot_time_series(ACE.D, x=years, title="Damages D", xlabel="Year", ylabel="Damage")
    plot_time_series(ACE.M[1:], x=years, title="Atmospheric Carbon M", xlabel="Year", ylabel="Carbon (GtC)")
    plot_time_series(ACE.E, x=years, title="Emissions policy (per year)", xlabel="Year", ylabel="E (per year)")






if __name__ == "__main__":
    main()