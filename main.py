from __future__ import annotations
import numpy as np
from src.helpers import E_SCC_reduction_function

from src.voting.OfficeMotiv_model import OfficeMotivPVM, VotingOutcome
from src.voting.PolicyOfficeMotiv_model import OfficePolicyMotivPVM, build_pvm_params
from src.ACE.ACE_reduced import ACEModel_RF
from src.plotting import plot_time_series, log_election, plot_election_series, plot_time_series_multi
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
    vote_G, vote_B = [], []
    E_G_series, E_B_series = [], []


    # logging containers
    elections: list[dict] = []
    vote_G: list[float] = []
    vote_B: list[float] = []
    E_G_series: list[float] = []
    E_B_series: list[float] = []
    E_SCC_level: list[float] = []


    #----- Set Start Beliefs ---------------------------------------
    beliefs = GroupBeliefUpdating.config(cfg)

    def policy_fn(t:int) -> float:
        xi_H, xi_L = beliefs.current_xi()

        E_star, V_G, V_B, E_G, E_B, E_SCC = voting.policy_and_election(
                t,
                xi_H,
                xi_L,
                E_BAU=cfg.E_bau,  
                #E_SCC=test,           
            )

        log_election(elections, t, E_star, xi_H, xi_L, vote_share=V_G)
        vote_G.append(V_G)
        vote_B.append(V_B)
        E_G_series.append(E_G)
        E_B_series.append(E_B)
        E_SCC_level.append(E_SCC)
            
        print(f"proposed E_G: ", E_G)
        print(f"proposed E_B: ", E_B)
        print(f"[t={t}] -> Election determines E*={E_star:.3f}  | xi_H={xi_H:.6f} xi_L={xi_L:.6f}")
        return E_star

    def learn_fn(t: int, M_t: float, D_t: float):
        beliefs.update_new(
            M_t=M_t,
            D_t=D_t,
            M_pre=cfg.M_init,
        )
        xH, xL = beliefs.current_xi()




    
    #----- Simulate ---------------------------------------------------
    ACE.simulate(policy_fn, learn_fn)

    #--- THESIS GRAPHS ------------------------------------------------


    #----- Plotting ---------------------------------------------------
    years = np.arange(cfg.start_year, cfg.start_year + cfg.period_len * ACE.T, cfg.period_len)
    years_elec = years[:len(elections)]

    #plot_time_series(E_SCC_level, x=years_elec, title="E_SCC", xlabel="Year", ylabel="E_SCC")
    print(cfg.BAU_E_CO2_init)
    print(cfg.E_bau)
    print(cfg.M_init)

    plot_time_series_multi(
        {"Green": vote_G, "Brown": vote_B},
        x=years_elec,
        title="Vote shares (Green vs Brown)",
        xlabel="Year",
        ylabel="Share (0–1)"
        )
    plot_time_series_multi(
        { "Brown": E_B_series, "Green": E_G_series,},
        x=years_elec,
        title="Party platform $E_G$ vs $E_B$",
        xlabel="Year",
        ylabel="Emissions policy",
        y_min=0,
        # colors={"Green": "#59db35", "Brown": "#fc9f47"},  # custom thesis colors
        colors={"Green": "#127FBF", "Brown": "#F9703E"},  # custom thesis colors

    )



    # elections chart (beliefs & E*)
    plot_election_series(elections, years_elec)

    # vote shares & platforms (use the arrays we collected)
    plot_time_series(vote_G, x=years_elec, title="Vote share – Green", xlabel="Year", ylabel="Share (0–1)")
    plot_time_series(vote_B, x=years_elec, title="Vote share – Brown", xlabel="Year", ylabel="Share (0–1)")
    plot_time_series(E_G_series, x=years_elec, title="Party platform E_G", xlabel="Year", ylabel="Emissions policy")
    plot_time_series(E_B_series, x=years_elec, title="Party platform E_B", xlabel="Year", ylabel="Emissions policy")

    # existing macro series
    plot_time_series(ACE.Y, x=years, title="GDP Y", xlabel="Year", ylabel="GDP")
    plot_time_series(ACE.K, x=years, title="Capital K", xlabel="Year", ylabel="Capital")
    plot_time_series(ACE.D, x=years, title="Damages D", xlabel="Year", ylabel="Damage")
    plot_time_series(ACE.M[1:], x=years, title="Atmospheric Carbon M", xlabel="Year", ylabel="Carbon (GtC)")
    plot_time_series(ACE.E, x=years, title="Emissions policy (per year)", xlabel="Year", ylabel="E (per year)")







if __name__ == "__main__":
    main()