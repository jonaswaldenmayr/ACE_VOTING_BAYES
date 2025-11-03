from __future__ import annotations
import numpy as np
from src.helpers import E_SCC_reduction_function

from src.voting.OfficeMotiv_model import OfficeMotivPVM, VotingOutcome
from src.voting.PolicyOfficeMotiv_model import OfficePolicyMotivPVM, build_pvm_params
from src.ACE.ACE_reduced import ACEModel_RF
from src.plotting import plot_all_beliefs_and_damages, plot_beliefs_and_damages, plot_dual_axis, plot_dual_axis_beliefs_M, plot_time_series, log_election, plot_election_series, plot_time_series_multi
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

    

    #----- Plotting ---------------------------------------------------
    years = np.arange(cfg.start_year, cfg.start_year + cfg.period_len * ACE.T, cfg.period_len)
    years_elec = years[:len(elections)]
    plot_all_beliefs_and_damages(
        years=years,
        elections=elections,
        damages=ACE.D,
        title="Beliefs and Damages",
        save_pdf=True,
    )
    plot_dual_axis(
        x=years,
        y_left=ACE.M[1:],          # atmospheric carbon
        y_right=ACE.D,             # damages
        label_left="Carbon stock (GtC)",
        label_right="Damages (fraction of GDP)",
        title="Carbon Stock vs. Damages",
        color_left="#127FBF",
        color_right="#F9703E",
    )

    plot_time_series_multi(
        {"Green": vote_G, "Brown": vote_B},
        x=years_elec,
        title="Party Vote shares (Green vs. Brown)",
        xlabel="Year",
        ylabel="Vote Share",
        y_line = 0.5,
        y_line_width=0.8,
        y_line_color="black",
        y_line_style="--",
        )
    
    plot_time_series_multi(
        { "$E_B^*$": E_B_series, "$E_G^*$": E_G_series,},
        x=years_elec,
        title="Party platforms $E_G^*$ and $E_B^*$",
        xlabel="Year",
        # ylabel="Emission policy",
        y_min=0,
        # colors={"Green": "#59db35", "Brown": "#fc9f47"},  # custom thesis colors
        colors={"$E_G^*$": "#127FBF", "$E_B^*$": "#F9703E"},  # custom thesis colors
    )


    # Extract beliefs and E* from the elections log
    xi_H_series   = [e["xi_H"]   for e in elections]
    xi_L_series   = [e["xi_L"]   for e in elections]
    E_star_series = [e["E_star"] for e in elections]

    plot_time_series_multi(
        {
            r"$\hat{\xi}_H$": xi_H_series,
            r"$\hat{\xi}_L$": xi_L_series,
        },
        x=years_elec,
        title="Climate sensitivity beliefs",
        xlabel="Year",
        colors={r"$\hat{\xi}_H$": "#127FBF", r"$\hat{\xi}_L$": "#F9703E"},
        legend_loc="best",
        y_line = 0.0002046,
        y_line_width=0.8,
        y_line_color="black",
        y_line_style="--",
    )

    plot_time_series_multi(
        {
            r"$E^*$": E_star_series,
        },
        x=years_elec,
        title=r"Election winning Policy $E_P^*$",
        xlabel="Year",
        legend_loc="best",
    )

    plot_time_series(ACE.E, x=years, title="Election winning Policy $E^*$", xlabel="Year", 
    #ylabel="$E^*$"
    )





    plot_beliefs_and_damages(
        years=years,
        elections=elections,
        damages=ACE.D,
        title="Climate Sensitivity Beliefs and Damages",
        save_pdf=True,
    )

    # formatted = ", ".join(f"{y:,.2f}" for y in ACE.Y)
    # print(f"[{formatted}]")

    # formatted = ", ".join(f"{y:,.2f}" for y in ACE.M)
    # print(f"[{formatted}]")



    plot_election_series(elections, years_elec)

    # # vote shares & platforms (use the arrays we collected)
    # plot_time_series(vote_G, x=years_elec, title="Vote share – Green", xlabel="Year", ylabel="Share (0–1)")
    # plot_time_series(vote_B, x=years_elec, title="Vote share – Brown", xlabel="Year", ylabel="Share (0–1)")
    # plot_time_series(E_G_series, x=years_elec, title="Party platform E_G", xlabel="Year", ylabel="Emissions policy")
    # plot_time_series(E_B_series, x=years_elec, title="Party platform E_B", xlabel="Year", ylabel="Emissions policy")

    # existing macro series
    # plot_time_series(ACE.Y, x=years, title="GDP Y", xlabel="Year", ylabel="GDP")
    # plot_time_series(ACE.K, x=years, title="Capital K", xlabel="Year", ylabel="Capital")
    # plot_time_series(ACE.D, x=years, title="Damages D", xlabel="Year", ylabel="Damage")
    # plot_time_series(ACE.M[1:], x=years, title="Atmospheric Carbon M", xlabel="Year", ylabel="Carbon (GtC)")







if __name__ == "__main__":
    main()