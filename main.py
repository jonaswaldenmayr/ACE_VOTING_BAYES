from __future__ import annotations
import numpy as np

from src.config.app import all_config
from src.voting.test_voting_model import VotingModel
from src.ACE.ACE_reduced import ACEModel_RF
from src.plotting import plot_time_series


def main():
    cfg = all_config()

    print("Parameters - core:", cfg.core)
    print("Parameters – ACE :", cfg.ace)
    print("Parameters – vote:", cfg.voting)

    # Build models
    ACE = ACEModel_RF(cfg.ace, cfg.core)
    #beliefs = 
    voting = VotingModel(ACE.controls.E_opti, cfg.ace.E_bau, cfg.voting)

    # policy function calls voting model
    def policy_fn(dmg: float) -> float: 
        tax, gv, bv = voting.run_election(dmg)
        print(f"  [Election] tax={tax:.3f} | green={gv} ({gv/cfg.voting.num_voters:.1%}) | brown={bv} ({bv/cfg.voting.num_voters:.1%})")
        return tax  # per-year emissions number
    
    # Simulate
    ACE.simulate(policy_fn)

    years = np.arange(cfg.core.start_year, cfg.core.start_year + cfg.core.period_len * ACE.T, cfg.core.period_len)
    plot_time_series(ACE.Y, x=years, title="GDP Y", xlabel="Year", ylabel="GDP")
    plot_time_series(ACE.K, x=years, title="Capital K", xlabel="Year", ylabel="Capital")
    plot_time_series(ACE.D, x=years, title="Damages D", xlabel="Year", ylabel="Damage")
    plot_time_series(ACE.M[1:], x=years, title="Atmospheric Carbon M", xlabel="Year", ylabel="Carbon (GtC)")
    plot_time_series(ACE.E, x=years, title="Emissions policy (per year)", xlabel="Year", ylabel="E (per year)")





if __name__ == "__main__":
    main()