# from multiprocessing import Pool

import numpy as np
from get_sm_for_sim import get_sm_for_sim, get_smf

def compute_likelihood(obs_smf, sim_smf):
    assert len(obs_smf) == len(sim_smf)

    ln_like = 0
    for i in range(len(obs_smf)):
        if sim_smf[i] < obs_smf["smf"][i]:
            err = np.abs(obs_smf["smf"][i] - obs_smf["smf_low"][i])
        else:
            err = np.abs(obs_smf["smf"][i] - obs_smf["smf_upp"][i])

        ln_like += np.power(
                (obs_smf["smf"][i] - sim_smf[i])/err,
                2
        )
    return ln_like


def single_step(
        params, # The position in parameter space
        sim_data, # The halos
        obs_smf, # HSC SMF
        sim_size,
):
    log_stellar_masses = get_sm_for_sim(sim_data, params[:3], params[3:])

    smf_bins = np.append(obs_smf["logm_0"], obs_smf["logm_1"][-1])

    sim_smf = get_smf(log_stellar_masses, smf_bins, sim_size**3)

    return compute_likelihood(obs_smf, sim_smf)

def single_step_avg(params, sim_data, obs_smf, n):
    print(params)
    chi2 = np.mean([single_step(params, sim_data, obs_smf) for i in range(n)])
    print(chi2)
    return chi2
