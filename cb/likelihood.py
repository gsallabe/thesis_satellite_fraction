# from multiprocessing import Pool

import numpy as np
from get_sm_for_sim import get_sm_for_sim, get_smf
import clustering as c

def compute_smf_chi2(obs_smf, sim_smf):
    assert len(obs_smf) == len(sim_smf)

    chi2 = 0
    for i in range(len(obs_smf)):
        if sim_smf[i] < obs_smf["smf"][i]:
            err = np.abs(obs_smf["smf"][i] - obs_smf["smf_low"][i])
        else:
            err = np.abs(obs_smf["smf"][i] - obs_smf["smf_upp"][i])

        chi2 += np.power(
                (obs_smf["smf"][i] - sim_smf[i]) / err,
                2
        )
    return chi2

def compute_chi2(
        params,         # The position in parameter space
        sim_data,       # The halos
        obs_smf,        # HSC SMF
        obs_clust,      # HSC clustering
        sim_size,       # Length of each side in the sim
        cen_sat_div,
        x_field,
):
    log_stellar_masses = get_sm_for_sim(sim_data, params[:5], params[5:], x_field)

    sim_clust = np.array(
            [c.compute_sim_clustering(sim_data, sim_size, log_stellar_masses, div) for div in cen_sat_div],
            dtype=[("clustering", np.float64), ("err", np.float64)])

    clust_delta = np.abs(sim_clust["clustering"] - obs_clust["clustering"])
    clust_delta_err = np.sqrt(sim_clust["err"]**2 + obs_clust["err"]**2)
    clust_chi2 = np.sum(np.power(clust_delta / clust_delta_err, 2))

    # We don't need the error on the sim_smf because it will be a lot smaller than on the observations
    # We didn't really need it on the clustering either but it doesn't cost anything...
    sim_smf = get_smf(
            log_stellar_masses,
            np.append(obs_smf["logm_0"], obs_smf["logm_1"][-1]),
            sim_size**3,
    )

    return compute_smf_chi2(obs_smf, sim_smf) + clust_chi2

def compute_chi2_n(params, sim_data, obs_smf, obs_clust, sim_size, cen_sat_div, x_field, n, extra_params=None):
    if extra_params is not None:
        params = _sub_extra_params(params, extra_params)

    chi2 = []
    for _ in range(n):
        chi2.append(compute_chi2(
            params, sim_data, obs_smf, obs_clust, sim_size, cen_sat_div, x_field,
        ))
        # We don't need to repro terrible points
        if np.mean(chi2) > 5:
            break

    chi2 = np.mean(chi2)
    print(params, chi2)
    return chi2


def _sub_extra_params(params, extra_params):
    # We can't mutate the params
    if type(params) is list:
        p = np.array(params)
    else:
        p = np.copy(params)

    for loc, val in extra_params.items():
        p = np.insert(p, loc, val)

    return p
