import matplotlib.pyplot as plt
import numpy as np

from get_sm_for_sim import get_sm_for_sim, get_smf
import likelihood as l

def plot_smfs(sim_data, obs_smf, b_params, s_params, sim_size, x_field):
    log_stellar_masses = get_sm_for_sim(sim_data, b_params, s_params, x_field)

    smf_bins = np.append(obs_smf["logm_0"], obs_smf["logm_1"][-1])
    smf_centers = obs_smf["logm_mean"]

    sim_smf = get_smf(log_stellar_masses, smf_bins, sim_size**3)

    _, ax = plt.subplots()
    ax.errorbar(obs_smf["logm_mean"], obs_smf["smf"], yerr=obs_smf["smf_err"], label="HSC")
    ax.scatter(obs_smf["logm_mean"], obs_smf["smf_low"], color="b")
    ax.scatter(obs_smf["logm_mean"], obs_smf["smf_upp"], color="b")

    ax.set(yscale="log", ylim=(np.min(obs_smf["smf_low"]), np.max(obs_smf["smf_upp"])))

    ax.plot(smf_centers, sim_smf, label="Sim")
    ax.legend()

    print(l.compute_smf_chi2(obs_smf, sim_smf))
    return ax

def plot_scatter(s_params):
    mvir = np.linspace(12, 15)

    _, ax = plt.subplots()
    ax.plot(mvir, s_params[0] * mvir + s_params[1])
    ax.set(
            xlabel="Mvir (Msun)",
            ylabel="M*cen scatter",
    )

    return ax
