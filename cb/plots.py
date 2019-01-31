import matplotlib.pyplot as plt
import numpy as np

from get_sm_for_sim import get_smf

# TODO: fix these colors
def plot_smfs(obs_smf, log_stellar_masses, sim_size):
    smf_bins = np.append(obs_smf["logm_0"], obs_smf["logm_1"][-1])
    smf_centers = obs_smf["logm_mean"]

    sim_smf = get_smf(log_stellar_masses, smf_bins, sim_size**3)

    fig, ax = plt.subplots()
    l = ax.errorbar(obs_smf["logm_mean"], obs_smf["smf"], yerr=obs_smf["smf_err"], label="HSC")
    ax.scatter(obs_smf["logm_mean"], obs_smf["smf_low"], color=l[0].get_color())
    ax.scatter(obs_smf["logm_mean"], obs_smf["smf_upp"], color=l[0].get_color())

    ax.set(yscale="log")#, ylim=(np.min(obs_smf["smf_low"]), np.max(obs_smf["smf_upp"])))

    ax.plot(smf_centers, sim_smf, label="Sim", marker=".")
    ax.legend()

    return fig, ax

def plot_scatter(s_params):
    mvir = np.linspace(12, 15)

    _, ax = plt.subplots()
    ax.plot(mvir, s_params[0] * mvir + s_params[1])
    ax.set(
            xlabel="Mvir (Msun)",
            ylabel="M*cen scatter",
    )

    return ax
