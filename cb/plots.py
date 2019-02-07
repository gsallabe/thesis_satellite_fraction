import matplotlib.pyplot as plt
import numpy as np

from get_sm_for_sim import get_smf

# TODO: fix these colors
def plot_smfs(obs_smf, log_stellar_masses, sim_size):
    smf_bins = np.append(obs_smf["logm_0"], obs_smf["logm_1"][-1])
    smf_centers = obs_smf["logm_mean"]

    sim_smf = get_smf(log_stellar_masses, smf_bins, sim_size**3)

    fig, ax = plt.subplots()
    ax.plot(smf_centers, sim_smf, label="Model", marker=".", zorder=100)

    yerr = np.vstack((obs_smf["smf"] - obs_smf["smf_low"], obs_smf["smf_upp"] - obs_smf["smf"]))
    ax.errorbar(obs_smf["logm_mean"], obs_smf["smf"], yerr=yerr, label="HSC")

    ax.set(yscale="log", ylabel=r"d$N$/d\,log $M_{\ast}$ [${\rm Mpc^{-3} dex^{-1}}$]", xlabel=r"${\rm log}(M_{\ast} / M_{\odot})$")

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
