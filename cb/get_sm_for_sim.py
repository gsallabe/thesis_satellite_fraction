import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology

import smhm_fit

# Given the b_params for the behroozi functional form, and the halos in the sim
# find the SM for each halo
def get_sm_for_sim(sim_data, b_params, s_params, sanity=False):
    assert len(b_params) == 3 and len(s_params) == 2
    b_bounds = [(10, 15), (8, 14), (0.1, 3)]
    for i in range(len(b_params)):
        if not b_bounds[i][0] < b_params[i] < b_bounds[i][1]:
            return np.zeros_like(sim_data["halo_mvir"])

    # default b_params [10**12.52, 10**10.91, 0.45, 0.6, 1.83]
    # from a paper by Alexie
    b_params = np.append(b_params[:2], np.array([0.45, b_params[2], 1.83]))
    for i in range(2):
        b_params[i] = 10**b_params[i]

    # default scatter params - we want ~0.3 at 13Mvir and 0.18 at 15Mvir
    # scatter = -0.06 * Mvir + 1.08

    log_halo_masses = np.log10(sim_data["halo_mvir"])
    min_mvir = np.min(log_halo_masses)
    max_mvir = np.max(log_halo_masses)

    sample_halo_masses = np.linspace(min_mvir, max_mvir, num=20)

    try:
        sample_stellar_masses = smhm_fit.f_shmr(
            sample_halo_masses,
            *b_params
        )
    except Exception as e:
        if e.args[0].startswith("Failure to invert"):
            return np.zeros_like(log_halo_masses)
        raise


    f_mvir_to_sm = scipy.interpolate.interp1d(sample_halo_masses, sample_stellar_masses)

    log_stellar_masses = f_mvir_to_sm(log_halo_masses)
    if not np.all(np.isfinite(log_stellar_masses)):
        print("infinite SM")
        return np.zeros_like(log_stellar_masses)

    # This adds some stochasticity... Ideally we would keep these as a distribution
    # But that is much harder. So we just accept the stochasticity and that the MCMC
    # will take longer to converge

    log_sm_scatter = s_params[0] * log_halo_masses + s_params[1]
    if not np.all(log_sm_scatter > 0):
        print("negative scatter")
        return np.zeros_like(log_stellar_masses)

    log_stellar_masses += np.random.normal(0, log_sm_scatter, size=len(log_sm_scatter))

    if sanity:
        return log_stellar_masses, sample_halo_masses, sample_stellar_masses, f_mvir_to_sm, min_mvir, max_mvir
    else:
        return log_stellar_masses

# Given all the stellar masses, the bins and the sim volume
# Return the SMF
def get_smf(log_stellar_masses, bins, sim_volume):
    counts, edges = np.histogram(log_stellar_masses, bins=bins)
    assert np.all(edges == bins)

    bin_widths = np.diff(bins) # In dex
    assert np.allclose(bin_widths, bin_widths[0])

    number_density = counts / sim_volume / bin_widths[0]

    return number_density


def _sanity_get_sm_for_sim(sim_data, b_params, s_params):
    log_stellar_masses, sample_halo_masses, sample_stellar_masses, f_mvir_to_sm, min_mvir, max_mvir = get_sm_for_sim(sim_data, b_params, s_params, sanity=True)

    # Check that the samples + interpolation look sane
    _, ax = plt.subplots()
    ax.scatter(sample_halo_masses, sample_stellar_masses, s=1)

    interp_x = np.linspace(min_mvir, max_mvir, 1001)
    ax.plot(interp_x, f_mvir_to_sm(interp_x), lw=1)
    ax.set(title="SM-HM relation", xlabel="Mvir", ylabel="SM")

    # Check that the SMF looks sane
    _, ax = plt.subplots()
    counts, edges = np.histogram(log_stellar_masses, bins=30)
    ax.plot(edges[:-1], counts)

    #ax.hist(stellar_masses[stellar_masses > 11.5], bins=30)
    ax.set(yscale="log", ylabel="Counts", xlabel="SM")

    # And just check the distribution
    _, ax = plt.subplots()
    ax.scatter(np.log10(sim_data["halo_mvir"][::100]), log_stellar_masses[::100], s=0.1)
