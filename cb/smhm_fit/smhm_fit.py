import numpy as np

# The functional form from https://arxiv.org/pdf/1103.2077.pdf
# This is the fitting function
# f_shmr finds SM given HM. As the inverse, this find HM given SM
def f_shmr_inverse(log_stellar_masses, m1, sm0, beta, delta, gamma):
    if np.max(log_stellar_masses) > 1e6:
        raise Exception("You are probably not passing log masses!")

    stellar_masses = np.power(10, log_stellar_masses)

    usm = stellar_masses / sm0 # unitless stellar mass is sm / characteristic mass
    log_halo_mass = np.log10(m1) + (beta * np.log10(usm)) + ((np.power(usm, delta)) / (1 + np.power(usm, -gamma))) - 0.5
    return log_halo_mass

# Given halo masses, finds stellar masses
def f_shmr(log_halo_masses, m1, sm0, beta, delta, gamma):
    sample_log_sm = np.linspace(5, 14, num=201)
    sample_log_hm = f_shmr_inverse(sample_log_sm, m1, sm0, beta, delta, gamma)
    return np.interp(log_halo_masses, sample_log_hm, sample_log_sm)
