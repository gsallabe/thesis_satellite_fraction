import numpy as np
import scipy.stats
from astropy.io import fits

def mock(cens, sats, bins):
    sats_counts, _, _ = scipy.stats.binned_statistic(
            sats["stellar_mass"], None, statistic="count", bins=10**bins)
    cens_counts, _, _ = scipy.stats.binned_statistic(
            cens["stellar_mass"], None, statistic="count", bins=10**bins)

    sat_frac = sats_counts / (cens_counts + sats_counts)
    sat_frac_unc = sat_frac * np.sqrt( 1/sats_counts + 1/(sats_counts + cens_counts) )

    return sat_frac, sat_frac_unc

# Fig 13 from https://arxiv.org/pdf/1207.2160.pdf
def reddick():

    # They are using are M/h^2. Assume h ~ 0.7
    bins = np.linspace(10.9, 11.9, num=6)
    bins = np.log10(10**bins / (0.7**2))

    # These are the observed values - fig 8. Not what we want!
    # f(sat_frac, group_finder)
    # sat_frac = [0.19, 0.18, 0.15, 0.09, 0.06, 0.07]
    # sat_frac_unc = [0.02, 0.02, 0.02, 0.02, 0.02, 0.03]

    sat_frac = [0.18, 0.15, 0.13, 0, 0, 0]
    sat_frac_unc = [0.01, 0.02, 0.05, 0, 0, 0]

    return bins, sat_frac, sat_frac_unc

# Fig 11 from https://arxiv.org/pdf/1509.00482.pdf
def saito():
    # We assume cmass is complete about 11.3.
    # Throw a fairly arbitrary x error to account for the fact that they don't mass cut
    # This doesn't need a
    return 11.4, 0.125, 0.1

def reid():
    return 11.4, 0.10, 0.1, 0.005

def ASAP_UM(cens, sats, bins):
    bins = np.log10(bins) # UM works in log space
    sats_counts, _, _ = scipy.stats.binned_statistic(
            sats["logms_tot_asap"], None, statistic="count", bins=10**bins)
    cens_counts, _, _ = scipy.stats.binned_statistic(
            cens["logms_tot_asap"], None, statistic="count", bins=10**bins)

    sat_frac = sats_counts / (cens_counts + sats_counts)
    sat_frac_unc = sat_frac * np.sqrt( 1/sats_counts + 1/(sats_counts + cens_counts) )

    return sat_frac, sat_frac_unc
