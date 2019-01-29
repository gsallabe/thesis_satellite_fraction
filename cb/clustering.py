import numpy as np
import astropy.cosmology
import halotools.mock_observables

from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.theory.DDrppi import DDrppi
from Corrfunc.utils import convert_3d_counts_to_cf

from colossus.cosmology import cosmology

from joblib import Memory

import os
memory = Memory((os.path.dirname(__file__) or ".") + "/joblib_cache", verbose=2)

import data as d

def compute_sim_clustering(sim_data, sim_size, log_stellar_masses, cen_sat_div):
    s1 = sim_data[log_stellar_masses > cen_sat_div]
    s2 = sim_data[
            (log_stellar_masses < cen_sat_div) & (log_stellar_masses > (cen_sat_div - 0.1))
    ]

    aRSD = True
    dd = _squash(sim_clustering(s1, s2, sim_size, applyRSD1=aRSD, applyRSD2=aRSD))
    assert len(dd) == 1

    random_len = max(len(s1), len(s2)) * 1000
    dr, rd, rr = predict_counts(sim_size, s1, s2, random_len)

    sim_clust = convert_3d_counts_to_cf(len(s1), len(s2), random_len, random_len, dd, dr, rd, rr)
    sim_clust_w_err = convert_3d_counts_to_cf(len(s1), len(s2), random_len, random_len, _add_poisson_err(dd), dr, rd, rr)

    for sample in [sim_clust, sim_clust_w_err]: assert len(sample) == 1

    return sim_clust[0], np.abs(sim_clust_w_err[0] - sim_clust[0])

def predict_counts(sim_size, s1, s2, random_len, pimax=10, rmax=1):
    v_cyl = np.pi * rmax**2 * (pimax * 2)
    frac_one_cyl = v_cyl / sim_size**3

    like_cf = lambda x: np.array([x], dtype=[("npairs", "f8")])
    # dr
    dr = like_cf((frac_one_cyl * len(s1)) * random_len)

    # rd
    rd = like_cf((frac_one_cyl * random_len) * len(s2))

    # RR - the number of r2 in r1's cylinder
    rr = like_cf((frac_one_cyl * random_len) * random_len)

    return dr, rd, rr


def sim_clustering(s1, s2, sim_size, applyRSD1=False, applyRSD2=False, test=False):
    z1 = s1["halo_z"]
    if applyRSD1:
        z1 = halotools.mock_observables.apply_zspace_distortion(
                s1["halo_z"],
                s1["vz"],
                0.35,
                astropy.cosmology.Planck15,
                sim_size,
        )

    z2 = s2["halo_z"]
    if applyRSD2:
        z2 = halotools.mock_observables.apply_zspace_distortion(
                s2["halo_z"],
                s2["vz"],
                0.35,
                astropy.cosmology.Planck15,
                sim_size,
        )

    res = DDrppi(
            autocorr=False,
            nthreads=12,
            pimax=10,
            binfile=np.linspace(0.000001, 1, num=2), # Just 1 bin out to 1Mpc
            X1=s1["halo_x"],
            Y1=s1["halo_y"],
            Z1=z1,
            periodic=True,
            boxsize=sim_size,
            X2=s2["halo_x"],
            Y2=s2["halo_y"],
            Z2=z2,
    )

    if test:
        # Use halotools to check that I am calling corrfunc correctly
        cnts = halotools.mock_observables.counts_in_cylinders(
                s1[["halo_x", "halo_y", "halo_z"]].view((np.float64, 3)),
                s2[["halo_x", "halo_y", "halo_z"]].view((np.float64, 3)),
                proj_search_radius=1,
                cylinder_half_length=10,
                period=sim_size,
        )
        assert np.sum(res["npairs"]) == np.sum(cnts)

    return res

# Given the galaxys (location, mass) and the mass cuts for the centrals and satellites
# Compute clustering (for some definition of clustering encoded in this function)
@memory.cache()
def compute_hsc_clustering(gals, cen_sat_div):
    s1 = gals[gals["logm_max"] > cen_sat_div] # Centrals
    s2 = gals[
        (gals["logm_max"] < cen_sat_div) & (gals["logm_max"] > (cen_sat_div - 0.1))
    ]

    # Load randoms ensuring that their z distibution is the same as our samples
    r1 = d.load_randoms(s1["z_best"])
    r2 = d.load_randoms(s2["z_best"])
    assert len(r1) > 20 * max(len(s1), len(s2)), "Randoms should be much longer than the sample"

    # Select half the randoms for r1 and the other half for r2
    r_div = np.arange(len(r1))
    np.random.shuffle(r_div)
    r1 = r1[r_div[:len(r_div) // 2]]
    r2 = r2[r_div[len(r_div) // 2:]]
    assert len(r1) == len(r2)

    # Sqaush because we don't care about having it in 10 pi bins. We just want the summary
    dd = _squash(obs_clustering(s1, s2))
    dr = _squash(obs_clustering(s1, r2))
    rd = _squash(obs_clustering(r1, s2))
    rr = _squash(obs_clustering(r1, r2))

    for sample in [dd, dr, rd, rr]: assert len(sample) == 1
    # dd is the smallest by a long way so poisson error on it will dominate
    for sample in [dr, rd, rr]: assert sample["npairs"] > 6 * dd["npairs"]

    # Go from counts to clustering (and estimate the error)
    obs_clust = convert_3d_counts_to_cf(len(s1), len(s2), len(r1), len(r2), dd, dr, rd, rr)
    obs_clust_w_err = convert_3d_counts_to_cf(len(s1), len(s2), len(r1), len(r2), _add_poisson_err(dd), dr, rd, rr)

    for sample in [obs_clust, obs_clust_w_err]: assert len(sample) == 1

    return obs_clust[0], np.abs(obs_clust_w_err[0] - obs_clust[0])

def obs_clustering(s1, s2, test=False):
    cosmo = cosmology.setCosmology("planck18")
    res = DDrppi_mocks(
            autocorr=False,
            cosmology=1, # This is ignored as is_comoving_dist == True and we convert z to Mpc/h
            nthreads=12,
            pimax=10, # We get this back in 10 bins
            binfile=np.linspace(0.000001, 1, num=2), # Just 1 bin out to 1 Mpc/h
            RA1=s1["ra"],
            DEC1=s1["dec"],
            CZ1=cosmo.comovingDistance(0., s1["z_best"]), # Returns Mpc/h
            RA2=s2["ra"],
            DEC2=s2["dec"],
            CZ2=cosmo.comovingDistance(0., s2["z_best"]),
            is_comoving_dist=True,
    )

    if test:
        print("Comparing corrfunc with my handrolled correlational func")
        print(handrolled_obs_clustering(s1, s2)[0])
        print(np.sum(res["npairs"]))

    return res

# My artisinal, handrolled correlation function for observations.
# Only used to check that I am calling corrfunc correctly
# And used in the obs part
def handrolled_obs_clustering(s1, s2):
    cnts = 0
    cosmo = cosmology.setCosmology("planck18")
    indexes = []
    for i, s in enumerate(s1):
        z_dist = cosmo.comovingDistance(s["z_best"], s2["z_best"])
        s2_sub = s2[np.abs(z_dist) < 10]
        angular_dist = np.arccos(
                np.sin(np.radians(s["dec"])) * np.sin(np.radians(s2_sub["dec"])) +
                np.cos(np.radians(s["dec"])) * np.cos(np.radians(s2_sub["dec"])) * np.cos(np.radians(s["ra"] - s2_sub["ra"])))

        xy_dist = angular_dist * cosmo.comovingDistance(0, s["z_best"])
        s2_match = np.count_nonzero(xy_dist < 1)
        cnts += s2_match
        for j in np.where(np.abs(z_dist) < 10)[0][xy_dist < 1]:
            indexes.append((i, j))
    indexes = np.array(indexes, dtype=[("i1", np.int32), ("i2", np.int32)])
    return cnts, indexes


# We get dd along the pi direction. Square into a single bin
def _squash(dd):
    dd[0]["npairs"] = np.sum(dd["npairs"])
    return dd[:1]

# Approximate dd error
def _add_poisson_err(dd):
    dd[0]["npairs"] += np.sqrt(dd[0]["npairs"])
    return dd

def analysis_obs_clustering(gals, cen_sat_div):
    s1 = gals[gals["logm_max"] > cen_sat_div] # Centrals
    s2 = gals[
        (gals["logm_max"] < cen_sat_div) & (gals["logm_max"] > (cen_sat_div - 0.01))
    ]

    cnts, indexes = handrolled_obs_clustering(s1, s2)
    assert cnts == np.sum(obs_clustering(s1, s2)["npairs"])
    return indexes, s2["logm_max"]


def analysis_sim_clustering(sim_data, cen_sat_div, sim_size):
    s1 = sim_data[sim_data["stellar_mass"] > 10**cen_sat_div]
    s2 = sim_data[
            (sim_data["stellar_mass"] < 10**cen_sat_div) &
            (sim_data["stellar_mass"] > 10**(cen_sat_div - 0.01))
    ]
    assert not np.may_share_memory(s1, sim_data)

    s1["halo_z"] = halotools.mock_observables.apply_zspace_distortion(
            s1["halo_z"], s1["vz"], 0.35, astropy.cosmology.Planck15, sim_size,
    )
    s2["halo_z"] = halotools.mock_observables.apply_zspace_distortion(
            s2["halo_z"], s2["vz"], 0.35, astropy.cosmology.Planck15, sim_size,
    )

    _, indexes = halotools.mock_observables.counts_in_cylinders(
            np.copy(s1)[["halo_x", "halo_y", "halo_z"]].view((np.float64, 3)),#s1_pos,
            np.copy(s2)[["halo_x", "halo_y", "halo_z"]].view((np.float64, 3)),#s2_pos,
            proj_search_radius=1,
            cylinder_half_length=10,
            period=sim_size,
            return_indexes=True,
    )
    return indexes, s2["stellar_mass"]
