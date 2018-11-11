import numpy as np
import astropy.cosmology
import halotools.mock_observables

from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.theory.DDrppi import DDrppi
from Corrfunc.utils import convert_3d_counts_to_cf

from colossus.cosmology import cosmology

import data as d

def compute_sim_clustering(sim_data, sim_size, log_stellar_masses, cen_cuts, sat_cuts):
    s1 = sim_data[
        (log_stellar_masses > cen_cuts[0]) & (log_stellar_masses < cen_cuts[1])
    ]
    s2 = sim_data[
            (log_stellar_masses > sat_cuts[0]) & (log_stellar_masses < sat_cuts[1])
    ]
    # Compromise between accuracy and run time
    random_len = max(len(s1), len(s2)) * 20

    r1 = sim_size * np.random.random(size=(random_len, 3))
    r1 = r1.ravel().view([("halo_x", np.float64), ("halo_y", np.float64), ("halo_z", np.float64)])

    r2 = sim_size * np.random.random(size=(random_len, 3))
    r2 = r2.ravel().view([("halo_x", np.float64), ("halo_y", np.float64), ("halo_z", np.float64)])

    dd = _squash(sim_clustering(s1, s2, sim_size, applyRSD1=True, applyRSD2=True))
    dr = _squash(sim_clustering(s1, r2, sim_size, applyRSD1=True))
    rd = _squash(sim_clustering(r1, s2, sim_size, applyRSD2=True))
    rr = _squash(sim_clustering(r1, r2, sim_size))


    for sample in [dd, dr, rd, rr]: assert len(sample) == 1
    # dd is the smallest (though not by a long way) but we still claim poisson error dominates
    # This not a huge deal as the uncertainty on the sim clustering << that on the obs clustering
    for sample in [dr, rd, rr]:
        if sample["npairs"] < dd["npairs"]: print("This is not good", sample["npairs"], dd["npairs"])

    sim_clust = convert_3d_counts_to_cf(len(s1), len(s2), len(r1), len(r2), dd, dr, rd, rr)
    sim_clust_w_err = convert_3d_counts_to_cf(len(s1), len(s2), len(r1), len(r2), _add_poisson_err(dd), dr, rd, rr)

    for sample in [sim_clust, sim_clust_w_err]: assert len(sample) == 1

    return sim_clust[0], np.abs(sim_clust_w_err[0] - sim_clust[0])

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
def compute_hsc_clustering(gals, cen_cuts, sat_cuts):
    s1 = gals[
        (gals["logm_max"] > cen_cuts[0]) & (gals["logm_max"] < cen_cuts[1])
    ]
    s2 = gals[
        (gals["logm_max"] > sat_cuts[0]) & (gals["logm_max"] < sat_cuts[1])
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
            cosmology=1, # This is ignored as is_comiving_dist == True and we convert z to Mpc/h
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
        print(handrolled_obs_clustering(s1, s2))
        print(np.sum(res["npairs"]))

    return res

# My artisinal, handrolled correlation function for observations.
# Only used to check that I am calling corrfunc correctly
def handrolled_obs_clustering(s1, s2):
    cnts = 0
    cosmo = cosmology.setCosmology("planck18")
    for s in s1:
        z_dist = cosmo.comovingDistance(s["z_best"], s2["z_best"])
        s2_sub = s2[np.abs(z_dist) < 10]
        angular_dist = np.arccos(
                np.sin(np.radians(s["dec"])) * np.sin(np.radians(s2_sub["dec"])) +
                np.cos(np.radians(s["dec"])) * np.cos(np.radians(s2_sub["dec"])) * np.cos(np.radians(s["ra"] - s2_sub["ra"])))

        xy_dist = angular_dist * cosmo.comovingDistance(0, s["z_best"])
        s2_match = np.count_nonzero(xy_dist < 1)
        cnts += s2_match
    return cnts


# We get dd along the pi direction. Square into a single bin
def _squash(dd):
    dd[0]["npairs"] = np.sum(dd["npairs"])
    return dd[:1]

# Approximate dd error
def _add_poisson_err(dd):
    dd[0]["npairs"] += np.sqrt(dd[0]["npairs"])
    return dd
