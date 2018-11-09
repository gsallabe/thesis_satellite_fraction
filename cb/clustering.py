import numpy as np
import astropy.cosmology
import halotools.mock_observables

from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.theory.DDrppi import DDrppi
from Corrfunc.utils import convert_3d_counts_to_cf

from colossus.cosmology import cosmology

import data as d
from get_sm_for_sim import get_sm_for_sim

# We get dd along the pi direction. Square into a single bin
def squash(dd):
    dd[0]["npairs"] = np.sum(dd["npairs"])
    return dd[:1]

# Approximate dd error
def add_poisson_err(dd):
    dd[0]["npairs"] += np.sqrt(dd[0]["npairs"])
    return dd

def compute_sim_clustering(sim_data, sim_size, log_stellar_masses, cen_cuts, sat_cuts):
    s1 = sim_data[
        (log_stellar_masses > cen_cuts[0]) & (log_stellar_masses < cen_cuts[1])
    ]
    s2 = sim_data[
            (log_stellar_masses > sat_cuts[0]) & (log_stellar_masses < sat_cuts[1])
    ]
    random_len = max(len(s1), len(s2)) * 10
    print("Randoms len is 10x the data {}".format(random_len))

    r1 = sim_size * np.random.random(size=(random_len, 3))
    r1 = r1.ravel().view([("halo_x", np.float64), ("halo_y", np.float64), ("halo_z", np.float64)])

    r2 = sim_size * np.random.random(size=(random_len, 3))
    r2 = r2.ravel().view([("halo_x", np.float64), ("halo_y", np.float64), ("halo_z", np.float64)])

    dd = squash(sim_clustering(s1, s2, sim_size, applyRSD1=True, applyRSD2=True))
    dr = squash(sim_clustering(s1, r2, sim_size, applyRSD1=True))
    rd = squash(sim_clustering(r1, s2, sim_size, applyRSD2=True))
    rr = squash(sim_clustering(r1, r2, sim_size))

    sim_clust_cf = convert_3d_counts_to_cf(len(s1), len(s2), len(r1), len(r2), dd, dr, rd, rr)
    dd, dr, rd, rr = add_poisson_err(dd), add_poisson_err(dr), add_poisson_err(rd), add_poisson_err(rr),
    sim_clust_w_err = convert_3d_counts_to_cf(len(s1), len(s2), len(r1), len(r2), dd, dr, rd, rr)
    assert len(sim_clust_cf) == 1, len(sim_clust_cf)

    return sim_clust_cf[0], sim_clust_w_err[0] - sim_clust_cf[0]


# Given the galaxys (location, mass) and the mass cuts for the centrals and satellites
# Compute clustering (for some hardcoded definition of clustering)
def compute_hsc_clustering(gals, cen_cuts, sat_cuts, for_plot=False):
    s1 = gals[
        (gals["logm_max"] > cen_cuts[0]) & (gals["logm_max"] < cen_cuts[1])
    ]
    s2 = gals[
        (gals["logm_max"] > sat_cuts[0]) & (gals["logm_max"] < sat_cuts[1])
    ]

    # Load randoms ensuring that their z distibution is the same as our samples
    r1 = d.load_randoms(s1["z_best"])
    r2 = d.load_randoms(s2["z_best"])
    print("Randoms should be much longer that sample:", len(r1), len(s1), len(s2))

    # Split randoms into two parts
    r_div = np.arange(len(r1))
    np.random.shuffle(r_div)
    r1 = r1[r_div[:len(r_div) // 2]]
    r2 = r2[r_div[len(r_div) // 2:]]
    assert len(r1) == len(r2)

    dd = squash(obs_clustering(s1, s2))
    dr = squash(obs_clustering(s1, r2))
    rd = squash(obs_clustering(r1, s2))
    rr = squash(obs_clustering(r1, r2))

    obs_clust_cf = convert_3d_counts_to_cf(len(s1), len(s2), len(r1), len(r2), dd, dr, rd, rr)
    dd, dr, rd, rr = add_poisson_err(dd), add_poisson_err(dr), add_poisson_err(rd), add_poisson_err(rr),
    obs_clust_w_err = convert_3d_counts_to_cf(len(s1), len(s2), len(r1), len(r2), dd, dr, rd, rr)

    assert len(obs_clust_cf) == 1, len(obs_clust_cf)
    return obs_clust_cf[0], obs_clust_w_err[0] - obs_clust_cf[0]


def sim_clustering(s1, s2, sim_size, applyRSD1=False, applyRSD2=False):
    # cnts = halotools.mock_observables.counts_in_cylinders(
    #         s1[["halo_x", "halo_y", "halo_z"]].view((np.float64, 3)),
    #         s2[["halo_x", "halo_y", "halo_z"]].view((np.float64, 3)),
    #         proj_search_radius=1,
    #         cylinder_half_length=10,
    #         period=sim_size,
    # )
    # pet_z = (s1["halo_z"] + np.random.normal(0, 5, len(s1))) % 400
    # assert np.sum(test["npairs"]) == np.sum(cnts)

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
            nthreads=1,
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

    return res

def obs_clustering(s1, s2, test=False):
    cosmo = cosmology.setCosmology("planck18")
    res = DDrppi_mocks(
            autocorr=False,
            cosmology=1, # This doesn't matter
            nthreads=1, # This is ignored because we didn't compile with it
            pimax=10,
            binfile=np.linspace(0.000001, 1, num=2), # Just 1 bin out to 1Mpc/h
            RA1=s1["ra"],
            DEC1=s1["dec"],
            CZ1=cosmo.comovingDistance(0., s1["z_best"]), # Returns Mpc/h
            RA2=s2["ra"],
            DEC2=s2["dec"],
            CZ2=cosmo.comovingDistance(0., s2["z_best"]),
            is_comoving_dist=True,
    )

    if test:
        print(test_obs_clustering(s1, s2))
        print(np.sum(res["npairs"]))

    return res

def test_obs_clustering(s1, s2):
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

def compute_clustering(s1_s2, s1_r, n_s2, n_r):
    # https://files.slack.com/files-pri/T5WPLGLAF-FD1NHDZ1R/image.png

    # The number of s2 and randoms near to s1
    # If these numbers are the same and the samples are the same size, xi == 0. s2 ~ random
    # If these numbers are the same but s2 is half the size, xi == 1. s2 twice as likely
    # to be near to s1 as random
    # etc
    xi = (s1_s2 / s1_r) * (n_r / n_s2) - 1
    return xi
