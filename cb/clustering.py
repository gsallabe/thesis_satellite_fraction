import numpy as np
from halotools.mock_observables import counts_in_cylinders
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.theory.DDrppi import DDrppi
from colossus.cosmology import cosmology

from get_obs_params import sim_size


def sim_clustering(s1, s2):
    cnts = counts_in_cylinders(
            s1[["halo_x", "halo_y", "halo_z"]].view((np.float64, 3)),
            s2[["halo_x", "halo_y", "halo_z"]].view((np.float64, 3)),
            proj_search_radius=1,
            cylinder_half_length=10,
            period=sim_size,
    )

    # Just make sure that corrfunc is being used correctly
    test = DDrppi(
            autocorr=False,
            nthreads=1,
            pimax=10,
            binfile=np.linspace(0.000001, 1, num=2), # Just 1 bin out to 1Mpc
            X1=s1["halo_x"],
            Y1=s1["halo_y"],
            Z1=s1["halo_z"],
            periodic=True,
            boxsize=sim_size,
            X2=s2["halo_x"],
            Y2=s2["halo_y"],
            Z2=s2["halo_z"],
    )
    assert np.sum(test["npairs"]) == np.sum(cnts)

    return np.sum(cnts)

def obs_clustering(s1, s2, test=False):
    cosmo = cosmology.setCosmology("planck18")
    res = DDrppi_mocks(
            autocorr=False,
            cosmology=1, # This doesn't matter
            nthreads=1, # This is ignored because we didn't compile with it
            pimax=10,
            binfile=np.linspace(0.000001, 1, num=2), # Just 1 bin out to 1Mpc
            RA1=s1["ra"],
            DEC1=s1["dec"],
            CZ1=cosmo.comovingDistance(0., s1["z_best"]), # Returns Mpc/h
            RA2=s2["ra"],
            DEC2=s2["dec"],
            CZ2=cosmo.comovingDistance(0., s2["z_best"]),
            is_comoving_dist=True,
    )

    assert res.shape == (10,)
    if test:
        print(test_obs_clustering(s1, s2))
        print(np.sum(res["npairs"]))

    return np.sum(res["npairs"])

def test_obs_clustering(s1, s2):
    cnts = 0
    cosmo = cosmology.setCosmology("planck18")
    # import pdb; pdb.set_trace()
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
