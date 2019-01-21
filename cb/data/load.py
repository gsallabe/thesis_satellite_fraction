import numpy as np
from astropy.io import fits
import numpy.lib.recfunctions as rfn
import scipy.interpolate


local_base = "/home/christopher/research/satellite_fraction/data/"

def load_smf():
    smf = np.load(local_base + "s16a_wide2_massive_smf_mmax_11.6.npy")

    # This starts in units of count per mpc^3 per msun
    # We want to use mpc/h throughout.
    # 1 mpc ~ 1*0.7 mpc/h

    # So basically multiply by 1/h cubed. This is the same as dividing by h^3
    for col in ["smf", "smf_err", "smf_low", "smf_upp"]:
        smf[col] = smf[col] / 0.7**3 # Song just uses 0.7
    return smf


def load_hsc_gals():
    gals = fits.open(local_base + "s16a_massive_logm100_11.45_z_0.25_0.45_all.fits")
    recarr = np.array(gals[1].data)
    structarr = recarr.view(recarr.dtype.fields, np.ndarray)
    return structarr[[
        "ra", "dec", "z_best", "logm_max",
    ]]

def load_randoms(s_z):
    randoms = np.load(local_base + "s16a_random_500k.npy")

    # We need to ensure that the randoms have the same redshift distribution as the sample
    # They are currently a uniform random [0, 1)
    hist, edges = np.histogram(s_z, bins=100)
    cdf = np.append(0, np.cumsum(hist))
    cdf = cdf / cdf[-1]
    f = scipy.interpolate.interp1d(cdf, edges)
    randoms["z"] = f(randoms["z"])

    return rfn.rename_fields(randoms, {"z": "z_best"})

def load_smdpl():
    f = np.load("/home/christopher/Data/data/universe_machine/sfr_catalog_insitu_exsitu_0.712400_final_wssfr_wv.npz")
    sim_data = np.append(f["centrals"], f["satellites"])
    sim_data = sim_data[sim_data["m"] > 10**12.2]

    # Remember that things are outside of the box...
    for col in "xyz":
        sim_data[col] %= 400


    # Make this look like MDPL
    return rfn.rename_fields(sim_data, {
            "x": "halo_x",
            "y": "halo_y",
            "z": "halo_z",
            "m": "halo_mvir",
    }), 400

def load_mdpl():
    sim_data = np.load("/home/christopher/Data/data/MDPL/hlist_0.73330.cut.4.npy")

    sim_data = rfn.rename_fields(sim_data, {
            "Mpeak": "halo_mvir",
            "x": "halo_x",
            "y": "halo_y",
            "z": "halo_z",
    })

    sim_data["halo_mvir"] = sim_data["halo_mvir"] / 0.6777

    return sim_data, 1000
