import numpy as np
from astropy.io import fits
from astropy.table import Table
import numpy.lib.recfunctions as rfn



local_base = "/home/christopher/research/satellite_fraction/data/"

def load_smf():
    smf = np.load(local_base + "s16a_wide2_massive_smf_mmax_11.6.npy")
    return smf


def load_gals():
    gals = fits.open(local_base + "s16a_massive_logm100_11.45_z_0.25_0.45_all.fits")
    recarr = np.array(gals[1].data)
    structarr = recarr.view(recarr.dtype.fields, np.ndarray)
    return structarr[[
        "ra", "dec", "z_best", "logm_max",
    ]]

def load_randoms():
    randoms = np.load(local_base + "s16a_random_500k.npy")
    return rfn.rename_fields(randoms, {"z": "z_best"})

sim_size = 400

def load_smdpl():
    print("Remember to change sim size to 400")
    f = np.load("/home/christopher/Data/data/universe_machine/sfr_catalog_insitu_exsitu_0.712400_final_wssfr_wv.npz")
    sim_data = np.append(f["centrals"], f["satellites"])
    sim_data = sim_data[sim_data["mp"] > 1e12]

    # Remember that things are outside of the box...
    for col in "xyz":
        sim_data[col] %= 400

    # Make this look like MDPL
    return rfn.rename_fields(sim_data, {
            "x": "halo_x",
            "y": "halo_y",
            "z": "halo_z",
            "mp": "halo_mvir", # This is an open question - use this or m?
    })

def load_mdpl():
    print("Remember to change sim size to 1000")
    # dtype([('halo_id', '<i8'), ('halo_upid', '<i8'), ('halo_mvir', '<f8'), ('halo_Vpeak', '<f8'), ('halo_Acc_Rate_2*Tdyn', '<f8'), ('halo_x', '<f8'), ('halo_y', '<f8'), ('halo_z', '<f8')])

    sim_data = np.load("/home/christopher/Data/data/MDPL/hlist_0.73330_vpeak3_mvir_gt_12_wxyz.npy")
    return sim_data
