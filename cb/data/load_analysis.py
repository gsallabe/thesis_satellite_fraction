import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
from astropy.io import fits

import pandas as pd

def load_mock(fname, complete_cut, plot_completeness=False):
    sim_data = np.load(fname)

    sats = sim_data[sim_data["upid"] != -1]
    cens = sim_data[sim_data["upid"] == -1]
    assert len(sim_data) == len(sats) + len(cens)

    if plot_completeness:
        _plot_completeness(cens, complete_cut)

    cens_complete = cens[cens["stellar_mass"] > complete_cut]
    sats_complete = sats[sats["stellar_mass"] > complete_cut]
    sim_data_complete = sim_data[sim_data["stellar_mass"] > complete_cut]

    return sim_data, cens, sats, sim_data_complete, cens_complete, sats_complete, 1000

def load_asap_um(fname, complete_cut):
    sim_data = fits.open(fname)
    sim_data = np.asarray(sim_data[1].data)
    sim_data = sim_data.view(sim_data.dtype.fields, np.ndarray)

    # I did this for the first data set and now everything needs it
    sim_data = rfn.rename_fields(sim_data, {
            "x": "halo_x",
            "y": "halo_y",
            "z": "halo_z",
    })

    sats = sim_data[sim_data["upid"] != -1]
    cens = sim_data[sim_data["upid"] == -1]
    assert len(sim_data) == len(sats) + len(cens)

    cens_complete = cens[cens["logms_tot_asap"] > complete_cut]
    sats_complete = sats[sats["logms_tot_asap"] > complete_cut]
    sim_data_complete = sim_data[sim_data["logms_tot_asap"] > complete_cut]

    return sim_data_complete, cens_complete, sats_complete, 400

def add_velocities_to_asap_um(sim_data):
    f = np.load("/home/christopher/Data/data/universe_machine/sfr_catalog_insitu_exsitu_0.712400_final_wssfr_wv.npz")
    t = pd.DataFrame(np.append(f["centrals"], f["satellites"])[["id", "vx", "vy", "vz"]]).set_index("id")

    sim_data = pd.DataFrame(sim_data)
    jsim_data = sim_data.join(t, on="upid", how="inner")

    print(len(sim_data), len(jsim_data))
    return jsim_data

def _plot_completeness(cens, complete_cut):
    _, ax = plt.subplots()
    ax.hist(np.log10(cens["stellar_mass"]), bins="fd")
    ax.axvline(np.log10(complete_cut), color="blue")
