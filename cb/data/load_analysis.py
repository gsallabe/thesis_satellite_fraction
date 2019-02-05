import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def load_mock(fname, complete_cut):
    sim_data = np.load(fname)

    sats = sim_data[sim_data["upid"] != -1]
    cens = sim_data[sim_data["upid"] == -1]
    assert len(sim_data) == len(sats) + len(cens)

    # _plot_completeness(cens, complete_cut)

    cens_complete = cens[cens["stellar_mass"] > complete_cut]
    sats_complete = sats[sats["stellar_mass"] > complete_cut]
    sim_data_complete = sim_data[sim_data["stellar_mass"] > complete_cut]

    return sim_data_complete, cens_complete, sats_complete, 1000

def load_asap_um(fname, complete_cut):
    sim_data = fits.open(fname)
    sim_data = np.asarray(sim_data[1].data)
    sim_data = sim_data.view(sim_data.dtype.fields, np.ndarray)

    sats = sim_data[sim_data["upid"] != -1]
    cens = sim_data[sim_data["upid"] == -1]
    assert len(sim_data) == len(sats) + len(cens)

    cens_complete = cens[cens["logms_tot_asap"] > complete_cut]
    sats_complete = sats[sats["logms_tot_asap"] > complete_cut]
    sim_data_complete = sim_data[sim_data["logms_tot_asap"] > complete_cut]

    return sim_data_complete, cens_complete, sats_complete



def _plot_completeness(cens, complete_cut):
    _, ax = plt.subplots()
    ax.hist(np.log10(cens["stellar_mass"]), bins="fd")
    ax.axvline(np.log10(complete_cut), color="blue")
