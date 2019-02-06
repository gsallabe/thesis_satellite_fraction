import matplotlib.pyplot as plt
import numpy as np
import data as d

from scipy.interpolate import interp1d


# HSC isn't actually fully spec-z. We need to add some photo-z uncertainty to
# some, mass dependent, fraction of galaxies.
def get_fraction(plot=False):
    hsc = d.load_hsc_gals(allCols=True)
    min_mass = np.min(hsc["logm_max"])
    print(f"Constraints down to mass: {min_mass}")
    bins = np.append(np.linspace(min_mass, 12.1, num=10), 12.4)
    bin_centers = (bins[1:] + bins[:-1]) / 2

    specz_frac = []
    for i in range(len(bins[:-1])):
        s = hsc[
                (hsc["logm_max"] > bins[i]) & (hsc["logm_max"] < bins[i+1])
        ]
        specz = np.count_nonzero(s[
            # These are true spec-z
            (s["z_source"] == b"z_gama") |
            (s["z_source"] == b"z_hsc") |
            (s["z_source"] == b"z_sdss") |
            # These are photo-z but with spec-z accuracy
            (s["z_source"] == b"z_camira_cen") |
            (s["z_source"] == b"z_redmapper_cen")
        ])
        photoz = np.count_nonzero(s[
            (s["z_source"] == b"z_camira_mem") |
            (s["z_source"] == b"z_franken")
        ])

        assert specz + photoz == len(s)
        specz_frac.append(specz / len(s))

    if plot:
        _, ax = plt.subplots()
        ax.plot(bin_centers, specz_frac)

    return interp1d(bin_centers, specz_frac, kind="cubic", bounds_error=False, fill_value=(0.8, 1))
