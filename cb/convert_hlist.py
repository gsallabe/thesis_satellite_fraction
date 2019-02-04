import numpy as np
import halotools.sim_manager


hlist_cols = {
    "id": (1, "int32"),
    "pid": (5, "int32"),
    "upid": (6, "int32"),
    "Mvir": (10, "float64"),
    "Mpeak": (60, "float64"),
    "Vmax@Mpeak": (74, "float64"),
    "x": (17, "float64"),
    "y": (18, "float64"),
    "z": (19, "float64"),
    "vx": (20, "float64"),
    "vy": (21, "float64"),
    "vz": (22, "float64"),
}

hlist_min_cuts = {
        "Mpeak": 1e11,
}

def main():
    data_base = "/home/christopher/Data/data/MDPL/"
    hlist_reader = halotools.sim_manager.TabularAsciiReader(
            data_base + "hlist_0.73330.list",
            hlist_cols,
            row_cut_min_dict=hlist_min_cuts,
    )
    reduced_catalog = hlist_reader.read_ascii()
    np.save(data_base + "hlist_0.73330.cut.5", reduced_catalog)


if __name__ == "__main__":
    main()
