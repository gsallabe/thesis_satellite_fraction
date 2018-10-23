import numpy as np
import halotools.sim_manager


hlist_cols = {
    "id": (1, "int32"),
    "Mpeak": (60, "float64"),
    "x": (17, "float32"),
    "y": (18, "float32"),
    "z": (19, "float32"),
    "vx": (20, "float32"),
    "vy": (21, "float32"),
    "vz": (22, "float32"),
}

hlist_min_cuts = {
        "Mpeak": 1e12,
}

def main():
    data_base = "/home/christopher/Data/data/MDPL/"
    hlist_reader = halotools.sim_manager.TabularAsciiReader(
            data_base + "hlist_0.73330.list",
            hlist_cols,
            row_cut_min_dict=hlist_min_cuts,
    )
    reduced_catalog = hlist_reader.read_ascii()
    np.save(data_base + "hlist_0.73330.cut", reduced_catalog)



if __name__ == "__main__":
    main()
