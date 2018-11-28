# Mock Making

A set of scripts to make an HSC like mock. Each step should be self documenting (e.g. at the top of the notebook) so here we just list the steps.

* convert_hlist.py: A script to take an MDPL halo list in consistent trees format and to remove unneeded columns and rows where the halo mass is small (< 12).
* main.ipynb: The notebook where we make the mock (mock.npy)
* analysis.ipynb: Some analysis on the mock
