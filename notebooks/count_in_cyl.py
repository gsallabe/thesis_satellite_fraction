import copy
import numpy as np
from astropy.coordinates import SkyCoord, Angle



def pdr_pairs(i, j, in_table, z_lim, parent_table): 
    
    # the object we choose here is going to be a central by the way we have defined things
    cent_gal = in_table[i]

    
    # this was a trick that song used to make a new catalog for the other objects of interest.
    # the utility is that we don't have to loop through the whole catalog EVERY TIME
    cat_use = copy.deepcopy(parent_table[(parent_table['logm_max'] > mass_bins[j+1])])
    
    
    # find the galaxies within the delta z limit
    dz = cat_use[np.abs(cent_gal['z_best'] - cat_use['z_best']) <= z_lim]
    
    pair_index = [] # list of indices of objects within cylinder
    
    if len(dz) > 0:
        # find separation for gals within dz
        dz['sep'] = SkyCoord(cent_gal['ra']*u.deg, cent_gal['dec']*u.deg).separation(SkyCoord(dz['ra']*u.deg, dz['dec']*u.deg)).degree 
    for i in range(len(dz)):
        if dz['sep'][i] <= cent_gal['r_vir']:
            pair_index.append(dz['index'][i])
            
    num_pairs = len(pair_index)
    
    return(num_pairs) # need the normalization factor for later. Length of the threshold sample

def model_pairs(i, j, in_table, l_lim):

    cent_gal = in_table[i]

    cat_use = copy.deepcopy(um_use[(um_use['logms_max'] > mass_bins[j+1])])

    # find the galaxies within the delta z limit
    delta_l_cut = cat_use[np.abs(cent_gal['z_dist'] - cat_use['z_dist']) <= l_lim]
    
    pair_index = [] # list of indices of objects within cylinder
    
    if len(delta_l_cut) > 0:
        # find separation for gals within dl (x-y dist)
        delta_l_cut['sep'] = np.sqrt( ((delta_l_cut['x'] - cent_gal['x'])**2) + ((delta_l_cut['y'] - cent_gal['y'])**2) )

    for i in range(len(delta_l_cut)):
        if delta_l_cut['sep'][i] <= cent_gal['r_vir']:
            pair_index.append(dz['index'][i])
            
    num_pairs = len(pair_index)
    
    return(num_pairs)
       
