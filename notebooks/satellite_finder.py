# Greg Sallaberry
# 2018

# import relevant packages
import numpy as np
import copy
from astropy.table import Table , Column



# The first function just finds the satellites of a single galaxy
def model_satellite(i, table, lcut):
    

    # the object we choose here is going to be a central by the way we have defined things
    cent_gal = table[i] 
    
    # this was a trick that song used to make a new catalog for the other objects of interest.
    # the utility is that we don't have to loop through the whole catalog EVERY TIME
    cat_use = copy.deepcopy(table[(table['logms_tot_mod'] < cent_gal['logms_tot_mod'])])
    
    
    
    # find the galaxies within the delta z limit
    delta_l_cut = cat_use[np.abs(cent_gal['z_dist'] - cat_use['z_dist']) <= lcut] #this creates a new table
    
    if len(delta_l_cut) > 0:
        # find separation for gals within dz (x-y dist)
        delta_l_cut['sep'] = np.sqrt( ((delta_l_cut['x'] - cent_gal['x'])**2) + ((delta_l_cut['y'] - cent_gal['y'])**2) )
    for i in range(len(delta_l_cut)):
        if delta_l_cut['sep'][i] <= cent_gal['r_vir']:
            table['flag'][int(delta_l_cut['index'][i])] = 1
            #print('satellite, yo')


# This function iterates over the table to find satellites. 
# Returns: satellite fraction
def run_model_satellite(in_table, l):
    # creating new columns useful for the satellite finding inside of the function
    in_table['flag'] = np.zeros(len(in_table['logms_tot_mod'])) 
    in_table['sep'] = np.zeros(len(in_table['logms_tot_mod'])) #column of separations


    for idx in range(len(in_table)): 
        model_satellite(idx, in_table, l)
    
    in_table.remove_columns(['sep']) # remove separation column because it's useless now

# -------- Put the stuff in new tables ------------------
    
    cen_table = Table(names = in_table.colnames) #make tables of cent/satdata
    sat_table = Table(names = in_table.colnames)
    for i in range(len(in_table['flag'])):
        if in_table['flag'][i] == 0:
            cen_table.add_row(in_table[i])
        if in_table['flag'][i] == 1:
            sat_table.add_row(in_table[i])
        
    in_table.remove_columns(['flag']) # now remove flag column

# ----------- Use histogram to calculate sat frac ---------
    # range of masses (for binning)
    mmin = np.min(in_table['logms_tot_mod'])
    mmax = np.max(in_table['logms_tot_mod'])

    hist_cen, edges_cen = np.histogram(cen_table['logms_tot_mod'],range = [11.5,mmax], bins=8)
    hist_sat, edges_sat = np.histogram(sat_table['logms_tot_mod'],range = [11.5,mmax], bins=8)

    mass_center = (edges_cen[1:] + edges_cen[:-1]) / 2

    f_sat = (hist_sat / hist_cen) * 100 # satellite fraction in percent

    err = np.sqrt(hist_sat)/hist_cen * 100
    return(mass_center, f_sat, err)
