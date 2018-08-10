# Greg Sallaberry
# 2018

# import relevant packages
import numpy as np
import copy
from astropy.table import Table , Column
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy import units as u
from halotools.mock_observables import counts_in_cylinders




# The first function just finds the satellites of a single galaxy
def model_satellite(i, table, lcut, use_halotools = False):
    
    if use_halotools == False:
        # the object we choose here is going to be a central by the way we have defined things
        cent_gal = table[i] 
    
        # this was a trick that Song used to make a new catalog for the other objects of interest.
        # the utility is that we don't have to loop through the whole catalog EVERY TIME
        cat_use = copy.deepcopy(table[(table['logms_tot_mod'] < cent_gal['logms_tot_mod'])])
    
    
    
        # find the galaxies within the delta l limit
        delta_l_cut = cat_use[np.abs(cent_gal['z_dist'] - cat_use['z_dist']) <= lcut] #this creates a new table
    
        if len(delta_l_cut) > 0:
         # find separation for gals within dl (x-y dist)
            delta_l_cut['sep'] = np.sqrt( ((delta_l_cut['x'] - cent_gal['x'])**2) + ((delta_l_cut['y'] - cent_gal['y'])**2) )

        for i in range(len(delta_l_cut)):
            if delta_l_cut['sep'][i] <= cent_gal['r_vir']:
                table['flag'][int(delta_l_cut['index'][i])] = 1 #flag galaxies in rvir and dl
                #print('satellite, yo')
    if use_halotools == True:
        cent_gal = table[i]
        cat_use = copy.deepcopy(table[(table['logms_tot_mod'] < cent_gal['logms_tot_mod'])])            


# This function iterates over the table to find satellites. 
# Returns: satellite fraction
def run_model_satellite(in_table, l, nbins):
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
    #mmin = np.min(in_table['logms_tot_mod'])
    #mmax = np.max(in_table['logms_tot_mod'])

    hist_all, edges_all = np.histogram(in_table['logms_tot_mod'], bins= nbins)
    hist_cen, edges_cen = np.histogram(cen_table['logms_tot_mod'], bins= edges_all)
    hist_sat, edges_sat = np.histogram(sat_table['logms_tot_mod'], bins= edges_all)

    mass_center = np.log10((10**edges_cen[1:] + 10**edges_cen[:-1]) / 2)

    f_sat = (hist_sat / hist_all) * 100 # satellite fraction in percent

    err = np.sqrt(hist_sat)/hist_all * 100
    return(mass_center, f_sat, err)

# ---------- PDR Functions ---------------------

def pdr_satellite(i, table, z_lim):
   
    # the object we choose here is going to be a central by the way we have defined things
    cent_gal = table[i] 
    
    # this was a trick that song used to make a new catalog for the other objects of interest.
    # the utility is that we don't have to loop through the whole catalog EVERY TIME
    cat_use = copy.deepcopy(table[(table['logm_max'] < cent_gal['logm_max'])])
    
    
    # this is code Marie recommended for a more sophistiated delta z cut. Originally, I used a flat
    # delta z cut
    #z_lim = 0.005
    #cvir = 5.
    #littleh = FlatLambdaCDM(H0=70 , Om0= 0.3).H0.value/100.
    #p_nfw = profile_nfw.NFWProfile(M=10**cent_gal['logmh_vir']*littleh,c=cvir,z=cent_gal['z_best'],mdef='200m')
    # Assume maximum circular velocity is 1.4 times the lD velocity dispersion in the halo, 
    # following Tormen+97.
    # Set the Delta z cut to be 3 times the 1D dispersion. 
    #v_lim = p_nfw.Vmax()[0]/1.4*3
    # Convert velocity cut to redshift cut
    #z_lim = ltu.dz_from_dv(v_lim*u.km/u.s,cent_gal['z_best'])
    
    
    # find the galaxies within the delta z limit
    dz= cat_use[np.abs(cent_gal['z_best'] - cat_use['z_best']) <= z_lim]
    
    if len(dz) > 0:
        # find separation for gals within dz
        dz['sep'] = cent_gal['coord'].separation(dz['coord']).degree 
    for i in range(len(dz)):
        if dz['sep'][i] <= cent_gal['r_halo']:
            table['flag'][int(dz['index'][i])] = 1
            #print('satellite, yo')


def run_pdr_satellite(in_table, dz, bin_edges):

    in_table['coord'] = SkyCoord(in_table['ra']*u.deg, in_table['dec']*u.deg) # angular coordinate of each galaxy

    in_table['flag'] = np.zeros(len(in_table['logm_max'])) 

    in_table['index'] = np.zeros(len(in_table['logm_max']))
    for i in range(len(in_table['index'])):
        in_table['index'][i] = i
        in_table['sep'] = np.zeros(len(in_table['logm_max'])) #column of angular separations

    for idx in range(len(in_table)):
        pdr_satellite(idx, in_table, dz)
    
    in_table.remove_columns(['sep','coord'])

    cen_table = Table(names = in_table.colnames)
    sat_table = Table(names = in_table.colnames)
    for i in range(len(in_table['flag'])):
        if in_table['flag'][i] == 0:
            cen_table.add_row(in_table[i])
        if in_table['flag'][i] == 1:
            sat_table.add_row(in_table[i])

    hist_all, edges_all = np.histogram(in_table['logm_max'], bins = bin_edges)

    hist_cen, edges_cen = np.histogram(cen_table['logm_max'], bins = edges_all)

    hist_sat, edges_sat = np.histogram(sat_table['logm_max'], bins = edges_all)

    mass_center = np.log10((10**(edges_cen[1:]) + 10**(edges_cen[:-1])) / 2)

    frac_sat = (hist_sat / hist_all) * 100.0
    for k in range(len(frac_sat)):
        if np.isnan(frac_sat[k]) == True:
            frac_sat[k] = 0 

    err = (np.sqrt(hist_sat) / hist_all) * 100.0 # poisson error bars 

    return(mass_center, frac_sat, err)
