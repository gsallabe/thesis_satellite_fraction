	# functions for doing shit that I am already doing in my analysis to clean it up
# Greg Sallaberry
# 2019

import numpy as np
import copy
from astropy.io import fits
from astropy.table import Table, Column
from astropy.cosmology import FlatLambdaCDM
from halotools.mock_observables import apply_zspace_distortion
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
import colossus
from colossus.cosmology import cosmology
from colossus.halo import mass_so

# Colossus params for HSC
params_hsc = {
			'flat': True , 
			'H0': 70.0 , 
			'Om0': 0.3 , 
			'Ob0': 0.049, 
			'sigma8': 0.81, 
			'ns': 0.95
			}

h_song = .7
cosmology.addCosmology('huang18',params_hsc)

# Colossus params for UM
params_um = {
			'flat': True , 
			'H0': 67.77 , 
			'Om0': 0.307115 , 
			'Ob0': 0.048206, 
			'sigma8': 0.8228, 
			'ns': 0.96
			}

h_smdpl = 0.6777
cosmology.addCosmology('SMDPL',params_um) # this is what Song had me use when doing the sat frac
cosmo_um = cosmology.setCosmology('SMDPL')



def make_data_table(file, stellar_mass, virial_mass, redshift, hsc): # get the data for the galaxies I want
	# file: path to file
	# stellar_mass: name of stellar mass column (str)
	# virial_mass: name of stellar mass column (str)
	# redshift: name of redshift column (str)
	# hsc: (boolean) - is the data from HSC or UM? 
	# output: table will be rank-ordered by mass, have the source column names
	# and also the useful flag colums that aren't added vis another method

	# general things
	h_hsc = 0.7
	h_smdpl = 0.6777
	hdu1 = fits.open(file)
	full_set = Table(hdu1[1].data)

	data_table = Table(names = full_set.colnames) # blank table


	if hsc == False: # for UM case

		cosmology.setCosmology('SMDPL') # set cosmology for um


		for i in range(len(np.array(full_set[ stellar_mass ]))):
			if full_set[ stellar_mass ][i] > 11.45:
				data_table.add_row( full_set[i] )

		# apply z-space distortions
		z_dist = apply_zspace_distortion(data_table['z'], data_table['vz'], 0.37, cosmo, 400.0)
		data_table['z_rsd'] = z_dist

		# get rid of h in spatial coords
		data_table['z_rsd'] = data_table['z_rsd'] / h_smdpl
		data_table['x'] = data_table['x'] / h_smdpl
		data_table['y'] = data_table['y'] / h_smdpl

		# get virial radius
		# m_to_r use 'vir' definition
		r_h = mass_so.M_to_R((10**data_table[ virial_mass ])*h_smdpl , .37, 'vir') * 1e-3 #Mpc/
		data_table['r_vir'] = (r_h / h_smdpl) * (1.37) #comoving Mpc

		# Make an index column
		data_table['index'] = np.zeros(len(data_table[ stellar_mass ]))

		# rank order by mass
		data_table.sort(stellar_mass)
		data_table.reverse()

		for i in range(len(data_table['index'])):
			data_table['index'][i] = i


	elif hsc == True:

		cosmology.setCosmology('huang18') # use this cosmology for hsc

		for i in range(len(np.array(full_set[ stellar_mass ]))):
			if full_set[ stellar_mass ][i] > 11.45 and full_set[ redshift ][i] > .25 and full_set[ redshift ][i] < .45:
				data_table.add_row(full_set[i]) 

		# Now to find r_halo
		r_halo = mass_so.M_to_R((10**data_table[ virial_mass ])*h_hsc , data_table[ redshift ], 'vir') / h_hsc # kpc
		# Turn the physical radius into an angular size in degrees
		theta_per_kpc = FlatLambdaCDM(H0=70 , Om0= 0.3).arcsec_per_kpc_proper(data_table[ redshift ])
		r_halo_deg = ((r_halo*u.kpc * theta_per_kpc).to(u.deg)) / u.deg

		data_table['r_halo'] = r_halo_deg

		# rank order by mass
		data_table.sort( stellar_mass )
		data_table.reverse()

	return ( data_table )

def get_mock_photoz_err(um_table, hsc_table):
	delta_z = 0.04 # based on the sigma from HSC

	l_comoving = cosmo_um.comovingDistance(0.37, 0.37 +delta_z)



	# determine fraction of hsc galaxies without specz by mass
	# find photo z fractions
	hsc_no_specz = np.isnan(hsc_table['z_spec'])

	hsc_photoz = Table(names = hsc_table.colnames) # table for only photo-z

	for i in range(len(hsc_table)):
	    if hsc_no_specz[i] == True:
	        hsc_photoz.add_row(hsc_table[i])

	mass_bins = [11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3]

	hist_hsc, edges_hsc = np.histogram(hsc_table['logm_max'], bins = mass_bins)
	hist_phot_z, edges_phot_z = np.histogram(hsc_photoz['logm_max'], bins = mass_bins)

	photoz_frac = hist_phot_z/hist_hsc

	# Find how many random numbers to draw (In the least efficient way possible)
	# This assigns random data points to adjust position based on fraction of HSC specz
	hist_um, edges_um = np.histogram(um_table['logms_max'], mass_bins)
	
	random_sample0 = np.random.rand(int(photoz_frac[0]*hist_um[0]))
	random_index0 = random_sample0 * hist_um[0]
	
	random_sample1 = np.random.rand(int(photoz_frac[1]*hist_um[1]))
	random_index1 = hist_um[0]+(random_sample1 * hist_um[1])
	
	random_sample2 = np.random.rand(int(photoz_frac[2]*hist_um[2]))
	random_index2 = hist_um[0]+hist_um[1]+ (random_sample2 * hist_um[2])
	
	random_sample3 = np.random.rand(int(photoz_frac[3]*hist_um[3]))
	random_index3 = hist_um[0]+hist_um[1]+hist_um[2]+(random_sample3 * hist_um[3])
	
	random_sample4 = np.random.rand(int(photoz_frac[1]*hist_um[4]))
	random_index4 = hist_um[0]+hist_um[1]+hist_um[2]+hist_um[3]+(random_sample4 * hist_um[4])
	
	random_index = np.concatenate((random_index0, random_index1, random_index2, random_index3, random_index4), axis = 0)

	# Turn array of indices into integers
	for i in range(len(random_index)):
		random_index[i] = np.int(random_index[i])

	# here are the random distortions
	l_kick = np.random.normal(0.0, l_comoving, size = len(random_index))
    

	random_index_integer = random_index.astype(int) # this is the final array of numbers to use

	um_kick = copy.deepcopy(um_table)

	# Add errors to the random rows
	for i in range(len(random_index_integer)):
	    um_kick['z_rsd'][random_index_integer[i]] = um_kick['z_rsd'][random_index_integer[i]] + l_kick[i]

	return(um_kick)


def open_table(file):
	# since I don't want tp run my code to make my data tables every time,
	# I'll use this to just open files where I store them
	hdu1 = fits.open(file)
	table = Table(hdu1[1].data)

	return( table )
