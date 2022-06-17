

# run_smemp_apf.py --> python version of the ipynb notebook. 
# Script to run Specmatch-Emp model on APF spectra and output results including derived stellar properties 
# and residual between target and best matching spectra.
#
# NOTE: This script is modified from smemp_multifile.ipynb (itself modified from C.N's smemp.py). Modifications from smemp.py include added ability to process mulitple stellar
#      targets from a directory, new deblazing function, option to simplistically remove cosmic rays, and new output
#      files including output of residual. See project project write-up for details.  
#      Modifications are noted by the initials ADZ and the date.
# Input: path to a directory containing APF fits spectra files. If more than one file corresponds to a star, 
#      they must be grouped into a subdirectory.
# Output: Specmatch-emp derived Stellar properties in specmatch_results.csv, fits file for each star containing normalized,
#       deblazed target spectrum, residual between target sepctrum and linear combinatino of best matching spectra, and 
#       shifted wavelength scale. Also produces a log file. Please update file paths for these if needed before running.



# ADZ: allow importing .ipynb scripts 
#get_ipython().system('pip install import-ipynb')

# send output to 'specmatch_output.txt' (this is a log file for the current run)
# CAUTION! Deletes existing log files

# define the options  -- need to do this here to set up output streaming to prevent in/out error on running in command line
import argparse
import os
import sys
parser = argparse.ArgumentParser(description="Run Specmatch-Emp model and isochrone analysis on APF stellar spectra.")
parser.add_argument("--get_only_NDR", help="Only compute the normalized, deblazed, registered spectra (NOT YET IMPLEMENTED -- use  get_all_NDR_corrected.ipynb)", action = 'store_true') # default is false
parser.add_argument("--save_unshifted", help="Save the deblazed, normalized, resampled and NOT shifted target to a fits file.", action = 'store_true') # default is false
parser.add_argument("--save_shifted", help="Save the deblazed, normalized, resampled and shifted target to a fits file.", action = 'store_true') # default is false
parser.add_argument("--save_shifts", help="Save the pixel shift values for each of the nine sections", action = 'store_true') # default is false
parser.add_argument("-unshifted_out","--unshifted_out_path", type=str, help= "Directory to save deblazed, normalized, resampled and NOT shifted target to (ie. 'NDRU_calib').") 
parser.add_argument("-shifted_out","--shifted_out_path", type=str, help= "Directory to save deblazed, normalized, resampled and shifted target to (ie. 'NDRS_calib').") 
parser.add_argument("--omit_properties", help= "Do NOT output the stellar property results from SM-Emp", action = 'store_true') 
parser.add_argument("--omit_resids", help= "Do NOT output the residuals", action = 'store_true') 
parser.add_argument("--dont_run_iso", help= "Do NOT run isochrone analysis to determine additional stellar properties", action = 'store_true') 
parser.add_argument("--save_SM_object", help= "Save the Specmatch objects themselves", action = 'store_true') 
parser.add_argument("--save_peak_scale_factors", help= "Save the factors by which the peaks of each order are scaled.", action = 'store_true')
parser.add_argument("--save_scale_factors", help= "Save the factors by which each pixel is scaled.", action = 'store_true')
parser.add_argument("--save_baselines", help= "Save baseline flux values (num photons).", action = 'store_true')
parser.add_argument("--save_photon_counts", help= "Save photon counts (resampled, non-normalized, non-deblazed spectrum).", action = 'store_true')
parser.add_argument("--display_plots", help= "Make additional plots (beyond standard outputs)", action = 'store_true') 
parser.add_argument("-make_new", "--make_new", help="Make new output directories (without this flag will add to existing directories, but careful if same output residuals or plots have already been run in existing output directories).", action = 'store_true')
parser.add_argument("-in", "--path_to_dir", type=str, help="Path to input directory containing spectra to run on (ie. '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/apf_spectra_highest_SNR' for calibration run.")
parser.add_argument("-resid_out","--residuals_out_path", type=str, help= "Directory to save residuals to (ie. 'NDRR_calib').") 
parser.add_argument("-results_out","--results_filename", type=str, help= "Filename to save steller property results to (ie. 'specmatch_results_calib.csv').") 
parser.add_argument("-log","--logfile", type=str, help= "Filename for streaming output.") 
parser.add_argument("-plots_out", "--plots_out_path", type=str, help="Directory to save plots to (ie. 'SM_output_plots_calib').")
parser.add_argument("-start", "--start", type=int, help="Index of file to start on.")
parser.add_argument("-stop", "--stop", type=int, help="Index of file to stop on.")
parser.add_argument("-results_type","--results_subdirectory", type=str, help= "What type of run is this. Must match a subdirectory name (all_apf, calibration, tests, or old).") 
#parser.add_argument("-lags_dir","--lags_out_dir", type=str, help= "Directory for lags output.") 
#parser.add_argument("-lags_file","--lags_file", type=str, help= "Filename for lags output.") 
args = parser.parse_args()
# options
get_only_NDR = args.get_only_NDR # only compute the normalized, deblazed, registered spectra (NOT YET IMPLEMENTED -- use get_all_NDR_corrected.ipynb)
get_properties = not args.omit_properties # run SM-Emp and output the stellar property results 
get_resids = not args.omit_resids # run SM-Emp and output the residuals
run_iso = not args.dont_run_iso # run SM-Emp and run isochrone analysis to determine additional stellar properties
save_shifted = args.save_shifted # save the shifted target
save_unshifted = args.save_unshifted # save the unshifted target
save_shifts = args.save_shifts # save the shift values
save_SM_object = args.save_SM_object # to save the Specmatch objects themselves
display_plots = args.display_plots # to make plots 
#save_peak_scale_factors = args.save_peak_scale_factors # to save scale factors at peak of blaze functions
#save_scale_factors = args.save_scale_factors # to save scale factors
#save_baselines = args.save_baselines # to save baselines
save_photon_counts = args.save_photon_counts # to save absolute photon counts
residuals_out_path = 'APF_spectra/' + args.residuals_out_path + '/' #'/NDRR_calib/' # directory to save residuals to 
properties_out_path = 'SM_stellar_properties/' # directory to save steller property results to
shifted_out_path = 'APF_spectra/' + args.shifted_out_path + '/' #'/NDRS_calib/' # directory to save shifted target to 
unshifted_out_path = 'APF_spectra/' + args.unshifted_out_path + '/' #'/NDRU_calib/' # directory to save unshifted target to 
#lags_out_path = 'SM_shift_values/' # directory to save shift value results to
results_subdir = args.results_subdirectory + '/' # subdirectoy to save property results
results_filename = args.results_filename #'specmatch_results_calib.csv' # filename for stellar property results within properties_out_path dir and results_subdir
plots_out_path =  args.plots_out_path + '/' #'SM_output_plots_calib' # directory to save plots to
properties_plots_path = args.plots_out_path + '/Stellar_properties/' # directory for property plots
spectra_plots_path = args.plots_out_path + '/Ref_lincomb_spectra/' # directory for spectra plots
#lags_path = lags_out_path  + args.lags_file # file for lags output
#scale_factor_path = 'SM_scale_factors/' + args.results_subdirectory # directory to save factors by which each pixel is scaled
#baseline_path = 'SM_baseline_fluxes/' + args.results_subdirectory # directory to save baseline fluxes
photon_counts_path = 'SM_photon_counts/' + args.results_subdirectory # directory to save absolute photon counts
logfile = args.logfile
path_to_dir = args.path_to_dir
start = args.start
stop = args.stop
make_new = args.make_new


# set up output streaming
if os.path.exists(logfile):
    os.remove(logfile)
old_stdout = sys.stdout
sys.stdout = open(logfile, 'w')

# standard imports
import import_ipynb #ADZ ADD 6/23/20
import pandas as pd #ADZ ADD 7/13/20
from astroquery.simbad import Simbad #ADZ ADD 8/6/20
import sys, os
from os import listdir
from os.path import isfile, join
import csv
from pylab import *
import pylab
from scipy import signal
import astropy.io.fits as pf
from astropy.io import fits
from astroquery.gaia import Gaia
import astropy.units as units
from astropy.coordinates import SkyCoord
import copy
import glob
import h5py,pdb
import pyvo
import datetime

# Specmatch imports (relative to current directory)
sys.path.insert(0, '/mnt_home/azuckerman/BL_APF_DAP/specmatch_emp')
from specmatch_emp.specmatchemp import library
from specmatch_emp.specmatchemp import plots as smplot
from specmatch_emp.specmatchemp.spectrum import Spectrum
from specmatch_emp.specmatchemp.specmatch import SpecMatch

# Isoclassify imports
os.environ['DUST_DIR'] = '/mnt_home/azuckerman/BL_APF_DAP/mwdust/dust_dir'
import mwdust
from isoclassify.direct import classify as classify_direct
from isoclassify.grid import classify as classify_grid
from isoclassify import DATADIR
PACKAGEDIR = '/mnt_home/azuckerman/BL_APF_DAP/isoclassify/isoclassify'

# Other local imports
#from deblaze import afs_deblaze # ADZ comment out
from rescale import get_rescaled_wave_soln
#from rescale import resample
from rescale import resample_order
from optparse import OptionParser
from bstar_deblaze import bstar_deblazed2 #ADZ ADD 7/17/20
#from bstar_deblaze import zoe_percentile_deblazed
from AFS_deblaze import AFS

# Example command-line run:
#python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/apf_spectra_highest_SNR' -resid_out 'NDRR_testing' -results_out #'specmatch_results_testing.csv' -plots_out 'SM_output_plots_testing' 

# define an exception to use in reading in filenames
class HIP_name_Exception(Exception):
    pass

# define an exception if iodine is found in the spectrum
class Iodine_Exception(Exception):
    pass

# define an exception if the star is a KNOWN binary
class Binary_Exception(Exception):
    pass

# define an exception if the observation KNOWN to be of a non-stellar object
class Non_Stellar_Exception(Exception):
    pass

# define functions to write results (for separate cases of successful and failed run)
def write_results(fd, my_spectrum, sm, iso_results, id_name, sim_name, HIP_name, filenames, Teff_bounds_flag, R_bounds_flag, 
                  SNR, iodine_flag, iso_fail_flag, warnings, write_new = False): 
                 
    """
    Write to a csv the derived properties of a target following
    the SpecMatch process
    Args:
        fd(File): object for the csv file to write detrended and un-detrended stellar property results to
        my_spectrum (spectrum.Spectrum): Target spectrum
        sm (specmatch.SpecMatch): Contains results of the algorithm
        write_new (Boolean): Whether to write to a new csv file
    """
    
    fieldnames = ['name','HIP_name', 'Simbad_resolvable_name', 'filenames', 'SNR', 'iodine_flag', 'binary_flag', 'non-stellar_flag', 'failure_code', 'failure_message', 'iso_fail_flag',
                  'warnings', 'Teff_bounds_flag', 'R_bounds_flag', 
                  'Teff', 'u_Teff', 'Teff_detrended', 'u_Teff_detrended', 'iso_Teff', 'iso_up_Teff', 'iso_um_Teff', 
                  'radius','u_radius', 'radius_detrended', 'u_radius_detrended', 'iso_radius', 'iso_up_radius', 'iso_um_radius',
                  'feh','u_feh', 'feh_detrended','u_feh_detrended', 'iso_feh', 'iso_up_feh', 'iso_um_feh',
                  'logg', 'u_logg', 'logg_detrended', 'u_logg_detrended', 'iso_logg', 'iso_up_logg', 'iso_um_logg',
                  'mass', 'u_mass', 'mass_detrended', 'u_mass_detrended', 'iso_mass', 'iso_up_mass', 'iso_um_mass',
                  'age', 'u_age', 'age_detrended', 'u_age_detrended', 'iso_age', 'iso_up_age', 'iso_um_age', 
                  'best_mean_chi_squared', 'timestamp']

    thewriter = csv.DictWriter(fd, fieldnames=fieldnames)
    
    # create timestamp
    dt = datetime.datetime.now()
    timestamp = dt.strftime("%d") + dt.strftime("%b") + dt.strftime("%Y") + '-' + dt.strftime("%X") 

    if (write_new): thewriter.writeheader()

    thewriter.writerow({'name' : id_name, #my_spectrum.name,
                        'HIP_name': HIP_name,
                        'Simbad_resolvable_name': sim_name,
                        'filenames': filenames,
                        'SNR': SNR,
                        'iodine_flag': iodine_flag, # At least one spectrum for this star had iodine; do not trust results
                        'binary_flag': False,
                        'non-stellar_flag': False,
                        'failure_code': 0, # Did not fail (sucessfully ran SM-Emp on this star)
                        'failure_message': 'No failure',
                        'iso_fail_flag': iso_fail_flag,
                        'warnings': warnings,
                        'Teff_bounds_flag': Teff_bounds_flag,
                        'R_bounds_flag': R_bounds_flag,
                        'Teff' : '{0:.3f}'.format(sm.results_nodetrend['Teff']),
                        'u_Teff' : '{0:.3f}'.format(sm.results_nodetrend['u_Teff']),
                        'Teff_detrended' : '{0:.3f}'.format(sm.results['Teff']),
                        'u_Teff_detrended' : '{0:.3f}'.format(sm.results['u_Teff']),      
                        'iso_Teff' : '{0:.3f}'.format(iso_results.teff),
                        'iso_up_Teff' : '{0:.3f}'.format(iso_results.teffep), # plus uncertianty
                        'iso_um_Teff' : '{0:.3f}'.format(iso_results.teffem), # minus uncertianty
                        'radius' : '{0:.3f}'.format(sm.results_nodetrend['radius']),
                        'u_radius' : '{0:.3f}'.format(sm.results_nodetrend['u_radius']),
                        'radius_detrended' : '{0:.3f}'.format(sm.results['radius']),
                        'u_radius_detrended' : '{0:.3f}'.format(sm.results['u_radius']),       
                        'iso_radius' : '{0:.3f}'.format(iso_results.rad),
                        'iso_up_radius' : '{0:.3f}'.format(iso_results.radep), # plus uncertianty
                        'iso_um_radius' : '{0:.3f}'.format(iso_results.radem), # minus uncertianty
                        'feh' : '{0:.3f}'.format(sm.results_nodetrend['feh']),
                        'u_feh' : '{0:.2f}'.format(sm.results_nodetrend['u_feh']),
                        'feh_detrended' : '{0:.3f}'.format(sm.results['feh']),
                        'u_feh_detrended' : '{0:.3f}'.format(sm.results['u_feh']),    
                        'iso_feh' : '{0:.3f}'.format(iso_results.feh),
                        'iso_up_feh' : '{0:.3f}'.format(iso_results.fehep), # plus uncertianty
                        'iso_um_feh' : '{0:.3f}'.format(iso_results.fehem), # minus uncertianty  
                        'logg' : '{0:.3f}'.format(sm.results_nodetrend['logg']),
                        'u_logg' : '{0:.3f}'.format(sm.results_nodetrend['u_logg']),
                        'logg_detrended' : '{0:.3f}'.format(sm.results['logg']),
                        'u_logg_detrended' : '{0:.3f}'.format(sm.results['u_logg']),   
                        'iso_logg' : '{0:.3f}'.format(iso_results.logg),
                        'iso_up_logg' : '{0:.3f}'.format(iso_results.loggep), # plus uncertianty
                        'iso_um_logg' : '{0:.3f}'.format(iso_results.loggem), # minus uncertianty
                        'mass' : '{0:.3f}'.format(sm.results_nodetrend['mass']),
                        'u_mass' : '{0:.3f}'.format(sm.results_nodetrend['u_mass']),
                        'mass_detrended' : '{0:.3f}'.format(sm.results['mass']),
                        'u_mass_detrended' : '{0:.3f}'.format(sm.results['u_mass']),   
                        'iso_mass' : '{0:.3f}'.format(iso_results.mass),
                        'iso_up_mass' : '{0:.3f}'.format(iso_results.massep), # plus uncertianty
                        'iso_um_mass' : '{0:.3f}'.format(iso_results.massem), # minus uncertianty
                        'age' : '{0:.2f}'.format(sm.results_nodetrend['age']),
                        'u_age' : '{0:.2f}'.format(sm.results_nodetrend['u_age']),
                        'age_detrended' : '{0:.3f}'.format(sm.results['age']),
                        'u_age_detrended' : '{0:.3f}'.format(sm.results['u_age']),   
                        'iso_age' : '{0:.3f}'.format(iso_results.age),
                        'iso_up_age' : '{0:.3f}'.format(iso_results.ageep), # plus uncertianty
                        'iso_um_age' : '{0:.3f}'.format(iso_results.ageem), # minus uncertianty
                        'best_mean_chi_squared' : '{0:.2f}'.format(best_mean_chi_squared),
                        'timestamp' : timestamp})

   # Teff_detrended, feh_detrended, R_detrended = detrend(sm)
    
   # thewriter = csv.DictWriter(fd_detrended, fieldnames=fieldnames)

   # if (write_new): thewriter.writeheader()

   # thewriter.writerow({'apf_name' : my_spectrum.name,
   #                     'Teff' : '{0:.3f}'.format(sm.results['Teff']),
   #                     'u_Teff' : '{0:.3f}'.format(sm.results['u_Teff']),
   #                     'radius' : '{0:.3f}'.format(sm.results['radius']),
   #                     'u_radius' : '{0:.3f}'.format(sm.results_nodetrend['u_radius']),
   #                     'logg' : '{0:.3f}'.format(sm.results['logg']),
   #                     'u_logg' : '{0:.3f}'.format(sm.results['u_logg']),
   #                     'feh' : '{0:.3f}'.format(sm.results['feh']),
   #                     'u_feh' : '{0:.2f}'.format(sm.results['u_feh']),
   #                     'mass' : '{0:.3f}'.format(sm.results['mass']),
   #                     'u_mass' : '{0:.3f}'.format(sm.results['u_mass']),
   #                     'age' : '{0:.2f}'.format(sm.results['age']),
   #                     'u_age' : '{0:.2f}'.format(sm.results['u_age']),
   #                     'best_mean_chi_squared' :
   #                     '{0:.2f}'.format(best_mean_chi_squared)})

def write_results_fail_case(fd, id_name, sim_name, HIP_name, filenames, Teff_bounds_flag, R_bounds_flag, fail_code, fail_message, SNR, iodine_flag, binary_flag, non_stellar_flag, warnings, write_new = False):   
    """
    Write to a csv a line denoting that SpecMatch-Emp has failed to run on a target.
    Args:
        fd(File): object for the csv file to write detrended and un-detrended stellar property results to
        write_new (Boolean): Whether to write to a new csv file
    """

    fieldnames = ['name','HIP_name', 'Simbad_resolvable_name', 'filenames', 'SNR', 'iodine_flag', 'binary_flag', 'non-stellar_flag', 'failure_code', 'failure_message', 'iso_fail_flag',
                  'warnings', 'Teff_bounds_flag', 'R_bounds_flag', 
                  'Teff', 'u_Teff', 'Teff_detrended', 'u_Teff_detrended', 'iso_Teff', 'iso_up_Teff', 'iso_um_Teff', 
                  'radius','u_radius', 'radius_detrended', 'u_radius_detrended', 'iso_radius', 'iso_up_radius', 'iso_um_radius',
                  'feh','u_feh', 'feh_detrended','u_feh_detrended', 'iso_feh', 'iso_up_feh', 'iso_um_feh',
                  'logg', 'u_logg', 'logg_detrended', 'u_logg_detrended', 'iso_logg', 'iso_up_logg', 'iso_um_logg',
                  'mass', 'u_mass', 'mass_detrended', 'u_mass_detrended', 'iso_mass', 'iso_up_mass', 'iso_um_mass',
                  'age', 'u_age', 'age_detrended', 'u_age_detrended', 'iso_age', 'iso_up_age', 'iso_um_age',
                  'best_mean_chi_squared', 'timestamp']

    thewriter = csv.DictWriter(fd, fieldnames=fieldnames)

    if (write_new): thewriter.writeheader()

    iodine_flag = np.nan
    if fail_message == '__main__.Iodine_Exception: At least one target spectrum file contains iodine.':
        iodine_flag = 1
        
    # create timestamp
    dt = datetime.datetime.now()
    timestamp = dt.strftime("%d") + dt.strftime("%b") + dt.strftime("%Y") + '-' + dt.strftime("%X") 
    
    thewriter.writerow({'name' : id_name, #HIP_name,
                        'HIP_name': HIP_name,
                        'Simbad_resolvable_name': sim_name, 
                        'filenames': filenames,
                        'SNR': SNR,
                        'iodine_flag': iodine_flag, 
                        'binary_flag': binary_flag, 
                        'non-stellar_flag': non_stellar_flag,
                        'failure_code': fail_code,
                        'failure_message': fail_message,
                        'iso_fail_flag': np.nan, # not assined False or True so that searching False or True gives instances of isoclassify (not) failing, not SM failing .  
                        'warnings': warnings,
                        'Teff_bounds_flag': Teff_bounds_flag,
                        'R_bounds_flag': R_bounds_flag,
                        'Teff' : np.nan,
                        'u_Teff' : np.nan,
                        'Teff_detrended' : np.nan,
                        'u_Teff_detrended' : np.nan, 
                        'iso_Teff' : np.nan,
                        'iso_up_Teff' : np.nan,
                        'iso_um_Teff' : np.nan,
                        'radius' : np.nan,
                        'u_radius' : np.nan,
                        'radius_detrended' : np.nan,
                        'u_radius_detrended' : np.nan,  
                        'iso_radius' : np.nan,
                        'iso_up_radius' : np.nan,
                        'iso_um_radius' : np.nan,
                        'feh' : np.nan,
                        'u_feh' : np.nan,
                        'feh_detrended' : np.nan,
                        'u_feh_detrended' : np.nan, 
                        'iso_feh' : np.nan,
                        'iso_up_feh' : np.nan,
                        'iso_um_feh' : np.nan,
                        'logg' : np.nan,
                        'u_logg' : np.nan,
                        'logg_detrended' : np.nan,
                        'u_logg_detrended' : np.nan, 
                        'iso_logg' : np.nan,
                        'iso_up_logg' : np.nan,
                        'iso_um_logg' : np.nan,
                        'mass' : np.nan,
                        'u_mass' : np.nan,
                        'mass_detrended' : np.nan,
                        'u_mass_detrended' : np.nan, 
                        'iso_mass' : np.nan,
                        'iso_up_mass' : np.nan,
                        'iso_um_mass' : np.nan,
                        'age' : np.nan,
                        'u_age' : np.nan,
                        'age_detrended' : np.nan,
                        'u_age_detrended' : np.nan, 
                        'iso_age' : np.nan,
                        'iso_up_age' : np.nan,
                        'iso_um_age' : np.nan,
                        'best_mean_chi_squared' : np.nan,
                        'timestamp' : timestamp})

# define function to get Gaia and 2MASS names -- NOT USED as of 11/2/21 (read in from log file)
#def get_names(sim_name):
#    # get Gaia and 2MASS names (for creating flags and for later isochrone analysis)
#    result_table = Simbad.query_objectids(sim_name)
#    names = result_table.to_pandas()['ID'].to_list()
#    gaia_name = [name.decode('utf-8').split(' ')[-1] for name in names if name.decode('utf-8').startswith('Gaia DR2')]
#    two_MASS_name = [name.decode('utf-8').split('J')[-1] for name in names if name.decode('utf-8').startswith('2MASS')]
#    (gaia_source_id,) = gaia_name
#    (id_2MASS,) = two_MASS_name
#    return gaia_source_id, id_2MASS

#def write_lags(sim_name, HIP_name, lags, center_pix, fd, write_new = False):
#def write_lags(sim_name, HIP_name, wl_deltas, shifted_wl_values, fd, write_new = False):
#    #fieldnames = ['sim_name', 'HIP_name', 'lags', 'center_pix']
#    fieldnames = ['sim_name', 'HIP_name', 'wl_deltas', 'shifted_wl_values']
#    thewriter = csv.DictWriter(fd, fieldnames=fieldnames)
#    
#    # create timestamp
#    dt = datetime.datetime.now()
#    timestamp = dt.strftime("%d") + dt.strftime("%b") + dt.strftime("%Y") + '-' + dt.strftime("%X") #
#
#    if write_new: 
#        thewriter.writeheader()
#
#    thewriter.writerow({'sim_name' : sim_name,
#                        'HIP_name': HIP_name,
#                        'wl_deltas': wl_deltas,
#                        'shifted_wl_values': shifted_wl_values})
#    #                    'lags': lags,
#    #                    'center_pix': center_pix})


# define function to get Gaia properties
def query_gaia_data(gaia_source_id):
    # Gaia properties to flag stars outside library bounds
    #gaia_data =  pd.read_csv('./Gaia_properties_by_HIP_name.csv') # gaia stellar properties (only R and Teff)
    query1 = "SELECT source_id, ra, dec, parallax, parallax_error, teff_val, radius_val  FROM gaiadr2.gaia_source WHERE source_id = " + str(gaia_source_id)
    job = Gaia.launch_job(query1)
    gaia_data = job.get_results()
    return gaia_data

# define new resampling function (resample each order individually, and then combine)
def new_resample(wave_soln_ref, wave_soln, spectrum):
    resampled_orders = np.empty(shape=(0)) 
    resampled_wl = np.empty(shape=(0)) # contains resampled wl for each order concatenated (so there are repeats)
    for order in np.arange(30,52,1):
        order_values = spectrum[order][100:-101] # truncate the ends becuase the deblazing doesn't work well here --> unless in future iteration do this in deblazing
        apf_wl_values = wave_soln[order][100:-100]
        first = apf_wl_values[0]
        last = apf_wl_values[-1]
        new_first = first - 0.017 
        new_last = last + 0.17 
        new_wl_section = np.array(wave_soln_ref)[(new_first <= wave_soln_ref) * (new_last >= wave_soln_ref)]
        resampled_order = resample_order(new_wl_section, apf_wl_values, order_values)
        resampled_orders = np.hstack([resampled_orders, resampled_order])
        resampled_wl = np.hstack([resampled_wl, new_wl_section])
    # average duplicate flux values
    resampled_spectrum = np.zeros(len(wave_soln_ref))
    i=0
    for wl in wave_soln_ref:
        values = resampled_orders[resampled_wl == wl]
        avg = np.nanmean(values)
        resampled_spectrum[i] = avg
        i += 1       
    return resampled_spectrum


# define new resampling function for baseline photons -> sum not average in overlap regions
def new_resample_baseline(wave_soln_ref, wave_soln, arr):
    resampled_orders = np.empty(shape=(0)) 
    resampled_wl = np.empty(shape=(0)) # contains resampled wl for each order concatenated (so there are repeats)
    for order in np.arange(30,52,1):
        #order_values = arr[order][100:-101] # truncate the ends becuase the deblazing doesn't work well here --> unless in future iteration do this in deblazing
        #apf_wl_values = wave_soln[order][100:-100]
        order_values = arr[order][:-1] 
        apf_wl_values = wave_soln[order][:-1]
        first = apf_wl_values[0]
        last = apf_wl_values[-1]
        new_first = first - 0.017 
        new_last = last + 0.17 
        new_wl_section = np.array(wave_soln_ref)[(new_first <= wave_soln_ref) * (new_last >= wave_soln_ref)]
        resampled_order = resample_order(new_wl_section, apf_wl_values, order_values)
        resampled_orders = np.hstack([resampled_orders, resampled_order])
        resampled_wl = np.hstack([resampled_wl, new_wl_section])
    # sum duplicate flux values
    resampled_arr = np.zeros(len(wave_soln_ref))
    i=0
    for wl in wave_soln_ref:
        values = resampled_orders[resampled_wl == wl]
        total = np.nansum(values)
        resampled_arr[i] = total
        i += 1       
    return resampled_arr

# define function to run isochrone analysis
def run_isoclassify(sm, gaia_data, phot_results):
    # NOTE: this is largely copied from the Grid classification tutorial on the isoclassify github page (grid.ipynb), by Dan hUber.
    
    k_mag = float(phot_results['k_m']) # [Mag]
    h_mag = float(phot_results['h_m']) # [Mag] 
    j_mag = float(phot_results['j_m']) # [Mag]
    u_k_mag = float(phot_results['k_msigcom']) # [Mag]
    u_h_mag = float(phot_results['h_msigcom']) # [Mag] 
    u_j_mag = float(phot_results['j_msigcom']) # [Mag]
    
    # load models
    fn = os.path.join(DATADIR,'mesa_from_github.h5')
    modfile = h5py.File(fn,'r', driver='core', backing_store=False)
    model = {'age':np.array(modfile['age']),    'mass':np.array(modfile['mass']),    'feh_init':np.array(modfile['feh']),    'feh':np.array(modfile['feh_act']),    'teff':np.array(modfile['teff']),    'logg':np.array(modfile['logg']),    'rad':np.array(modfile['rad']),    'lum':np.array(modfile['rad']),    'rho':np.array(modfile['rho']),    'dage':np.array(modfile['dage']),    'dmass':np.array(modfile['dmass']),    'dfeh':np.array(modfile['dfeh']),    'eep':np.array(modfile['eep']),    'bmag':np.array(modfile['bmag']),    'vmag':np.array(modfile['vmag']),    'btmag':np.array(modfile['btmag']),    'vtmag':np.array(modfile['vtmag']),    'gmag':np.array(modfile['gmag']),    'rmag':np.array(modfile['rmag']),    'imag':np.array(modfile['imag']),    'zmag':np.array(modfile['zmag']),    'jmag':np.array(modfile['jmag']),    'hmag':np.array(modfile['hmag']),    'kmag':np.array(modfile['kmag']),    'bpmag':np.array(modfile['bpmag']),    'gamag':np.array(modfile['gamag']),    'rpmag':np.array(modfile['rpmag']),    'fdnu':np.array(modfile['fdnu']),    'avs':np.zeros(len(np.array(modfile['gamag']))),    'dis':np.zeros(len(np.array(modfile['gamag'])))}

    #ebf.read(os.path.join(DATADIR,'mesa.ebf'))
    # prelims to manipulate some model variables (to be automated soon ...)
    #pdb.set_trace()
    model['rho'] = np.log10(model['rho'])
    model['lum'] = model['rad']**2*(model['teff']/5777.)**4
    # next line turns off Dnu scaling relation corrections
    model['fdnu'][:]=1.
    model['avs']=np.zeros(len(model['teff']))
    model['dis']=np.zeros(len(model['teff']))
    
    # define class that contains observables
    x = classify_grid.obsdata()
    
    # add [Teff, logg, FeH] and [sigma_Teff, sigma_logg, sigma_FeH] 
    # currently just copied from a results file -- will read these in from SM
    sm_R = 	sm.results['radius']
    sm_Teff = sm.results['Teff']
    sm_feh = sm.results['feh']
    sm_logg = sm.results['logg']
    sm_u_R = sm.results['u_radius']
    sm_u_Teff = sm.results['u_Teff']
    sm_u_feh = sm.results['u_feh']
    sm_u_logg = sm.results['u_logg']
    x.addspec([sm_Teff, sm_logg , sm_feh], [sm_u_Teff, sm_u_logg, sm_u_feh])
    #x.addspec([5777.,4.44,0.0],[60.,0.07,0.04])
    
    # perform classification based on those inputs
    # paras = classify_grid.classify(input=x, model=model, dustmodel=0,plot=1,band='kmag')
    
    # add some more observables 
    # PARALLAX
    x.addplx(parallax/1000,u_parallax/1000) # /1000 to convert to arsec
    # JHK (2MASS) photometry
    x.addjhk([j_mag,h_mag,k_mag],[u_j_mag,u_h_mag,u_k_mag])
    
    # using photometry requires some treatment of reddening and extinction. the following functions
    # required for this.
    def query_dustmodel_coords(ra,dec,dust):
        if dust == 'allsky':
            reddenMap = mwdust.Combined19()
            ext = extinction('green19')
        if dust == 'green19':
            reddenMap = mwdust.Green19()
            ext = extinction('green19')
        if dust == 'zero':
            reddenMap = mwdust.Zero()
            ext = extinction('cardelli')
        if dust == 'none':
            reddenMap = 0
            ext = extinction('cardelli')
            print('Fitting for reddening.')
            return reddenMap,ext

        sightLines = SkyCoord(ra*units.deg,dec*units.deg,frame='icrs')
        sightLines = sightLines.transform_to('galactic')

        distanceSamples = np.loadtxt(f"{PACKAGEDIR}/data/distance-samples-green19.txt",delimiter=',')*1000.

        reddenContainer = reddenMap(sightLines.l.value,sightLines.b.value,distanceSamples/1000.)

        del reddenMap # To clear reddenMap from memory

        dustModelDF = pd.DataFrame({'ra': [ra], 'dec': [dec]})

        for index in range(len(reddenContainer)):
            dustModelDF['av_'+str(round(distanceSamples[index],6))] = reddenContainer[index]

        return dustModelDF,ext


    # using photometry requires some treatment of reddening and extinction. the following functions
    # required for this

    def query_dustmodel_coords(ra,dec,dust):
        if dust == 'allsky':
            reddenMap = mwdust.Combined19()
            ext = extinction('green19')
        if dust == 'green19':
            reddenMap = mwdust.Green19()
            ext = extinction('green19')
        if dust == 'zero':
            reddenMap = mwdust.Zero()
            ext = extinction('cardelli')
        if dust == 'none':
            reddenMap = 0
            ext = extinction('cardelli')
            print('Fitting for reddening.')
            return reddenMap,ext

        sightLines = SkyCoord(ra*units.deg,dec*units.deg,frame='icrs')
        sightLines = sightLines.transform_to('galactic')

        #PACKAGEDIR='../isoclassify/'

        distanceSamples = np.loadtxt(f"{PACKAGEDIR}/data/distance-samples-green19.txt",delimiter=',')*1000.

        reddenContainer = reddenMap(sightLines.l.value,sightLines.b.value,distanceSamples/1000.)

        del reddenMap # To clear reddenMap from memory

        dustModelDF = pd.DataFrame({'ra': [ra], 'dec': [dec]})

        for index in range(len(reddenContainer)):
            dustModelDF['av_'+str(round(distanceSamples[index],6))] = reddenContainer[index]

        return dustModelDF,ext

    def extinction(law):

        #PACKAGEDIR='./BL_APF_DAP/isoclassify/isoclassify' # ADZ modification 
        if (law == 'cardelli'):
            out = {}

            with open(f"{PACKAGEDIR}/data/extinction-vector-cardelli-iso.txt") as f:

                for line in f:
                    (key,val) = line.split(',')
                    out[key] = float(val)

        if (law == 'schlafly11'):
            out = {}

            with open(f"{PACKAGEDIR}/data/extinction-vector-schlafly11-iso.txt") as f:

                for line in f:
                    (key,val) = line.split(',')
                    out[key] = float(val)

        if (law == 'schlafly16'):
            # see http://argonaut.skymaps.info/usage under "Gray Component". this is a lower limit.
            grayoffset=0.063
            out = {}

            with open(f"{PACKAGEDIR}/data/extinction-vector-schlafly16-iso.txt") as f:

                for line in f:
                    (key,val) = line.split(',')
                    out[key] = float(val)+grayoffset

        if (law == 'green19'):
            out = {}

            with open(f"{PACKAGEDIR}/data/extinction-vector-green19-iso.txt") as f:

                for line in f:
                    (key,val) = line.split(',')
                    out[key] = float(val)
        return out

    # if we don't want to use a reddening map, isoclassify fits for Av. However, we need to 
    # define an extinction law
    ext = extinction('cardelli')
    
    # perform classification
    paras = classify_grid.classify(input=x, model=model, dustmodel=0,plot=0,ext=ext,band='kmag')
    return paras 


# define function to run SM-Emp
def run_specmatch(sim_name, id_name, path_name, lib, display_plots): #ADZ: made this into a function 6/29/20

    
    #parser = OptionParser()
    #parser.add_option("-f", "--file", action='store', type='string',
    #                  dest="pathname",
    #                  help="pass the path of the FITS file(s) as an argument")
    #parser.add_option("-o", action='store', type='string',
    #                  dest="outputpath",
    #                  help="pass the path to a csv file to write to "
    #                       "as an argument")
    #parser.add_option("-p", action="store_true", dest="plot",
    #                  help='plot')
    #parser.add_option("--all", action="store_true", dest="all",
    #                  help='plots all wavelength regions')
    #parser.add_option("--best", action="store_true", dest="best",
    #                  help='plots the reference, modified reference and residuals '
    #                       'for each of the best matches.')
    #parser.add_option("--chi", action="store_true", dest="chi",
    #                  help='plot the chi-squared surface from the pairwise \
    #                  matching procedure')
    #parser.add_option("--ref", action="store_true", dest="ref",
    #                  help='plot the locations of the best references used in the \
    #                  linear combination step')
    #parser.add_option("--sr", action="store_true", dest="ref",
    #                  help='save the residuals ')
    #
    #(options, sys.argv) = parser.parse_args(sys.argv)

    # if no path given in command, prompt user for a path to a file or a directory
    # from which to acquire fits files
    # NOTE: program currently only known to work if all the files in the directory
    # are fits files and are intended targets

    #ADZ 6/29/20: removed this if statement; pathname is given when this function is called.
    #if (options.pathname == None):
    #path_name = input('Please enter the path to the FITS file(s) of a star: ') 
    #print()
    #else:
    #    path_name = options.pathname
    
    # initialize this here in case error occurs before it is assigned a truth value
    iodine_flag = np.nan

    try:
        filenames = [f for f in listdir(path_name) if isfile(join(path_name, f))]
    except NotADirectoryError: # path to one file
        path_split = path_name.split('/')
        path_split[:-1] = ['/'.join(path_split[:-1])]
        filenames = []
        filename = path_split[-1]
        filenames.append(filename)
        path_name = path_split[0]

    # ADZ: remove test below 7/8/20 -> See function check2 in check_file_labeling.ipynb
    #                                (I tested that all files in each dir are for same star in that func.)
    
    # check to see if files are for the same star
    # NOTE: program currently does not work if the input directory contains fits
    # files for multiple stars
    #names = set() #ADZ comment out 
    #for filename in filenames:
    #    file = pf.open(path_name + '/' + filename,ignore_missing_end=True) #ADZ ADD ignore_missing_end=True
    #    header = file[0].header
    #    name = header['TOBJECT']
    #    names.add(name)
    #    if (len(names) > 1):
    #        print('Spectra Addition Error: ')
    #        print('This program sums the spectra for a star.')
    #        print('Please only provide the path to FITS files for the same star' +
    #              ' for a run of this program.')
     #       sys.exit()

    #display_plots = False
    #if (options.plot or options.chi or options.best or options.ref):
    #   display_plots = True # bool var for whether or not to display plots

    # Prompt for regions to plot
    if ((display_plots) and (options.all == None)):
        print("0 : 5000 to 5100 Å")
        print("1 : 5100 to 5200 Å")
        print("2 : 5200 to 5300 Å")
        print("3 : 5300 to 5400 Å")
        print("4 : 5400 to 5500 Å")
        print("5 : 5500 to 5600 Å")
        print("6 : 5600 to 5700 Å")
        print("7 : 5700 to 5800 Å")
        print("8 : 5800 to 5900 Å")
        print("Please enter the corresponding numbers for " +
              "the wavelength regions to be plotted.")
        print("Separate the numbers with spaces.")
        print("Default option is only region 1. Simply press enter for " +
              "default option.")
        print("Enter \'all\' to plot all the regions.")
        
        while(True):
            inp = input('→ ')
            try:
                if (inp == ''): # Default - plot region 1
                    regions = [1]
                elif (inp == 'all'):
                    regions = (list(range(9)))
                else:
                    regions = [int(region) for region in sort(inp.split(" "))]
                    if (False in [(0 <= region <= 8) for region in regions]):
                        continue
            except ValueError: continue
            break

    else: # plot all
        regions = (list(range(9)))    
    
    # Read in data from wavelength solution
    wave_soln = (pf.open('apf_wave_2022.fits'))[0].data

    # Sum all of the data files for a star
    data = np.zeros((79, 4608))

    ve = False
    counter  = 0
    iodine_flag = 0
    for filename in filenames:
        file = pf.open(path_name + '/' + filename)
        data_part = file[0].data
        # check for iodine in the spectrum, and skip this star if there is iodine
        if file[0].header['ICELNAM'] == 'In':
            iodine_flag = True
            SNR = np.nan
            raise Iodine_Exception('At least one target spectrum file contains iodine.')
        if counter == 0: #ADZ 7/26/20: get the header from the first file for this star, to use for the residual fits file 
            use_header = file[0].header
        counter += 1
        if (str(np.shape(data_part)) != '(79, 4608)'):
            print(str(np.shape(data_part)) + ' is an incompatible data shape.')
            print('Cannot perform shift-and-match process.')
            sys.exit()
        try:
            data += data_part
        except ValueError:
            ve = True
    
    # calculate a single-to-noise-ratio for the star
    SNR = np.sqrt(np.median(data[45,:])) # we assume the errors in flux are Poisson, so that sqrt(flux) gives the SNR  
    
    if (ve):
        print("Value Error occurred during spectra summation.")

    header = file[0].header
    #name = header['TOBJECT']
    print('Running SpecMatch-Emp on ' + sim_name + ':')
    for filename in filenames:
        print(filename)  
       
   
    #ve = False
    #Deblaze the orders: 31 to 52 (range of focus in the SM-Emp library)
    peak_scale_factors = np.zeros(22)*np.nan
    baseline_num_ph = np.zeros(np.shape(data))
    scale_factors = np.zeros(np.shape(data))
    data_no_deblaze = np.copy(data)
    for order_inc in range(22):
        #try: #ADZ 7/17/20:  use B-star deblaze instead of afs_deblaze
           # data[30 + order_inc, :4607] = afs_deblaze(data[30 + order_inc],
           #                                           30 + order_inc)[0]
        
        # prepare data in expected format for AFS algorithm 
        #order_data_AFS = pd.DataFrame(np.vstack([wave_soln[30 + order_inc][:4600], data[30 + order_inc][:4600]]).T, columns = ["wv","intens"])  
        #data[30 + order_inc, :4600] = AFS(order_data_AFS) 
        #data[30 + order_inc, :4600], current_blaze = zoe_percentile_deblazed(data, 30 + order_inc)
        data[30 + order_inc, :4600], peak_scale_factors[order_inc], scale_factors[30 + order_inc, :4600], baseline_num_ph[30 + order_inc, :4600] = bstar_deblazed2(data, 30 + order_inc)
        #except ValueError: ve = True

    if (ve): print("Value Error occurred during blaze correction.")
        
        
    #if save_peak_scale_factors:
    #    scale_factor_file = open(scale_factor_path + '/peak_scale_factors_' + sim_name, "w")
    #    for i in range(len(peak_scale_factors)):
    #        element = str(peak_scale_factors[i])
    #        if i < len(peak_scale_factors) - 1:
    #            scale_factor_file.write(element + ", ")
    #        elif i == len(peak_scale_factors) - 1:
    #            scale_factor_file.write(element)
    #    scale_factor_file.close()
    
    #ADZ: option to remove cosmic rays (simplistically) from normalized, deblazed spectrum 
    #     NOTE: set to FALSE when running this for results other than calibration
    remove_cosmic_rays = False
    def remove_cosmic_rays(spect): # must input a normalized, deblazed spectrum
        new_spect = spect 
        for i in range(len(spect)):
            old_value = spect[i]
            if old_value > 1.4:
                new_value = np.median(spect[i-3:i+3])
                new_spect[i] = new_value 
                print('replaced value ' + str(old_value) + ' with '+ str(new_value) +' at ' + str(i))
        return new_spect
    if remove_cosmic_rays == True:
        data = remove_cosmic_rays(data)   
    
    # Get a wavelength solution rescaled onto the scale of the library
    wave_soln_ref = get_rescaled_wave_soln()

    # Resample the spectrum onto the new wavelength scale
    #data_new = resample(wave_soln_ref, wave_soln, data)
    data_new = new_resample(wave_soln_ref, wave_soln, data)
    
    # Resample the array of scale factors
    resampled_scale_factors = new_resample(wave_soln_ref, wave_soln, scale_factors)
    
    # resample the array of baseline absolute (num photons) fluxes
    resampled_baseline_num_ph = new_resample_baseline(wave_soln_ref, wave_soln, baseline_num_ph)
    
    # resample the raw photon values without deblazing
    resamp_no_deblaze = new_resample_baseline(wave_soln_ref, wave_soln, data_no_deblaze)
    
    # save array of scale factors
    #if save_scale_factors:
    #    all_scale_factor_path = scale_factor_path + '/scale_factors_' + sim_name + '.csv'
    #    scale_factors_df = pd.DataFrame(resampled_scale_factors)
    #    scale_factors_df.to_csv(all_scale_factor_path, index = False)
        
    # save array of abolsute baseline fluxes
    #if save_baselines:
    #    baselines_file = baseline_path + '/baseline_fluxes_' + sim_name + '.csv'
    #    baselines_df = pd.DataFrame(resampled_baseline_num_ph)
    #    baselines_df.to_csv(baselines_file, index = False)
        
    # save array of absolute fluxes values
    if save_photon_counts:
        photons_file = photon_counts_path + '/photon_counts_' + id_name + '.csv'
        photons_df = pd.DataFrame(resamp_no_deblaze)
        photons_df.to_csv(photons_file, index = False)

    # Create spectrum object
    my_spectrum = Spectrum(np.asarray(wave_soln_ref), np.asarray(data_new))
    my_spectrum.name = sim_name

    #lib = specmatchemp.library.read_hdf() ADZ 8/10/20 moved this to outer loop so can remove stars from library
    
    sm = SpecMatch(my_spectrum, lib)
    
    # save the unshifted spectrum to a fits file
    if save_unshifted:
        target = sm.target.s
        target_wl = sm.target.w   
        new_header = use_header
        new_header.set('NRDU', 'YES','Normalized, resampled, deblazed, unshifted')
        data_hdu = fits.PrimaryHDU(target, new_header) 
        wl_hdu = fits.ImageHDU(target_wl)
        hdu = fits.HDUList([data_hdu, wl_hdu])
        hdu.writeto(unshifted_out_path + id_name + '_NDRU.fits')  

    # Perform shift
    sm.shift()
    
    # save the shifted spectrum to a fits file
    if save_shifted:
        target = sm.target.s
        target_wl = sm.target.w   
        new_header = use_header
        new_header.set('NRDS', 'YES','Normalized, resampled, deblazed, shifted')
        data_hdu = fits.PrimaryHDU(target, new_header) 
        wl_hdu = fits.ImageHDU(target_wl)
        hdu = fits.HDUList([data_hdu, wl_hdu])
        hdu.writeto(shifted_out_path + id_name + '_NDRS.fits')  
    
    # produce cross-correlation plots
    #solar_reference = astropy.io.fits.open('./APF_spectra/HD10700/NDR.fits')[0].data # --- read in tau ceti spectrum --- #
    #M_dwarf_reference = astropy.io.fits.open('./APF_spectra/GJ699/NDR.fits')[0].data # --- read in M-dwarf spectrum --- #
    
    #solar_x_corr =  signal.correlate(sm.target.s, solar_reference)
    #mdwarf_x_corr =  signal.correlate(sm.target.s, M_dwarf_reference)
    

    # Perform match
    sm.match()

    # Perform lincomb
    # NOTE: detrend() is called within lincomb(),
    #       so after this sm.results() gives detrended and sm.results_nodetrend() gives non-detrended results.
    sm.lincomb()

    # Chi squared values of the best match
    chi_squares = []

    chi_squares.append(sm.match_results.iloc[0]['chi_squared_5000'])
    chi_squares.append(sm.match_results.iloc[0]['chi_squared_5100'])
    chi_squares.append(sm.match_results.iloc[0]['chi_squared_5200'])
    chi_squares.append(sm.match_results.iloc[0]['chi_squared_5300'])
    chi_squares.append(sm.match_results.iloc[0]['chi_squared_5400'])
    chi_squares.append(sm.match_results.iloc[0]['chi_squared_5500'])
    chi_squares.append(sm.match_results.iloc[0]['chi_squared_5600'])
    chi_squares.append(sm.match_results.iloc[0]['chi_squared_5700'])
    chi_squares.append(sm.match_results.iloc[0]['chi_squared_5800'])

    best_mean_chi_squared = np.mean(np.asarray(chi_squares))

    # Plot HR diagram
    fig1 = figure(figsize=(12, 10))
    sm.plot_references(verbose=True) # NOTE: ADZ modified plot_references to omit plotting the derived properties for an example matching region
                                     # (I think this section was copied directly from the SM-Emp quickstart page, but in that case it is actually the library properties
                                     # plotted under 'plot target onto HR digram' which isn't possible for any general star.)
   

    # plot target onto HR diagram
    axes = fig1.axes
    axes[0].plot(sm.results['Teff'], sm.results['radius'], '*', ms=15, color='red', label='Target')
    axes[1].plot(sm.results['Teff'], sm.results['radius'], '*', ms=15, color='red')
    axes[2].plot(sm.results['feh'], sm.results['radius'], '*', ms=15, color='red')
    axes[3].plot(sm.results['feh'], sm.results['radius'], '*', ms=15, color='red')
    axes[0].legend(numpoints=1, fontsize='small', loc='best')
    plt.savefig(properties_plots_path + 'stellar_properties_'+ id_name.replace('.','_'))
    #plt.savefig(plots_out_path +'Stellar_properties/stellar_properties_'+ name)
    fig1.show() 
    
    # Plot reference spectra and linear combinations
    fig2 = plt.figure(figsize=(12,6))
    sm.plot_lincomb()
    plt.savefig(spectra_plots_path + 'ref_and_lincomb_spectra_' + id_name.replace('.','_'))
    #plt.savefig(plots_out_path + 'Ref_lincomb_spectra/ref_and_lincomb_spectra_' + name)
    fig2.show()
    
    # method of plotting currently not used
    if (display_plots):
        # Plot figures
        fignum = 0
        for r in regions:
            if (options.chi):
                fignum += 1
                plt.figure(fignum)
                fig3 = pylab.gcf()
                fig3.canvas.set_window_title('Chi-Squared Surface: Region ' + str(r))
                sm.plot_chi_squared_surface(region=r, num_best=None)
            if (options.best):
                fignum += 1
                plt.figure(fignum)
                fig1 = pylab.gcf()
                fig1.canvas.set_window_title('Best Match Spectra: Region ' + str(r))
                sm.plot_best_match_spectra(region=r, wavlim='all', num_best=None)
            if (options.ref):
                fignum += 1
                plt.figure(fignum)
                fig2 = pylab.gcf()
                fig2.canvas.set_window_title('References: Region ' + str(r))
                sm.plot_references(region=r, num_best=None, verbose=True)
            fignum += 1
            plt.figure(fignum)
            fig = pylab.gcf()
            fig.canvas.set_window_title('Linear Combination: Region ' + str(r))
            sm.plot_lincomb(region=r, wavlim='all')
            plt.show()

        # Plot shift
        fignum += 1
        plt.figure(fignum, figsize=(10,5))
        fig4 = pylab.gcf()
        fig4.canvas.set_window_title('Shift')
        sm.target_unshifted.plot(normalize=True, plt_kw={'color':'forestgreen'},
                                 text='Target (unshifted)')
        sm.target.plot(offset=0.5, plt_kw={'color':'royalblue'},
                       text='Target: ' + sim_name + ' (shifted)')
        sm.shift_ref.plot(offset=1, plt_kw={'color':'firebrick'},
                          text='Reference: ' +sm.shift_ref.name)
        plt.xlim(5160,5190)
        plt.ylim(0,2.2)

        plt.show()

    # ADZ 7/10/20: 
    # return the (normalized, deblazed) target and the residual between the target spectrum and the linear combination of  
    # best matched spectra
    mt_lincomb = sm.lincomb_matches[0:9]
    residual_all_regions = np.zeros([0,0])
    target_all_regions = np.zeros([0,0])
    wl_all_regions = np.zeros([0,0])
    for n in range(9): 
        residual = mt_lincomb[n].target.s - mt_lincomb[n].modified.s # Changed this 7/27, was opposite order
        target = mt_lincomb[n].target.s
        residual_all_regions = np.append(residual_all_regions, residual)
        target_all_regions = np.append(target_all_regions, target)
        wl_all_regions = np.append(wl_all_regions, mt_lincomb[n].target.w) # Added 8/7, was calculating wl scale below
    
    #ADZ: create header-data-units to store output
    obs_name = filenames[0].split('.')[0] # letters corresponding to the set of observations used for this target
                                          # NOTE: assumes all spectra used are from single set of observations!
    use_header.set('RESID', 'YES','Residual output of Specmatch-emp (HDU 1)')
    use_header.set('SPECT', str(filenames).replace(']','').replace('[','').replace('\'','') ,'Spectra files used for target spectrum')
    use_header.set('NDR', 'YES','Normalized, deblazed, registered spctrm (HDU 2)')
           
    print(str(sim_name) + ' COMPLETE') #ADZ 6/29/20: include name in statement
    return target_all_regions, residual_all_regions, wl_all_regions, use_header, obs_name, my_spectrum, sm, best_mean_chi_squared, args, SNR, iodine_flag, peak_scale_factors


# ------------------------ Driver code to run SM-Emp and isoclassify -------------------- #

# options -- for running as Juptyer notebook
#get_only_NDR = False # only compute the normalized, deblazed, registered spectra (NOT YET IMPLEMENTED -- use get_all_NDR_corrected.ipynb)
#get_properties = True # run SM-Emp and output the stellar property results 
#get_resids = True # run SM-Emp and output the residuals
#run_iso = True # run SM-Emp and run isochrone analysis to determine additional stellar properties
#save_SM_object = False # to save the Specmatch objects themselves
#display_plots = False # to make plots
#residuals_out_path = 'APF_spectra/NDRR_calib/' # directory to save residuals to 
#properties_out_path = 'SM_stellar_properties/' # directory to save steller property results to
#results_filename = 'specmatch_results_calib.csv' # filename for stellar property results within properties_out_path dir
#plots_out_path =  'SM_output_plots_calib/' # directory to save plots to
#properties_plots_path = plots_out_path + 'Stellar_properties/' # directory for property plots
#spectra_plots_path = plots_out_path + 'Ref_lincomb_spectra/' # directory for spectra plots

# rename existing output files
result_path = properties_out_path + results_subdir + results_filename
if make_new:
    dt = datetime.datetime.now()
    timestamp = dt.strftime("%d") + dt.strftime("%b") + dt.strftime("%Y") + '-' + dt.strftime("%X")
    if os.path.isdir(residuals_out_path):
        os.rename(residuals_out_path, residuals_out_path[:-1] + '_' + timestamp)
    if os.path.isfile(result_path):
        os.rename(result_path, result_path.split('.')[0] + '_' + timestamp + '.csv')  
    if os.path.isdir(properties_plots_path):
        os.rename(properties_plots_path, properties_plots_path[:-1] + '_' + timestamp)
    if os.path.isdir(spectra_plots_path):    
        os.rename(spectra_plots_path, spectra_plots_path[:-1] + '_' + timestamp)

    # create output directories
    os.mkdir(residuals_out_path)
    if not os.path.isdir(plots_out_path): os.mkdir(plots_out_path)
    os.mkdir(properties_plots_path)
    os.mkdir(spectra_plots_path)
    
if not make_new: # add to the old directories, but make them if they don't already exist
    if not os.path.isdir(residuals_out_path): os.mkdir(residuals_out_path)
    if not os.path.isdir(plots_out_path): os.mkdir(plots_out_path)
    if not os.path.isdir(properties_plots_path): os.mkdir(properties_plots_path)
    if not os.path.isdir(spectra_plots_path): os.mkdir(spectra_plots_path)


#  Get spectra filelist
#path_to_dir = input('Enter the path to the directory that contains the spectra: ') 
filelist = os.listdir(path_to_dir)
# for running on a subset of the files

try:
    filelist.remove('.ipynb_checkpoints') # remove hidden file in this directory
    filelist.remove('HIP5643_spectra') # remove problematic spectrum; produces an error but not due to labeling (GJ54.1)
except ValueError:
    pass

filelist = filelist[start:stop]
# note if more than one spectra for a star, place in a subdirectory. 

    
# for restarting a run if it has stopped partway through 
#number_run = 40 # the number already successfully completed
#filelist = filelist[number_run:]
#filelist = filelist[39:42]

# get wavelengths for residuals 
#wl_regions = [[5000,5100],[5101,5200],[5201,5300],[5301,5400],[5401,5500],[5501,5600],[5601,5700],[5701,5800],[5801,5900]]
#region_lens = [5835, 5722, 5612, 5508, 5406, 5309, 5216, 5124, 5037]
#wl_all_regions = np.zeros([0,0])
#for n in range(9): 
#    wl = np.linspace(wl_regions[n][0], wl_regions[n][1], region_lens[n])
#    wl_all_regions = np.append(wl_all_regions, wl) 

#lib = specmatchemp.library.read_hdf() # Moved below so can remove stars and replace after running each star

# Run Specmatch-emp and save results, normalized, deblazed, registered target, and residuals
nameslist = []
#pixel_shifts = []
empty_dirs = []

apf_name_conversion = pd.read_csv('apf_name_conversion_updated.csv')
apf_log_file = pd.read_csv('./apf_log_full_current.csv')
# for each star
for filename in filelist: 
    try:
        print(filename)

        # add a warnings field to save with results
        warnings = []

        # add a flag for failure of isochrone analysis, binary stars, and galaxies
        iso_fail_flag = False
        non_stellar_flag = False
        binary_flag = False

        # get list of filenames used for this star for logging in the results file    
        path_name = str(path_to_dir) + '/' + filename
        try:
            filenames = [f for f in listdir(path_name) if isfile(join(path_name, f))]
        except NotADirectoryError: # path to one file
            path_split = path_name.split('/')
            filenames = path_split[-1]

        # get star name from filename
        if os.path.isdir(path_to_dir + '/' + filename): #filename.endswith('_spectra'): # is a directory of spectra
            sim_name = filename.split('_')[0] #filename.replace('_spectra', '') # read simbad resolvable name from directory name
            # define name for saving shifted spectrum, residual, plots, etc to distinguish files in the case that there are multiple
            # subdirectories for the same star in this directory (NOTE: in that case must follow naming structure star_spectra_number
            # for each subdirectory)
            if filename.endswith('spectra'):
                id_name = sim_name
            elif filename[-1].isdigit():
                id_name = sim_name + '_' + filename.split('_')[-1]
            #if name == 'etaCrv':
            #    sim_name = 'eta Crv'
            #elif name == 'epsCep':
            #    sim_name = 'eps Cep'
            #elif name == 'bTau':
            #    sim_name = 'b Tau'
            #else:
            #    sim_name = name # name as resolvable by Simbad
            row = apf_log_file[apf_log_file['Simbad_resolvable_name'] == sim_name]
            alt_names = row['Alt_names'].tolist()[0]
            alt_names = alt_names.strip('][').replace('\'', '').split(', ') # list of all alternative names as listed in Simbad
            HIP_name = row['HIP_name'].values.tolist()[0]
            non_stellar_flag = row['Known_non-stellar'].all()
            binary_flag = row['Known_binary'].all()
            main_type = row['Main_type'].to_numpy()[0]
            if main_type == 'SB*': # spectroscopic binary
                binary_flag = True
            if len(os.listdir(path_to_dir + '/' + filename)) < 1:
                print('Skipping ' + filename + ' due to empty directory.')
                empty_dirs += [filename]
                continue 
        elif filename.endswith('fits'): # is a single spectrum -- really should just never give it a single spectrum that isn't enclosed in it's own directory 
            warnings += ['Please enclose single file in a directory with naming convention: name_spectra, where name is Simbad resolvable.']
            try:
                row = apf_log_file[apf_log_file['Filename'] == (filename.split('.')[0] + '.' + filename.split('.')[1] + '.fits')]
                HIP_name = row['HIP_name'].values.tolist()[0]
                if HIP_name == 'None': raise HIP_name_Exception('HIP_name \'None\' in APF log')
            except (IndexError, HIP_name_Exception): # File is not in BL APF database log file or HIP name is None
                row = apf_name_conversion[apf_name_conversion['FILENAME'] == (filename.split('.')[0] + '.' + filename.split('.')[1])]
                HIP_name = row['HIP_NAME'].values.tolist()[0]
            sim_name = HIP_name
            id_name = sim_name

        if binary_flag:
            Teff_bounds_flag = 3
            R_bounds_flag = 3
            SNR = np.nan
            iodine_flag = np.nan
            raise Binary_Exception('This star is a known (spectroscopic) binary.')
        if non_stellar_flag:
            Teff_bounds_flag = 3
            R_bounds_flag = 3
            SNR = np.nan
            iodine_flag = np.nan
            raise Non_Stellar_Exception('This is a known non-stellar object.')


        # get the Gaia and 2MASS (for later isochrone analysis) names, and get certain Gaia properties
        try:  
            run_iso_this_iter = run_iso 
            #gaia_source_id, id_2MASS = get_names(sim_name)
            try: 
                gaia_source_id = [name for name in alt_names if name.startswith('Gaia DR2')][0].split(' ')[-1]
            except IndexError:
                warnings += ['Failed to find Gaia name in Simbad; cannot run isochrone analysis.']
                run_iso_this_iter = False  
                iso_fail_flag = True
            try: 
                id_2MASS = [name for name in alt_names if name.startswith('2MASS')][0].split('J')[-1]
            except IndexError:
                warnings += ['Failed to find 2MASS name in Simbad; cannot run isochrone analysis.']
                run_iso_this_iter = False  
                iso_fail_flag = True
            # get the gaia values
            if os.path.exists('gaia_values.csv'): 
                gaia_values = pd.read_csv('gaia_values.csv')
                if gaia_values['source_id'].isin([gaia_source_id]).any(): # if this star already exists in the gaia csv file
                    gaia_data = gaia_values.loc[gaia_values['source_id'] == int(gaia_source_id)]
                else: # query gaia for the values and add this star to the gaia csv file
                    gaia_data = query_gaia_data(gaia_source_id).to_pandas()
                    gaia_data.to_csv('gaia_values.csv', mode='a', header=False)
            else: # create the csv file, query gaia for the values and add this star to the gaia csv file
                gaia_data = query_gaia_data(gaia_source_id).to_pandas()
                gaia_data.to_csv('gaia_values.csv', mode='w')

            # read in Gaia properties (for later isochrone analysis)
            if gaia_data.empty:
                warnings += ['Gaia returned empty properties array.']
                run_iso_this_iter = False  
                iso_fail_flag = True
            parallax = np.nanmedian(gaia_data['parallax']) #[mas --> milliarsec]
            u_parallax = np.nanmedian(gaia_data['parallax_error']) #[mas]
            if np.isnan(parallax) and not gaia_data.empty: 
                warnings += ['Gaia returned Nan parallax'] # can return Nan value even if returns other values successfully   
                run_iso_this_iter = False  
                iso_fail_flag = True
            ra = np.nanmedian(gaia_data['ra']) #[deg]
            if np.isnan(ra) and not gaia_data.empty: 
                warnings += ['Gaia returned Nan RA']
                run_iso_this_iter = False  
                iso_fail_flag = True
            dec = np.nanmedian(gaia_data['dec'])  #[deg]
            if np.isnan(dec) and not gaia_data.empty: 
                warnings += ['Gaia returned Nan DEC']
                run_iso_this_iter = False  
                iso_fail_flag = True
            #run_iso_this_iter = run_iso
        except (AttributeError, ValueError, NameError): # the name was not resolved in Simbad, for instance
            warnings += ['Failed to query Gaia properties; cannot run isochrone analysis.']
            run_iso_this_iter = False  
            iso_fail_flag = True
            print('Failed to query Gaia catalog.')


        # get the 2MASS photometry
        try:
            query2 = "SELECT designation,ra,dec,k_m,h_m,j_m,k_msigcom,h_msigcom,j_msigcom FROM fp_psc WHERE designation = '" + str(id_2MASS) + "'"
            service = pyvo.dal.TAPService('https://irsa.ipac.caltech.edu/TAP')
            phot_results = service.run_async(query2)
        except Exception:
            warnings += ['Failed to acquire 2MASS photometry; cannot run isochrone analysis.']
            run_iso_this_iter = False  
            iso_fail_flag = True
            print('Failed to query for 2MASS photometry.')
            pass

        # flag stars outside library bounds in Teff and R (becuase these won't produce great matches)
        try:
            #gaia_Teff = float(gaia_data[gaia_data['HIP_name'] == HIP_name]['teff_val'])
            gaia_Teff = np.nanmedian(gaia_data['teff_val'])
            Teff_bounds_flag = int((gaia_Teff < 3056) or (gaia_Teff > 6738)) # 1 if outside library bounds in Teff
            if np.isnan(gaia_Teff):
                Teff_bounds_flag = 3 # 3 == could not find Teff value for star in Gaia (b/c this star is missing Teff)
        except Exception:
            Teff_bounds_flag = 3 # 3 == could not find Teff value for star in Gaia (b/c this star is not in the Gaia list)
        try:
            gaia_R = np.nanmedian(gaia_data['radius_val'])
            R_bounds_flag = int((gaia_R < 0.168) or (gaia_R > 15.781)) # 1 if outside library bounds in radius
            if np.isnan(gaia_R):
                R_bounds_flag = 3 # 3 == could not find Teff value for star in Gaia (b/c this star is missing Teff)
        except Exception:
            R_bounds_flag = 3 # 3 == could not find radius value for star in Gaia

        # Remove star from library (for use in calibration run)
        lib = library.read_hdf()
        idx_GL570B = lib.get_index('GL570B') # remove this one as it is in error according to S. Yee
        lib.pop(idx_GL570B)
        lib_names = lib.library_params['cps_name'].to_list()
        #result_table = Simbad.query_objectids(HIP_name)
        #alt_names = result_table.to_pandas()
        #alt_names = alt_names.iloc[:,0].str.decode('utf-8') # gets rid of weird formatting
        if HIP_name == 'HIP80824': # This is the only GJ star (that we are running) that isn't listed in library as GL
            lib_name = 'GJ628'
        else: # get the library name
            lib_name = [name.replace(' ','').replace('HD','').replace('GJ','GL') for name in alt_names if name.replace(' ','').replace('HD','').replace('GJ','GL') in lib_names]
        idx = lib.get_index(lib_name) # get the idx in the library
        if idx == []:
            print('Could not find star ' + HIP_name + ' in catalog in order to remove.')
        else:
            star = lib.pop(idx) # remove star from library
            print('For ' + sim_name + ', removing corresponding star: ' + star[0]['cps_name'])
    
        # run Specmatch!
        if get_only_NDR:
            # TO BE IMPLEMENTED (10/15/21). For now use the script get_all_NDR_corrected.ipynb for this purpose.
            print('\'get_only_NDR\' option not yet implemented. For now use the script get_all_NDR_corrected.ipynb for this purpose.')
            pass
            if not(get_resids) and not(run_SMEMP) and not(run_iso_this_iter):
                continue

        if get_properties or get_resids or run_iso_this_iter:       
            # run SM-Emp
            star_target, star_residual, wl_scale, use_header, obs_name, my_spectrum, sm, best_mean_chi_squared, args, SNR, iodine_flag, peak_scale_factors = run_specmatch(sim_name, id_name, str(path_to_dir) + '/' + filename, lib, display_plots) # Run Specmatch on each star  
            
        # save residual to fits file
        if get_resids:
            target_hdu = fits.PrimaryHDU(star_target, use_header)
            resid_hdu = fits.ImageHDU(star_residual)
            wl_hdu = fits.ImageHDU(wl_scale)
            hdu = fits.HDUList([target_hdu, resid_hdu, wl_hdu])
            hdu.writeto(residuals_out_path + id_name + '_NDRR.fits')  # change 4/1/22 from sim_name

        # perform the isochrone analysis to determine better logg, mass, and age values
        if run_iso_this_iter:
            try: 
                print('Running isoclassify on ' + sim_name)
                iso_results = run_isoclassify(sm, gaia_data, phot_results)
                print('Isoclassify successful.')
            except Exception:
                print('Isoclassify failed!')
                iso_fail_type, value, traceback = sys.exc_info()
                iso_fail_flag = True # Isoclassify failed to run on this star
                iso_fail_message = str(iso_fail_type).split('\'')[1] + ': ' + str(value)
                warnings += [iso_fail_message]
                # set the results to Nan
                iso_results = classify_grid.resdata() 
                iso_results.rad = np.nan
                iso_results.teff = np.nan
                iso_results.feh = np.nan
                iso_results.logg = np.nan
                iso_results.mass = np.nan
                iso_results.age = np.nan
        else: 
            # set the results to Nan as placeholders
            iso_results = classify_grid.resdata() 
            iso_results.rad = np.nan
            iso_results.teff = np.nan
            iso_results.feh = np.nan
            iso_results.logg = np.nan
            iso_results.mass = np.nan
            iso_results.age = np.nan

        # Record results to csv file
        # if not provided an output file name, specmatch_results.csv is written to
        # (and created if not already)
        #if (options.outputpath != None):
        #    if (not options.outputpath.endswith('.csv')):
        #        if (not options.outputpath.endswith('/')):
        #            options.outputpath += '/'
        #        options.outputpath += 'specmatch_results.csv'
        #    if isfile(options.outputpath):
        #        with open(options.outputpath,'a') as fd:
        #            write_results(fd, my_spectrum, sm, iso_results, HIP_name, filenames, Teff_bounds_flag, R_bounds_flag, SNR, iodine_flag)
        #    else:
        #        with open(options.outputpath, 'w', newline='') as fd:
        #            write_results(fd, my_spectrum, sm, iso_results, HIP_name, filenames, Teff_bounds_flag, R_bounds_flag, SNR, iodine_flag, write_new = True)
        #else:
        if get_properties:
            if isfile(result_path):
                with open(result_path,'a') as fd: 
                    #with open(properties_out_path + 'specmatch_results_detrended_test.csv','a') as fd_detrended:
                    write_results(fd, my_spectrum, sm, iso_results, id_name, sim_name, HIP_name, filenames, Teff_bounds_flag, R_bounds_flag, SNR, iodine_flag, iso_fail_flag, warnings) 
            else:
                with open(result_path, 'w', newline='') as fd:
                   # with open(properties_out_path + 'specmatch_results_detrended_test.csv', 'w', newline='') as fd_detrended:
                   write_results(fd, my_spectrum, sm, iso_results, id_name, sim_name, HIP_name, filenames, Teff_bounds_flag, R_bounds_flag, SNR, iodine_flag, iso_fail_flag, warnings, write_new = True)

        # find the (approximate) shift used during shifting 
        # NOTE: this is currently not saved anywhere, and has not yet been shown to provide the correct shfit value! 
        #lags_dir, lags_file, lags_path, save_shifts
        #pixel_shifts = sm.shift_data['lag']
        #center_pix = sm.shift_data['center_pix']
        #shifted_wl_values = sm.shift_data['shifted_wl_values']
        #wl_deltas = sm.shift_data['wl_deltas']
        #if save_shifts:
        #    if isfile(lags_path):
        #        with open(result_path,'a') as fd: 
        #            #write_lags(sim_name, HIP_name, pixel_shifts, center_pix, fd)
        #            write_lags(sim_name, HIP_name, wl_deltas, shifted_wl_values, fd)
        #    else:
        #        with open(lags_path, 'w', newline='') as fd:
        #            #write_lags(sim_name, HIP_name, pixel_shifts, center_pix, fd, write_new = True)
        #            write_lags(sim_name, HIP_name, wl_deltas, shifted_wl_values, fd, write_new = True)
                    
        # WRITE THIS TO A CSV FILE, ONE LINE PER STAR WITH THE SHIFTS AND TIMESTAMP, NEW CSV FOR EACH RUN
        #pixel_shifts = pixel_shifts + [pixel_shift]

        # Can save the entire SpecMatch object using: 
        if save_SM_object:
            save_sm_path = './Specmatch_objects/' + HIP_name + '.hdf'
            sm.to_hdf(save_sm_path)

    except Exception:
        print('SpecMatch-Emp failed!')
        fail_type, value, traceback = sys.exc_info()
        fail_code = 1 # SM-Emp failed to run on this star
        fail_message = str(fail_type).split('\'')[1] + ': ' + str(value)
        if 'Iodine_Exception' in fail_message:
            SNR = np.nan
            iodine_flag = True
            
        #if (options.outputpath != None):
        #   if (not options.outputpath.endswith('.csv')):
        #        if (not options.outputpath.endswith('/')):
        #            options.outputpath += '/'
        #        options.outputpath += 'specmatch_results.csv'
        #    if isfile(options.outputpath):
        #        with open(options.outputpath,'a') as fd:
        #            write_results_fail_case(fd, HIP_name, filenames, Teff_bounds_flag, R_bounds_flag, fail_code, fail_message, SNR, iodine_flag)
        #    else:
        #        with open(options.outputpath, 'w', newline='') as fd:
        #            write_results_fail_case(fd, HIP_name, filenames, Teff_bounds_flag, R_bounds_flag, fail_code, fail_message, SNR, iodine_flag, write_new = True)
        #else: 
        if get_properties:
            if isfile(result_path):
                with open(result_path,'a') as fd: 
                    #with open(properties_out_path + 'specmatch_results_detrended_test.csv','a') as fd_detrended:
                    write_results_fail_case(fd, id_name, sim_name, HIP_name, filenames, Teff_bounds_flag, R_bounds_flag, fail_code, fail_message, SNR, iodine_flag, binary_flag, non_stellar_flag, warnings)
            else:
                with open(result_path, 'w', newline='') as fd:
                   # with open(properties_out_path + 'specmatch_results_detrended_test.csv', 'w', newline='') as fd_detrended:
                   write_results_fail_case(fd, id_name, sim_name, HIP_name, filenames, Teff_bounds_flag, R_bounds_flag, fail_code, fail_message, SNR, iodine_flag, binary_flag, non_stellar_flag, warnings, write_new = True)

        # save Nans for the shift values    
        #wl_deltas = [np.nan]
        #shifted_wl_values = [np.nan]
        #if save_shifts:
        #    if isfile(lags_path):
        #        with open(result_path,'a') as fd: 
        #            write_lags(sim_name, HIP_name, wl_deltas, shifted_wl_values, fd)
        #    else:
        #        with open(lags_path, 'w', newline='') as fd:
        #            write_lags(sim_name, HIP_name, wl_deltas, shifted_wl_values, fd, write_new = True)                   
                    
                    
                    
sys.stdout = old_stdout


# In[10]:


# Save pixel shifts, if desired
#HIP_names = pd.read_csv('Star_list.csv')['HIP_NAME'].to_list()
#pix_shift_array = np.array(pixel_shifts)
#df = pd.DataFrame(pix_shift_array, index = HIP_names, columns =['Pixel_shifts']) 
#df.to_csv('Pixel_shifts.csv')


# In[ ]:


#./APF_spectra/apf_spectra_highest_SNR/HIP93873_spectra
#./APF_spectra/apf_spectra_lite # for testing only
#./APF_spectra/apf_spectra_highest_SNR # for calibration set
#/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/apf_spectra_highest_SNR


# In[ ]:


#ranw.242.NDR.fits (GJ 244), ranw.314.NDR.fits (HIP69673), ranx.273.NDR.fits (HIP83207) and raqt.232.NDR.fits (HIP91262)


# # sandbox below here

# In[ ]:




