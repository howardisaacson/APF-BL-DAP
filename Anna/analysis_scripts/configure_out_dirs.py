import argparse 
import os
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-resid_out","--residuals_out_path", type=str, help= "Directory to save residuals to (ie. 'NDRR_calib').") 
parser.add_argument("-results_out","--results_filename", type=str, help= "Filename to save steller property results to (ie. 'specmatch_results_calib.csv').") 
parser.add_argument("-results_type","--results_subdirectory", type=str, help= "What type of run is this. Must match a subdirectory name (all_apf, calibration, tests, or old).") 
parser.add_argument("-plots_out", "--plots_out_path", type=str, help="Directory to save plots to (ie. 'SM_output_plots_calib').")
parser.add_argument("-logfiles", "--logfile_directory", type=str, help="Directory to store log and output files (ie. 'logfiles').")
args = parser.parse_args()

residuals_out_path = 'APF_spectra/' + args.residuals_out_path + '/' #'/NDRR_calib/' # directory to save residuals to 
properties_out_path = 'SM_stellar_properties/' + args.results_subdirectory + '/' # directory to save steller property results to
results_filename = args.results_filename #'specmatch_results_calib.csv' # filename for stellar property results within properties_out_path dir
result_path = properties_out_path + results_filename
plots_out_path =  args.plots_out_path + '/' #'SM_output_plots_calib' # directory to save plots to
properties_plots_path = args.plots_out_path + '/Stellar_properties/' # directory for property plots
spectra_plots_path = args.plots_out_path + '/Ref_lincomb_spectra/' # directory for spectra plots
logfile_dir = args.logfile_directory


dt = datetime.datetime.now()
timestamp = dt.strftime("%d") + dt.strftime("%b") + dt.strftime("%Y") + '-' + dt.strftime("%X")
if os.path.isdir(residuals_out_path):
    os.rename(residuals_out_path, residuals_out_path[:-1] + '_' + timestamp)
if os.path.isfile(result_path):
    os.rename(result_path, result_path.split('.')[0] + '_' + timestamp + '.csv')  
#for file in os.listdir(properties_out_path):
#    if file.startswith(results_filename.split('.')[0]) and file[-5].isdigit() and (file[-7] != ':'):
#        os.rename(properties_out_path + file, properties_out_path + file.split('.')[0] + '_' + timestamp + '.csv')   
if os.path.isdir(properties_plots_path):
    os.rename(properties_plots_path, properties_plots_path[:-1] + '_' + timestamp)
if os.path.isdir(spectra_plots_path):    
    os.rename(spectra_plots_path, spectra_plots_path[:-1] + '_' + timestamp)
if os.path.isdir(logfile_dir):    
    os.rename(logfile_dir, logfile_dir + '_' + timestamp)


# create output directories
os.mkdir(residuals_out_path)
if not os.path.isdir(plots_out_path): os.mkdir(plots_out_path)
os.mkdir(properties_plots_path)
os.mkdir(spectra_plots_path)
os.mkdir(logfile_dir)
os.mkdir(logfile_dir + '/SM_outputs')
os.mkdir(logfile_dir + '/SM_logs')


