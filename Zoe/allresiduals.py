# Run the laser detection algorithm on all stars.
# See AllStars.ipynb for code documentation.

# INSTRUCTIONS
# Create a folder to store all results, and set variable folder_name to that folder.
# Determine which stars to run algorithm on by setting total_range.

import astropy.io.fits as fits
import matplotlib.pylab as plt
import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from scipy.interpolate import splev, splrep
import math
from mpmath import mp
import os
import os.path 
import random
from random import seed
from random import choice
from random import uniform
import mplcursors
import pandas as pd
from operator import itemgetter

from astropy import stats
import astropy


# for establishing threshold: how many sigmas to go above median
# n = 50
# gaussian_threshold = 0.07
# width_threshold = 3.2
# SNR_limit = 150


n = 15 # 25
gaussian_threshold = 0.5
width_threshold = 2.7
SNR_limit = -1

results_folder = '/mnt_home/zoek/code/APF-BL-DAP/Zoe/AnnaInjectionResults/'
directory = '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/NDRR_inj_narrow'

# directory = '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/NDRR_all_apf'

sp_results = pd.read_csv('/home/zoek/code/APF-BL-DAP/Zoe/specmatch_results.csv')

plt.rcParams["figure.figsize"] = (8,5)
plt.rcParams.update({'font.size': 15})
plt.rcParams. update({'font.family':'serif'}) 


mp.dps=100
exp_array = np.frompyfunc(mp.exp, 1, 1)

# function to insert simulated gaussians by adding a gaussian into a given location in the spectrum
def insert_gaussian(spectrum, gaussian_params, midpoint, numpoints):
    height = gaussian_params[0]
    position = gaussian_params[1] #position within segment, not index in spectrum
    FWHM = gaussian_params[2]
    offset = gaussian_params[3]
    x = np.linspace(0,numpoints-1,numpoints) # numpoints must be even
    width = FWHM/(2*np.sqrt(2*np.log(2)))    
    gauss = gaussian(x,height,position,width,offset)
    new_spect = spectrum.copy()
    new_spect[midpoint - math.floor(numpoints/2):midpoint + math.floor(numpoints/2)] = new_spect[midpoint - math.floor(numpoints/2):midpoint + math.floor(numpoints/2)] + gauss
    return new_spect

# same as above, but REMOVES the part of the data where the gaussian is inserted
def insert_gaussian_with_data_removal(spectrum, gaussian_params, midpoint, numpoints):
    height = gaussian_params[0]
    position = gaussian_params[1] #position within segment, not index in spectrum
    FWHM = gaussian_params[2]
    offset = gaussian_params[3]
    x = np.linspace(0,numpoints-1,numpoints) # numpoints must be even
    width = FWHM/(2*np.sqrt(2*np.log(2)))    
    gauss = gaussian(x,height,position,width,offset)
    new_spect = spectrum.copy()
    new_spect[midpoint - math.floor(numpoints/2):midpoint + math.floor(numpoints/2)] = gauss
    return new_spect
    
def gaussian(x,a,b,c,d): # a = height, b = position of peak, c = width, x = numpy array of x values
    f = a*exp_array((-(x-b)**2)/(2*c)**2) + d
    return f 

def chi(model, data):
    '''given two arrays of the same length, calculate reuduced chi-squared'''
#     return np.mean(((data - model) ** 2) / abs(model))
    chi_squared = sum(((data - model) / np.std(data)) ** 2)
    K = len(data) - 2
    return chi_squared / K

plot = False  # plt.show()
save_figs = True  # save figures into folders

# detected signals information
detected_widths = []
detected_heights = []
detected_indicies = []
detected_wavelengths = []
detected_max_flux_vals = []

star_indicies = [] # which star in the list the other values correspond to
threshold_vals = []

num_injections = 0
num_injections_above_threshold = 0
num_recoveries = 0

# table containing all the stars and detections
column_names = ['star', 'index', 'ndetections']
total_detections = pd.DataFrame(columns = column_names)

residuals_target_list= pd.read_csv('/mnt_home/zoek/code/APF-BL-DAP/Zoe/residual_target_list.csv')['Simbad_resolvable_name']


# every file in Anna's NDRR folder that is also in the residual target list
list_of_files = []
for filename in os.listdir(directory):
    if filename.endswith(".fits"): 
        file_path = os.path.join(directory, filename)
        file = fits.open(file_path)
        header = file[0].header
        star = header['OBJECT']
        if residuals_target_list.str.contains(star).any():
            list_of_files = np.append(list_of_files, file_path)
        
        
# create Laser Detection Results folder
# if save_figs == True:
#     path = '/mnt_home/zoek/code/APF-BL-DAP/Zoe/ResidualsResults/'
#     if not (os.path.isdir(path)):
#         os.mkdir(path)

# create Laser Detection Results folder
if save_figs == True:
    path = results_folder
    if not (os.path.isdir(path)):
        os.mkdir(path)
        
        
# # for which_star in np.array([732]): # 896
# for which_star in random.sample(np.arange(len(list_of_files)).tolist(), 50):

for which_star in np.arange(len(list_of_files)):

    print(which_star)
    
    # Get one star from list of all stars
    APF_flux_path = list_of_files[which_star]

    file = fits.open(APF_flux_path)
    flux = file[1].data
    wl = file[2].data

    header = file[0].header
    star = header['OBJECT']
    
    # get SNR
    
    SNR = 0
    if sp_results['Simbad_resolvable_name'].str.contains(star).any():
        x = sp_results[sp_results['name'] == star]
        SNR = float(x['SNR'])
    
    if SNR < SNR_limit:
        continue

#     print('File path: ' + APF_flux_path)
#     print('Star: ' + star)
        
    num_detections_this_star = 0
    # table containing info about this observation
    column_names = ['description', 'indicies']
    detections = pd.DataFrame(columns = column_names)

    spect = flux

    idxs1 = [] # indicies that are above continuum level
    idxs2 = [] # indicies in idxs1 that are above the threshold value and are local maximas
    idxs3 = [] # indicies in idxs2 that are gaussian-shaped
    idxs4 = [] # indicies in idxs3 that are greater than 5 pixels in width
    
    idxs4_heights = []


    
#     clipped = astropy.stats.sigma_clip(spect, sigma=5).filled()
#     abs_dev = stats.median_absolute_deviation(clipped)
#     med = np.median(clipped)

    abs_dev = stats.median_absolute_deviation(spect)
    med = np.median(spect)
    
    T = med + n * abs_dev
    threshold_vals = np.append(threshold_vals, T)
    
    for idx in np.arange(len(spect)):
        if spect[idx] > T:
            idxs2 = idxs2 + [idx]

    consecutive_indicies_idxs2 = []
    i = 0
    while i < (len(idxs2)):
        lst = [idxs2[i]]
        while (i < len(idxs2) - 1) and (idxs2[i + 1] == idxs2[i] + 1):
            lst = np.append(lst, idxs2[i+1])
            i += 1
        consecutive_indicies_idxs2 = consecutive_indicies_idxs2 + [lst]
        i +=1

    median_indicies = []
    for idxs in consecutive_indicies_idxs2:
        max_index = max(idxs, key=lambda i: spect[i])
        median_indicies = np.append(median_indicies, int(max_index))

    idxs2 = np.array(median_indicies, dtype=int)
    num_injections_above_threshold += len(idxs2)
    
    
    
    

    detection_plot = False

    for idx in idxs2:
        # fit a gaussian to the peak, see if the width is greater than or equal to 2 pixels
        # see how much signal resembles a gaussian
        # if last test is commented out, ALSO check FWHM of gaussian

        # DETERMINING EDGES OF SIGNAL: mark edge when flux reaches a local minimum
        # PRO: can identify two signals together
        # CON: can't deal with noise in signal

        temp_ind = idx
        left_bound = 0
        while temp_ind > 1:
            temp_ind -= 1
            if spect[temp_ind] < spect[temp_ind - 1] and spect[temp_ind] < spect[temp_ind + 1]:
                left_bound = temp_ind
                break
        temp_ind = idx
        right_bound = len(spect) - 1
        while temp_ind < len(spect) - 4:
            temp_ind += 1
            if (spect[temp_ind] < spect[temp_ind - 1]) and (spect[temp_ind] < spect[temp_ind + 1]):
                right_bound = temp_ind
                break

        x = wl[left_bound:right_bound + 2]
        y = spect[left_bound:right_bound + 2]

        # oversample detected signal to determine precise bounds on the edges of the signal
        # use this to determine the FWHM of the signal in pixels
        oversampled_x = np.linspace(x[0], x[-1], len(x) * 10)
        spl = splrep(x, y, k=2)
        oversampled_y = splev(oversampled_x, spl)

        max_y = max(oversampled_y)
        min_y = np.percentile(oversampled_y, 3) 
        height = max_y - min_y
        ind = oversampled_y.tolist().index(max_y)
        pos = oversampled_x[ind]
        min_width = 0.00001
        max_width = oversampled_x[len(oversampled_x) - 1] - oversampled_x[0]
        width_spacing = 0.001

        chi_squared_values = []
        width_vals = np.arange(min_width, max_width, width_spacing)
        
        # 2/6 change
        for w in width_vals:
            gaus = gaussian(oversampled_x, height, pos, w, min_y)
            gaus2 = gaussian(x, height, pos, w, min_y)
            
            chi_squared = chi(gaus2, y)
            chi_squared_values = np.append(chi_squared_values, chi_squared)
        min_chi_squared = min(chi_squared_values)
        
        ind_of_min_chisquared = chi_squared_values.tolist().index(min_chi_squared)
        width = width_vals[ind_of_min_chisquared]
        gaus = gaussian(oversampled_x, height, pos, width, min_y)
        gaus2 = gaussian(x, height, pos, width, min_y)

        width_threshold = False
        gauss_threshold = False

        # see if the signal fits a gaussian
        if min_chi_squared < gaussian_threshold:
            gauss_threshold = True
            idxs3 = idxs3 + [idx]

            # find the width of the gaussian in pixels

            peak = max(gaus)
            half_max = peak - height / 2

            peak_index = gaus.tolist().index(peak)
            temp_left_bound = peak_index
            temp_right_bound = peak_index

            while gaus[temp_left_bound] > half_max and temp_left_bound > 0:
                temp_left_bound -=1

            while gaus[temp_right_bound] > half_max and temp_right_bound < len(gaus) - 1:
                temp_right_bound += 1

            pixel_width = (temp_right_bound - temp_left_bound) / 10

            if pixel_width > width_threshold:
                width_threshold = True
                idxs4 = idxs4 + [idx]
                idxs4_heights = idxs4_heights + [height]
                num_detections_this_star += 1
        
        
            if plot == True or save_figs == True:

                fig = plt.figure()
                plt.step(x, y, label = 'Detected Signal at ' + str(round(wl[idx], 2)) + ' A')
#                 plt.plot(oversampled_x, gaus, label = 'Gaussian')
                plt.plot(x, gaus2, label = 'Gaussian')
                if width_threshold == True:
                    # passed width threshold AND gaussian threshold
                    plt.title('PASS: chi-squared of ' + str(round(min_chi_squared, 4)) + ' and pixel width of ' + str(pixel_width))
                elif gauss_threshold == True and width_threshold == False:
                    # failed width threshold
                    plt.title('FAIL: too narrow with pixel width of ' + str(pixel_width))
                    break
                else:
                    # failed gaussian threshold
                    plt.title('FAIL: not gaussian-shaped: chi-squared of ' + str(round(min_chi_squared, 4)))
                    break



                plt.xlabel('Wavelength [A]')
                plt.ylabel('Flux')
                for ind in np.arange(left_bound, right_bound):
                    plt.axvline(x=wl[ind], color='gray', linestyle='-', linewidth=0.2)
                plt.legend()
                if plot == True:
                    plt.show()
                if save_figs == True:

                    path = results_folder + star
                    if not (os.path.isdir(path)):
                        os.mkdir(path)

                    name = 'r_' + star + '_' + str(round(wl[idx], 2))
                    fig.savefig(path + '/' + name + '.png')

                if detection_plot == False:
                    detection_plot = True

                    fig = plt.figure()
                    plt.plot(wl, spect, label = 'Original Data')
                    if (len(idxs2) > 0):
                        for ind in idxs2:
                            plt.axvline(x=wl[ind], color='gray', linestyle='--')
                    plt.axhline(y=T, linestyle='--', label='Threshold')
                    plt.title('SNR: ' + str(round(SNR, 0)))
                    plt.suptitle(star)
                    plt.xlabel('Wavelength [A]')
                    plt.ylabel('Flux')
                    plt.legend() 
    #                 plt.ylim(-1, 2)
                    if plot == True:
                        plt.show()
                    if save_figs == True:
                        fig.savefig(path + '/' + 'r_data' + star + '.png')
    wavelengths = []
    for i in idxs4:
        w = wl[i]
        wavelengths += [w]

    new1 = {'description': ['indicies above threshold', 'indicies that are gaussian-shaped', 'indicies wider than PSF'],
            'indicies': [idxs2.tolist(), idxs3, idxs4], 'wavelengths': [[], [], wavelengths], 'heights': [[], [], idxs4_heights]}
    
    df1 = pd.DataFrame(new1)
    detections = detections.append(df1)
    name = str(which_star) + '_' + star
    if save_figs == True and detection_plot == True:
        detections.to_csv(path + '/' + 'r_' + name + '.csv')
    
    
    new2 = {'star': [star], 'index': [which_star], 'ndetections': [num_detections_this_star]}
    df2 = pd.DataFrame(new2)
    total_detections = total_detections.append(df2)
    
if save_figs == True:
    total_detections.to_csv(results_folder + 'r_results.csv')
else:
    print(total_detections)