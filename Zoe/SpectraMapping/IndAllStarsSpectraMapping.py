# exact same as IndAllResultsSpectraMapping.ipynb, just in .py form


import astropy.io.fits as fits
import matplotlib.pylab as plt
import numpy as np
import glob
import pickle
import pandas as pd
import os

# directory to save files into 
directory = '/home/zoek/code/APF-BL-DAP/Zoe/6sEvents/'

# directory to read data from
repeated_events = '/home/zoek/code/APF-BL-DAP/Zoe/6sResults/repeated_events.csv'

# how many pixels in the x and y direction to include in the 2-D spectrum plot
y_pixels = 16
x_pixels = 40

APF_wavelength_path = '/mnt_home/zoek/code/APF-BL-DAP/Zoe/APFTutorial/apf_wav.fits'
wl_file = fits.open(APF_wavelength_path)
wl_image = wl_file[0].data

def find_location_of_closest_val(val, array, round_by = 2):
    '''given a wavelength value, find the indicies that give its location in wl_image
    returns order #, pixel #'''
    rounded_val = np.round(val, round_by)
    for sub_array in array:
        rounded_sub_array = np.round(sub_array, round_by)
        if rounded_val in rounded_sub_array:
            return(array.tolist().index(sub_array.tolist()), rounded_sub_array.tolist().index(rounded_val))
        
# read in dictionary and coefficient array 
# (coefficients of polynomials describing the curve of the 2-d spectra)
spect_dict = pd.read_pickle(r'/home/zoek/code/APF-BL-DAP/Zoe/SpectraMapping/spect2d.pkl')

text = open('/mnt_home/zoek/code/APF-BL-DAP/Zoe/SpectraMapping/order_coefficients.txt', "r")
lines = text.read().splitlines()
coeff_array = np.zeros((79,5))
for i in range(len(lines)):
    a0 = float(lines[i][6:13].strip())
    a1 = float(lines[i][17:26].strip())
    a2 = float(lines[i][27:39].strip())
    a3 = float(lines[i][40:52].strip())
    a4 = float(lines[i][54:].strip())
    coeffs_one_line = np.array([a0,a1,a2,a3,a4])
    coeff_array[i] += coeffs_one_line
    
repeated_events = pd.read_csv(repeated_events).drop(columns=['Unnamed: 0'])

for star in repeated_events['star']:
#     print(star)
    star_table = repeated_events[repeated_events['star'] == star]
    
    for i in np.arange(len(star_table)):
        wl = star_table.iloc[i, :]['r shifted wl']
        identifier = star_table.iloc[i, :]['identifier']
    
        save_dir = directory + star + '/' + str(np.round(wl, 2))
        if not os.path.isdir(directory + star + '/'):
            os.mkdir(directory + star + '/')

        result = find_location_of_closest_val(wl, wl_image, round_by = 2)
        if result == None:
            result = find_location_of_closest_val(wl, wl_image, round_by = 1)
        order, pixel = result



        # find corresponding 2-D spectrum
        if star in spect_dict:
            spect2d = spect_dict[star]
        else:
            break

        for subsubfile in spect2d:
            apf_2d = fits.open(subsubfile)
            image_2d = apf_2d[0].data
            image_rot = np.rot90(image_2d)
            image_flip = np.fliplr(image_rot)
            star = apf_2d[0].header['TOBJECT']

            a0 = coeff_array[order,0]
            a1 = coeff_array[order,1]
            a2 = coeff_array[order,2]
            a3 = coeff_array[order,3]
            a4 = coeff_array[order,4]

            y = a0 + a1*pixel + a2*pixel**2 + a3*pixel**3 + a4*pixel**4
            y = int(y)

            new_image = image_flip[y-(y_pixels//2):y+(y_pixels//2),pixel-(x_pixels//2):pixel+(x_pixels//2)] 
            # y coords, then x coords

            lower_bound = pixel - (x_pixels//2)
            if lower_bound < 0:
                lower_bound = 0

            upper_bound = pixel + (x_pixels//2)
            if upper_bound > 4606:
                upper_bound = 4606

            extent = [wl_image[order][lower_bound] - wl_image[order][pixel], 
              wl_image[order][upper_bound]- wl_image[order][pixel], 
              wl_image[order][lower_bound] - wl_image[order][pixel], 
              wl_image[order][upper_bound]- wl_image[order][pixel]]

            fig = plt.figure()
            plt.imshow(new_image, cmap = 'gray', 
                       vmin = np.median(new_image), 
                       vmax = np.max(new_image), origin = 'lower',
                      extent=extent)
            plt.xlabel('Wavelength [A] - ' + str((round(wl, 2))) + ' A')
            plt.title(star + ' at ' + str((round(wl, 2))) + ' A')
            spect_fname = subsubfile.split('/')[-1].split('.')[-2]
            ax = fig.gca()
            ax.get_yaxis().set_visible(False)
            
        
            pathname = save_dir + '/' + '2d_' + identifier + '.png'
            print(pathname)
            if os.path.isfile(pathname):
                os.remove(pathname)
            fig.savefig(pathname)
            plt.close(fig)