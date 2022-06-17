# imports
import pandas as pd
import numpy as np 
import astropy.io.fits as pf
import os
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from mpmath import mp
mp.dps=100
exp_array = np.frompyfunc(mp.exp, 1, 1)
from astroquery.simbad import Simbad
import sys
gaia_values = pd.read_csv('/mnt_home/azuckerman/BL_APF_DAP/gaia_values.csv')
velocity_shifts = pd.read_csv('Velocity_shift_inj_8.csv') #velocity_shifts = pd.read_csv('Velocity_shift_all_targets_updated.csv')
apf_wl_soln = pf.open('apf_wav.fits')[0].data
import import_ipynb
from rescale import get_rescaled_wave_soln
wave_soln_ref = get_rescaled_wave_soln()
import datetime
c = 2.99 * 10**5 # km/s

# define gaussian function for fitting
def gaussian(x,a,b,c,d): # a = height, b = position of peak, c = width, x = numpy array of x values
    f = a*exp_array((-(x-b)**2)/(2*c)**2) + d
    return f.astype(float)
     
def get_names(sim_name):
    # get Gaia and 2MASS names 
    result_table = Simbad.query_objectids(sim_name)
    names = result_table.to_pandas()['ID'].to_list()
    gaia_name = [name.decode('utf-8').split(' ')[-1] for name in names if name.decode('utf-8').startswith('Gaia DR2')]
    two_MASS_name = [name.decode('utf-8').split('J')[-1] for name in names if name.decode('utf-8').startswith('2MASS')]
    (gaia_source_id,) = gaia_name
    (id_2MASS,) = two_MASS_name
    return gaia_source_id, id_2MASS


# recover the peaks

#star_list = os.listdir('APFRpectra/all_apf_spectra_highest_SNR')[:100]
test = '8'
key = '3'
path = 'APF_spectra/NDRR_testing_inj_' + test                                 

star_list =  sorted([star.split('_')[0] for star in os.listdir(path)])
try: star_list.remove('.ipynb_checkpoints')
except: pass
#try: star_list.remove('GJ1002_spectra') # seems something went wrong with the injections for this star
#except: pass

injection_key_file = 'Injection_key_'  + key + '.csv'
injection_key = pd.read_csv(injection_key_file)
injected_stars = injection_key['Star_name']
expected_peak_photons_R = np.zeros(len(injected_stars)) * np.nan
expected_baseline_photons_R = np.zeros(len(injected_stars)) * np.nan
expected_wl_arr_R = np.zeros(len(injected_stars)) * np.nan
recovered_peak_photons_R = np.zeros(len(injected_stars)) * np.nan
recovered_peak_photons_gauss_R = np.zeros(len(injected_stars)) * np.nan
recovered_baseline_photons_R = np.zeros(len(injected_stars)) * np.nan
original_wl_arr_R = np.zeros(len(injected_stars)) * np.nan
gaia_Teff_arr =  np.zeros(len(injected_stars)) * np.nan
types = np.zeros(len(injected_stars)) * np.nan #[None] * len(injected_stars)
failed = []
num_inj = 3 # injections per star

n = 0 # total injections fit
for star in star_list: #[star_list[1]]: #['HIP108092']:# ['HIP25878', 'HIP46343', 'HIP46733']:#star_list:
    try:

        star = star.split('_')[0]
        print(star)

        # read in un-process injected spectra
        j = 0
        injected_dir = 'APF_spectra/injected_spectra_'  + key + '/'
        file_list = os.listdir(injected_dir + star + '_spectra') 
        try:
            file_list.remove('.ipynb_checkpoints')
        except:
            pass
        types[n:n+3] = 1 #'center'
        print('injected_spectra_' + key)
        hdu_injected = [None] * len(file_list)
        for file in file_list:
            if not '_3_injection' in file:
                print(file)
                hdu_injected[j] =  pf.open(injected_dir + star + '_spectra/' + file)
            j += 1

        # get gaia properties
        try:
            gaia_source_id, id_2MASS = get_names(star)
            gaia_row = gaia_values.loc[gaia_values['source_id'] == int(gaia_source_id)] # star has been run so should be here
            gaia_row = gaia_row.median(axis = 0, skipna = True)
            gaia_Teff_arr[n] = gaia_row['teff_val']
        except ValueError:
            gaia_Teff_arr[n] = np.nan


        if np.isnan(types[n]):
                print()
                print('star:' + star)   
                print('NO TYPE RECORDED FOR THIS STAR!')
                break


        hdu_shifted = pf.open('APF_spectra/NDRR_testing_inj_' + test + '/' + star + '_NDRR.fits')
        hdu_unshifted = pf.open('APF_spectra/NDRU_testing_inj_' + test + '/' + star + '_NDRU.fits')
        hdu_shifted_no_inj = pf.open('APF_spectra/NDRS_all_apf/' + star + '_NDRS.fits')
        #try:
        photon_counts = np.array(pd.read_csv('SM_photon_counts_all_injections/test_' + key + '/photon_counts_' + star + '.csv'))
        #except:
        #    photon_counts = np.array(pd.read_csv('SM_photon_counts/testing/photon_counts_' + star + '.csv'))
        shifted_no_inj  = hdu_shifted_no_inj[0].data
        wl_shifted_no_inj = hdu_shifted_no_inj[1].data
        shifted  = hdu_shifted[1].data
        wl_shifted = hdu_shifted[2].data
        unshifted  = hdu_unshifted[0].data
        wl_unshifted = hdu_unshifted[1].data
        shift_value = float(velocity_shifts[velocity_shifts['Star_name'] == star]['Velocity_shift [km/s]'])
        pixels_per_A = 1 / np.median(wl_unshifted[1:] - wl_unshifted[:-1])
        pixel_shift = pixels_per_A * shift_value

        expected_injections = injection_key[injection_key['Star_name'] == star]

        # plot the injected signals
        make_plots = False
        if make_plots:
            lw = np.linspace(4,1,len(file_list))
            ms = 3

        # plot the recovered signals
        #if make_plots:
        #    plt.figure(figsize = [20,5])   
        orders = [39,40,41]
        expected_wl = list(expected_injections['Injection_wavelength'])
        expected_locs = sorted(list(expected_injections['Injection_wavelength'] - expected_injections['Injection_wavelength'] * shift_value / c))
        for i in range(num_inj):  # for each injection
            order = orders[i]
            low_bound = apf_wl_soln[order, 0]#500#[38,0]
            up_bound = apf_wl_soln[order,-1]#-3000#[42,-1]
            idxs_order_shifted = (wl_shifted > low_bound) * (wl_shifted < up_bound)
            idxs_order_unshifted = (wl_unshifted > low_bound) * (wl_unshifted < up_bound)
            idxs_order_shifted_no_inj = (wl_shifted_no_inj > low_bound) * (wl_shifted_no_inj < up_bound)
            idxs_order_raw_phot = (wave_soln_ref > low_bound) * (wave_soln_ref < up_bound)
            if make_plots:
                #plt.subplot(1,3,i+1)
                plt.figure(figsize = [15,5]) 
                lw = 1    
                #plt.vlines([expected_locs[i]], ymin = np.min(shifted[idxs_order_shifted]),  ymax = 1.1 * np.max(shifted[idxs_order_shifted]), ls = 'dashed', alpha = 0.5, label = 'Expected injection locations')
                plt.plot(wl_shifted[idxs_order_shifted], shifted[idxs_order_shifted]/np.max(shifted[idxs_order_shifted]), '.-', ms = ms, lw = lw, label = 'Shifted spectrum')
                #plt.plot(wl_shifted_no_inj[idxs_order_shifted_no_inj], shifted_no_inj[idxs_order_shifted_no_inj], '.-', ms = ms, lw = lw, alpha = 0.3, label = 'Shifted spectrum (no injection)')
                plt.plot([],[], ' ', label = 'Velocity shift: ' + str(np.round(shift_value,2)) + ' km/s')
                #plt.xlabel('Wavelength [A]'); plt.ylabel('Intensity')


            # ----- Find the observed signal (for real signals this is done in Zoe's algorithm) -----#    
            popt, pcov = curve_fit(gaussian, wl_shifted[idxs_order_shifted], shifted[idxs_order_shifted], 
                                   bounds=([0.2, expected_locs[i] - 3, 0, 0.5], [5, expected_locs[i] + 3, 2, 1.5]))
            fit_height_norm = popt[0]
            fit_peak_pos = popt[1]
            fit_width = popt[2]
            fit_baseline_flux_norm = popt[3]
            # fit the photon counts array peak to a guassian to better estimate the recovered properties
            shifted_wl = fit_peak_pos + fit_peak_pos * shift_value / c
            original_loc_guess = np.absolute(wave_soln_ref - shifted_wl).argmin()
            initial_height_guess = photon_counts[original_loc_guess]


            # --------------------------- Recover the original signal -----------------------------# 
            popt2, pcov2 = curve_fit(gaussian, wl_unshifted[idxs_order_unshifted], photon_counts[idxs_order_raw_phot].flatten(), 
                                   bounds=([0.8 * initial_height_guess, shifted_wl - 0.5, 0, 0], [2 * initial_height_guess, shifted_wl + 0.5, np.inf, np.inf]))

            # do the guassian fitting again with a wwider window if the fit is essentially flat
            if popt2[0]  < 1.1 * popt2[3]:
                popt2, pcov2 = curve_fit(gaussian, wl_unshifted[idxs_order_unshifted], photon_counts[idxs_order_raw_phot].flatten(), 
                               bounds=([0.8 * initial_height_guess, shifted_wl - 2, 0, 0], [2 * initial_height_guess, shifted_wl + 2, np.inf, np.inf]))

            original_wl = popt2[1]
            original_wl_arr_R[n] = original_wl
            original_loc = np.absolute(wave_soln_ref - original_wl).argmin()
            recovered_baseline_photons_R[n] = np.median(np.hstack([photon_counts[original_loc - 100 : original_loc - 50], photon_counts[original_loc + 50 : original_loc + 100]]))

            recovered_baseline_photons_R[n] = np.median(np.hstack([photon_counts[original_loc - 100 : original_loc - 50], photon_counts[original_loc + 50 : original_loc + 100]]))
            prev_ord_edge = apf_wl_soln[order - 1, -1]
            next_ord_edge = apf_wl_soln[order + 1, 0]
            wl_50_pix = 50 / pixels_per_A

            # fit the baseline, dealing with order edges
            case_a = (np.abs(original_wl - prev_ord_edge) <= 0.5*wl_50_pix) and (original_wl < prev_ord_edge) # near, to left of previous order edge
            case_b = (np.abs(original_wl - next_ord_edge) <= 0.5*wl_50_pix) and (original_wl < prev_ord_edge) # near, to left of next order edge
            case_c = (np.abs(original_wl - prev_ord_edge) <= 0.5*wl_50_pix) and (original_wl > prev_ord_edge) # near, to right of previous order edge
            case_d = (np.abs(original_wl - next_ord_edge) <= 0.5*wl_50_pix) and (original_wl > prev_ord_edge) # near, to right of next order edge
            if case_a or case_b: 
                recovered_baseline_photons_R[n] = np.median(photon_counts[original_loc - 100 : original_loc - 50]) # only use the left side 
            if case_c or case_d: 
                recovered_baseline_photons_R[n] = np.median(photon_counts[original_loc + 50 : original_loc + 100]) # only use the right side        

            recovered_peak_photons_R[n] = photon_counts[original_loc] - recovered_baseline_photons_R[n] #fit_height_norm / float(scale)  

            if recovered_peak_photons_R[n] < 100:
                print()
                print('star:' + star)           
                print('shift:' + str(shift_value))
                print('T:' + str(gaia_Teff_arr[n]))

            #recovered_baseline_photons_R[n] = fit_baseline_flux_norm / float(scale)
            expected_peak_photons_R[n] = list(expected_injections['Injected_peak_photons'])[i]
            expected_baseline_photons_R[n] = list(expected_injections['Local_median_photons'])[i]
            expected_wl_arr_R[n] = expected_wl[i]

            #fractional_diff_arr[n] = (recovered_peak_photons_R[n]- expected_peak_photons_R[n])/expected_peak_photons_R[n]  

            verbosity = 0
            if verbosity > 0:#or abs(fractional_diff_arr[n]) > 0.5:
                print()
                print(star)
                print('expected peak position: ' + str(expected_wl[i]))
                print('fit peak position: ' + str(fit_peak_pos)) 
                print('recovered peak position (fit - shift): ' + str(original_wl)) 
                print('wl shift: ' + str(fit_peak_pos * shift_value / c))
                print('original loc: ' + str(original_loc))
                print()
                print('expected peak: ' + str(expected_peak_photons_R[n]))
                print('expected baseline: ' + str(expected_baseline_photons_R[n]))
                print('recovered baseline: ' + str(recovered_baseline_photons_R[n]))
                print('fit peak: ' + str(photon_counts[original_loc]))
                print('fit peak using gaussian: ' + str(recovered_peak_photons_gauss_R[n]))
                print('recovered peak (fit - baseline): ' + str(recovered_peak_photons_R[n]))
                print('(fit - expected)/expected height: ' + str((recovered_peak_photons_R[n]- expected_peak_photons_R[n])/expected_peak_photons_R[n]))


            if make_plots:# | (recovered_peak_photons_R[n] < 100):
                if not make_plots:
                    plt.figure(figsize = [20,5])
                #plt.subplot(1,3,i+1)
                lw = 1    
               # plt.vlines([expected_locs[i]], ymin = np.min(shifted[idxs_order_shifted]),  ymax = 1.1 * np.max(shifted[idxs_order_shifted]), ls = 'dashed', alpha = 0.5, label = 'Expected injection locations')
                #plt.plot(wl_shifted[idxs_order_shifted], shifted[idxs_order_shifted], '.-', ms = ms, lw = lw, label = 'Shifted spectrum')
                plt.plot(wl_unshifted[idxs_order_unshifted], unshifted[idxs_order_unshifted]/np.max(unshifted[idxs_order_unshifted]), '.-', ms = ms, lw = lw, label = 'Normalized unshifted spectrum')
                plt.plot(wl_unshifted[idxs_order_unshifted], photon_counts[idxs_order_raw_phot]/np.max(photon_counts[idxs_order_raw_phot]), '.-', ms = ms, lw = lw, label = 'Normalized photon counts')
                plt.plot([],[], ' ', label = 'recovered value: ' + str(np.round(recovered_peak_photons_R[n],2)))
                plt.plot([],[], ' ', label = 'expected value: ' + str(np.round(expected_peak_photons_R[n],2)))
                ymin, ymax = plt.gca().get_ylim()
                plt.vlines([original_wl], ymin = np.min(shifted[idxs_order_shifted]),  ymax = ymax, ls = 'dashed', alpha = 0.5, label = 'recovered injection location in unshifted spectrum')            
                plt.vlines([expected_wl[i]], ymin = np.min(shifted[idxs_order_shifted]),  ymax = ymax, color = 'r', alpha = 0.5, label = 'expected location')            
                xmin, xmax = plt.gca().get_xlim(); ran = xmax - xmin
                plt.xlim([xmin + 3*ran/8, xmax - 3*ran/8])
                #plt.plot(wl_shifted_no_inj[idxs_order_shifted_no_inj], shifted_no_inj[idxs_order_shifted_no_inj], '.-', ms = ms, lw = lw, alpha = 0.3, label = 'Shifted spectrum (no injection)')
                plt.xlabel('Wavelength [A]'); plt.ylabel('Intensity')
                #plt.plot(wl_shifted[idxs_order_shifted], gaussian(wl_shifted[idxs_order_shifted], *popt), label = 'Fit')
                plt.title(star + ' shifted spectrum, order ' + str(order))
                plt.legend()
            i += 1
            n += 1
           
    except :
        fail_type, value, traceback = sys.exc_info()
        print(str(fail_type).split('\'')[1] + ': ' + str(value))
        expected_peak_photons_R[n:n+num_inj] = np.nan
        expected_baseline_photons_R[n:n+num_inj] = np.nan
        recovered_peak_photons_R[n:n+num_inj] = np.nan
        recovered_baseline_photons_R[n:n+num_inj] = np.nan
        original_wl_arr_R[n:n+num_inj] = np.nan
        gaia_Teff_arr[n:n+num_inj] =  np.nan
        types[n:n+num_inj] = np.nan
        n += 1 * num_inj
        failed += [star]
        print()
        print('Failed: ' + star)
        print()

    
df = pd.DataFrame(list(zip(expected_peak_photons_R, expected_baseline_photons_R, recovered_peak_photons_R, recovered_baseline_photons_R,
                           original_wl_arr_R, gaia_Teff_arr)), columns = ['Expected_peak', 'Expected_baseline', 'Recovered_peak', 
                                                                                'Recovered_baseline', 'Recovered_wavelength', 'Gaia Teff [K]'])

dt = datetime.datetime.now()
timestamp = dt.strftime("%d") + dt.strftime("%b") + dt.strftime("%Y") + '-' + dt.strftime("%X")
if os.path.exists('residuals_injection_recovery.csv'):
    os.rename('residuals_injection_recovery.csv', 'residuals_injection_recovery_' + timestamp + '.csv')
df.to_csv('residuals_injection_recovery.csv', index = False)
                                                                                