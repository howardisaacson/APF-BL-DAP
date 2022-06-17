# update the wavelength solution for all apf (highest SNR) run
# python update_wl_soln_all_apf.py > wl_update_log_all_apf.txt 2>&1 &

# update the wavelength solution for all obs run
#python update_wl_soln_all_obs.py > wl_update_log_all_obs.txt 2>&1 &

# update the wavelength solution for all obs run
python update_wl_soln_ind.py -start 5080 -stop 5380 > wl_update_log_ind1.txt 2>&1 &
python update_wl_soln_ind.py -start 5380 -stop 5680 > wl_update_log_ind2.txt 2>&1 &
python update_wl_soln_ind.py -start 5680 -stop 5980 > wl_update_log_ind3.txt 2>&1 &
python update_wl_soln_ind.py -start 5980 -stop 6280 > wl_update_log_ind4.txt 2>&1 &
python update_wl_soln_ind.py -start 6280 -stop 6461 > wl_update_log_ind5.txt 2>&1 &