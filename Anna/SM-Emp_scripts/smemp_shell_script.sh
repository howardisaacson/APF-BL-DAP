#!/bin/bash

#num_files = ls /APF_spectra/all_apf_spectra_highest_SNR -1 | wc -l

python configure_out_dirs.py -resid_out 'NDRR_all_apf' -results_out 'specmatch_results_all_apf.csv' -results_type 'all_apf' -plots_out 'SM_output_plots_all_apf' -logfiles 'logfiles'

#python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/additional_apf_spectra_highest_SNR' -resid_out 'NDRR_testing' -results_out 'specmatch_results_tests10.csv' -plots_out 'SM_output_plots_testing' -log 'specmatch_output10.txt' -start 0 -stop 25 > logfiles/log10.txt 2>&1 & 
#python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/additional_apf_spectra_highest_SNR' -resid_out 'NDRR_testing' -results_out 'specmatch_results_tests11.csv' -plots_out 'SM_output_plots_testing' -log 'specmatch_output11.txt' -start 25 -stop 50 > logfiles/log11.txt 2>&1 & 
#python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/additional_apf_spectra_highest_SNR' -resid_out 'NDRR_testing' -results_out 'specmatch_results_tests12.csv' -plots_out 'SM_output_plots_testing' -log 'specmatch_output12.txt' -start 50 -stop 75 > logfiles/log12.txt 2>&1 & 
#python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/additional_apf_spectra_highest_SNR' -resid_out 'NDRR_testing' -results_out 'specmatch_results_tests13.csv' -plots_out 'SM_output_plots_testing' -log 'specmatch_output13.txt' -start 75 -stop 100 > logfiles/log13.txt 2>&1 & 
#python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/additional_apf_spectra_highest_SNR' -resid_out 'NDRR_testing' -results_out 'specmatch_results_tests14.csv' -plots_out 'SM_output_plots_testing' -log 'specmatch_output14.txt' -start 100 -stop 114 > logfiles/log14.txt 2>&1 & 

python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_all_apf' -results_out 'specmatch_results_all1_05Nov2021.csv' -results_type 'all_apf' -plots_out 'SM_output_plots_all_apf' -log 'logfiles/SM_outputs/specmatch_output1.txt' -start 0 -stop 50 > logfiles/SM_logs/log1.txt 2>&1 & 

python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_all_apf' -results_out 'specmatch_results_all2_05Nov2021.csv' -results_type 'all_apf' -plots_out 'SM_output_plots_all_apf' -log 'logfiles/SM_outputs/specmatch_output2.txt' -start 50 -stop 100 > logfiles/SM_logs/log2.txt 2>&1 & 

python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_all_apf' -results_out 'specmatch_results_all3_05Nov2021.csv' -results_type 'all_apf' -plots_out 'SM_output_plots_all_apf' -log 'logfiles/SM_outputs/specmatch_output3.txt' -start 100 -stop 150 > logfiles/SM_logs/log3.txt 2>&1 & 

python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_all_apf' -results_out 'specmatch_results_all4_05Nov2021.csv' -results_type 'all_apf' -plots_out 'SM_output_plots_all_apf' -log 'logfiles/SM_outputs/specmatch_output4.txt' -start 150 -stop 200 > logfiles/SM_logs/log4.txt 2>&1 & 

python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_all_apf' -results_out 'specmatch_results_all5_05Nov2021.csv' -results_type 'all_apf' -plots_out 'SM_output_plots_all_apf' -log 'logfiles/SM_outputs/specmatch_output5.txt' -start 200 -stop 250 > logfiles/SM_logs/log5.txt 2>&1 & 

python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_all_apf' -results_out 'specmatch_results_all6_05Nov2021.csv' -results_type 'all_apf' -plots_out 'SM_output_plots_all_apf' -log 'logfiles/SM_outputs/specmatch_output6.txt' -start 250 -stop 300 > logfiles/SM_logs/log6.txt 2>&1 & 

python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_all_apf' -results_out 'specmatch_results_all7_05Nov2021.csv' -results_type 'all_apf' -plots_out 'SM_output_plots_all_apf' -log 'logfiles/SM_outputs/specmatch_output7.txt' -start 300 -stop 350 > logfiles/SM_logs/log7.txt 2>&1 & 

python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_all_apf' -results_out 'specmatch_results_all8_05Nov2021.csv' -results_type 'all_apf' -plots_out 'SM_output_plots_all_apf' -log 'logfiles/SM_outputs/specmatch_output8.txt' -start 350 -stop 400 > logfiles/SM_logs/log8.txt 2>&1 & 

python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_all_apf' -results_out 'specmatch_results_all9_05Nov2021.csv' -results_type 'all_apf' -plots_out 'SM_output_plots_all_apf' -log 'logfiles/SM_outputs/specmatch_output9.txt' -start 400 -stop 450 > logfiles/SM_logs/log9.txt 2>&1 & 

python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_all_apf' -results_out 'specmatch_results_all10_054Nov2021.csv' -results_type 'all_apf' -plots_out 'SM_output_plots_all_apf' -log 'logfiles/SM_outputs/specmatch_output10.txt' -start 450 -stop 500 > logfiles/SM_logs/log10.txt 2>&1 & 

python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_all_apf' -results_out 'specmatch_results_all11_05Nov2021.csv' -results_type 'all_apf' -plots_out 'SM_output_plots_all_apf' -log 'logfiles/SM_outputs/specmatch_output11.txt' -start 500 -stop 550 > logfiles/SM_logs/log11.txt 2>&1 & 

python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_all_apf' -results_out 'specmatch_results_all12_05Nov2021.csv' -results_type 'all_apf' -plots_out 'SM_output_plots_all_apf' -log 'logfiles/SM_outputs/specmatch_output12.txt' -start 550 -stop 600 > logfiles/SM_logs/log12.txt 2>&1 & 

python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_all_apf' -results_out 'specmatch_results_all13_05Nov2021.csv' -results_type 'all_apf' -plots_out 'SM_output_plots_all_apf' -log 'logfiles/SM_outputs/specmatch_output13.txt' -start 600 -stop 650 > logfiles/SM_logs/log13.txt 2>&1 & 

python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_all_apf' -results_out 'specmatch_results_all14_05Nov2021.csv' -results_type 'all_apf' -plots_out 'SM_output_plots_all_apf' -log 'logfiles/SM_outputs/specmatch_output14.txt' -start 650 -stop 700 > logfiles/SM_logs/log14.txt 2>&1 & 

python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_all_apf' -results_out 'specmatch_results_all15_05Nov2021.csv' -results_type 'all_apf' -plots_out 'SM_output_plots_all_apf' -log 'logfiles/SM_outputs/specmatch_output15.txt' -start 700 -stop 750 > logfiles/SM_logs/log15.txt 2>&1 & 

python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_all_apf' -results_out 'specmatch_results_all16_05Nov2021.csv' -results_type 'all_apf' -plots_out 'SM_output_plots_all_apf' -log 'logfiles/SM_outputs/specmatch_output16.txt' -start 750 -stop 800 > logfiles/SM_logs/log16.txt 2>&1 & 

python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_all_apf' -results_out 'specmatch_results_all17_05Nov2021.csv' -results_type 'all_apf' -plots_out 'SM_output_plots_all_apf' -log 'logfiles/SM_outputs/specmatch_output17.txt' -start 800 -stop 850 > logfiles/SM_logs/log17.txt 2>&1 & 

python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_all_apf' -results_out 'specmatch_results_all18_05Nov2021.csv' -results_type 'all_apf' -plots_out 'SM_output_plots_all_apf' -log 'logfiles/SM_outputs/specmatch_output18.txt' -start 850 -stop 902 > logfiles/SM_logs/log18.txt 2>&1 & 


#python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_testing' -results_out 'specmatch_results_testing10.csv' -plots_out 'SM_output_plots_testing' -log 'specmatch_output10.txt' -start 450 -stop 500  & 
#python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_testing' -results_out 'specmatch_results_testing11.csv' -plots_out 'SM_output_plots_testing' -log 'specmatch_output11.txt' -start  500 -stop  550  & 
#python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_testing' -results_out 'specmatch_results_testing12.csv' -plots_out 'SM_output_plots_testing' -log 'specmatch_output12.txt' -start  550 -stop  600  & 
#python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_testing' -results_out 'specmatch_results_testing13.csv' -plots_out 'SM_output_plots_testing' -log 'specmatch_output13.txt' -start  600 -stop  650  & 
#python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_testing' -results_out 'specmatch_results_testing14.csv' -plots_out 'SM_output_plots_testing' -log 'specmatch_output14.txt' -start  650 -stop  700  & 
#python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_testing' -results_out 'specmatch_results_testing15.csv' -plots_out 'SM_output_plots_testing' -log 'specmatch_output15.txt' -start  700 -stop  750  & 
#python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_testing' -results_out 'specmatch_results_testing16.csv' -plots_out 'SM_output_plots_testing' -log 'specmatch_output16.txt' -start  750 -stop  800  & 
#python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/all_apf_spectra_highest_SNR' -resid_out 'NDRR_testing' -results_out 'specmatch_results_testing17.csv' -plots_out 'SM_output_plots_testing' -log 'specmatch_output17.txt' -start  800 -stop  810  
#python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/apf_spectra_highest_SNR' -resid_out 'NDRR_testing' -results_out 'specmatch_results_testing.csv' -plots_out 'SM_output_plots_testing' -start = 850 -stop = 900  & 
#python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/apf_spectra_highest_SNR' -resid_out 'NDRR_testing' -results_out 'specmatch_results_testing.csv' -plots_out 'SM_output_plots_testing' -start = 900 -stop = 950  & 
#python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/apf_spectra_highest_SNR' -resid_out 'NDRR_testing' -results_out 'specmatch_results_testing.csv' -plots_out 'SM_output_plots_testing' -start = 950 -stop = 1000  & 
#python run_smemp_apf.py -in '/mnt_home/azuckerman/BL_APF_DAP/APF_spectra/apf_spectra_highest_SNR' -resid_out 'NDRR_testing' -results_out 'specmatch_results_testing.csv' -plots_out 'SM_output_plots_testing' -start = 1000 -stop = 1050  & 

