# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:09:53 2018

@author: mhayman
"""

import os

# Software dir
software_path = os.environ['HOME']+'/git/hsrl_configuration/projDir/postProcessing/'
# Cal file path
cal_file_path = os.environ['HOME']+'/git/hsrl_configuration/projDir/calfiles/'

# HSRL data dirs
basepath = {
        'CSET':'/scr/snow2/rsfdata/projects/cset/hsrl/raw/raw_denoised_tempPress/', #CSET basepath to denoised data
        'SOCRATES':'/scr/rain1/rsfdata/projects/socrates/hsrl/raw/raw_denoised/' #SOCRATES basepath to denoised data
        }

# Aircraft data dirs
aircraft_basepath = {
    'CSET':'/scr/snow2/rsfdata/projects/cset/GV/gv_data/',
    'SOCRATES':'/scr/raf_data/SOCRATES/'       
    } 
    
# Save data dirs
save_plots_path = '/scr/snow2/rsfdata/projects/cset/hsrl/qc2/pythonOut/plotsModel/'
save_data_path = '/scr/snow2/rsfdata/projects/cset/hsrl/qc2/pythonOut/dataModel/'

paths = {
    'software_path':software_path,
    'cal_file_path':cal_file_path,
	'basepath':basepath,
	'aircraft_basepath':aircraft_basepath,
	'save_data_path':save_data_path,
	'save_plots_path':save_plots_path
	}
