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
#basepath = '/scr/rain1/rsfdata/projects/socrates/hsrl/raw/' #SOCRATES basepath to non denoised data
basepath = '/scr/rain1/rsfdata/projects/socrates/hsrl/raw/raw_denoised/' #SOCRATES basepath to non denoised data

# Aircraft data dirs
aircraft_basepath = {
    'CSET':'/scr/raf_data/CSET/24_Mar_17_BU/',
    'SOCRATES':'/scr/raf_data/SOCRATES/'       
    } 
    
# Save data dirs
save_plots_path = '/scr/sci/romatsch/HSRL/pythonOut/plots/'
save_data_path = '/scr/sci/romatsch/HSRL/pythonOut/data/'

paths = {
    'software_path':software_path,
    'cal_file_path':cal_file_path,
	'basepath':basepath,
	'aircraft_basepath':aircraft_basepath,
	'save_data_path':save_data_path,
	'save_plots_path':save_plots_path
	}