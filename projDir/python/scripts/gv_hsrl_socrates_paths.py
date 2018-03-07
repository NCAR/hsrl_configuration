# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:09:53 2018

@author: mhayman
"""

import os

#/scr/rain1/rsfdata/projects/socrates/hsrl/raw/2018/01/02/raw/gvhsrl_201801

#basepath = '/scr/eldora1/HSRL_data/'  # old path - still works with link from HSRL_data to /hsrl/raw/
#basepath = '/scr/eldora1/rsfdata/hsrl/raw/'  # new absolute path

#basepath = '/Users/mhayman/Documents/HSRL/GVHSRL_data/'  # local computer data path
#basepath = '/scr/rain1/rsfdata/projects/socrates/hsrl/raw/' #SOCRATES basepath
#basepath = '/mnt/nas/data/SOCRATES/data_synced/hsrl/raw/' #SOCRATES old field basepath
basepath = '/mnt/nas/data/SOCRATES/hsrl_synced/raw/' #SOCRATES new field basepath as of 1/20/2018 23:00 (UTC) (after RF02, before RF03)
#basepath = '/data/gvhsrldata/SOCRATES/raw/' #SOCRATES local basepath

# Server path (FL)
#aircraft_basepath = {
#    'CSET':'/scr/raf_data/CSET/24_Mar_17_BU/',
#    'SOCRATES':'/scr/raf_data/SOCRATES/' #SOCRATEStf01.nc       
#    } 
    
# Local Paths
#aircraft_basepath = {
#    'CSET':'/Users/mhayman/Documents/HSRL/aircraft_data/',
#    'SOCRATES':'/Users/mhayman/Documents/HSRL/aircraft_data/' #SOCRATEStf01.nc       
#    } 

# SOCRATES Field Paths
aircraft_basepath = {
    'CSET':'/Users/mhayman/Documents/HSRL/aircraft_data/',
    'SOCRATES':'/mnt/nas/data/SOCRATES/data_synced/'    
    } 

# local save paths
#save_data_path = '/data/gvhsrldata/SOCRATES/data/'
#save_plots_path = '/data/gvhsrldata/SOCRATES/plots/'
save_data_path = '/mnt/nas/data/SOCRATES/hsrl_synced/preliminary_data/'
save_plots_path = '/mnt/nas/data/SOCRATES/hsrl_synced/preliminary_images/5minImages/'


software_path = os.environ['HOME']+'/Python/lidar/GV-HSRL-Python/'

paths = {
        'basepath':basepath,
        'aircraft_basepath':aircraft_basepath,
        'save_data_path':save_data_path,
        'save_plots_path':save_plots_path,
        'software_path':software_path
        }

#Processor = os.environ['HOME']+'/Documents/Python/Lidar/GV-HSRL-Python/processors/Airborne_Processor_GVHSRL.py'
