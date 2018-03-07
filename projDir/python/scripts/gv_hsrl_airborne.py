# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 11:38:22 2018

@author: mhayman
"""
import os
import datetime

proj = 'SOCRATES'
flt = proj+'tf01'
flight_time_start = datetime.timedelta(hours=20,minutes=0)
flight_time_stop = datetime.timedelta(hours=22,minutes=0)

settings = {
    'tres':0.5,  # resolution in time in seconds (0.5 sec) before altitude correction
    'tres_post':10, # resolution after altitude correction (in seconds) -  set to zero to not use
    'zres':7.5,  # altitude resolution in meters (7.5 m minimum)
    
    #mol_gain = 1.133915#1.0728915  # gain adjustment to molecular channel
    
    # index for where to treat the profile as background only
    'BGIndex': -100, # negative number provides an index from the end of the array
    'platform':'airborne', # 'ground' or 'airborne'.  If 'airborne' it needs an aircraft netcdf.
    'MaxAlt':10e3,
    'MinAlt':-30,
    
    'RemoveCals':True,  # don't include instances where the I2 cell is removed
                        # scan files are not included in the file search so they
                        # are removed anyway
    
    'diff_geo_correct':True,  # apply differential overlap correction
    
    'load_reanalysis':False, # load T and P reanalysis from NCEP/NCAR Model
    
    'plot_2D':True,   # pcolor plot the BSR and depolarization profiles
    'plot_date':True,  # plot results in date time format.  Otherwise plots as hour floats
    
    'Estimate_Mol_Gain':True, # use statistics on BSR to estimate the molecular gain
    
    'hsrl_rb_adjust':True, # adjust for Rayleigh Brillouin Spectrum
    
    'Denoise_Mol':False, # run PTV denoising on molecular channel
    
    
    'Airspeed_Threshold':15, # threshold for determining start and end of the flight (in m/s)
    
    'loadQWP':'fixed',  # load 'fixed','rotating', or 'all' QWP data
    }
    
    
#basepath = '/scr/eldora1/HSRL_data/'  # old path - still works with link from HSRL_data to /hsrl/raw/
#basepath = '/scr/eldora1/rsfdata/hsrl/raw/'  # new absolute path
#basepath = '/Users/mhayman/Documents/HSRL/GVHSRL_data/'  # local computer data path
        
aircraft_basepath = {
    'CSET':'/scr/raf_data/CSET/24_Mar_17_BU/',
    'SOCRATES':'/scr/raf_data/SOCRATES/' #SOCRATEStf01.nc       
    } 
    
basepath = '/scr/rain1/rsfdata/projects/socrates/hsrl/raw/'

Processor = os.environ['HOME']+'/Documents/Python/Lidar/GV-HSRL-Python/processors/Airborne_Processor_GVHSRL.py'

fullpath = os.path.abspath(Processor)
g = globals().copy()
g['__file__'] = fullpath

   
#exec(open(os.path.abspath(__file__+'/../Path_Settings.py')).read())
exec(open(Processor).read(),g)
