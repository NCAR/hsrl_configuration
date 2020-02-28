# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 11:38:22 2018

@author: mhayman
"""
import os
import datetime
import matplotlib.pyplot as plt

import Airborne_GVHSRL_DataProcessor as dp
import Airborne_GVHSRL_DataSelection as ds

plt.close('all')
process_start_time = datetime.datetime.now()

process_vars = {}
process_vars['proj'] = 'CSET' # SOCRATES, CSET
process_vars['flt'] = process_vars['proj']+'rf01'

"""
Set start and end time
"""
# As time in hour and minutes --> set 'time_ref2takeoff':False in settings
process_vars['flight_time_start'] = datetime.timedelta(hours=17,minutes=40)
process_vars['flight_time_stop'] = datetime.timedelta(hours=17,minutes=55)

"""
Set up processing settings
"""

settings = {
    # Time and aircraft settings
    'full_flight':True, # process the entire flight
    'time_ref2takeoff':False,
    'Airspeed_Threshold':25, # threshold for determining start and end of the flight (in m/s)
    
    'aircraft_time_shift':0.859822,  # shift in aircraft time needed to align to HSRL time 0.75 - 3.0
    
    'load_reanalysis_from_file':True, # load T and P reanalysis from netcdf file
    
    # Plot  settings
    'plot_2D':True,   # pcolor plot the BSR and depolarization profiles
    'show_plots':True, # show plots in a matplotlib window
    'plot_date':True,  # plot results in date time format.  Otherwise plots as hour floats
    'time_axis_scale':5.0,  # scale for horizontal axis on pcolor plots    
    
    # Save settings
    'save_plots':True, # save the plot data    
    'save_mol_gain_plot':True,  # save the plots used to estimate molecular gain
    'save_data':True, # save data as netcdf    
    'save_flight_folder':False, # save data/plots in folders according to flight name
    
    # Data processing settings
    'tres':0.5,  # resolution in time in seconds (0.5 sec) before altitude correction
    'tres_post':0.5, # resolution after altitude correction (in seconds) -  set to zero to not use
    'zres':7.5,  # altitude resolution in meters (7.5 m minimum)

    'mol_smooth':False, # smooth molecular profile
        
    # index for where to treat the profile as background only
    'BGIndex': -100, # negative number provides an index from the end of the array
    'platform':'airborne', # 'ground' or 'airborne'.  If 'airborne' it needs an aircraft netcdf.
    'MaxAlt':15e3,
    'MinAlt':-1e3,
    
    'RemoveCals':True,  # don't include instances where the I2 cell is removed
                        # scan files are not included in the file search so they
                        # are removed anyway    
    'Remove_Off_Data':True, # remove instances where the lidar does not appear
                            # to be running
    
    'get_extinction':True, # retrieve extinction estimate
    
    'diff_geo_correct':True,  # apply differential overlap correction
    
    'deadtime_correct':True,  # perform deadtime correction
        
    'count_mask_threshold':2.0,  # count mask threshold (combined_hi).  If set to zero, no mask applied 
    
    'Estimate_Mol_Gain':True, # use statistics on BSR to estimate the molecular gain
    
    'hsrl_rb_adjust':True, # adjust for Rayleigh Brillouin Spectrum
       
    'loadQWP':'fixed',  # load 'fixed','rotating', or 'all' QWP data
 
    'use_BM3D':True   
    }
    
"""
Load path file
"""
PathFile = os.path.abspath(__file__+'/../')+'/gv_hsrl_paths.py'
print('Paths stored in: ' +PathFile)
exec(open(PathFile).read())

"""
Get the right times
"""
time_start0,time_stop0,settings,paths,process_vars = ds.SelectAirborneData(settings=settings,paths=paths,process_vars=process_vars)
day_start = datetime.datetime(year=time_start0.year,month=time_start0.month,day=time_start0.day)

"""
Run processing
"""
prof_list = dp.ProcessAirborneDataChunk(time_start0,time_stop0,
                             settings=settings,paths=paths,process_vars=process_vars,date_reference=day_start)

process_stop_time = datetime.datetime.now()
plt.show()

print('processing began ' + process_start_time.strftime('%Y-%b-%d %H:%M'))
print('processing completed ' + process_stop_time.strftime('%Y-%b-%d %H:%M'))
print('processing duration %f hours' %((process_stop_time-process_start_time).total_seconds()/3600.0))
