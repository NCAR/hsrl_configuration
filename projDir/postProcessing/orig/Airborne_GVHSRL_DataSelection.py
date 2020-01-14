# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:56:12 2018

@author: mhayman
"""

#import sys
import os

##filename = inspect.getframeinfo(inspect.currentframe()).filename
##filename = __file__
#try:
#    filename = fullpath
#except NameError:
#    filename = __file__


## add the path to GVHSRLlib manually
#library_path = os.path.abspath(filename+'/../../libraries/')
#print(library_path)
#if library_path not in sys.path:
#    sys.path.append(library_path)


import numpy as np
import LidarProfileFunctions as lp
import datetime


import json

import GVHSRLlib as gv


    


# input and raw_input are the same for python 3 and 2 respectively
# this makes it so input always accepts a string
try:
    input=raw_input
except NameError:
    pass

def SelectAirborneData(settings={},paths={},process_vars={}):
    
    filename=__file__
#    print(filename)    
    
    cal_file_path = os.path.abspath(filename+'/../../calfiles/')+'/'
    cal_file = cal_file_path + 'gv_calvals.json'
    
#    reanalysis_path = os.path.abspath(filename+'/../../../external_data/')+'/'
    
    default_settings = {
        'full_flight':False,  # process the entire flight
#        'tres':0.5,  # resolution in time in seconds (0.5 sec) before altitude correction
#        'tres_post':1*60, # resolution after altitude correction -  set to zero to not use
#        'zres':7.5,  # altitude resolution in meters (7.5 m minimum)
        
        #mol_gain = 1.133915#1.0728915  # gain adjustment to molecular channel
        
        # index for where to treat the profile as background only
#        'BGIndex': -100, # negative number provides an index from the end of the array
        'platform':'airborne', # 'ground' or 'airborne'.  If 'airborne' it needs an aircraft netcdf.
#        'MaxAlt':10e3,
#        'MinAlt':-30,
        
        'get_extinction':False,  # process data for extinction    
        
        'time_ref2takeoff':False,    # flight_time_start and 
                                     # time_stop are referened to takeoff time
        
#        'RemoveCals':True,  # don't include instances where the I2 cell is removed
                            # scan files are not included in the file search so they
                            # are removed anyway
        
#        'Remove_Off_Data':True, # remove data where the lidar appears to be off
                                # currently only works if RemoveCals is True
        
#        'diff_geo_correct':True,  # apply differential overlap correction
        
        'load_reanalysis':False, # load T and P reanalysis from NCEP/NCAR Model
        
#        'plot_2D':True,   # pcolor plot the BSR and depolarization profiles
#        'show_plots':True, # show plots in matplotlib window
#        'plot_date':True,  # plot results in date time format.  Otherwise plots as hour floats
        'save_plots':False, # save the plot data
        
        'save_data':False, # save data as netcdf
        
        'save_flight_folder':False, # save data/plots in folders according to flight name
        
        'save_mol_gain_plot':False, # save plots from molecular gain estimate
        
#        'time_axis_scale':5.0,  # scale for horizontal axis on pcolor plots
#        'count_mask_threshold':2.0,  # count mask threshold (combined_hi).  If set to zero, no mask applied  
#        'd_part_res_lim':0.25,  # resolution limit to decide where to mask particle depolarization data
        
#        'Estimate_Mol_Gain':True, # use statistics on BSR to estimate the molecular gain
        
#        'hsrl_rb_adjust':True, # adjust for Rayleigh Brillouin Spectrum
        
#        'Denoise_Mol':False, # run PTV denoising on molecular channel
        
        
        'Airspeed_Threshold':25, # threshold for determining start and end of the flight (in m/s)
        
        'use_aircraft_tref':True, # use the start time of the aircraft netcdf as the date reference
        
#        'loadQWP':'fixed',  # load 'fixed','rotating', or 'all' QWP data
        
#        'as_altitude':False, # process in altitude centered format or range centered format
        
#        'SNRlimit':40.0  # minimum integrated SNR to treat the lidar as transmitting
                         # used to filter instances where the shutter is closed
                         # toggle this with 'Remove_Off_Data'
        }
      
    
    # check if any settings have been defined.  If not, define it as an empty dict.
    try: settings
    except NameError: settings = {}
        
    # If a paremeter isn't supplied, use the default setting
    for param in default_settings.keys():
        if not param in settings.keys():
            settings[param] = default_settings[param]    
        
#    tres = settings['tres']
#    tres_post = settings['tres_post']
#    zres = settings['zres']
#    BGIndex = settings['BGIndex']
#    MaxAlt = settings['MaxAlt']
#    MinAlt = settings['MinAlt']
    
#    #default_basepath = '/scr/eldora1/rsfdata/hsrl/raw/'  # new absolute path
#    #default_basepath = '/scr/rain1/rsfdata/projects/socrates/hsrl/raw/'
#    default_basepath = '/Users/mhayman/Documents/HSRL/GVHSRL_data/'  # local path
#    
#    try:
#        basepath
#    except NameError:
#        basepath = default_basepath
#    #basepath = '/scr/eldora1/HSRL_data/'  # old path - still works with link from HSRL_data to /hsrl/raw/
#    #basepath = '/Users/mhayman/Documents/HSRL/GVHSRL_data/'  # local computer data path
     
           
    #default_aircraft_basepath = {
    #    'CSET':'/scr/raf_data/CSET/24_Mar_17_BU/',
    #    'SOCRATES':'/scr/raf_data/SOCRATES/' #SOCRATEStf01.nc       
    #    } 
           
    # Local Paths
    default_aircraft_basepath = {
        'CSET':'/Users/mhayman/Documents/HSRL/aircraft_data/',
        'SOCRATES':'/Users/mhayman/Documents/HSRL/aircraft_data/' #SOCRATEStf01.nc       
        } 
    
    if 'aircraft_basepath' in paths.keys():
        aircraft_basepath = paths['aircraft_basepath']
    else:
        aircraft_basepath = default_aircraft_basepath
    
    """
    Load variable lists
    """
    
#    # list of 1D variables to load
#    var_1d_list = ['total_energy','RemoveLongI2Cell'\
#        ,'TelescopeDirection','TelescopeLocked','polarization','DATA_shot_count','builduptime']  # 'DATA_shot_count'
#    
#    # list of 2D variables (profiles) to load
#    var_2d_list = ['molecular','combined_hi','combined_lo','cross']
    
    # list of aircraft variables to load
    var_aircraft = ['Time','TASX']
        #['Time','GGALT','ROLL','PITCH','THDG','GGLAT','GGLON','TASX','ATX','PSXC']
    
    # grab calval data from json file
    with open(cal_file,"r") as f:
        cal_json = json.loads(f.read())
    f.close()
    
    proj_list = []
    year_list = []
    for ai in range(len(cal_json['Flights'])):
        if not cal_json['Flights'][ai]['Project'] in proj_list:
            proj_list.extend([cal_json['Flights'][ai]['Project']])
            year_list.extend([lp.json_str_to_datetime(cal_json['Flights'][ai]['date'])])
            print('%d.) '%(len(proj_list)) + proj_list[-1] + ', ' + year_list[-1].strftime('%Y'))
    print('')
    # check if the project/flight has been passed in 
    # if not, ask the user for it    
    try: 
        proj = process_vars['proj']
    except KeyError:    
        # interactive prompt to determine desired flight
      
        usr_proj = np.int(input('Select Project: '))-1
        if usr_proj < 0 or usr_proj > len(proj_list)-1:
            print('Selection is not recognized')
        else:
            proj = proj_list[usr_proj]
    
    flight_list = []
    flight_date = []
    flight_label = []
    try:
        flt = process_vars['flt']
        for ai in range(len(cal_json['Flights'])):
            if cal_json['Flights'][ai]['Project'] == proj:
                flight_list.extend([proj+cal_json['Flights'][ai]['Flight Designation'] + str(cal_json['Flights'][ai]['Flight Number']).zfill(2)])
                flight_date.extend([lp.json_str_to_datetime(cal_json['Flights'][ai]['date'])])
                flight_label.extend([cal_json['Flights'][ai]['Flight Designation'].upper()+str(cal_json['Flights'][ai]['Flight Number']).zfill(2)])
                print('%d.) '%len(flight_list) + ' ' + flight_list[-1] + ', ' + flight_date[-1].strftime('%d-%b, %Y'))
                if flight_list[-1] == flt:
                    usr_flt = len(flight_list)-1
                    print('^^^^^^^^^^')
                    
    except KeyError:
        for ai in range(len(cal_json['Flights'])):
            if cal_json['Flights'][ai]['Project'] == proj:
                flight_list.extend([proj+cal_json['Flights'][ai]['Flight Designation'] + str(cal_json['Flights'][ai]['Flight Number']).zfill(2)])
                flight_date.extend([lp.json_str_to_datetime(cal_json['Flights'][ai]['date'])])
                flight_label.extend([cal_json['Flights'][ai]['Flight Designation'].upper()+str(cal_json['Flights'][ai]['Flight Number']).zfill(2)])
                print('%d.) '%len(flight_list) + ' ' + flight_list[-1] + ', ' + flight_date[-1].strftime('%d-%b, %Y'))
        
        usr_flt = np.int(input('Select Flight: '))-1
        if usr_flt < 0 or usr_flt > len(flight_list)-1:
            print('Selection is not recognized')
        else:
            flt = flight_list[usr_flt]
            
    filePathAircraft = aircraft_basepath[proj] + flt + '.nc'
    
    paths['filePathAircraft'] = filePathAircraft
            
    #  load aircraft data    
    air_data,aircraft_t_ref = gv.load_aircraft_data(filePathAircraft,var_aircraft)
    process_vars['aircraft_t_ref'] = aircraft_t_ref
    
    # locate time range where aircraft is flying
    iflight = np.nonzero(air_data['TASX'] > settings['Airspeed_Threshold'])[0]
    it0 = iflight[0]  # index when aircraft starts moving
    it1 = iflight[-1]  # index when aircraft stops moving
    time_takeoff = flight_date[usr_flt]+datetime.timedelta(seconds=np.int(air_data['Time'][it0]))
    time_landing = flight_date[usr_flt]+datetime.timedelta(seconds=np.int(air_data['Time'][it1]))
    print('Flight time is: ')
    print('   '+time_takeoff.strftime('%H:%M %d-%b, %Y to'))
    print('   '+time_landing.strftime('%H:%M %d-%b, %Y'))
    print('')
    
        
    
    if settings['full_flight']:
        if settings['use_aircraft_tref']:
            time_start = aircraft_t_ref+datetime.timedelta(seconds=np.int(air_data['Time'][it0]))
            time_stop = aircraft_t_ref+datetime.timedelta(seconds=np.int(air_data['Time'][it1]))
        else:
            time_start = flight_date[usr_flt]+datetime.timedelta(seconds=np.int(air_data['Time'][it0]))
            time_stop = flight_date[usr_flt]+datetime.timedelta(seconds=np.int(air_data['Time'][it1]))
        
    else:
        try: 
            if settings['time_ref2takeoff']:
                time_start = time_takeoff + process_vars['flight_time_start']
                time_stop = time_takeoff + process_vars['flight_time_stop']
            else:
                time_start = flight_date[usr_flt]+process_vars['flight_time_start']
                time_stop = flight_date[usr_flt]+process_vars['flight_time_stop']
        except KeyError:
            time_sel = input('Enter start hour (UTC) or press [Enter] select start of flight: ')
            if time_sel == '':
                time_start = flight_date[usr_flt]+datetime.timedelta(seconds=np.int(air_data['Time'][it0]))
                
            else:
                start_hr = np.float(time_sel)
                time_start = flight_date[usr_flt]+datetime.timedelta(seconds=np.int(start_hr*3600))
                
            
            time_sel = input('Enter duration or press [Enter] for end of flight: ')
            if time_sel == '':
                time_stop = flight_date[usr_flt]+datetime.timedelta(seconds=np.int(air_data['Time'][it1]))
            else:
                stop_hr = np.float(time_sel)
                time_stop = time_start+datetime.timedelta(seconds=np.int(stop_hr*3600))
    
    # check for out of bounds time limits
    if time_start > time_landing:
        time_start = time_landing - datetime.timedelta(minutes=1)
    if time_stop < time_takeoff:
        time_stop = time_takeoff + datetime.timedelta(minutes=1)
        
    process_vars['proj_label'] = proj + ' ' + flight_label[usr_flt] + ', '
    process_vars['flight_date'] = flight_date
    process_vars['usr_flt'] = usr_flt
    process_vars['flt'] = flt
    
    
    # set the save paths based on the flight name if 
    # save_flight_folder is enabled    
    
    # unecessary condition?  and settings['save_flight_folder']
    if settings['save_data']:
        try:
            flt_save_path = paths['save_data_path']+process_vars['flt']+'/'
            if not os.path.exists(flt_save_path):
                os.makedirs(flt_save_path)
            paths['save_data_path'] = flt_save_path
        except KeyError:
            print('Save data is disabled')
            print('  No save path (save_data_path) is provided')
            settings['save_data'] = False
            
    # unnecessary condition ?  and settings['save_flight_folder']
    if settings['save_plots'] or settings['save_mol_gain_plot']:
        try:
            flt_plot_path = paths['save_plots_path']+process_vars['flt']+'/'
            if not os.path.exists(flt_plot_path):
                os.makedirs(flt_plot_path)
            paths['save_plots_path'] = flt_plot_path
        except KeyError:
            print('Save plots is disabled')
            print('  No save path (save_plots_path) is provided')
            settings['save_plots'] = False    
    
    return time_start,time_stop,settings,paths,process_vars