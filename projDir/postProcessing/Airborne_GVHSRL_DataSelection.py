# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:56:12 2018

@author: mhayman
"""

import sys
import os
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
    
    cal_file = paths['cal_file_path'] + 'gv_calvals.json'

    default_settings = {
        'full_flight':False,  # process the entire flight
        'time_ref2takeoff':False,    # flight_time_start and 
                                     # time_stop are referened to takeoff time
        'save_plots':False, # save the plot data        
        'save_data':False, # save data as netcdf
        'save_mol_gain_plot':False, # save plots from molecular gain estimate
        'Airspeed_Threshold':25, # threshold for determining start and end of the flight (in m/s)     
        'use_aircraft_tref':True, # use the start time of the aircraft netcdf as the date reference
        }
      
    
    # check if any settings have been defined.  If not, define it as an empty dict.
    try: settings
    except NameError: settings = {}
        
    # If a paremeter isn't supplied, use the default setting
    for param in default_settings.keys():
        if not param in settings.keys():
            settings[param] = default_settings[param]    
                   
    # Default dirs
    default_aircraft_basepath = {
        'CSET':'.',
        'SOCRATES':'.'    
        } 
    
    if 'aircraft_basepath' in paths.keys():
        aircraft_basepath = paths['aircraft_basepath']
    else:
        aircraft_basepath = default_aircraft_basepath
    
    """
    Load variable lists
    """
    var_aircraft = ['Time','TASX']

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
                    print('--> Requested flight')
                    
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
        
    flight_date_start=datetime.datetime(year=time_takeoff.year,month=time_takeoff.month,day=time_takeoff.day)
    flight_date_end=datetime.datetime(year=time_landing.year,month=time_landing.month,day=time_landing.day)
    
    if settings['full_flight']:
        if settings['use_aircraft_tref']:
            time_start = aircraft_t_ref+datetime.timedelta(seconds=np.int(air_data['Time'][it0]))
            time_stop = aircraft_t_ref+datetime.timedelta(seconds=np.int(air_data['Time'][it1]))
        else:
            time_start = time_takeoff
            time_stop = time_landing
        
    else:
        try: 
            if settings['time_ref2takeoff']:
                time_start = time_takeoff + process_vars['flight_time_start']
                time_stop = time_takeoff + process_vars['flight_time_stop']
            else:
                if process_vars['flight_time_start'].seconds/3600>=time_takeoff.hour:
                    flight_date_1=flight_date_start
                else:
                    flight_date_1=flight_date_end
                if process_vars['flight_time_stop'].seconds/3600>=time_takeoff.hour:
                    flight_date_2=flight_date_start
                else:
                    flight_date_2=flight_date_end
                time_start = flight_date_1+process_vars['flight_time_start']
                time_stop = flight_date_2+process_vars['flight_time_stop']
        except KeyError:
            sys.exit('Stop or start time is not valid!')
                
    # check for out of bounds time limits
    if time_start > time_landing:
        sys.exit('Start time is not valid!')
    if time_stop < time_takeoff:
        sys.exit('Stop time is not valid!')
        
    process_vars['proj_label'] = proj + ' ' + flight_label[usr_flt] + ', '
    process_vars['flight_date'] = flight_date
    process_vars['usr_flt'] = usr_flt
    process_vars['flt'] = flt
    
    """
    Create output folders if they don't exist
    """
       
    # Output data folder
    if settings['save_data']:
        try:
            if settings['save_flight_folder']:                
                paths['save_data_path'] = paths['save_data_path']+process_vars['flt']+'/'
            if not os.path.exists(paths['save_data_path']):
                os.makedirs(paths['save_data_path'])
        except KeyError:
            print('Save data is disabled')
            print('  No save path (save_data_path) is provided')
            settings['save_data'] = False
            
    # Output plot folder
    if settings['save_plots'] or settings['save_mol_gain_plot']:
        try:
            if settings['save_flight_folder']:                
                paths['save_plots_path'] = paths['save_plots_path']+process_vars['flt']+'/'
            if not os.path.exists(paths['save_plots_path']):
                os.makedirs(paths['save_plots_path'])
        except KeyError:
            print('Save plots is disabled')
            print('  No save path (save_plots_path) is provided')
            settings['save_plots'] = False    
    
    return time_start,time_stop,settings,paths,process_vars