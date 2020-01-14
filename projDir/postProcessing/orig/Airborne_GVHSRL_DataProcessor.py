# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:59:17 2018

@author: mhayman
"""

#import sys
import os

#filename = inspect.getframeinfo(inspect.currentframe()).filename
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
import matplotlib.pyplot as plt
import LidarProfileFunctions as lp
import LidarPlotFunctions as lplt
import datetime
#import glob

import matplotlib.dates as mdates

import json

import GVHSRLlib as gv

#import ExternalDataFunctions as ex

import MLELidarProfileFunctions as mle
    

# input and raw_input are the same for python 3 and 2 respectively
# this makes it so input always accepts a string
try:
    input=raw_input
except NameError:
    pass

def ProcessAirborneDataChunk(time_start,time_stop,
                             settings={},paths={},process_vars={},
                             date_reference=0):

    filename = __file__
#    print(filename)
    cal_file_path = os.path.abspath(filename+'/../../calfiles/')+'/'
    cal_file = cal_file_path + 'gv_calvals.json'
    
#    reanalysis_path = os.path.abspath(filename+'/../../../external_data/')+'/'
    
    default_settings = {
        'tres':0.5,  # resolution in time in seconds (0.5 sec) before altitude correction
        'tres_post':1*60, # resolution after altitude correction -  set to zero to not use
        'zres':7.5,  # altitude resolution in meters (7.5 m minimum)
        
        'mol_smooth':False, # smooth molecular profile
        't_mol_smooth':0.0, # smoothing in time of molecular profile
        'z_mol_smooth':30.0, # smoothing in range of molecular profile
        
        'use_BM3D':True, # use BM3D denoised raw data where it is available

        'range_min':150.0,  # closest range in m where data is treated as valid
        
        #mol_gain = 1.133915#1.0728915  # gain adjustment to molecular channel
        
        # index for where to treat the profile as background only
        'BGIndex': -100, # negative number provides an index from the end of the array
        'platform':'airborne', # 'ground' or 'airborne'.  If 'airborne' it needs an aircraft netcdf.
        'MaxAlt':10e3,
        'MinAlt':-30,
        
        'get_extinction':False,  # process data for extinction   
        'ext_sg_width':21,  # savitsky-gouley window width (must be odd)
        'ext_sg_order':3,   # savitsky-gouley polynomial order
        'ext_tres':15,      # extinction convolution kernel time width in seconds
        'ext_zres':60,      # extinction convolution kernel range width in meters
        
        'baseline_subtract':False, # use baseline subtraction
        'deadtime_correct':False,  # correct for APD deadtime
        'merge_hi_lo':True,         # merge combined high and low gain channels into a single estimate
        
        'time_ref2takeoff':False,    # flight_time_start and 
                                     # time_stop are referened to takeoff time
        
        'RemoveCals':True,  # don't include instances where the I2 cell is removed
                            # scan files are not included in the file search so they
                            # are removed anyway
        
        'Remove_Off_Data':True, # remove data where the lidar appears to be off
                                # currently only works if RemoveCals is True
        
        'diff_geo_correct':True,  # apply differential overlap correction
        
        'load_reanalysis':False, # load T and P reanalysis from NCEP/NCAR Model
        
        'plot_2D':True,   # pcolor plot the BSR and depolarization profiles
        'show_plots':True, # show plots in matplotlib window
        'plot_date':True,  # plot results in date time format.  Otherwise plots as hour floats
        'save_plots':False, # save the plot data
        
        'save_data':False, # save data as netcdf
        'save_raw':True,    # save raw profiles (with matching time axes to processed data)
        
        'time_axis_scale':5.0,  # scale for horizontal axis on pcolor plots
        'alt_axis_scale':1.0,   # scale for vertical axis on pcolor plots
        'count_mask_threshold':2.0,  # count mask threshold (combined_hi).  If set to zero, no mask applied  
        'd_part_res_lim':0.25,  # resolution limit to decide where to mask particle depolarization data
        
        'Estimate_Mol_Gain':True, # use statistics on BSR to estimate the molecular gain
        'save_mol_gain_plot':False, # save the results of the molecular gain estimate
        
        'hsrl_rb_adjust':True, # adjust for Rayleigh Brillouin Spectrum
        
        'Denoise_Mol':False, # run PTV denoising on molecular channel
        'denoise_accel':True, # run accelerated denoising (reduced scan region)
        'denoise_debug_plots':False, # plot denoising results (use only for debugging.  Slows processing down.)
        'denoise_eps':1e-7,  # eps float precision for optimization.  Smaller numbers are slower but provide more accurate denoising 
        
        'time_denoise':False,  # run horizontal (time) denoising on profiles
        'time_denoise_accel':True, # run accelerated denoising (reduced scan region)
        'time_denoise_debug_plots':False, # plot denoising results (use only for debugging.  Slows processing down.)
        'time_denoise_eps':1e-5,  # eps float precision for optimization.  Smaller numbers are slower but provide more accurate denoising 
        'time_denoise_verbose':False,  # output optimizor status at each step
        'time_denoise_max_range':10e3,  # maximum range to which denoising occurs
        
        
        'Airspeed_Threshold':15, # threshold for determining start and end of the flight (in m/s)
        
        'loadQWP':'fixed',  # load 'fixed','rotating', or 'all' QWP data
        
        'as_altitude':False, # process in altitude centered format or range centered format
        
        'SNRlimit':40.0,  # minimum integrated SNR to treat the lidar as transmitting
                         # used to filter instances where the shutter is closed
                         # toggle this with 'Remove_Off_Data'
        
        'use_aircraft_tref':True,  # set the time reference based on aircraft data
        'aircraft_time_shift':0.75,  # shift in aircraft time needed to align to HSRL time 0.75
        'Estimate Time Shift':True   # estimate the time shift between aircraft and HSRL data systems based on roll and pitch manuevers
        }
        
    
    # check if any settings have been defined.  If not, define it as an empty dict.
    try: settings
    except NameError: settings = {}
   
    save_other_data = {}  # dict containing extra variables to be saved to the netcdf file   
   
    # If a paremeter isn't supplied, use the default setting
    for param in default_settings.keys():
        if not param in settings.keys():
            settings[param] = default_settings[param]    
    
    tres = settings['tres']
    tres_post = settings['tres_post']
    zres = settings['zres']
    BGIndex = settings['BGIndex']
    MaxAlt = settings['MaxAlt']
    MinAlt = settings['MinAlt']
    
    if not hasattr(date_reference,'year'):
        date_reference = datetime.datetime(year=time_start.year,month=time_start.month,day=time_start.day)
        
    
    #default_basepath = '/scr/eldora1/rsfdata/hsrl/raw/'  # new absolute path
    #default_basepath = '/scr/rain1/rsfdata/projects/socrates/hsrl/raw/'
    default_basepath = '/Users/mhayman/Documents/HSRL/GVHSRL_data/'  # local path
    
    try:
        basepath = paths['basepath']
    except KeyError:
        basepath = default_basepath
        paths['basepath'] = basepath
    #basepath = '/scr/eldora1/HSRL_data/'  # old path - still works with link from HSRL_data to /hsrl/raw/
    #basepath = '/Users/mhayman/Documents/HSRL/GVHSRL_data/'  # local computer data path
    
           
    #default_aircraft_basepath = {
    #    'CSET':'/scr/raf_data/CSET/24_Mar_17_BU/',
    #    'SOCRATES':'/scr/raf_data/SOCRATES/' #SOCRATEStf01.nc       
    #    } 
           
#    # Local Paths
#    default_aircraft_basepath = {
#        'CSET':'/Users/mhayman/Documents/HSRL/aircraft_data/',
#        'SOCRATES':'/Users/mhayman/Documents/HSRL/aircraft_data/' #SOCRATEStf01.nc       
#        } 
    
#    if 'aircraft_basepath' in paths.keys():
#        aircraft_basepath = paths['aircraft_basepath']
#    else:
#        aircraft_basepath = default_aircraft_basepath
    
    """
    Load variable lists
    """
    
    # list of 1D variables to load
    var_1d_list = ['total_energy','RemoveLongI2Cell'\
        ,'TelescopeDirection','TelescopeLocked','polarization','DATA_shot_count','builduptime']  # 'DATA_shot_count'
    
    # list of 2D variables (profiles) to load
    var_2d_list = ['molecular','combined_hi','combined_lo','cross']
    
    # list of aircraft variables to load
    var_aircraft = ['Time','GGALT','ROLL','PITCH','THDG','GGLAT','GGLON','TASX','ATX','PSXC']
    
    # grab calval data from json file
    with open(cal_file,"r") as f:
        cal_json = json.loads(f.read())
    f.close()
    bin0 = lp.get_calval(time_start,cal_json,"Bin Zero")[0]  # bin number where t/range = 0
    
       
    filePathAircraft = paths['filePathAircraft']
    #  load aircraft data    
    air_data,aircraft_t_ref = gv.load_aircraft_data(filePathAircraft,var_aircraft)
#    air_data['Time'] = air_data['Time']+settings['aircraft_time_shift']

    print('Processing: ')
    print('   '+time_start.strftime('%H:%M %d-%b, %Y to'))
    print('   '+time_stop.strftime('%H:%M %d-%b, %Y'))
    print('')
    
    if settings['use_aircraft_tref']:
        date_reference = aircraft_t_ref
    
#    plt.figure()
#    plt.plot(air_data['Time']/3600.0,air_data['TASX'])
#    plt.show()    
    
    flt = process_vars['flt']
    
    if settings['save_data']:
        try:
            save_data_path = paths['save_data_path']
            save_data_file = save_data_path+flt+'_GVHSRL_'+time_start.strftime('%Y%m%dT%H%M')+'_'+time_stop.strftime('%Y%m%dT%H%M')+'.nc'
        except KeyError:
            print('Save data is disabled')
            print('  No save path (save_data_path) is provided')
            settings['save_data'] = False
    
    if settings['save_plots'] or settings['save_mol_gain_plot']:
        try:
            save_plots_path = paths['save_plots_path']
            save_plots_base = flt+'_GVHSRL_'+time_start.strftime('%Y%m%dT%H%M')+'_'+time_stop.strftime('%Y%m%dT%H%M')
        except KeyError:
            print('Save plots is disabled')
            print('  No save path (save_plots_path) is provided')
            settings['save_plots'] = False
    

    
    # grab raw data from netcdf files
    time_list,var_1d_data, profs = gv.load_raw_data(time_start,time_stop,var_2d_list,var_1d_list,basepath=basepath,verbose=True,as_prof=True,loadQWP=settings['loadQWP'],date_reference=date_reference,time_shift=settings['aircraft_time_shift'],bin0=bin0,loadBM3D=settings['use_BM3D'])
    
    if settings['use_BM3D']:
        # if using BM3D data, try to load the pthinned denoised profiles for 
        # filter optimization
        thin_list = ['molecular_pthin_fit_BM3D','molecular_pthin_ver_BM3D']
        _,_, thin_profs = gv.load_raw_data(time_start,time_stop,thin_list,[],basepath=basepath,verbose=True,as_prof=True,loadQWP=settings['loadQWP'],date_reference=date_reference,time_shift=settings['aircraft_time_shift'],bin0=bin0,loadBM3D=True)
        for tvar in thin_profs.keys():
            if isinstance(thin_profs[tvar], lp.LidarProfile):
                print('found '+tvar)
                # check that dimensions agree with other profiles first?
                profs[tvar] = thin_profs[tvar]
            else:
                print(tvar + ' may not exist in the requested data')
                print(thin_profs[tvar].shape)
    
    run_processing = len(profs) > 0  
    
    if run_processing:
        # plot raw profiles
#        lp.plotprofiles(profs)
        # estimate where bin0 ()
        pbin0 = np.sum(profs['molecular'].profile,axis=0)
        try:
            
            ipbin0 = np.nonzero(pbin0 > 4*pbin0[0])[0][0] # locate intial outgoing pulse
            ipbin1 = ipbin0 + np.nonzero(np.diff(pbin0[ipbin0:]) < 0)[0][0] # locate zero crossing after the pulse
    #        print('start index: %d'%ipbin0)
            # interp expects xp to be monotonically increasing.
            # by negating the xp term in the function, we assume a negative slope
            est_bin0 = np.interp(np.zeros(1),-np.diff(pbin0[ipbin0:ipbin1+2]),np.arange(ipbin0,ipbin1+1)+0.5) 
            print('')
            print('Estimated bin0: %f'%est_bin0)
            print('Current bin0: %f'%bin0)
            print('')
            save_other_data['est_bin0']={'data':est_bin0,'description':'estimated MCS bin corresponding to t=0 on the lidar pulse','units':'MCS bin number'}
            save_other_data['bin0'] = {'data':bin0,'description':'actual MCS bin corresponding to t=0 on the lidar pulse used in this processing','units':'MCS bin number'}
            
    #        plt.figure()
    #        plt.plot(np.arange(ipbin0,ipbin0+30)+0.5,np.diff(pbin0[ipbin0:ipbin0+31]))
    #        plt.plot(np.arange(ipbin0,ipbin0+30),pbin0[ipbin0:ipbin0+30])
    #        plt.plot(np.arange(ipbin0,ipbin1+1)+0.5,np.diff(pbin0[ipbin0:ipbin1+2]))
    #        plt.show()
        except IndexError:
            print('No bin0 estimate')
    
    
    while run_processing:
        #execute processing if data was found
    
        timeD = time_list[0]
        time_dt = time_list[1]
        time_sec = time_list[2]
    
        
        # find instances in raw data where I2 cell is removed
        if 'RemoveLongI2Cell' in var_1d_data.keys():
            cal_indices = np.nonzero(var_1d_data['RemoveLongI2Cell'] < 50)[0]
        else:
            cal_indices = []
        
        # find instances where the lidar is not transmitting
        if settings['Remove_Off_Data']:
            _,off_indices = profs['combined_hi'].trim_to_on(ret_index=True,delete=False,SNRlim=settings['SNRlimit'])
            cal_indices = np.unique(np.concatenate((off_indices,cal_indices)))
        
        
        # grab calibration data files
        # down_gain is the molecular gain when the telescope points down
        if settings['hsrl_rb_adjust']:
            mol_gain_up,diff_geo_file,mol_gain_down,diff_geo_file_down = lp.get_calval(time_start,cal_json,'Molecular Gain',cond=[['RB_Corrected','=','True']],returnlist=['value','diff_geo','down_gain','down_diff'])  
        else:
            mol_gain_up,diff_geo_file,mol_gain_down,diff_geo_file_down = lp.get_calval(time_start,cal_json,"Molecular Gain",returnlist=['value','diff_geo','down_gain','down_diff'])
            
        baseline_file = lp.get_calval(time_start,cal_json,"Baseline File")[0]
        diff_pol_file = lp.get_calval(time_start,cal_json,"Polarization",returnlist=['diff_geo'])
        i2_file = lp.get_calval(time_start,cal_json,"I2 Scan")
        dead_time_list = lp.get_calval(time_start,cal_json,"Dead_Time",returnlist=['combined_hi','cross','combined_lo','molecular'])
        dead_time = dict(zip(['combined_hi','cross','combined_lo','molecular'],dead_time_list))
        
        if settings['get_extinction']:
            geo_file_up,geo_file_down = lp.get_calval(time_start,cal_json,"Geo File",returnlist=['value','down_file'])
            geo_up = np.load(cal_file_path+geo_file_up)
            if len(geo_file_down) > 0:
                geo_down = np.load(cal_file_path+geo_file_down)
            else:
                geo_data = geo_up
                
        
        
        # load differential overlap correction
        diff_data_up = np.load(cal_file_path+diff_geo_file)
        if len(diff_geo_file_down) > 0:
            diff_data_down = np.load(cal_file_path+diff_geo_file_down)
        else:
            diff_data = diff_data_up
        
        baseline_data = np.load(cal_file_path+baseline_file)
        
        if len(diff_pol_file):
            diff_pol_data = np.load(cal_file_path+diff_pol_file[0])
        
        # load i2 scan from file
        if len(i2_file):
            i2_data = np.load(cal_file_path+i2_file[0])
        else:
            # if no i2 scan available, don't correct for Rayleigh Brillouin spectrum
            settings['hsrl_rb_adjust'] = False
        
        lidar_location = lp.get_calval(time_start,cal_json,"Location",returnlist=['latitude','longitude'])
        
        
        flight_date = process_vars['flight_date']
        usr_flt = process_vars['usr_flt']
        
        print('flight_date: '+flight_date[usr_flt].strftime('%Y-%b-%d %H:%M'))
        print('date_reference: '+date_reference.strftime('%Y-%b-%d %H:%M'))
        
        # set the master time to match all 2D profiles to
        # (1d data will not be resampled)
        #master_time = np.arange(time_sec[0]-tres/2,time_sec[-1]+tres/2,tres) #
#        sec_start = np.max([time_sec[0],(time_start-flight_date[usr_flt]).total_seconds()])
#        sec_stop = np.min([time_sec[-1],(time_stop-flight_date[usr_flt]).total_seconds()])
#        sec_start = np.max([time_sec[0],(time_start-date_reference).total_seconds()])
#        sec_stop = np.min([time_sec[-1],(time_stop-date_reference).total_seconds()])
        sec_start = (time_start-date_reference).total_seconds()
        sec_stop = (time_stop-date_reference).total_seconds()
        print('found data for')
        print('   %f h-UTC to'%(sec_start/3600.0))
        print('   %f h-UTC'%(sec_stop/3600.0))
        if tres > 0.5:
            master_time = np.arange(sec_start-tres/2,sec_stop+tres/2,tres)
            time_1d,var_1d = gv.var_time_resample(master_time,time_sec,var_1d_data,average=True)
            # estimate the cal grid points that need to be removed in the new time resolution.
            cal_ind_tres = np.unique(np.digitize(profs['combined_hi'].time[cal_indices],master_time))
        else:
            time_1d = time_sec.copy()
            var_1d = var_1d_data.copy()
            cal_ind_tres = cal_indices.copy()
#        print('Time Axes')
#        print(time_1d.shape)
#        print(time_1d[0]/3600.0)
#        print(time_1d[-1]/3600.0)
#        print(time_sec[0]/3600.0)
#        print(time_sec[-1]/3600.0)
#        print(time_sec.shape)
#        print(master_time[0]/3600.0)
#        print(master_time[-1]/3600.0)
#        print(master_time.shape)
#        print(air_data['Time'][0]/3600.0)
#        print(air_data['Time'][-1]/3600.0)
#        print(air_data['Time'].shape)
        
        
        
        air_data_t = gv.interp_aircraft_data(time_1d,air_data)
        
        if settings['RemoveCals']:
            time_1d = np.delete(time_1d,cal_indices)
            var_1d = gv.delete_indices(var_1d,cal_ind_tres)
            air_data_t = gv.delete_indices(air_data_t,cal_ind_tres)
        
        # if there is no valid data don't process this chunk
        if time_1d.size == 0:
            print('No atmospheric data, skipping this data set')
            run_processing = False
            break
#        print(time_1d.size)
#        print(time_1d[0]/3600.0)
#        print(time_1d[-1]/3600.0)
        
        if settings['as_altitude']:
            master_alt = np.arange(MinAlt,MaxAlt+zres,zres)
        
#        print(air_data_t['Time'][0]/3600.0)
#        print(air_data_t['Time'][-1]/3600.0)
#        print(air_data_t['Time'].shape)
#        print(np.max(air_data_t['GGALT']))
#        print(np.min(air_data_t['GGALT']))
#        print(air_data_t['Time'].shape)
        
#        # find instances in raw data where I2 cell is removed
#        if 'RemoveLongI2Cell' in var_1d_data.keys():
#            cal_indices = np.nonzero(var_1d_data['RemoveLongI2Cell'] < 50)[0]
#        else:
#            cal_indices = []
#        
#        # find instances where the lidar is not transmitting
#        if settings['Remove_Off_Data']:
#            _,off_indices = profs['combined_hi'].trim_to_on(ret_index=True,delete=False,SNRlim=settings['SNRlimit'])
#            cal_indices = np.unique(np.concatenate((off_indices,cal_indices)))
#        
#        if settings['RemoveCals']:
#            time_1d = np.delete(time_1d,cal_indices)
#            var_1d = gv.delete_indices(var_1d,cal_indices)
                
        
        
        # time resolution after range to altitude conversion
        if tres_post > 0:
            print('post time res: %f seconds'%tres_post)
            master_time_post = np.arange(sec_start-tres_post/2,sec_stop+tres_post/2,tres_post)
        #    time_post,var_post = gv.var_time_resample(master_time_post,time_sec,var_1d_data,average=True)
        #    air_data_post = gv.interp_aircraft_data(time_post,air_data)
        elif tres > 0.5:
            print('using tres')
            master_time_post = master_time
            time_post = time_1d
            var_post = var_1d
            air_data_post = air_data_t
        else:
            print('no change to time resolution')
            master_time_post = np.arange(sec_start-tres/2,sec_stop+tres/2,tres)
        print('master_time_post limits')
        print('   %f - %f h UTC'%(master_time_post[0]/3600.0,master_time_post[-1]/3600.0))
        print('   %f second resolution'%np.mean(np.diff(master_time_post)))
        print('   %d data points'%master_time_post.size)
        
        ## setup molecular gain vector based on telescope pointing direction
        #mol_gain = np.zeros(var_post['TelescopeDirection'].shape)
        #mol_gain[np.nonzero(var_post['TelescopeDirection']==1.0)] = mol_gain_up
        #mol_gain[np.nonzero(var_post['TelescopeDirection']==0.0)] = mol_gain_down
        #mol_gain = mol_gain[:,np.newaxis]
        
        # setup variable diffierential overlap (up vs down pointing) if supplied
        if len(diff_geo_file_down) > 0:
            diff_data = {}
            key_list = ['hi_diff_geo','lo_diff_geo']
            for var in key_list:
                diff_data[var] = np.ones((var_1d['TelescopeDirection'].size,diff_data_up[var].size))
                diff_data[var][np.nonzero(var_1d['TelescopeDirection']==1.0)[0],:] = diff_data_up[var]
                diff_data[var][np.nonzero(var_1d['TelescopeDirection']==0.0)[0],:] = diff_data_down[var]
           
        
        # setup variable geo overlap (up vs down pointing) if supplied
        if settings['get_extinction']:
            if len(geo_file_down) > 0:
                geo_data = {}
                key_list = ['geo_mol','geo_mol_var','Nprof']
                for var in key_list:
                    if var in geo_up.keys():
                        geo_data[var] = np.ones((var_1d['TelescopeDirection'].size,geo_up[var].size))
                        geo_data[var][np.nonzero(var_1d['TelescopeDirection']==1.0)[0],:] = geo_up[var]
                        if var in geo_down.keys():
                            geo_data[var][np.nonzero(var_1d['TelescopeDirection']==0.0)[0],:] = geo_down[var]
                        else:
                            geo_data[var][np.nonzero(var_1d['TelescopeDirection']==0.0)[0],:] = geo_up[var]
                    else:
                        geo_data[var] = np.ones((var_1d['TelescopeDirection'].size,1))
        
        """
        Main Profile Processing Loop
        loop through each lidar profile in profs and perform basic processing 
        operations
        """
        
        if settings['as_altitude']:
            # maximum range required for this dataset
            range_trim = np.max([np.max(MaxAlt-air_data_t['GGALT']),np.max(air_data_t['GGALT']-MinAlt)])+4*zres
        else:
            # set maximum range to MaxAlt if range centered processing
            range_trim = MaxAlt
        
        int_profs = {}  # obtain time integrated profiles
#        save_profs = {}
        tv_denoise = {}
        raw_profs = {}
        for var in profs.keys():
            if settings['RemoveCals']:
                # remove instances where the I2 cell is removed
                profs[var].remove_time_indices(cal_indices)
        #        profs[var].trim_to_on()  # remove points where lidar isn't transmitting
            if tres > 0.5:
                profs[var].time_resample(tedges=master_time,update=True,remainder=False)
#            int_profs[var] = profs[var].copy()
#            int_profs[var].time_integrate()
            
            if settings['deadtime_correct'] and var in dead_time.keys():
#                if var == 'combined_hi':
#                    p_before = profs[var].copy()
                if hasattr(profs[var],'NumProfsList') and (var in dead_time.keys()):
                    profs[var].nonlinear_correct(dead_time[var],laser_shot_count=2000*profs[var].NumProfsList[:,np.newaxis],std_deadtime=5e-9)
                else:
                    # number of laser shots is based on an assumption that there is one 0.5 second profile per time bin
                    profs[var].nonlinear_correct(dead_time[var],laser_shot_count=2000,std_deadtime=5e-9)
                    
            if 'molecular_pthin' in var:
                if 'fit' in var:
                    fit_mol = profs[var].copy()

                elif 'ver' in var:
                    ver_mol = profs[var].copy()


            if settings['save_raw']:
                raw_profs[var]=profs[var].copy()
                raw_profs[var].label = 'Raw '+raw_profs[var].label
#                if var == 'combined_hi':
#                    p_after = profs[var].copy()
#                    lp.plotprofiles([p_before,p_after],varplot=True,time=18.3*3600)
#                    lp.plotprofiles([p_after],varplot=True,time=18.1*3600)
#            if settings['time_denoise'] and var != 'combined_lo':
            if settings['time_denoise'] and var == 'molecular':
                print('Running temporal denoising on '+var)
                #save_profs[var] = profs[var].copy()
                # save the profile before it is denoised
                profs[var+'_raw'] = profs[var].copy()

                profs[var],tv_denoise[var] = gv.DenoiseTime(profs[var],MaxAlt=min([range_trim,settings['time_denoise_max_range']]),n=1,
                    verbose=settings['time_denoise_verbose'],accel = settings['time_denoise_accel'],tv_lim =[0.10, 2.0],N_tv_pts=24,
                    eps_opt=settings['time_denoise_eps'],plot_result=settings['time_denoise_debug_plots'],
                    MinAlt=300)  # 300.0
                
                # save the denoised profile
                profs[var+'_denoise'] = profs[var].copy()
            
            if var == 'molecular' and settings['Denoise_Mol']:
                MolRaw = profs['molecular'].copy() 
                
            if var == 'molecular' and settings['get_extinction']:
                mol_ext = profs['molecular'].copy()
                if (not 'molecular_pthin_fit_BM3D' in profs.keys()) or (not 'molecular_pthin_ver_BM3D' in profs.keys()):
                    print('No Poisson thinned data provided.  Poisson thinning for extinction estimation now.')
                    mol_ext.multiply_piecewise(mol_ext.NumProfList[:,np.newaxis])
                    fit_mol,ver_mol = mol_ext.p_thin()
    
            if settings['baseline_subtract']:        
                # baseline subtract  profiles
                profs[var].baseline_subtract(baseline_data['save_data'][var]['fit'], \
                    baseline_var = baseline_data['save_data'][var]['variance'], \
                    tx_norm=var_1d['total_energy'][:,np.newaxis]/baseline_data['avg_energy'])
                    
            
            # background subtract the profile
            profs[var].bg_subtract(BGIndex)
#            if settings['time_denoise'] and var != 'combined_lo':
#                [log_tune,valid_val] = mle.DenoiseBG(profs[var],BGIndex,verbose=False,plot_sol = False, tv_lim =[0.10, 2.0],N_tv_pts=24)
#            else:
#                profs[var].bg_subtract(BGIndex)
            
            # profile specific processing routines
            if var == 'combined_hi' and settings['diff_geo_correct']:
                if settings['Estimate Time Shift']:
                    print('Estimate Time Shift enabled')
                    # if the software settings state we are supposed to check for time shift, look to see if there are any
                    # roll or pitch manuevers that will help us do that.
                    # this is determined by looking at the derivative signals over a long strech of time
                    weights = np.convolve(np.ones(2000)*0.5e-3,np.concatenate((np.zeros(1),np.diff(air_data_t['ROLL'])**2+np.diff(air_data_t['PITCH'])**2)),'same')
#                    print(weights.max())
#                    plt.figure()
#                    plt.plot(weights)
#                    plt.show()
                    if (weights > 0.02).any():
                        print('    Manuevers found')
                        print('    Estimating time shift between aircraft and HSRL time')
                        
                        t_dir_set = np.sign(var_1d['TelescopeDirection']-0.5)
                        # calculate the expected location of the sea surface
                        Rg_exp = air_data_t['GGALT']/(np.cos((air_data_t['ROLL']-4.0*t_dir_set)*np.pi/180)*np.cos((air_data_t['PITCH'])*np.pi/180))
                        # and the corresponding profile index of that expected sea surface location
                        iRg_exp = np.argmin(np.abs(Rg_exp[:,np.newaxis]-profs['combined_hi'].range_array[np.newaxis,:]),axis=1).astype(np.int)                
                        
                        # define a range around the expected sea surface location to look for a ground return
                        iRg_min = iRg_exp - 80
                        iRg_min[np.nonzero(iRg_min < 0)] = 0
                        iRg_max = iRg_exp + 80
                        iRg_max[np.nonzero(iRg_max >= profs['combined_hi'].range_array.size)] = profs['combined_hi'].range_array.size -1

                        iRg = np.zeros(iRg_exp.size,dtype=np.int)
                        Rg_SNR = np.zeros(iRg_exp.size)
                        Rg_count = np.zeros(iRg_exp.size)
#                        Rg_weight = np.exp(-(np.arange(iRg_exp.size)-20)**2/20**2)
                        for ri in range(iRg_exp.size):
                            Rg_weight = np.exp(-(np.arange(iRg_min[ri],iRg_max[ri])-iRg_exp[ri])**2/40**2)
                            # find the sea return by just looking for the largest backscatter signal in the specified range
                            iRg[ri] = np.int(np.argmax(Rg_weight*profs['combined_hi'].profile[ri,iRg_min[ri]:iRg_max[ri]])+iRg_min[ri])
                            if var_1d['TelescopeDirection'][ri] <= 0:
                                # if the telescope is pointing up, treat the estimate as valid
                                # and estimate the signal to noise from it
                                Rg_std = np.sqrt(np.var(profs['combined_hi'].profile[ri,iRg_min[ri]:iRg[ri]])+np.var(profs['combined_hi'].profile[ri,iRg[ri]+1:iRg_max[ri]]))
                                Rg_SNR[ri] = profs['combined_hi'].profile[ri,iRg[ri]]/Rg_std #np.std(profs['combined_hi'].profile[ri,iRg_min[ri]:iRg_max[ri]])
                                Rg_count[ri] = profs['combined_hi'].profile[ri,iRg[ri]]
                        
                        # get the ranges associated with the sea returns
                        Rg = profs['combined_hi'].range_array[iRg]
                        # remove data points with low SNR
                        Rg_filt = Rg.copy()
                        Rg_out = np.nonzero(Rg_SNR < 7.0)
#                        Rg_keep = np.nonzero(Rg_SNR >=7.0)
                        Rg_filt[Rg_out] = np.nan
#                        Rg_interp = np.interp(np.arange(Rg.size),Rg_keep[0],Rg_filt[Rg_keep])
                        
                        
                        # estimate the time offset by shifting the data in time and calculating the
                        # mean squared error
                        # weight the error by the amount of manuevers taking place in that time period
                        i_offset = np.arange(-10,10,0.25)
                        Rg_corr1 = np.zeros(i_offset.size)
                        Rg_corr2 = np.zeros(i_offset.size)
                        Rg_corr3 = np.zeros(i_offset.size)
                        try:
                            for ri in range(i_offset.size):
            
                                i0 = np.int(np.ceil(np.abs(i_offset[ri])))
            
                                
                                if i_offset[ri] < 0:
                                    xinterp = np.arange(Rg_exp.size-i0)
                                    Rg1 = np.interp(xinterp,np.arange(Rg_exp.size)-i_offset[ri],Rg_exp)
                                    weights_Rg1 = np.interp(xinterp,np.arange(Rg_exp.size)-i_offset[ri],weights)
                                    Rg2 = Rg_filt.flatten()[:-1*i0]
                                else:
                                    xinterp = np.arange(Rg_exp.size-i0)+i0
                                    Rg1 = np.interp(xinterp,np.arange(Rg_exp.size)-i_offset[ri],Rg_exp)
                                    weights_Rg1 = np.interp(xinterp,np.arange(Rg_exp.size)-i_offset[ri],weights)
                                    Rg2 = Rg_filt.flatten()[i0:]
    
                                Rg_corr1[ri] = np.nanmean(weights_Rg1 * (Rg1-Rg2)**2)
                                Rg_corr2[ri] = np.nanstd(Rg1-Rg2)
    #                            Rg_corr3[ri] = np.nanmean(((Rg1-np.nanmean(Rg1))*np.nanstd(Rg1)*(Rg2-np.nanmean(Rg2))*np.nanstd(Rg2)))
    #                            Rg_corr3[ri] = np.nanmean(np.diff(Rg1)*np.diff(Rg2))
                                Rg_corr3[ri] = np.nanmean((np.diff(Rg1)-np.diff(Rg2))**2/profs['combined_hi'].mean_dt**2)
                                
                            Rg_tot = Rg_corr3/np.nanmean(Rg_corr3)+Rg_corr1/np.nanmean(Rg_corr1)/4
                            i_min_tot = np.argmin(Rg_tot)
    #                        x_0x = 0.5*(i_offset[:-1]+i_offset[1:])
    #                        i_t_off = np.interp(np.zeros(1),np.diff(Rg_corr1),x_0x)
                            x_0x = 0.5*(i_offset[i_min_tot-1:i_min_tot+1]+i_offset[i_min_tot:i_min_tot+2])
                            i_t_off_tot = np.interp(np.zeros(1),np.diff(Rg_tot)[i_min_tot-1:i_min_tot+1],x_0x)  # mimimum of total error function
                            
                            i_min_lms = np.argmin(Rg_corr1)
                            x_0x = 0.5*(i_offset[i_min_lms-1:i_min_lms+1]+i_offset[i_min_lms:i_min_lms+2])
                            i_t_off_lms = np.interp(np.zeros(1),np.diff(Rg_tot)[i_min_lms-1:i_min_lms+1],x_0x)  # minimum of lms error
                            
                            i_min_deriv = np.argmin(Rg_corr3)
                            x_0x = 0.5*(i_offset[i_min_deriv-1:i_min_deriv+1]+i_offset[i_min_deriv:i_min_deriv+2])
                            i_t_off_deriv = np.interp(np.zeros(1),np.diff(Rg_tot)[i_min_deriv-1:i_min_deriv+1],x_0x)  # minimum from derivative lms
                            
                            print('')
                            print('Current time offset: %f'%settings['aircraft_time_shift'])
                            print('Estimated time offset (total): %f s'%(i_t_off_tot*profs['combined_hi'].mean_dt+settings['aircraft_time_shift']))
                            print('Estimated time offset (attitude fit): %f s'%(i_t_off_lms*profs['combined_hi'].mean_dt+settings['aircraft_time_shift']))
                            print('Estimated time offset (derivative attitude fit): %f s'%(i_t_off_deriv*profs['combined_hi'].mean_dt+settings['aircraft_time_shift']))
                            print('')
                            save_other_data['time_offset_total']={'data':i_t_off_tot*profs['combined_hi'].mean_dt+settings['aircraft_time_shift'],'description':'estimated time offset between aircraft and HSRL data systems based on combined aircraft attitude and derivative of aircraft attitude signals','units':'seconds'}
                            save_other_data['time_offset_lms']={'data':i_t_off_lms*profs['combined_hi'].mean_dt+settings['aircraft_time_shift'],'description':'estimated time offset between aircraft and HSRL data systems based on combined aircraft attitude signal','units':'seconds'}
                            save_other_data['time_offset_deriv']={'data':i_t_off_deriv*profs['combined_hi'].mean_dt+settings['aircraft_time_shift'],'description':'estimated time offset between aircraft and HSRL data systems based on derivative of aircraft attitude signal','units':'seconds'}
                            save_other_data['time_offset']={'data':settings['aircraft_time_shift'],'description':'time offset between aircraft and HSRL data systems used in processing this dataset','units':'seconds'}
    #                    Rg_corr1[ri] = np.nanmean((Rg1-Rmean)*(Rg2-Rmean))
                        
                            textstr = 'Current time offset: %f s\n'%settings['aircraft_time_shift'] + \
                                'Estimated time offset (total): %f s\n'%(i_t_off_tot*profs['combined_hi'].mean_dt + settings['aircraft_time_shift'])  + \
                                'Estimated time offset (lms): %f s\n'%(i_t_off_lms*profs['combined_hi'].mean_dt + settings['aircraft_time_shift'])  + \
                                'Estimated time offset (derivative lms): %f s\n'%(i_t_off_deriv*profs['combined_hi'].mean_dt + settings['aircraft_time_shift'])
    #                            'New Range Std: %f m\n'%Rg_corr2.min()
                            # piggy back on save_mol_gain_plot setting
                            
                            plt.figure()
                            plt.subplot(211)
                            plt.plot(-1*i_offset,np.sqrt(Rg_tot),label='Total Error')
                            plt.plot(-1*i_offset,np.sqrt(Rg_corr3/np.nanmean(Rg_corr3)),label='Derivative Error')
                            plt.plot(-1*i_offset,np.sqrt(Rg_corr1/np.nanmean(Rg_corr1)),label='Weighted Error')
                            plt.plot(-1*i_offset[i_min_tot],np.sqrt(Rg_tot[i_min_tot]),'k.')
                            plt.plot(-1*i_offset[i_min_lms],np.sqrt(Rg_corr1[i_min_lms]/np.nanmean(Rg_corr1)),'k.')
                            plt.plot(-1*i_offset[i_min_deriv],np.sqrt(Rg_corr3[i_min_deriv]/np.nanmean(Rg_corr3)),'k.')
                            plt.grid(b=True)
                            plt.text(np.mean(i_offset),np.mean(plt.ylim()),textstr,verticalalignment='center',horizontalalignment='center',size=9)
                            plt.xlabel('Aircraft Time shift [s]')
                            plt.ylabel('Weighted RMS Error')
                            plt.legend(fontsize=9)
    
    ##                        plt.figure()
    #                        fig,ax1 = plt.subplots()
    #                        ax1.plot(-1*i_offset,np.sqrt(Rg_corr3),label='Derivative Error',color='b')
    #                        ax1.plot(-1*i_offset[i_min],np.sqrt(Rg_corr3[i_min]),'kx')
    #                        ax2=ax1.twinx()
    #                        ax2.plot(-1*i_offset,np.sqrt(Rg_corr1),label='Weighted Error',color='r')
    #                        ax1.set_xlabel('Aircraft Time shift [s]')
    #                        ax1.set_ylabel('RMS Time Derivative Error [m/s]',color='b')
    #                        ax2.set_ylabel('RMS Error [m]',color='r')
    #                        plt.grid(b=True)
    #                        plt.text(np.mean(i_offset),np.mean(np.sqrt(Rg_corr1)),textstr,verticalalignment='center',horizontalalignment='center')
        #                        plt.text(0.95*(np.max(bbsr)-np.min(bbsr))+np.min(bbsr),0.95*(np.max(balt)-np.min(balt))+np.min(balt),text_str,color='white',fontsize=8,verticalalignment='top',horizontalalignment='right')
    
#                            if settings['save_mol_gain_plot']:
#                                # double zero at beginning of filename just to put plots at the front of a sorted list of profiles
#                                plt.savefig(save_plots_path+'01_Time_Delay_Estimate_'+save_plots_base,dpi=300)
                                
                                
                #                plt.plot(Rg_corr2)
    #                        plt.figure()
    #                        plt.plot(-1*i_offset,np.sqrt(Rg_corr3),label='Cross Correlation')
    #                        plt.title('Cross Correlation')
                                
            #                Rg_filt[np.nonzero(Rg_count < 50)] = np.nan
#                            plt.figure()
                            plt.subplot(212)
                            plt.plot(air_data_t['Time']/3600.0,Rg_exp,label='aircraft data estimate')
    #                        plt.plot(air_data_t['Time']/3600.0,Rg,label='ground return (unfiltered)')
    #                        plt.plot(Rg_interp,'--',label='From Ground Return')
                            plt.plot(air_data_t['Time']/3600.0,Rg_filt,'r.-',label='ground return (filtered)')
                            plt.xlabel('Flight Time [h]')
                            plt.ylabel('Range to Surface [m]')
                            plt.grid(b=True)
                            plt.legend(fontsize=9)
#                            if settings['save_mol_gain_plot']:
#                                plt.savefig(save_plots_path+'02_Range_to_Surface_'+save_plots_base,dpi=300)
                                
                            if settings['save_mol_gain_plot']:
                                # double zero at beginning of filename just to put plots at the front of a sorted list of profiles
                                plt.savefig(save_plots_path+'01_Time_Delay_Estimate_'+save_plots_base,dpi=300)
                                
    #                        plt.figure()
    #                        plt.plot(Rg_SNR)
    #                        plt.plot(Rg_count)
    #                        plt.show()
                            
    #                        plt.figure()
    #                        plt.plot(profs['combined_hi'].profile[325,iRg_min[325]:iRg_max[325]])
    #                        plt.figure()
    #                        plt.plot(weights*Rg_exp)
    #                        plt.show()
                        except:
                            print('   Skipping due to data alignment error (probably not enough valid ground returns)')
                    else:
                        print('  Manuevers not found')
                
                
                profs[var].diff_geo_overlap_correct(diff_data['hi_diff_geo'],geo_reference='molecular')
            elif var == 'combined_lo' and settings['diff_geo_correct']:
                profs[var].diff_geo_overlap_correct(diff_data['lo_diff_geo'],geo_reference='molecular')
                try:
                    profs[var].gain_scale(1.0/diff_data['lo_norm'])
                except KeyError:
                    print('No lo gain scale factor')
                    print('   lo gain channel may need adjustment')
            elif var == 'cross' and settings['diff_geo_correct']:
                if len(diff_pol_file):
                    profs[var].diff_geo_overlap_correct(diff_pol_data['cross_diff_geo'],geo_reference='combined_hi')
                    profs[var].diff_geo_overlap_correct(diff_data['hi_diff_geo'],geo_reference='molecular')  
                else:
                    profs[var].diff_geo_overlap_correct(diff_data['hi_diff_geo'],geo_reference='molecular')
            elif var == 'molecular' and settings['get_extinction'] and settings['as_altitude']:
                # if retrieving extinction, use a range centered profile to obtain it
                mol_ext = profs['molecular'].copy()
                mol_ext.multiply_piecewise(geo_data['geo_mol'])
                mol_ext.slice_range(range_lim=[0,range_trim])
                if tres_post > 0 or tres <= 0.5:
                    mol_ext.time_resample(tedges=master_time_post,update=True,remainder=False)
                
                
            profs[var].slice_range(range_lim=[settings['range_min'],range_trim])
          
            if settings['as_altitude']:
                profs[var].range2alt(master_alt,air_data_t,telescope_direction=var_1d['TelescopeDirection'])
                if settings['save_raw']:
                    raw_profs[var].range2alt(master_alt,air_data_t,telescope_direction=var_1d['TelescopeDirection'])
               
            
            if tres_post > 0 or tres <= 0.5:
#                print(var+' resample:')
#                print('  [start,end] = [%f,%f] h UTC'%(profs[var].time[0]/3600.0,profs[var].time[-1]/3600.0))
#                print('  resolution = %f sec'%profs[var].mean_dt)
#                print('  %d data points'%profs[var].profile.shape[0])
                profs[var].time_resample(tedges=master_time_post,update=True,remainder=False)
                if settings['save_raw']:
                    raw_profs[var].time_resample(tedges=master_time_post,update=True,remainder=False)
            if profs[var].profile.size == 0:
                run_processing = False
                print('Processing will terminate.  Profile of ' +var+ ' is empty')
#                print('  -- Now --')
#                print('  [start,end] = [%f,%f]'%(profs[var].time[0]/3600.0,profs[var].time[-1]/3600.0))
#                print('  resolution = %f sec'%profs[var].mean_dt)
#                print('  %d data points'%profs[var].profile.shape[0])
#                print(profs[var].ProcessingStatus)
                
            else:
                int_profs[var] = profs[var].copy()
                int_profs[var].time_integrate()
        if not run_processing:
            print('Terminating processing due to empty profiles')
            break
        
#        print('optimizing extinction filter design')
#        settings['ext_sg_width'],settings['ext_sg_order']=lp.optimize_sg_raw(raw_profs['molecular'],axis=1,full=True,order=[1,5],window=[3,23],range_lim=[300,5e3])
##        ext_sg_wid = settings['ext_sg_width']
##        ext_sg_order = settings['ext_sg_order']
#        if not hasattr(settings['ext_sg_width'],'__iter__'):
#            print('   window width: %d'%settings['ext_sg_width'])
#            print('   order: %d'%settings['ext_sg_order'])

        
        # merge high and low gain combined profiles if option is true
        if settings['merge_hi_lo']:
            profs['combined'],_ = gv.merge_hi_lo(profs['combined_hi'],profs['combined_lo'],plot_res=False)
        else:
            # if merging is not enabled, use combined high for all calculations
            profs['combined'] = profs['combined_hi']
        
        if settings['mol_smooth']:
            profs['molecular'].conv(settings['t_mol_smooth']/profs['molecular'].dt,settings['z_mol_smooth']/profs['molecular'].mean_dR)
        
        # reformulate the master time based on the time that appears in the profiles
        # after processing
        if tres_post > 0:
#            print(master_time_post.shape)
            
            master_time_post = np.concatenate((np.array([profs['molecular'].time[0]-tres_post*0.5]), \
                0.5*np.diff(profs['molecular'].time)+profs['molecular'].time[:-1], \
                np.array([profs['molecular'].time[-1]+tres_post*0.5])))
#            print(master_time_post.shape)
            time_post,var_post = gv.var_time_resample(master_time_post,time_sec,var_1d_data,average=True)
            air_data_post = gv.interp_aircraft_data(time_post,air_data)
        else:
            time_post = time_1d
            var_post = var_1d
            air_data_post = air_data_t
        
        # setup molecular gain vector based on telescope pointing direction
        mol_gain = np.zeros(var_post['TelescopeDirection'].shape)
        mol_gain[np.nonzero(var_post['TelescopeDirection']==1.0)] = mol_gain_up
        mol_gain[np.nonzero(var_post['TelescopeDirection']==0.0)] = mol_gain_down
        mol_gain = mol_gain[:,np.newaxis]
        
        #if load_reanalysis:
        #    pres,temp = ex.load_fixed_point_NCEP_TandP(profs['molecular'],lidar_location,reanalysis_path)
        
        
        lp.plotprofiles(profs)
        if settings['as_altitude']:
            temp,pres = gv.get_TP_from_aircraft(air_data,profs['molecular'])
        else:
            temp,pres = gv.get_TP_from_aircraft(air_data,profs['molecular'],telescope_direction=var_post['TelescopeDirection'])
        beta_m = lp.get_beta_m(temp,pres,profs['molecular'].wavelength)
        
        nLidar = gv.lidar_pointing_vector(air_data_post,var_post['TelescopeDirection'],lidar_tilt=4.0)
        # plots for evaluating lidar pointing direction
#        plt.figure() 
#        plt.subplot(211)
#        plt.plot(nLidar.T)
#        plt.plot(np.sqrt(np.sum(nLidar**2,axis=0)),'k')
#        plt.subplot(212)
#        plt.plot(air_data_post['ROLL'])
#        plt.plot(air_data_post['PITCH'])
#        plt.plot(air_data_post['THDG'])
#        plt.figure()
#        plt.scatter(air_data_post['ROLL'],air_data_post['THDG'],c=np.sqrt(np.sum(nLidar**2,axis=0))-1)
#        plt.colorbar()
#        plt.show()

        save_other_data['lidar_pointing']={'data':nLidar,'description':'Lidar pointing vector in global coordinate frame. index 0 = North, index 1 = East, index 2 = Down','units':'none'}
        
        if (settings['get_extinction']  or settings['Denoise_Mol']) and settings['as_altitude']:
#            print(var_post['TelescopeDirection'].size)
#            print(mol_ext.profile.shape)
#            print(profs['molecular'].time.size)
#            print(time_post.size)
    #        temp_ext,pres_ext = gv.get_TP_from_aircraft(air_data,mol_ext,telescope_direction=var_1d['TelescopeDirection'])
            temp_ext,pres_ext = gv.get_TP_from_aircraft(air_data,mol_ext,telescope_direction=var_post['TelescopeDirection'])
            beta_m_ext = lp.get_beta_m(temp_ext,pres_ext,profs['molecular'].wavelength)
        else:
            beta_m_ext = beta_m.copy()
        
        
        if settings['Denoise_Mol']:
            
            if len(geo_file_down) > 0:
                geo_denoise = {}
                key_list = ['geo_mol','geo_mol_var','Nprof']
                for var in key_list:
                    if var in geo_up.keys():
                        geo_denoise[var] = np.ones((var_1d['TelescopeDirection'].size,geo_up[var].size))
                        geo_denoise[var][np.nonzero(var_1d['TelescopeDirection']==1.0)[0],:] = geo_up[var]
                        if var in geo_down.keys():
                            geo_denoise[var][np.nonzero(var_1d['TelescopeDirection']==0.0)[0],:] = geo_down[var]
                        else:
                            geo_denoise[var][np.nonzero(var_1d['TelescopeDirection']==0.0)[0],:] = geo_up[var]
                    else:
                        geo_data[var] = np.ones((var_1d['TelescopeDirection'].size,1))            
                geo_denoise['range_array'] = geo_up['range_array'].copy()
                
            print('Denoising Molecular Channel')
            MolDenoise,tune_list = gv.DenoiseMolecular(MolRaw,beta_m_sonde=beta_m_ext.copy(), \
                                    MaxAlt=range_trim,accel = settings['denoise_accel'],tv_lim =[1.5, 2.2],N_tv_pts=21, \
                                    bg_index=-10,n=1,geo_data=geo_denoise,geo_key='geo_mol',verbose=False, \
                                    plot_result=settings['denoise_debug_plots'],eps_opt=settings['denoise_eps']) # dict(geo_prof=np.array([2e14])), geo_data=geo_data,geo_key='geo_mol'  #tv_lim =[1.5, 2.8],N_tv_pts=59
#            # testing and debugging
#            MolRaw.bg_subtract(-10)
#            lp.plotprofiles([MolRaw,MolDenoise],time=22.12*3600)
#            lp.plotprofiles([MolRaw,MolDenoise],time=22.04*3600)
#            plt.show()
            
            MolDenoise.slice_range(range_lim=[0,range_trim])

            if settings['as_altitude']:
                MolDenoise.range2alt(master_alt,air_data_t,telescope_direction=var_1d['TelescopeDirection'])
            if tres_post > 0 or tres <= 0.5:
                MolDenoise.time_resample(tedges=master_time_post,update=True,remainder=False)
               
            
#            if tres_post > 0 or tres <= 0.5:
#                MolDenoise.time_resample(tedges=master_time_post,update=True,remainder=False)
        
        if settings['hsrl_rb_adjust']:
            print('Obtaining Rayleigh-Brillouin Correction')
            dnu = 20e6  # resolution
            nu_max = 10e9 # max frequency relative to line center
            nu = np.arange(-nu_max,nu_max,dnu)
            
            if 'mol_fit' in i2_data.keys():
                Ti2 = np.interp(nu,i2_data['freq']*1e9,i2_data['mol_fit'])  # molecular transmission
                Tam = np.interp(0,i2_data['freq']*1e9,i2_data['mol_fit'])  # aersol transmission into molecular channel
                
                Tc2 = np.interp(nu,i2_data['freq']*1e9,i2_data['comb_fit'])  # combined transmission
                Tac = np.interp(0,i2_data['freq']*1e9,i2_data['comb_fit'])  # aersol transmission into combined channel
                
                mol_gain = mol_gain*(i2_data['combined_mult']/i2_data['molecular_mult'])
            else:
                Ti2 = np.interp(nu,i2_data['freq']*1e9,i2_data['mol_scan'])  # molecular transmission
                Tam = np.interp(0,i2_data['freq']*1e9,i2_data['mol_scan'])  # aersol transmission into molecular channel
                
                Tc2 = np.interp(nu,i2_data['freq']*1e9,i2_data['combined_scan'])  # combined transmission
                Tac = np.interp(0,i2_data['freq']*1e9,i2_data['combined_scan'])  # aersol transmission into combined channel
            
            
            
            [eta_i2,eta_c] = lp.RB_Efficiency([Ti2,Tc2],temp.profile.flatten(),pres.profile.flatten()*9.86923e-6,profs['molecular'].wavelength,nu=nu,norm=True,max_size=10000)
            
        #    beta_mol_norm = lp.RB_Spectrum(temp.profile.flatten(),pres.profile.flatten()*9.86923e-6,profs['molecular'].wavelength,nu=nu,norm=True)
        #    eta_i2 = np.sum(Ti2[:,np.newaxis]*beta_mol_norm,axis=0)
            eta_i2 = eta_i2.reshape(temp.profile.shape)
            
            
            profs['molecular'].gain_scale(mol_gain,gain_var = (mol_gain*0.05)**2)
        
                
        #    eta_c = np.sum(Tc2[:,np.newaxis]*beta_mol_norm,axis=0)
            eta_c = eta_c.reshape(temp.profile.shape)
            
#            print('Tam = %f'%Tam)
#            print('Tac = %f'%Tac)
#            plt.figure()
#            plt.pcolor(eta_i2)
#            plt.colorbar()
#            plt.title('\eta_mm')
#            plt.figure()
#            plt.pcolor(eta_c)
#            plt.colorbar()
#            plt.title('\eta_mc')
#            plt.show()

            beta_a,dPart,BSR,param_profs = gv.AerosolBackscatter(profs['molecular'],profs['combined'],profs['cross'],beta_m, \
                eta_am=Tam,eta_ac=Tac,eta_mm=eta_i2,eta_mc=eta_c,eta_x=0.0,gm=1.0)            
            
            
#            profs['molecular'].multiply_piecewise(1.0/eta_i2)
#            profs['combined'].multiply_piecewise(1.0/eta_c)
#            profs['cross'].multiply_piecewise(1.0/eta_c)
            
            if settings['Denoise_Mol']:
#                MolDenoise.multiply_piecewise(1.0/eta_i2)
                MolDenoise.gain_scale(mol_gain,gain_var = (mol_gain*0.05)**2)
            if settings['get_extinction']:    
                if not settings['as_altitude']:
                    mol_ext = profs['molecular'].copy()
                    eta_i2_ext = eta_i2
                else:
                    [eta_i2_ext] = lp.RB_Efficiency([Ti2],temp_ext.profile.flatten(),pres_ext.profile.flatten()*9.86923e-6,profs['molecular'].wavelength,nu=nu,norm=True,max_size=10000)
                    eta_i2_ext = eta_i2_ext.reshape(temp_ext.profile.shape)
                
                mol_ext.multiply_piecewise(1.0/eta_i2_ext)
                mol_ext.range_correct()
#            else:
#                mol_ext = profs['molecular'].copy()

                
        else:
            # Rescale molecular channel to match combined channel gain
            profs['molecular'].gain_scale(mol_gain,gain_var = (mol_gain*0.05)**2)
            if settings['Denoise_Mol']:
                MolDenoise.gain_scale(mol_gain,gain_var = (mol_gain*0.05)**2)
        
#        beta_a = lp.AerosolBackscatter(profs['molecular'],(profs['combined']+profs['cross']),beta_m)
        if settings['Denoise_Mol']:
            beta_a_denoise,dPart_denoise,BSR_denoise,param_profs_denoise = gv.AerosolBackscatter(MolDenoise,profs['combined'],profs['cross'],beta_m, \
                eta_am=Tam,eta_ac=Tac,eta_mm=eta_i2,eta_mc=eta_c,eta_x=0.0,gm=1.0)    
#            beta_a_denoise = lp.AerosolBackscatter(MolDenoise,(profs['combined']+profs['cross']),beta_m)
            beta_a_denoise.descript = 'Poisson total variation denoised calibrated measurement of Aerosol Backscatter Coefficient in m^-1 sr^-1'
            beta_a_denoise.label = 'Denoised Aerosol Backscatter Coefficient'
        
        if settings['get_extinction']:
            print('Estimating Extinction')
#            ext_sg_wid = settings['ext_sg_width']
#            ext_sg_order = settings['ext_sg_order']
#            ext_tres = settings['ext_tres']
#            ext_zres = settings['ext_zres']
            
#            beta_m_ext = beta_m.copy()
            beta_m_ext.slice_range(range_lim=[settings['range_min'],range_trim])
            
#            print('geo_pre')
#            print(geo_data['geo_mol'].shape)
            
#            plt.figure()
#            plt.plot(fit_mol.profile[10,:])
#            plt.plot(ver_mol.profile[10,:])
            
            t_geo = fit_mol.time.copy()
            r_geo = fit_mol.range_array.copy()
            
            
            if tres_post > 0 or tres <= 0.5:
                fit_mol.time_resample(tedges=master_time_post,update=True,remainder=False,average=False)
                ver_mol.time_resample(tedges=master_time_post,update=True,remainder=False,average=False)
#                beta_m_ext.time_resample(tedges=master_time_post,update=True,remainder=False,average=True)
            
            ext_time_filt = fit_mol.profile.shape[0] > 120
    
            ext_range_filt = fit_mol.profile.shape[1] > 200

            
            if not settings['use_BM3D'] or (not 'molecular_pthin_fit_BM3D' in profs.keys() and not 'molecular_pthin_ver_BM3D' in profs.keys()):
                # if not using BM3D, run a filter optimization on the photon counts
                print('Using filter optimization on raw profiles instead of BM3D')
                # check to make sure the profiles are of reasonable size before trying to optimize a filter for them
                if ext_time_filt:
                    t_win_ord=lp.optimize_sg_raw(fit_mol,axis=0,full=False,order=[1,5],window=[3,23],range_lim=[],bg_subtract=True,AdjCounts=False)    
                    t_window0=t_win_ord[0]['window']
                    t_ord0=t_win_ord[0]['order']
                if ext_range_filt:
                    r_win_ord=lp.optimize_sg_raw(fit_mol,axis=1,full=False,order=[1,5],window=[3,23],range_lim=[settings['range_min'],range_trim],bg_subtract=True,AdjCounts=False)
                    r_window0=t_win_ord[0]['window']
                    r_ord0=t_win_ord[0]['order']
                    fit_mol.sg_filter(r_window0,r_ord0,axis=1)
                    ver_mol.sg_filter(r_window0,r_ord0,axis=1)
                
                if ext_time_filt:
                    fit_mol.sg_filter(t_window0,t_ord0,axis=0)
                    ver_mol.sg_filter(t_window0,t_ord0,axis=0)
                    
#            plt.plot(fit_mol.profile[10,:])
#            plt.plot(ver_mol.profile[10,:])
            
            # update the geo array using nearest neighbor
            igeo_t = np.argmin(np.abs(t_geo[:,np.newaxis]-fit_mol.time[np.newaxis,:]),axis=0)
            geo_new = geo_data['geo_mol'][igeo_t,:]
#            t_geo = fit_mol.time.copy()                    
                    
#            if t_geo.size > fit_mol.time.size:
#                igeo_t = np.nonzero(np.in1d(np.round(2*t_geo).astype(np.int),np.round(2*fit_mol.time).astype(np.int)))[0]
#                geo_new = geo_data['geo_mol'][igeo_t[0]:igeo_t[-1]+1,:]
#                geo_trim_case = 0
#                print('extinction geo case 0')
#                print(t_geo.size)
#                print(fit_mol.time.size)
#                print(igeo_t.shape)
#            elif t_geo.size == fit_mol.time.size:
#                igeo_t = np.nonzero(np.in1d(np.round(2*t_geo).astype(np.int),np.round(2*fit_mol.time).astype(np.int)))[0]
#                igeo_t2 = np.nonzero(np.in1d(np.round(2*fit_mol.time).astype(np.int),np.round(2*t_geo).astype(np.int)))[0]
#                geo_new = np.ones(fit_mol.profile.shape)
##                print('geo 1');
##                print(geo_new.shape)
#                geo_new[igeo_t2,:]=geo_data['geo_mol'][igeo_t,:]
##                print('geo 2')
##                print(geo_new.shape)
#                t_geo = fit_mol.time.copy()
#                geo_trim_case = 1
#                print('extinction geo case 1')
#                print(t_geo.size)
#                print(fit_mol.time)
#            else:
#                geo_new = geo_data['geo_mol'].copy()
#                geo_trim_case = 2
#                print('extinction geo case 2')
#                print(t_geo.size)
#                print(fit_mol.time)
            
            fit_mol.bg_subtract(BGIndex)
            fit_mol.multiply_piecewise(geo_new)
            
            
#            fit_mol.multiply_piecewise(geo_data['geo_mol'])
#            print('eta-pre')
#            print(eta_i2_ext.shape)
            
            
            fit_mol.slice_range(range_lim=[0,range_trim])
            ver_mol.slice_range(range_lim=[settings['range_min'],range_trim])
            
            
            fit_mol.slice_range(range_lim=[settings['range_min'],range_trim])
            
            fit_mol.multiply_piecewise(1.0/eta_i2_ext)
            r_eta = fit_mol.range_array.copy()
            
            fit_mol.range_correct()
            
            
            fit_mol.multiply_piecewise(1.0/beta_m_ext.profile)
            fit_mol.log(update=True)
            
            
            # update the geo array using nearest neighbor
            iforward_t = np.argmin(np.abs(t_geo[:,np.newaxis]-fit_mol.time[np.newaxis,:]),axis=0)
            iforward_r = np.argmin(np.abs(r_geo[:,np.newaxis]-fit_mol.range_array[np.newaxis,:]),axis=0)
            geo_forward = geo_data['geo_mol'][iforward_t,:][:,iforward_r]  # two step indexing (grab profiles in time, then range)
            
            # update molecular efficiency using nearest neighbor
            iforward = np.argmin(np.abs(r_eta[:,np.newaxis]-fit_mol.range_array[np.newaxis,:]),axis=0)
            eta_i2_forward = eta_i2_ext[:,iforward]
            
#            geo_forward = np.interp(fit_mol.range_array,r_geo,geo_data['geo_mol'])
            
#            iforward_r = np.nonzero(np.in1d(np.round(2*r_geo).astype(np.int),np.round(2*fit_mol.range_array).astype(np.int)))[0]
#            iforward_t = np.nonzero(np.in1d(np.round(2*t_geo).astype(np.int),np.round(2*fit_mol.time).astype(np.int)))[0]
#            if geo_trim_case == 0:
#                geo_forward = geo_data['geo_mol'][iforward_t[0]:iforward_t[-1]+1,iforward_r[0]:iforward_r[-1]+1]
#            elif geo_trim_case > 0:
#                geo_forward = geo_new[iforward_t[0]:iforward_t[-1]+1,iforward_r[0]:iforward_r[-1]+1]
#            
##            geo_forward = geo_data['geo_mol'][iforward_t[0]:iforward_t[-1]+1,iforward_r[0]:iforward_r[-1]+1]
##            geo_forward = geo_data['geo_mol'][:,iforward_r[0]:iforward_r[-1]+1]
#            iforward = np.nonzero(np.in1d(np.round(2*r_eta).astype(np.int),np.round(2*fit_mol.range_array).astype(np.int)))[0]
#            eta_i2_forward = eta_i2_ext[:,iforward[0]:iforward[-1]+1]
            
            
            print('optimizing extinction filter design')
            fit_error_t =[]
            fit_error_r =[]
            twin = []
            rwin = []
            tord = []
            rord = []
            tord_max = 13
            rord_max = 13
            twin_max = 21
            rwin_max = 21
            iterations_t = np.sum(np.minimum(np.arange(3,twin_max,2)-2,tord_max))
            iterations_r = np.sum(np.minimum(np.arange(3,rwin_max,2)-2,rord_max))
            
            if ext_time_filt:
                print('expected time evaluations: %d'%iterations_t)
                iternum = 0
                itertarg = 10  # point  (in percent) at which we update the completion status
                for ext_sg_wid_t in range(3,twin_max,2):
                    for ext_sg_ord_t in range(1,min([ext_sg_wid_t-1,tord_max])):
                        filt_mol = fit_mol.copy()
                        filt_mol.sg_filter(ext_sg_wid_t,ext_sg_ord_t,axis=0)
    #                    print('profile')
    #                    print(filt_mol.profile.shape)
    #                    print('beta_m')
    #                    print(beta_m_ext.profile.shape)
    #                    print('eta')
    #                    print(eta_i2_forward.shape)
    #                    print('geo')
    #                    print(geo_forward.shape)
                        
                        
                        forward_model = np.exp(filt_mol.profile)*beta_m_ext.profile*eta_i2_forward/(filt_mol.range_array[np.newaxis,:]**2)/geo_forward+filt_mol.bg[:,np.newaxis]
                        fit_error_fm = np.nansum(forward_model-ver_mol.profile*np.log(forward_model),axis=0)
                        tord+=[ext_sg_ord_t]
                        twin+=[ext_sg_wid_t]
                        fit_error_t+=[fit_error_fm]
                        
                        iternum+=1                   
                        if iternum*100.0/iterations_t >= itertarg:
                            print('time: %d %%'%itertarg)
                            itertarg+=10
                
                
                imin_t = np.nanargmin(np.array(fit_error_t),axis=0)
    #            print(imin_t.shape)
    #            print(len(twin))
    #            print(np.array(twin).shape)
                ext_sg_wid_t = np.array(twin)[imin_t]
                ext_sg_order_t = np.array(tord)[imin_t]
                
                save_other_data['ext_sg_width_t'] = {'data':ext_sg_wid_t,'description':'window width of Savitzky-Golay filter applied in time','units':'bins'}
                save_other_data['ext_sg_order_t'] = {'data':ext_sg_order_t,'description':'polynomial order of Savitzky-Golay filter applied in time','units':'unitless'}
            
            if ext_range_filt:
                print('expected range evaluations: %d'%iterations_r)
                iternum = 0
                itertarg = 10
    #            fit_error_min = 0
                for ext_sg_wid_r in range(3,rwin_max,2):
                    for ext_sg_ord_r in range(1,min([ext_sg_wid_r-1,rord_max])):
                        filt_mol = fit_mol.copy()
                        filt_mol.sg_filter(ext_sg_wid_r,ext_sg_ord_r,axis=1)
    
                        forward_model = np.exp(filt_mol.profile)*beta_m_ext.profile*eta_i2_forward/(filt_mol.range_array[np.newaxis,:]**2)/geo_forward+filt_mol.bg[:,np.newaxis]
                        
                        fit_error_fm = np.nansum(forward_model-ver_mol.profile*np.log(forward_model),axis=1)
                        fit_error_r+=[fit_error_fm]
                        rwin+=[ext_sg_wid_r]
                        rord+=[ext_sg_ord_r]
                        
                        iternum+=1                   
                        if iternum*100.0/iterations_r >= itertarg:
                            print('range: %d %%'%itertarg)
                            itertarg+=10
                        
    #                    if fit_error_fm[10] < fit_error_min or len(fit_error_r) == 1:
    #                        fit_error_min = fit_error_fm[10]
    #                        print('new min min at %d:  %f'%(iternum,fit_error_fm[10]))
    #                        plt.figure()
    #                        plt.plot(ver_mol.profile[10,:])
    #                        plt.plot(forward_model[10,:])
    #                        plt.title('%d window, %d order'%(ext_sg_wid_r,ext_sg_ord_r))
    #                        
    #                        plt.figure()
    #                        plt.plot(fit_mol.profile[10,:])
    #                        plt.plot(filt_mol.profile[10,:])
    #                        plt.title('%d window, %d order'%(ext_sg_wid_r,ext_sg_ord_r))
                                    
                                
                imin_r = np.nanargmin(np.array(fit_error_r),axis=0)
                ext_sg_wid_r = np.array(rwin)[imin_r]
                ext_sg_order_r = np.array(rord)[imin_r]
            
                save_other_data['ext_sg_width_r'] = {'data':ext_sg_wid_r,'description':'window width of Savitzky-Golay filter applied in range','units':'bins'}
                save_other_data['ext_sg_order_r'] = {'data':ext_sg_order_r,'description':'polynomial order of Savitzky-Golay filter applied in range','units':'unitless'}
#            print('optimized sg filter parameters:')
#            print('   range (width,order) : %d,  %d'%(ext_sg_wid_r,ext_sg_order_r))
#            print('   time (width,order)  : %d,  %d'%(ext_sg_wid_t,ext_sg_order_t))
            
#            ext_sg_wid = 21
#            ext_sg_order = 4
#            ext_tres = 20  # extinction time resolution in seconds
#            ext_zres = 15   # extinction range resolution in meters
            
#            mol_ext.conv(ext_tres/mol_ext.mean_dt,ext_zres/mol_ext.mean_dR)
#            beta_m_ext.conv(ext_tres/beta_m_ext.mean_dt,ext_zres/beta_m_ext.mean_dR)
#            print(mol_ext.profile.shape)
#            print(beta_m_ext.profile.shape)
#            OD = mol_ext/beta_m_ext
#            OD.conv(ext_tres/OD.mean_dt,ext_zres/OD.mean_dR)
            
            OD = fit_mol.copy()
            OD.descript = 'Total optical depth from aircraft altitude'
            OD.label = 'Optical Depth'
            OD.profile_type = 'unitless'
            alpha_a = OD.copy()
            if ext_time_filt:
                OD.sg_filter(ext_sg_wid_t,ext_sg_order_t,axis=0)
            if ext_range_filt:
                OD.sg_filter(ext_sg_wid_r,ext_sg_order_r,axis=1)
#            OD.profile_variance = OD.profile_variance/OD.profile**2
#            OD.profile = np.log(OD.profile)
#            ODdata = OD.profile.copy()
#            OD.sg_filter(ext_sg_wid,ext_sg_order,deriv=0)
            OD.profile = -0.5*(OD.profile-np.nanmean(OD.profile[:,0:3],axis=1)[:,np.newaxis])
            
#            for ai in range(alpha_a.profile.shape[0]):
#                alpha_a.profile[ai,:] = -0.5*gv.savitzky_golay(np.log(alpha_a.profile[ai,:].flatten()), ext_sg_wid, ext_sg_order, deriv=1)
            if ext_time_filt:
                alpha_a.sg_filter(ext_sg_wid_t,ext_sg_order_t,axis=0)
            if ext_range_filt:
                alpha_a.sg_filter(ext_sg_wid_r,ext_sg_order_r,axis=1,deriv=1)
            else:
                alpha_a.sg_filter(3,1,axis=1,deriv=1)
            alpha_a = 0.5*alpha_a/alpha_a.mean_dR # not sure this is the right scaling factor
            alpha_a = alpha_a - beta_m_ext*(8*np.pi/3)  # remove molecular extinction
#            alpha_a.profile = ODdata.copy()
#            alpha_a.sg_filter(ext_sg_wid,ext_sg_order,deriv=0)
#            alpha_a.gain_scale(-0.5)
            if settings['as_altitude']:
                alpha_a.range2alt(master_alt,air_data_post,telescope_direction=var_post['TelescopeDirection'])
                
    #        if tres_post > 0 or tres <= 0.5:
    #            alpha_a.time_resample(tedges=master_time_post,update=True,remainder=False)
            
            alpha_a.descript = 'Aerosol Extinction Coefficient'
            alpha_a.label = 'Aerosol Extinction Coefficient'
            alpha_a.profile_type = '$m^{-1}$'
            
            # set OD profile to accumulate past valid data
            # and avoid saving nans to the final dataset
#            OD.mask(np.isnan(profs['Optical_Depth'].profile))
#            OD.profile = np.maximum.accumulate(OD.profile.filled(0),axis=1)
#            OD.remove_mask()
        
#        BSR = (profs['combined']+profs['cross'])/profs['molecular']
#        BSR.descript = 'Ratio of combined to molecular backscatter'
#        BSR.label = 'Backscatter Ratio'
#        BSR.profile_type = 'unitless'
        
        #BSR_mask = (BSR.profile-1)/np.sqrt(BSR.profile_variance) < 5.0
        #BSR_mask = BSR.profile < 2.0
        
        #BSR2 = profs['combined_hi'].copy()
        #BSR2.divide_prof(profs['molecular'])
        #BSR2.descript = 'Ratio of combined to molecular backscatter'
        #BSR2.label = 'Backscatter Ratio'
        #BSR2.profile_type = 'unitless'
        
#        d_mol = 2*0.000365/(1+0.000365) # molecular depolarization        
#        dVol = beta_a*dPart+beta_m*d_mol
        dVol = profs['cross']/(profs['combined']+profs['cross'])
        #dVol = profs['combined_hi'].copy()
        dVol.descript = 'Propensity of Volume to depolarize (d).  This is not identical to the depolarization ratio.  See Gimmestad: 10.1364/AO.47.003795 or Hayman and Thayer: 10.1364/JOSAA.29.000400'
        dVol.label = 'Volume Depolarization'
        dVol.profile_type = 'unitless'
        
        deltaLVol = dVol/(2-dVol)  
        deltaLVol.descript = 'Theoretically determined linear depolarization of the volume.  Depolarization is measured using circular polarizations assuming the volume consists of randomly oriented particles.'
        deltaLVol.label = 'Volume Linear Depolarization Ratio'
        deltaLVol.profile_type = 'unitless'
        
        deltaLPart = dPart/(2-dPart)  
        deltaLPart.descript = 'Theoretically determined linear depolarization of particles (molecular removed).  Depolarization is measured using circular polarizations assuming the volume consists of randomly oriented particles.'
        deltaLPart.label = 'Particle Linear Depolarization Ratio'
        deltaLPart.profile_type = 'unitless'
        
#        d_mol = 2*0.000365/(1+0.000365) # molecular depolarization
        
#        #Particle Depolarization = dVol/(1.0-1.0/BSR) - d_mol/(BSR-1)
#        dPart = (BSR*dVol-d_mol)/(BSR-1)
#        dPart.descript = 'Propensity of Particles to depolarize (d).  This is not identical to the depolarization ratio.  See Gimmestad: 10.1364/AO.47.003795 or Hayman and Thayer: 10.1364/JOSAA.29.000400'
#        dPart.label = 'Particle Depolarization'
#        dPart.profile_type = 'unitless'
        
        
        if settings['Estimate_Mol_Gain']:
            # This segment estimates what the molecular gain should be 
            # based on a histogram minimum in BSR over the loaded data
            
            iUp = np.nonzero(var_post['TelescopeDirection']==1.0)[0]            
            mol_adj_up = lp.Estimate_Mol_Gain(BSR,iKeep=iUp,mol_gain=mol_gain_up,alt_lims=[2000,4000],label='Telescope Up',plot=True)
            if settings['save_mol_gain_plot'] and mol_adj_up != 1.0:
                # double zero at beginning of filename just to put plots at the front of a sorted list of profiles
                plt.savefig(save_plots_path+'00_Molecular_Gain_Up_'+save_plots_base,dpi=300)
            
            
            iDown = np.nonzero(var_post['TelescopeDirection']==0.0)[0]
            mol_adj_down = lp.Estimate_Mol_Gain(BSR,iKeep=iDown,mol_gain=mol_gain_down,alt_lims=[2000,4000],label='Telescope Down',plot=True)
            if settings['save_mol_gain_plot'] and mol_adj_down != 1.0:
                # double zero at beginning of filename just to put plots at the front of a sorted list of profiles
                plt.savefig(save_plots_path+'00_Molecular_Gain_Down_'+save_plots_base,dpi=300)
            
        
        
        # add a diagnostic for counts/backscatter coeff
        
        # add a diagnostic for diff overlap between lo and hi channels as a function
        # of count rate or backscatter coeff
        
        count_mask = profs['combined_hi'].profile < settings['count_mask_threshold']
        
        #dPartMask = dPart.profile_variance > 1.0
#        dPartMask = dPart.profile_variance > settings['d_part_res_lim']**2
        #dPart.mask(dPartMask)
#        dPart.mask(dPart.profile_variance > settings['d_part_res_lim']**2)
        
        dPart.mask(dPart.profile > 1.1)
        dPart.mask(dPart.profile < -0.1)
        
        
        try:
            proj_label = process_vars['proj_label']
        except KeyError:
            proj_label = ''
    #    proj_label = proj + ' ' + flight_label[usr_flt] + ', '
        
        beta_a.mask(np.isnan(beta_a.profile))
        BSR.mask(np.isnan(BSR.profile))
        dPart.mask(np.isnan(dPart.profile))
        dVol.mask(np.isnan(dVol.profile))
        
#        beta_a_gv.mask(np.isnan(beta_a_gv.profile))
#        dPart_gv.mask(np.isnan(dPart_gv.profile))
#        dPart_gv.mask(dPart.profile.mask)
        
        if settings['Denoise_Mol']:
            beta_a_denoise.mask(np.isnan(beta_a_denoise.profile))
        
        
        if settings['count_mask_threshold'] > 0:
            beta_a.mask(count_mask)
            dPart.mask(count_mask)
            dVol.mask(count_mask)
            profs['combined'].mask(count_mask)
            
#            beta_a_gv.mask(count_mask)
#            dPart_gv.mask(count_mask)
            
            if settings['Denoise_Mol']:
                beta_a_denoise.mask(count_mask)
            
            if settings['get_extinction']:
                alpha_a.mask(count_mask)
#                alpha_a.mask(dPart.profile.mask)
        
        ParticleMask = np.logical_and(beta_a.profile < 1e-4,beta_a.SNR() < 1.0)
        dPart.mask(ParticleMask)      
        if settings['get_extinction']:
            alpha_a.mask(dPart.profile.mask)
        
        save_prof_list = [beta_a,dPart,dVol,BSR,beta_m,temp,pres,deltaLPart,deltaLVol] # test profiles: beta_a_gv,dPart_gv
        return_prof_list = [beta_a,dPart,dVol,BSR,beta_m,temp,pres,profs,param_profs]
        # add all channels to list of profilse to save
        for var in profs.keys():
            save_prof_list.extend([profs[var]])
            if settings['save_raw'] and var in raw_profs.keys():
                save_prof_list.extend([raw_profs[var]])
                
        if settings['get_extinction']:
            # add extinction to list of profilse to save
            save_prof_list.extend([alpha_a])
            save_prof_list.extend([OD])
        if settings['Denoise_Mol']:
            # add denoised molecular observations to list of profilse to save
            save_prof_list.extend([MolDenoise])
            save_prof_list.extend([beta_a_denoise])
        save_var1d_post = {'TelescopeDirection':{'description':'1-Lidar Pointing Up, 0-Lidar Pointing Down','units':'none'},
                           'polarization':{'description':'System Quarter Waveplate orientation','units':'radians'}}
        save_air_post = {'THDG': {'description':'aircraft heading','units':'degrees'},
                         'TASX': {'description':'airspeed','units':'meters/second'},
                         'GGLAT': {'description':'latitude','units':'degrees'},
                         'PITCH': {'description':'aircraft pitch angle','units':'degrees'},
                         'GGALT': {'description':'altitude','units':'meters'},
                         'PSXC': {'description':'ambient pressure','units':'hPa'},
                         'ROLL': {'description':'aircraft roll angle','units':'degrees'},
                         'GGLON': {'description':'longitude','units':'degrees'},
                         'ATX': {'description':'ambient temperature', 'units':'C'}}

        if settings['save_data']:
            print('saving profiles to')
            print(save_data_file)
            for ai in range(len(save_prof_list)):
                save_prof_list[ai].write2nc(save_data_file) #,name_override=True,tag=var_name)
                
            print('saving lidar status data to')
            print(save_data_file)
            for var in save_var1d_post.keys():
                lp.write_var2nc(var_post[var],str(var),save_data_file,description=save_var1d_post[var]['description'],units=save_var1d_post[var]['units'])
            
            print('saving aircraft variables to')
            print(save_data_file)
            for var in save_air_post.keys():
                lp.write_var2nc(air_data_post[var],str(var),save_data_file,description=save_air_post[var]['description'],units=save_air_post[var]['units'])
            
            print('saving additional variables to')
            print(save_data_file)
            for var in save_other_data.keys():
                lp.write_var2nc(save_other_data[var]['data'],str(var),save_data_file,description=save_other_data[var]['description'],units=save_other_data[var]['units'])
            
            lp.write_proj2nc(save_data_file,proj_label)
        else:
            print('save_data setting is False.  This data will not be saved.')
            
        if settings['plot_2D']:
            if settings['as_altitude']:
                if settings['plot_date']:
                    t1d_plt = mdates.date2num([datetime.datetime.fromordinal(BSR.StartDate.toordinal()) \
                                + datetime.timedelta(seconds=sec) for sec in time_1d])   
                else:
                    t1d_plt = time_1d/3600.0
                
                tlims = [(time_start-flight_date[usr_flt]).total_seconds()/3600.0,
                          (time_stop-flight_date[usr_flt]).total_seconds()/3600.0]
        #        tlims = [time_post[0]/3600.0, time_post[-1]/3600.0]
            #    rfig = lp.pcolor_profiles([BSR,dPart],scale=['log','linear'],climits=[[1,1e2],[0,0.7]],ylimits=[MinAlt*1e-3,MaxAlt*1e-3],title_add=proj_label,plot_date=plot_date)
                rfig = lp.pcolor_profiles([beta_a],scale=['log'],
                                          climits=[[1e-8,1e-3]],
                                          ylimits=[MinAlt*1e-3,MaxAlt*1e-3],
                                          tlimits=tlims,
                                          title_add=proj_label,
                                          plot_date=settings['plot_date'],
                                          t_axis_scale=settings['time_axis_scale'],
                                          h_axis_scale=settings['alt_axis_scale'],
                                          minor_ticks=5,major_ticks=1,cmap=['jet'])
                if settings['as_altitude']:
                    for ai in range(len(rfig[1])):
                        rfig[1][ai].plot(t1d_plt,air_data_t['GGALT']*1e-3,color='gray',linewidth=1.2)  # add aircraft altitude      
                if settings['save_plots']:
                    plt.savefig(save_plots_path+'Aerosol_Backscatter_'+save_plots_base,dpi=300)
                
                rfig = lp.pcolor_profiles([dPart],scale=['linear'],
                                          climits=[[0,1.0]],
                                          ylimits=[MinAlt*1e-3,MaxAlt*1e-3],tlimits=tlims,
                                          title_add=proj_label,
                                          plot_date=settings['plot_date'],
                                          t_axis_scale=settings['time_axis_scale'],
                                          h_axis_scale=settings['alt_axis_scale'],
                                          minor_ticks=5,major_ticks=1)
                if settings['as_altitude']:
                    for ai in range(len(rfig[1])):
                        rfig[1][ai].plot(t1d_plt,air_data_t['GGALT']*1e-3,color='gray',linewidth=1.2)  # add aircraft altitude      
                if settings['save_plots']:
                    plt.savefig(save_plots_path+'Aerosol_Depolarization_'+save_plots_base,dpi=300)        
                
                rfig = lp.pcolor_profiles([profs['combined']],scale=['log'],
                                          climits=[[1e-1,1e4]],
                                          ylimits=[MinAlt*1e-3,MaxAlt*1e-3],tlimits=tlims,
                                          title_add=proj_label,
                                          plot_date=settings['plot_date'],
                                          t_axis_scale=settings['time_axis_scale'],
                                          h_axis_scale=settings['alt_axis_scale'],
                                          minor_ticks=5,major_ticks=1)
                if settings['as_altitude']:
                    for ai in range(len(rfig[1])):
                        rfig[1][ai].plot(t1d_plt,air_data_t['GGALT']*1e-3,color='gray',linewidth=1.2)  # add aircraft altitude
                if settings['save_plots']:
                    plt.savefig(save_plots_path+'AttenuatedBackscatter_'+save_plots_base,dpi=300)
                
                rfig = lp.pcolor_profiles([dVol],scale=['linear'],
                                          climits=[[0,1.0]],
                                          ylimits=[MinAlt*1e-3,MaxAlt*1e-3],tlimits=tlims,
                                          title_add=proj_label,
                                          plot_date=settings['plot_date'],
                                          t_axis_scale=settings['time_axis_scale'],
                                          h_axis_scale=settings['alt_axis_scale'],
                                          minor_ticks=5,major_ticks=1)
                if settings['as_altitude']:
                    for ai in range(len(rfig[1])):
                        rfig[1][ai].plot(t1d_plt,air_data_t['GGALT']*1e-3,color='gray',linewidth=1.2)  # add aircraft altitude
                if settings['save_plots']:
                    plt.savefig(save_plots_path+'Volume_Depolarization_'+save_plots_base,dpi=300)
                        
                
                if settings['get_extinction']:
                    rfig = lp.pcolor_profiles([alpha_a],scale=['log'],
                                              climits=[[1e-5,1e-2]],
                                              ylimits=[MinAlt*1e-3,MaxAlt*1e-3],tlimits=tlims,
                                              title_add=proj_label,
                                              plot_date=settings['plot_date'],
                                              t_axis_scale=settings['time_axis_scale'],
                                              h_axis_scale=settings['alt_axis_scale'],
                                              minor_ticks=5,major_ticks=1)
                    if settings['as_altitude']:
                        for ai in range(len(rfig[1])):
                            rfig[1][ai].plot(t1d_plt,air_data_t['GGALT']*1e-3,color='gray',linewidth=1.2)  # add aircraft altitude
                    if settings['save_plots']:
                        plt.savefig(save_plots_path+'Extinction_'+save_plots_base,dpi=300)
                        
                if settings['Denoise_Mol']:
                    rfig = lp.pcolor_profiles([beta_a_denoise],scale=['log'],
                                          climits=[[1e-8,1e-3]],
                                          ylimits=[MinAlt*1e-3,MaxAlt*1e-3],
                                          tlimits=tlims,
                                          title_add=proj_label,
                                          plot_date=settings['plot_date'],
                                          t_axis_scale=settings['time_axis_scale'],
                                          h_axis_scale=settings['alt_axis_scale'],
                                          minor_ticks=5,major_ticks=1)
                    if settings['as_altitude']:
                        for ai in range(len(rfig[1])):
                            rfig[1][ai].plot(t1d_plt,air_data_t['GGALT']*1e-3,color='gray',linewidth=1.2)  # add aircraft altitude      
                    if settings['save_plots']:
                        plt.savefig(save_plots_path+'Denoised_Aerosol_Backscatter_'+save_plots_base,dpi=300)
                    
                    
                    rfig = lp.pcolor_profiles([beta_a,beta_a_denoise],scale=['log','log'],
                                          climits=[[1e-8,1e-3],[1e-8,1e-3]],
                                          ylimits=[MinAlt*1e-3,MaxAlt*1e-3],
                                          tlimits=tlims,
                                          title_add=proj_label,
                                          plot_date=settings['plot_date'],
                                          t_axis_scale=settings['time_axis_scale'],
                                          h_axis_scale=settings['alt_axis_scale'],
                                          minor_ticks=5,major_ticks=1)
                    if settings['as_altitude']:
                        for ai in range(len(rfig[1])):
                            rfig[1][ai].plot(t1d_plt,air_data_t['GGALT']*1e-3,color='gray',linewidth=1.2)  # add aircraft altitude      
                    if settings['save_plots']:
                        plt.savefig(save_plots_path+'Compare_Denoised_Aerosol_Backscatter_'+save_plots_base,dpi=300)
            else:
                t1d_plt = time_1d/3600.0
                
                tlims = [(time_start-flight_date[usr_flt]).total_seconds()/3600.0,
                          (time_stop-flight_date[usr_flt]).total_seconds()/3600.0]
        #        tlims = [time_post[0]/3600.0, time_post[-1]/3600.0]
            #    rfig = lp.pcolor_profiles([BSR,dPart],scale=['log','linear'],climits=[[1,1e2],[0,0.7]],ylimits=[MinAlt*1e-3,MaxAlt*1e-3],title_add=proj_label,plot_date=plot_date)
                rfig = lplt.scatter_z(beta_a,scale=['log'],
                                          climits=[[1e-8,1e-3]],
                                          ylimits=[MinAlt*1e-3,MaxAlt*1e-3],
                                          tlimits=tlims,
                                          title_add=proj_label,
                                          t_axis_scale=settings['time_axis_scale'],
                                          h_axis_scale=settings['alt_axis_scale'],
                                          cmap='jet')
#                rfig.plot(t1d_plt,air_data_t['GGALT']*1e-3,color='gray',linewidth=1.2)  # add aircraft altitude  
                
                if settings['save_plots']:
                    plt.savefig(save_plots_path+'Aerosol_Backscatter_'+save_plots_base,dpi=300)
                
                rfig = lplt.scatter_z(dPart,scale=['linear'],
                                          climits=[[0,1.0]],
                                          ylimits=[MinAlt*1e-3,MaxAlt*1e-3],tlimits=tlims,
                                          title_add=proj_label,
                                          t_axis_scale=settings['time_axis_scale'],
                                          h_axis_scale=settings['alt_axis_scale'])
                
#                rfig.plot(t1d_plt,air_data_t['GGALT']*1e-3,color='gray',linewidth=1.2)  # add aircraft altitude        
                if settings['save_plots']:
                    plt.savefig(save_plots_path+'Aerosol_Depolarization_'+save_plots_base,dpi=300)        
                
                rfig = lplt.scatter_z(profs['combined'],scale=['log'],
                                          climits=[[1e-1,1e4]],
                                          ylimits=[MinAlt*1e-3,MaxAlt*1e-3],tlimits=tlims,
                                          title_add=proj_label,
                                          t_axis_scale=settings['time_axis_scale'],
                                          h_axis_scale=settings['alt_axis_scale'])
#                rfig.plot(t1d_plt,air_data_t['GGALT']*1e-3,color='gray',linewidth=1.2)  # add aircraft altitude  
                if settings['save_plots']:
                    plt.savefig(save_plots_path+'AttenuatedBackscatter_'+save_plots_base,dpi=300)
                
                rfig = lplt.scatter_z(dVol,scale=['linear'],
                                          climits=[[0,1.0]],
                                          ylimits=[MinAlt*1e-3,MaxAlt*1e-3],tlimits=tlims,
                                          title_add=proj_label,
                                          t_axis_scale=settings['time_axis_scale'],
                                          h_axis_scale=settings['alt_axis_scale'])
#                rfig.plot(t1d_plt,air_data_t['GGALT']*1e-3,color='gray',linewidth=1.2)  # add aircraft altitude  
                if settings['save_plots']:
                    plt.savefig(save_plots_path+'Volume_Depolarization_'+save_plots_base,dpi=300)
                        
                
                if settings['get_extinction']:
                    rfig = lplt.scatter_z(alpha_a,scale='log',
                                              climits=[1e-5,1e-2],
                                              ylimits=[MinAlt*1e-3,MaxAlt*1e-3],tlimits=tlims,
                                              title_add=proj_label,
                                              t_axis_scale=settings['time_axis_scale'],
                                              h_axis_scale=settings['alt_axis_scale'])
#                    rfig.plot(t1d_plt,air_data_t['GGALT']*1e-3,color='gray',linewidth=1.2)  # add aircraft altitude  
                    if settings['save_plots']:
                        plt.savefig(save_plots_path+'Extinction_'+save_plots_base,dpi=300)
                        
                if settings['Denoise_Mol']:
                    rfig = lplt.scatter_z(beta_a_denoise,scale='log',
                                          climits=[[1e-8,1e-3]],
                                          ylimits=[MinAlt*1e-3,MaxAlt*1e-3],
                                          tlimits=tlims,
                                          title_add=proj_label,
                                          plot_date=settings['plot_date'],
                                          t_axis_scale=settings['time_axis_scale'],
                                          h_axis_scale=settings['alt_axis_scale'])
                    rfig.plot(t1d_plt,air_data_t['GGALT']*1e-3,color='gray',linewidth=1.2)  # add aircraft altitude     
                    if settings['save_plots']:
                        plt.savefig(save_plots_path+'Denoised_Aerosol_Backscatter_'+save_plots_base,dpi=300)
                    
                    
                    rfig = lp.pcolor_profiles([beta_a,beta_a_denoise],scale=['log','log'],
                                          climits=[[1e-8,1e-3],[1e-8,1e-3]],
                                          ylimits=[MinAlt*1e-3,MaxAlt*1e-3],
                                          tlimits=tlims,
                                          title_add=proj_label,
                                          t_axis_scale=settings['time_axis_scale'],
                                          h_axis_scale=settings['alt_axis_scale'])
                    rfig.plot(t1d_plt,air_data_t['GGALT']*1e-3,color='gray',linewidth=1.2)  # add aircraft altitude       
                    if settings['save_plots']:
                        plt.savefig(save_plots_path+'Compare_Denoised_Aerosol_Backscatter_'+save_plots_base,dpi=300)
                    
                    
           
        if settings['show_plots']:
            plt.show()
            
        return return_prof_list
    else:
        # no data was loaded
        return 0
