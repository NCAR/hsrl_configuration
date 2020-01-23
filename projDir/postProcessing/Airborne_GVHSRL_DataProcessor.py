# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:59:17 2018

@author: mhayman
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import LidarProfileFunctions as lp
import LidarPlotFunctions as lplt
import datetime

import matplotlib.dates as mdates
import json

import GVHSRLlib as gv

# input and raw_input are the same for python 3 and 2 respectively
# this makes it so input always accepts a string
try:
    input=raw_input
except NameError:
    pass

def ProcessAirborneDataChunk(time_start,time_stop,
                             settings={},paths={},process_vars={},
                             date_reference=0):
    
    default_settings = {
        'tres':0.5,  # resolution in time in seconds (0.5 sec) before altitude correction
        'tres_post':1*60, # resolution after altitude correction -  set to zero to not use
        'zres':7.5,  # altitude resolution in meters (7.5 m minimum)
        
        'mol_smooth':False, # smooth molecular profile
        't_mol_smooth':0.0, # smoothing in time of molecular profile
        'z_mol_smooth':30.0, # smoothing in range of molecular profile
        
        'use_BM3D':True, # use BM3D denoised raw data where it is available

        'range_min':150.0,  # closest range in m where data is treated as valid

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
                
        'Airspeed_Threshold':15, # threshold for determining start and end of the flight (in m/s)
        
        'loadQWP':'fixed',  # load 'fixed','rotating', or 'all' QWP data
        
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
      
    # If a paremeter isn't supplied, use the default setting
    for param in default_settings.keys():
        if not param in settings.keys():
            settings[param] = default_settings[param]    
    
    tres = settings['tres']
    tres_post = settings['tres_post']
#    zres = settings['zres']
    BGIndex = settings['BGIndex']
    MaxAlt = settings['MaxAlt']
    MinAlt = settings['MinAlt']
      
    try:
        basepath = paths['basepath']
    except KeyError:
        sys.exit('No data path found.')
        
    """
    Set up to save data
    """
    
    flt = process_vars['flt']   
    
    save_other_data = {}  # dict containing extra variables to be saved to the netcdf file   
    
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
            save_plots_base = flt+'_GVHSRL_'+time_start.strftime('%Y%m%d_%H%M')+'_to_'+time_stop.strftime('%Y%m%d_%H%M')
        except KeyError:
            print('Save plots is disabled')
            print('  No save path (save_plots_path) is provided')
            settings['save_plots'] = False

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
    cal_file = paths['cal_file_path'] + 'gv_calvals.json'
    with open(cal_file,"r") as f:
        cal_json = json.loads(f.read())
    f.close()
    bin0 = lp.get_calval(time_start,cal_json,"Bin Zero")[0]  # bin number where t/range = 0
    
       
    filePathAircraft = paths['filePathAircraft']
    #  load aircraft data    
    air_data,aircraft_t_ref = gv.load_aircraft_data(filePathAircraft,var_aircraft)

    print('Processing: ')
    print('   '+time_start.strftime('%H:%M %d-%b, %Y to'))
    print('   '+time_stop.strftime('%H:%M %d-%b, %Y'))
    print('')
    
    if settings['use_aircraft_tref']:
        date_reference = aircraft_t_ref
    
    """
    Get raw data from netcdf files
    """
    proj = process_vars['proj']
    time_list,var_1d_data, profs = gv.load_raw_data(time_start,time_stop,var_2d_list,var_1d_list,basepath=basepath[proj],verbose=True,as_prof=True,loadQWP=settings['loadQWP'],date_reference=date_reference,time_shift=settings['aircraft_time_shift'],bin0=bin0,loadBM3D=settings['use_BM3D'])
    
    if settings['use_BM3D']:
        # if using BM3D data, try to load the pthinned denoised profiles for 
        # filter optimization
        thin_list = ['molecular_pthin_fit_BM3D','molecular_pthin_ver_BM3D']
        _,_, thin_profs = gv.load_raw_data(time_start,time_stop,thin_list,[],basepath=basepath[proj],verbose=True,as_prof=True,loadQWP=settings['loadQWP'],date_reference=date_reference,time_shift=settings['aircraft_time_shift'],bin0=bin0,loadBM3D=True)
        for tvar in thin_profs.keys():
            if isinstance(thin_profs[tvar], lp.LidarProfile):
                print('found '+tvar)
                # check that dimensions agree with other profiles first?
                profs[tvar] = thin_profs[tvar]
            else:
                print(tvar + ' may not exist in the requested data')
                print(thin_profs[tvar].shape)
    
    """
    Start data processing
    """
    
    run_processing = len(profs) > 0  
    
    if run_processing:
        # estimate where bin0 ()
        pbin0 = np.sum(profs['molecular'].profile,axis=0)
        try:
            
            ipbin0 = np.nonzero(pbin0 > 4*pbin0[0])[0][0] # locate intial outgoing pulse
            ipbin1 = ipbin0 + np.nonzero(np.diff(pbin0[ipbin0:]) < 0)[0][0] # locate zero crossing after the pulse
            # interp expects xp to be monotonically increasing.
            # by negating the xp term in the function, we assume a negative slope
            est_bin0 = np.interp(np.zeros(1),-np.diff(pbin0[ipbin0:ipbin1+2]),np.arange(ipbin0,ipbin1+1)+0.5) 
            print('')
            print('Estimated bin0: %f'%est_bin0)
            print('Current bin0: %f'%bin0)
            print('')
            save_other_data['est_bin0']={'data':est_bin0,'description':'estimated MCS bin corresponding to t=0 on the lidar pulse','units':'MCS bin number'}
            save_other_data['bin0'] = {'data':bin0,'description':'actual MCS bin corresponding to t=0 on the lidar pulse used in this processing','units':'MCS bin number'}
        except IndexError:
            print('No bin0 estimate')
    
    
    while run_processing:
        #execute processing if data was found

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
            geo_up = np.load(paths['cal_file_path']+geo_file_up)
            if len(geo_file_down) > 0:
                geo_down = np.load(paths['cal_file_path']+geo_file_down)
            else:
                geo_data = geo_up
                        
        # load differential overlap correction
        diff_data_up = np.load(paths['cal_file_path']+diff_geo_file)
        if len(diff_geo_file_down) > 0:
            diff_data_down = np.load(paths['cal_file_path']+diff_geo_file_down)
        else:
            diff_data = diff_data_up
        
        baseline_data = np.load(paths['cal_file_path']+baseline_file)
        
        if len(diff_pol_file):
            diff_pol_data = np.load(paths['cal_file_path']+diff_pol_file[0])
        
        # load i2 scan from file
        if len(i2_file):
            i2_data = np.load(paths['cal_file_path']+i2_file[0])
        else:
            # if no i2 scan available, don't correct for Rayleigh Brillouin spectrum
            settings['hsrl_rb_adjust'] = False
         
        flight_date = process_vars['flight_date']
        usr_flt = process_vars['usr_flt']
        
        print('flight_date: '+flight_date[usr_flt].strftime('%Y-%b-%d %H:%M'))
        print('date_reference: '+date_reference.strftime('%Y-%b-%d %H:%M'))
        
        # set the master time to match all 2D profiles to
        # (1d data will not be resampled)
        sec_start = (time_start-date_reference).total_seconds()
        sec_stop = (time_stop-date_reference).total_seconds()
        print('Found data for')
        print('   %f - %f hours after UTC flight start date.'%(sec_start/3600.0,sec_stop/3600.0))
        if tres > 0.5:
            master_time = np.arange(sec_start-tres/2,sec_stop+tres/2,tres)
            time_1d,var_1d = gv.var_time_resample(master_time,time_sec,var_1d_data,average=True)
            # estimate the cal grid points that need to be removed in the new time resolution.
            cal_ind_tres = np.unique(np.digitize(profs['combined_hi'].time[cal_indices],master_time))
        else:
            time_1d = time_sec.copy()
            var_1d = var_1d_data.copy()
            cal_ind_tres = cal_indices.copy()
        
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
        
        # time resolution after range to altitude conversion
        if tres_post > 0:
            print('post time res: %f seconds'%tres_post)
            master_time_post = np.arange(sec_start-tres_post/2,sec_stop+tres_post/2,tres_post)
        elif tres > 0.5:
            print('Using tres')
            master_time_post = master_time
            time_post = time_1d
            var_post = var_1d
            air_data_post = air_data_t
        else:
            print('No change to time resolution')
            master_time_post = np.arange(sec_start-tres/2,sec_stop+tres/2,tres)
        print('master_time_post limits')
        print('   %f - %f hours after UTC flight start date.'%(master_time_post[0]/3600.0,master_time_post[-1]/3600.0))
        print('   %f second resolution'%np.mean(np.diff(master_time_post)))
        print('   %d data points'%master_time_post.size)

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
        
        # set maximum range to MaxAlt
        range_trim = MaxAlt
        
        int_profs = {}  # obtain time integrated profiles
        raw_profs = {}
        for var in profs.keys():
            if settings['RemoveCals']:
                # remove instances where the I2 cell is removed
                profs[var].remove_time_indices(cal_indices)
            if tres > 0.5:
                profs[var].time_resample(tedges=master_time,update=True,remainder=False)

            if settings['deadtime_correct'] and var in dead_time.keys():
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

            # profile specific processing routines
            if var == 'combined_hi' and settings['diff_geo_correct']:
                if settings['Estimate Time Shift']:
                    print('Estimate Time Shift enabled')
                    # if the software settings state we are supposed to check for time shift, look to see if there are any
                    # roll or pitch manuevers that will help us do that.
                    # this is determined by looking at the derivative signals over a long strech of time
                    weights = np.convolve(np.ones(2000)*0.5e-3,np.concatenate((np.zeros(1),np.diff(air_data_t['ROLL'])**2+np.diff(air_data_t['PITCH'])**2)),'same')
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
                        Rg_filt[Rg_out] = np.nan
                  
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
                                Rg_corr3[ri] = np.nanmean((np.diff(Rg1)-np.diff(Rg2))**2/profs['combined_hi'].mean_dt**2)
                                
                            Rg_tot = Rg_corr3/np.nanmean(Rg_corr3)+Rg_corr1/np.nanmean(Rg_corr1)/4
                            i_min_tot = np.argmin(Rg_tot)
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

                            textstr = 'Current time offset: %f s\n'%settings['aircraft_time_shift'] + \
                                'Estimated time offset (total): %f s\n'%(i_t_off_tot*profs['combined_hi'].mean_dt + settings['aircraft_time_shift'])  + \
                                'Estimated time offset (lms): %f s\n'%(i_t_off_lms*profs['combined_hi'].mean_dt + settings['aircraft_time_shift'])  + \
                                'Estimated time offset (derivative lms): %f s\n'%(i_t_off_deriv*profs['combined_hi'].mean_dt + settings['aircraft_time_shift'])
         
                            fig1=plt.figure(figsize=(8,8))
                            ax1 = fig1.add_subplot(2,1,1,xmargin=0.0)
                            ax1.plot(-1*i_offset,np.sqrt(Rg_tot),label='Total Error')
                            ax1.plot(-1*i_offset,np.sqrt(Rg_corr3/np.nanmean(Rg_corr3)),label='Derivative Error')
                            ax1.plot(-1*i_offset,np.sqrt(Rg_corr1/np.nanmean(Rg_corr1)),label='Weighted Error')
                            ax1.plot(-1*i_offset[i_min_tot],np.sqrt(Rg_tot[i_min_tot]),'k.')
                            ax1.plot(-1*i_offset[i_min_lms],np.sqrt(Rg_corr1[i_min_lms]/np.nanmean(Rg_corr1)),'k.')
                            ax1.plot(-1*i_offset[i_min_deriv],np.sqrt(Rg_corr3[i_min_deriv]/np.nanmean(Rg_corr3)),'k.')
                            ax1.grid(b=True)
                            ax1.text(np.mean(i_offset),np.mean(plt.ylim()),textstr,verticalalignment='baseline',horizontalalignment='center',size=9)
                            ax1.set_xlabel('Aircraft Time shift [s]',fontsize=9)
                            ax1.set_ylabel('Weighted RMS Error',fontsize=9)
                            ax1.legend(fontsize=9)
                            ax1.set_title(save_plots_base, fontsize=10, fontweight='bold')

                            ax2=fig1.add_subplot(2,1,2,xmargin=0.0)
                            # Set up plot times
                            time_plot=[]
                            for sec1 in range(air_data_t['Time'].size):
                                time_plot.append(datetime.timedelta(seconds=air_data_t['Time'][sec1])+date_reference)
                            # axis date format
                            myFmt = mdates.DateFormatter("%H:%M")
                            ax2.plot(time_plot,Rg_exp,label='aircraft data estimate')
                            ax2.plot(time_plot,Rg_filt,'r.-',label='ground return (filtered)')
                            ax2.xaxis.set_major_formatter(myFmt)
                            ax2.set_xlabel('Time [UTC]',fontsize=9)
                            ax2.set_ylabel('Range to surface [m]',fontsize=9)
                            ax2.grid(b=True)
                            ax2.legend(fontsize=9)

                            if settings['save_mol_gain_plot']:
                                plt.savefig(save_plots_path+'Debug01_Time_Delay_Estimate_'+save_plots_base,dpi=300)
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

            profs[var].slice_range(range_lim=[settings['range_min'],range_trim])
    
            if tres_post > 0 or tres <= 0.5:
                profs[var].time_resample(tedges=master_time_post,update=True,remainder=False)
                if settings['save_raw']:
                    raw_profs[var].time_resample(tedges=master_time_post,update=True,remainder=False)
            if profs[var].profile.size == 0:
                run_processing = False
                print('Processing will terminate.  Profile of ' +var+ ' is empty')
            else:
                int_profs[var] = profs[var].copy()
                int_profs[var].time_integrate()
        if not run_processing:
            print('Terminating processing due to empty profiles')
            break

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
            master_time_post = np.concatenate((np.array([profs['molecular'].time[0]-tres_post*0.5]), \
                0.5*np.diff(profs['molecular'].time)+profs['molecular'].time[:-1], \
                np.array([profs['molecular'].time[-1]+tres_post*0.5])))
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
 
        lp.plotprofiles(profs)
        if settings['save_mol_gain_plot']:
            plt.title(save_plots_base)
            plt.savefig(save_plots_path+'Debug02_Lidar_Profiles_'+save_plots_base,dpi=300)

        temp,pres = gv.get_TP_from_aircraft(air_data,profs['molecular'],telescope_direction=var_post['TelescopeDirection'])
        beta_m = lp.get_beta_m(temp,pres,profs['molecular'].wavelength)
        
        nLidar = gv.lidar_pointing_vector(air_data_post,var_post['TelescopeDirection'],lidar_tilt=4.0)

        save_other_data['lidar_pointing']={'data':nLidar,'description':'Lidar pointing vector in global coordinate frame. index 0 = North, index 1 = East, index 2 = Down','units':'none'}

        beta_m_ext = beta_m.copy()

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

            eta_i2 = eta_i2.reshape(temp.profile.shape)
                        
            profs['molecular'].gain_scale(mol_gain,gain_var = (mol_gain*0.05)**2)
        
            eta_c = eta_c.reshape(temp.profile.shape)

            beta_a,dPart,BSR,param_profs = gv.AerosolBackscatter(profs['molecular'],profs['combined'],profs['cross'],beta_m, \
                eta_am=Tam,eta_ac=Tac,eta_mm=eta_i2,eta_mc=eta_c,eta_x=0.0,gm=1.0)            

            if settings['get_extinction']:    
                mol_ext = profs['molecular'].copy()
                eta_i2_ext = eta_i2
                mol_ext.multiply_piecewise(1.0/eta_i2_ext)
                mol_ext.range_correct()
           
        else:
            # Rescale molecular channel to match combined channel gain
            profs['molecular'].gain_scale(mol_gain,gain_var = (mol_gain*0.05)**2)

        if settings['get_extinction']:
            print('Estimating Extinction')
            beta_m_ext.slice_range(range_lim=[settings['range_min'],range_trim])
 
            t_geo = fit_mol.time.copy()
            r_geo = fit_mol.range_array.copy()
                        
            if tres_post > 0 or tres <= 0.5:
                fit_mol.time_resample(tedges=master_time_post,update=True,remainder=False,average=False)
                ver_mol.time_resample(tedges=master_time_post,update=True,remainder=False,average=False)

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
#                    r_win_ord=lp.optimize_sg_raw(fit_mol,axis=1,full=False,order=[1,5],window=[3,23],range_lim=[settings['range_min'],range_trim],bg_subtract=True,AdjCounts=False)
                    r_window0=t_win_ord[0]['window']
                    r_ord0=t_win_ord[0]['order']
                    fit_mol.sg_filter(r_window0,r_ord0,axis=1)
                    ver_mol.sg_filter(r_window0,r_ord0,axis=1)
                
                if ext_time_filt:
                    fit_mol.sg_filter(t_window0,t_ord0,axis=0)
                    ver_mol.sg_filter(t_window0,t_ord0,axis=0)

            # update the geo array using nearest neighbor
            igeo_t = np.argmin(np.abs(t_geo[:,np.newaxis]-fit_mol.time[np.newaxis,:]),axis=0)
            geo_new = geo_data['geo_mol'][igeo_t,:]

            fit_mol.bg_subtract(BGIndex)
            fit_mol.multiply_piecewise(geo_new)
         
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
                ext_sg_wid_t = np.array(twin)[imin_t]
                ext_sg_order_t = np.array(tord)[imin_t]
                
                save_other_data['ext_sg_width_t'] = {'data':ext_sg_wid_t,'description':'window width of Savitzky-Golay filter applied in time','units':'bins'}
                save_other_data['ext_sg_order_t'] = {'data':ext_sg_order_t,'description':'polynomial order of Savitzky-Golay filter applied in time','units':'unitless'}
            
            if ext_range_filt:
                print('expected range evaluations: %d'%iterations_r)
                iternum = 0
                itertarg = 10
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
                        
                imin_r = np.nanargmin(np.array(fit_error_r),axis=0)
                ext_sg_wid_r = np.array(rwin)[imin_r]
                ext_sg_order_r = np.array(rord)[imin_r]
            
                save_other_data['ext_sg_width_r'] = {'data':ext_sg_wid_r,'description':'window width of Savitzky-Golay filter applied in range','units':'bins'}
                save_other_data['ext_sg_order_r'] = {'data':ext_sg_order_r,'description':'polynomial order of Savitzky-Golay filter applied in range','units':'unitless'}
 
            OD = fit_mol.copy()
            OD.descript = 'Total optical depth from aircraft altitude'
            OD.label = 'Optical Depth'
            OD.profile_type = 'unitless'
            alpha_a = OD.copy()
            if ext_time_filt:
                OD.sg_filter(ext_sg_wid_t,ext_sg_order_t,axis=0)
            if ext_range_filt:
                OD.sg_filter(ext_sg_wid_r,ext_sg_order_r,axis=1)
            OD.profile = -0.5*(OD.profile-np.nanmean(OD.profile[:,0:3],axis=1)[:,np.newaxis])

            if ext_time_filt:
                alpha_a.sg_filter(ext_sg_wid_t,ext_sg_order_t,axis=0)
            if ext_range_filt:
                alpha_a.sg_filter(ext_sg_wid_r,ext_sg_order_r,axis=1,deriv=1)
            else:
                alpha_a.sg_filter(3,1,axis=1,deriv=1)
            alpha_a = 0.5*alpha_a/alpha_a.mean_dR # not sure this is the right scaling factor
            alpha_a = alpha_a - beta_m_ext*(8*np.pi/3)  # remove molecular extinction

            alpha_a.descript = 'Aerosol Extinction Coefficient'
            alpha_a.label = 'Aerosol Extinction Coefficient'
            alpha_a.profile_type = '$m^{-1}$'
 
        dVol = profs['cross']/(profs['combined']+profs['cross'])
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

        if settings['Estimate_Mol_Gain']:
            # This segment estimates what the molecular gain should be 
            # based on a histogram minimum in BSR over the loaded data
            
            iUp = np.nonzero(var_post['TelescopeDirection']==1.0)[0]            
            mol_adj_up = lp.Estimate_Mol_Gain(BSR,iKeep=iUp,mol_gain=mol_gain_up,alt_lims=[2000,4000],label='Telescope Up',plot=True)
            if settings['save_mol_gain_plot'] and mol_adj_up != 1.0:
                plt.savefig(save_plots_path+'Debug03_Molecular_Gain_Up_'+save_plots_base,dpi=300)
                        
            iDown = np.nonzero(var_post['TelescopeDirection']==0.0)[0]
            mol_adj_down = lp.Estimate_Mol_Gain(BSR,iKeep=iDown,mol_gain=mol_gain_down,alt_lims=[2000,4000],label='Telescope Down',plot=True)
            if settings['save_mol_gain_plot'] and mol_adj_down != 1.0:
                plt.savefig(save_plots_path+'Debug04_Molecular_Gain_Down_'+save_plots_base,dpi=300)
                   
        # add a diagnostic for counts/backscatter coeff
        # add a diagnostic for diff overlap between lo and hi channels as a function
        # of count rate or backscatter coeff
        
        count_mask = profs['combined_hi'].profile < settings['count_mask_threshold']

        dPart.mask(dPart.profile > 1.1)
        dPart.mask(dPart.profile < -0.1)
                
        try:
            proj_label = process_vars['proj_label']
        except KeyError:
            proj_label = ''

        beta_a.mask(np.isnan(beta_a.profile))
        BSR.mask(np.isnan(BSR.profile))
        dPart.mask(np.isnan(dPart.profile))
        dVol.mask(np.isnan(dVol.profile))

        if settings['count_mask_threshold'] > 0:
            beta_a.mask(count_mask)
            dPart.mask(count_mask)
            dVol.mask(count_mask)
            profs['combined'].mask(count_mask)

            if settings['get_extinction']:
                alpha_a.mask(count_mask)

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
            
            if settings['get_extinction']:
                numPlots=5
            else:
                numPlots=4

#            tlims = [(time_start-flight_date[usr_flt]).total_seconds()/3600.0,
#                      (time_stop-flight_date[usr_flt]).total_seconds()/3600.0]
            
            fig1=plt.figure(figsize=(10,13))
            ax1 = fig1.add_subplot(numPlots,1,1)
            
            rfig = lplt.scatter_z(beta_a,scale=['log'],
                                      ax=ax1,
                                      lidar_pointing = nLidar,
                                      lidar_alt = air_data_post['GGALT'],
                                      climits=[[1e-8,1e-3]],
                                      ylimits=[MinAlt*1e-3,MaxAlt*1e-3],
#                                      tlimits=tlims,
                                      title_add=proj_label,
                                      t_axis_scale=settings['time_axis_scale'],
                                      h_axis_scale=settings['alt_axis_scale'],
                                      cmap='jet')
            
            ax2 = fig1.add_subplot(numPlots,1,2)
            rfig = lplt.scatter_z(dPart,scale=['linear'],
                                      ax=ax2,
                                      lidar_pointing = nLidar,
                                      lidar_alt = air_data_post['GGALT'],
                                      climits=[[0,1.0]],
                                      ylimits=[MinAlt*1e-3,MaxAlt*1e-3],
#                                      tlimits=tlims,
                                      title_add=proj_label,
                                      t_axis_scale=settings['time_axis_scale'],
                                      h_axis_scale=settings['alt_axis_scale'])
            
            ax3 = fig1.add_subplot(numPlots,1,3)
            rfig = lplt.scatter_z(profs['combined'],scale=['log'],
                                      ax=ax3,
                                      lidar_pointing = nLidar,
                                      lidar_alt = air_data_post['GGALT'],
                                      climits=[[1e-1,1e4]],
                                      ylimits=[MinAlt*1e-3,MaxAlt*1e-3],
#                                      tlimits=tlims,
                                      title_add=proj_label,
                                      t_axis_scale=settings['time_axis_scale'],
                                      h_axis_scale=settings['alt_axis_scale'])

            ax4 = fig1.add_subplot(numPlots,1,4)
            rfig = lplt.scatter_z(dVol,scale=['linear'],
                                      ax=ax4,
                                      lidar_pointing = nLidar,
                                      lidar_alt = air_data_post['GGALT'],
                                      climits=[[0,1.0]],
                                      ylimits=[MinAlt*1e-3,MaxAlt*1e-3],
#                                      tlimits=tlims,
                                      title_add=proj_label,
                                      t_axis_scale=settings['time_axis_scale'],
                                      h_axis_scale=settings['alt_axis_scale'])
                                      
            if settings['get_extinction']:
                ax5 = fig1.add_subplot(numPlots,1,5)
                rfig = lplt.scatter_z(alpha_a,scale='log',
                                          ax=ax5,
                                          lidar_pointing = nLidar,
                                          lidar_alt = air_data_post['GGALT'],
                                          climits=[1e-5,1e-2],
                                          ylimits=[MinAlt*1e-3,MaxAlt*1e-3],
#                                          tlimits=tlims,
                                          title_add=proj_label,
                                          t_axis_scale=settings['time_axis_scale'],
                                          h_axis_scale=settings['alt_axis_scale'])
                
            fig1.tight_layout()

            if settings['save_plots']:
                plt.savefig(save_plots_path+'LidarFields_'+save_plots_base,dpi=300)
  
        if settings['show_plots']:
            plt.show()
            
        return return_prof_list
    else:
        # no data was loaded
        return 0
