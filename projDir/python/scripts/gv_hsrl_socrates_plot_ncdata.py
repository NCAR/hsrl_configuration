# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 16:21:08 2018

@author: mhayman
"""

import numpy as np
import matplotlib.pyplot as plt
import LidarProfileFunctions as lp
import datetime

import os
import sys

import glob

import matplotlib.dates as mdates

import netCDF4 as nc4
#import json

import GVHSRLlib as gv


prof_list = ['Aerosol_Backscatter_Coefficient','Aerosol_Extinction_Coefficient',
             'Particle_Depolarization','Volume_Depolarization'] #'Aerosol_Extinction_Coefficient'

MaxAlt = 8e3 #49e3
MinAlt = 0e3 #-2e3

# size of each processing step
time_increment = datetime.timedelta(hours=0,minutes=20)
# size of a processesed data set
time_duration = datetime.timedelta(hours=1,minutes=0)
#time_duration = time_increment


new_settings = {
            'plot_date':True,
	    'plot_kft':False,
            'time_axis_scale':5.0,
            'alt_axis_scale':1.0,
            'save_data':True,
            'save_plots':True,
            'full_flight':True,
            'save_flight_folder':True
            }

try: settings
except NameError:
    settings={}
    
for val in new_settings.keys():
    settings[val] = new_settings[val]

PathFile = os.path.abspath(__file__+'/../')+'/gv_hsrl_socrates_paths.py'

plot_settings = {
    'Aerosol_Backscatter_Coefficient':{
        'climits':[1e-8,1e-3],
        'scale':'log'
        },
    'Aerosol_Extinction_Coefficient':{
        'climits':[1e-5,1e-2],
        'scale':'log'
        },
    'Particle_Depolarization':{
        'climits':[0,1.0],
        'scale':'linear'
        },
    'Volume_Depolarization':{
        'climits':[0,1.0],
        'scale':'linear'
        },
    }

# load path data for this computer
exec(open(PathFile).read())

# add the path to GVHSRLlib manually
library_path = os.path.abspath(paths['software_path']+'/processors/')
print(library_path)
if library_path not in sys.path:
    sys.path.append(library_path)

import Airborne_GVHSRL_DataSelection as ds

try:process_vars
except NameError:
    process_vars = {}
    process_vars['proj'] = 'SOCRATES'



time_start0,time_stop0,settings,paths,process_vars = \
    ds.SelectAirborneData(settings=settings,paths=paths,process_vars=process_vars)

#ncfilename = '/Users/mhayman/Documents/HSRL/GVHSRL/data/SOCRATEStf02/SOCRATEStf02_GVHSRL_20180104T1700_20180104T1800.nc'
try:
	day_start = process_vars['aircraft_t_ref']
except KeyError:
	day_start = datetime.datetime(year=time_start0.year,month=time_start0.month,day=time_start0.day)

t0 = time_start0-day_start




#tlims = [17.0,18.0]



data_path = paths['save_data_path']
nclist = sorted(glob.glob(data_path+'*.nc'))

print('\nSaving plots to:')
print(paths['save_plots_path']+'\n')

time_start = day_start + datetime.timedelta(seconds=np.floor(t0.total_seconds()/time_increment.total_seconds())*time_increment.total_seconds())
time_stop = time_start + time_duration
#time_stop = day_start + datetime.timedelta(seconds=np.floor(t0.total_seconds()/time_increment.total_seconds())*time_increment.total_seconds())+time_duration
loop_data = True




while loop_data:
    findex = []
    for ai in range(len(nclist)):
        filestart = datetime.datetime.strptime(nclist[ai][-30:-17],'%Y%m%dT%H%M')
        filestop = datetime.datetime.strptime(nclist[ai][-16:-3],'%Y%m%dT%H%M')

        if time_start <= filestart and time_stop >= filestart:
            findex.extend([ai])
        elif time_start <= filestop and time_stop >= filestart:
            findex.extend([ai])
        
    
    save_plots_base = process_vars['flt']+'_GVHSRL_'+time_start.strftime('%Y%m%dT%H%M')+'_'+time_stop.strftime('%Y%m%dT%H%M')    
    tlims = [(time_start-day_start).total_seconds()/3600.0,(time_stop-day_start).total_seconds()/3600.0]
    print(time_start.strftime('%H:%M') + ' - ' + time_stop.strftime('%H:%M')+':')
    #print('[%f,%f]'%(tlims[0],tlims[1]))
    prof_start = True
    profs = {}
    for fi in findex:
        ncfilename = nclist[fi]
        f = nc4.Dataset(ncfilename,'r')
        alt_data0 = lp.ncvar(f,'GGALT')
        t_data0 = lp.ncvar(f,'time').astype(np.float)
        f.close()
        
        #print(alt_data0.shape)
        #print(t_data0.shape)
        
        
        if prof_start:
            alt_data = alt_data0.copy()
            t_data = t_data0.copy()
            prof_start = False
        else:
            alt_data = np.concatenate((alt_data,alt_data0))
            t_data = np.concatenate((t_data,t_data0))
#        print('alt_data')
#        print(alt_data.shape)
        
        for var in prof_list:
            prof0 = lp.load_nc_Profile(ncfilename,var,mask=True)
            if var in profs.keys():
                profs[var].cat_time(prof0,front=False)  
            else:
                profs[var] = prof0.copy()
#            print(var)
#            print(profs[var].profile.shape)
        
    if len(nclist) > 0:  
        for var in prof_list:
	    	try:
		    profs[var].slice_time([tlims[0]*3600,tlims[1]*3600])
	#            profs[var] = lp.load_nc_Profile(ncfilename,var,mask=True)
		    
		#    tlims = [profs[var].time[0]/3600.0,profs[var].time[-1]/3600.0]
		    print('   ' + var + ' %d data points'%profs[var].profile.shape[0])
		    if profs[var].time.size > 10:
		    	rfig = lp.pcolor_profiles([profs[var]],scale=[plot_settings[var]['scale']],
		                                      climits=[plot_settings[var]['climits']],
		                                      ylimits=[MinAlt*1e-3,MaxAlt*1e-3],
		                                      tlimits=tlims,
		                                      title_add=process_vars['proj_label'],
		                                      plot_date=settings['plot_date'],
		                                      t_axis_scale=settings['time_axis_scale'],
		                                      h_axis_scale=settings['alt_axis_scale'],
		                                      minor_ticks=10,major_ticks=1,plt_kft=settings['plot_kft']
		                                      )
		    	if settings['plot_date']:
		        	t1d_plt = mdates.date2num([datetime.datetime.fromordinal(profs[var].StartDate.toordinal()) \
		                    	+ datetime.timedelta(seconds=sec) for sec in t_data])   
		    	else:
		        	t1d_plt = t_data/3600.0
			if settings['plot_kft']:
				range_adj = 3.28084
			else:
				range_adj = 1.0

		    	for ai in range(len(rfig[1])):
		        	rfig[1][ai].plot(t1d_plt,alt_data*1e-3*range_adj,color='gray',linewidth=1.2)  # add aircraft altitude     
		    	if settings['save_plots']:
		        	plt.savefig(paths['save_plots_path']+var+'_'+save_plots_base,dpi=300)
		        #                print('   ' + paths['save_plots_path']+var+'_'+save_plots_base)
		    	plt.close('all') 
		except KeyError:
			print('   ' + var + ' not found')

        
    if time_stop >= time_stop0:
        loop_data = False
    else:
        time_start = time_start+time_increment
        time_stop = time_stop+time_increment
