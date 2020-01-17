# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:28:07 2018

@author: mhayman
"""

import numpy as np

import matplotlib.pyplot as plt
import datetime

import matplotlib
import matplotlib.dates as mdates

from mpl_toolkits.axes_grid1 import make_axes_locatable

import LidarProfileFunctions as lp

#from matplotlib.mlab import griddata
from scipy.interpolate import griddata

from mpl_toolkits.mplot3d import Axes3D

# standard plot settings definitions
plot_settings = {
    'Aerosol_Backscatter_Coefficient':{
        'climits':[1e-8,1e-3],
        'scale':'log',
	'colormap':'jet'
        },
    'Denoised_Aerosol_Backscatter_Coefficient':{
        'climits':[1e-8,1e-3],
        'scale':'log',
	'colormap':'jet'
        },
    'Aerosol_Extinction_Coefficient':{
        'climits':[1e-5,1e-2],
        'scale':'log',
	'colormap':'viridis'
        },
    'Denoised_Aerosol_Extinction_Coefficient':{
        'climits':[1e-5,1e-2],
        'scale':'log',
	'colormap':'viridis'
        },
    'Particle_Depolarization':{
        'climits':[0,0.8],
        'scale':'linear',
	'colormap':'viridis'
        },
    'Denoised_Particle_Depolarization':{
        'climits':[0,0.8],
        'scale':'linear',
	'colormap':'viridis'
        },
    'Volume_Depolarization':{
        'climits':[0,1.0],
        'scale':'linear',
	'colormap':'jet'
        },
    
    'Merged_Combined_Channel':{
        'climits':[1e-1,1e4],
        'scale':'log',
	'colormap':'jet'
        },
    'Lidar_Ratio':{
        'climits':[15,40],
        'scale':'linear',
	'colormap':'viridis'
        },
    'Denoised_Lidar_Ratio':{
        'climits':[15,40],
        'scale':'linear',
	'colormap':'viridis'
        },
    'Denoised_Absolute_Humidity':{
        'climits':[0,20],
        'scale':'linear',
	'colormap':'jet'
        },
    'Absolute_Humidity':{
        'climits':[0,20],
        'scale':'linear',
	'colormap':'jet'
        },
    'Denoised_Attenuated_Backscatter':{
        'climits':[1e-3,1e3],
        'scale':'log',
	'colormap':'jet'
        },
    'Attenuated_Backscatter':{
        'climits':[1e-3,1e3],
        'scale':'log',
	'colormap':'jet'
        },
    'WV_Online_Backscatter_Channel':{
        'climits':[1e0,1e4],
        'scale':'log',
        'colormap':'jet'},
    'WV_Offline_Backscatter_Channel':{
        'climits':[1e0,1e4],
        'scale':'log',
        'colormap':'jet'}
    }


def plotprofiles(proflist,varplot=False,time=None,scale='log',fignum=None,cindex=0,loc=1,marker='-'):
    """
    plot profiles listed on an axis.
    proflist - a list of lidar profiles
    varplot - if set to True, plots the standard deviation of the signal
    time - if set, it only plots the profile corresponding to the time provided.
        time is provided in seconds from 0000 UTC on the profile start date
    scale - plot on a 'log' or 'linear' axis.
    fignum - sets the figure number you want to plot to
    cindex - sets the color index you want to start at
    loc - location of the legend
    marker - set the marker type wanted for all plots in the list
    """
    colorlist = ['b','g','r','c','m','y','k']
    if fignum is None:
        fignum = plt.figure()
    else:
        plt.figure(fignum)
#    for ai in range(len(proflist)):
    for ai,p0 in enumerate(proflist):
        if isinstance(proflist,dict):
            p1 = proflist[p0].copy()
        else:
            p1 = p0.copy()
#            p1 = proflist[ai].copy()
        if time is None:
            p1.time_integrate()
            if varplot:
                
                if scale == 'log':
                    plt.semilogx(np.sqrt(p1.profile_variance.flatten()),p1.range_array.flatten(),colorlist[np.mod(ai+cindex,len(colorlist))]+'--',label=p1.label+' std.')
                else:
                    plt.fill_betweenx(p1.range_array,p1.profile.flatten()-np.sqrt(p1.profile_variance.flatten()),p1.profile.flatten()+np.sqrt(p1.profile_variance.flatten()),facecolor=colorlist[np.mod(ai+cindex,len(colorlist))],alpha=0.2)
#                    plt.plot(np.sqrt(p1.profile_variance.flatten()),p1.range_array.flatten(),colorlist[np.mod(ai+cindex,len(colorlist))]+'--',label=p1.label+' std.')
            
            if scale == 'log':
                plt.semilogx(p1.profile.flatten(),p1.range_array.flatten(),colorlist[np.mod(ai+cindex,len(colorlist))]+marker,label=p1.label)
            else:
                plt.plot(p1.profile.flatten(),p1.range_array.flatten(),colorlist[np.mod(ai+cindex,len(colorlist))]+marker,label=p1.label)
            
        else:
            itime = np.argmin(np.abs(p1.time-time))
            if varplot:
                
                if scale == 'log':
                    plt.semilogx(np.sqrt(p1.profile_variance[itime,:]),p1.range_array.flatten(),colorlist[np.mod(ai+cindex,len(colorlist))]+'--',label=p1.label+' std.')
                else:
                    plt.fill_betweenx(p1.range_array,p1.profile[itime,:]-np.sqrt(p1.profile_variance[itime,:]),p1.profile[itime,:]+np.sqrt(p1.profile_variance[itime,:]),facecolor=colorlist[np.mod(ai+cindex,len(colorlist))],alpha=0.2)
#                    plt.plot(np.sqrt(p1.profile_variance[itime,:]),p1.range_array.flatten(),colorlist[np.mod(ai+cindex,len(colorlist))]+'--',label=p1.label+' std.')
            if scale == 'log':
                plt.semilogx(p1.profile[itime,:],p1.range_array.flatten(),colorlist[np.mod(ai+cindex,len(colorlist))]+marker,label=p1.label)
            else:
                plt.plot(p1.profile[itime,:],p1.range_array.flatten(),colorlist[np.mod(ai+cindex,len(colorlist))]+marker,label=p1.label)
            
        units = p1.profile_type
        if ('_' in units or '^' in units) and not ('$' in units):
                units = units.replace('(','{')
                units = units.replace(')','}')
                units = '$'+units+'$'
        
        plt.grid(b=True);
        plt.legend(loc=loc)
        plt.ylabel('Range [m]')
        plt.xlabel(units)
    
    return fignum
        
        
def pcolor_profiles(proflist0,ylimits=None,tlimits=None,
                    climits=None,plotAsDays=False,scale=None,cmap=None,
                    title_font_size=0,title_add ='',plot_date=False,
                    t_axis_scale=1.0, h_axis_scale=1.0,
                    minor_ticks=0,major_ticks=1.0,plt_kft=False,
                    tick_format=None,title_date_format="%A %B %d, %Y",
                    colorbar_width=0.1,colorbar_pad=0.2):
    """
    pcolor_profiles(proflist0,ylimits=None,tlimits=None,
                    climits=None,plotAsDays=False,scale=None,cmap=None,
                    title_font_size=0,title_add ='',plot_date=False,
                    t_axis_scale=1.0, h_axis_scale=1.0,
                    minor_ticks=0,major_ticks=1.0,plt_kft=False,
                    tick_format=None,title_date_format="%A %B %d, %Y",
                    colorbar_width=0.1,colorbar_pad=0.2)
    
    plot time and range resolved profiles as pcolors
    proflist - a list of lidar profiles
    ylimits - list containing upper and lower bounds of plots in km
    tlimits - list containing upper and lower bounds of plots in days or hours (plotAsDays=True or plotAsDays=False)
    climits - list containing a list of the colorbar limits.  e.g. [[0,5],[4,10]] for two profiles.
            if the list of limits includes a nan, that profile will use the default scale
    plotAsDays - True: sets the time axis units as days
                 False: sets the time axis as hours
    scale - a list containing the desired scale for the plots.  set to 'linear' or 'log'.  Defaults to 'log'
    cmap - plotting colormap.  If not specified, it uses 'jet' (standard default).
    title_font_size - sets the size of the font in the title.  If unspecified it adjusts it based on the figure length.
    title_add - string for additional information at the start of the title
    plot_date - if true, it plots the x ticks as dates/time instead of floats representing fraction of an hour
    t_axis_scale - scale factor for sizing the width of the plot.  
                    >1.0 - bigger plot width per unit time
    h_axis_scale - scale factor for vertical sizing of the plot
                    >1.0 - bigger plot per unit altitude/range
    minor_ticks - interval in minutes for minor ticks on x axis
    major_ticks - interval in hours for major ticks on the x axis
    plt_kft - boolean to plot the data in kft instead of km
    tick_format - string indicating matplotlib date format
        e.g. '%b %d %H:%M is month day hours:minutes
    title_date_format - string indicating the printed date format
        default is "%A %B %d, %Y".
        set to empty string if no date is desired
    colorbar_width - fraction of figure to use for the colorbar.  default is 0.1
    colorbar_pad - fraction of figure to pad the colorbar from the main figure.  default is 0.2
    """
    
    if ylimits is None:
        ylimits = [0,np.nan]
    if tlimits is None:
        tlimits = [np.nan,np.nan]
    if climits is None:
        climits = []
    if scale is None:
        scale = []
    if cmap is None:
        cmap = []
    
    
    Nprof = np.double(len(proflist0))
    
    proflist =[]
    for iprof in range(np.int(Nprof)):
        proflist = proflist+[proflist0[iprof]]
    
    if plotAsDays:
        time_scale = 3600*24.0
        span_scale = 24.0
    else:
        time_scale = 3600.0
        span_scale = 1.0

    # if plotting in kft, adjust the range scales    
    # this does not affect range limits.
    if plt_kft:
        range_factor = 3.28084
        range_label = 'kft'
    else:
        range_factor = 1
        range_label = 'km'
        
    # if scale is not provided or it is not provided for all profiles
    # assign the not provided plots with a log scale
    if len(scale) < len(proflist):
        scale.extend(['log']*(len(proflist)-len(scale)))
       
    # if the color scale limits is not provided or not enough color scale 
    # entries are provided, set the color limits to auto (nan = auto)
    if len(climits) < len(proflist):
        climits.extend([[np.nan,np.nan]]*(len(proflist)-len(climits)))
    
    # if cmap is not provided or it is not provided for all profiles
    # assign the not provided plots with 'jet'
    if len(cmap) < len(proflist):
        cmap.extend(['jet']*(len(proflist)-len(cmap)))    
    
    tmin = 1e9
    tmax = 0
    ymin = 1e9
    ymax = 0
    
    for ai in range(len(proflist)):
        tmin = np.min(np.array([tmin,proflist[ai].time[0]/time_scale]))
        tmax = np.max(np.array([tmax,proflist[ai].time[-1]/time_scale]))
        ymin = np.min(np.array([ymin,proflist[ai].range_array[0]*1e-3*range_factor]))
        ymax = np.max(np.array([ymax,proflist[ai].range_array[-1]*1e-3*range_factor]))
    if np.isnan(tlimits[0]):
        tlimits[0] = tmin
    if np.isnan(tlimits[1]):
        tlimits[1] = tmax
    if np.isnan(ylimits[0]):
        ylimits[0] = ymin
    if np.isnan(ylimits[1]):
        ylimits[1] = ymax
        
#    print ('time limits:')
#    print(tlimits)
    # scale figure dimensions based on time and altitude dimensions
    time_span = tlimits[1]*span_scale-tlimits[0]*span_scale  # time domain of plotted data
    range_span = (ylimits[1]-ylimits[0])/range_factor  # range domain of plotted data
    
    if title_font_size == 0:
        # adjust title line based on the amount of plotted time data
        if time_span*t_axis_scale < 8.0:
            # short plots (in time)
            line_char = '\n'  # include a newline to fit full title
            y_top_edge = 1.2  # top edge set for double line title
            title_font_size = 12  # use larger title font
        elif time_span*t_axis_scale <= 16.0:
            # medium plots (in time)
            line_char = ' '  # no newline in title
            y_top_edge = 0.9  # top edge set for single line title
            title_font_size = 12  # use smaller title font
        else:
            # long plots (in time)
            line_char = ' '  # no newline in title
            y_top_edge = 0.9  # top edge set for single line title
            title_font_size = 16  # use larger title font
    else:
        line_char = ' '
        y_top_edge = 0.9
    
    max_len = 18.0
    min_len = 2.0
    max_h = 8.0
    min_h = 0.2
    x_left_edge =1.0
    x_right_edge = 2.0
    y_bottom_edge = 0.6

    
    ax_len = np.max(np.array([np.min(np.array([max_len,time_span*18.0/24.0*t_axis_scale])),min_len])) # axes length
    ax_h = np.max(np.array([np.min(np.array([max_h,range_span*2.1/12*h_axis_scale])),min_h]))  # axes height
    fig_len = x_left_edge+x_right_edge+ax_len  # figure length
    fig_h =y_bottom_edge+y_top_edge+ax_h  # figure height
    
    axL = []   # axes list
    caxL = []  # color axes list
    imL = []   # image list
    
    fig = plt.figure(figsize=(fig_len,Nprof*fig_h))
        
    for ai in range(len(proflist)): 
        axlim = [x_left_edge/fig_len,y_bottom_edge/fig_h/Nprof+(Nprof-ai-1)/Nprof,1-x_right_edge/fig_len,(1-y_top_edge/fig_h)/Nprof]
   
        
        ax = plt.axes(axlim) 
        proflist[ai].fill_blanks()  # fill in missing times with masked zeros
        # only actually create the plot if there is actual data to plot
        if proflist[ai].profile.size > 0 and np.sum(proflist[ai].profile.mask) < proflist[ai].profile.size:
            if plot_date:
                x_time = mdates.date2num([datetime.datetime.fromordinal(proflist[ai].StartDate.toordinal()) \
                    + datetime.timedelta(seconds=sec) for sec in proflist[ai].time])
            else:
                x_time = proflist[ai].time/time_scale
            if scale[ai] == 'log':
                im = plt.pcolor(x_time,proflist[ai].range_array*1e-3*range_factor,proflist[ai].profile.T \
                    ,norm=matplotlib.colors.LogNorm(),cmap=cmap[ai])  # np.real(np.log10(proflist[ai].profile)).T
            else:
                im = plt.pcolor(x_time,proflist[ai].range_array*1e-3*range_factor,proflist[ai].profile.T,cmap=cmap[ai])
           
            if not any(np.isnan(climits[ai])):
                plt.clim(climits[ai])  # if no nans in color limits, apply to the plot.  otherwise go with the auto scale
            
            if plot_date:
                plt.gcf().autofmt_xdate()
                if tick_format is None:
                    if plotAsDays:
                        myFmt = mdates.DateFormatter('%b %d %H:%M')
                    else:
                        myFmt = mdates.DateFormatter('%H:%M')
                else:
                    myFmt = mdates.DateFormatter(tick_format)
                plt.gca().xaxis.set_major_formatter(myFmt)
                if major_ticks >= 1.0:
                    # if major ticks is > 1, treat it as hourly ticks
                    major_ticks = np.int(np.max(np.array([major_ticks,1.0])))
                    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=major_ticks))  # mod for 5 min socrates plots
                else:
                    # if major ticks is < 1, treat it as mintue ticks
                    # and disable minor ticks to avoid overlaping labels
                    major_ticks = np.int(np.round(60*major_ticks))
                    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=major_ticks))
                    plt.gca().tick_params(axis='x', which='major', labelsize=8)
                    minor_ticks = 0
                if minor_ticks > 0:
                    minor_ticks_array = np.arange(minor_ticks,60,minor_ticks)
                    minor_ticks_array = np.setdiff1d(minor_ticks_array,major_ticks)
                    if major_ticks < 1.0:
                        # only include hours in label if major ticks are at hour increments
                        plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
                    else:
                        plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter(':%M'))
                    plt.gca().xaxis.set_minor_locator(mdates.MinuteLocator(byminute=minor_ticks_array,interval=1))
                    plt.gca().tick_params(axis='x', which='minor', labelsize=8)
                plt.setp(plt.gca().xaxis.get_majorticklabels(),rotation=0,horizontalalignment='center')
                if plotAsDays:
                    xl1 = mdates.date2num(datetime.datetime.fromordinal(proflist[ai].StartDate.toordinal())+\
                        datetime.timedelta(seconds=tlimits[0]*3600*24))
                    xl2 = mdates.date2num(datetime.datetime.fromordinal(proflist[ai].StartDate.toordinal())+\
                        datetime.timedelta(seconds=tlimits[1]*3600*24))
                else:
                    xl1 = mdates.date2num(datetime.datetime.fromordinal(proflist[ai].StartDate.toordinal())+\
                        datetime.timedelta(seconds=tlimits[0]*3600))
                    xl2 = mdates.date2num(datetime.datetime.fromordinal(proflist[ai].StartDate.toordinal())+\
                        datetime.timedelta(seconds=tlimits[1]*3600))
                plt.xlim(np.array([xl1,xl2]))  
            else:
                plt.xlim(np.array(tlimits))
                
            plt.ylim(ylimits)
            
            if len(title_date_format) > 0:
                DateLabel = proflist[ai].StartDate.strftime("%A %B %d, %Y")
            else:
                DateLabel = ""
            units = proflist[ai].profile_type
            if ('_' in units or '^' in units) and not ('$' in units):
                units = units.replace('(','{')
                units = units.replace(')','}')
                units = '$'+units+'$'
            plt.title(title_add+DateLabel + ', ' +proflist[ai].lidar + line_char + proflist[ai].label ,fontsize=title_font_size) # +' [' + units + ']'
            plt.ylabel('Altitude ['+range_label+']')
            if plotAsDays:
                plt.xlabel('Days [UTC]')
            else:
                plt.xlabel('Time [UTC]')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right",size=colorbar_width,pad=colorbar_pad) # size=0.1 pad=0.2, size=0.05,pad=0.5
            plt.colorbar(im,cax=cax,label=units)
            
            axL.extend([ax])
            caxL.extend([cax])
            imL.extend([imL])
        else:
            print('pcolor_profiles() is ignoring request to plot ' + proflist[ai].label + '.')
            print('   This profile has no valid data')
    return fig,axL,caxL,imL
        

def pcolor_profiles_official(proflist,ylimits=None,tlimits=None,climits=None,plotAsDays=False,scale=None,cmap=None,plot_mask=None,title_font_size=0):
    """
    Used to produce field catalog plots
    
    pcolor_profiles_official(proflist,ylimits=[0,np.nan],tlimits=[np.nan,np.nan],climits=[],plotAsDays=False,scale=[],cmap=[],title_font_size=0)
    
    plot time and range resolved profiles as pcolors
    proflist - a list of lidar profiles
    ylimits - list containing upper and lower bounds of plots in km
    tlimits - list containing upper and lower bounds of plots in days or hours (plotAsDays=True or plotAsDays=False)
    climits - list containing a list of the colorbar limits.  e.g. [[0,5],[4,10]] for two profiles.
            if the list of limits includes a nan, that profile will use the default scale
    plotAsDays - True: sets the time axis units as days
                 False: sets the time axis as hours
    scale - a list containing the desired scale for the plots.  set to 'linear' or 'log'.  Defaults to 'log'
    cmap - plotting colormap.  If not specified, it uses 'jet' (standard default).
    title_font_size - sets the size of the font in the title.  If unspecified it adjusts it based on the figure length.
    """
    
    Nprof = np.double(len(proflist))
    if plotAsDays:
        time_scale = 3600*24.0
        span_scale = 24.0
    else:
        time_scale = 3600.0
        span_scale = 1.0
    
    if ylimits is None:
        ylimits=[]
    if tlimits is None:
        tlimits = [0,24.0]
    if climits is None:
        climits =[]
    if scale is None:
        scale = []
    if cmap is None:
        cmap = []
    if plot_mask is None:
        plot_mask = []
      
    # if scale is not provided or it is not provided for all profiles
    # assign the not provided plots with a log scale
    if len(scale) < len(proflist):
        scale.extend(['log']*(len(proflist)-len(scale)))
       
    # if the color scale limits is not provided or not enough color scale 
    # entries are provided, set the color limits to auto (nan = auto)
    if len(climits) < len(proflist):
        climits.extend([[np.nan,np.nan]]*(len(proflist)-len(climits)))
    
    # if cmap is not provided or it is not provided for all profiles
    # assign the not provided plots with 'jet'
    if len(cmap) < len(proflist):
        cmap.extend(['jet']*(len(proflist)-len(cmap)))    
    
    # default to blank masked data points
    if len(plot_mask) < len(proflist):
        plot_mask.extend([True]*(len(proflist)-len(cmap))) 
    
    tmin = 1e9
    tmax = 0
    ymin = 1e9
    ymax = 0
    
    if len(ylimits) < len(proflist):
        ylimits.extend([0,np.nan]*(len(proflist)-len(cmap)))     
    
    for ai in range(len(proflist)):
        tmin = np.min(np.array([tmin,proflist[ai].time[0]/time_scale]))
        tmax = np.max(np.array([tmax,proflist[ai].time[-1]/time_scale]))
        if np.isnan(ylimits[ai][0]):
            ymin = np.min(np.array([ymin,proflist[ai].range_array[0]*1e-3]))
        else:
            ymin = np.min(np.array([ymin,ylimits[ai][0]]))
        if np.isnan(ylimits[ai][1]):
            ymax = np.max(np.array([ymax,proflist[ai].range_array[-1]*1e-3]))
        else:
            ymax = np.max(np.array([ymax,ylimits[ai][1]]))
#    if np.isnan(tlimits[0]):
#        tlimits[0] = tmin
#    if np.isnan(tlimits[1]):
#        tlimits[1] = tmax
    for ai in range(len(proflist)):
        if np.isnan(ylimits[ai][0]):
            ylimits[ai][0] = ymin
        if np.isnan(ylimits[ai][1]):
            ylimits[ai][1] = ymax
    
    tlimits=[0,24]
    
    # scale figure dimensions based on time and altitude dimensions
    time_span = tlimits[1]*span_scale-tlimits[0]*span_scale  # time domain of plotted data
    range_span = ymax-ymin #  ylimits[1]-ylimits[0]  # range domain of plotted data
    
    if title_font_size == 0:
        # adjust title line based on the amount of plotted time data
        if time_span < 8.0:
            # short plots (in time)
            line_char = '\n'  # include a newline to fit full title
            y_top_edge = 1.2  # top edge set for double line title
            title_font_size = 12  # use larger title font
        elif time_span <= 16.0:
            # medium plots (in time)
            line_char = ' '  # no newline in title
            y_top_edge = 0.9  # top edge set for single line title
            title_font_size = 12  # use smaller title font
        else:
            # long plots (in time)
            line_char = ' '  # no newline in title
            y_top_edge = 0.9  # top edge set for single line title
            title_font_size = 16  # use larger title font
    else:
        line_char = ' '
        y_top_edge = 0.9
    
    max_len = 9.0
    min_len = 2.0
    max_h = 8.0
    min_h = 0.2
    x_left_edge =1.0
    x_right_edge = 2.0
    y_bottom_edge = 0.6

    
    ax_len = np.max(np.array([np.min(np.array([max_len,time_span*max_len/24.0])),min_len])) # axes length
    ax_h = np.max(np.array([np.min(np.array([max_h,range_span*2.1/12])),min_h]))  # axes height
    fig_len = x_left_edge+x_right_edge+ax_len  # figure length
    fig_h =y_bottom_edge+y_top_edge+ax_h  # figure height
    
    axL = []   # axes list
    caxL = []  # color axes list
    imL = []   # image list
    
    fig = plt.figure(figsize=(fig_len,Nprof*fig_h))
        
    for ai in range(len(proflist)): 
        axlim = [x_left_edge/fig_len,y_bottom_edge/fig_h/Nprof+(Nprof-ai-1)/Nprof,1-x_right_edge/fig_len,(1-y_top_edge/fig_h)/Nprof]
   
        
        ax = plt.axes(axlim) 
        proflist[ai].fill_blanks()  # fill in missing times with masked zeros
        if lp.has_mask(proflist[ai].profile) and not plot_mask[ai]:
            plot_prof = proflist[ai].profile.data.T        
        else:
            plot_prof = proflist[ai].profile.copy().T   
        
        if scale[ai] == 'log':
            im = plt.pcolor(proflist[ai].time/time_scale,proflist[ai].range_array*1e-3,plot_prof \
                ,norm=matplotlib.colors.LogNorm(),cmap=cmap[ai])  # np.real(np.log10(proflist[ai].profile)).T
        else:
            im = plt.pcolor(proflist[ai].time/time_scale,proflist[ai].range_array*1e-3,plot_prof,cmap=cmap[ai])
       
        if not any(np.isnan(climits[ai])):
            plt.clim(climits[ai])  # if no nans in color limits, apply to the plot.  otherwise go with the auto scale
            
        plt.ylim(ylimits[ai])
        plt.xlim(np.array(tlimits))
        plt.xticks(np.arange(np.ceil(tlimits[0]),np.floor(tlimits[1])+1,1.0))
        DateLabel = proflist[ai].StartDate.strftime("%A %B %d, %Y")
        units = proflist[ai].profile_type
        if '_' in units or '^' in units and not ('$' in units):
            units = units.replace('(','{')
            units = units.replace(')','}')
            units = '$'+units+'$'
        plt.title(DateLabel + ', ' +proflist[ai].lidar + line_char + proflist[ai].label +' [' + units + ']',fontsize=title_font_size)
        plt.ylabel('Altitude AGL [km]')
        if plotAsDays:
            plt.xlabel('Days [UTC]')
        else:
            plt.xlabel('Time [UTC]')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",size=0.1,pad=0.2)
        plt.colorbar(im,cax=cax)
        
        axL.extend([ax])
        caxL.extend([cax])
        imL.extend([imL])
        
    return fig,axL,caxL,imL

def pcolor_airborne(lidar_prof0,lidar_pointing = np.array([]),lidar_alt=None,
                  tlimits=[np.nan,np.nan],ylimits=[0,np.nan],climits=[],scale='log',
                  cmap='jet',title_font_size=0,title_add ='',s=2,plotAsDays=False,
                  t_axis_scale=1.0,h_axis_scale=1.0,plt_kft=False,plot_date=False,
                  minor_ticks=0,major_ticks=1.0):
                      
    """
        plots range centered data on a time/altitude grid
        profile - profile to be plotted
        lidar_pointing - normalized vector describing the lidar pointing direction in 3D space
            index 0 - North direction
            index 1 - East direction
            index 2 - Down direction
        
        s-plot point size
            
    """
    lidar_prof = lidar_prof0.copy()
    
    Nprof = 1
    
    if lidar_alt is None:
        lidar_alt = np.array([0])
    
    if plotAsDays:
        time_scale = 3600*24.0
        span_scale = 24.0
    else:
        time_scale = 3600.0
        span_scale = 1.0    
    
    # if plotting in kft, adjust the range scales    
    # this does not affect range limits.
    if plt_kft:
        range_factor = 3.28084
        range_label = 'kft'
    else:
        range_factor = 1
        range_label = 'km'
    
    if lidar_pointing.size > 0:
        alt_data = (-lidar_pointing[2,:][:,np.newaxis]*lidar_prof.range_array+lidar_alt[:,np.newaxis]).flatten()
    else:
        print('in scatter_z():  No lidar pointing data.  Assuming vertical.')
        alt_data = (lidar_prof.range_array+lidar_alt[:,np.newaxis]).flatten()
    t_data_plt = (np.ones((1,lidar_prof.range_array.size))*(lidar_prof.time[:,np.newaxis])).flatten()
    t_data_1d = lidar_prof.time[:,np.newaxis]/time_scale
    
    # trim data that is outside the time and range limits
    if hasattr(lidar_prof.profile,'mask'):
        plot_prof = (lidar_prof.profile.data.copy()).flatten()
        plot_prof_mask = lidar_prof.profile.mask.flatten()
    else:
        plot_prof = (lidar_prof.profile.copy()).flatten()
        plot_prof_mask = np.zeros(plot_prof.size,dtype=bool)
    irm2d = np.nonzero((alt_data*1e-3*range_factor > ylimits[1]) + (alt_data*1e-3*range_factor < ylimits[0]) + (t_data_plt/time_scale < tlimits[0]) +(t_data_plt/time_scale > tlimits[1]) )#+np.isnan(plot_prof)+plot_prof_mask)
    irm1d = np.nonzero((t_data_1d/time_scale < tlimits[0]) +(t_data_1d/time_scale > tlimits[1]))
    
    plot_prof[plot_prof_mask] = np.nan
    
    alt_data = np.delete(alt_data,irm2d)
    t_data_plt = np.delete(t_data_plt,irm2d)
    plot_prof = np.delete(plot_prof,irm2d)
    
    lidar_alt = np.delete(lidar_alt,irm1d)
    t_data_1d = np.delete(t_data_1d,irm1d)
    
    
    
    if np.isnan(tlimits[0]):
        tlimits[0] = t_data_1d[0]
    if np.isnan(tlimits[1]):
        tlimits[1] = t_data_1d[-1]
    if np.isnan(ylimits[0]):
        ylimits[0] = np.min(alt_data)*1e-3*range_factor
    if np.isnan(ylimits[1]):
        ylimits[1] = np.max(alt_data)*1e-3*range_factor
        
#    print ('time limits:')
#    print(tlimits)
    # scale figure dimensions based on time and altitude dimensions
    time_span = tlimits[1]*span_scale-tlimits[0]*span_scale  # time domain of plotted data
    range_span = (ylimits[1]-ylimits[0])/range_factor  # range domain of plotted data
    
    if title_font_size == 0:
        # adjust title line based on the amount of plotted time data
        if time_span*t_axis_scale < 8.0:
            # short plots (in time)
            line_char = '\n'  # include a newline to fit full title
            y_top_edge = 1.2  # top edge set for double line title
            title_font_size = 12  # use larger title font
        elif time_span*t_axis_scale <= 16.0:
            # medium plots (in time)
            line_char = ' '  # no newline in title
            y_top_edge = 0.9  # top edge set for single line title
            title_font_size = 12  # use smaller title font
        else:
            # long plots (in time)
            line_char = ' '  # no newline in title
            y_top_edge = 0.9  # top edge set for single line title
            title_font_size = 16  # use larger title font
    else:
        line_char = ' '
        y_top_edge = 0.9
    
    max_len = 18.0
    min_len = 2.0
    max_h = 8.0
    min_h = 0.2
    x_left_edge =1.0
    x_right_edge = 2.0
    y_bottom_edge = 0.6

    
    ax_len = np.max(np.array([np.min(np.array([max_len,time_span*18.0/24.0*t_axis_scale])),min_len])) # axes length
    ax_h = np.max(np.array([np.min(np.array([max_h,range_span*2.1/12*h_axis_scale])),min_h]))  # axes height
    fig_len = x_left_edge+x_right_edge+ax_len  # figure length
    fig_h =y_bottom_edge+y_top_edge+ax_h  # figure height
    
#    axL = []   # axes list
#    caxL = []  # color axes list
#    imL = []   # image list
    
    fig = plt.figure(figsize=(fig_len,Nprof*fig_h))

    ai = 0
    axlim = [x_left_edge/fig_len,y_bottom_edge/fig_h/Nprof+(Nprof-ai-1)/Nprof,1-x_right_edge/fig_len,(1-y_top_edge/fig_h)/Nprof]
           
    ax = plt.axes(axlim) 
      
    
    lidar_prof.fill_blanks()  # fill in missing times with masked zeros
    tgrid = lidar_prof.time
    ygrid = np.arange(alt_data.min(),alt_data.max(),lidar_prof.mean_dR)
    tigrid,yigrid = np.meshgrid(tgrid,ygrid) # scipy version of griddata requires meshgrid instead of axes definitions

    
#    plot_profile = griddata(t_data_plt,alt_data,plot_prof,tgrid,ygrid,interp='linear')  # matplotlib.mlab.griddata
    plot_profile = griddata((t_data_plt,alt_data,),plot_prof,(tigrid,yigrid,),method='linear')  # scipy.interpolate.griddata
    
    plot_profile_mask = np.isnan(plot_profile)
    plot_profile = np.ma.array(plot_profile,mask=plot_profile_mask)

        
    # only actually create the plot if there is actual data to plot
    if plot_profile.size > 0 and np.sum(lidar_prof.profile.mask) < lidar_prof.profile.size:
        if plot_date:
            x_time = mdates.date2num([datetime.datetime.fromordinal(lidar_prof.StartDate.toordinal()) \
                + datetime.timedelta(seconds=sec) for sec in tgrid])
        else:
            x_time =tgrid/time_scale
        if scale == 'log':
            im = plt.pcolor(x_time,ygrid*1e-3*range_factor,plot_profile \
                ,norm=matplotlib.colors.LogNorm(),cmap=cmap)  # np.real(np.log10(proflist[ai].profile)).T
        else:
            im = plt.pcolor(x_time,ygrid*1e-3*range_factor,plot_profile,cmap=cmap)
       
        if not any(np.isnan(climits)):
            plt.clim(climits)  # if no nans in color limits, apply to the plot.  otherwise go with the auto scale
        
        if plot_date:
            plt.gcf().autofmt_xdate()
            if plotAsDays:
                myFmt = mdates.DateFormatter('%b %d %H:%M')
            else:
                if major_ticks >= 1.0:
                    myFmt = mdates.DateFormatter('%H:%M')
                elif major_ticks < 1.0/60:
                    myFmt = mdates.DateFormatter('%H:%M:%S')
                else:
                    myFmt = mdates.DateFormatter('%H:%M')
            plt.gca().xaxis.set_major_formatter(myFmt)
            if major_ticks >= 1.0:
                # if major ticks is > 1, treat it as hourly ticks
                major_ticks = np.int(np.max(np.array([major_ticks,1.0])))
                plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=major_ticks))  # mod for 5 min socrates plots
            elif major_ticks < 1.0/60:
                # if major ticks is < 1 minute (1/60), treat it as second ticks
                # and disable minor ticks to avoid overlaping labels
                major_ticks = np.int(np.round(3600*major_ticks))
                plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=major_ticks))
                plt.gca().tick_params(axis='x', which='major', labelsize=8)
                minor_ticks = 0
            else:
                # if major ticks is < 1, treat it as mintue ticks
                # and disable minor ticks to avoid overlaping labels
                major_ticks = np.int(np.round(60*major_ticks))
                plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=major_ticks))
                plt.gca().tick_params(axis='x', which='major', labelsize=8)
                minor_ticks = 0
            if minor_ticks > 0:
                minor_ticks_array = np.arange(minor_ticks,60,minor_ticks)
                minor_ticks_array = np.setdiff1d(minor_ticks_array,major_ticks)
                if major_ticks < 1.0 and minor_ticks < 1.0:
                    # only include hours in label if major ticks are at hour increments
                    plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('::%S'))
                elif major_ticks < 1.0 and minor_ticks > 1.0:
                    plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter(':%M'))
                elif major_ticks > 1.0 and minor_ticks < 1.0:
                    plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter(':%M:%S'))
                else:
                    plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
                if minor_ticks < 1.0:
                    plt.gca().xaxis.set_minor_locator(mdates.SecondLocator(bysecond=minor_ticks_array,interval=np.int(minor_ticks*60)))
                else:
                    plt.gca().xaxis.set_minor_locator(mdates.MinuteLocator(byminute=minor_ticks_array,interval=minor_ticks))
                plt.gca().tick_params(axis='x', which='minor', labelsize=8)
            plt.setp(plt.gca().xaxis.get_majorticklabels(),rotation=0,horizontalalignment='center')
            if plotAsDays:
                xl1 = mdates.date2num(datetime.datetime.fromordinal(lidar_prof.StartDate.toordinal())+\
                    datetime.timedelta(seconds=tlimits[0]*3600*24))
                xl2 = mdates.date2num(datetime.datetime.fromordinal(lidar_prof.StartDate.toordinal())+\
                    datetime.timedelta(seconds=tlimits[1]*3600*24))
            else:
                xl1 = mdates.date2num(datetime.datetime.fromordinal(lidar_prof.StartDate.toordinal())+\
                    datetime.timedelta(seconds=tlimits[0]*3600))
                xl2 = mdates.date2num(datetime.datetime.fromordinal(lidar_prof.StartDate.toordinal())+\
                    datetime.timedelta(seconds=tlimits[1]*3600))
            if len(lidar_alt) > 1:
                plt.plot(x_time,lidar_alt*1e-3*range_factor,color='gray',linewidth=1.5)
            plt.xlim(np.array([xl1,xl2]))  
        else:
            plt.xlim(np.array(tlimits))
            
        plt.ylim(ylimits)
        
        DateLabel = lidar_prof.StartDate.strftime("%A %B %d, %Y")
#        plt.title(title_add+DateLabel + ', ' +lidar_prof.lidar + line_char + lidar_prof.label +' [' + lidar_prof.profile_type + ']',fontsize=title_font_size)
        plt.title(title_add+DateLabel + ', ' +lidar_prof.lidar + line_char + lidar_prof.label,fontsize=title_font_size)
        plt.ylabel('Altitude ['+range_label+']')
        if plotAsDays:
            plt.xlabel('Days [UTC]')
        else:
            plt.xlabel('Time [UTC]')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",size=0.1,pad=0.2)
        units = lidar_prof.profile_type
        if ('_' in units or '^' in units) and not ('$' in units):
            units = units.replace('(','{')
            units = units.replace(')','}')
            units = '$'+units+'$'
        plt.colorbar(im,cax=cax,label=units)
        
        
        
        return fig,ax,cax,im
    else:
        # no data valid data was availabe to be plotted
        return fig,ax,np.nan,np.nan
    
def scatter_z(lidar_prof,ax=None,lidar_pointing = None,lidar_alt=None,
                  tlimits=None,ylimits=None,climits=None,scale='log',
                  cmap='jet',title_font_size=12,title_add ='',s=2,
                  t_axis_scale=1.0,h_axis_scale=1.0,plt_kft=False):
    """
    plots range centered data on a time/altitude grid
    profile - profile to be plotted
    lidar_pointing - normalized vector describing the lidar pointing direction in 3D space
        index 0 - North direction
        index 1 - East direction
        index 2 - Down direction
    
    s-plot point size
        
    """
    ax = ax or plt.gca()

    if lidar_alt is None:
        lidar_alt = np.zeros(lidar_prof.time.size)
    elif np.isscalar(lidar_alt):
        lidar_alt = lidar_alt*np.ones(lidar_prof.time.size)
        
    if tlimits is None:
        tlimits = [np.nan,np.nan]
    if ylimits is None:
        ylimits = [0,np.nan]
    
    if climits is None:
        climits = []
    
    time_scale = 3600.0

    # if plotting in kft, adjust the range scales    
    # this does not affect range limits.
    if plt_kft:
        range_factor = 3.28084
        range_label = 'kft'
    else:
        range_factor = 1
        range_label = 'km'
    
    if lidar_pointing is None:
        print('in scatter_z():  No lidar pointing data.  Assuming vertical.')
        alt_data = (lidar_prof.range_array[np.newaxis,:]+lidar_alt[:,np.newaxis]).flatten()*1e-3*range_factor  
    else:
        alt_data = (-lidar_pointing[2,:][:,np.newaxis]*lidar_prof.range_array+lidar_alt[:,np.newaxis]).flatten()*1e-3*range_factor     
        
    t_data_plt = (np.ones((1,lidar_prof.range_array.size))*(lidar_prof.time[:,np.newaxis])/time_scale).flatten()
#    # Set up plot times
#    t_data_plt=[]
#    for sec1 in range(t_data_plt1.size):
#        t_data_plt.append(datetime.timedelta(seconds=t_data_plt1[sec1])+lidar_prof.StartDate)
    t_data_1d = lidar_prof.time[:,np.newaxis]/time_scale
        
    # trim data that is outside the time and range limits
    if hasattr(lidar_prof.profile,'mask'):
        plot_prof = (lidar_prof.profile.data.copy()).flatten()
        plot_prof_mask = lidar_prof.profile.mask.flatten()
    else:
        plot_prof = (lidar_prof.profile.copy()).flatten()
        plot_prof_mask = np.zeros(plot_prof.size,dtype=bool)
    irm2d = np.nonzero((alt_data > ylimits[1]) + (alt_data < ylimits[0]) + (t_data_plt < tlimits[0]) +(t_data_plt > tlimits[1])+np.isnan(plot_prof)+plot_prof_mask)
    irm1d = np.nonzero((t_data_1d < tlimits[0]) +(t_data_1d > tlimits[1]))
    
    alt_data = np.delete(alt_data,irm2d)
    t_data_plt = np.delete(t_data_plt,irm2d)
    plot_prof = np.delete(plot_prof,irm2d)
    
    lidar_alt = np.delete(lidar_alt,irm1d)
    t_data_1d = np.delete(t_data_1d,irm1d)
            
    if np.isnan(tlimits[0]):
        tlimits[0] = t_data_1d[0]
    if np.isnan(tlimits[1]):
        tlimits[1] = t_data_1d[-1]
    if np.isnan(ylimits[0]):
        ylimits[0] = np.min(alt_data)
    if np.isnan(ylimits[1]):
        ylimits[1] = np.max(alt_data)
 
    if plot_prof.size > 0:
    
        if scale == 'log':
            im = ax.scatter(t_data_plt,alt_data,c=plot_prof,s=s,linewidth=0,norm=matplotlib.colors.LogNorm(),cmap=cmap)
        else:
            im = ax.scatter(t_data_plt,alt_data,c=plot_prof,s=s,linewidth=0,cmap=cmap)
        if len(climits) > 0:    
            im.set_clim(climits[0])
        if len(lidar_alt) > 0:
            ax.plot(t_data_1d,lidar_alt*1e-3*range_factor,color='red',linewidth=1.5)
        ax.set_xlim(tlimits)
        ax.set_ylim(ylimits)
        getXticks1=ax.get_xticks()
        xTicksSecs=getXticks1*time_scale
        tempTicks=np.round(xTicksSecs/60)*60/time_scale
        ax.set_xticks(np.arange(tempTicks[1],tempTicks[-1],step=np.min(tempTicks[2:getXticks1.size]-tempTicks[1:getXticks1.size-1])))
        getXticks=ax.get_xticks()
        newLabAll=[]
        addTime=datetime.datetime(lidar_prof.StartDate.year,lidar_prof.StartDate.month,lidar_prof.StartDate.day,0,0,0)
        for lab in range(getXticks.size):
            currLab=getXticks[lab]
            newLab=datetime.timedelta(seconds=currLab*time_scale)+addTime
            newLabAll.append(newLab.strftime("%H:%M"))        
        ax.set_xticklabels(newLabAll)
        ax.set_xlim(tlimits)
        startLab=datetime.timedelta(seconds=t_data_plt[0]*time_scale)+addTime
        endLab=datetime.timedelta(seconds=t_data_plt[-1]*time_scale)+addTime
        ax.set_title(lidar_prof.label+ ', '+ title_add+startLab.strftime("%Y%m%d %H:%M") + ' to ' +endLab.strftime("%Y%m%d %H:%M"),fontsize=title_font_size)
        ax.set_ylabel('Altitude ['+range_label+']')
        ax.set_xlabel('Time [UTC]')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",size=0.1,pad=0.2)
        units = lidar_prof.profile_type
        if '_' in units or '^' in units and not ('$' in units):
            units = units.replace('(','{')
            units = units.replace(')','}')
            units = '$'+units+'$'
        cbar=plt.colorbar(im,cax=cax,label=units)
        
        return ax.boxplot
    else:
        # no data valid data was availabe to be plotted
        return ax.boxplot
    
def scatter_3d(lidar_prof,lidar_pointing = np.array([]),lat=np.array([]),lon=np.array([]),pos_ref=[np.nan,np.nan],
               lidar_alt=np.array([0]),tlimits=[np.nan,np.nan],ylimits=[0,np.nan],climits=[],scale='log',
                  cmap='jet',title_font_size=12,title_add ='',s=2,alpha=0.4,plot_aicraft=True):
    """
    generate 3d plot of lidar observations over a time span
    pos_ref [lat,lon] to reference distances to
    """
    # time data is still needed to trim the data set
    t_data_plt = (np.ones((1,lidar_prof.range_array.size))*(lidar_prof.time[:,np.newaxis]/3600.0)).flatten()
    t_data_1d = lidar_prof.time[:,np.newaxis]/3600.0
    
    # get aircraft (x,y) from latitude and longitude data
    a_pos_x,a_pos_y = lp.dist_from_latlon(lat,lon,lat0=pos_ref[0],lon0=pos_ref[1])
    a_pos_x = a_pos_x*1e-3
    a_pos_y = a_pos_y*1e-3
    a_pos_z = lidar_alt.copy()*1e-3
    
    # get (x,y,z) position of every lidar observation
    l_pos_x = (a_pos_x[:,np.newaxis]+lidar_pointing[0,:][:,np.newaxis]*lidar_prof.range_array*1e-3).flatten()
    l_pos_y = (a_pos_y[:,np.newaxis]+lidar_pointing[1,:][:,np.newaxis]*lidar_prof.range_array*1e-3).flatten()
    l_pos_z = (a_pos_z[:,np.newaxis]-lidar_pointing[2,:][:,np.newaxis]*lidar_prof.range_array*1e-3).flatten()
    
    if hasattr(lidar_prof.profile,'mask'):
        plot_prof = (lidar_prof.profile.data.copy()).flatten()
        plot_prof_mask = lidar_prof.profile.mask.flatten()
    else:
        plot_prof = (lidar_prof.profile.copy()).flatten()
        plot_prof_mask = np.zeros(plot_prof.size,dtype=bool)
        
    irm2d = np.nonzero((l_pos_z > ylimits[1]) + (l_pos_z < ylimits[0]) + (t_data_plt < tlimits[0]) +(t_data_plt > tlimits[1])+plot_prof_mask+np.isnan(plot_prof))
    irm1d = np.nonzero((t_data_1d < tlimits[0]) +(t_data_1d > tlimits[1]))
    
    l_pos_z = np.delete(l_pos_z,irm2d)
    l_pos_y = np.delete(l_pos_y,irm2d)
    l_pos_x = np.delete(l_pos_x,irm2d)
    plot_prof = np.delete(plot_prof,irm2d)
    
    t_data_plt = np.delete(t_data_plt,irm2d)
    
    a_pos_x = np.delete(a_pos_x,irm1d)
    a_pos_y = np.delete(a_pos_y,irm1d)
    a_pos_z = np.delete(a_pos_z,irm1d)
    
    t_data_1d = np.delete(t_data_1d,irm1d)
    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    if scale == 'log':
        l_plot = ax.scatter(l_pos_x,l_pos_y,l_pos_z,c=plot_prof, \
            vmin=climits[0],vmax=climits[1], \
            s=s,linewidth=0,alpha=alpha,norm=matplotlib.colors.LogNorm(),cmap=cmap)
    else:
        l_plot = ax.scatter(l_pos_x,l_pos_y,l_pos_z,c=plot_prof, \
            vmin=climits[0],vmax=climits[1], \
            s=s,linewidth=0,alpha=alpha,cmap=cmap)
    if plot_aicraft:
        ax.scatter(a_pos_x,a_pos_y,a_pos_z,c='gray',s=2,linewidth=0)
    ax.set_xlabel('E/W [km]')
    ax.set_ylabel('N/S [km]')
    ax.set_zlabel('Altitude [km]')
    units = lidar_prof.profile_type
    if '_' in units or '^' in units and not ('$' in units):
        units = units.replace('(','{')
        units = units.replace(')','}')
        units = '$'+units+'$'
    fig.colorbar(l_plot,label=lidar_prof.label +' [' + units + ']')
    
    DateLabel = lidar_prof.StartDate.strftime("%A %B %d, %Y")
    timeLabel = (lidar_prof.StartDate+datetime.timedelta(seconds=np.int(np.nanmean(t_data_1d*3600.0)))).strftime("%H:%M:%S")
    plt.title(title_add+DateLabel + ', ' +lidar_prof.lidar + '\n' + lidar_prof.label+ '\n'+timeLabel,fontsize=title_font_size)
    
    return fig,ax,l_plot