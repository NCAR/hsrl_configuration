# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 07:59:40 2018

@author: mhayman

Runs two python scripts.
The first processes flight data and saves it as a netcdf.
The second plots the netcdf data and saves the plots.

These scripts are configured for SOCRATES processing.  The user will
be prompted to select which SOCRATES flight to process.


"""
import os

ProcessFile = os.path.abspath(__file__+'/../')+'/gv_hsrl_socrates_process_hour.py'
PlotFile = os.path.abspath(__file__+'/../')+'/gv_hsrl_socrates_plot_ncdata_5min.py'

# Process the data (and save it)
exec(open(ProcessFile).read())

# Plot the data (and save the plots)
exec(open(PlotFile).read())

