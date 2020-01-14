# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:45:55 2016

@author: mhayman
"""
import numpy as np
import scipy.special.lambertw
from scipy.special import wofz
import matplotlib.pyplot as plt
import scipy.signal
import datetime

import scipy as sp

import matplotlib
import matplotlib.dates as mdates

from mpl_toolkits.axes_grid1 import make_axes_locatable

import os

#from scipy.io import netcdf
import netCDF4 as nc4

kB = 1.3806504e-23;     # Boltzman Constant
Mair = 28.95*1.66053886e-27;  # Average mass of molecular air
N_A = 6.0221413e23; #Avagadro's number, mol^-1
mH2O = 18.018; # mass water molecule g/mol
c = 2.99792458e8  # speed of light in a vacuum

# custom exception classes

class LidarProfileDimensionError(ValueError):
    # Non-identical dimensions between inputs on an arithmatic operation
    pass

class OtherProfileNotRecognized(ValueError):
    # An arithmatic operation was attempted with an unsupported object
    pass

class ListDimensionsDontAgree(ValueError):
    # a function was expecting two input lists to have the same length
    # but this was not the case
    pass


class LidarProfile():
    """
    LidarProfile((self,profile,time,label='none',descript='none',lidar='none',
        bin0=0,shot_count=np.array([]),binwidth=0,wavelength=0,
        StartDate=datetime.datetime(1970,1,1,0))
    LidarProfile Class provides functionality of consolidating operations performed
    on single profiles and storing all instances of those operations in a single
    spot (i.e. background subtracted, etc)
    
    LidarProfile(profile,time,label=none,descript=none)
    profile - data read from raw data
    time - corresponding time axis for profile
    label - optional string label for profile.  This is used to help set the 
        operational parameters of the system
    descript - longer string that can be used to provide a detailed profile 
        description
    
    range can be calculed from the profile bin widths
    
    Methods are written in order in which they are expected to be applied.
    """
    def __init__(self,profile,time,label='none',descript='none',lidar='none',bin0=0,shot_count=None,binwidth=0,wavelength=0,StartDate=None):
        if hasattr(profile,'binwidth_ns'):
            if profile.binwidth_ns == 0:
                if (lidar == 'GV-HSRL' or lidar == 'gv-hsrl'):
                    self.binwidth_ns = 50
                    self.wavelength = 532e-9
                elif  (lidar == 'WV-DIAL' or lidar == 'wv-dial'):
                    self.binwidth_ns = 500  
                    self.wavelength = 828e-9
                elif  (lidar == 'DLB-HSRL' or lidar == 'dlb-hsrl'):
                    self.binwidth_ns = 500
                    self.wavelength = 780.24e-9
                else:
                    self.binwidth_ns = 50
                    self.wavelength = 532e-9
            else:
                self.binwidth_ns = profile.binwidth_ns
                if (lidar == 'GV-HSRL' or lidar == 'gv-hsrl'):
                    self.wavelength = 532e-9
                elif  (lidar == 'WV-DIAL' or lidar == 'wv-dial'):
                    self.wavelength = 828e-9
                elif  (lidar == 'DLB-HSRL' or lidar == 'dlb-hsrl'):
                    self.wavelength = 780.24e-9
                else:
                    self.wavelength = 532e-9
#            self.raw_profile = profile             # unprocessed profile - how do I copy a netcdf_variable?
#            self.raw_profile.data = profile.data.copy()
#            self.raw_profile.dimensions = profile.dimensions
#            self.raw_profile.binwidth_ns = self.binwidth_ns
                    
            self.profile = profile.data
        else:
            if binwidth == 0:
                if (lidar == 'GV-HSRL' or lidar == 'gv-hsrl'):
                    self.binwidth_ns = 50
                elif  (lidar == 'WV-DIAL' or lidar == 'wv-dial'):
                    self.binwidth_ns = 500  
                elif  (lidar == 'DLB-HSRL' or lidar == 'dlb-hsrl'):
                    self.binwidth_ns = 250
                else:
                    # default to GV-HSRL
                    self.binwidth_ns = 50   
            else:
                self.binwidth_ns = binwidth*1e9
            
            if wavelength == 0:
                if (lidar == 'GV-HSRL' or lidar == 'gv-hsrl'):
                    self.wavelength = 532e-9
                elif  (lidar == 'WV-DIAL' or lidar == 'wv-dial'):
                    self.wavelength = 828e-9
                elif  (lidar == 'DLB-HSRL' or lidar == 'dlb-hsrl'):
                    self.wavelength = 780.24e-9
                else:
                    # default to GV-HSRL
                    self.wavelength = 532e-9
            else:
                self.wavelength = wavelength
                    #profile_data,profile_dimensions=('time','bincount'),profile_binwidth_ns=0
            self.profile = profile.copy()         # highest processessed level of lidar profile - updates at each processing stage
            
##            self.raw_profile = profile             # unprocessed profile - how do I copy a netcdf_variable?
#            self.raw_profile.data = profile.copy()
#            self.raw_profile.dimensions = profile.shape
#            self.raw_profile.binwidth_ns = self.binwidth_ns
        if StartDate is None: 
            StartDate = datetime.datetime(1970,1,1,0)
#        self.dimensions = self.profile    # time and altitude dimensions from netcdf
#        self.binwidth_ns = profile.binwidth_ns  # binwidth - used to calculate bin ranges
        self.time = time.copy()                        # 
        self.label = label                      # label for profile (used in error messages to identify the profile)
        self.descript = descript                # description of profile
        self.lidar = lidar                      # name of lidar corresponding to this profile
        self.StartDate = StartDate              # start date of the profile
        
        self.ProcessingStatus = ['Raw Data']    # status of highest level of lidar profile - updates at each processing stage
        self.profile_variance = self.profile+1   # variance of the highest level lidar profile      
        self.mean_dt = np.nanmean(np.diff(time))   # average profile integration time
        self.range_array = (np.arange(np.shape(self.profile)[1])-bin0)*self.binwidth_ns*1e-9*c/2  # array containing the range bin data
        self.diff_geo_Refs = [];                # list containing the differential geo overlap reference sources (answers: differential to what?)
        self.profile_type = 'Photon Counts'     # measurement type of the profile (either 'Photon Counts' or 'Photon Arrival Rate [Hz]')
        self.bin0 = bin0                        # bin corresponding to range = 0        
        
        self.bg = np.array([])                  # profile background levels
        self.bg_var = np.array([])              # variance of background levels
        self.mean_dR = c*self.binwidth_ns*1e-9/2  # binwidth in range [m]
        
        self.NumProfList = np.ones(np.shape(self.profile)[0])    # number of raw profile accumlations in each profile
        
        if shot_count is None:
            if (lidar == 'GV-HSRL' or lidar == 'gv-hsrl'):
                laser_freq = 4e3
            elif  (lidar == 'WV-DIAL' or lidar == 'wv-dial'):
                laser_freq = 7e3
            elif  (lidar == 'DLB-HSRL' or lidar == 'dlb-hsrl'):
                laser_freq = 7e3
            else:
                laser_freq = 4e3
            self.shot_count = np.concatenate((np.array([self.mean_dt]),np.diff(time)))*laser_freq
        else:
            self.shot_count = shot_count
        
        # Check for dimensions or set altitude/time dimension indices?        
    
    def __repr__(self):
        return 'LidarProfile: ' + self.lidar + ' ' + self.label+' (%d, %d)'%self.profile.shape
    
    def __str__(self):
        return self.label
    
    def __add__(self,other_profile):
        """
        definition for additon between two LidarProfiles
        """
        
        if other_profile.__class__ == LidarProfile:
            sum_prof = self.copy()
    #        sum_prof.ProcessingStatus = []
            
            
            # Consistency Checks (e.g. size, time, range compatability)
            if check_piecewise_compatability(self,other_profile,'addition'):
                # Perform sum operation
                sum_prof.profile = self.profile + other_profile.profile
                sum_prof.variance_profile = self.profile_variance + other_profile.profile_variance
                sum_prof.bg = self.bg + other_profile.bg
                sum_prof.bg_var = self.bg_var + other_profile.bg_var
                
                sum_prof.descript = self.descript + ' added to ' + other_profile.descript  
                sum_prof.label =  self.label + '+' + other_profile.label  
                
                sum_prof.ProcessingStatus.extend([ 'Sum of ' + self.label + ' and ' + other_profile.label ])
                
                setmask = np.zeros(self.profile.shape,dtype=bool)
                if has_mask(self.profile): #hasattr(self.profile,'mask'):
                    setmask = self.profile.mask
                elif has_mask(other_profile.profile): #hasattr(other_profile.profile,'mask'):
                    setmask = np.logical_or(setmask,other_profile.profile.mask)
                if np.sum(setmask) > 0:
                    sum_prof.mask(setmask)
                return sum_prof
            else:
                raise LidarProfileDimensionError(other_profile)
#                return 0
        elif not hasattr(other_profile, "__len__"):
            # scalar evaluation
            sum_prof = self.copy()
            sum_prof.profile = sum_prof.profile+other_profile
            sum_prof.ProcessingStatus.extend([ 'Added %f'%other_profile + ' to '  + self.label])
            return sum_prof
            
        elif hasattr(other_profile, "__len__"):
            # array evaluation
            sum_prof = self.copy()
            try:
                sum_prof.profile = sum_prof.profile+other_profile
                sum_prof.ProcessingStatus.extend([ 'Added array' + ' to '  + self.label])
                return sum_prof
            except ValueError:
                # shape of profiles did not align
                print('Failed to add ' +self.label+' with an array' )
                print('  ' +self.label+' dimensions:')
                print(self.profile.shape)
                print('   array dimensions:')
                print(other_profile.shape)
                raise LidarProfileDimensionError(other_profile)
#                return 0
            
        else:
            raise OtherProfileNotRecognized(other_profile)
            
            
            
    def __radd__(self,other_profile):
        """
        definition for reverse additon between two LidarProfiles
        """
        
        return self.__add__(other_profile)
        
    def __sub__(self,other_profile):
        """
        definition for additon between two LidarProfiles
        """
        
        if other_profile.__class__ == LidarProfile:
            sub_prof = self.copy()
    #        sub_prof.ProcessingStatus = []
            
            
            # Consistency Checks (e.g. size, time, range compatability)
            if check_piecewise_compatability(self,other_profile,'subtraction'):
                # Perform sum operation
                sub_prof.profile = self.profile - other_profile.profile
                sub_prof.variance_profile = self.profile_variance + other_profile.profile_variance
                sub_prof.bg = self.bg - other_profile.bg
                sub_prof.bg_var = self.bg_var + other_profile.bg_var
                
                sub_prof.descript = other_profile.descript + ' subtracted from ' + self.descript
                sub_prof.label =  self.label + '-' + other_profile.label  
                
                sub_prof.ProcessingStatus.extend( [ 'Subtraction of ' + other_profile.label + ' from ' + self.label ])
                
                setmask = np.zeros(self.profile.shape,dtype=bool)
                if has_mask(self.profile): #hasattr(self.profile,'mask'):
                    setmask = self.profile.mask
                if has_mask(other_profile.profile): #hasattr(other_profile.profile,'mask'):
                    setmask = np.logical_or(setmask,other_profile.profile.mask)
                if np.sum(setmask) > 0:
                    sub_prof.mask(setmask)
                
                return sub_prof
            else:
                raise LidarProfileDimensionError(other_profile)
#                return 0
        elif not hasattr(other_profile, "__len__"):
            # scalar argument
            sub_prof = self.copy()
            sub_prof.profile = sub_prof.profile-other_profile
            sub_prof.ProcessingStatus.extend([ 'Subtracted %f'%other_profile + ' from '  + self.label])
            return sub_prof
        
        elif hasattr(other_profile, "__len__"):
            # array argument
            sub_prof = self.copy()
            try:
                sub_prof.profile = sub_prof.profile-other_profile
                sub_prof.ProcessingStatus.extend([ 'Subtracted array' + ' from '  + self.label])
                return sub_prof
            except ValueError:
                # shape of profiles did not align
                print('Failed to subtract array from ' +self.label )
                print('  ' +self.label+' dimensions:')
                print(self.profile.shape)
                print('   array dimensions:')
                print(other_profile.shape)
                raise LidarProfileDimensionError(other_profile)
#                return 0
        else:
            raise OtherProfileNotRecognized(other_profile)
#            return 0
            
    def __rsub__(self,other_profile):
        """
        definition for reverse subtraction
        """
        return (self.__neg__()).__add__(other_profile)
        
    def __neg__(self):
        """
        definition for profile negation
        """
        neg_prof = self.copy()
        neg_prof.profile = -neg_prof.profile
        neg_prof.ProcessingStatus.extend([ 'Negated profile'])
        return neg_prof
        
    def __div__(self,other_profile):
        div_prof = self.__truediv__(other_profile)
        return div_prof
        
    def __rdiv__(self,other_profile):
        div_prof = self.__rtruediv__(other_profile)
        return div_prof
        
    def __truediv__(self,other_profile):
        """
        definition for division between two LidarProfiles
        """
        
        if other_profile.__class__ == LidarProfile:
            div_prof = self.copy()
    #        div_prof.ProcessingStatus = []
    
            # Consistency Checks (e.g. size, time, range compatability)
            if check_piecewise_compatability(self,other_profile,'division'):
                # Perform division operation
                SNRnum = self.SNR()
                SNRden = other_profile.SNR()
                div_prof.profile = self.profile / other_profile.profile
                div_prof.profile_variance = div_prof.profile**2*(1.0/SNRnum**2+1.0/SNRden**2)
                div_prof.ProcessingStatus.extend([ 'Division of ' + self.label + ' by ' + other_profile.label ])
                div_prof.profile_type = self.profile_type + '/' + other_profile.profile_type  # units are not set after division
                div_prof.descript = self.descript + ' divided by ' + other_profile.descript  
                div_prof.label =  self.label + '/' + other_profile.label     
                
                setmask = np.zeros(self.profile.shape,dtype=bool)
                if has_mask(self.profile): #hasattr(self.profile,'mask'):
                    setmask = self.profile.mask
                if has_mask(other_profile.profile): #hasattr(other_profile.profile,'mask'):
                    setmask = np.logical_or(setmask,other_profile.profile.mask)
                if np.sum(setmask) > 0:
                    div_prof.mask(setmask)                
                
                return div_prof
            else:
                print('Uncompatable profile dimensions in true divide')
                raise LidarProfileDimensionError(other_profile)
#                return 0
        elif not hasattr(other_profile, "__len__"):
            # scalar argument
            div_prof = self.copy()
            div_prof.profile = self.profile/other_profile
            div_prof.profile_variance = self.profile_variance/other_profile**2
            div_prof.ProcessingStatus.extend(['Divided by %f'%other_profile])
            return div_prof
        elif hasattr(other_profile, "__len__"):
            # array argument
            div_prof = self.copy()
            try:
                div_prof.profile = self.profile/other_profile
                div_prof.profile_variance = self.profile_variance/other_profile**2
                div_prof.ProcessingStatus.extend(['Divided by array'])
                return div_prof
            except ValueError:
                # shape of profiles did not align
                print('Failed ' +self.label+'  divide by array' )
                print('  ' +self.label+' dimensions:')
                print(self.profile.shape)
                print('   array dimensions:')
                print(other_profile.shape)
                raise LidarProfileDimensionError(other_profile)
#                return 0
        else:
            print('unrecognized input class in true divide by ' + self.label)
#            raise ValueError
            raise OtherProfileNotRecognized(other_profile)
#            return 0
    
    def __rtruediv__(self,other_profile):
        """
        definition for reverse division
        """      
        if other_profile.__class__ == LidarProfile:
            div_prof = self.copy()
    #        div_prof.ProcessingStatus = []
    
            # Consistency Checks (e.g. size, time, range compatability)
            if check_piecewise_compatability(self,other_profile,'division'):
                # Perform division operation
                SNRnum = other_profile.SNR()
                SNRden = self.SNR()
                div_prof.profile = other_profile.profile / self.profile
                div_prof.profile_variance = div_prof.profile**2*(1.0/SNRnum**2+1.0/SNRden**2)
                div_prof.ProcessingStatus.extend([ 'Division of ' + other_profile.label + ' by ' + self.label ])
                div_prof.profile_type = other_profile.profile_type + '/' + self.profile_type  # units are not set after division
                div_prof.descript = other_profile.descript + ' divided by ' + self.descript  
                div_prof.label =  other_profile.label + '/' + self.label     
                
                setmask = np.zeros(self.profile.shape,dtype=bool)
                if has_mask(self.profile): #hasattr(self.profile,'mask'):
                    setmask = self.profile.mask
                if has_mask(other_profile.profile): #hasattr(other_profile.profile,'mask'):
                    setmask = np.logical_or(setmask,other_profile.profile.mask)
                if np.sum(setmask) > 0:
                    div_prof.mask(setmask)                
                
                return div_prof
            else:
                print('Uncompatable profile dimensions in true divide')
                raise LidarProfileDimensionError(other_profile)
#                return 0
        elif not hasattr(other_profile, "__len__"):
            # scalar argument
            div_prof = self.copy()
            div_prof.profile = other_profile/self.profile
            div_prof.profile_variance = (other_profile**2/self.profile**4)*self.profile_variance
            div_prof.ProcessingStatus.extend(['%f Divided by '%other_profile + self.label])
            setmask = np.zeros(self.profile.shape,dtype=bool)
            if has_mask(self.profile): #hasattr(self.profile,'mask'):
                setmask = self.profile.mask
            if np.sum(setmask) > 0:
                div_prof.mask(setmask)
            return div_prof
        elif hasattr(other_profile, "__len__"):
            # array argument
            div_prof = self.copy()
            try:
                div_prof.profile = other_profile/self.profile
                div_prof.profile_variance = (other_profile**2/self.profile**4)*self.profile_variance
                div_prof.ProcessingStatus.extend(['array Divided by ' + self.label])
                setmask = np.zeros(self.profile.shape,dtype=bool)
                if has_mask(self.profile): #hasattr(self.profile,'mask'):
                    setmask = self.profile.mask
                if has_mask(other_profile.profile): #hasattr(other_profile.profile,'mask'):
                    setmask = np.logical_or(setmask,other_profile.profile.mask)
                if np.sum(setmask) > 0:
                    div_prof.mask(setmask)
                return div_prof
            except ValueError:
                # shape of profiles did not align
                print('Failed divied array by '+self.label )
                print('  ' +self.label+' dimensions:')
                print(self.profile.shape)
                print('   array dimensions:')
                print(other_profile.shape)
                raise LidarProfileDimensionError(other_profile)
#                return 0
        else:
            print('unrecognized input class in reverse true divide with ' + self.label)
#            raise ValueError
            raise OtherProfileNotRecognized(other_profile)
            
    def __mul__(self,other_profile):
        """
        definition for multiplication between two LidarProfiles
        """
        
        if other_profile.__class__ == LidarProfile:
            mul_prof = self.copy()
    #        mul_prof.ProcessingStatus = []
    
            # Consistency Checks (e.g. size, time, range compatability)
            if check_piecewise_compatability(self,other_profile,'multiplication'):
                # Perform multiplication operation
                mul_prof.profile = self.profile * other_profile.profile
                mul_prof.profile_variance = self.profile**2*other_profile.profile_variance+self.profile_variance*other_profile.profile**2
                mul_prof.ProcessingStatus.extend( [ 'Multiplication of ' + self.label + ' and ' + other_profile.label ])
                mul_prof.profile_type = self.profile_type + '*' + other_profile.profile_type  # units are not set after division
                mul_prof.descript = self.descript + ' multiplied by ' + other_profile.descript  
                mul_prof.label =  self.label + '*' + other_profile.label   
                setmask = np.zeros(self.profile.shape,dtype=bool)
                if has_mask(self.profile): #hasattr(self.profile,'mask'):
                    setmask = self.profile.mask
                if has_mask(other_profile.profile): #hasattr(other_profile.profile,'mask'):
                    setmask = np.logical_or(setmask,other_profile.profile.mask)
                if np.sum(setmask) > 0:
                    mul_prof.mask(setmask)
                return mul_prof
            else:
                raise LidarProfileDimensionError(other_profile)
#                return 0
        elif not hasattr(other_profile, "__len__"):
            # scalar argument
            mul_prof = self.copy()
            mul_prof.profile = self.profile * other_profile
            mul_prof.profile_variance = self.profile_variance*other_profile**2
            mul_prof.ProcessingStatus.extend( [ 'Multipled by %f'%other_profile])
            return mul_prof
        elif hasattr(other_profile, "__len__"):
            # array argument
            mul_prof = self.copy()
            try:
                mul_prof.profile = self.profile * other_profile
                mul_prof.profile_variance = self.profile_variance*other_profile**2
                mul_prof.ProcessingStatus.extend( [ 'Multipled by array'])
                return mul_prof
            except ValueError:
                # shape of profiles did not align
                print('Failed array multiply with '+self.label )
                print('  ' +self.label+' dimensions:')
                print(self.profile.shape)
                print('   array dimensions:')
                print(other_profile.shape)
                raise LidarProfileDimensionError(other_profile)
#                return 0
            
        else:
            raise OtherProfileNotRecognized(other_profile)
#            return 0 
        
    def __rmul__(self,other_profile):
        """
        definition for revers multiplication
        """        
        
        return self.__mul__(other_profile)
        
    def __getitem__(self, index):
        """
        indexing into profile is time based
        returns a range array of the indexed profile
        """
        return self.profile[index,:]
    
    @property
    def shape(self):
        return self.profile.shape

    def set_bin0(self,bin0):
        # set the bin number corresponding to range = 0.  Float or integer are accepted.
        self.bin0 = bin0
        self.range_array = (np.arange(np.shape(self.profile)[1])-self.bin0)*self.binwidth_ns*1e-9*c/2
        self.ProcessingStatus.extend(['Reset Bin 0'])
        
    def nonlinear_correct(self,deadtime,laser_shot_count=0,std_deadtime=2e-9,override=False,productlog=False,newstats=False):
        # Apply nonlinear counting correction to data
        # Requires detector dead time
        # User can provide an array of the laser shots fired for each time bin
        #   otherwise it will estimate the count based on a 4kHz laser repition
        #   rate and the average integration bin duration.
    
        # override skips errors due to order in which corrections are applied
        if any('Background Subtracted' == s for s in self.ProcessingStatus):
            print ('Warning: Nonlinear correction is being applied AFTER background subtracting %s.'  %self.label)
            print ('   Nonlinear correction is generally applied BEFORE backsground subtracting.')
            print ('   Applying correction anyway.')
        
        if any('Nonlinear CountRate Correction' == s for s in self.ProcessingStatus) and not override:
            print ('Warning:  Attempted Nonlinear correction on %s after it has already been applied.' %self.label)
            print ('   Skipping this step.')
            
        else:
            if productlog:
                self.profile_count_rate(update=True)
                prodlog = scipy.special.lambertw(-deadtime*self.profile)
                self.profile=-prodlog/deadtime
                self.profile_variance = self.profile_variance*(prodlog/(deadtime*self.profile*(1+prodlog)))
                self.profile_to_photon_counts(update=True)
            elif newstats:
                bintime = self.binwidth_ns*1e-9
                self.profile = self.profile/self.shot_count[:,np.newaxis]
                self.profile = (1+self.profile*(bintime-self.profile*deadtime)/(bintime+self.profile*deadtime))*(2*bintime)**self.profile \
                    /(((1+(bintime-self.profile*deadtime)/(bintime+self.profile*deadtime)))**self.profile*(bintime+self.profile*deadtime)**(self.profile-1)*(bintime+self.profile*deadtime))
                second_moment = -4*self.profile*(self.profile**2-1)*bintime*deadtime+(bintime+self.profile*deadtime)**2*(2+self.profile**2+3*self.profile*np.sqrt(1-4*self.profile*deadtime*bintime/(bintime-self.profile*deadtime)**2)) \
                    /(bintime-self.profile*deadtime)**4
                self.profile_variance = second_moment-self.profile**2
                
                self.profile= self.profile*bintime*self.shot_count[:,np.newaxis]
                self.profile_variance= self.profile_variance*bintime**2*self.shot_count[:,np.newaxis]**2
                
                    
            else:
                count_factor = 1.0/((self.binwidth_ns*1e-9)*self.shot_count[:,np.newaxis])
                lam = self.profile*count_factor#  self.profile_count_rate()
                CorrectionFactor = 1.0/(1-deadtime*lam)
                CorrectionFactor[np.nonzero(CorrectionFactor < 0)] = np.nan;  # if the correction factor goes negative, just set it to 1.0 or NaN
            
                # does not include variance for count rate in the denominator                
#                self.profile_variance = self.profile_variance*CorrectionFactor**2+std_deadtime**2*(self.profile*lam/CorrectionFactor**2)**2
                # modified to include count rate in the denominator
                self.profile_variance = CorrectionFactor**4*(self.profile_variance+std_deadtime**2*self.profile**4*count_factor**2)               
                self.profile = self.profile*CorrectionFactor
                self.mask(np.isnan(self.profile))
                
#                self.profile_variance = self.profile_variance*CorrectionFactor**4  # power of 4 due to count dependance in denominator of the correction factor
#                self.profile_variance = std_deadtime**2*self.profile**4*(self.binwidth_ns*1e-9)**2/(self.binwidth_ns*1e-9-self.profile*deadtime)**4
            self.ProcessingStatus.extend(['Nonlinear CountRate Correction for dead time %.1f ns'%(deadtime*1e9)])
        
    def bg_subtract(self,start_index,stop_index=-1):
        # HSRL usually uses preshot data for background subtraction.  That
        # should be added to this function
        # Estimate background based on indicies passed to the method,
        # then create a background subtracted profile
        self.bg = np.nanmean(self.profile[:,start_index:stop_index],axis=1)
        # This needs to be adjusted to account for any nans in each profile (only divide by the number of non-nans)
        self.bg_var = np.nansum(self.profile_variance[:,start_index:stop_index],axis=1)/(np.shape(self.profile_variance[:,start_index:stop_index])[1])**2
        self.profile = self.profile-self.bg[:,np.newaxis]
        self.profile_variance = self.profile_variance+self.bg_var[:,np.newaxis]
        self.ProcessingStatus.extend(['Background Subtracted over [%.2f, %.2f] m'%(self.range_array[start_index],self.range_array[stop_index])])
    
    def baseline_subtract(self,baseline,baseline_var = np.array([]),tx_norm=1.0):
        """
        Subtract baseline data from lidar profile:
        baseline - full baseline profile
        baseline_var - variance of the baseline estimate
        norm - normalization factor (tx power from profile/tx power during calibration)
        """
        
        self.profile = self.profile-tx_norm*baseline
        if baseline_var.size > 0:
            self.profile_variance = self.profile_variance+tx_norm**2*baseline_var
        self.ProcessingStatus.extend(['Subtracted baseline'])
    
    def trim_to_on(self,SNRlim=12.5,ret_index=False,delete=True,plot=False):
        """
        Remove profile components where the lidar does not appear to be turned
        on.  This is done by looking at the cumulative signal in range and
        background subtracting.
        The method returns a Lidar_On boolean that reports if it believes
        the lidar was turned on at all during the supplied data set.  This can
        be used to turn off that lidar processing
        
        ret_index - if True returns the indices of the removed times
        delete - if False, it won't actually delete the times
        plot- set to True to see what the analyzed data looks like and what
            data is being filtered.
        """
        
        if any('Background Subtracted' in s for s in self.ProcessingStatus):
            totalSNR = np.abs(np.nansum(self.profile[:,1:],axis=1)/(np.sqrt(np.nansum(self.profile_variance[:,1:],axis=1)+self.bg_var)))
        else:
            tmpProf = self.copy()
            tmpProf.bg_subtract(-100)
            totalSNR = np.abs(np.nansum(tmpProf.profile[:,1:],axis=1)/(np.sqrt(np.nansum(tmpProf.profile_variance[:,1:],axis=1)+tmpProf.bg_var)))
            if plot:
                plt.figure()
                plt.plot(tmpProf.time/3600.0,totalSNR,label='SNR')
        
        totalSNR[np.nonzero(np.isnan(totalSNR))] = 0  # set NANs to zero so the get removed
        
        i_lidar_off = np.nonzero(totalSNR < SNRlim)[0]
        if plot:
            plt.plot(tmpProf.time[i_lidar_off]/3600.0,totalSNR[i_lidar_off],'r--',label='Removed points')
            plt.legend();
            
        if i_lidar_off.size == self.time.size:
            Lidar_On = False
        else:
            Lidar_On = True
        
        if i_lidar_off.size > 0 and delete:
            self.remove_time_indices(i_lidar_off) 
            self.ProcessingStatus.extend(['Removed profiles where lidar appears to be off'])
        if ret_index:
            return Lidar_On,i_lidar_off
        else:
            return Lidar_On
    
#    def cumtrap(self,axis=0,keepgrid=True):
#        """
#        cumulative integration using the trapazoidal rule
#        along the time (0) axis or the range (1) axis
#        
#        keepgrid = True - keep the current axis grid.  Set to False to use a
#            more accurate grid that will change the number of elements
#        """
#        if axis == 0:
#            # define the trapazoidal time grid 
#            time_new = self.time[1:]-np.diff(self.time)/2.0
#            
#        if axis == 1:
#            # define the trapazoidal range grid        
#            drange_new = np.diff(self.rrange_array)
##            range_new = self.range_array[1:]-drange_new/2.0
#            if keepgrid:
#                self.profile = np.zeros(self.profile.shape)
#                self.profile[:,1:] = sp.integrate.cumtrapz(self.time,x=self.range_array,axis=1)
#                self.profile_variance = np.zeros(self.profile.shape)
#                self.profile_variance[:,1:-1] = self.profile_variance[:,1:-1]*(drange_new[1:]+drange_new[:-1])[np.newaxis,:]**2/4.0
#                self.profile_variance[]
#            
#            
#            
#        else:
#            print('Unknown axis in cumulative integration call')
        
    def cumsum(self,axis=0):
        """
        Performs a cumulative sum on the profile
        along either the time (0) or range(1) axis
        """
        self.profile[np.nonzero(np.isnan(self.profile))] = 0
        self.profile = np.cumsum(self.profile,axis=axis)*self.mean_dR
        self.profile_variance[np.nonzero(np.isnan(self.profile_variance))] = 0
        self.profile_variance = np.cumsum(self.profile_variance,axis=axis)*self.mean_dR**2
        
        if axis == 0:
            self.cat_ProcessingStatus('Cumulative sum along time axis')
        if axis == 1:
            self.cat_ProcessingStatus('Cumulative sum along range axis')
        else:
            self.cat_ProcessingStatus('Cumulative sum along undefined axis')
        
    def energy_normalize(self,PulseEnergy,override=False):
        # Normalize each time bin to the corresponding transmitted energy
        # passed in as PulseEnergy
        
        if (any('Transmit Energy Normalized' == s for s in self.ProcessingStatus) and not override):
            print ('Warning:  Attempted Energy Normalization on %s after it has already been applied.' %self.label)
            print ('   Skipping this step.')
        else:
            # Only execute Energy Normalization if the profile has been background
            # subtracted.
            if any('Background Subtracted' == s for s in self.ProcessingStatus):
#                PulseEnergy = PulseEnergy/np.nanmean(PulseEnergy)      # normalize to averge pulse energy to preserve count-rate information
                # Averge Energy normalization needs to happen outside of the routine to maintain uniformity across all profiles and time data
                self.profile = self.profile/PulseEnergy[:,np.newaxis]       # normalize each vertical profile
                self.profile_variance = self.profile_variance/PulseEnergy[:,np.newaxis]**2
                self.ProcessingStatus.extend(['Transmit Energy Normalized'])
            else:
                print ('Error: Cannot energy normalize on profile %s.\n   Profile must be background subtracted first' %self.label)
        
    def geo_overlap_correct(self,geo_correction):
        # Apply a geometric overlap correction to the recorded profile
        if any('Geometric Overlap Correction' == s for s in self.ProcessingStatus):
            print ('Warning:  Attempted Geometric Overlap Correction on %s after it has already been applied.' %self.label)
            print ('   Applying correction anyway.')

        geo_correction[np.nonzero(np.isnan(geo_correction[:,1]))[0],1] = 0  # set value to zero if it is a nan
        #print ('Geometric Overlap Correction initiated for %s but processing code is not complete.' %self.label)
        geo_corr = np.interp(self.range_array,geo_correction[:,0],geo_correction[:,1])[np.newaxis,:]        
        if geo_correction.shape[1] > 2:
#            geo_corr_var = (0.05*geo_corr)**2
            geo_corr_var = np.interp(self.range_array,geo_correction[:,0],geo_correction[:,2])[np.newaxis,:]+(0.5*geo_corr)**2     
#            self.profile_variance = self.profile_variance*geo_corr**2+geo_corr_var*self.profile**2
        else:
            geo_corr_var = (0.5*geo_corr)**2
#            self.profile_variance = self.profile_variance*geo_corr**2
        self.profile_variance = self.profile_variance*geo_corr**2+geo_corr_var*self.profile**2
        self.profile = self.profile*geo_corr
        self.ProcessingStatus.extend(['Geometric Overlap Correction Applied'])
    
    def diff_geo_overlap_correct(self,diff_geo_correction,geo_reference = 'none',diff_geo_var=np.array([])):
        # Apply a differential geometric overlap correction to this profile.
        # An optional label allows the user to define the reference channel 
        # (what this channel is being compared to).
        if diff_geo_correction.ndim == 1:
            self.profile = self.profile*diff_geo_correction[np.newaxis,:]
            self.profile_variance = self.profile_variance*diff_geo_correction[np.newaxis,:]**2
            if diff_geo_var.size == self.profile.shape[1]:
                self.profile_variance+=diff_geo_var[np.newaxis,:]*self.profile**2
            
        else:
            self.profile = self.profile*diff_geo_correction
            self.profile_variance = self.profile_variance*diff_geo_correction**2
            if diff_geo_var.size == self.profile.size:
                self.profile_variance+=diff_geo_var*self.profile**2
            
        self.diff_geo_Refs.extend([geo_reference])
        self.ProcessingStatus.extend(['Differential Geometric Overlap Correction referenced to %s'%geo_reference])
    def range_correct(self):
        # apply correction for 1/r^2 loss in the profile
        self.profile = self.profile*self.range_array[np.newaxis,:]**2
        self.profile_variance = self.profile_variance*self.range_array[np.newaxis,:]**4
        self.ProcessingStatus.extend(['Applied R^2 Range Correction'])
        
    def profile_count_rate(self,laser_shot_count=0,update=False,bg=False):
        # Calculate the profile a photon arrival rate instead of photon counts.
        # if an array of laser shots per profile is not passed, the number of
        # shots are assumed based on the profile integration time and a 4kHz
        # laser rep rate.
        #
        # if update=True, the profile in the class is updated.  Otherwise
        # the profile count rates are returned.
        # if bg=True, the background is folded into the count rate calculation
    
        if self.profile_type == 'Photon Counts':
            if not type(laser_shot_count).__module__==np.__name__:
                laser_shot_count = self.shot_count
                
#            if np.size(laser_shot_count) != np.shape(self.profile)[0] or laser_shot_count <= 0:
#                if (self.lidar == 'GV-HSRL' or self.lidar == 'gv-hsrl'):
#                    laser_shot_count = 4e3*self.mean_dt
#                elif  (self.lidar == 'WV-DIAL' or self.lidar == 'wv-dial'):
#                    laser_shot_count = 7e3*2.0
#                elif  (self.lidar == 'DLB-HSRL' or self.lidar == 'dlb-hsrl'):
#                    laser_shot_count = 7e3*2.0
#                else:
#                    laser_shot_count = 4e3*self.mean_dt
#                bin_count_rate = self.profile/((self.binwidth_ns*1e-9)*laser_shot_count)
#                var_count_rate = self.profile/((self.binwidth_ns*1e-9)*laser_shot_count)**2
#            else:
#                bin_count_rate = self.profile/((self.binwidth_ns*1e-9)*laser_shot_count[:,np.newaxis])
#                var_count_rate = self.profile/((self.binwidth_ns*1e-9)*laser_shot_count[:,np.newaxis])**2
                
            if any('Background Subtracted' == s for s in self.ProcessingStatus) and bg:
                bin_count_rate = (self.profile+self.bg[:,np.newaxis])/((self.binwidth_ns*1e-9)*laser_shot_count[:,np.newaxis])
                var_count_rate = self.profile_variance/((self.binwidth_ns*1e-9)*laser_shot_count[:,np.newaxis])**2 
            else:
                bin_count_rate = self.profile*self.NumProfList[:,np.newaxis]/((self.binwidth_ns*1e-9)*laser_shot_count[:,np.newaxis])
                var_count_rate = self.profile_variance*self.NumProfList[:,np.newaxis]**2/((self.binwidth_ns*1e-9)*laser_shot_count[:,np.newaxis])**2
            if update:
                # update the processed profile, background and profile_type
                self.profile = bin_count_rate
                self.profile_variance = var_count_rate
                if self.bg.size > 0:
                    self.bg = self.bg*self.NumProfList/((self.binwidth_ns*1e-9)*laser_shot_count)
                    self.bg_var=self.bg_var*self.NumProfList**2/((self.binwidth_ns*1e-9)*laser_shot_count)**2
                self.profile_type = 'Counts/s'
            else:
                return bin_count_rate
        else:
            if not update:
                return self.profile
    def profile_to_photon_counts(self,laser_shot_count=0,update=False):
        # Calculate the photon counts corresponding to a profile currently defined in Photon Arrival Rate [Hz]        
        # if an array of laser shots per profile is not passed, the number of
        # shots are assumed based on the profile integration time and a 4kHz
        # laser rep rate.
        #
        # if update=True, the profile in the class is updated.  Otherwise
        # the profile count rates are returned.        
        
        
        if self.profile_type == 'Photon Arrival Rate [Hz]' or 'Counts/s':
            if laser_shot_count == 0:
                if (self.lidar == 'GV-HSRL' or self.lidar == 'gv-hsrl'):
                    laser_shot_count = 4e3*self.mean_dt
                elif  (self.lidar == 'WV-DIAL' or self.lidar == 'wv-dial'):
                    laser_shot_count = 7e3*self.mean_dt
                elif  (self.lidar == 'DLB-HSRL' or self.lidar == 'dlb-hsrl'):
                    laser_shot_count = 7e3*self.mean_dt
                else:
                    laser_shot_count = 4e3*self.mean_dt
                
                bin_counts = self.profile*((self.binwidth_ns*1e-9)*laser_shot_count)/self.NumProfList[:,np.newaxis]
                var_counts = self.profile_variance*((self.binwidth_ns*1e-9)*laser_shot_count)**2/self.NumProfList[:,np.newaxis]**2
            else:
                bin_counts = self.profile*((self.binwidth_ns*1e-9)*laser_shot_count[:,np.newaxis])/self.NumProfList[:,np.newaxis]
                var_counts = self.profile_variance*((self.binwidth_ns*1e-9)*laser_shot_count[:,np.newaxis])**2/self.NumProfList[:,np.newaxis]**2
            bg_counts_var = self.bg_var*((self.binwidth_ns*1e-9)*laser_shot_count)**2/self.NumProfList**2
            bg_counts = self.bg*((self.binwidth_ns*1e-9)*laser_shot_count)/self.NumProfList
            if update:
                # update the processed profile and profile_type
                self.profile = bin_counts.copy()
                self.profile_variance = var_counts.copy()
                self.bg = bg_counts.copy()
                self.bg_var = bg_counts_var.copy()
                self.profile_type = 'Photon Counts'
            else:
                return bin_counts
        else:
            if not update:
                return self.profile
        
    def time_resample(self,tedges=None,delta_t=0,i=1,t0=np.nan,update=False,remainder=False,average=True):
        # note that background data does not get adjusted in this routine
#        print ('time_resample() initiated for %s but no processing code has been written for this.' %self.label)
        if not tedges is None:
#            tedges = self.time[0]+np.arange(1,np.int((self.time[-1]-self.time[0])/delta_t)+1)*delta_t
#            if tedges.size > 0:    
#            print(self.time/3600.0)
            
            itime = np.digitize(self.time,tedges)
#            plt.figure()
#            plt.plot(itime,self.time/3600.0,'.')
#            plt.figure()
#            plt.plot((itime>0)*1.0,self.time/3600.0,'o')
#            plt.plot((itime<tedges.size)*1.0,self.time/3600.0,'^')
#            plt.plot((itime>0)*(itime<tedges.size),self.time/3600.0,'x')
#            plt.show()
            # Only run if the profiles fit in the master timing array (tedges), otherwise everything is a remainder
            if np.nansum((itime>0)*(itime<tedges.size))!=0:
#                iremain = np.nonzero(self.time > tedges[-1])[0]
                iremain = np.int(np.max(itime))  # the remainder starts at the maximum bin where data exists
#                if not remainder and iremain < tedges.size:
#                    iremain = iremain+1  # make sure to include the last bin of data if remainder isn't being used.
                iremainList = np.nonzero(self.time > tedges[iremain-1])[0]
                iprofstart = np.int(np.max(np.array([1,np.min(itime)])))
#                print('start index: %d\nstop index: %d'%(iprofstart,iremain))
                
#                profNew = np.zeros((np.size(tedges)-1,self.profile.shape[1]))
#                timeNew = 0.5*tedges[1:]+0.5*tedges[:-1] 
                profNew = np.zeros((iremain-iprofstart,self.profile.shape[1]))
                timeNew = -np.diff(tedges[iprofstart-1:iremain])*0.5+tedges[iprofstart:iremain]
#                itimeNew = np.arange(iprofstart,iremain)
                var_profNew = np.zeros(profNew.shape)
                shot_countNew = np.zeros(timeNew.shape)
                bg_New = np.zeros(timeNew.shape)
                bg_var_New = np.zeros(timeNew.shape)
                if has_mask(self.profile): #hasattr(self.profile,'mask'):
                    mask_new = np.zeros(profNew.shape)
                self.NumProfList = np.zeros(timeNew.shape)
                          
    
#                for ai in range(0,np.size(tedges)-1):
                for ai in range(np.size(timeNew)):
                    if has_mask(self.profile): #hasattr(self.profile,'mask'):
#                        NumProf = np.nansum(np.logical_not(self.profile[itime == ai+iprofstart,:].mask),axis=0)
#                        NumProfDiv = NumProf.copy()
#                        NumProfDiv[np.nonzero(NumProf==0)] = 1
                        NumProf = np.nanmax(np.nansum(np.logical_not(self.profile[itime == ai+iprofstart,:].mask),axis=0))
                        
                        if NumProf == 0:
                            NumProfDiv = 1.0
                        else:
                            NumProfDiv = np.float(NumProf)
                    else:
                        NumProf = self.profile[itime == ai+iprofstart,:].shape[0]
                        if NumProf == 0:
                            NumProfDiv = 1.0
                        else:
                            NumProfDiv = np.float(NumProf)
                    if not average:
                        NumProfDiv = 1.0

                    profNew[ai,:] = np.nansum(self.profile[itime == ai+iprofstart,:].astype(np.float),axis=0)/NumProfDiv
                    var_profNew[ai,:] = np.nansum(self.profile_variance[itime == ai+iprofstart,:].astype(np.float),axis=0)/NumProfDiv**2 
                    shot_countNew[ai] =1.0* np.nansum(self.shot_count[itime == ai+iprofstart].astype(np.float)) #/NumProfDiv
                    if len(self.bg):
                        bg_New[ai] = 1.0*np.nansum(self.bg[itime == ai+iprofstart].astype(np.float))/NumProfDiv
                        bg_var_New[ai] = 1.0*np.nansum(self.bg[itime == ai+iprofstart].astype(np.float))/NumProfDiv**2
                    if has_mask(self.profile): #hasattr(self.profile,'mask'):
#                        mask_new[ai,:] = np.nanprod(self.profile.mask[itime == ai+iprofstart,:],axis=0).astype(bool)
                        mask_new[ai,:] = np.prod(self.profile.mask[itime == ai+iprofstart,:],axis=0).astype(bool)
                    if average:
                        self.NumProfList[ai] = NumProf
                    else:
                        self.NumProfList[ai] = 1.0
                    
                if remainder:
                    RemainderProfile = self.copy();
                    RemainderProfile.profile = self.profile[iremainList,:].copy()
                    RemainderProfile.profile_variance = self.profile_variance[iremainList,:].copy()
                    RemainderProfile.time = self.time[iremainList].copy()
                    RemainderProfile.shot_count = self.shot_count[iremainList].copy()
                    if len(self.bg):
                        RemainderProfile.bg = self.bg[iremainList].copy()
                        RemainderProfile.bg_var = self.bg_var[iremainList].copy()
                        
                if update:
                    self.profile = profNew.copy()
                    self.profile_variance = var_profNew.copy()
                    self.time = timeNew.copy()
                    self.mean_dt = np.mean(np.diff(timeNew))
                    self.shot_count = shot_countNew.copy()
                    if len(self.bg):
                        self.bg = bg_New.copy()
                        self.bg_var = bg_var_New.copy()
                    if has_mask(self.profile): #hasattr(self.profile,'mask'):
                        self.mask(mask_new)
#                    self.Nprof = NumProfList
                    self.ProcessingStatus.extend(['Time Resampled to dt= %.1f s'%(self.mean_dt)])
            else:
                RemainderProfile = self.copy()
                self.profile = np.array([])
                self.profile_variance = np.array([])
                self.time = np.array([])
                self.mean_dt = np.array([])
                self.shot_count = np.array([])
                self.bg = np.array([])
                self.bg_var = np.array([])
                    
            if remainder:
                return RemainderProfile
        ############  Functions below this point need updated to include self.shot_count updates
        elif delta_t != 0:
            if np.isnan(t0):
                tedges = self.time[0]+np.arange(1,np.int((self.time[-1]-self.time[0])/delta_t)+1)*delta_t
            else:
                tedges = t0+np.arange(1,np.int((self.time[-1]-t0)/delta_t)+1)*delta_t
            if tedges.size > 0:            
                itime = np.digitize(self.time,tedges)
                
                iremain = np.nonzero(self.time > tedges[-1])[0]
                
                profNew = np.zeros((np.size(tedges)-1,self.profile.shape[1]))
                var_profNew = np.zeros(profNew.shape)
                timeNew = 0.5*tedges[1:]+0.5*tedges[:-1]  
                NumProfList = np.zeros(timeNew.shape)
    
                for ai in range(0,np.size(tedges)-1):
                    if has_mask(self.profile) and average: # hasattr(self.profile,'mask')
                        NumProf = np.nansum(np.logical_not(self.profile[itime == ai,:].mask),axis=0)
                        NumProf[np.nonzero(NumProf==0)] = 1.0
                    elif average:
                        NumProf = self.profile[itime == ai,:].shape[0]
                        if NumProf == 0:
                            NumProf = 1.0
                    else:
                        NumProf = 1.0
                    profNew[ai,:] = np.nansum(self.profile[itime == ai,:],axis=0)/NumProf
                    var_profNew[ai,:] = np.nansum(self.profile_variance[itime == ai,:],axis=0)/NumProf**2    
                    NumProfList[ai] = NumProf
                    
                if remainder:
                    RemainderProfile = self.copy();
                    RemainderProfile.profile = self.profile[iremain,:]
                    RemainderProfile.profile_variance = self.profile_variance[iremain,:]
                    RemainderProfile.time = self.time[iremain]
                    
                if update:
                    self.profile = profNew
                    self.profile_variance = var_profNew
                    self.time = timeNew
                    self.mean_dt = delta_t
#                    self.Nprof = NumProfList
                    self.ProcessingStatus.extend(['Time Resampled to dt= %.1f s'%(self.mean_dt)])
            elif remainder:
                RemainderProfile = self.copy()
                    
            if remainder:
                return RemainderProfile
            
#            i = np.int(np.round(delta_t/self.mean_dt))
        elif i > 1.0:
            i = np.int(i)
            profNew = np.zeros((np.floor(self.profile.shape[0]/i),self.profile.shape[1]))
            var_profNew = np.zeros(profNew.shape)
            timeNew = np.zeros(profNew.shape[0])
            
            # Calculate remainders on the end of the profile
            if remainder:
                RemainderProfile = self.copy();
                RemainderProfile.profile = self.profile[i*np.floor(self.profile.shape[0]/i):,:]
                RemainderProfile.profile_variance = self.profile_variance[i*np.floor(self.profile.shape[0]/i):,:]
                RemainderProfile.time = self.time[i*np.floor(self.profile.shape[0]/i):]
            
            
            
            for ai in range(i):
                if average:
                    NumProf = i
                else:
                    NumProf = 1
                profNew = profNew + self.profile[ai:(i*profNew.shape[0]):i,:]/NumProf
                var_profNew = var_profNew + self.profile_variance[ai:(i*profNew.shape[0]):i,:]/NumProf**2
                timeNew = timeNew + self.time[ai:(i*profNew.shape[0]):i]*1.0/i
        
#            for ai in range(profNew.shape[0]):
#                profNew[ai,:] = np.sum(self.profile[(ai*i):(i*(ai+1)),:],axis=0)
#                var_profNew[ai,:] = np.sum(self.profile_variance[(ai*i):(i*(ai+1)),:],axis=0)
            if update:
                self.profile = profNew
                self.profile_variance = var_profNew
                self.time = timeNew
                self.mean_dt = self.mean_dt*i
#                    self.Nprof = np.ones(time.shape)*i
                self.ProcessingStatus.extend(['Time Resampled to dt= %.1f s'%(self.mean_dt)])
            if remainder:
                return RemainderProfile
                
    def range_resample(self,R0=0,delta_R=0,update=False):
        # resample or integrate profile in range
        # R0 - center of first range bin
        # delta_R - range bin resolution
        # update - boolean to update the profile or return the profile
        #   True - don't return the profile, update the profile with the new range integrated one
        #   False - return the range integrated profile and don't update this one.
#        print ('range_resample() initiated for %s but no processing code has been written for this.' %self.label)
#        if delta_R <= 0:
#            new_range_profile = np.nansum(self.profile,axis=1)
#            new_range_variance = np.nansum(self.profile_variance,axis=1)
#            new_range_binwidth_ns = self.binwidth_ns*np.shape(self.profile)[1]
        if delta_R > self.mean_dR:
            i = np.int(np.round(delta_R/self.mean_dR))
            if i > 1.0:
                profNew = np.zeros((self.profile.shape[0],np.int(np.floor(self.profile.shape[1]/i))))
                var_profNew = np.zeros(profNew.shape)
                rangeNew = np.zeros(profNew.shape[1])
                for ai in range(i):
                    profNew = profNew + self.profile[:,ai:(i*profNew.shape[1]):i]
                    var_profNew = var_profNew + self.profile_variance[:,ai:(i*profNew.shape[1]):i]
                    rangeNew = rangeNew + self.range_array[ai:(i*profNew.shape[1]):i]*1.0/i
            # Probably need to LP filter and resample
            
#            new_range_profile = self.profile
#            do other stuff
        
                if update:
                    self.profile = profNew
                    self.profile_variance = var_profNew
                    self.range_array = rangeNew
                    self.binwidth_ns = self.binwidth_ns*i
                    self.mean_dR = c*self.binwidth_ns*1e-9/2
#                    self.ProcessingStatus.extend(['Range Resample to dR = %.1f m'%(self.mean_dR)])
                    self.cat_ProcessingStatus('Range Resample to dR = %.1f m'%(self.mean_dR))
                else:
                    return profNew  # needs to be updated to return a LidarProfile type
    def range_interp(self,range_new,update=True):
        """
        Force the profile onto a new range grid using linear interpolation
        range_new - new range grid in meters
        if update = True, updates this profile
        if update = False, returns a new interpolated profile
        """
        
        drange = np.mean(np.diff(range_new))
        if update:
            if has_mask(self.profile): #hasattr(self.profile,'mask'):
                profile_new = np.zeros((self.time.size,range_new.size))
                profile_var_new = np.zeros((self.time.size,range_new.size))
                profile_mask = np.zeros((self.time.size,range_new.size),dtype=bool)
                for itime in range(self.time.size):
                    profile_new[itime,:] =  np.interp(range_new,self.range_array,self.profile.data[itime,:])
                    profile_mask[itime,:] = np.interp(range_new,self.range_array,self.profile.mask.astype(np.float)[itime,:]) > 0.05
                    profile_var_new[itime,:] =  np.interp(range_new,self.range_array,self.profile_variance[itime,:])
                self.profile = np.ma.array(profile_new,mask=profile_mask)
                self.profile_variance = np.ma.array(profile_var_new,mask=profile_mask)
            else:
                profile_new = np.zeros((self.time.size,range_new.size))
                profile_var_new = np.zeros((self.time.size,range_new.size))
                for itime in range(self.time.size):
                    profile_new[itime,:] =  np.interp(range_new,self.range_array,self.profile[itime,:])
                    profile_var_new[itime,:] =  np.interp(range_new,self.range_array,self.profile_variance[itime,:])
                self.profile = profile_new.copy()
                self.profile_variance = profile_var_new.copy()

            self.binwidth_ns = drange/c*2e9
            self.mean_dR = drange
            self.range_array = range_new
            self.cat_ProcessingStatus('Interpolated onto new range grid')
        
        else:
            NewProf = self.copy()
            profile_new = np.zeros((self.time.size,range_new.size))
            profile_var_new = np.zeros((self.time.size,range_new.size))
            for itime in range(self.time.size):
                profile_new[itime,:] =  np.interp(range_new,self.range_array,self.profile[itime,:])
                profile_var_new[itime,:] =  np.interp(range_new,self.range_array,self.profile_variance[itime,:])
            NewProf.profile = profile_new.copy()
            NewProf.profile_variance = profile_var_new.copy()
            NewProf.binwidth_ns = drange/c*2e9
            NewProf.mean_dR = drange
            NewProf.range_array = range_new
            NewProf.cat_ProcessingStatus('Interpolated onto new range grid')
            return NewProf
            
        
    def mask(self,new_mask0):
        """
        adds new_mask to the current lidar profile mask
        """
        new_mask = new_mask0.astype(bool).copy()
        if new_mask.shape == self.profile.shape:
            new_mask[np.nonzero(np.isnan(new_mask))] = 0
            if has_mask(self.profile): #hasattr(self.profile,'mask'):
                new_mask = np.logical_or(new_mask.astype(np.bool),self.profile.mask)
                self.profile.mask = new_mask
            else:
                self.profile = np.ma.array(self.profile,mask=new_mask.astype(np.bool))
            self.cat_ProcessingStatus('Applied Pointwise Mask')
        else:
            print("Warning: mask applied to %s does not have the same dimensions as the profile"%self.label)
            print("  profile dimensions: [%d,%d]"%(self.profile.shape[0],self.profile.shape[1]))
            if new_mask.ndim == 2:
                print("  mask dimensions: [%d,%d]"%(new_mask.shape[0],new_mask.shape[1]))
            elif new_mask.ndim == 1:
                print("  mask dimensions: [%d,]"%(new_mask.shape[0]))
            else:
                print("  mask has %d dimensions"%(new_mask.ndim))
            self.cat_ProcessingStatus('Failed to apply mask due to different dimensions')
            
    def mask_range(self,type_str,limits):
        range_mask_defined = False
        if type_str == 'index':
            range_mask = np.zeros(self.profile.shape)
            range_mask[:,limits] = 1 
            range_mask_defined = True
            StatusString = 'Range Mask Applied based on supplied range indices between %d and %d'%(np.nanmin(limits),np.nanmax(limits))
        elif type_str == 'less than' or '<':
            range_mask = np.zeros(self.profile.shape)
            range_mask[:,np.nonzero(self.range_array < limits)[0]] = 1
            range_mask_defined = True
            StatusString = 'Range Mask Applied on range < %.1f m'%(limits)
        elif type_str == 'less than or equal' or '<=':
            range_mask = np.zeros(self.profile.shape)
            range_mask[:,np.nonzero(self.range_array <= limits)[0]] = 1
            range_mask_defined = True
            StatusString = 'Range Mask Applied on range <= %.1f m'%(limits)
        else:
            print('Warning: Unrecongized mask type in %s call to range_mask.  No action will be taken.' %self.label)
            
        if range_mask_defined:
            self.profile = np.ma.array(self.profile,mask=range_mask)
            self.profile_variance = np.ma.array(self.profile_variance,mask=range_mask)
            self.cat_ProcessingStatus(StatusString)
            
    def remove_mask(self):
        """
        Clears the mask on the profile so no data is masked
        """
        if has_mask(self.profile): #hasattr(self.profile,'mask'):
            newmask  = np.zeros(self.profile.mask.shape,dtype=bool)
            self.profile.mask = newmask
        if has_mask(self.profile_variance): #hasattr(self.profile_variance,'mask'):
            self.profile_variance.mask = newmask
        self.cat_ProcessingStatus('Removed mask')
        
    def regrid_data(self,timeData,rangeData):
        print ('regrid_data() initiated for %s but no processing code has been written for this.' %self.label)
      
    def slice_time_index(self,time_lim=[0,10000]):
        if time_lim[1] >= np.size(self.time):
            time_lim[1] = np.size(self.time)-1
#            print('Warning: requested upper time slice exceeds time dimensions of the profile:')
#            print('Time dimension: %d'%np.size(self.time))
#            print('Requested upper index: %d'%time_lim[1])
        if time_lim[0] >= np.size(self.time):
            time_lim[0] = np.size(self.time)
#            print('Warning: requested lower time slice exceeds time dimensions of the profile:')
#            print('Time dimension: %d'%np.size(self.time))
#            print('Requested lower index: %d'%time_lim[1])
        keep_index = np.arange(time_lim[0],time_lim[1]+1)
        lower_remainder_index = np.arange(time_lim[0])
        upper_remainder_index = np.arange(time_lim[1]+1,self.profile.shape[0])
        
        lower_remainder = self.profile[lower_remainder_index,:]
        upper_remainder = self.profile[upper_remainder_index,:]
        
        if self.bg.size == self.time.size:
            self.bg = self.bg[keep_index]
            self.bg_var = self.bg_var[keep_index]
        self.profile = self.profile[keep_index,:]
        self.time = self.time[keep_index]
        self.profile_variance = self.profile_variance[keep_index,:]
        self.shot_count = self.shot_count[keep_index]
        self.NumProfList = self.NumProfList[keep_index]
        
        self.cat_ProcessingStatus('Grab Time Slice from index range [%d , %d]'%(time_lim[0],time_lim[1]))
        
        return lower_remainder,upper_remainder,lower_remainder_index,upper_remainder_index 
        
        # Grab a slice of time in data and return the remainder(s)
        # Should be useful for processing multiple netcdf files to avoid discontinuities at file edges.
#        print ('slice_time() initiated for %s but no processing code has been written for this.' %self.label)
        #return end_remainder,start_remainder      
    def remove_time(self,time_array):
        """
        deletes the times supplied in time_array from the profile
        """
#        # find the time indices that match the supplied list
#        itimes = np.nonzero(np.sum(self.time[:,np.newaxis]==time_array[np.newaxis,:],axis=1))[0]
#        
#        # if using a masked array, the delete function corrupts the mask.
#        # this will update and reset it after we are done.
#        set_mask = False
#        if hasattr(self.profile,'mask'):
#            set_mask = True
#            new_mask = self.profile.mask.copy()
#            new_mask = np.delete(new_mask,itimes,axis=0)
#            
#        self.time = np.delete(self.time,itimes)
#        self.shot_count = np.delete(self.shot_count,itimes)
#        self.NumProfList = np.delete(self.NumProfList,itimes)
#        self.bg = np.delete(self.bg,itimes)
#        self.bg_var = np.delete(self.bg_var,itimes)
#        self.profile = np.delete(self.profile,itimes,axis=0)
#        self.profile_variance = np.delete(self.profile_variance,itimes,axis=0)
#        if set_mask:
#            self.mask(new_mask)
#        self.cat_ProcessingStatus('Removed specified times')
        
        itimes = np.nonzero(np.sum(self.time[:,np.newaxis]==time_array[np.newaxis,:],axis=1)==0)[0]
        
        # if using a masked array, the delete function corrupts the mask.
        # this will update and reset it after we are done.
#        set_mask = False
#        if hasattr(self.profile,'mask'):
#            set_mask = True
#            new_mask = self.profile.mask.copy()
#            new_mask = np.delete(new_mask,itimes,axis=0)
        
        if self.bg.size == self.time.size:
            self.bg = self.bg[itimes]
            self.bg_var = self.bg_var[itimes]
            
        self.time = self.time[itimes]
        self.shot_count = self.shot_count[itimes]
        self.NumProfList = self.NumProfList[itimes]
        
        self.profile = self.profile[itimes,:]
        self.profile_variance = self.profile_variance[itimes,:]
#        if set_mask:
#            self.mask(new_mask)
        self.cat_ProcessingStatus('Removed specified times')
    def remove_time_indices(self,itimes,label=''):
        """
        deletes the times indices supplied in time_array from the profile
        """
        # if using a masked array, the delete function corrupts the mask.
        # this will update and reset it after we are done.
        set_mask = False
        if has_mask(self.profile): #hasattr(self.profile,'mask'):
            set_mask = True
            new_mask = self.profile.mask.copy()
            new_mask = np.delete(new_mask,itimes,axis=0)
            
        self.time = np.delete(self.time,itimes)
        self.shot_count = np.delete(self.shot_count,itimes)
        self.NumProfList = np.delete(self.NumProfList,itimes)
        self.bg = np.delete(self.bg,itimes)
        self.bg_var = np.delete(self.bg_var,itimes)
        self.profile = np.delete(self.profile,itimes,axis=0)
        self.profile_variance = np.delete(self.profile_variance,itimes,axis=0)
        if set_mask:
            self.mask(new_mask)
        if len(label) > 0:
            self.cat_ProcessingStatus('Removed specified times: '+label)
        else:
            self.cat_ProcessingStatus('Removed specified times')
            
    def slice_time(self,time_range):
        itime1 = np.argmin(np.abs(time_range[0]-self.time))
        itime2 = np.argmin(np.abs(time_range[1]-self.time))
        time_slice = self.slice_time_index(time_lim=[itime1,itime2])
        self.cat_ProcessingStatus('Grab Time Slice %.1f - %.1f hours'%(time_range[0]/3600,time_range[1]/3600))
        return time_slice
    def slice_range(self,range_lim=[0,1e6]):
        # Slices profile to range_lim[start,stop] (set in m)
        # If range_lim is not supplied, negative ranges will be removed.
        
        keep_index = np.nonzero(np.logical_and(self.range_array >= range_lim[0], self.range_array <= range_lim[1]))[0]
        lower_remainder_index = np.nonzero(self.range_array < range_lim[0])
        upper_remainder_index = np.nonzero(self.range_array > range_lim[1])
        
        lower_remainder = self.profile[:,lower_remainder_index]
        range_lower_remainder = self.range_array[lower_remainder_index]
        upper_remainder = self.profile[:,upper_remainder_index]        
        range_upper_remainder = self.range_array[upper_remainder_index]        
        
        self.profile = self.profile[:,keep_index]
        self.profile_variance = self.profile_variance[:,keep_index]
        self.range_array = self.range_array[keep_index]
        
        self.cat_ProcessingStatus('Grab Range Slice %.1f - %.1f m'%(range_lim[0],range_lim[1]))
        
        # returned profiles should still by LidarProfile type - needs update
        return lower_remainder,upper_remainder,range_lower_remainder,range_upper_remainder
    def remove_range_indices(self,irange):
        self.range_array = self.range_array[irange]
        self.profile = self.profile[:,irange]
        self.profile_variance = self.profile_variance[:,irange]
#        if set_mask:
#            self.mask(new_mask)
        self.cat_ProcessingStatus('Removed specified range indices')
        
    def remove_range(self,range_array):
        """
        deletes the ranges supplied in range_array from the profile
        """
#        # find the time indices that match the supplied list
#        irange = np.nonzero(np.sum(self.range_array[:,np.newaxis]==range_array[np.newaxis,:],axis=0))[0]
#        
#        # if using a masked array, the delete function corrupts the mask.
#        # this will update and reset it after we are done.
#        set_mask = False
#        if hasattr(self.profile,'mask'):
#            set_mask = True
#            new_mask = self.profile.mask.copy()
#            new_mask = np.delete(new_mask,irange,axis=0)
#            
#        self.range_array = np.delete(self.range_array,irange)
#        self.profile = np.delete(self.profile,irange,axis=0)
#        self.profile_variance = np.delete(self.profile_variance,irange,axis=0)
#        if set_mask:
#            self.mask(new_mask)
#        self.cat_ProcessingStatus('Removed specified ranges')
        
        # find the time indices that match the supplied list
        irange = np.nonzero(np.sum(self.range_array[:,np.newaxis]==range_array[np.newaxis,:],axis=1)==0)[0]
        
#        # if using a masked array, the delete function corrupts the mask.
#        # this will update and reset it after we are done.
#        set_mask = False
#        if hasattr(self.profile,'mask'):
#            set_mask = True
#            new_mask = self.profile.mask.copy()
#            new_mask = np.delete(new_mask,irange,axis=0)
            
        self.range_array = self.range_array[irange]
        self.profile = self.profile[:,irange]
        self.profile_variance = self.profile_variance[:,irange]
#        if set_mask:
#            self.mask(new_mask)
        self.cat_ProcessingStatus('Removed specified ranges')
        
    def slice_range_index(self,range_lim=[0,1e6]):
        # Slices profile to range_lim[start,stop] (set in m)
        # If range_lim is not supplied, negative ranges will be removed.
        
        if range_lim[1] > self.range_array.size:
            range_lim[1] = self.range_array.size
        range_indicies = np.arange(self.range_array.size)
        keep_index = range_indicies[range_lim[0]:range_lim[1]]
#        lower_remainder_index = np.nonzero(self.range_array < range_lim[0])
#        upper_remainder_index = np.nonzero(self.range_array > range_lim[1])
#        
#        lower_remainder = self.profile[:,lower_remainder_index]
#        range_lower_remainder = self.range_array[lower_remainder_index]
#        upper_remainder = self.profile[:,upper_remainder_index]        
#        range_upper_remainder = self.range_array[upper_remainder_index]        
        
        self.profile = self.profile[:,keep_index]
        self.profile_variance = self.profile_variance[:,keep_index]
        self.range_array = self.range_array[keep_index]
            
        self.cat_ProcessingStatus('Grab Range Index Slice %.1f - %.1f indices'%(range_lim[0],range_lim[1]))    
    def copy(self,range_index=[0,0],time_index=[0,0],label='none',descript='none'):
        # needed to copy the profile and perform alternate manipulations
        # Code needs significant work!  Not sure how to best do this.
#        print ('copy() initiated for %s but no processing code has been written for this.' %self.label)
        if label == 'none':
            label = self.label
        if descript == 'none':
            descript = self.descript
        
        tmp_raw_profile = self.profile.copy()
        tmp_time = self.time.copy()
        tmp_range= self.range_array.copy()
        
        # Slice range according to requested indices range_index
        if range_index!= [0,0]:
            # check for legitimate indices before slicing
            if range_index[0] < -np.shape(tmp_raw_profile)[1]:
                print ('Warning: range_index out of bounds on LidarProfile.copy()')
                range_index[0] = 0;
            if range_index[0] > np.shape(tmp_raw_profile)[1]-1:
                print ('Warning: range_index out of bounds on LidarProfile.copy()')
                range_index[0] = np.shape(tmp_raw_profile)[1]-1;
            if range_index[1] < -np.shape(tmp_raw_profile)[1]:
                print ('Warning: range_index out of bounds on LidarProfile.copy()')
                range_index[1] = 0;
            if range_index[1] > np.shape(tmp_raw_profile)[1]-1:
                print ('Warning: range_index out of bounds on LidarProfile.copy()')
                range_index[1] = np.shape(tmp_raw_profile)[1]-1;
            
            # slice in range
            tmp_raw_profile = tmp_raw_profile[:,range_index[0]:range_index[1]]
            tmp_range = tmp_range[range_index[0]:range_index[1]]
            if np.size(tmp_range) == 0:
                print ('Warning: range_index on LidarProfile.copy() produces an empty array')
        
        # Slice rime according to requested indices time_index  
        if time_index != [0,0]:
            # check to make sure the indices are in the array bounds
            if time_index[0] < -np.shape(tmp_raw_profile)[0]:
                print ('Warning: time_index out of bounds on LidarProfile.copy()')
                time_index[0] = 0;
            if time_index[0] > np.shape(tmp_raw_profile)[0]-1:
                print ('Warning: time_index out of bounds on LidarProfile.copy()')
                time_index[0] = np.shape(tmp_raw_profile)[0]-1;
            if time_index[1] < -np.shape(tmp_raw_profile)[0]:
                print ('Warning: time_index out of bounds on LidarProfile.copy()')
                time_index[1] = 0;
            if time_index[1] > np.shape(tmp_raw_profile)[0]-1:
                print ('Warning: time_index out of bounds on LidarProfile.copy()')
                time_index[1] = np.shape(tmp_raw_profile)[0]-1;
            
            # slice time
            tmp_raw_profile = tmp_raw_profile[time_index[0]:time_index[1],:]
            tmp_time = tmp_time[time_index[0]:time_index[1]]
            if np.size(tmp_time) == 0:
                print ('Warning: time_index on LidarProfile.copy() produces an empty array')
        
        # Create the new profile
        NewProfile = LidarProfile(tmp_raw_profile,tmp_time,label=label,descript=descript)
        
        # Copy over everything else that was not transfered in initialization
        NewProfile.profile = self.profile.copy()        # highest processessed level of lidar profile - updates at each processing stage
        NewProfile.ProcessingStatus = list(self.ProcessingStatus)     # status of highest level of lidar profile - updates at each processing stage
        NewProfile.profile_variance = self.profile_variance.copy()   # variance of the highest level lidar profile      
        NewProfile.mean_dt = self.mean_dt                       # average profile integration time
        NewProfile.range_array = tmp_range.copy()                     # array containing the range bin data
        NewProfile.diff_geo_Refs = self.diff_geo_Refs           # list containing the differential geo overlap reference sources (answers: differential to what?)
        NewProfile.profile_type =  self.profile_type            # measurement type of the profile (either 'Photon Counts' or 'Photon Arrival Rate [Hz]')
        NewProfile.bin0 = self.bin0                             # bin corresponding to range = 0        
        NewProfile.lidar = self.lidar
        NewProfile.wavelength = self.wavelength
        
        NewProfile.bg = self.bg.copy()                 # profile background levels
        NewProfile.bg_var = self.bg_var.copy()
        NewProfile.mean_dR = self.mean_dR       # binwidth in range [m]
        
        NewProfile.time = self.time.copy()
        NewProfile.StartDate = self.StartDate
        NewProfile.binwidth_ns = self.binwidth_ns
        NewProfile.NumProfList = self.NumProfList.copy()
        NewProfile.shot_count = self.shot_count.copy()
        
        NewProfile.ProcessingStatus.extend(['Copy of previous profile: %s'%self.label])
        
        return NewProfile
        
        
    def cat_time(self,NewProfile,front=True):
        # concatenate the added profile to the end of this profile and store it
#        print ('cat_time() initiated for %s but no processing code has been written for this.' %self.label)
#        self.ProcessingStatus.extend(['Concatenate Time Data'])        
        
        ### Add checks for consistency - e.g. lidar type, photon counts vs arrival rate, etc
        if NewProfile.profile.size != 0:
            self.ProcessingStatus.extend(['Concatenate Time Data']) 
            
            use_mask = False
            if has_mask(self.profile): #hasattr(self.profile,'mask'):
                current_mask = self.profile.mask.copy()
                use_mask = True
            else:
                current_mask = np.zeros(self.profile.shape,dtype=bool)
            if has_mask(NewProfile.profile): #hasattr(NewProfile.profile,'mask'):
                new_mask = NewProfile.profile.mask.copy()
                use_mask = True
            else:
                new_mask = np.zeros(NewProfile.profile.shape,dtype=bool)
                
                
            if front:
                if self.StartDate > datetime.datetime(year=2005,month=1,day=1) and NewProfile.StartDate > datetime.datetime(year=2005,month=1,day=1):
                    time_offset = (self.StartDate-NewProfile.StartDate).total_seconds()
                
                # Concatenate NewProfile in the front
                self.time = np.concatenate((NewProfile.time,self.time+time_offset))
                self.profile = np.vstack((NewProfile.profile,self.profile))
                self.profile_variance = np.vstack((NewProfile.profile_variance,self.profile_variance))
                self.bg = np.concatenate((NewProfile.bg,self.bg))
                self.bg_var = np.concatenate((NewProfile.bg_var,self.bg_var))
                self.shot_count = np.concatenate((NewProfile.shot_count,self.shot_count))
                self.NumProfList = np.concatenate((NewProfile.NumProfList,self.NumProfList))
                self.StartDate = NewProfile.StartDate  # use start date of the first profile
                if use_mask:
                    set_mask = np.vstack((new_mask,current_mask))
                    self.mask(set_mask)
            else:
                if self.StartDate > datetime.datetime(year=2005,month=1,day=1) and NewProfile.StartDate > datetime.datetime(year=2005,month=1,day=1):
                    time_offset = (NewProfile.StartDate-self.StartDate).total_seconds()
                
                # Concatenate NewProfile in the back
                self.time = np.concatenate((self.time,NewProfile.time+time_offset))
                self.profile = np.vstack((self.profile,NewProfile.profile))
                self.profile_variance = np.vstack((self.profile_variance,NewProfile.profile_variance))
                self.bg = np.concatenate((self.bg,NewProfile.bg))
                self.bg_var = np.concatenate((self.bg_var,NewProfile.bg_var))
                self.shot_count = np.concatenate((self.shot_count,NewProfile.shot_count))
                self.NumProfList = np.concatenate((self.NumProfList,NewProfile.NumProfList))
                if use_mask:
                    set_mask = np.vstack((current_mask,new_mask))
                    self.mask(set_mask)
        else:
            self.ProcessingStatus.extend(['Concatenate Time Data Skipped (Empty Profile Supplied) between %s and %s'%(self.label,NewProfile.label)]) 
        
    def cat_range(self,NewProfile,bottom=True):
        if NewProfile.time.size == self.time.size:
            if all(NewProfile.time == self.time):
                use_mask = False
                if has_mask(self.profile): #hasattr(self.profile,'mask'):
                    current_mask = self.profile.mask.copy()
                    use_mask = True
                else:
                    current_mask = np.zeros(self.profile.shape)
                if has_mask(NewProfile.profile): #hasattr(NewProfile.profile,'mask'):
                    new_mask = NewProfile.profile.mask.copy()
                    use_mask = True
                else:
                    new_mask = np.zeros(NewProfile.profile.shape)
                
                if bottom:
                    self.range_array = np.concatenate((NewProfile.range_array,self.range_array))
                    self.profile = np.hstack((NewProfile.profile,self.profile))
                    self.profile_variance = np.hstack((NewProfile.profile_variance,self.profile_variance))
                    if use_mask:
                        set_mask = np.hstack((new_mask,current_mask))
                        self.mask(set_mask)
#                        self.profile.mask = set_mask
    #                    self.profile = np.ma.array(self.profile,mask=set_mask)
    #                    self.profile = np.ma.array(self.profile_variance,mask=set_mask)
                    self.ProcessingStatus.extend(['Concatenate Range Data from %s to profile bottom'%NewProfile.label])
                else:
                    self.range_array = np.concatenate((self.range_array,NewProfile.range_array))
                    self.profile = np.hstack((self.profile,NewProfile.profile))
                    self.profile_variance = np.hstack((self.profile_variance,NewProfile.profile_variance))
                    if use_mask:
                        set_mask = np.hstack((current_mask,new_mask))
                        self.mask(set_mask)
#                        self.profile.mask = set_mask
    #                    set_mask = np.hstack((current_mask,new_mask))
    #                    self.profile = np.ma.array(self.profile,mask=set_mask)
    #                    self.profile = np.ma.array(self.profile_variance,mask=set_mask)
                    self.ProcessingStatus.extend(['Concatenate Range Data to top'])
                    
            else:
                print('Warning: Concatenate Range Data Skipped (time axes did not match)')
                print('     between %s and %s'%(self.label,NewProfile.label))
                self.ProcessingStatus.extend(['Concatenate Range Data Skipped (time axes did not match) between %s and %s'%(self.label,NewProfile.label)])
        else:
            print('Warning: Concatenate Range Data Skipped (time sizes did not match)')
            self.ProcessingStatus.extend(['Concatenate Range Data Skipped (time sizes did not match) between \n   %s: %d \n   and\n   %s: %d'%(self.label,self.time.size,NewProfile.label,NewProfile.time.size)])
    def cat_ProcessingStatus(self,ProcessingUpdate):
        self.ProcessingStatus.extend([ProcessingUpdate]) 
    
    def time_integrate(self,avg=False):
        if avg and self.profile_type.lower()=='photon counts':
#            num = np.shape(self.profile)[0]-np.sum(np.isnan(self.profile),axis=0)
#            self.profile = np.nanmean(self.profile,axis=0)[np.newaxis,:]
            self.profile = np.nansum(self.profile*self.NumProfList[:,np.newaxis]/np.nansum(self.NumProfList),axis=0)[np.newaxis,:]
#            self.profile_variance = np.nansum(self.profile_variance,axis=0)[np.newaxis,:]/(num[np.newaxis,:]**2)
            self.profile_variance = np.nansum(self.profile_variance*(self.NumProfList[:,np.newaxis]/np.nansum(self.NumProfList))**2,axis=0)[np.newaxis,:]
            self.bg = np.array([np.nanmean(self.bg)])
            self.shot_count = np.array([np.nanmean(self.shot_count)])
            self.NumProfList = np.array([np.nansum(self.NumProfList)])
            update_string = 'Integrated In Time (averaged)'
        elif avg:
            self.profile = np.nanmean(self.profile,axis=0)[np.newaxis,:]
            self.profile_variance = np.nansum(self.profile_variance,axis=0)[np.newaxis,:]/self.profile_variance.shape[0]**2
            self.bg = np.array([np.nanmean(self.bg)])
            self.shot_count = np.array([np.nanmean(self.shot_count)])
            self.NumProfList = np.array([np.nansum(self.NumProfList)])
            update_string = 'Integrated In Time (averaged)'
        else:
            self.profile = np.nansum(self.profile,axis=0)[np.newaxis,:]
            self.profile_variance = np.nansum(self.profile_variance,axis=0)[np.newaxis,:]
            self.bg = np.array([np.nansum(self.bg)])
            self.shot_count = np.array([np.nansum(self.shot_count)])
            self.NumProfList = np.array([np.nanmean(self.NumProfList)])
            update_string = 'Integrated In Time (summed)'
        self.mean_dt = (self.time[-1]-self.time[0])+self.mean_dt
        self.time = np.array([np.nanmean(self.time)])
        ## needs update to variance terms
        self.ProcessingStatus.extend([update_string])
    
    def gain_scale(self,gain,gain_var = 0.0):
        """
        Scale the profile by a factor gain.
        gain_var can be used to account for possible variance in the 
            scale factor
        """
        
        self.profile = self.profile*gain
        if gain_var.__class__ == np.ndarray:
            self.profile_variance = self.profile_variance*gain**2 + self.profile**2*gain_var
            self.bg_var = self.bg_var*gain.flatten()**2+self.bg*gain_var.flatten()
            
        elif gain_var != 0:
            self.profile_variance = self.profile_variance*gain**2 + self.profile**2*gain_var
            self.bg_var = self.bg_var*gain**2+self.bg*gain_var
            
        else:
            self.profile_variance = self.profile_variance*gain**2
            self.bg_var = self.bg_var*gain**2
            

        if gain.__class__ == np.ndarray:
            self.bg = self.bg.flatten()*gain.flatten()
            self.ProcessingStatus.extend(['Profile Rescaled by array betwen %e and %e'%(np.nanmin(gain),np.nanmax(gain))])
        else:
            self.bg = self.bg*gain
            self.ProcessingStatus.extend(['Profile Rescaled by %e'%gain])
        print(self.bg.shape)
        
    
    def get_conv_kernel(self,sigt,sigz,norm=True):
        """
        Generates a Gaussian convolution kernel for
        standard deviations sigt and sigz in units of grid points.
        This should probably be moved to be an independent function
        
        Replaced by the function get_conv_kernel which is now called
        directly from this method
        """        
        z,t,kconv = get_conv_kernel(sigt,sigz,norm=True)
        
        
        return z,t,kconv
        
    def conv(self,sigt,sigz,keep_mask=True):
        """
        Convolve a Gaussian with std sigt in the time dimension (in points)
        and sigz in the altitude dimension (also in points)
        """
        z,t,kconv = get_conv_kernel(sigt,sigz,norm=True)
        
        # Replaced code with get_conv_kernel
#        t = np.arange(-np.round(4*sigt),np.round(4*sigt))      
#        z = np.arange(-np.round(4*sigz),np.round(4*sigz))  
#        zz,tt = np.meshgrid(z,t)
#        
#        kconv = np.exp(-tt**2*1.0/(sigt**2)-zz**2*1.0/(sigz**2))
#        kconv = kconv/(1.0*np.sum(kconv))
        
        if has_mask(self.profile): #hasattr(self.profile,'mask'):
            prof_mask = np.ma.getmask(self.profile).copy()
            prof0 = np.ma.getdata(self.profile).copy()
            scale = np.ones(self.profile.shape)
            scale[prof_mask] = 0
            scale = scipy.signal.convolve2d(scale,kconv,mode='same')
            scale[scale==0] = 1  # avoid a divide by zero
#            scale = np.ma.array(np.ones(self.profile.shape),mask=prof_mask)
#            scale = scipy.signal.convolve2d(scale.filled(0),kconv,mode='same')  # adjustment factor for the number of points included due to masking
#            scale[np.nonzero(scale==0)] = 1  # avoid a divide by zero
            self.profile = scipy.signal.convolve2d(self.profile.filled(0),kconv,mode='same')/scale
            self.profile[prof_mask] = prof0[prof_mask]
            if keep_mask:
                self.profile = np.ma.array(self.profile,mask=prof_mask)
#            else:
#                self.profile = pnew.copy()
        else:
            self.profile = scipy.signal.convolve2d(self.profile,kconv,mode='same')
        
        if has_mask(self.profile_variance): #hasattr(self.profile_variance,'mask'):
            prof_mask = np.ma.getmask(self.profile_variance).copy()
            prof_var0 = np.ma.getdata(self.profile_variance).copy()
            scale = np.ones(self.profile_variance.shape)
            scale[prof_mask] = 0
            scale = scipy.signal.convolve2d(scale,kconv,mode='same')
#            scale = np.ma.array(np.ones(self.profile_variance.shape),mask=prof_mask)
#            scale = scipy.signal.convolve2d(scale.filled(0),kconv,mode='same')  # adjustment factor for the number of points included due to masking
#            scale[np.nonzero(scale==0)] = 1  # avoid a divide by zero
            self.profile_variance = scipy.signal.convolve2d(self.profile_variance.filled(0),kconv**2,mode='same')/scale**2
            self.profile_variance[prof_mask] = prof_var0[prof_mask]
            if keep_mask:
                self.profile_variance = np.ma.array(self.profile_variance,mask=prof_mask)
        else:
            self.profile_variance = scipy.signal.convolve2d(self.profile_variance,kconv**2,mode='same')
        
        
        self.ProcessingStatus.extend(['Convolved Profile with Gaussian, sigma_t = %f, sigma_z = %f'%(sigt,sigz)])
    def conv2d(self,kernel,keep_mask=True,descript='',update=True,edge_scale=True,mode='same'):
        """
        Perform a 2D convolution with the profile using a provided convolution kernel
        """
        if has_mask(self.profile): #hasattr(self.profile,'mask'):
#            print(self.profile.shape)
            prof_mask = np.ma.getmask(self.profile)
            prof0 = np.ma.getdata(self.profile.data)
            if edge_scale:
                scale = np.ones(self.profile.shape)
                scale[prof_mask] = 0
                scale = scipy.signal.convolve2d(scale,kernel,mode='same')  # adjustment factor for the number of points included due to masking
    #            scale = np.ma.array(np.ones(self.profile.shape),mask=prof_mask)
    #            scale = scipy.signal.convolve2d(scale.filled(0),kernel,mode='same')  # adjustment factor for the number of points included due to masking
                scale[scale==0] = 1  # avoid a divide by zero
                prof = scipy.signal.convolve2d(self.profile.filled(0),kernel,mode=mode)/scale
            else:
                prof = scipy.signal.convolve2d(prof0,kernel,mode=mode)
#            plt.semilogy(prof[30,:])
            prof[prof_mask] = prof0[prof_mask]
#            plt.semilogy(prof[30,:])
#            print(self.profile.shape)
#            prof = scipy.signal.convolve2d(self.profile,kernel,mode='same')
            if keep_mask:
                prof = np.ma.array(prof,mask=prof_mask)
#            plt.semilogy(prof[30,:])
        else:
            prof = scipy.signal.convolve2d(self.profile,kernel,mode=mode)
            if edge_scale:
                scale = np.ones(self.profile.shape)
                scale = scipy.signal.convolve2d(scale,kernel,mode=mode)
                prof = prof/scale
            
        
        if has_mask(self.profile_variance): #hasattr(self.profile_variance,'mask'):
            prof_mask = np.ma.getmask(self.profile_variance)
            prof_var0 = np.ma.getdata(self.profile_variance)
            if edge_scale:
                scale = np.ones(self.profile_variance.shape)
                scale[prof_mask] = 0
                scale = scipy.signal.convolve2d(scale,kernel,mode=mode)  # adjustment factor for the number of points included due to masking
    #            scale = np.ma.array(np.ones(self.profile_variance.shape),mask=prof_mask)
    #            scale = scipy.signal.convolve2d(scale.filled(0),kernel,mode='same')  # adjustment factor for the number of points included due to masking
                scale[np.nonzero(scale==0)] = 1  # avoid a divide by zero
                prof_var = scipy.signal.convolve2d(self.profile_variance.filled(0),kernel**2,mode=mode)/scale**2
            else:
                prof_var = scipy.signal.convolve2d(prof_var0,kernel**2,mode=mode)
            prof_var[prof_mask] = prof_var0[prof_mask]

#            prof_var = scipy.signal.convolve2d(self.profile_variance,kernel**2,mode='same')
            if keep_mask:
                prof_var = np.ma.array(prof,mask=prof_mask)
        else:
            prof_var = scipy.signal.convolve2d(self.profile_variance,kernel**2,mode=mode)
            if edge_scale:
                prof_var=prof_var/scale**2
        
        if update:
            self.profile = prof.copy()
            try:
                self.profile_variance = prof_var.copy()
            except:
                pass
            self.ProcessingStatus.extend(['Convolved Profile with a kernel: '+descript])
        else:
            pnew = self.copy()
            pnew.profile = prof.copy()
            try:
                pnew.profile_variance = prof_var.copy()
            except:
                pass
            pnew.ProcessingStatus.extend(['Convolved Profile with a kernel: '+descript])
            
            return pnew
        
    def SNR(self):
        """
        Calculate and return the SNR of the profile
        """
        return np.abs(self.profile)/np.sqrt(self.profile_variance)
    
    def time_edges(self):
        """
        Returns the edges of the time bins, where self.time is the center.
        Some assumptions go into this calculation.
        It is assumed that the first bin edge is mean_dt/2 before the center of the first point.
        It is assumed that the last bin edge is mean_dt/2 after the center of the last point.
        It is assumed that the edges are half way between the centers of all the other points.
        """
        t_edges = np.concatenate((np.array([self.time[0]-self.mean_dt*0.5]),0.5*(self.time[:-1]+self.time[1:]),np.array([self.time[-1]+self.mean_dt*0.5])))
        return t_edges
    
    def p_thin(self,n=2):
        """
        Use Poisson thinning to creat n statistically independent copies of the
        profile.  This should only be used for photon count profiles where the
        counts are a poisson random number 
        (before background subtraction and overlap correction)
        The copied profiles are returned in a list
        """
        
        if any('Background Subtracted' in s for s in self.ProcessingStatus):
            print('Warning:  poisson thinning (self.p_thin) called on %s \n   %s has been background subtracted so it is \n   not strictly a poisson random number \n   applying anyway.' %(self.label,self.label))
        
        proflist = []
        p = 0.5
        apply_mask = False
        if has_mask(self.profile):
            prof_mask = np.ma.getmask(self.profile)
            apply_mask = True
#        p = 1.0/n
#        for ai in range(n):
#            copy = self.copy()
#            copy.profile = np.random.binomial(self.profile.astype(np.int),p,size=self.profile.shape)
#            copy.profile_variance = copy.profile_variance*p
#            
#            copy.label = copy.label + 'Poisson Thinned %d'%ai
#            copy.ProcessingStatus.extend(['Poisson Thinned copy %d out of %d'%(ai,n)])
#            proflist.extend([copy])
        copy = self.copy()
        copy.profile = np.random.binomial(self.profile.astype(np.int),p,size=self.profile.shape)
        copy.profile_variance = copy.profile_variance*p**2
        
        copy.label = self.label + ' Poisson Thinned 0'
        copy.ProcessingStatus.extend(['Poisson Thinned copy 0 out of 2'])
        proflist.extend([copy])
        
        copy2 = self.copy()
        copy2.profile = self.profile-copy.profile
        copy2.profile_variance = copy.profile_variance*p**2
        
        copy2.label = self.label + ' Poisson Thinned 1'
        copy2.ProcessingStatus.extend(['Poisson Thinned copy 1 out of 2'])
        proflist.extend([copy2])
        
        if apply_mask:
            # transfer the mask
            for p in proflist:
                p.profile = np.ma.array(p.profile,mask=prof_mask)
                p.profile_variance = np.ma.array(p.profile_variance,mask=prof_mask)
        
            
        return proflist
        
    def divide_prof(self,denom_profile):
        """
        Divides the current profile by another lidar profile (denom_profile)
        Propagates the profile error from the operation.
        """
        SNRnum = self.SNR()
        SNRden = denom_profile.SNR()
        self.profile = self.profile/denom_profile.profile
        self.profile_variance = self.profile**2*(1.0/SNRnum**2+1.0/SNRden**2)   
        self.ProcessingStatus.extend([ 'Divided by ' + denom_profile.label ])
        
    def multiply_prof(self,profile2):
        """
        multiplies the current profile by another lidar profile (profile2)
        Propagates the profile error from the operation.
        """
        self.profile = self.profile*profile2.profile
        self.profile_variance = self.profile**2*profile2.profile_variance+self.profile_variance*profile2.profile**2
        self.ProcessingStatus.extend([ 'Multiplied by ' + profile2.label ])
    
    def multiply_piecewise(self,mult_array):
        """
        multiplies the current profile by an array
        Propagates the profile error from the operation.
        """
        self.profile = self.profile*mult_array
        self.profile_variance = self.profile_variance*mult_array**2
        self.ProcessingStatus.extend(['Performed piecewise multiplication'])
    def log(self,update=False):
        """
        Performs the natural log operation on the profile
        """
        if update:
            self.profile_variance = self.profile_variance/self.profile**2
            self.profile = np.log(self.profile)
            self.cat_ProcessingStatus('Performed natural log operation')
        else:
            pnew = self.copy()
            pnew.profile_variance = pnew.profile_variance/pnew.profile**2
            pnew.profile = np.log(pnew.profile)
            pnew.cat_ProcessingStatus('Performed natural log operation')
            return pnew
        
    def diff(self,axis=1,deriv=True,adj_axis=True,update=False):
        """
        Performs a difference operation on the profie along the defined axis
        axis = 1 -range dimension
        axis = 0 -time dimension
        deriv = True - divide by the independent variable resolution
        adj_axis = True - center the independent variable axis
        update = True - apply to the current profile
                = False - return a new profile
        """
        
        
        if update:
            if axis == 0:
                self.profile = np.diff(self.profile,axis=0)
                self.profile_variance = self.profile_variance[1:,:]+self.profile_variance[:-1,:]
                if len(self.bg) > 0:
                    self.bg = np.diff(self.bg)
                    self.bg_var = self.bg_var[1:]+self.bg_var[:-1]
                
                if deriv or adj_axis:
                    dtime = np.diff(self.time)
                    if deriv:
                        self.profile = self.profile/dtime[:,np.newaxis]
                        self.profile_variance=self.profile_variance/dtime[:,np.newaxis]**2
                        if len(self.bg) > 0:
                            self.bg = self.bg/dtime
                            self.bg_var = self.bg_var/dtime**2
                        self.cat_ProcessingStatus('Apply time derivative')
                    else:
                        self.cat_ProcessingStatus('Apply time difference')
                    if adj_axis:
                        self.time = self.time[:-1]+dtime/2.0
                    else:
                        self.time = self.time[:-1]
            if axis == 1:
                self.profile = np.diff(self.profile,axis=1)
                self.profile_variance = self.profile_variance[:,1:]+self.profile_variance[:,:-1]
                
                if deriv or adj_axis:
                    drange = np.diff(self.range_array)
                    if deriv:
                        self.profile = self.profile/drange[np.newaxis,:]
                        self.profile_variance=self.profile_variance/drange[np.newaxis,:]**2
                        self.cat_ProcessingStatus('Apply range derivative')
                    else:
                        self.cat_ProcessingStatus('Apply range difference')
                    if adj_axis:
                        self.range_array = self.range_array[:-1]+drange/2.0 
                    else:
                        self.range_array = self.range_array[:-1]
        else:
            pnew = self.copy()
            if axis == 0:
                pnew.profile = np.diff(pnew.profile,axis=0)
                pnew.profile_variance = pnew.profile_variance[1:,:]+pnew.profile_variance[:-1,:]
                if len(pnew.bg) > 0:
                    pnew.bg = np.diff(self.bg)
                    pnew.bg_var = pnew.bg_var[1:]+pnew.bg_var[:-1]
                
                if deriv or adj_axis:
                    dtime = np.diff(pnew.time)
                    if deriv:
                        pnew.profile = self.profile/dtime[:,np.newaxis]
                        pnew.profile_variance=self.profile_variance/dtime[:,np.newaxis]**2
                        pnew.cat_ProcessingStatus('Apply time derivative')
                        if len(pnew.bg) > 0:
                            pnew.bg = self.bg/dtime
                            pnew.bg_var = self.bg_var/dtime**2
                    else:
                        pnew.cat_ProcessingStatus('Apply time difference')
                    if adj_axis:
                        pnew.time = self.time[:-1]+dtime/2.0
                    else:
                        pnew.time = self.time[:-1]
            if axis == 1:
                pnew.profile = np.diff(pnew.profile,axis=1)
                pnew.profile_variance = pnew.profile_variance[:,1:]+pnew.profile_variance[:,:-1]
                
                if deriv or adj_axis:
                    drange = np.diff(pnew.range_array)
                    if deriv:
                        pnew.profile = pnew.profile/drange[np.newaxis,:]
                        pnew.profile_variance=self.profile_variance/drange[np.newaxis,:]**2
                        pnew.cat_ProcessingStatus('Apply range derivative')
                    else:
                        pnew.cat_ProcessingStatus('Apply range difference')
                    if adj_axis:
                        pnew.range_array = pnew.range_array[:-1]+drange/2.0 
                    else:
                        pnew.range_array = pnew.range_array[:-1]
                        
            return pnew
                
                
    
    def sg_filter(self,window,order,deriv=0,axis=1,keep_mask=False,edges=False,norm=False):
        """
        apply a savitzky golay filter to the profile along the specified
        axis.
        window - SG filter window width.  must be odd
        order - polynomial order of the sg filter.  must be < window-1
        deriv - derivative order desired
        axis - axis along which to apply the filter.  range = 1, time =0
        edges- (True) use manual polynomial fitting to span the edge cases
        norm - (True) normalize based on the number of points covered by the filter
        """
        if axis == 1:
            if not hasattr(window,'__iter__') and not hasattr(order,'__iter__'):
                if edges:
#                    pwin=(window-1)//2
#                    porder = np.maximum(order//2,1)
                    pwin= window
                    porder = order
                    iedge = np.int((window-1)/2)
                    pedges1 = self.profile[:,:pwin]
                    pedges2 = self.profile[:,-pwin:]
                    
                    xedges0 = -np.arange(pwin)
                    pow0 = np.arange(order+1)
                    dpow0 = np.maximum(pow0-deriv,0)
                    dcoeff = sp.special.factorial(pow0)/sp.special.factorial(pow0-deriv)
                    dcoeff[pow0-deriv<0] = 0
                    xedges = np.matrix(xedges0[:,np.newaxis]**pow0[np.newaxis,:])
                    dxedges = np.matrix(dcoeff[np.newaxis,:]*xedges0[:,np.newaxis]**dpow0[np.newaxis,:])
#                    print('order= %d, deriv= %d'%(order,deriv))
#                    print(pow0)
#                    print(dpow0)
#                    print(dcoeff)
                    
                    ydat = np.matrix(pedges1.T)
                    pfit = np.linalg.pinv(xedges)*ydat
                    yedges1=np.array(dxedges*pfit).T
                    
                    ydat = np.matrix(pedges2.T)
                    pfit = np.linalg.pinv(xedges)*ydat
                    yedges2=np.array(dxedges*pfit).T
                                       
                    if deriv > 1:
                        yedges1*=-1
                        yedges2*=-1
                    
#                    # use padding instead of poly fit
#                    if deriv > 0:
#                        self.profile = np.pad(self.profile,[(0,0),(iedge,iedge)],mode='reflect',reflect_type='odd')
#                        self.profile_variance = np.pad(self.profile_variance,[(0,0),(iedge,iedge)],mode='reflect',reflect_type='odd')
#                    else:
#                        self.profile = np.pad(self.profile,[(0,0),(iedge,iedge)],mode='reflect',reflect_type='even')
#                        self.profile_variance = np.pad(self.profile_variance,[(0,0),(iedge,iedge)],mode='reflect',reflect_type='even')
                    
                # a single kernal definition for the entire profile
                sg_kern = sg_kernel([1,window],[1,order],deriv=[[0,deriv]],grid_space=[1,1])[0]
                
#                # pad edges
#                self.conv2d(sg_kern,keep_mask=keep_mask,edge_scale=norm,mode='valid')
                
                # poly fit edges
                self.conv2d(sg_kern,keep_mask=keep_mask,edge_scale=norm,mode='same')
                if edges:
                    self.profile[:,:iedge] = yedges1[:,:iedge]
                    self.profile[:,-iedge:] = yedges2[:,-iedge:]

            else: 
                if not hasattr(window,'__iter__'):
                    window = window*np.ones(self.time.size)
                if not hasattr(order,'__iter__'):
                    order = order*np.ones(self.time.size)
                for ai in range(self.profile.shape[0]):
                    if edges:
#                        pwin=(window[ai]-1)//2
#                        porder = np.maximum(order[ai]//2,1)
                        pwin= window[ai]
                        porder = order[ai]
                        iedge = np.int((window[ai]-1)/2)
                        pedges1 = self.profile[ai,:pwin]
                        pedges2 = self.profile[ai,-pwin:]
                        xedges0 = -np.arange(pwin)
                        pow0 = np.arange(porder+1)
                        dpow0 = np.maximum(pow0-deriv,0)
                        dcoeff = sp.special.factorial(pow0)/sp.special.factorial(pow0-deriv)
                        dcoeff[pow0-deriv<0] = 0
                        xedges = np.matrix(xedges0[:,np.newaxis]**pow0[np.newaxis,:])
                        dxedges = np.matrix(dcoeff[np.newaxis,:]*xedges0[:,np.newaxis]**dpow0[np.newaxis,:])
                        
                        ydat = np.matrix(pedges1[:,np.newaxis])
                        pfit = np.linalg.pinv(xedges)*ydat
                        yedges1=np.array(dxedges*pfit).flatten()
                        
                        ydat = np.matrix(pedges2[:,np.newaxis])
                        pfit = np.linalg.pinv(xedges)*ydat
                        yedges2=np.array(dxedges*pfit).flatten()
                        
#                        if deriv > 1:
#                            yedges1*=-1
#                            yedges2*=-1
                        
                    # generate the sg kernal
                    sg_kern = sg_kernel([1,window[ai]],[1,order[ai]],deriv=[[0,deriv]],grid_space=[1,1])[0]
                    self.profile[ai,:] = np.convolve(self.profile[ai,:],sg_kern.flatten(),mode='same')
                    self.profile_variance[ai,:] = np.convolve(self.profile_variance[ai,:],sg_kern.flatten()**2,mode='same')
                    
                    if edges:
                        self.profile[ai,:iedge] = yedges1[:iedge]
                        self.profile[ai,-iedge:] = yedges2[-iedge:]

#                    self.profile[ai,:] = savitzky_golay(self.profile[ai,:].flatten(), window[ai], order[ai], deriv=deriv)
        else:
            if not hasattr(window,'__iter__') and not hasattr(order,'__iter__'):
                if edges:
#                    pwin=(window-1)//2
#                    porder = np.maximum(order//2,1)
                    pwin= window
                    porder = order
                    
                    iedge = np.int((window-1)/2)
                    pedges1 = self.profile[:pwin,:]
                    pedges2 = self.profile[-pwin:,:]
                    
                    xedges0 = -np.arange(pwin)
                    pow0 = np.arange(porder+1)
                    dpow0 = np.maximum(pow0-deriv,0)
                    dcoeff = sp.special.factorial(pow0)/sp.special.factorial(pow0-deriv)
                    dcoeff[pow0-deriv<0] = 0
                    xedges = np.matrix(xedges0[:,np.newaxis]**pow0[np.newaxis,:])
                    dxedges = np.matrix(dcoeff[np.newaxis,:]*xedges0[:,np.newaxis]**dpow0[np.newaxis,:])
                    
                    ydat = np.matrix(pedges1)
                    pfit = np.linalg.pinv(xedges)*ydat
                    yedges1=np.array(dxedges*pfit)
                    
                    ydat = np.matrix(pedges2)
                    pfit = np.linalg.pinv(xedges)*ydat
                    yedges2=np.array(dxedges*pfit)
                    
#                    if deriv > 1:
#                        yedges1*=-1
#                        yedges2*=-1
                    
                # a single kernal definition for the entire profile
                sg_kern = sg_kernel([window,1],[order,1],deriv=[[deriv,0]],grid_space=[1,1])[0]
                self.conv2d(sg_kern,keep_mask=keep_mask,edge_scale=norm)
                
                if edges:
                    self.profile[:iedge,:] = yedges1[:iedge,:]
                    self.profile[-iedge:,:] = yedges2[-iedge:,:]

            else:
                if not hasattr(window,'__iter__'):
                    window = window*np.ones(self.range_array.size)
                if not hasattr(order,'__iter__'):
                    order = order*np.ones(self.range_array.size)
                for ai in range(self.profile.shape[1]):
                    if edges:
#                        pwin=(window[ai]-1)//2
#                        porder = np.maximum(order[ai]//2,1)
                        
                        pwin= window[ai]
                        porder = order[ai]
                    
                        iedge = np.int((window[ai]-1)/2)
                        pedges1 = self.profile[:pwin,ai]
                        pedges2 = self.profile[-pwin:,ai]
                        xedges0 = -np.arange(pwin)
                        pow0 = np.arange(porder+1)
                        dpow0 = np.maximum(pow0-deriv,0)
                        dcoeff = sp.special.factorial(pow0)/sp.special.factorial(pow0-deriv)
                        dcoeff[pow0-deriv<0] = 0
                        xedges = np.matrix(xedges0[:,np.newaxis]**pow0[np.newaxis,:])
                        dxedges = np.matrix(dcoeff[np.newaxis,:]*xedges0[:,np.newaxis]**dpow0[np.newaxis,:])
                        
                        ydat = np.matrix(pedges1[:,np.newaxis])
                        pfit = np.linalg.pinv(xedges)*ydat
                        yedges1=np.array(dxedges*pfit).flatten()
                        
                        ydat = np.matrix(pedges2[:,np.newaxis])
                        pfit = np.linalg.pinv(xedges)*ydat
                        yedges2=np.array(dxedges*pfit).flatten()
                        
#                        if deriv > 1:
#                            yedges1*=-1
#                            yedges2*=-1
                        
                    sg_kern = sg_kernel([window[ai],1],[order[ai],1],deriv=[[deriv,0]],grid_space=[1,1])[0]
                    self.profile[:,ai] = np.convolve(self.profile[:,ai],sg_kern.flatten(),mode='same')
                    self.profile_variance[:,ai] = np.convolve(self.profile_variance[:,ai],sg_kern.flatten()**2,mode='same')
                    if edges:
                        self.profile[:iedge,ai] = yedges1[:iedge]
                        self.profile[-iedge:,ai] = yedges2[-iedge:]

                        
#                    self.profile[:,ai] = savitzky_golay(self.profile[:,ai].flatten(), window[ai], order[ai], deriv=deriv)
    def gaussian_filter(self,std,axis=1,edges=True):
        """
        apply a Gaussian filter to the profile along the specified
        axis.
        std - Guassian filter window width.
        axis - axis along which to apply the filter.  range = 1, time =0
        edges - if true, adjust for edge overlap
        """
        apply_mask = False
        if has_mask(self.profile):
            save_mask = np.ma.getmask(self.profile).copy()
            apply_mask = True
        
        warn=True
        if axis == 1:
            if not hasattr(std,'__iter__'):
                # a single kernal definition for the entire profile
                _,_,g_kern = get_conv_kernel(0,std)
                if g_kern.size > self.profile.shape[1]:
                    size_diff = np.int((g_kern.size-self.profile.shape[1])/2)
                    g_kern = g_kern[:,size_diff:-size_diff]
                self.conv2d(g_kern,descript='Gaussian (0,%f)'%std)
            else: 
                for ai in range(self.profile.shape[0]):
                    # generate the Gaussian kernal
                    _,_,g_kern = get_conv_kernel(0,std[ai])
                    if g_kern.size > self.profile[ai,:].size:
                        if warn:
                            print('Warning, convolution kernel is larger than the range dimension')
                            print('kernel std: %e'%std[ai])
                            print('kernel size: %d'%g_kern.size)
                            print('profile range dimension: %d'%self.profile[:,ai].size)
                        
                        size_diff = np.int((g_kern.size-self.profile[:,ai].size)/2)
                        g_kern = g_kern[:,size_diff:-size_diff-1]
                        if warn:
                            print('new kernel size: %d'%g_kern.size)
                            warn = False # supress future warnings
                        
                    if edges:               
                        scale = np.ones(self.profile[ai,:].size,dtype=np.float)
                        if apply_mask:
                            scale[save_mask[ai,:]] = 0
                            prof0 = self.profile[ai,:].filled(0)
                            profvar0 = self.profile_variance[ai,:].filled(0)
                        else:
                            prof0 = self.profile[ai,:]
                            profvar0 = self.profile_variance[ai,:]
                            
                        scale = np.convolve(scale,g_kern.flatten(),mode='same')
                        scale[scale == 0] = 1
                        self.profile[ai,:] = np.convolve(prof0,g_kern.flatten(),mode='same')/scale
                        self.profile_variance[ai,:] = np.convolve(profvar0,g_kern.flatten()**2,mode='same')/scale**2
                    else:
                        self.profile[ai,:] = np.convolve(self.profile[ai,:],g_kern.flatten(),mode='same')
                        self.profile_variance[ai,:] = np.convolve(self.profile_variance[ai,:],g_kern.flatten()**2,mode='same')
                    if apply_mask:
                        self.profile=np.ma.array(self.profile,mask=save_mask)
                        self.profile_variance=np.ma.array(self.profile_variance,mask=save_mask)
        else:
            if not hasattr(std,'__iter__'):
                # a single kernal definition for the entire profile
                _,_,g_kern = get_conv_kernel(std,0)
                if g_kern.size > self.profile.shape[0]:
                    size_diff = np.int((g_kern.size-self.profile.shape[0])/2)
                    g_kern = g_kern[size_diff:-size_diff,:]
                self.conv2d(g_kern,descript='Gaussian (%f,0)'%std)
            else:
                for ai in range(self.profile.shape[1]):
                    _,_,g_kern = get_conv_kernel(std[ai],0)
                    if g_kern.size > self.profile[:,ai].size:
                        if warn:
                            print('Warning, convolution kernel is larger than the time dimension')
                            print('kernel std: %e'%std[ai])
                            print('kernel size: %d'%g_kern.size)
                            print('profile time dimension: %d'%self.profile[:,ai].size)
                        size_diff = np.int((g_kern.size-self.profile[:,ai].size)/2)
                        g_kern = g_kern[size_diff:-size_diff-1,:]
                        if warn:
                            print('new kernel size: %d'%g_kern.size)
                            warn = False # supress future warnings
                    if edges:
                        scale = np.ones(self.profile[:,ai].size,dtype=np.float)
                        if apply_mask:
                            scale[save_mask[:,ai]] = 0
                            prof0 = self.profile[:,ai].filled(0)
                            profvar0 = self.profile_variance[:,ai].filled(0)
                        else:
                            prof0 = self.profile[:,ai]
                            profvar0 = self.profile_variance[:,ai]
                            
                        scale = np.convolve(scale,g_kern.flatten(),mode='same')
                        scale[scale == 0] = 1
                        self.profile[:,ai] = np.convolve(prof0,g_kern.flatten(),mode='same')/scale
                        self.profile_variance[:,ai] = np.convolve(profvar0,g_kern.flatten()**2,mode='same')/scale**2
                    else:   
                        self.profile[:,ai] = np.convolve(self.profile[:,ai],g_kern.flatten(),mode='same')
                        self.profile_variance[:,ai] = np.convolve(self.profile_variance[:,ai],g_kern.flatten()**2,mode='same')
                    if apply_mask:
                        self.profile=np.ma.array(self.profile,mask=save_mask)
                        self.profile_variance=np.ma.array(self.profile_variance,mask=save_mask)

    def fill_blanks(self,update=True):
        """
        Returns a profile with blank time segments filled with zeros and masked.
        Use this for plotting.  It is inefficient for standard processing.
        You can un-do this by calling self.trim_to_on() which removes
        data where the lidar does not appear to be operating
        """
        if np.isnan(self.mean_dt):
            self.mean_dt = np.nanmedian(np.diff(self.time))
            
        if not 'Filled blank times with masked zeros' in self.ProcessingStatus:
            t_master = np.arange(self.time[0],self.time[-1]+2*self.mean_dt,self.mean_dt)-self.mean_dt/2.0
            bin_dig = np.digitize(self.time,t_master)-1
            prof = np.ma.array(np.zeros((t_master.size-1,self.range_array.size)))
            pmask = np.ones(prof.shape,dtype=np.bool)
            
            prof[bin_dig,:] = self.profile 
            
            if has_mask(self.profile): #hasattr(self.profile,'mask'):
                pmask[bin_dig,:] = self.profile.mask 
            else:
                pmask[bin_dig,:] = np.zeros(self.range_array.size,dtype=np.bool)
            
            prof.mask = pmask
            time = t_master[1:]-self.mean_dt/2.0        
            
            if update:
                self.time = time.copy()
                
                shot_count = self.shot_count.copy()
                self.shot_count = np.zeros(self.time.size)
                self.shot_count[bin_dig] = shot_count
                
                NumProfList = self.NumProfList.copy()
                self.NumProfList = np.zeros(self.time.size)
                self.NumProfList[bin_dig] = NumProfList
                
                # only fill in background if it has been subtracted
                if len(self.bg) > 0:
                    try:
                        bg = self.bg.copy()
                        self.bg = np.zeros(self.time.size)
                        self.bg[bin_dig] = bg
                    except ValueError:
                        pass
                    
                    try:
                        bg_var = self.bg_var.copy()
                        self.bg_var = np.zeros(self.time.size)
                        self.bg_var[bin_dig] = bg_var
                    except ValueError:
                        pass
                
                prof_var = np.ma.array(np.zeros((t_master.size-1,self.range_array.size)))
                prof_var[bin_dig,:] = self.profile_variance
                prof_var.mask = pmask
                self.profile_variance = prof_var.copy()
    
                self.profile = prof.copy()
    
                self.cat_ProcessingStatus('Filled blank times with masked zeros')
            else:
                return prof,time
    
    def range2alt(self,altitude_grid,aircraft_data,telescope_direction=np.array([]),lidar_tilt=[0,4.0]):
        """
        converts range lidar data into altitude based lidar data
        altitude_grid - the desired master altitude grid
        aircraft_data - dict containing the aircraft pitch,roll and altitudes
        lidar_tilt - the lidar tilt angles along the [forward,lateral] directions
            in degrees
        """
        alt_res = np.abs(np.mean(np.diff(altitude_grid)))
        
        # convert telescope direction data into a more useful [-1,1] set
        if len(telescope_direction) > 0:
            telescope_direction = np.sign(telescope_direction-0.5)
        else:
            telescope_direction = np.ones(self.time.size)
        
        # create a 2D array of all the raw altitude data caputured by the lidar
#        alt_raw = \
#            (self.range_array[:,np.newaxis]*telescope_direction[np.newaxis,:]\
#            *np.cos((aircraft_data['ROLL']+lidar_tilt[1])*np.pi/180)*np.cos((aircraft_data['PITCH']+lidar_tilt[0])*np.pi/180)\
#            +aircraft_data['GGALT'][np.newaxis,:]).T
        
        # correct for lidar tilt being additive with roll when pointing down, 
        # but subtracting when pointing up
        # 
        alt_raw = \
            (self.range_array[:,np.newaxis]*telescope_direction[np.newaxis,:]\
            *np.cos((aircraft_data['ROLL']-lidar_tilt[1]*telescope_direction)*np.pi/180)*np.cos((aircraft_data['PITCH']+lidar_tilt[0])*np.pi/180)\
            +aircraft_data['GGALT'][np.newaxis,:]).T
            
        new_profile = np.zeros((self.time.size,altitude_grid.size))*np.nan
        new_variance = np.zeros((self.time.size,altitude_grid.size))*np.nan
        
#        conv_kernel = np.ones(np.round(alt_res/self.mean_dR))
        for ai in range(alt_raw.shape[0]):
            conv_size = np.round(alt_res/np.abs(np.mean(np.diff(alt_raw[ai,:]))))
            if np.isinf(conv_size):
                conv_size = 1
            elif np.isnan(conv_size):
                conv_size = 1
            conv_kernel = np.ones(np.int(conv_size))
            conv_area = np.sum(conv_kernel)
            if telescope_direction[ai] < 0:
                new_profile[ai,:] = np.interp(altitude_grid,alt_raw[ai,::-1],np.convolve(self.profile[ai,:],conv_kernel,'same')[::-1]/conv_area,left=np.nan,right=np.nan)
                new_variance[ai,:] = np.interp(altitude_grid,alt_raw[ai,::-1],np.convolve(self.profile_variance[ai,:],conv_kernel,'same')[::-1]/conv_area**2,left=np.nan,right=np.nan)
            else:
                new_profile[ai,:] = np.interp(altitude_grid,alt_raw[ai,:],np.convolve(self.profile[ai,:],conv_kernel,'same')/conv_area,left=np.nan,right=np.nan)
                new_variance[ai,:] = np.interp(altitude_grid,alt_raw[ai,:],np.convolve(self.profile_variance[ai,:],conv_kernel,'same')/conv_area**2,left=np.nan,right=np.nan)
        mask_no_obs = np.isnan(new_profile) # mask the regions that are not filled in
        self.range_array = altitude_grid
        self.profile = new_profile.copy()
        self.profile_variance = new_variance
        self.mean_dR = alt_res
        self.cat_ProcessingStatus('Converted range to altitude data')
        self.mask(mask_no_obs)  # apply the mask
    
    def write2nc(self,ncfilename,tag='',name_override=False,overwrite=False,write_axes=False,axes_names=None,dim_names=None,dtype='float32'):
        """
        Writes the current profile out to a netcdf file named ncfilename
        adds the string tag the variable name if name_override==False
        if name_override=True, it names the variable according to the string
        tag.
        if overwrite = True, the routine will overwrite nc data of the same name.
        if overwrite = False, the routine will abort without overwriting
        if write_axes = True, writes out unique axes (time/range) definitions for each profile
        axes_names - list of the time and range names to be used in the netcdf file.  If these are in use, but don't
            match the dimensions, it will create a new dimension axes_name[0]+variable_name
        dim_names - list of dimension names to use for the dimension axes
        """

        if axes_names is None:
            axes_names = ['time','range']
        if dim_names is None:
            dim_names = ['time','range']
        
        tdim = dim_names[0]
        rdim = dim_names[1]

        nc_error = False
        if os.path.isfile(ncfilename):
            # if the file already exists set to modify it
            mod_arg = 'r+'
#            fnc = nc4.Dataset(ncfilename,'r+') #'w' stands for write, format='NETCDF4'
        else:
            mod_arg = 'w'
#            fnc = nc4.Dataset(ncfilename,'w') #'w' stands for write, format='NETCDF4'
        
        with nc4.Dataset(ncfilename,mod_arg) as fnc:
        
            # determine the first axes (time) variable name
            # try to use the requested time name, but update if necessary
            if write_axes:
                # write out a unique time axis for this profile
                name_count = 1
                taxis = axes_names[0]+'_'+self.label.replace(' ','_')+tag
            else:
                name_count = 0
                taxis = axes_names[0]
                 
            write_time = True
            while name_count < 2:          
                if taxis in fnc.variables.keys():
                    if list(fnc.variables[taxis][:]) == list(self.time):
                        # the existing time variable matchis this profile
                        # no need to rewrite it
                        tdim = fnc.variables[taxis].dimensions[0]
                        write_time = False
                        name_count = 2
                    elif name_count < 1:
                        # try adding a tag to the time variable name
                        taxis = taxis+'_'+self.label.replace(' ','_')+tag
                    else:
                        # if the tagged variable exists and doesn't match, throw a warning
                        nc_error = True
                        write_time = False
                        print('Warning in %s write2nc to %s ' %(self.label,ncfilename))
                        print('  Attempted to use <%s> as the time variable'%taxis)
                        print('  Time variable already exists.')
                        print('  No data was written.')
                else:
                    name_count=2
                    
                name_count+=1
                
            # determine the first (time) dimension name
            name_count=0
            while name_count < 2 and write_time and not nc_error:
                if tdim in fnc.dimensions.keys():
                    if fnc.dimensions[tdim].size != self.time.size:
                        if name_count < 1:
                            tdim = tdim+'_'+self.label.replace(' ','_')+tag
                        else:
                            write_time = False
                            nc_error = True
                            print('Warning in %s write2nc to %s ' %(self.label,ncfilename))
                            print('  Attempted to use <%s> as the time dimension'%tdim)
                            print('  Time dimension already exists.')
                            print('  No data was written.')
                    else:
                        name_count = 2
                else:
                    fnc.createDimension(tdim,self.time.size)
                    name_count = 2
                
                name_count+=1
            
            if write_time:
                timeNC = fnc.createVariable(taxis,'float32',(tdim,))
                timeNC[:] = self.time.copy()
                timeNC.units = self.StartDate.strftime("seconds since %Y-%m-%dT%H:%M:%SZ").encode('ascii')
                
        
            
            # determine the second axes (range) variable name
            # try to use the requested time name, but update if necessary
            if write_axes:
                # write out a unique range axis for this profile
                name_count = 1
                raxis = axes_names[1]+'_'+self.label.replace(' ','_')+tag
            else:
                name_count = 0
                raxis = axes_names[1]
                 
            write_range = True
            while name_count < 2:
                if raxis in fnc.variables.keys():
                    if list(fnc.variables[raxis][:]) == list(self.range_array):
                        # the existing time variable matchis this profile
                        # no need to rewrite it
                        rdim = fnc.variables[raxis].dimensions[0]
                        write_range = False
                        name_count = 2
                    elif name_count < 1:
                        # try adding a tag to the time variable name
                        raxis = raxis+'_'+self.label.replace(' ','_')+tag
                    else:
                        # if the tagged variable exists and doesn't match, throw a warning
                        nc_error = True
                        write_range = False
                        print('Warning in %s write2nc to %s ' %(self.label,ncfilename))
                        print('  Attempted to use <%s> as the range variable'%raxis)
                        print('  Time variable already exists.')
                        print('  No data was written.')
                else:
                    name_count=2
                    
                name_count+=1
                
            # determine the second (range) dimension name
            name_count=0
            while name_count < 2 and write_range and not nc_error:
                if rdim in fnc.dimensions.keys():
                    if fnc.dimensions[rdim].size != self.range_array.size:
                        if name_count < 1:
                            rdim = rdim+'_'+self.label.replace(' ','_')+tag
                        else:
                            write_range = False
                            nc_error = True
                            print('Warning in %s write2nc to %s ' %(self.label,ncfilename))
                            print('  Attempted to use <%s> as the time dimension'%tdim)
                            print('  Time dimension already exists.')
                            print('  No data was written.')
                    else:
                        name_count = 2
                else:
                    fnc.createDimension(rdim,self.range_array.size)
                    name_count = 2
                
                name_count+=1
            
            if write_range:
                rangeNC = fnc.createVariable(raxis,'float32',(rdim,))
                rangeNC[:] = self.range_array.copy()
                rangeNC.units = 'meters'.encode('ascii')
        

            
            
            if not nc_error:
                if name_override:
                    varname = tag.replace(' ','_')
                else:
                    varname = self.label.replace(' ','_')+tag
                
                ancillary_str = taxis+' '+raxis+' '+varname+'_variance'+' '+varname+'_ProfileCount'
                
                if not varname in fnc.variables.keys(): 
                    ancillary_str = taxis+' '+raxis+' '+varname+'_variance'+' '+varname+'_ProfileCount'
                    if not varname in fnc.variables.keys():
                        profileNC = fnc.createVariable(varname,dtype,(tdim,rdim))
                        var_profileNC = fnc.createVariable(varname+'_variance',dtype,(tdim,rdim))
                    else:
                        profileNC = fnc.variables[varname]
                        var_profileNC = fnc.variables[varname+'_variance']
                    if has_mask(self.profile): 
                        profileNC[:,:] = self.profile.data.copy()
                        ancillary_str+=' '+varname+'_mask'
                    else:
                        profileNC[:,:] = self.profile.astype(dtype) #copy()
                        
                    profileNC.units = (((self.profile_type.replace('$','')).replace('{','(')).replace('}',')')).encode('ascii')
                    profileNC.description = self.descript.encode('ascii')
                    profileNC.lidar = self.lidar.encode('ascii')
                    profileNC.wavelength = self.wavelength
                    profileNC.classdef = 'LidarProfile'.encode('ascii')  # store that this is written from a LidarProfile class
                    profileNC.ancillary_variables = ancillary_str.encode('ascii')  # store a space delimited list of related variables
                    
                    if has_mask(self.profile_variance): 
                        var_profileNC[:,:] = self.profile_variance.data.astype(dtype)# .copy()
                    else:
                        var_profileNC[:,:] = self.profile_variance.astype(dtype) # .copy()
                    var_profileNC.ancillary_variables = varname
                    
                    
                    StatusStr = ''
                    for str_index in range(len(self.ProcessingStatus)):
                        if str_index == 0:
                            StatusStr =  self.ProcessingStatus[str_index]
                        else:
                            StatusStr = StatusStr + ',' + self.ProcessingStatus[str_index]
                    profileNC.ProcessingStatus = StatusStr
                    
                    
                    if has_mask(self.profile): #hasattr(self.profile,'mask'):
                        if not any(varname+'_mask' == s for s in fnc.variables):
                            mask_profileNC = fnc.createVariable(varname+'_mask','i1',(tdim,rdim))
                        else:
                            mask_profileNC = fnc.variables[varname+'_mask']
                            
                        mask_profileNC[:,:] = self.profile.mask.astype(np.int8)
                        mask_profileNC.units = '1 = Masked, 0 = Not Masked'.encode('ascii')
                        mask_profileNC.ancillary_variables = varname.encode('ascii')
                    # write out the profile counts as a separate variable
                    if not any(varname+'_ProfileCount'== s for s in fnc.variables):
                        pcount_profileNC = fnc.createVariable(varname+'_ProfileCount','i2',(tdim))
                    else:
                        pcount_profileNC = fnc.variables[varname+'_ProfileCount']
                    pcount_profileNC[:]=self.NumProfList.astype(np.int16)
                    pcount_profileNC.units = 'Count'.encode('ascii')
                    pcount_profileNC.description = 'Number of raw profiles integrated into this profile'.encode('ascii')
                    pcount_profileNC.ancillary_variables = varname.encode('ascii')
                        
                    try:
                        fnc.history = (fnc.history + '\nModified: wrote '+varname +' on '+ datetime.datetime.today().strftime('%m/%d/%Y')).encode('ascii')
                    except AttributeError:
                        fnc.history = ('Created ' + datetime.datetime.today().strftime('%d/%m/%y')).encode('ascii')
                else:
                    print('Warning in %s write2nc to %s ' %(self.label,ncfilename))
                    print('  %s data already exists.'%varname)
                    print('  No %s data was written.'%varname)
            else:
                print('No netcdf written due to error')
            


    def ProfileStart(self,*args):
        """
        Provide the start date and time of the profile
        if a format string is provided a string in the requested datetime format
            is returned
        if formatstr is empty, the datetime object is returned
        """
        start_datetime = self.StartDate+datetime.timedelta(microseconds=self.time[0]*1e6)
  
        if args:
            # return a string according to the requested format
            ret_obj = start_datetime.strftime(args[0])
        else:
            # just return the datetime object
            ret_obj = start_datetime            
            
        return ret_obj
    
    def ProfileEnd(self,*args):
        """
        Provide the end date and time of the profile
        if a format string is provided a string in the requested datetime format
            is returned
        if formatstr is empty, the datetime object is returned
        """
        end_datetime = self.StartDate+datetime.timedelta(microseconds=self.time[-1]*1e6)
        
        if args:
            # return a string according to the requested format
            ret_obj = end_datetime.strftime(args[0])
        else:
            # just return the datetime object
            ret_obj = end_datetime 
            
        return ret_obj
            
            
def check_piecewise_compatability(prof1,prof2,operation):
    """
    Checks for alignment between two profiles for a piecewise operation
    operation is a string to be used if an error is thrown
    """
    complete_operation = True
    if not prof1.profile.shape == prof2.profile.shape:
        complete_operation = False
        print('Error in LidarProfile addition: profile dimensions do not align')
        print(prof1.label +  ' has dimensions (%d, %d)' %prof1.profile.shape )
        print(prof2.label + ' has dimensions (%d, %d)' %prof2.profile.shape )
    elif not all(prof1.time == prof2.time):
        complete_operation = False
        print('Error in LidarProfile '+ operation+ ': time arrays do not align')
    elif not all(prof1.range_array == prof2.range_array):
        complete_operation = False
        print('Error in LidarProfile '+ operation +': range arrays do not align')
    
        
    return complete_operation

def has_mask(a):
    """
    accepts an array input, a, and returns a boolean for 
    if it has a valid mask attribute
    Function is added to maintain backward compatability with new ma functions
    where absense of a mask is indicated by .mask=False instead of absense of
    the attribute
    """
    
    mask_attr = False
    # check if there is a mask attribute in the array
    if hasattr(a,'mask'):
        # check if the mask attribute is array-like
        if hasattr(a.mask,'__len__'):
            mask_attr = True
    
    return mask_attr
    
      
def write_proj2nc(ncfilename,project,flight=''):
    """
    Write project information to netcdf file.
    ncfilename - name of netcdf file
    project - string containing the project name
    flight - if applicable, write the flight information (e.g. RF03)
    """
    if os.path.isfile(ncfilename):
        # if the file already exists set to modify it
        fnc = nc4.Dataset(ncfilename,'r+') #'w' stands for write, format='NETCDF4'
    else:
        fnc = nc4.Dataset(ncfilename,'w') #'w' stands for write, format='NETCDF4'
    
    # if flight not provided check for flight information in the project string
    projstr = project.split('RF')
    project = (projstr[0].replace(' ','')).replace(',','')
    if len(flight) <= 0 and len(projstr) > 1:
        flight = 'RF'+projstr[1]
            
    # note:  using ascii encoding so attributes are written as char instead of
    # strings.  
    # strings cannot be read in netcdf4 readers
    fnc.project = project.encode('ascii')
    if len(flight) > 0:
        fnc.flight= flight.encode('ascii')
    fnc.close()
    
def write_var2nc(var,varname,ncfilename,tag='',units='',description='',dtype='double',name_override=False,overwrite=False,dim_name=None):
    """
    Writes the var out to a netcdf file named ncfilename
    adds the string tag the variable name if name_override==False
    if name_override=True, it names the variable according to the string
    tag.
    varname = name for the variable
    units = units of variable or descriptor
    discription = description of the variable
    dtype = variable data type (defaults to double)
    if overwrite = True, the routine will overwrite nc data of the same name.
    if overwrite = False, the routine will abort without overwriting
    dim_name - list of names corresponding to dimensions of the data
    """

    if os.path.isfile(ncfilename):
        # if the file already exists set to modify it
        fnc = nc4.Dataset(ncfilename,'r+') #'w' stands for write, format='NETCDF4'
#        file_exists = True
    else:
        fnc = nc4.Dataset(ncfilename,'w') #'w' stands for write, format='NETCDF4'
#        file_exists = False
    
    # For mutable objects, an empty default definition means the contents
    # will be carried across function calls.
    # To avoid this,  mutable objects should be called as None then filled
    # in with default values in the actual code.
    if dim_name is None:
        dim_name = []    
        
    #  check if that exact variable is in the variable list
    if not (varname in fnc.variables) or overwrite:
        if len(tag)!=0:
            varname = varname+tag
        if not hasattr(var,"__len__"):
            # Scalar Instance
            profileNC = fnc.createVariable(varname,dtype)
            profileNC[:] = var
            profileNC.units = (units.replace('$','')).encode('ascii') 
            profileNC.description = description.encode('ascii')
            profileNC.classdef = 'scalar'.encode('ascii')
        else:
            # Array Instance
#            dim_name = []
            for ai in range(var.ndim):
                if ai < len(dim_name):
                    if len(dim_name) > 0:
                        # check if an empty string is passed, 
                        
                        if dim_name[ai] in fnc.dimensions:
                            # check if the dimension already exists
                            if len(fnc.dimensions[dim_name[ai]]) != var.shape[ai]:
                                # if the dimension exists, check to make sure its size
                                # matches the variable.  If not, create a new dimension
                                fnc.createDimension(varname+'_'+dim_name[ai],var.shape[ai])
                                dim_name[ai] = varname+'_'+dim_name[ai]
                        else:
                            # if the dimension does not exist already, create it
                            fnc.createDimension(dim_name[ai],var.shape[ai])
                    else:
                        # emply string was passed. make a dummy dimension name
                        dim_name[ai] = varname + 'dim%d'%ai
                        fnc.createDimension(dim_name[ai],var.shape[ai])
                else:
                    # no more dimension names passed.  Create dummy names for
                    # the rest
                    dim_name.extend([varname + 'dim%d'%ai])
                    fnc.createDimension(dim_name[ai],var.shape[ai])
            profileNC = fnc.createVariable(varname,dtype,tuple(dim_name))
            profileNC[:] = var.copy()
            profileNC.units = (((units.replace('$','')).replace('{','(')).replace('}',')')).encode('ascii')
            profileNC.description = description.encode('ascii')
            profileNC.classdef = 'array'.encode('ascii')
        
            if has_mask(var): #hasattr(var,'mask'):
                mask_profileNC = fnc.createVariable(varname+'_mask','i8',var.shape)
                mask_profileNC[:] = var.mask.copy()
                mask_profileNC.units = '1 = Masked, 0 = Not Masked'.encode('ascii')
                mask_profileNC.ancillary_variables = varname.encode('ascii')
                profileNC.ancillary_variables = (varname+'_mask').encode('ascii')
        
        try:
            fnc.history = (fnc.history + "\nModified " + datetime.datetime.today().strftime("%m/%d/%Y")).encode('ascii')
        except AttributeError:
            fnc.history = ("Created " + datetime.datetime.today().strftime("%d/%m/%y")).encode('ascii')
#        if file_exists:
#            fnc.history = (fnc.history + "\nModified " + datetime.datetime.today().strftime("%m/%d/%Y")).encode('ascii')
#        else:
#            fnc.history = ("Created " + datetime.datetime.today().strftime("%d/%m/%y")).encode('ascii')
    else:
        print('No %s netcdf data written because the variable already exists.'%varname)
        print('    set overwrite = True to overwrite existing data.')
    fnc.close()     
   
        
#class profile_netcdf():
#    def __init__(self,netcdf_var):
#        self.data = netcdf_var.data.copy()
#        self.dimensions = netcdf_var.dimensions
#        self.binwidth_ns = netcdf_var.binwidth_ns
#    def copy(self):
#        tmp = profile_netcdf(self)
#        return tmp


def get_conv_kernel(sigt,sigz,norm=True):
        """
        Generates a Gaussian convolution kernel for
        standard deviations sigt and sigz in units of grid points.
        """        
        
        nt = np.round(4*sigt)
        nz = np.round(4*sigz)
        t = np.arange(-nt,nt+1)      
        z = np.arange(-nz,nz+1)  
        
        kconv_t = np.exp(-t**2*1.0/(sigt**2))
        if kconv_t.size > 1:
            if np.sum(kconv_t) == 0:
                it0 = np.argmin(np.abs(t))
                kconv_t[it0] = 1.0
        else: 
            kconv_t = np.ones(1)
            
        kconv_z = np.exp(-z**2*1.0/(sigz**2))
        if kconv_z.size > 1:
            if np.sum(kconv_z) == 0:
                iz0 = np.argmin(np.abs(z))
                kconv_z[iz0] = 1.0
        else:
            kconv_z = np.ones(1)
            
        kconv = kconv_t[:,np.newaxis]*kconv_z[np.newaxis,:]
        
#        zz,tt = np.meshgrid(z,t)
#        kconv = np.exp(-tt**2*1.0/(sigt**2)-zz**2*1.0/(sigz**2))
        if norm:
            kconv = kconv/(1.0*np.sum(kconv))
        
        return z,t,kconv
def conv2d(Data,kconv,keep_mask=True,mode='same'):
    """
    performs a 2d convolution on data using the supplied kernel
    Automatically adjusts the scale at the edges
    """

    if has_mask(Data): # hasattr(Data,'mask'):
        prof_mask = Data.mask
        scale = np.ma.array(np.ones(Data.shape),mask=prof_mask)
        scale = scipy.signal.convolve2d(scale.filled(0),kconv,mode=mode)  # adjustment factor for the number of points included due to masking
        Data = scipy.signal.convolve2d(Data.filled(0),kconv,mode=mode)/scale
        if keep_mask:
            Data = np.ma.array(Data,mask=prof_mask)
    else:
        Data = scipy.signal.convolve2d(Data,kconv,mode=mode)
    return Data

def create_ncfilename(ncbase,Years,Months,Days,Hours,tag=''):
    """
    filestring = create_ncfilename(ncbase,Years,Months,Days,Hours)
    Creates a netcdf filename based on the requested processing interval
    and the current date.  Returns a string with the netcdf filename
    
    ncbase = a string containing the base to which dates and times should 
        tagged
    Years, Months, Days, Hours - array outputs generated by 
        generate_WVDIAL_day_list()
    tag = extra string to concatenate on the filename if desired
    """
    
    runday = datetime.datetime.today().strftime("%Y%m%d")
    startstr = str(Years[0])
    if Months[0] < 10:
        startstr = startstr+'0'+str(Months[0])
    else:
        startstr = startstr+str(Months[0])
    if Days[0] < 10:
        startstr = startstr+'0'+str(Days[0])
    else:
        startstr = startstr+str(Days[0])
    if Hours[0,0] < 10:
        startstr = startstr+'T'+'0'+str(np.int(Hours[0,0]))
    else:
        startstr = startstr+'T'+str(np.int(Hours[0,0]))
    Minutes = np.int(60*np.remainder(Hours[0,0],1))
    if Minutes < 10:
        startstr = startstr+'0'+str(Minutes)
    else:
        startstr = startstr+str(Minutes)
    
    stopstr = str(Years[-1])
    if Months[-1] < 10:
        stopstr = stopstr+'0'+str(Months[-1])
    else:
        stopstr = stopstr+str(Months[-1])
    if Days[-1] < 10:
        stopstr = stopstr+'0'+str(Days[-1])
    else:
        stopstr = stopstr+str(Days[-1])
    if Hours[-1,-1] < 10:
        stopstr = stopstr+'T'+'0'+str(np.int(Hours[-1,-1]))
    else:
        stopstr = stopstr+'T'+str(np.int(Hours[-1,-1]))
    Minutes = np.int(60*np.remainder(Hours[-1,-1],1))
    if Minutes < 10:
        stopstr = stopstr+'0'+str(Minutes)
    else:
        stopstr = stopstr+str(Minutes)
    if len(tag) > 0:
        ncfilename = ncbase + '_' + startstr + '_' + stopstr + '_created_' + runday + '_' + tag + '.nc'
    else:
        ncfilename = ncbase + '_' + startstr + '_' + stopstr + '_created_' + runday + '.nc'

    return ncfilename

def get_nc_class(datafile,class_type='LidarProfile'):
    """
    returns a list of all variables of a particular class type
    as written in the routines defined in this library
        class_type:
            'LidarProfile'
            'array'
            'scalar'
    datafile - netcdf file name
    allows us to load all variables of a particular type
    """

    var_list = []    
    with nc4.Dataset(datafile,'r') as f:
        for var in f.variables.keys():
            try:
                if f.variables[var].classdef == class_type:
                    var_list+=[var]
            except AttributeError:
                pass
    return var_list

def load_nc_Profile(datafile,var_name,label='',mask=True):
    """
    load_nc_Profile(datafile,var_name,label='',mask=True)
    loads netcdf data into a lidar profile variable
    datafile - string reporting the location of the data file
    var_name - string giving the name of the netcdf variable
    mask - If True, apply the netcdf supplied mask
        (if one exists)
    """

    
    # Open the netcdf file and load the relevant variables
    with nc4.Dataset(datafile,'r') as f:
        
        # edit to replace all instances of below 
        # if not any(var_name in s for s in f.variables):
        if not var_name in f.variables:
#            f.close()
            print('Cannot load %s from \n%s\nVariable does not exist'%(var_name,datafile))
            return None
        else:
            if hasattr(f.variables[var_name],'ancillary_variables'):
                anc_var = f.variables[var_name].ancillary_variables.split(' ')
                time_str = []
                for ancstr in anc_var:
                    if 'time' in ancstr:
                        time_str+= [ancstr]
                if len(time_str) == 0:
                    time_str = [anc_var[0]]
                    print('  Found ancillary variables but none with \'time\' in the name' )
                    print('     using the first variable: '+time_str[0])
                elif len(time_str) > 1:               
                    print('  Found multiple ancillary variables with \'time\' in the name' )
                    print('     using the variable: '+time_str[0])
                
                # If multiple possible time variables are found, this should really iterate to find one with a date label
                # to make the code more robust.
                # for now loading the start date will cause it to fail if we chose a variable that is not time
                try:
                    ProcStart = datetime.datetime.strptime(f.variables[time_str[0]].units, 'seconds since 0000 UTC on %A %B %d, %Y')
                except ValueError:
                    ProcStart = datetime.datetime.strptime(f.variables[time_str[0]].units, 'seconds since %Y-%m-%dT%H:%M:%SZ')
                    
                timeD = np.array(f.variables[time_str[0]][:]).copy()
            elif 'time_'+var_name in f.variables: 
                # Read in start date
                try:
                    ProcStart = datetime.datetime.strptime(f.variables['time_'+var_name].units, 'seconds since 0000 UTC on %A %B %d, %Y')
                except ValueError:
                    ProcStart = datetime.datetime.strptime(f.variables['time_'+var_name].units, 'seconds since %Y-%m-%dT%H:%M:%SZ')
        #        DateLabel = ProcStart.strftime("%A %B %d, %Y")     
                timeD = np.array(f.variables['time_'+var_name][:].copy());
            else:
                # Read in start date
                try:
                    ProcStart = datetime.datetime.strptime(f.variables['time'].units, 'seconds since 0000 UTC on %A %B %d, %Y')
                except ValueError:
                    ProcStart = datetime.datetime.strptime(f.variables['time'].units, 'seconds since %Y-%m-%dT%H:%M:%SZ')
        #        DateLabel = ProcStart.strftime("%A %B %d, %Y")          
                timeD = np.array(f.variables['time'][:].copy());
        
    #        if any('range_'+var_name in s for s in f.variables):
            if hasattr(f.variables[var_name],'ancillary_variables'):
                anc_var = f.variables[var_name].ancillary_variables.split(' ')
                range_str = []
                for ancstr in anc_var:
                    if 'range' in ancstr:
                        range_str += [ancstr]
                if len(range_str) == 0:
                    range_str = [anc_var[1]]
                    print('  Found ancillary variables but none with \'range\' in the name' )
                    print('     using the second variable: '+range_str[0])
                elif len(range_str) > 1:
                    if 'range' in anc_var[1]:
                        range_str[0] = anc_var[1]   # give priority to the second variable in the list                 
                    print('  Found multiple ancillary variables with \'range\' in the name' )
                    print('     using the variable: '+range_str[0])
                        
                altitude = np.array(f.variables[range_str[0]][:]).copy()
            elif 'range_'+var_name in f.variables:
                altitude = f.variables['range_'+var_name][:].copy()
            else:
                altitude = f.variables['range'][:].copy()
             
            if hasattr(f.variables[var_name],'lidar'):
                lidar = f.variables[var_name].lidar
            else:
                lidar = 'GV-HSRL'
            
            try:
                description = f.variables[var_name].description
            except AttributeError:
                description = ''
            
            # estimate bin width
            binwidth = np.nanmean(np.diff(altitude))*2.0/c       
            
            if len(label)==0:
                label = var_name.replace('_',' ')
            loaded_prof = LidarProfile(f.variables[var_name][:],timeD,\
                label=label,\
                descript = description,\
                lidar=lidar,StartDate=ProcStart,binwidth=binwidth)
            
            
            loaded_prof.range_array = altitude.copy()
            loaded_prof.profile_type = f.variables[var_name].units
            
            if hasattr(f.variables[var_name],'ProfileCount'):
                loaded_prof.NumProfList = f.variables[var_name].ProfileCount.copy()
            elif var_name+'_ProfileCount' in f.variables:
                loaded_prof.NumProfList = f.variables[var_name+'_ProfileCount'][:].copy()
            
            if hasattr(f.variables[var_name],'wavelength'):
                loaded_prof.wavelength = f.variables[var_name].wavelength
            elif any('wavelength_'+var_name in s for s in f.variables):
                loaded_prof.wavelength = f.variables['wavelength_'+var_name][:].copy()
             
    #        if any(var_name+'_mask' in s for s in f.variables) and mask:
            if (var_name+'_mask' in f.variables) and mask:
                data_mask = (f.variables[var_name+'_mask'][:].astype('bool')).copy()
                loaded_prof.mask(data_mask)
    #        if any(var_name+'_variance' in s for s in f.variables):
            if (var_name+'_variance' in f.variables):
                loaded_prof.profile_variance = f.variables[var_name+'_variance'][:].copy()
        
#            f.close()

            return loaded_prof

def load_nc_vars(filename,var_list,readfull=False,var_data0={}):
    """
    Loads data from netcdf
    filename - string with path and filename to load
    var_list - strings of the netcdf variables to load
    readfull - read all variable attributes and store as a dict
        actual data is stored as 'data'
    var_data0 - existing data dictionary to concatenate onto
    """    
    import copy

    print('Loading netcdf data from file:')
    print('   '+filename)
    
#    var_data = dict(zip(var_list,[np.array([])]*len(var_list)))
#    var_data = {}
    if len(var_data0) > 0:
        var_data = copy.deepcopy(var_data0)
    else:
        var_data = {}
        
    
    try:
        with nc4.Dataset(filename,'r') as f:
    #        f = nc4.Dataset(filename,'r')
            
    #        i_nan = []  # list of invalid values from the data system     
    #        for var in var_data.keys():
            for var in var_list:
                if var in f.variables:
    #            if any(var in s for s in f.variables):
                    data = ncvar(f,var,readfull=readfull)
                    if readfull:
                        if var in var_data.keys() and not np.isscalar(data['data']):
    #                    if len(var_data[var]['data']) > 0:
                            if data['data'].ndim > 1:
                                # merge along the last axis that is not equal between the arrays
                                cat_axis = np.nonzero(np.logical_not(np.array(data['data'].shape)==np.array(var_data[var]['data'].shape)))[0]
                                if len(cat_axis)>0:
                                    var_data[var]['data'] = np.concatenate((var_data[var]['data'],data['data']),axis=cat_axis[-1])
                                else:
                                    print('Error concatenating ' + var +'.')
                                    print('Dimensions do not match')
                            else:
                                var_data[var]['data'] = np.concatenate((var_data[var]['data'],data['data']))
                        else:
                            var_data[var] = copy.deepcopy(data)                            
                            
                    else:
                        if var in var_data.keys() and not np.isscalar(data):
                            if data.ndim > 1:
                                cat_axis = np.nonzero(np.array(data.shape)==np.array(var_data[var].shape))[0]
                                if len(cat_axis>0):
                                    var_data[var] = np.concatenate((var_data[var],data),axis=cat_axis[0])
                                else:
                                    print('Error concatenating ' + var +'.')
                                    print('Dimensions do not match')
                            else:
                                if data.ndim > 0:
                                    var_data[var] = np.concatenate((var_data[var],data)) 
                                else:
                                    var_data[var] = np.concatenate((var_data[var],np.array([data.item()]))) 
                        else:
                            if np.isscalar(data):
                                var_data[var] = data
                            elif data.ndim > 0:
                                var_data[var] = data.copy()
                            else:
                                var_data[var] = np.array([data.item()])
                else:
                    print('  '+var+' not found in '+filename)
#                i_nan.extend(list(np.nonzero(var_data[var]==-32767)[0]))
#        i_nan = np.unique(np.array(i_nan))  # force list to be unique
#        
#        # delete data where nans are present
#        for var in var_data.keys():
#            var_data[var] = np.delete(var_data[var],i_nan)

#        f.close()
    except RuntimeError:
        print('    Data file NOT found')
    
    return var_data

def align_profile_times(profile_list):
    """
    align_profile_times(profile_list):
    accepts a list of profiles built on the same master grid
    trims the profiles to eliminate instances where there is not
    data in all of them.
    """
    time_set = np.array([])
    for ai in range(len(profile_list)):
        time_set = np.concatenate((time_set,np.unique(profile_list[ai].time)))
    t_unique,t_count = np.unique(time_set,return_counts=True)
    
    # find times to remove
    # if the time does not exist in all profiles, cut it
    time_rem = t_unique[np.nonzero(t_count != len(profile_list))]
    
    for ai in range(len(profile_list)):
        profile_list[ai].remove_time(time_rem)

def align_profile_ranges(profile_list):
    """
    align_profile_ranges(profile_list):
    accepts a list of profiles built on the same master grid
    trims the profiles to eliminate instances where there is not
    data in all of them.
    """
    range_set = np.array([])
    for ai in range(len(profile_list)):
        range_set = np.concatenate((range_set,np.unique(profile_list[ai].range_array)))
    r_unique,r_count = np.unique(range_set,return_counts=True)
    
    # find times to remove
    # if the time does not exist in all profiles, cut it
    range_rem = r_unique[np.nonzero(r_count != len(profile_list))]
    
    for ai in range(len(profile_list)):
        profile_list[ai].remove_range(range_rem)

def align_profiles(profile_list):
    """
    aligns profiles in time and range
    """
    align_profile_times(profile_list)
    align_profile_ranges(profile_list)


def load_geofile(filename):
    geo = np.loadtxt(filename)
    return geo
    
def load_diff_geofile(filename,chan='backscatter'):
    diff_geo0 = np.loadtxt(filename)
    if chan == 'backscatter':
        diff_geo = {'bins':diff_geo0[:,0],'hi':1.0/diff_geo0[:,1],'lo':1.0/diff_geo0[:,2]}
    else:
        diff_geo = {'bins':diff_geo0[:,0],'cross':1.0/diff_geo0[:,1]}
    return diff_geo
    
def ncvar(ncID,varname,fillnan=False,readfull=False):
    if varname in ncID.variables:
        if readfull:
            data = ncID.variables[varname][:].copy()
            if data.shape == ():
                var = {'data':data.item()}
            else:
                var = {'data':data.copy()}
            if hasattr(ncID.variables[varname],'description'):
                var['description'] = ncID.variables[varname].description
            if hasattr(ncID.variables[varname],'units'):
                var['units'] = ncID.variables[varname].units
            if hasattr(ncID.variables[varname],'classdef'):
                var['classdef'] = ncID.variables[varname].classdef
            if hasattr(ncID.variables[varname],'ancillary_variables'):
                var['ancillary_variables'] = ncID.variables[varname].ancillary_variables
        else:
            var = np.array(ncID.variables[varname][:].copy())
    else:
        if fillnan:
            var = np.array([np.nan])
        else:
            var = np.array([])
    
#    try:
#        var = np.array(ncID.variables[varname][:].copy())
#    except KeyError:
#        print('Warning: Variable ' + varname + ' not found in ' + ncID.filepath())
#        if fillnan:
#            var = np.array([np.nan])
#        else:
#            var = np.array([])
##    var = np.array(ncID.variables[varname].data.copy())
    return var
    
    
def plotprofiles(proflist,varplot=False,time=np.nan,scale='log',fignum=np.nan,cindex=0,loc=1):
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
    """
    colorlist = ['b','g','r','c','m','y','k']
    if np.isnan(fignum):
        plt.figure()
    else:
        plt.figure(fignum)
#    for ai in range(len(proflist)):
    for ai,p0 in enumerate(proflist):
        if isinstance(proflist,dict):
            p1 = proflist[p0].copy()
        else:
            p1 = p0.copy()
#            p1 = proflist[ai].copy()
        if np.isnan(time):
            p1.time_integrate()
            if varplot:
                
                if scale == 'log':
                    plt.semilogx(np.sqrt(p1.profile_variance.flatten()),p1.range_array.flatten(),colorlist[np.mod(ai+cindex,len(colorlist))]+'--',label=p1.label+' std.')
                else:
                    plt.fill_betweenx(p1.range_array,p1.profile.flatten()-np.sqrt(p1.profile_variance.flatten()),p1.profile.flatten()+np.sqrt(p1.profile_variance.flatten()),facecolor=colorlist[np.mod(ai+cindex,len(colorlist))],alpha=0.2)
#                    plt.plot(np.sqrt(p1.profile_variance.flatten()),p1.range_array.flatten(),colorlist[np.mod(ai+cindex,len(colorlist))]+'--',label=p1.label+' std.')
            
            if scale == 'log':
                plt.semilogx(p1.profile.flatten(),p1.range_array.flatten(),colorlist[np.mod(ai+cindex,len(colorlist))]+'-',label=p1.label)
            else:
                plt.plot(p1.profile.flatten(),p1.range_array.flatten(),colorlist[np.mod(ai+cindex,len(colorlist))]+'-',label=p1.label)
            
        else:
            itime = np.argmin(np.abs(p1.time-time))
            if varplot:
                
                if scale == 'log':
                    plt.semilogx(np.sqrt(p1.profile_variance[itime,:]),p1.range_array.flatten(),colorlist[np.mod(ai+cindex,len(colorlist))]+'--',label=p1.label+' std.')
                else:
                    plt.fill_betweenx(p1.range_array,p1.profile[itime,:]-np.sqrt(p1.profile_variance[itime,:]),p1.profile[itime,:]+np.sqrt(p1.profile_variance[itime,:]),facecolor=colorlist[np.mod(ai+cindex,len(colorlist))],alpha=0.2)
#                    plt.plot(np.sqrt(p1.profile_variance[itime,:]),p1.range_array.flatten(),colorlist[np.mod(ai+cindex,len(colorlist))]+'--',label=p1.label+' std.')
            if scale == 'log':
                plt.semilogx(p1.profile[itime,:],p1.range_array.flatten(),colorlist[np.mod(ai+cindex,len(colorlist))]+'-',label=p1.label)
            else:
                plt.plot(p1.profile[itime,:],p1.range_array.flatten(),colorlist[np.mod(ai+cindex,len(colorlist))]+'-',label=p1.label)
            

        
        plt.grid(b=True);
        plt.legend(loc=loc)
        plt.ylabel('Range [m]')
        plt.xlabel(p1.profile_type)
        
def pcolor_profiles(proflist,ylimits=[0,np.nan],tlimits=[np.nan,np.nan],
                    climits=[],plotAsDays=False,scale=[],cmap=[],
                    title_font_size=0,title_add ='',plot_date=False,
                    t_axis_scale=1.0, h_axis_scale=1.0,
                    minor_ticks=0,major_ticks=1.0,plt_kft=False):
    """
    pcolor_profiles(proflist,ylimits=[0,np.nan],tlimits=[np.nan,np.nan],
                    climits=[],plotAsDays=False,scale=[],cmap=[],
                    title_font_size=0,title_add ='',plot_date=False,
                    t_axis_scale=1.0, h_axis_scale=1.0)
    
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
    """
    
    Nprof = np.double(len(proflist))
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
                if plotAsDays:
                    myFmt = mdates.DateFormatter('%b %d %H:%M')
                else:
                    myFmt = mdates.DateFormatter('%H:%M')
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
            
            DateLabel = proflist[ai].StartDate.strftime("%A %B %d, %Y")
            plt.title(title_add+DateLabel + ', ' +proflist[ai].lidar + line_char + proflist[ai].label +' [' + proflist[ai].profile_type + ']',fontsize=title_font_size)
            plt.ylabel('Altitude ['+range_label+']')
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
        else:
            print('pcolor_profiles() is ignoring request to plot ' + proflist[ai].label + '.')
            print('   This profile has no valid data')
    return fig,axL,caxL,imL
        

def pcolor_profiles_official(proflist,ylimits=[],tlimits=[0,24.0],climits=[],plotAsDays=False,scale=[],cmap=[],plot_mask=[],title_font_size=0):
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
    if np.isnan(tlimits[0]):
        tlimits[0] = tmin
    if np.isnan(tlimits[1]):
        tlimits[1] = tmax
    for ai in range(len(proflist)):
        if np.isnan(ylimits[ai][0]):
            ylimits[ai][0] = ymin
        if np.isnan(ylimits[ai][1]):
            ylimits[ai][1] = ymax
        
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
        if has_mask(proflist[ai].profile) and not plot_mask[ai]: # hasattr(proflist[ai].profile,'mask')
            plot_prof = proflist[ai].profile.data.T        
        else:
            plot_prof = proflist[ai].profile.T   
        
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
        plt.title(DateLabel + ', ' +proflist[ai].lidar + line_char + proflist[ai].label +' [' + proflist[ai].profile_type + ']',fontsize=title_font_size)
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

        
def read_WVDIAL_binary(filename,MCSbins):    
    """
    Function for reading WV-DIAL (.dat) binary files
    Function for reading WV-DIAL (.dat) binary files
    filename - .dat file containing WV-DIAL data
    MCSbins - int containing the number of bins the MCS is writing out
    profileData - 2D array containing the time resolved profiles (time along the first dimension,
              range along the second dimension)
    varData - 2D array containing additional variables stuffed into the header
              each row contains the time resolved value.
    """
    f = open(filename,"rb")
    data = np.fromfile(f,dtype=np.double)
    f.close()

    extraVar = 6  # number of extra variables preceeding the profile
    
    data = data.reshape((MCSbins+extraVar,-1),order='F')
    data = data.newbyteorder()
    profileData = data[extraVar:,:]
    varData = data[0:extraVar+1,:]
    
    return profileData,varData
        
def generate_WVDIAL_day_list(startYr,startMo,startDay,startHr=0,duration=0.0,stopYr=0,stopMo=0,stopDay=0,stopHr=24):
    """
    Generates a list of processing Year,Day,Month,Hour to process an arbitrary
    amount chunk of time for WV-DIAL or DLB-HSRL
    The function requires the start year, month, day, (start hour is optional)
    The processing chunk must then either be specified by:
        duration in hours
        stop year, month, day and hour (stopYr,stopMo,stopDay,stopHr) 
            - stop hour does not have to be specified in which case it is the 
            end of the day
    """    
    startDate = datetime.datetime(startYr,startMo,startDay)+datetime.timedelta(hours=startHr)
    
    # start the day processing list based on start times
    YrList = np.array([startYr])    
    MoList = np.array([startMo])
    DayList = np.array([startDay])
    HrList = np.array([[startHr],[24]])
    
    if duration != 0:
        stopDate = startDate+datetime.timedelta(hours=duration)
    else:
        if stopYr == 0:
            stopYr = startDate.year
        if stopMo == 0:
            stopMo = startDate.month
        if stopDay == 0:
            stopDay = startDate.day
        stopDate = datetime.datetime(stopYr,stopMo,stopDay)+datetime.timedelta(hours=stopHr) 
        
    nextday = True
        
    while nextday:
        if stopDate.date() == startDate.date():
            HrList[1,-1] = stopDate.hour+stopDate.minute/60.0+stopDate.second/3600.0
            nextday = False
        else:
            startDate = startDate+datetime.timedelta(days=1)
            YrList = np.concatenate((YrList,np.array([startDate.year])))
            MoList = np.concatenate((MoList,np.array([startDate.month])))
            DayList = np.concatenate((DayList,np.array([startDate.day])))
            HrList = np.hstack((HrList,np.array([[0],[24]])))
        
    return YrList,MoList,DayList,HrList
    
def AerosolBackscatter(MolProf,CombProf,Sonde,negfilter=True):
    """
    Calculate the Aerosol Backscatter Coeffcient LidarProfiles: Molecular and Combined Channels
    Expects a 2d sonde profile that has the same dimensions as the Molecular and combined channels
    
    Set negfilter = False to avoid filtering out negative values
    """
    
    Beta_AerBS = MolProf.copy()

    # calculate backscatter ratio
    BSR = CombProf.profile/MolProf.profile
    
#    Beta_AerBS.profile = (BSR-1)*beta_m_sonde[np.newaxis,:]    # only aerosol backscatter
    Beta_AerBS.profile = (BSR.copy()-1)
    Beta_AerBS.profile_variance = MolProf.profile_variance*(CombProf.profile)**2/(MolProf.profile)**4+CombProf.profile_variance*1/(MolProf.profile)**2
    Beta_AerBS.multiply_prof(Sonde)    
    
    Beta_AerBS.descript = 'Calibrated Measurement of Aerosol Backscatter Coefficient in m^-1 sr^-1'
    Beta_AerBS.label = 'Aerosol Backscatter Coefficient'
    Beta_AerBS.profile_type = '$m^{-1}sr^{-1}$'
    
    if negfilter:
        Beta_AerBS.profile[np.nonzero(Beta_AerBS.profile <= 0)] = 1e-10;
    
    return Beta_AerBS
        
def Calc_AerosolBackscatter(MolProf,CombProf,Temp = np.array([np.nan]),Pres = np.array([np.nan]),negfilter=True,beta_sonde_scale=1.0):
    """
    Calculate the Aerosol Backscatter Coeffcient LidarProfiles: Molecular and Combined Channels
    Temperature and Pressure profiles can be specified.  Otherwise it uses a standard atmosphere.
    Pressure is expected in Pa ( 101325 Pa /atm )
    Temperature is expected in K
    If either is only one element long, the first element will be used to seed the base conditins
    in the standard atmosphere.
    
    Set negfilter = False to avoid filtering out negative values
    """
    
    Beta_AerBS = MolProf.copy()
    
    if np.isnan(Temp).any() or np.isnan(Pres).any():
        if np.isnan(Temp[0]):
            Temp = 300-5*Beta_AerBS.range_array*1e-3
        else:
            Temp = Temp[0]-5*Beta_AerBS.range_array*1e-3
        if np.isnan(Pres[0]):
            Pres = 101325.0*np.exp(-Beta_AerBS.range_array*1e-3/8)
        else:
            Pres = Pres[0]*np.exp(-Beta_AerBS.range_array*1e-3/8)
    # Calculate expected molecular backscatter profile based on temperature and pressure profiles
    beta_m_sonde = 5.45*(550.0e-9/Beta_AerBS.wavelength)**4*1e-32*Pres/(Temp*kB)
   
    # calculate backscatter ratio
    BSR = CombProf.profile/MolProf.profile
    
#    Beta_AerBS.profile = (BSR-1)*beta_m_sonde[np.newaxis,:]    # only aerosol backscatter
    Beta_AerBS.profile = (BSR.copy()-1)
    Beta_AerBS.profile_variance = MolProf.profile_variance*(CombProf.profile)**2/(MolProf.profile)**4+CombProf.profile_variance*1/(MolProf.profile)**2
    Beta_AerBS.diff_geo_overlap_correct(beta_m_sonde,geo_reference='sonde')  # only aerosol backscatter
    
    Beta_AerBS.descript = 'Calibrated Measurement of Aerosol Backscatter Coefficient in m^-1 sr^-1'
    Beta_AerBS.label = 'Aerosol Backscatter Coefficient'
    Beta_AerBS.profile_type = '$m^{-1}sr^{-1}$'
    
    if negfilter:
        Beta_AerBS.profile[np.nonzero(Beta_AerBS.profile <= 0)] = 1e-10;
    
    return Beta_AerBS
        
def Calc_Extinction(MolProf,MolConvFactor = 1.0,Temp = np.array([np.nan]),Pres = np.array([np.nan]),Cam=0.005,AerProf=np.array([0])):


    
    OptDepth = MolProf.copy()
    OptDepth.gain_scale(MolConvFactor)  # Optical depth requires scaling to avoid bias
    
    OptDepth.descript = 'Optical Depth of the altitude Profile starting at lidar base'
    OptDepth.label = 'Optical Depth'
    OptDepth.profile_type = ''
    
    if np.isnan(Temp).any() or np.isnan(Pres).any():
        if np.isnan(Temp[0]):
            Temp = 300-5*OptDepth.range_array*1e-3
        else:
            Temp = Temp[0]-5*OptDepth.range_array*1e-3
        if np.isnan(Pres[0]):
            Pres = 101325.0*np.exp(-OptDepth.range_array*1e-3/8)
        else:
            Pres = Pres[0]*np.exp(-OptDepth.range_array*1e-3/8)
    # Calculate expected molecular backscatter profile based on temperature and pressure profiles
    beta_m_sonde = 5.45*(550.0e-9/OptDepth.wavelength)**4*1e-32*Pres/(Temp*kB)
    
    OptDepth.diff_geo_overlap_correct(1.0/beta_m_sonde)   
    OptDepth.profile_variance = OptDepth.profile_variance*(1/OptDepth.profile)**2
#    Transmission = OptDepth.copy()
    OptDepth.profile = np.log(OptDepth.profile)
    OptDepth.gain_scale(-0.5)
    
    ODmol = np.cumsum(4*np.pi*beta_m_sonde*OptDepth.mean_dR)
    
    # if an aerosol backscatter profile is passed in, use it to help estimate uncertainty
    if hasattr(AerProf,'profile'):
        varCam = (Cam/2.0)**2
        ErrAerVar = 1.0/(beta_m_sonde+Cam*AerProf.profile)**2*(AerProf.profile**2*varCam+Cam**2*AerProf.profile_variance)
        OptDepth.profile_variance = 0.5**2*ErrAerVar
    
    Alpha = OptDepth.copy()   
    Alpha.profile[:,1:] = np.diff(Alpha.profile,axis=1)
    Alpha.profile_variance[:,1:] = Alpha.profile_variance[:,:-1] + Alpha.profile_variance[:,1:]
    Alpha.gain_scale(1.0/Alpha.mean_dR)
    
    Alpha.descript = 'Atmospheric Extinction Coefficient in m^-1'
    Alpha.label = 'Extinction Coefficient'
    Alpha.profile_type = '$m^{-1}$'
    
    return Alpha,OptDepth,ODmol    


def FilterCrossTalkCorrect(MolProf,CombProf,Cam,smart=False):
    """
    FilterCrossTalkCorrect(MolProf,CombProf,Cam)
    remove the aerosol coupling into the molecular channel.
    MolProf - LidarProfile type of Molecular channel
    CombProf - LidarProfile type of Combined channel
    Cam - normalized coupling coefficient of aerosol backscatter to the molecular
        channel.
        Typically < 0.01
    smart - if set to True, corrections will only be applied where the added 
            noise effect is less than that of the coupling error        
        
    Correcting this cross talk has some tradeoffs.
        It improves backscatter coefficient estimates by a percentage coparable to Cam but
        It couples additional noise from the aerosol channel into the molecular channel
        It can introduce more error in cases where one channel is driven into more nonlinearity
            than another
    """
    
    
    
    if smart:
#        DeltaBSR = Cam*(CombProf.profile**2/MolProf.profile**2+(CombProf.profile/MolProf.profile))
        DeltaBSR = Cam*CombProf.profile*(MolProf.profile+CombProf.profile)/(MolProf.profile*(MolProf.profile+Cam*CombProf.profile))
        BSR0 = MolProf.profile_variance*(CombProf.profile)**2/(MolProf.profile)**4+CombProf.profile_variance*1/(MolProf.profile)**2
        BSRcor = (MolProf.profile_variance+Cam**2*CombProf.profile_variance)*(CombProf.profile)**2/(MolProf.profile)**4+CombProf.profile_variance*1/(MolProf.profile)**2
        CorrectMask = np.abs(DeltaBSR) > np.sqrt(BSRcor)-np.sqrt(BSR0)  # mask for locations to apply the correction
        MolProf.profile[CorrectMask] = 1.0/(1-Cam)*(MolProf.profile[CorrectMask]-CombProf.profile[CorrectMask]*Cam)
        MolProf.profile_variance[CorrectMask] = MolProf.profile_variance[CorrectMask]+Cam**2*CombProf.profile_variance[CorrectMask]
    else:    
        MolProf.profile = 1.0/(1-Cam)*(MolProf.profile-CombProf.profile*Cam)
        MolProf.profile_variance = MolProf.profile_variance+Cam**2*CombProf.profile_variance
        
def AerBackscatter_DynamicIntegration(MolProf,CombProf,Temp = np.array([np.nan]),Pres = np.array([np.nan]),num=3,snr_th=1.2,sigma = np.array([1.5,1.0]),beta_sonde_scale=1.0):
    beta0 = Calc_AerosolBackscatter(MolProf,CombProf,Temp=Temp,Pres=Pres,beta_sonde_scale=beta_sonde_scale)
    mask1 = np.log10(beta0.SNR()) > snr_th
    MolNew = MolProf.copy()
    CombNew = CombProf.copy()
    
    MolNew.profile = np.ma.array(MolNew.profile,mask=mask1)    
    CombNew.profile = np.ma.array(CombNew.profile,mask=mask1) 
    
    # avoid applying convolutions to the bottom rows of the profile
    zk,tk,kconv = MolNew.get_conv_kernel(sigma[0],sigma[1])
    mask1[:,:zk.shape[1]] = True
    
    MolNew.conv(sigma[0],sigma[1])
    CombNew.conv(sigma[0],sigma[1])
    
    if num == 0 or mask1.all():
        # merge top profile with new profile
        MolNew.profile[mask1] = MolProf.profile[mask1]
        MolNew.profile_variance[mask1] = MolProf.profile_variance[mask1]
        CombNew.profile[mask1] = CombProf.profile[mask1]
        CombNew.profile_variance[mask1] = CombProf.profile_variance[mask1]
        ProfResZ = np.ones(CombNew.profile.shape)*sigma[1]
        ProfResZ[mask1] = 0
        ProfResT = np.ones(CombNew.profile.shape)*sigma[0]
        ProfResT[mask1] = 0
        
        #return profiles
        return MolNew,CombNew,beta0,ProfResT,ProfResZ
    else:
        M1,C1,b1,Rt,Rz = AerBackscatter_DynamicIntegration(MolNew,CombNew,Temp=Temp,Pres=Pres,num=(num-1),snr_th=snr_th,beta_sonde_scale=beta_sonde_scale)
        # merge with next profile level up
        M1.profile[mask1] = MolProf.profile[mask1]
        M1.profile_variance[mask1] = MolProf.profile_variance[mask1]
        C1.profile[mask1] = CombProf.profile[mask1]
        C1.profile_variance[mask1] = CombProf.profile_variance[mask1]
        betaNew = Calc_AerosolBackscatter(M1,C1,Temp=Temp,Pres=Pres) 
        # update the resolution arrays
        Rt = Rt + sigma[0]
        Rt[mask1] = 0
        Rz = Rz + sigma[1]        
        Rz[mask1] = 0
        
        # return layered molecular profle, combined profile, backscatter profile, time resolution, altitude resolution
        return M1,C1,betaNew,Rt,Rz
        
def Retrieve_Ext_MLE(OptDepth,aer_beta,ODmol,blocksize=8,overlap=0,SNR_th=1.1,lam=np.array([10.0,3.0]),max_iterations=200): 
    if overlap == 0:
        overlap = blocksize/2
    
    Extinction = OptDepth.copy()
    Extinction.descript = 'Atmospheric Extinction Coefficient in m^-1 retrieved with MLE'
    Extinction.label = 'Extinction Coefficient'
    Extinction.profile_type = '$m^{-1}$'
    
    LidarRatio = OptDepth.copy()
    LidarRatio.descript = 'Atmospheric LidarRatio in sr retrieved with MLE'
    LidarRatio.label = 'LidarRatio'
    LidarRatio.profile_type = '$sr$'
    
    OptDepthMLE = OptDepth.copy()
    OptDepthMLE.descript = 'Optical Depth of the altitude Profile starting at lidar base, determined with MLE'
    
    
    # keep a record of which points were set to zero
    x_fit = np.ones(Extinction.profile.shape)    
    # keep a record of the Optical Depth Bias estimate
    ODbiasProf = np.zeros(Extinction.profile.shape[0])
    
    iprof = 0
    while iprof < Extinction.profile.shape[0]:    
        if iprof+blocksize < Extinction.profile.shape[0]:
            aerfit = aer_beta.profile[iprof:iprof+blocksize,:]
            aerfit_std = np.sqrt(aer_beta.profile_variance[iprof:iprof+blocksize,:])
            ODfit = OptDepth.profile[iprof:iprof+blocksize,:]-ODmol[np.newaxis,:]
            ODvar = OptDepth.profile_variance[iprof:iprof+blocksize,:]                
            aerfit[np.nonzero(np.isnan(aerfit))] = 0
            x_invalid = np.nonzero(aerfit < SNR_th*aerfit_std)
            x_fit[x_invalid[0]+iprof,x_invalid[1]] = 0
            sLR,ODbias = Retrieve_Ext_Block_MLE(ODfit,ODvar,aerfit,x_invalid,maxblock=blocksize+1,minLR=OptDepth.mean_dR,lam=lam,max_iterations=max_iterations)
            ODbiasProf[iprof:iprof+blocksize]=ODbias
            if iprof == 0:            
                LidarRatio.profile[iprof:iprof+blocksize,:] = sLR
            else:
                # Average the overlapping chunk
                LidarRatio.profile[iprof:iprof+overlap,:]=0.5*(LidarRatio.profile[iprof:iprof+overlap,:]+sLR[:overlap,:])
                # Fill in the new chunk
                LidarRatio.profile[iprof+overlap:iprof+blocksize,:] = sLR[overlap:,:]
        else:
            aerfit = aer_beta.profile[iprof:,:]
            aerfit_std = np.sqrt(aer_beta.profile_variance[iprof:,:])
            ODfit = OptDepth.profile[iprof:,:]-ODmol[np.newaxis,:]
            ODvar = OptDepth.profile_variance[iprof:,:]                
            aerfit[np.nonzero(np.isnan(aerfit))] = 0
            x_invalid = np.nonzero(aerfit < SNR_th*aerfit_std)
            x_fit[x_invalid[0]+iprof,x_invalid[1]] = 0
            sLR,ODbias = Retrieve_Ext_Block_MLE(ODfit,ODvar,aerfit,x_invalid,maxblock=blocksize+1,minLR=OptDepth.mean_dR,lam=lam,max_iterations=max_iterations)
            if iprof == 0:            
                LidarRatio.profile[iprof:iprof+blocksize,:] = sLR
            else:
                # Average the overlapping chunk
                LidarRatio.profile[iprof:iprof+overlap,:]=0.5*(LidarRatio.profile[iprof:iprof+overlap,:]+sLR[:overlap,:])
                # Fill in the new chunk
                LidarRatio.profile[iprof+overlap:,:] = sLR[overlap:,:]
        print('Completed %d of %d'%(iprof,Extinction.profile.shape[0]))        
        iprof=iprof+(blocksize-overlap)
    
    Extinction.profile = aer_beta.profile*x_fit*LidarRatio.profile
    Extinction.profile[np.nonzero(np.isnan(Extinction.profile))] = 0
    OptDepthMLE.profile= np.cumsum(Extinction.profile,axis=1)
    Extinction.profile = Extinction.profile/OptDepth.mean_dR
    LidarRatio.profile = LidarRatio.profile/OptDepth.mean_dR
    
    return Extinction,OptDepthMLE,LidarRatio,ODbiasProf,x_fit
    
    
def Retrieve_Ext_Block_MLE(ODfit,ODvar,aerfit,x_invalid,maxblock=10,maxLR=1e5,minLR=75.0,lam=np.array([10.0,3.0]),grad_gain=1e-1,max_iterations=200,optout=-1):
    """
    Retrieve_Ext_Block_MLE(ODfit,ODvar,aerfit,x_invalid,maxblock=10,SNR_th=3.3,maxLR=1e5)
    Uses optimization to estimate the extinction coefficent of aerosols and clouds
    Reduces the estimation problem to cases where aerosols are present (determined by SNR threshold-SNR_th)
    ODfit - array of Optical Depth profiles to be fit
    ODvar - uncertainty in those OD measurements in ODfit
    aerfit - array of aerosol backscatter coefficeints
    x_invalid - points in the array that are ignored due to lack of aerosol signal
    maxblock - number of profiles the function is allowed to operate on
    maxLR - maximum Lidar Ratio the optimizor is allowed to use
    minLR - minimum Lidar Ratio the optimizor is allowed to use (typically the range resolution)
    lam - two element array indicating the TV norm sensitivity in altitude and time respectively
    grad_gain - use smaller numbers (<1) to speed up convergence of the optimizer at the cost of accuracy
    max_iterations - maxium iterations allowed from the optimizor
    optout - optimizor output setting
        optout <= 0 : Silent operation
        optout == 1 : Print summary upon completion (default)
        optout >= 2 : Print status of each iterate and summary

    """
    
    rangeIndex = 20  # sets the minimum range where the OD will be used to retrieve extinction    
    
    if ODfit.shape[0] > maxblock:
        print('Max block size exceeded in Retrieve_Ext_Block_MLE')
        print('Current limit: maxblock=%d'%maxblock)
        print('Current number of profiles: %d'%ODfit.shape[0])
        print('Expect issues with x_invalid')
        ODfit = ODfit[:,maxblock,:]
        ODvar = ODvar[:,maxblock,:]
        aerfit = aerfit[:,maxblock,:]
        
#    timeLim = np.array([10.0,15])
#    NumProfs = 8
#    
#    b1 = np.argmin(np.abs(OptDepth.time/3600.0-timeLim[0]))  # 6.3, 15
#    #b2 = np.argmin(np.abs(OptDepth.time/3600-timeLim[1]))  # 6.3, 15
#    b2 = b1+NumProfs
#    
#    aerfit = aer_beta_dlb.profile[b1:b2,:]
#    #aerfit[np.nonzero(aerfit<0)] = 0
#    #aerfit[np.nonzero(np.isnan(aerfit))] = np.nan
#    aerfit_std = np.sqrt(aer_beta_dlb.profile_variance[b1:b2,:])
#    ODfit = OptDepth.profile[b1:b2,:]-ODmol[np.newaxis,:]
#    ODvar = OptDepth.profile_variance[b1:b2,:]
#    
#    sLR = np.zeros(aerfit.shape)
#    aerfit[np.nonzero(np.isnan(aerfit))] = 0
#    x_invalid = np.nonzero(aerfit < 3.3*aerfit_std)
    
    aerfit[x_invalid] = 0.0
    aerfit[:,:rangeIndex] = 0.0
    
#    x0 = np.ones(aerfit.size)*75*50*np.rand.random()
    x0 = 75*(25*np.random.rand(aerfit.size)+50)
    
#    x0 = (-np.log10(aerfit)-2)*1000.0
#    x0[np.nonzero(x0==0)] = 2000.0
#    x0[np.nonzero(x0==np.inf)] = 2000.0
    
    #x0[:] = 1200 #np.log10(1200)
    #x0[1:] = Soln[1:]
    bnds = np.zeros((x0.size,2))
    bnds[:,1] = maxLR
    bnds[:,0] = minLR
    
#    ODbias = -np.nanmin(ODfit[:,:2]) # -np.nanmin(ODfit[:,1:30],axis=1)[:,np.newaxis] # 19.05
    ODbias = -np.nanmean(ODfit[:,rangeIndex:rangeIndex+4])
    
    FitError = lambda x: Fit_LR_2D(x,aerfit,ODfit,ODbias,ODvar,lam=lam)  # substitute x[0] for ODbias to adaptively find bias
    gradFun0 = lambda x: Fit_LR_2D_prime(x,aerfit,ODfit,ODbias,ODvar,lam=lam)*grad_gain
    
    Soln = scipy.optimize.fmin_slsqp(FitError,x0,fprime=gradFun0,bounds=bnds,iter=max_iterations,iprint=optout) #fprime=gradFun0 acc=1e-14 fprime=gradFun0
    
    #sLR[xvalid] = 1200 #np.random.rand(xvalid.shape)*10
    
    #
    #ODsLR = np.cumsum(aerfit*sLR)+ODbias
    sLR = Soln.reshape(aerfit.shape)
#    extinction = aerfit*sLR
#    extinction[np.nonzero(np.isnan(extinction))] = 0
#    ODSoln = np.cumsum(extinction,axis=1)-ODbias
    
    return sLR, ODbias #,extinction,ODSoln,ODbias
#    extinction = extinction/OptDepth.mean_dR  # adjust for bin width    
        
def Fit_LR_2D(x,aerfit,ODfit,ODbias,ODvar,lam=np.array([0,0])):
    """
    Fit_LR_A_2D(x,aerfit,ODfit,ODbias,ODvar,lam=np.array([0,0]))
    Obtain an error estimate for the Optical Depth as a function of retrieved
    Lidar Ratio
    """    
    x2D = x.reshape(aerfit.shape)
    extinction = x2D*aerfit
    extinction[np.nonzero(np.isnan(extinction))] = 0
    ODsLR = np.cumsum(extinction,axis=1)-ODbias    
    ODfit = ODfit
    
    if any(lam > 0):
        deriv = lam[0]*np.nansum(np.abs(np.diff(x2D,axis=0)))+lam[1]*np.nansum(np.abs(np.diff(x2D,axis=1)))
        ErrorOut = np.nansum(0.5*(ODfit-ODsLR)**2/ODvar)+deriv
    else:
        ErrorOut = np.nansum(0.5*(ODfit-ODsLR)**2/ODvar)
    return ErrorOut

def Fit_LR_2D_prime(x,aerfit,ODfit,ODbias,ODvar,lam=np.array([0,0])):

    x2D = x = x.reshape(aerfit.shape)
    extinction = x2D*aerfit
    extinction[np.nonzero(np.isnan(extinction))] = 0
    ODsLR = np.cumsum(extinction,axis=1)-ODbias  


    gradErr = np.cumsum(aerfit,axis=1)*(-(ODfit-ODsLR)/ODvar)   
    gradErr[np.nonzero(np.isnan(gradErr))] = 0
    
    if any(lam >0):
        gradpen = lam[0]*np.sign(np.diff(x2D,axis=0))
        gradpen[np.nonzero(np.isnan(gradpen))] = 0
        gradErr[:-1,:] = gradErr[:-1,:]-gradpen
        gradErr[1:,:] = gradErr[1:,:]+gradpen
        
        gradpen = lam[1]*np.sign(np.diff(x2D,axis=1))
        gradpen[np.nonzero(np.isnan(gradpen))] = 0
        gradErr[:,:-1] = gradErr[:,:-1]-gradpen
        gradErr[:,1:] = gradErr[:,1:]+gradpen   
    
    return gradErr.flatten()  

def Klett_Inv(Comb,RefAlt,Temp = np.array([np.nan]),Pres = np.array([np.nan]),avgRef=False,geo_corr=np.array([]),BGIndex=-50,Nmean=0,kLR=1.0):
    """
    Klett_Inv(Comb,Temp,Pres,RefAlt,avgRef=False,geo_corr=np.array([]),BGIndex=-50)
    
    Accepts a raw photon count profile and estimates the aerosol backscatter
    using the Klett inversion (Klett, Appl. Opt. 1981)
    
    Comb - raw lidar profile.
    RefAlt - altitude in meters used a known reference point
    Temp - Temperature Profile in K - uses standard atmosphere if not provided
    Pres - Pressure Profile in Atm - uses standard atmosphere if not provided
    avgRef - if set to True, the signal value used for the reference altitude
            uses the time average of all data at that altitude
    geo_corr - input for a geometric overlap correction if known
    BGIndex - index for where to begin averaging to determine background levels
    Nmean - if avgRef=False, this sets the smoothing interval for estimating
            the signal at the reference altitude.  Noise at the reference
            altitude couples into the profiles.  
            If Nmean = 0, no smoothing is performed
    kLR - lidar ratio exponent to be used under 
            backscatter = const * extinction^kLR
            defaults to 1.0

    """
    CombK = Comb.copy()
    if not any('Background Subtracted over' in s for s in CombK.ProcessingStatus):
        CombK.bg_subtract(BGIndex)
    if not any('Applied R^2 Range Correction' in s for s in CombK.ProcessingStatus):
        CombK.range_correct()
    if geo_corr.size > 0 and not any('Geometric Overlap Correction' in s for s in CombK.ProcessingStatus):
        CombK.geo_overlap_correct(geo_corr)
    
    iref = np.argmin(np.abs(CombK.range_array-RefAlt))
    
#    kLR = 1.0
    
    if np.isnan(Temp).any() or np.isnan(Pres).any():
        if np.isnan(Temp[0]):
            Temp = 300-5*CombK.range_array*1e-3
        else:
            Temp = Temp[0]-5*CombK.range_array*1e-3
        if np.isnan(Pres[0]):
            Pres = 101325.0*np.exp(-CombK.range_array*1e-3/8)
        else:
            Pres = Pres[0]*np.exp(-CombK.range_array*1e-3/8)
    # Calculate expected molecular backscatter profile based on temperature and pressure profiles
    beta_m_sonde = 5.45*(550.0e-9/CombK.wavelength)**4*1e-32*Pres/(Temp*kB)    
    
    sigKref = (8*np.pi/3.0)*beta_m_sonde[iref]
    
    Sc = np.log(CombK.profile)
    Sc[np.nonzero(np.isnan(Sc))] = 0
    if avgRef:
        Scm = np.nanmean(Sc[:,iref])
    elif Nmean > 0:
        Nmean = 2*np.int(Nmean/2)  # force the conv kernel to be even (for simplicity)
        convKer = np.ones(Nmean)*1.0/Nmean
        GainMask = np.ones(Sc.shape[0])
        GainMask[:Nmean/2] = 1.0*Nmean/np.arange(Nmean/2,Nmean)
        GainMask[-Nmean/2+1:] = 1.0*Nmean/np.arange(Nmean-1,Nmean/2,-1)
        Scm = Sc[:,iref]
        izero = np.nonzero(Scm==0)[0]
        inonzero = np.nonzero(Scm)[0]
        ScmInterp = np.interp(izero,inonzero,Scm[inonzero])
        Scm[izero] = ScmInterp
        Scm = np.convolve(Scm,convKer,mode='same')[:,np.newaxis]
#        plt.figure();
#        plt.plot(Sc[:,iref]);
#        plt.plot(Scm.flatten());
        Scm = Scm*GainMask[:,np.newaxis]
#        plt.plot(Scm.flatten())
        
    else:
        Scm = (Sc[:,iref])[:,np.newaxis]
    
    Beta_AerBS = CombK.copy()
    Beta_AerBS.descript = 'Klett Estimate of Aerosol Backscatter Coefficient in m^-1 sr^-1\n using a %d m Reference Altitude\n '%RefAlt
    Beta_AerBS.label = 'Klett Aerosol Backscatter Estimate'
    Beta_AerBS.profile_type = '$m^{-1}sr^{-1}$' 
    Beta_AerBS.profile_variance = np.zeros(Beta_AerBS.profile_variance.shape)
    
    sigK = np.zeros(Sc.shape)
#    sigK[:,:iref+1] = np.exp((Sc-Scm)[:,:iref+1]/kLR)/(1.0/sigKref+2/kLR*np.cumsum((Sc-Scm)[:,iref::-1]/kLR,axis=1)[:,::-1]*CombK.mean_dR)
    sigK[:,:iref+1] = np.exp((Sc[:,:iref+1]-Scm)/kLR)/(1.0/sigKref+2/kLR*np.fliplr(np.cumsum(np.fliplr((Sc[:,:iref+1]-Scm)/kLR),axis=1))*CombK.mean_dR)
    Beta_AerBS.profile = sigK/(8*np.pi/3.0)-beta_m_sonde[np.newaxis,:]
    
    return Beta_AerBS
    

def Poisson_Thin(y,n=2):
    """
    Poisson_Thin(y,n=2)
    For y an array of poisson random numbers, this function
    builds n statistically independent copies of y
    returns a list of the copies as numpy arrays
    """
    p = 1.0/n
    
    copylist = [];
    for ai in range(n):
        copy = np.zeros(y.shape)
        copy = np.random.binomial(y,p)
        copylist.extend([copy])
    return copylist

def RB_Efficiency(Tlist,T,P,lam,nu=np.array([]),norm=True,max_size=100000):
    """
    Calculates relative transmission efficiency resulting from 
    Rayleigh-Brillouin molecular broadening
    
    Tlist is a list that contains the transmission functions of each channel
        on the same grid as nu (optionally passed in)
    T is the temperature profile in K
    P is the pressure profile in Atm
    lam is the wavelength
    nu - differential frequency basis (center frequency is zero).  If not
        supplied it runs at the native frequency from the PCA analysis
    norm - normalize the spectrum so its intergral (by sum) is zero 
    max_size - the maximum number of data points that will be evaluated at
        a time.  This is used to avoid memory overruns with large files.
        
    """
    i0 = np.arange(0,T.size,max_size)
    i0 = np.concatenate((i0,np.array([T.size])))  # add the last data point to the end
    eta_list = [np.array([])]*len(Tlist)
    for ipt in range(i0.size-1):
        beta_mol_norm = RB_Spectrum(T[i0[ipt]:i0[ipt+1]],P[i0[ipt]:i0[ipt+1]],lam,nu=nu,norm=True)
        for ai in range(len(Tlist)):
            if ipt == 0:
                eta_list[ai] = np.sum(Tlist[ai][:,np.newaxis]*beta_mol_norm,axis=0)
            else:
                eta_list[ai] = np.concatenate((eta_list[ai],np.sum(Tlist[ai][:,np.newaxis]*beta_mol_norm,axis=0)))
            
    return eta_list
#        eta_i2 = np.sum(Ti2[:,np.newaxis]*beta_mol_norm,axis=0)
    #    eta_i2 = eta_i2.reshape(temp.profile.shape)
    #    profs['molecular'].multiply_piecewise(1.0/eta_i2)
    #    profs['molecular'].gain_scale(mol_gain)
        
#        eta_c = np.sum(Tc2[:,np.newaxis]*beta_mol_norm,axis=0)
#    eta_c = eta_c.reshape(temp.profile.shape)
#    profs['combined_hi'].multiply_piecewise(1.0/eta_c)

def RB_Spectrum(T,P,lam,nu=np.array([]),norm=True):
    """
    RB_Spectrum(T,P,lam,nu=np.array([]))
    Obtain the Rayleigh-Brillouin Spectrum of Earth atmosphere
    T - Temperature in K.  Accepts an array
    P - Pressure in Atm.  Accepts an array with size equal to size of T
    lam - wavelength in m
    nu - differential frequency basis.  If not supplied uses native frequency from the PCA
        analysis.
    
    
    """
    # Obtain the y parameters from inputs
    yR = RayleighBrillouin_Y(T,P,lam);

    #Load results from PCA analysis (RB_PCA.m)
    # Loads M, Mavg, x1d os.path.abspath(__file__+'/../../calibrations/')
    filename = os.path.abspath(__file__+'/../DataFiles/') + '/RB_PCA_Params.npz'
    RBpca = np.load(filename);
#    RBpca = np.load('/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/RB_PCA_Params.npz');
    M = RBpca['M']
    Mavg = RBpca['Mavg']
    x = RBpca['x']
    RBpca.close()
    
    # Calculate spectrum based from yR and PCA data
    Spca = Get_RB_PCA(M,Mavg,yR);

    

    if nu.size > 0:
        # if nu is provided, interpolate to obtain requrested frequency grid
        xR = RayleighBrillouin_X(T,lam,nu)
        SpcaI = np.zeros(xR.shape)
   
        for ai in range(T.size):
            SpcaI[:,ai] = np.interp(xR[:,ai],x.flatten(),Spca[:,ai],left=0,right=0)
            if norm:
                SpcaI[:,ai] = SpcaI[:,ai]/np.sum(SpcaI[:,ai])
        return SpcaI
#        S1 = interp1(xR,Spca(:,1),xR(:,1));
        
    else:
        # if nu is not provided, return the spectra and the native x axis
        if norm:
            Spca = Spca/np.sum(Spca,axis=0)[np.newaxis,:]
        return Spca,x

def Get_RB_PCA(M,Mavg,y):
    y = y.flatten()
    yvec = y[np.newaxis,:]**np.arange(M.shape[1])[:,np.newaxis]
    Spect = Mavg+np.dot(M,yvec)
    return Spect

def RayleighBrillouin_Y(T,P,lam):
    """
    y = RayleighBrillouin_Y(T,P,nu,lambda)
    Calculates the RB parameter y for a given 
    Temperature (T in K), Pressure (P in atm) and wavelength (lambda in m)
    """

#    kB = 1.3806504e-23;
#    Mair = 28.95*1.66053886e-27;
    
    k=np.sin(np.pi/2)*4*np.pi/lam;
    v0=np.sqrt(2*kB*T/Mair);
    
    viscosity=17.63e-6;
#    bulk_vis=viscosity*0.73;
#    thermal_cond=25.2e-3;
#    c_int=1.0;
    
    p_pa=P*1.01325e5;
    n0=p_pa/(T*kB);
    
    y=n0*kB*T/(k*v0*viscosity);
    
    return y

def RayleighBrillouin_X(T,lam,nu):
    """
    [x,y] = RayleighBrillouin_XY(T,P,nu,lambda)
    Calculates the RB parameters x and y for a given 
    Temperature (T in K) and wavelength (lambda in m)
    The parameter x is calculated for the supplied frequency grid nu (in Hz)
    If an array of P and T are passed in, x will be a matrix where each
    column is the x values corresponding to the P and T values
    """

#    kB = 1.3806504e-23;
#    Mair = 28.95*1.66053886e-27;
    
    if isinstance(T, np.ndarray):
        k=np.sin(np.pi/2)*4*np.pi/lam;
        v0=np.sqrt(2*kB*T[np.newaxis,:]/Mair);
        x = nu[:,np.newaxis]/(k*v0/(2*np.pi));
    else:
        k=np.sin(np.pi/2)*4*np.pi/lam;
        v0=np.sqrt(2*kB*T/Mair);
        x = nu/(k*v0/(2*np.pi));
        
    return x

def RayleighBrillouin_XY(T,P,lam,nu):
    """
    [x,y] = RayleighBrillouin_XY(T,P,nu,lambda)
    Calculates the RB parameters x and y for a given 
    Temperature (T in K), Pressure (P in atm) and wavelength (lambda in m)
    The parameter x is calculated for the supplied frequency grid nu (in Hz)
    """

    kB = 1.3806504e-23;
    Mair = 28.95*1.66053886e-27;
    
    k=np.sin(np.pi/2)*4*np.pi/lam;
    v0=np.sqrt(2*kB*T/Mair);
    x = nu/(k*v0/(2*np.pi));
    
    viscosity=17.63e-6;
#    bulk_vis=viscosity*0.73;
#    thermal_cond=25.2e-3;
#    c_int=1.0;
    
    p_pa=P*1.01325e5;
    n0=p_pa/(T*kB);
    
    y=n0*kB*T/(k*v0*viscosity);
    
    return x,y

def voigt(x,alpha,gamma,norm=True):
    """
    voigt(x,alpha,gamma)
    Calculates a zero mean voigt profile for spectrum x 
    alpha - Gaussian HWHM
    gamma - lorentzian HWMM
    norm - True: normalize area under the profile's curve
           False:  max value of profile = 1
    
    for instances where the profile is not zero mean substitute x-xmean for x
    
    see scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/
    """
    sigma = alpha / np.sqrt(2*np.log(2))
    if norm:
        v_prof = np.real(wofz((x+1j*gamma)/sigma/np.sqrt(2)))/sigma/np.sqrt(2*np.pi)
        return v_prof
    else:
        v_prof = np.real(wofz((x+1j*gamma)/sigma/np.sqrt(2))) #np.pi/gamma
#        v_prof/ np.real(wofz((0.0+1j*gamma)/sigma/np.sqrt(2))) # normalize so V(x=0) = 1
#        v_prof = (np.pi*sigma/gamma*np.exp(gamma**2/sigma**2)*(1-scipy.special.erf(gamma/sigma)))*v_prof/ np.real(wofz((0.0+1j*gamma)/sigma/np.sqrt(2))) # normalize so V(x=0) = np.pi*sigma/gamma
#        v_prof = (np.pi*sigma/gamma)*v_prof/ np.real(wofz((0.0+1j*gamma)/sigma/np.sqrt(2))) # normalize so V(x=0) = np.pi*sigma/gamma
        return v_prof

def WV_ExtinctionFromHITRAN(nu,TempProf,PresProf,filename='',freqnorm=False,nuLim=np.array([])):
    """
    WV_ExtinctionFromHITRAN(nu,TempProf,PresProf)
    returns a WV extinction profile in m^-1 for a given
    nu - frequency grid in Hz
    TempProf - Temperature array in K
    PresProf - Pressure array in Atm (must be same size as TempProf)
    
    Note that the height of the extinction profile will change based on the
    grid resolution of nu.  
    Set freqnorm=True
    To obtain a grid independent profile to obtain extinction in m^-1 Hz^-1
    
    This function requires access to the HITRAN ascii data:
    '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/WV_HITRAN2012_815_841.txt'
    The data file can be subtituted with something else by using the optional
    filename input.  This accepts a string with a path to the desired file.
    
    If a full spectrum is not needed (nu only represents an on and off line),
    use nuLim to define the frequency limits over which the spectral lines 
    should be included.
    nuLim should be a two element numpy.array
    nuLim[0] = minimum frequency
    nuLim[1] = maximum frequency
    
    """
    nuL = np.mean(nu);
    
    if not filename:
        filename = os.path.abspath(__file__+'/../DataFiles/') + '/WV_HITRAN2012_815_841.txt'
#        filename = '/h/eol/mhayman/PythonScripts/NCAR-LidarProcessing/libraries/WV_HITRAN2012_815_841.txt';
    
    Mh2o = (mH2O*1e-3)/N_A; # mass of a single water molecule, kg/mol
    
    # read HITRAN data
    data = np.loadtxt(filename,delimiter=',',usecols=(0,1,2,3,4,5,6,7,8,9),skiprows=13)
    
    if nuLim.size == 0:
        nuSpan = np.max(nu) - np.min(nu)
        nuLim = np.array([np.min(nu)-nuSpan*0.0,np.max(nu)+nuSpan*0.0])
    
    #Voigt profile calculation
    wn_nu  = nu/c*1e-2; # convert to wave number in cm^-1
    wn_nuL  = nuL/c*1e-2; # convert laser frequency to wave number in cm^-1
    wn_nuLim = nuLim/c*1e-2  # convert frequency span of included lines to wave number in cm^-1
    #Find lines from WNmin to WNmax to calculate voigt profile
    hitran_line_indices = np.nonzero(np.logical_and(data[:,2] > wn_nuLim[0],data[:,2] < wn_nuLim[1]))[0];
#    print('%d'%hitran_line_indices.size)
    
    hitran_T00 = 296;              # HITRAN reference temperature [K]
    hitran_P00 = 1;                # HITRAN reference pressure [atm]
    hitran_nu0_0 = data[hitran_line_indices,2];      # absorption line center wavenumber from HITRAN [cm^-1]
    hitran_S0 = data[hitran_line_indices,3];         # initial linestrength from HITRAN [cm^-1/(mol*cm^-2)]   
    hitran_gammal0 = data[hitran_line_indices,5];    # air-broadened halfwidth at T_ref and P_ref from HITRAN [cm^-1/atm]
#    hitran_gamma_s = data[hitran_line_indices,6];    # self-broadened halfwidth at T_ref and P_ref from HITRAN [cm^-1/atm]
    hitran_E = data[hitran_line_indices,7];          # ground state transition energy from HITRAN [cm^-1]  
    hitran_alpha = data[hitran_line_indices,8];      # linewidth temperature dependence factor from HITRAN
    hitran_delta = data[hitran_line_indices,9];     # pressure shift from HiTRAN [cm^-1 atm^-1]
    
    
    voigt_sigmav_f = np.zeros((np.size(TempProf),np.size(wn_nu)));
    
    dnu = np.mean(np.diff(nu))
    dnu_sign = np.sign(dnu)
    dwn_nu = np.mean(np.diff(wn_nu))
    
    # calculate the absorption cross section at each range
    for ai in range(np.size(TempProf)): 
        #    %calculate the pressure shifts for selected lines as function of range
        hitran_nu0 = hitran_nu0_0+hitran_delta*(PresProf[ai]/hitran_P00); # unclear if it should be Pi/P00
        hitran_gammal = hitran_gammal0*(PresProf[ai]/hitran_P00)*((hitran_T00/TempProf[ai])**hitran_alpha);    # Calculate Lorentz lineweidth at P(i) and T(i)
        hitran_gammad = (hitran_nu0)*((2.0*kB*TempProf[ai]*np.log(2.0))/(Mh2o*c**2))**(0.5);  # Calculate HWHM Doppler linewidth at T(i)                                        ^
        
        # term 1 in the Voigt profile
#        voigt_y = (hitran_gammal/hitran_gammad)*((np.log(2.0))**(0.5));
        voigt_x_on = ((wn_nuL-hitran_nu0)/hitran_gammad)*(np.log(2.0))**(0.5);
    
        # setting up Voigt convolution
#        voigt_t = np.arange(-np.shape(hitran_line_indices)[0]/2.0,np.shape(hitran_line_indices)[0]/2); # set up the integration spectral step size
        
        voigt_f_t = np.zeros((np.size(hitran_line_indices),np.size(wn_nu)));
        for bi in range(voigt_x_on.size):
            voigt_f_t[bi,:] = voigt(wn_nu-hitran_nu0[bi],hitran_gammad[bi],hitran_gammal[bi],norm=True); 
            if freqnorm:
                voigt_f_t[bi,:] = voigt_f_t[bi,:]
#                voigt_f_t[bi,:] = dnu_sign*voigt_f_t[bi,:]/np.trapz(voigt_f_t[bi,:],x=nu);
            else:
                voigt_f_t[bi,:] = voigt_f_t[bi,:]*np.abs(dwn_nu)
#                voigt_f_t[bi,:] = voigt_f_t[bi,:]/np.trapz(voigt_f_t[bi,:]);  # add x=wn_nu to add frequency normalization
    
        # Calculate linestrength at temperature T
        hitran_S = hitran_S0*((hitran_T00/TempProf[ai])**(1.5))*np.exp(1.439*hitran_E*((1.0/hitran_T00)-(1.0/TempProf[ai])));
      
        # Cross section is normalized for spectral integration (no dnu multiplier required)
        voigt_sigmav_f[ai,:] = np.nansum(hitran_S[:,np.newaxis]*voigt_f_t,axis=0);  
    
    
    ExtinctionProf = (1e-4)*voigt_sigmav_f;  # convert to m^2
    return ExtinctionProf
    
def get_beta_m_sonde(Profile,Years,Months,Days,sonde_basepath,interp=False,returnTP=False):
    """
    Returns a 2D array containing the expected molecular backscatter component
    
    StartDateTime - initial data set time (where time=0 starts)
        set by datetime.datetime(Years[0],Months[0],Days[0],0)
    
    StopDateTime - only needed to make sure and get all the right sonde files
        set by datetime.datetime(Years[-1],Months[-1],Days[-1],0)  
    
    interp (False).  If True, interpolates profiles in time
    returnTP (False).  If True, returns teemperature and pressure profiles
        if interp is also True, those profiles are also interpolated
        
    """
    
    # brute force step through and load each month's data
    if np.unique(Years).size==1 and np.unique(Months).size == 1:
        # if we know for sure the data is confined to one month
        Num_Sonde_Iterations = 1
    else:
        Num_Sonde_Iterations = Years.size
    
    for ai in range(Num_Sonde_Iterations):
        if ai == 0:
            YearStr = str(Years[ai])
            if Months[ai] < 10:
                MonthStr = '0'+str(Months[ai])
            else:
                MonthStr = str(Months[ai])
            ### Grab Sonde Data
            sondefilename = '/scr/eldora1/HSRL_data/'+YearStr+'/'+MonthStr+'/sondes.DNR.nc'
            
            print('Accessing %s' %sondefilename)
            #(Man or SigT)
#            f = netcdf.netcdf_file(sondefilename, 'r')
            f = nc4.Dataset(sondefilename,'r')
            TempDat = f.variables['tpSigT'][:].copy()  # Kelvin
            PresDat = f.variables['prSigT'][:].copy()*100.0  # hPa - convert to Pa (or Man or SigT)
            SondeTime = f.variables['relTime'][:].copy() # synoptic time: Seconds since (1970-1-1 00:00:0.0) 
            SondeAlt = f.variables['htSigT'][:].copy()  # geopotential altitude in m
            StatElev = f.variables['staElev'][:].copy()  # launch elevation in m
            f.close()
            
        elif Months[ai-1] != Months[ai]:
            YearStr = str(Years[ai])
            if Months[ai] < 10:
                MonthStr = '0'+str(Months[ai])
            else:
                MonthStr = str(Months[ai])
                
            ### Grab Sonde Data
            sondefilename = '/scr/eldora1/HSRL_data/'+YearStr+'/'+MonthStr+'/sondes.DNR.nc'
            #(Man or SigT)
#            f = netcdf.netcdf_file(sondefilename, 'r')
            f = nc4.Dataset(sondefilename,'r')
            TempDat = np.hstack((TempDat,f.variables['tpSigT'][:].copy()))  # Kelvin
            PresDat = np.hstack((f.variables['prSigT'][:].copy()*100.0))  # hPa - convert to Pa (or Man or SigT)
            SondeTime = np.concatenate((SondeTime,f.variables['relTime'][:].copy())) # synoptic time: Seconds since (1970-1-1 00:00:0.0) 
            SondeAlt = np.hstack((SondeAlt,f.variables['htSigT'][:].copy()))  # geopotential altitude in m
            StatElev = np.concatenate((StatElev,f.variables['staElev'][:].copy()))  # launch elevation in m
            f.close()
        
        
    # set unrealistic sonde data to nans    
    TempDat[np.nonzero(np.logical_or(TempDat < 173.0, TempDat > 373.0))] = np.nan;
    PresDat[np.nonzero(np.logical_or(PresDat < 1.0*100, PresDat > 1500.0*100))] = np.nan;
   
   
    # get sonde time format into the profile time reference    
    StartDateTime = datetime.datetime(Years[0],Months[0],Days[0],0)
    sonde_datetime0 = datetime.datetime(1970,1,1,0,0)
    sonde_datetime = []
    tref = np.zeros(SondeTime.size)
    for ai in range(SondeTime.size):
        # obtain sonde date/time in datetime format
        sonde_datetime.extend([sonde_datetime0+datetime.timedelta(SondeTime[ai]/(3600*24))])
        # calculate the sonde launch time in profile time
        tref[ai] =  (sonde_datetime[ai]-StartDateTime).total_seconds()
        
    # find the sonde index that best matches the time of the profile
    sonde_index_prof = np.argmin(np.abs(Profile.time[np.newaxis,:]-tref[:,np.newaxis]),axis=1)  # profile index for a given sonde launch
    sonde_index = np.argmin(np.abs(Profile.time[np.newaxis,:]-tref[:,np.newaxis]),axis=0)  # sonde index for a given profile
    sonde_index_u = np.unique(sonde_index)  # unique list of sonde launches used to build the profle
    
    
    beta_m_sonde = np.zeros(Profile.profile.shape) 
    TsondeR = np.zeros((Profile.time.size,Profile.range_array.size))
    PsondeR = np.zeros((Profile.time.size,Profile.range_array.size))
    
    if interp:
        # if possible, add additional endpoints to ensure interpolation
        if np.min(sonde_index_u) > 0:
            sonde_index_u = np.concatenate((np.array([np.min(sonde_index_u)-1]),sonde_index_u))
        if np.max(sonde_index_u) < SondeTime.size-1:
            sonde_index_u = np.concatenate((sonde_index_u,np.array([np.max(sonde_index_u)+1])))

        Tsonde = np.zeros((sonde_index_u.size,Profile.range_array.size))
        Psonde = np.zeros((sonde_index_u.size,Profile.range_array.size))
        
        for ai in range(sonde_index_u.size):
            Tsonde[ai,:] = np.interp(Profile.range_array,SondeAlt[sonde_index_u[ai],:]-StatElev[sonde_index_u[ai]],TempDat[sonde_index_u[ai],:])
            Psonde[ai,:] = np.interp(Profile.range_array,SondeAlt[sonde_index_u[ai],:]-StatElev[sonde_index_u[ai]],PresDat[sonde_index_u[ai],:])
            
        for ai in range(Profile.range_array.size):
            TsondeR[:,ai] = np.interp(Profile.time,tref[sonde_index_u],Tsonde[:,ai])
            PsondeR[:,ai] = np.interp(Profile.time,tref[sonde_index_u],Psonde[:,ai])
            beta_m_sonde[:,ai] = 5.45*(550.0e-9/Profile.wavelength)**4*1e-32*PsondeR[:,ai]/(TsondeR[:,ai]*kB)
    else:
    
        for ai in range(sonde_index_u.size):
            Tsonde = np.interp(Profile.range_array,SondeAlt[sonde_index_u[ai],:]-StatElev[sonde_index_u[ai]],TempDat[sonde_index_u[ai],:])
            Psonde = np.interp(Profile.range_array,SondeAlt[sonde_index_u[ai],:]-StatElev[sonde_index_u[ai]],PresDat[sonde_index_u[ai],:])
            beta_m_sondes0 = 5.45*(550.0e-9/Profile.wavelength)**4*1e-32*Psonde/(Tsonde*kB)
            ifill = np.nonzero(sonde_index==sonde_index_u[ai])[0]
            TsondeR[ifill,:] = Tsonde
            PsondeR[ifill,:] = Psonde
            beta_m_sonde[ifill,:] = beta_m_sondes0[np.newaxis,:]*np.ones((ifill.size,beta_m_sondes0.size))
    beta_mol = Profile.copy()
    beta_mol.profile = beta_m_sonde.copy()
    beta_mol.profile_variance = (beta_mol.profile*0.01)**2  # force SNR of 100 in sonde profile.
    beta_mol.ProcessingStatus = []     # status of highest level of lidar profile - updates at each processing stage
    beta_mol.lidar = 'sonde'
    
    beta_mol.diff_geo_Refs = ['none']           # list containing the differential geo overlap reference sources (answers: differential to what?)
    beta_mol.profile_type =  '$m^{-1}sr^{-1}$'
    
    beta_mol.bg = np.zeros(beta_mol.bg.shape) # profile background levels
    
    beta_mol.descript = 'Sonde Estimated Molecular Backscatter Coefficient in m^-1 sr^-1'
    beta_mol.label = 'Molecular Backscatter Coefficient'
    
    if not returnTP:
        return beta_mol,tref,sonde_index_u
    
    else:
    
        temp = Profile.copy()
        temp.profile = TsondeR.copy()
        temp.profile_variance = (temp.profile*0.01)**2  # force SNR of 100 in sonde profile.
        temp.ProcessingStatus = []     # status of highest level of lidar profile - updates at each processing stage
        temp.lidar = 'sonde'
        
        temp.diff_geo_Refs = ['none']           # list containing the differential geo overlap reference sources (answers: differential to what?)
        temp.profile_type =  '$K$'
        
        temp.bg = np.zeros(temp.bg.shape) # profile background levels
        
        temp.descript = 'Sonde Measured Temperature in K'
        temp.label = 'Temperature'
        
        pres = Profile.copy()
        pres.profile = PsondeR.copy()
        pres.profile_variance = (pres.profile*0.01)**2  # force SNR of 100 in sonde profile.
        pres.ProcessingStatus = []     # status of highest level of lidar profile - updates at each processing stage
        pres.lidar = 'sonde'
        
        pres.diff_geo_Refs = ['none']           # list containing the differential geo overlap reference sources (answers: differential to what?)
        pres.profile_type =  '$Pa$'
        
        pres.bg = np.zeros(temp.bg.shape) # profile background levels
        
        pres.descript = 'Sonde Measured Pressure in Pa'
        pres.label = 'Pressure'
    
        return beta_mol,tref,sonde_index_u,temp,pres,sonde_index
    
def get_beta_m_model(Profile,base_temp,base_pres,interp=False,returnTP=False):
    """
    Uses standard atmosphere and base station to obtain a model of the
    molecular backscatter, temperature and pressure
    
    Currently requires base_temp and base_pres are supplied with the same
    time dimension as the actual profile.
    
    expects temperature in K
    expects pressure in atm
    """    
    
    
    TsondeR = base_temp[:,np.newaxis]-0.0065*Profile.range_array[np.newaxis,:]
    PsondeR = base_pres[:,np.newaxis]*(base_temp[:,np.newaxis]/TsondeR)**(-5.5)/9.86923e-6  # give pressure in Pa
    beta_m_sonde = 5.45*(550.0e-9/Profile.wavelength)**4*1e-32*PsondeR/(TsondeR*kB)
    
    # mismatch in gridding can occur when Profiles are trimmed for consistency    
    if beta_m_sonde.shape[0] > Profile.time.size:
        beta_m_sonde = beta_m_sonde[:Profile.time.size,:]
        TsondeR = TsondeR[:Profile.time.size,:]
        PsondeR = PsondeR[:Profile.time.size,:]
    elif beta_m_sonde.shape[0] < Profile.time.size:
        tdif = Profile.time.size-beta_m_sonde.shape[0]
        beta_m_sonde = np.hstack((beta_m_sonde,beta_m_sonde[-1,:][np.newaxis,:]*np.ones(tdif,Profile.range_array.size)))
        TsondeR = np.hstack((TsondeR,TsondeR[-1]*np.ones(tdif,Profile.range_array.size)))
        PsondeR = np.hstack((PsondeR,PsondeR[-1]*np.ones(tdif,Profile.range_array.size)))
        
    
    beta_mol = Profile.copy()
    beta_mol.profile = beta_m_sonde.copy()
    beta_mol.profile_variance = (beta_mol.profile*0.01)**2  # force SNR of 100 in sonde profile.
    beta_mol.ProcessingStatus = []     # status of highest level of lidar profile - updates at each processing stage
    beta_mol.lidar = 'ideal atmosphere'
    
    beta_mol.diff_geo_Refs = ['none']           # list containing the differential geo overlap reference sources (answers: differential to what?)
    beta_mol.profile_type =  '$m^{-1}sr^{-1}$'
    
    beta_mol.bg = np.zeros(beta_mol.bg.shape) # profile background levels
    
    beta_mol.descript = 'Ideal Atmosphere Molecular Backscatter Coefficient in m^-1 sr^-1'
    beta_mol.label = 'Molecular Backscatter Coefficient'
    
    if not returnTP:
        return beta_mol
    
    else:
    
        temp = Profile.copy()
        temp.profile = TsondeR.copy()
        temp.profile_variance = (temp.profile*0.1)**2  # force SNR of 10 in sonde profile.
        temp.ProcessingStatus = []     # status of highest level of lidar profile - updates at each processing stage
        temp.lidar = 'ideal atmosphere'
        
        temp.diff_geo_Refs = ['none']           # list containing the differential geo overlap reference sources (answers: differential to what?)
        temp.profile_type =  '$K$'
        
        temp.bg = np.zeros(temp.bg.shape) # profile background levels
        
        temp.descript = 'Ideal Atmosphere Temperature in K'
        temp.label = 'Temperature'
        
        pres = Profile.copy()
        pres.profile = PsondeR.copy()
        pres.profile_variance = (pres.profile*0.1)**2  # force SNR of 10 in sonde profile.
        pres.ProcessingStatus = []     # status of highest level of lidar profile - updates at each processing stage
        pres.lidar = 'ideal atmosphere'
        
        pres.diff_geo_Refs = ['none']           # list containing the differential geo overlap reference sources (answers: differential to what?)
        pres.profile_type =  '$Pa$'
        
        pres.bg = np.zeros(temp.bg.shape) # profile background levels
        
        pres.descript = 'Ideal Atmosphere Pressure in Pa'
        pres.label = 'Pressure'
    
        return beta_mol,temp,pres

def get_beta_m(temp,pres,wavelength):
    """
    Calculate molecular backscatter given a temperature and pressure profile
    and a scattering wavelength.
    Inputs:
    temp - lidar profile of temeprature in K
    pres - lidar profile of pressure in Pa
    wavelength - incident wavelength in m
    
    returns beta_mol, a lidar profile of the molecular backscatter coefficient
    
    """
    beta_mol = temp.copy()
    beta_mol.diff_geo_Refs = ['none']           # list containing the differential geo overlap reference sources (answers: differential to what?)
    beta_mol.profile_type =  '$m^{-1}sr^{-1}$'
    beta_mol.descript = 'Ideal Atmosphere Molecular Backscatter Coefficient in m^-1 sr^-1'
    beta_mol.label = 'Molecular Backscatter Coefficient'
    
    if pres.profile_type == '$Pa$' or pres.profile_type == 'Pa':
        beta_mol.profile = 5.45*(550.0e-9/wavelength)**4*1e-32*pres.profile/(temp.profile*kB)
    elif pres.profile_type == 'atm.' or pres.profile_type == '$atm.$':
        beta_mol.profile = 5.45*(550.0e-9/wavelength)**4*1e-32*pres.profile/9.86923e-6/(temp.profile*kB)
    else:
        beta_mol.profile = np.ones(temp.profile.shape)
        print('In LidarProfileFunctions.get_beta_m(temp,pres,wavelength)')
        print('  Unrecogized pressure units: '+ pres.profile_type)
        print('  Expecting $Pa$ or atm.')
        beta_mol.profile_type =  'unitless'
        beta_mol.descript = 'Unassigned Molecular Backscatter data due to unexpected pressure units'
        beta_mol.label = 'Unassigned Molecular Backscatter Coefficient'

    beta_mol.profile_variance = (beta_mol.profile*0.01)**2  # force SNR of 100 in sonde profile.
    beta_mol.ProcessingStatus = []     # status of highest level of lidar profile - updates at each processing stage

    beta_mol.bg = np.zeros(beta_mol.bg.shape) # profile background levels
    
    return beta_mol

def get_calval(data_date,json_data,cal_name,cond = [],returnlist=['value']):
    """
    retrieve the best value for a calibration value based on the date
    of the data under processing
    
    data_date - datetime of the processed data
    json_data - calibration data loaded from the json file
    cal_name - string giving the library look-up for the calibration
    cond - additional conditions.  e.g. use [['diff_geo','=','none']] to require
        the Molecular Gain has no diff_geo associated with it.
        [['diff_geo','!=','none']] means the diff_geo cannot equal 'none'
        cond only accepts an 'equal' or 'not equal' argument.  It can't handle
        'greater than' or 'less than' and will treat them as 'not equal'.
        Handles multiple conditions by adding more to the list.  e.g.
        [['diff_geo','!=','none'],['RB Corrected','=','True']].
        Multiple conditions are treated as AND conditions where they all
        must be true.
    returnlist - list of the fields to be returned.  defaults to just the
        value of the calibration variable
    """
    
    
    # For better handling use try with except KeyError:
    cal_name = cal_name.replace(' ','_')  # replace spaces with underscores
                                            # json names were updated to accomidate
                                            # matlab reader
    
    if cal_name in json_data:    
        # replace spaces in requested variables with underscores    
        for ai,val in enumerate(returnlist):
            returnlist[ai] = val.replace(' ','_') 
        for bi in range(len(cond)):
            for ai,val in enumerate(cond[bi]):
                cond[bi][ai] = val.replace(' ','_') 
        
        min_time = 1e15
        min_abs_time = min_time
        cal_index = -1
        # for some calibrations, we need to check if the change is 'abrupt' or 'gradual'
        test_not_equal = []
        if len(cond) > 0:
            for c_ind in range(len(cond)):
                if cond[c_ind][1] == '=':
                    test_not_equal.extend([False])
    #                test_not_equal = False
                else:
                    test_not_equal.extend([True])
    #                test_not_equal = True
        else:
            test_not_equal.extend([False])
    #        test_not_equal = False
                
        # check for changes that have the 'abrupt' 'gradual' designation.
    #    if cal_name == 'Molecular_Gain' or cal_name == 'Geo_File_Record' or cal_name == 'Location':        
        if 'change_type' in json_data[cal_name][0].keys():
            for ai in range(len(json_data[cal_name])):
                if len(cond) == 0:
                    cond_true = True
                else:
                    cond_true = True
                    for c_ind in range(len(cond)):
                        if not boolean_not_switch(json_data[cal_name][ai][cond[c_ind][0]]==cond[c_ind][2],test_not_equal[c_ind]):
                            cond_true = False
    #            elif boolean_not_switch(json_data[cal_name][ai][cond[0]]==cond[2],test_not_equal):
    #                cond_true = True
    #            else:
    #                cond_true = False
                
                if cond_true:
                    time_diff = (data_date-json_str_to_datetime(json_data[cal_name][ai]['date'])).total_seconds()
                    if json_data[cal_name][ai]['change_type']  == 'abrupt' :
                        if time_diff >= 0 and time_diff < min_time:
                            cal_index = ai
                            min_time = time_diff
                    elif np.abs(time_diff) < min_abs_time:
                        cal_index = ai
                        min_time = np.abs(time_diff)
                    if np.abs(time_diff) < min_abs_time:
                        min_abs_time = np.abs(time_diff)
        else:
            for ai in range(len(json_data[cal_name])):
                if len(cond) == 0:
                    cond_true = True
                else:
                    cond_true = True
                    for c_ind in range(len(cond)):
                        if not boolean_not_switch(json_data[cal_name][ai][cond[c_ind][0]]==cond[c_ind][2],test_not_equal[c_ind]):
                            cond_true = False
    #            elif boolean_not_switch(json_data[cal_name][ai][cond[0]]==cond[2],test_not_equal):
    #                cond_true = True
    #            else:
    #                cond_true = False
                
                if cond_true:
                    time_diff = (data_date-json_str_to_datetime(json_data[cal_name][ai]['date'])).total_seconds()
                    if time_diff < min_time and time_diff >= 0:
                        cal_index = ai
                        min_time = np.abs(time_diff)  
        
        return_data = []
        for ai in range(len(returnlist)):
            try:
                return_data.extend([json_data[cal_name][cal_index][returnlist[ai]]])
            except IndexError:
                print('No valid data for ' +cal_name+ ' found')
            except KeyError:
                print('No valid data for ' +cal_name+ ' found during this time period')
                        
        return return_data
    else:
        print('No entry for ' +cal_name+' found')
        return []

def json_str_to_datetime(json_str):
    try:
        json_datetime = datetime.datetime.strptime(json_str,'%d-%b-%Y, %H:%M')
    except ValueError:
        json_datetime = datetime.datetime.strptime(json_str,'%d-%b-%Y %H:%M')
    return json_datetime
    
def boolean_not_switch(in_bool,switch):
    """
    Switch that determines whether to not an expression
    if switch == True
    in_bool is not-ed
    
    if switch == False
    in_bool is unchanged
    """
    if switch:
        ret_bool = not in_bool
    else:
        ret_bool = in_bool
    return ret_bool



def Estimate_Mol_Gain(BSR,iKeep=np.array([np.nan]),mol_gain=1.0,alt_lims=[2000,6500],label='',plot=True):
    """
    Estimate_Mol_Gain(mol_prof,comb_prof,iKeep=np.array([]),mol_gain=1.0)
    
    Statistically estimates the gain multiplier needed for the molecular
    channel.
    
    BSR - lidar profile of the backscatter ratio
    iKeep - time indices for the data to be used in the analysis
        (this can be used to define the cal for a telescope pointing direction)
    mol_gain - if provided the result supplies the actual gain setting needed
        otherwise the current gain needs to be scaled by the solution output
        by the function
    alt_lims - limits the altitude range for searching for the molecular gain
    label is a string for identifying the analysis.  E.g. "Telescope Up" to 
        specify the analysis is categorized based on telescope direction
    plot - boolian specifying if the analysis should be plotted
    """
    # This segment estimates what the molecular gain should be 
    # based on a histogram minimum in BSR over the loaded data
    
    if len(iKeep == 1) and np.isnan(iKeep[0]):
        BSRprof = BSR.profile.flatten()
        BSRalt = (np.ones(BSR.profile.shape)*BSR.range_array[np.newaxis,:]).flatten()
    else:
        BSRprof = BSR.profile[iKeep,:].flatten()
        BSRalt = (np.ones(BSR.profile[iKeep,:].shape)*BSR.range_array[np.newaxis,:]).flatten()
        
    if len(iKeep > 0):
        BSRalt = np.delete(BSRalt,np.nonzero(np.isnan(BSRprof)))
        BSRprof = np.delete(BSRprof,np.nonzero(np.isnan(BSRprof)))
    
        bbsr = np.linspace(0,4,400)
    
        balt = np.concatenate((BSR.range_array-BSR.mean_dR/2,BSR.range_array[-1:]+BSR.mean_dR/2))
    
        """
        perform analysis by altitude
        """
        
        hbsr = np.histogram2d(BSRprof,BSRalt,bins=[bbsr,balt])
        
        i_hist_median = np.argmax(hbsr[0],axis=0)
        
        # check for selecting end points
        iadj = np.nonzero(i_hist_median >= hbsr[0].shape[0]-1)
        i_hist_median[iadj] = hbsr[0].shape[0] - 2
        
        # check for selecting start points
        iadj = np.nonzero(i_hist_median == 0)
        i_hist_median[iadj] = 1
        
        iset = np.arange(hbsr[0].shape[1])
        dh1 = hbsr[0][i_hist_median,iset]-hbsr[0][i_hist_median-1,iset]
        dh2 = hbsr[0][i_hist_median+1,iset]-hbsr[0][i_hist_median,iset]
        dbsr = np.mean(np.diff(bbsr))
        bsr1 = bbsr[i_hist_median]
    #    bsr2 = bbsr[i_hist_median+1]
        
        m_0 = (dh2-dh1)/dbsr
        b_0 = dh1-m_0*bsr1
        
        Nsm = 20  # number of bins to smooth over
        hist_median = (-b_0)/m_0
        hist_med_sm = np.convolve(hist_median,np.ones(Nsm)*1.0/Nsm,mode='same')
        
        
        
        i_alt_lim = np.nonzero(np.logical_and(balt > alt_lims[0],balt < alt_lims[1]))[0]
        
        mol_gain_adj = np.nanmin(hist_med_sm[i_alt_lim])
        if np.isnan(mol_gain_adj):
            mol_gain_adj = np.nanmin(hist_med_sm)
            if np.isnan(mol_gain_adj):
                i_gain = 0
                print('\nNo Altitude Reference Found')
                plot_point = 1.0
                text_str = 'No Altitude Reference'
            else:
                i_gain = np.nanargmin(hist_med_sm)
                print('\nAltitude Reference: %f m' %(balt[i_gain]))
                plot_point = balt[i_gain]
                text_str = 'Altitude Reference: %f m' %(balt[i_gain])
        else:
            i_gain = np.nanargmin(hist_med_sm[i_alt_lim])
            print('\nAltitude Reference: %f m' %(balt[i_alt_lim])[i_gain])
            plot_point = (balt[i_alt_lim])[i_gain]
            text_str = 'Altitude Reference: %f m' %(balt[i_alt_lim])[i_gain]

        
#        print('\nAltitude Reference: %f m' %(balt[i_alt_lim])[i_gain])
        print('Current Molecular ('+label+') Gain: %f'%mol_gain)
        print('Suggested Molecular ('+label+') Gain: %f\n'%(mol_gain*mol_gain_adj))
        
        if plot:
            plt.figure()
            plt.pcolor(bbsr,balt,hbsr[0].T)
#            plt.plot(hist_median,balt[1:],'r--')
            plt.plot(hist_med_sm,balt[1:],'g--')
            plt.plot(mol_gain_adj,plot_point+BSR.mean_dR,'go',linewidth=2,markeredgecolor='w',markersize=6.0)
            plt.xlabel('BSR')
            plt.ylabel('Altitude [m]')
            plt.title(label)
#            text_str = 'Altitude Reference: %f m' %(balt[i_alt_lim])[i_gain]
            text_str = text_str+'\nCurrent Molecular Gain: %f'%mol_gain
            text_str = text_str+'\nSuggested Molecular Gain: %f\n'%(mol_gain*mol_gain_adj)
#            plt.text(0.95,0.95,verticalalignment='top',horizontalalignment='right',transform=ax.transAxes,color='white')
#            plt.grid(b=True)
            plt.plot(np.ones(2),np.array([balt[0],balt[-1]]),'w--')
            plt.text(0.95*(np.max(bbsr)-np.min(bbsr))+np.min(bbsr),0.95*(np.max(balt)-np.min(balt))+np.min(balt),text_str,color='white',fontsize=8,verticalalignment='top',horizontalalignment='right')
        
        
    else:
        mol_gain_adj = 1.0
        print('\nNo data for ' + label + ' molecular gain estimate\n')
        
    return mol_gain_adj
    
def LogProfile(profile,order=3):
    """
    Takes the natural log of a profile using higher order moments to estimate
    the mean value of ln(x) from x.
    order is the order used for mean value and variance estimation based on
    Mekid and Vaja doi:10.1016/jmeasurement.2007.07.004
    maximum value is 3, and minimum value is 1
    1 is the same as standard propagation of error
    returns a profile with the log of the profile and associated variance
    """
    
    retprof = profile.copy()    
    
    order = min([order,3])
    order = max([order,1])    
    
    # first order mean and variance
    Mean1 = np.log(profile.profile)
    var1 = profile.profile_variance/profile.profile**2
        
    if order >= 2:
        # recover original photon counts to calculate higher order moments
        Phcnts = (retprof.profile + retprof.bg[:,np.newaxis])*profile.NumProfList[:,np.newaxis]
        # calculate skewness and kurtosis
        skewPh = 1.0/np.sqrt(Phcnts)  # skewness of photon counts
        kurtPh = 1.0/Phcnts             # kurtosis of photon counts
        # calculate mean and variance from second order analysis
        Mean2 = Mean1-0.5*profile.profile_variance/(profile.profile**2)
        var2 = var1 - skewPh*(1.0/profile.profile**3)*profile.profile_variance**(3.0/2) \
            +(kurtPh-1)/4.0*(1/profile.profile**4)*profile.profile_variance**2 
    else:
        # return first order mean value and variance
        retprof.profile = Mean1
        retprof.profile_variance = var1
        retprof.cat_ProcessingStatus('ln of Profile with order 1 estimation')
        retprof.profile_type = 'ln Photon Counts'
        return retprof
    
    if order >= 3:
        # calculate higher order moments for third order analysis
        mom5 = Phcnts+10*Phcnts**2  # fifth order moment needed for 3rd order variance
        mom6 = Phcnts+25*Phcnts**2+15*Phcnts**3  # sixth order moment needed for 3rd order variance
        # calculate mean and variance from third order analysis
        Mean3 = Mean2+profile.profile_variance**(3.0/2)*skewPh/(3.0*profile.profile**3)
        var3 = var2 + kurtPh/3*(2.0/profile.profile**4)*profile.profile_variance**2 \
            - 1.0/3*(1.0/profile.profile**5)*(mom5-skewPh*profile.profile_variance**(5.0/2)) \
            + 1.0/9*(1.0/profile.profile**6)*(mom6-skewPh**2*profile.profile_variance**3)
        # return third order mean value and variance
        retprof.profile = Mean3
        retprof.profile_variance = var3
        retprof.cat_ProcessingStatus('ln of Profile with order 3 estimation')
        retprof.profile_type = 'ln Photon Counts'
        return retprof
    else:
        # return second order mean value and variance
        retprof.profile = Mean2
        retprof.profile_variance = var2
        retprof.cat_ProcessingStatus('ln of Profile with order 2 estimation')
        retprof.profile_type = 'ln Photon Counts'
        return retprof
        
def DivideProfile(profile,order=3):
    """
    Takes the natural log of a profile using higher order moments to estimate
    the mean value of 1/x from x.
    order is the order used for mean value and variance estimation based on
    Mekid and Vaja doi:10.1016/jmeasurement.2007.07.004
    maximum value is 3, and minimum value is 1
    1 is the same as standard propagation of error
    returns a profile with the log of the profile and associated variance
    """
    
    retprof = profile.copy()    
    
    order = min([order,3])
    order = max([order,1])    
    
    # first order mean and variance
    Mean1 = 1/(profile.profile)
    var1 = profile.profile_variance/profile.profile**4
        
    if order >= 2:
        # recover original photon counts to calculate higher order moments
        Phcnts = (retprof.profile + retprof.bg[:,np.newaxis])*profile.NumProfList[:,np.newaxis]
        # calculate skewness and kurtosis
        skewPh = 1.0/np.sqrt(Phcnts)  # skewness of photon counts
        kurtPh = 1.0/Phcnts             # kurtosis of photon counts
        # calculate mean and variance from second order analysis
        Mean2 = Mean1-0.5*profile.profile_variance/(profile.profile**2)
        var2 = var1 - skewPh*(1.0/profile.profile**3)*profile.profile_variance**(3.0/2) \
            +(kurtPh-1)/4.0*(1/profile.profile**4)*profile.profile_variance**2 
    else:
        # return first order mean value and variance
        retprof.profile = Mean1
        retprof.profile_variance = var1
        retprof.cat_ProcessingStatus('ln of Profile with order 1 estimation')
        retprof.profile_type = 'ln Photon Counts'
        return retprof
    
    if order >= 3:
        # calculate higher order moments for third order analysis
        mom5 = Phcnts+10*Phcnts**2  # fifth order moment needed for 3rd order variance
        mom6 = Phcnts+25*Phcnts**2+15*Phcnts**3  # sixth order moment needed for 3rd order variance
        # calculate mean and variance from third order analysis
        Mean3 = Mean2+profile.profile_variance**(3.0/2)*skewPh/(3.0*profile.profile**3)
        var3 = var2 + kurtPh/3*(2.0/profile.profile**4)*profile.profile_variance**2 \
            - 1.0/3*(1.0/profile.profile**5)*(mom5-skewPh*profile.profile_variance**(5.0/2)) \
            + 1.0/9*(1.0/profile.profile**6)*(mom6-skewPh**2*profile.profile_variance**3)
        # return third order mean value and variance
        retprof.profile = Mean3
        retprof.profile_variance = var3
        retprof.cat_ProcessingStatus('ln of Profile with order 3 estimation')
        retprof.profile_type = 'ln Photon Counts'
        return retprof
    else:
        # return second order mean value and variance
        retprof.profile = Mean2
        retprof.profile_variance = var2
        retprof.cat_ProcessingStatus('ln of Profile with order 2 estimation')
        retprof.profile_type = 'ln Photon Counts'
        return retprof


def dist_from_latlon(latitude,longitude,lat0=np.nan,lon0=np.nan):
    """
    Calculates an E/W and N/S distance (in meters) between a base latitude and longitude
    and an array of latitude and longitude
    latitude - array containing the the disired latitude points for conversion to distance
    longitude - array containing the desired longitude points for conversion to distance
    lat0 - latitude reference for distance.  If not assigned, it will use the mean of latitude
    lon0 - longitude reference for distance.  If not assigned, it will use the mean of longitude
    
    returns dist_x, dist_y
    dist_x - distance in E/W direction
    dist_y - distance in N/S direction
    """
    if np.isnan(lat0):
        lat0 = np.nanmean(latitude)
    if np.isnan(lon0):
        lon0 = np.nanmean(longitude)
    
    a_dist = np.sin(np.pi*(latitude-lat0)/360.0)**2+np.cos(latitude*np.pi/180)*np.cos(lat0*np.pi/180)*np.sin(np.pi*(longitude-lon0)/360.0)**2
    c_dist = 2*np.arctan2(np.sqrt(a_dist),np.sqrt(1-a_dist))
    dist = 6371e3*c_dist
    bearing = np.arctan2(np.sin(np.pi*(longitude-lon0)/180.0)*np.cos(latitude*np.pi/180),np.cos(lat0*np.pi/180)*np.sin(latitude*np.pi/180)-np.sin(lat0*np.pi/180)*np.cos(latitude*np.pi/180)*np.cos(np.pi*(longitude-lon0)/180.0))
    dist_x = np.sin(bearing)*dist
    dist_y = np.cos(bearing)*dist

    return dist_x, dist_y
    
def depol_to_ratio(lidar_prof):
    """
    Converts a depolarization (d) LidarProfile to a linear depolarization ratio
    lidar_prof - LidarProfile of depolarization (d)
    """
    
    depR = lidar_prof.copy()
    
    depR.profile = lidar_prof.profile/(2-lidar_prof.profile)
    depR.profile_variance = lidar_prof.profile_variance*4*(lidar_prof.profile-1)**2/(lidar_prof.profile-2)**4
    depR.descript = 'Linear depolarization ratio assuming particles are randomly oriented'
    depR.profile_type = 'unitless'
    depR.label = 'Linear Depolarization Ratio'
    depR.cat_ProcessingStatus('Converted to linear depolarization ratio')
    
    return depR
    
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
#    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def optimize_sg_raw(profile0,axis=(1,),full=False,order=None,window=None,
                    range_lim=None,bg_subtract=False,scale=None,AdjCounts=False,
                    return_metrics=False,return_sol=False):
    """
    optimize_sg_raw(profile0,axis=(1,),full=False,order=None,window=None,
                    range_lim=None,bg_subtract=False,scale=None,AdjCounts=False,
                    return_metrics=False,return_sol=False)
    
    for a photon count profile, estimate the optimal savitzky-golay filter 
    parameters
    
    profile0 - profile to optimize against.  This should consist of raw, true
        photon counts (no averaging or background subtracting)
    
    axis - tuple of axes to filter in desired order
    
    full - True: find the optimal filter for the entire profile
           False: find the optimal filter for each individual profile
    
    order - [[min,max]] range to evaluate the SG polynomial order for each axis
    
    window - [[min,max]] range over which to evaluate the SG window size for each axis
    
    range_lim - altitude range over which to evaluate the filter
    
    bg_subtract - optimize on background subtracted data
    
    scale - scale terms to divide by before filtering
    
    AdjCounts - multiply by number of profiles to get the actual photon counts
    
    return_metrics - return inverse log-likelihood for each evaluation
    
    return_sol - return the filtered profile and the verification profile
        
    """
    # attempt to handle variety of input formats
    if not hasattr(axis,'__len__'):
        axis = (axis,)
        
    if (np.asarray(order).ndim)==1:
        order=[order]
    
    if order is None:
        order = [[1,5]]*len(axis)
    elif isinstance(order,np.ndarray):
        if order.ndim != len(axis):
            order = [order]*len(axis)
    elif hasattr(order,'__len__'):
        if len(order) != len(axis):
            if len(order) == 1:
                order = [order]*len(axis)
            else:
                print('optimize_sg_raw error')
                print('  axis has length: %d'%len(axis))
                print('  order has length: %d'%len(order))
                print('  these dimensions must agree')
                raise ListDimensionsDontAgree(order)
    
    if window is None:
        window = [[3,15]]*len(axis)
    elif isinstance(window,np.ndarray):
        if window.ndim != len(axis):
            window = [window]*len(axis)
    elif hasattr(window,'__len__'):
        if len(window) != len(axis):
            if len(order) == 1:
                window = [window]*len(axis)
            else:
                print('optimize_sg_raw error')
                print('  axis has length: %d'%len(axis))
                print('  window has length: %d'%len(window))
                print('  these dimensions must agree')
                raise ListDimensionsDontAgree(window)
    
    profile = profile0.copy()
    if AdjCounts:    
        profile.multiply_piecewise(profile.NumProfList[:,np.newaxis])
    
    # thin the profile into fit and verification profiles
    pfit,pver = profile.p_thin()
    
    if bg_subtract:
        bg = 0.5*np.nanmean(profile0.profile[:,-100:],axis=1)[:,np.newaxis]
    else:
        bg = 0
        
#    if not range_lim is None:
#        pfit.slice_range(range_lim=range_lim)
#        pver.slice_range(range_lim=range_lim)
    
    pfit.profile = pfit.profile-bg
    if not scale is None:
        if scale.size == pfit.profile.size:
            pfit.profile = pfit.profile/scale
        else:
            print('scale dimensions in optimize_gaussian_raw() does not match the profile')
            print('   scale: '+str(scale.shape))
            print('   profile: '+str(pfit.profile.shape))
            print(' Continuing without scale')
            scale = None
            
    filt_opt ={}
    InvLL = {}
    
    for iax,ax in enumerate(axis):

        # define the "other" axis needed for profile-by-profile evaluation
        if ax == 0:
            bx = 1
        elif ax == 1:
            bx = 0
        else:
            print('Unrecognized axis in optimize_gaussian_raw()')
            print('  expecting a list or tuple with 1 or 0')
            print('  instead received '+str(axis))
            print('  skipping filter optimization step at %d'%iax)
            
        
        
        # build a list of order/window combinations to evaluate
        order_list,window_list = np.meshgrid(np.arange(order[iax][0],order[iax][1]),
                                             np.arange(window[iax][0],window[iax][1],2))
        order_list=order_list.flatten()
        window_list=window_list.flatten()
        # locate cases where the order value is too large for the window size
        # then remove them
        irm = np.nonzero(order_list > window_list-1) 
        window_list = np.delete(window_list,irm)
        order_list = np.delete(order_list,irm)
        
        assert window_list.size==order_list.size
        
        if full:
            fit_error_list = np.zeros((window_list.size,1))
        else:
            fit_error_list = np.zeros((window_list.size,pfit.profile.shape[bx]))
            
        verprof = pver.copy()  # make a copy to avoid altering the original
#        if not range_lim is None and ax==1:
#            verprof.slice_range(range_lim=range_lim)
        
        for ai,(sgwid,sgord) in enumerate(zip(window_list,order_list)):
            fitprof = pfit.copy()  # make a copy to avoid altering the original
            if not range_lim is None and ax ==1:
                range_mask = np.zeros(fitprof.profile.shape,dtype=bool)+(fitprof.range_array < range_lim[0])+(fitprof.range_array > range_lim[1])
                fitprof.mask(range_mask)
#                fitprof.slice_range(range_lim=range_lim)
            
            fitprof.sg_filter(sgwid,sgord,axis=ax,norm=True) # apply filter
            
            # rebuild the raw profile for evaluation
            if not scale is None:
                fitprof.profile*=scale
            fitprof.profile+=bg
            
            if full:
                fiterror = np.nansum(fitprof.profile-verprof.profile*np.log(fitprof.profile))
            else:
                fiterror = np.nansum(fitprof.profile-verprof.profile*np.log(fitprof.profile),axis=ax)

            # store the inverse log-likelihood result
            fit_error_list[ai,:] = fiterror
        
        
        if return_metrics:
            InvLL[ax] = {'fit':fiterror.copy(),
                         'window':window_list.copy(),
                         'order':order_list.copy()}
        
        if full:
            imin = np.argmin(fit_error_list)
        else:
            imin = np.argmin(fit_error_list,axis=0)
        
        filt_opt[ax] = {'window':window_list[imin],
                    'order':order_list[imin]}
        
        if return_sol or iax < len(axis)-1:
            # update the fit profile if
            # the final optimized result is requested
            # OR
            # another optimization is expected
            pfit.sg_filter(filt_opt[ax]['window'],filt_opt[ax]['order'],axis=ax,norm=True)
            
            
    return_set = (filt_opt,)
    if return_metrics:
        return_set = return_set+(InvLL,)
    if return_sol:
        if not scale is None:
            pfit.profile*=scale
        pfit.profile+=bg
        
        return_set=return_set+(pfit,pver,)
    
    if len(return_set) == 1:
        return_set = return_set[0]
        
    return return_set



def optimize_sg(pfit0,pver0,axis=1,full=False,order=[1,5],window=[3,23],range_lim=[-100,1e8],return_metrics=False):
    """
    optimize a savitzky golay filter for non-poisson data.
    The user manually passes in a fit and verification profile
    (generally obtained by thinning the original captured data).
    We then assume the data is roughly Gaussian for the evaluation step.
    
    
    """
    
    # make copies so we don't change anything in the calling routine
    pfit = pfit0.copy()
    pver = pver0.copy()
    
    # slice to the desired range
    pfit.slice_range(range_lim=range_lim)
    pver.slice_range(range_lim=range_lim)
    
    order_list = []
    window_list = []
    fit_error_list = []
    if axis == 1:
        for window_size in range(window[0],window[1]+2,2):
            for order_sg in range(order[0],np.min([window_size-1,order[1]+1])):
                fitprof = np.zeros(pfit.profile.shape)
                for ai in range(pfit.profile.shape[0]):
                    fitprof[ai,:] = savitzky_golay(pfit.profile[ai,:],window_size,order_sg,deriv=0)
                if full:
                    fiterror = np.nansum((fitprof-pver.profile)**2/pver.profile_variance)
                else:
                    fiterror = np.nansum((fitprof-pver.profile)**2/pver.profile_variance,axis=1)
                fit_error_list+=[fiterror]
                order_list+=[order_sg]
                window_list+=[window_size]

    else:
        for window_size in range(window[0],window[1]+2,2):
            for order_sg in range(order[0],np.min([window_size-1,order[1]+1])):
                fitprof = np.zeros(pfit.profile.shape)
                for ai in range(pfit.profile.shape[1]):
                    fitprof[:,ai] = savitzky_golay(pfit.profile[:,ai],window_size,order_sg,deriv=0)
                if full:
                    fiterror = np.nansum((fitprof-pver.profile)**2/pver.profile_variance)
                else:
                    fiterror = np.nansum((fitprof-pver.profile)**2/pver.profile_variance,axis=0)
                fit_error_list+=[fiterror]
                order_list+=[order_sg]
                window_list+=[window_size]
    if full:
        imin = np.nanargmin(np.array(fit_error_list))
        order_opt = order_list[imin]
        window_opt = window_list[imin]
    else:
        imin = np.nanargmin(np.array(fit_error_list),axis=0)
        order_opt = np.array(order_list)[imin]
        window_opt = np.array(window_list)[imin]
            
    if return_metrics:
        return window_opt,order_opt,np.array(fit_error_list)
    else:
        return window_opt,order_opt

def sg_kernel(window,order,deriv=None,grid_space=None):
    
    """
    Obtain the convolution kernel for a savitsky-golay filter
    with 
    window - window width [time,range] in points
    order - sg polynomial order in [time,range]
    deriv - list of desired derivatives [[time0,range0],[time1,range1],[time2,range2],...]
            negative values imply integration over the half intervals of the grid domain
    grid_space - the grid spacing in [time,range] ([dt,dR]) needed for derivatives 
            if not applied outside the function
    
    returns a list of convolution kernels with the requested derivatives
    """
    from math import factorial

    #    # defintions in [time,range]
    #    window = [7,15]  # sg window width in [time,range]
    #    order = [3,5]    # sg polynomial order in [time,range]
    #    deriv=[[0,0]]    # list of derivative orders desired in [[time0,range0],[time1,range1],[time2,range2],...]
    #    grid_space = [1,1]  # grid spacing in [time,range] for derivative evaluation
    
    
    if grid_space is None:
        grid_space = [1,1]
        
    if deriv is None:
        deriv = [[0,0]]
    
    # Need to add code to handle special case of 1D kernels when window size <= 1
    
    for ai in range(len(window)):
        window[ai] = np.abs(np.int(window[ai]))
        if window[ai] % 2 != 1 and window[ai] > 1:
            raise TypeError("window_size size must be a positive odd number")
        order[ai] = np.abs(np.int(order[ai]))
        if window[ai] < order[ai] + 1 and window[ai] > 1:
            raise TypeError("window_size %d is too small for the polynomial order %d"%(window[ai],order[ai]))
    for ai in range(len(deriv)):  
        for bi in range(len(deriv[ai])):
            deriv[ai][bi] = np.int(deriv[ai][bi])
    
    if window[0] > 1 and window[1] > 1:   
        # 1d filter in time axis
        max_order = max(order)
        nval = np.arange(-(window[0]-1) // 2, ((window[0]-1) //2 )+1)
        mval = np.arange(-(window[1]-1) // 2, ((window[1]-1) //2 )+1)
        
        mval2d,nval2d = np.meshgrid(mval,nval)
        
        Pmat = []
        Pcolname = []
        npow = []
        mpow = []
        for q in range(max_order+1):
            for j in range(max([q-order[1],0]),min([q,order[0]])+1):
                Pcolname += ['n^%d m^%d'%(j,q-j)]
                npow+=[j]   # store the power of n for later lookup
                mpow+=[q-j] # store the power of m for later lookup
                Pmat+=[nval2d.flatten()**j*mval2d.flatten()**(q-j)]
        Pmat = np.array(Pmat).T
        npow = np.array(npow)
        mpow = np.array(mpow)
        
        Cmat = np.linalg.pinv(Pmat)
        Kkern = []
        for d in deriv:
            if d[0] >= 0 and d[1] >= 0:
                dindex = np.nonzero((npow==d[0])*(mpow==d[1]))[0][0]  #find the row index for the requested derivative orders
                # obtain the convolution kernel for the SG filter with the requested derivative orders
                Kkern += [np.array((Cmat[dindex,:]*factorial(d[0])*factorial(d[1])*1.0/(grid_space[0]**d[0]*grid_space[1]**d[1])).reshape(window[0],window[1]))]
            elif d[0] < 0 and d[1] < 0:
#                # negative numbers imply integration
                Pint = grid_space[0]**np.abs(d[0])*(0.5**(npow+np.abs(d[0]))-(-0.5)**(npow+np.abs(d[0])))*sp.special.factorial(npow)/sp.special.factorial(npow+np.abs(d[0]))
                Pint *= grid_space[1]**np.abs(d[1])*(0.5**(mpow+np.abs(d[1]))-(-0.5)**(mpow+np.abs(d[1])))*sp.special.factorial(mpow)/sp.special.factorial(mpow+np.abs(d[1]))
                Kkern += [np.array((np.matrix(Pint)*np.matrix(Cmat)).reshape(window[0],window[1]))]
            elif d[0] >= 0 and d[1] < 0:
                dindex = np.nonzero((npow==d[0]))[0]
                Pint = grid_space[1]**np.abs(d[1])*(0.5**(mpow+np.abs(d[1]))-(-0.5)**(mpow+np.abs(d[1])))*sp.special.factorial(mpow)/sp.special.factorial(mpow+np.abs(d[1]))
                Pint = Pint[dindex]
                Cmat2 = np.matrix(Cmat[dindex,:]*factorial(d[0])*1.0/(grid_space[0]**d[0]*grid_space[0]**d[0]))
                Kkern += [np.array((np.matrix(Pint)*Cmat2).reshape(window[0],window[1]))]
            elif d[0] < 0 and d[1] >= 0:
                dindex = np.nonzero((npow==d[1]))[0]
                Pint = grid_space[0]**np.abs(d[0])*(0.5**(npow+np.abs(d[0]))-(-0.5)**(npow+np.abs(d[0])))*sp.special.factorial(npow)/sp.special.factorial(npow+np.abs(d[0]))
                Pint = Pint[dindex]
                Cmat2 = np.matrix(Cmat[dindex,:]*factorial(d[1])*1.0/(grid_space[1]**d[1]*grid_space[1]**d[1]))
                Kkern += [np.array((np.matrix(Pint)*Cmat2).reshape(window[0],window[1]))]
            else:
                print('LidarProfileFunctions.sg_kernel():  Unexpected derivative case')
    elif window[0] > 1 and window[1] <= 1:
        # 1d filter in range axis
        max_order = order[0]
        nval = np.arange(-(window[0]-1) // 2, ((window[0]-1) // 2 )+1)
        
        Pmat = []
        qpow = np.arange(max_order+1)
    
        for q in range(max_order+1):
            Pmat+=[nval**q]

        Pmat = np.array(Pmat).T
        
        Cmat = np.linalg.pinv(Pmat)
        Kkern = []
        for d in deriv:
            if d[0] >= 0:
                dindex = d[0]
                # obtain the convolution kernel for the SG filter with the requested derivative orders
                Kkern += [(Cmat[dindex,:]*factorial(d[0])*1.0/(grid_space[0]**d[0])).reshape(window[0],1)]
            else:
                # perform integration if deriv is negative
                Pint = np.matrix(grid_space[0]**np.abs(d[0])*(0.5**(qpow+np.abs(d[0]))-(-0.5)**(qpow+np.abs(d[0])))*sp.special.factorial(qpow)/sp.special.factorial(qpow+np.abs(d[0])))
                Kkern += [np.array((Pint*np.matrix(Cmat)).reshape(window[0],1))]
    elif window[0] <= 1 and window[1] > 1:
        max_order = order[1]
        mval = np.arange(-(window[1]-1) // 2, ((window[1]-1) // 2 )+1)
        
        Pmat = []
        qpow = np.arange(max_order+1)
    
        for q in range(max_order+1):
            Pmat+=[mval**q]
        Pmat = np.array(Pmat).T
        
        Cmat = np.linalg.pinv(Pmat)
        Kkern = []
        for d in deriv:
            if d[1] >= 0:
                dindex = d[1]
                # obtain the convolution kernel for the SG filter with the requested derivative orders
                Kkern += [(Cmat[dindex,:]*factorial(d[1])*1.0/(grid_space[1]**d[1])).reshape(1,window[1])]
            else:
                Pint = np.matrix(grid_space[1]**np.abs(d[1])*(0.5**(qpow+np.abs(d[1]))-(-0.5)**(qpow+np.abs(d[1])))*sp.special.factorial(qpow)/sp.special.factorial(qpow+np.abs(d[1])))
                Kkern += [np.array((Pint*np.matrix(Cmat)).reshape(1,window[1]))]
    
    return Kkern

def optimize_gaussian_raw(profile0,axis=(1,),full=False,std_list=None,
                          range_lim=None,scale=None,bg_subtract=False,
                          AdjCounts=False,return_metrics=False,return_sol=False,
                          mask=None):
    """
    optimize_gaussian_raw(profile0,axis=(1,),full=False,std_list=None,
                          range_lim=None,scale=None,bg_subtract=False,
                          AdjCounts=False,return_metrics=False,return_sol=False)
    
    for a photon count profile, estimate the optimal Gaussian filter 
    parameters for noise supression without distorting the profile
    
    profile0 - profile to optimize against.  This should consist of raw, true
        photon counts (no averaging or background subtracting)
    
    axis - tuple of axes to perform the filter on in order of the desired operation
    
    full - True: find the optimal filter for the entire profile
           False: find the optimal filter for each individual profile
    
    std_list - array of Gaussian widths to evaluate.  
            For multiple axes proivide a list with an array for each
    
    range_lim - altitude range over which to evaluate the filter    
    
    scale - multiplication array for the foward model
    
    bg_subtract - optimize on background subtracted data
    
    AdjCounts - multiply by number of profiles to get the actual photon counts
    
    return_metrics - return the Inverse LogLikelihood evaluation of each case
    
    return_sol - return the optimally filtered profile and the verification profile
    
    mask - mask data points from the smoothing and optimization process
        
    """
    
    if not hasattr(axis,'__len__'):
        axis = (axis,)
    
    if std_list is None:
        std_list = [np.linspace(0.5,20,40)]*len(axis)
    elif isinstance(std_list,np.ndarray):
        if std_list.ndim != len(axis):
            std_list = [std_list]*len(axis)
    elif hasattr(std_list,'__len__'):
        if len(std_list) != len(axis):
            if len(std_list) == 1:
                std_list = [std_list]*len(axis)
            else:
                print('optimize_gaussian_raw error')
                print('  axis has length: %d'%len(axis))
                print('  std_list has length: %d'%len(std_list))
                print('  these dimensions must agree')
                raise ListDimensionsDontAgree(std_list)
    
    profile = profile0.copy()
    if AdjCounts:    
        profile.multiply_piecewise(profile.NumProfList[:,np.newaxis])
    
    # thin the profile into fit and verification profiles
    pfit,pver = profile.p_thin()
    
    if bg_subtract:
        bg = 0.5*np.nanmean(profile0.profile[:,-100:],axis=1)[:,np.newaxis]
    else:
        bg = 0
    
    if not mask is None:
        pfit.mask(mask)
        
#    if not range_lim is None:
#        pfit.slice_range(range_lim=range_lim)
#        pver.slice_range(range_lim=range_lim)
    
    pfit.profile = pfit.profile-bg
#    plt.figure()
#    plt.semilogy(pfit.profile[30,:])
    if not scale is None:
        if scale.size == pfit.profile.size:
            pfit.profile = pfit.profile/scale
#            plt.semilogy(pfit.profile[30,:])
        else:
            print('scale dimensions in optimize_gaussian_raw() does not match the profile')
            print('   scale: '+str(scale.shape))
            print('   profile: '+str(pfit.profile.shape))
            print(' Continuing without scale')
            scale = None
#    plt.semilogy(pfit.profile[30,:])
    std_opt = {}
    InvLL ={}
    
    # iterate through each requested filter axis
    for iax,ax in enumerate(axis):
        
        # define the "other" axis needed for profile-by-profile evaluation
        if ax == 0:
            bx = 1
        elif ax == 1:
            bx = 0
        else:
            print('Unrecognized axis in optimize_gaussian_raw()')
            print('  expecting a list or tuple with 1 or 0')
            print('  instead received '+str(axis))
            print('  skipping filter optimization step at %d'%iax)

        # preallocate the array for storing inverse log-likelihood of each
        # filter evaluated
        if full:
            fit_error_list = np.zeros((std_list[iax].size,1))
        else:
            fit_error_list = np.zeros((std_list[iax].size,pfit.profile.shape[bx]))
        
        verprof = pver.copy()  # make a copy to avoid altering the original
#        if not range_lim is None and ax == 1:
#            verprof.slice_range(range_lim=range_lim)
        # evaluate Gaussian filters of the requested size
#        plt.figure()
#        plt.plot(verprof.profile[30,:])
#        plt.plot(pfit.profile[30,:])
        for ai,sig in enumerate(std_list[iax]):
            fitprof = pfit.copy()  # make a copy to avoid altering the original
            range_mask = np.zeros(fitprof.profile.shape,dtype=bool)
            if not range_lim is None and ax == 1:
                range_mask+=(fitprof.range_array < range_lim[0])[np.newaxis,:]+(fitprof.range_array > range_lim[1])[np.newaxis,:]
            fitprof.mask(range_mask)
#                fitprof.slice_range(range_lim=range_lim)
#            plt.plot(fitprof.profile.mask[30,:])
            fitprof.gaussian_filter(sig,axis=ax) # filter the profile
#            print('%d: %f'%(ax,sig))
#            plt.plot(fitprof.profile[30,:],label='%f'%sig)
                        
            # rebuild the raw profile for evaluation
            if not scale is None:
                fitprof.profile*=scale
            fitprof.profile+=bg
#            plt.plot(fitprof.profile[30,:],label='%f'%sig)
            
            if full:
                fiterror = np.nansum(fitprof.profile.filled(1e-30)-verprof.profile*np.log(fitprof.profile.filled(1e-30)))
            else:
                fiterror = np.nansum(fitprof.profile.filled(1e-30)-verprof.profile*np.log(fitprof.profile.filled(1e-30)),axis=ax)
            
            # store the inverse log-likelihood result
            fit_error_list[ai,:] = fiterror

        if full:
            imin = np.argmin(fit_error_list)
            std_opt[ax] = std_list[iax][imin]
        else:
            imin = np.argmin(fit_error_list,axis=0)
            std_opt[ax] = std_list[iax][imin]

        if return_metrics:
            InvLL[ax] = fit_error_list.copy()
        
        if return_sol or iax < len(axis)-1:
            # update the fit profile if
            # the final optimized result is requested
            # OR
            # another optimization is expected
            pfit.gaussian_filter(std_opt[ax],axis=ax)
        
        
        
    return_set = (std_opt,)
    if return_metrics:
        return_set=return_set+(InvLL,)
    if return_sol:
        if not scale is None:
            pfit.profile*=scale
        pfit.profile+=bg
        
        return_set=return_set+(pfit,pver,)
        
    if len(return_set) == 1:
        return_set = return_set[0]
        
    return return_set
    
    
    
#    if 1 in axis:
#        fit_error_list = np.zeros((std_list[0].size,pfit.profile[0]))
#        for sigz in std_list[0]:
##            gkern = get_conv_kernel(0,sigz,norm=True)
#            fitprof = pfit.copy()
#            fitprof.conv(0,sigz,keep_mask=True)
#            if scale.size == pfit.profile.size:
#                fitprof.profile*=scale
#            fitprof.profile+=bg
#            
#            if full:
#                fiterror = np.nansum(fitprof.profile-pver.profile*np.log(fitprof.profile))
#            else:
#                fiterror = np.nansum(fitprof.profile-pver.profile*np.log(fitprof.profile),axis=1)
#            
#            fit_error_list[ai,:] = fiterror
##            fit_error_list+=[fiterror]
#        fit_error_list 
#        if full:
#            imin = np.argmin(np.array(fit_error_list))
#            std_opt[1] = std_list[imin]
#        else:
#            imin = np.argmin(np.array(fit_error_list),axis=0)
#            std_opt[1] = std_list[imin]
#
#        if return_metrics:
#            InvLL[1] = fit_error_list.copy()
##            std_list+=[sigz]
#    if 0 in axis:
#        fit_error_list = np.zeros((std_list[1].size,pfit.profile[1]))
#        for sigt in std_list:
##            gkern = get_conv_kernel(0,sigz,norm=True)
#            fitprof = pfit.copy()
#            fitprof.conv(sigt,0,keep_mask=True)
#            if scale.size == pfit.profile.size:
#                fitprof.profile*=scale
#            fitprof.profile+=bg
#            if full:
#                fiterror = np.nansum(fitprof.profile-pver.profile*np.log(fitprof.profile))
#            else:
#                fiterror = np.nansum(fitprof.profile-pver.profile*np.log(fitprof.profile),axis=0)
#                    
#            fit_error_list+=[fiterror]
#            
#        if full:
#            imin = np.argmin(np.array(fit_error_list))
#            std_opt[0] = std_list[imin]
#        else:
#            imin = np.argmin(np.array(fit_error_list),axis=0)
#            std_opt[0] = std_list[imin]
#        
#        if return_metrics:
#            InvLL[0] = fit_error_list.copy()
##            std_list+=[sigt]
#    if not 1 in axis and not 0 in axis:
#        print('Unrecognized axis in optimize_gaussian_raw()')
#        print('  expecting a list or tuple with 1 or 0')
#        print('  instead received '+str(axis))
#        print('  skipping filter optimization')

#    if full:
#        imin = np.argmin(np.array(fit_error_list))
#        std_opt = std_list[imin]
#    else:
#        imin = np.argmin(np.array(fit_error_list),axis=0)
#        std_opt = std_list[imin]
     
#    return_set = (std_opt)    
#    if return_metrics:
#        return_set=return_set+(InvLL,)
#    return return_set

#        return std_opt,np.array(fit_error_list)
#    else:
#        return std_opt






def polyfit(x,y,order,yvar=None):
    """
    Polynomial fitting routine that computes fit uncertainty if desired
    x - input x data
    y - input (noisy) y data
    order - polynomial order
    yvar - variance of each y data point
    """
    xMat = np.matrix(x[:,np.newaxis]**np.arange(order+1)[np.newaxis,:])
#    if yvar.size==y.size:
    if not yvar is None:
        Cy = np.matrix(np.diag(yvar))
        xInv = (xMat.T*Cy.I*xMat).I*xMat.T*Cy.I
        pfit = xInv*np.matrix(y[:,np.newaxis])
        Cpfit = xInv*Cy*xInv.T
        
        return pfit,Cpfit 
    else:
        xInv=np.linalg.pinv(xMat)
        if y.ndim == 1:
            pfit = xInv*np.matrix(y[:,np.newaxis])
        else:
            pfit = xInv*np.matrix(y)
        return pfit

def polyval(x,pfit,Cm=None,deriv=0):
    """
    Applies polynomial fit data from polyfit to estimate y values for a given
    x input.  It also estimates the uncertainty in y (variance)
    x - x values corresponding to desired y values
    pfit - polynomial fit data returned from polyfit
    Cm - covariance matrix returned from polyfit
    """
    order = pfit.size-1
    if deriv == 0:
        xMat = np.matrix(x[:,np.newaxis]**np.arange(order+1)[np.newaxis,:])
    else:
        pow0 = np.arange(order+1)
        dpow = np.maximum(pow0-deriv,0)
        dcoeff = sp.special.factorial(pow0)/sp.special.factorial(pow0-deriv)
        dcoeff[pow0-deriv<0] = 0
        print(pow0)
        print(dpow)
        print(dcoeff)
        xMat = np.matrix(dcoeff[np.newaxis,:]*x[:,np.newaxis]**dpow[np.newaxis,:])
        print(xMat)
    y = xMat*pfit
#    if Cm.shape[0] == pfit.size:
    if not Cm is None:
        Cy = xMat*Cm*xMat.T
        return np.array(y),np.diag(np.array(Cy))
    else:
        return np.array(y)
