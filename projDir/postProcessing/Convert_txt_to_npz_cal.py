# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 19:29:09 2017

@author: mhayman

Converts text based cal file into an npz for running the NCAR
GV-HSRL processing scripts

"""

import numpy as np
import datetime

#fname = 'diff_default_geofile_20171025T1805.geo'
#fname  = 'diff_default_geofile_20120201T0000.geo'
#fname  = 'cross_pol_diff_geofile_20161024T0000.geo'
#fname = 'i2-default-scan-20150701T0000.cal'
fname  = 'cross_pol_diff_geofile_20150722T1920.geo'

#read_path = '/h/eol/romatsch/git/hsrl_configuration/projDir/calfiles/'
read_path = '/scr/snow2/rsfdata/projects/hsrl/raw/2015/07/'

write_path = '/h/eol/romatsch/git/hsrl_configuration/projDir/calfiles/'



cal = np.loadtxt(read_path + fname)

if '.geo' in fname:
    if 'pol' in fname:
        """
        Convert a polarization diff geo
        """
        print('recongized as cross polarization differential overlap file')
        f= open(read_path+fname, "r")
        file_list = f.readlines()[:1]
        f.close()
        date_str = file_list[0].split('profile')[1]
        effective_date = datetime.datetime.strptime(fname.split('T')[0][-8:],'%Y%m%d')
        cal_date = datetime.datetime.strptime(date_str,' data from %d-%b-%y %H:%M  UT \n')
        cross_diff_geo = 1.0/cal[:,1]  # setup so the saved parameter is multiplied by
                                    # the cross pol profile
        save_file_name = write_path+'cross_pol_diff_geo_GVHSRL'+ effective_date.strftime('%Y%m%d')
        np.savez(save_file_name,cross_diff_geo=cross_diff_geo,cal_date=cal_date)
        
    elif 'diff' in fname:
        """
        Convert a diff geo file
        """
        print('recongized as differential overlap file')
        f= open(read_path+fname, "r")
        file_list = f.readlines()[:2]
        f.close()
        date_str = file_list[0].split('profile')[1]
        effective_date = datetime.datetime.strptime(fname[-17:-4],'%Y%m%dT%H%M')
        cal_date = datetime.datetime.strptime(date_str.split(' -->')[0],' data from %d-%b-%y %H:%M')
        stop_time = datetime.datetime.strptime(cal_date.strftime('%d-%b-%y ')+date_str.split(' -->')[1],'%d-%b-%y %H:%M UTC\n')
        save_file_name = write_path+'diff_geo_GVHSRL'+ effective_date.strftime('%Y%m%d')
        np.savez(save_file_name,hi_diff_geo=cal[:,1],lo_diff_geo=cal[:,2],start_time = cal_date,stop_time=stop_time)
elif 'baseline' in fname:
    """
    Convert a baseline file
    """
    print('recongized as baseline file')
    f= open(read_path+fname, "r")
    file_list = f.readlines()[:2]
    f.close()
    date_str = file_list[0].split('profile')[1]
    effective_date = datetime.datetime.strptime(fname[-17:-4],'%Y%m%dT%H%M')
    cal_date = datetime.datetime.strptime(date_str.split(' -->')[0],' data from %d-%b-%y %H:%M')
    stop_time = datetime.datetime.strptime(cal_date.strftime('%d-%b-%y ')+date_str.split(' -->')[1],'%d-%b-%y %H:%M UTC\n')
    save_file_name = write_path+'diff_geo_GVHSRL'+ effective_date.strftime('%Y%m%d')
    np.savez(save_file_name,hi_diff_geo=cal[:,1],lo_diff_geo=cal[:,2],start_time = cal_date,stop_time=stop_time)
elif 'i2' in fname:
    """
    Convert an I2 scan file
    """
    print('recongized as I2 scan file')
    f= open(read_path+fname, "r")
    file_list = f.readlines()[:2]
    f.close()
    date_str = file_list[1].split(' on ')[1]
    cal_date = datetime.datetime.strptime(date_str.split(' -->')[0],'%d-%b-%y at %H:%M UT\n')
    effective_date = datetime.datetime.strptime(fname[-17:-4],'%Y%m%dT%H%M')
    save_file_name =  write_path + 'i2-default-scan-'+effective_date.strftime('%Y%m%dT%H%M')
    np.savez(save_file_name, \
        freq=cal[:,0],combined_scan=cal[:,1],mol_scan=cal[:,2],\
        i2_theory=cal[:,4],created_str = cal_date.strftime('%d-%b-%y at %H:%m UT'))