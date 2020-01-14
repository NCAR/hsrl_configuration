# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:03:03 2017

@author: mhayman
"""

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from time import time,gmtime,strftime
import LidarProfileFunctions as lp

#import ptv.hsrl.denoise as denoise
#from ptv.estimators.poissonnoise import poissonmodel0

import copy

cond_fun_default_gv_hsrl = {'xB':lambda x,y: cond_pass(x,operation=y),
                       'xS':lambda x,y: cond_pass(x,operation=y),
                       'xP':lambda x,y: cond_pass(x,operation=y)}

cond_fun_default_wv_dial = {'xN':lambda x,y: cond_pass(x,operation=y),
                       'xPhi':lambda x,y: cond_pass(x,operation=y)}
                       
cond_fun_default_dlb_hsrl = {'xB':lambda x,y: cond_pass(x,operation=y),
                       'xS':lambda x,y: cond_pass(x,operation=y)}

def Num_Gradient(func,x0,step_size=1e-3):
    """
    Numerically estimate the gradient of a function at x0.
    Useful for validating analytic gradient functions.
    """
    Gradient = np.zeros((x0.size))
    for ai in range(x0.size):
        xu = x0.astype(np.float)
        xl = x0.astype(np.float)
        if x0[ai] != 0:
            xu[ai] = x0[ai]*(1+step_size)            
            xl[ai] = x0[ai]*(1-step_size)
#            Gradient[ai] = (func(xu)-func(xl))/(2*step_size)
        else:
            xu[ai] = step_size
            xl[ai] = -step_size
#            Gradient[ai] = (func(step_size)-func(-step_size))/(2*step_size)

        Gradient[ai] = (func(xu)-func(xl))/(xu[ai]-xl[ai])
    return Gradient

def Num_Gradient_Dict(func,x0,step_size=1e-3):
    """
    Numerically estimate the gradient of a function at x0 which consists
    of a dict of independent variables.
    Useful for validating analytic gradient functions.
    
    """
    
    Gradient = {}
    for var in x0.keys():
        xu = copy.deepcopy(x0)
        xl = copy.deepcopy(x0)
        if x0[var].ndim > 0:
            # handle cases where the parameter is an array
            Gradient[var] = np.zeros(x0[var].shape)
            for ai in np.ndindex(x0[var].shape):
                xu[var] = x0[var].astype(np.float)
                xl[var] = x0[var].astype(np.float)
                if x0[var][ai] != 0:
                    xu[var][ai] = x0[var][ai]*(1+step_size)            
                    xl[var][ai] = x0[var][ai]*(1-step_size)
        #            Gradient[ai] = (func(xu)-func(xl))/(2*step_size)
                else:
                    xu[var][ai] = step_size
                    xl[var][ai] = -step_size
        #            Gradient[ai] = (func(step_size)-func(-step_size))/(2*step_size)
                
                Gradient[var][ai] = (func(xu)-func(xl))/(xu[var][ai]-xl[var][ai])
        else:
            # handle cases where the parameter is a scalar
            xu[var] = np.float(x0[var])
            xl[var] = np.float(x0[var])
            if x0[var] != 0:
                xu[var] = x0[var]*(1+step_size)            
                xl[var] = x0[var]*(1-step_size)
    #            Gradient[ai] = (func(xu)-func(xl))/(2*step_size)
            else:
                xu[var] = step_size
                xl[var] = -step_size
    #            Gradient[ai] = (func(step_size)-func(-step_size))/(2*step_size)
            
            Gradient[var] = (func(xu)-func(xl))/(xu[var]-xl[var])
    return Gradient

def DenoiseMolecular(MolRaw,beta_m_sonde=np.array([np.nan]),
                     geo_data=dict(geo_prof=np.array([1])),
                    MaxAlt=np.nan,n=1,start_time=0,end_time=np.nan,
                    verbose=False,accel = False,tv_lim =[0.4, 1.8],N_tv_pts=48,
                    bg_index = -50,geo_key='geo_prof',MolGain_Adj = 0.75):
    """
    Use Willem Marais' functions to denoise the molecular signal in an 
    HSRL signal.
    MolRaw - raw profile
    beta_m_sonde - estimated molecular backscatter coefficient.  Not used if
                not provided.
    geo_data - geometric overlap function.  Not used if not provided
    MaxAlt - maximum altitude in meters to fit.  Defaults to full profile.
    n = number of time profiles to process at a time
    start_time - the profile time to start on.  Defaults to first profile.
    end_time - the profile time to end on.  Defaults to last profile
    verbose - set to True to have text output from optimizer routine
    accel - attempt to accelerate by using previous TV results
    tv_lim - limits for the tv search space.  default of [0.4, 1.8] are obtained
        for the DLB-HSRL  other settings may be desirable for other lidar
    N_tv_pts - number of tv points to evalute in the tv_lim space.  Defaults
        to 48 used in DLB-HSRL
    bg_index - index above which to treat as background
    geo_key - key to access geo overlap data
    MolGain_Adj - factor to multiply the molecular gain to get a decent agreement
        between estimated backscatter and observed.
    """
    
    import ptv.hsrl.denoise as denoise
    from ptv.estimators.poissonnoise import poissonmodel0
    
#    bg_index = -50    
    if not np.isnan(end_time):
        MolRaw.slice_time([start_time,end_time])
    if any('Background Subtracted' in s for s in MolRaw.ProcessingStatus):
        BG_Sub_Flag = True
        MolRaw.profile = MolRaw.profile+MolRaw.bg[:,np.newaxis]
        Mol_BG_Total = MolRaw.bg*MolRaw.NumProfList
    else:
        BG_Sub_Flag = False    
        Mol_BG_Total = np.mean(MolRaw.profile[:,bg_index:],axis=1)*MolRaw.NumProfList  # get background before we remove the top of the profile
    
    
    if isinstance(beta_m_sonde,lp.LidarProfile):
        if not np.isnan(MaxAlt): 
            beta_m_sonde.slice_range(range_lim=[0,MaxAlt])
        MolRaw.slice_range(range_lim=beta_m_sonde.range_array[[0,-1]]) 
      
    elif not np.isnan(MaxAlt):     
            MolRaw.slice_range(range_lim=[0,MaxAlt]) 

    
    
    
    MolDenoise = MolRaw.copy()
    MolDenoise.label = 'Denoised Molecular Backscatter Channel'
    MolDenoise.descript = 'Total Variation Denoised\nUnpolarization\nMolecular Backscatter Returns'
#    MolDenoise.profile_type = '$m^{-3}$'
        
    
    if n  > MolRaw.time.size:
        n = MolRaw.time.size
        
    tune_list = []
    
    fit_range_array = MolRaw.range_array.copy()
    fit_range_array[np.nonzero(fit_range_array==0)] = 1
    
    for i_prof in range(np.ceil(MolRaw.time.size*1.0/n).astype(np.int)):
    
        istart = i_prof*n
        iend = np.min(np.array([istart + n,MolRaw.time.size]))

#        MolGain_Adj = 0.75 # 0.5

        MolFit = (MolRaw.profile[istart:iend,:]*MolRaw.NumProfList[istart:iend,np.newaxis]).astype (np.int)
        NumProf = MolRaw.NumProfList[istart:iend,np.newaxis]
        
        Mol_BG = Mol_BG_Total[istart:iend]# np.mean(MolFit[:,bg_index:],axis=1)


        # Check what a priori data the user is providing
        try:
            geo_len = len(geo_data[geo_key])
            if geo_len == 1:
                # no geo data provided.
                geo_est = np.ones(MolFit.shape[1])*geo_data[geo_key]
            else:
                if not 'Nprof' in geo_data.keys():
                    geo_data['Nprof'] = 1
                if geo_data[geo_key].shape[1] == 3:
                    # could also consider testing if the time axis of geo data is equal to the time axis on the MolRaw.profile
                
                    # 1d array for geo overlap estimate
                    geofun = 1/np.interp(MolRaw.range_array,geo_data[geo_key][:,0],geo_data[geo_key][:,1])
                    geo_est = 1.0/geo_data['Nprof']*geofun
                else:
                    # 2d array for geo overlap estimate
                    # typically used for up and down (airborne) pointing
                    geofun = 1/geo_data[geo_key][istart:iend,:]
                    geo_est = 1.0/geo_data['Nprof']*geofun
    #            geo_est = MolRaw.mean_dt/geo_data['Nprof']/geo_data['tres']*geofun
        except TypeError:
            geo_est = np.ones(MolFit.shape[1])*geo_data
            
        if hasattr(beta_m_sonde,'profile'):
            # Estimated molecular backscatter is passed to the function
            if not np.isnan(MaxAlt):    
                beta_m_sonde.slice_range(range_lim=[0,MaxAlt]) 
            beta_m_2D = beta_m_sonde.profile[istart:iend,:]
            
        else:
            beta_m_2D = np.ones(MolFit.shape)
        
#        print(beta_m_2D.shape)
#        print(geo_est.shape)        
#        print(fit_range_array.shape)
        
        
        # Create the Poisson thin object so that we can do cross-validation
        poisson_thn_obj = denoise.poissonthin (MolFit.T, p_trn_flt = 0.5, p_vld_flt = 0.5)
        
        # define coefficients in fit
        A_arr = (NumProf*MolGain_Adj*beta_m_2D*geo_est[np.newaxis,:]/fit_range_array**2).T
        A_arr[np.nonzero(A_arr==0)] = 1e-20
        A_arr[np.nonzero(np.isnan(A_arr))] = 1e-20
        A_arr[np.nonzero(np.isinf(A_arr))] = 1e-20
        
#        plt.figure()
#        plt.semilogy(MolFit.flatten())
#        plt.semilogy(A_arr.flatten()+Mol_BG[0])

        sparsa_cfg_obj = denoise.sparsaconf (eps_flt = 1e-5, verbose_int = 1e6)
        
        # check if the fit data is 1D or 2D.  1D can be run faster.
        if MolFit.shape[0] == 1:
            # Use for 1D denoising
            est_obj = poissonmodel0 (poisson_thn_obj, A_arr = A_arr, b_arr=Mol_BG[np.newaxis], log_model_bl = True, penalty_str = 'condatTV', 
                sparsaconf_obj = sparsa_cfg_obj)
        else:
            # Use for 2D denoising
            est_obj = poissonmodel0 (poisson_thn_obj, A_arr = A_arr, b_arr=Mol_BG[np.newaxis], log_model_bl = True, penalty_str = 'TV', 
                sparsaconf_obj = sparsa_cfg_obj)    
            
        # Create the denoiser object
        if MolFit.shape[0] == 1:
            # Use for 1D denoising:  
            # Defaults: log10_reg_lst = [-2, 2], nr_reg_int = 48
        
            if accel and i_prof > 0:        
                tv_reg = [log_tune[np.argmin(valid_val)]*0.8, log_tune[np.argmin(valid_val)]*1.2]  # log of range of TV values to test
                nr_int = 5  # number of TV values to test
            else:
                tv_reg = tv_lim  # log of range of TV values to test
                nr_int = 17  # number of TV values to test
                
            denoise_cnf_obj = denoise.denoiseconf (log10_reg_lst = tv_reg, nr_reg_int =nr_int, 
                pen_type_str = 'condatTV', verbose_bl = verbose)
        else:
            # Use for 2D denoising
            denoise_cnf_obj = denoise.denoiseconf (log10_reg_lst = [-2, 2], nr_reg_int = 48, 
                pen_type_str = 'TV', verbose_bl = verbose)
            
        denoiser_obj = denoise.denoisepoisson (est_obj, denoise_cnf_obj)
        # Start the denoising
        denoiser_obj.denoise ()
        
        MolDenoise.profile[istart:iend,:] = denoiser_obj.getdenoised().T/NumProf   
        
#        plt.semilogy(MolDenoise.profile[istart:iend,:].flatten()*NumProf.flatten()+Mol_BG[0]/NumProf.flatten())      
        
        log_tune,valid_val = denoiser_obj.get_validation_loss()
        tune_list.extend([[log_tune,valid_val]])  # store the results from the tuning parameters
    
    if BG_Sub_Flag:
        MolRaw.profile = MolRaw.profile - MolRaw.bg[:,np.newaxis]
    
    MolDenoise.bg = Mol_BG_Total/MolRaw.NumProfList
    MolDenoise.bg_var = Mol_BG_Total/MolRaw.NumProfList**2
    MolDenoise.profile = MolDenoise.profile-MolDenoise.bg[:,np.newaxis]
    
    MolDenoise.ProcessingStatus.extend(['Applied range PTV denoising'])
    
    return MolDenoise,tune_list
        

#    plt.figure(); 
#    plt.plot(ln_tune,valid_val)
#    
#    print('index: %d'%istart)
#    print('log TV parameter: %f'%ln_tune[np.argmin(valid_val)])
#    tv_lam = 10**(ln_tune[np.argmin(valid_val)])
#    
#    plt.figure(); 
#    plt.semilogy(MolProf_Denoised[:,0]-Mol_BG[0],label='Denoised'); 
#    plt.semilogy(MolFit[0,:]-Mol_BG[0],label='Original')
#    plt.grid(b=True)


def DenoiseTime(ProfRaw,MaxAlt=1e8,n=1,start_time=-1,end_time=1e16,
                    verbose=False,accel = False,tv_lim =[0.4, 3.0],N_tv_pts=48,
                    eps_val = 1e-5):
    """
    Use Willem Marais' functions to denoise the molecular signal in an 
    HSRL signal.
    MolRaw - raw profile
    beta_m_sonde - estimated molecular backscatter coefficient.  Not used if
                not provided.
    geo_data - geometric overlap function.  Not used if not provided
    MaxAlt - maximum altitude in meters to fit.  Defaults to full profile.
    n = number of time profiles to process at a time
    start_time - the profile time to start on.  Defaults to first profile.
    end_time - the profile time to end on.  Defaults to last profile
    verbose - set to True to have text output from optimizer routine
    accel - attempt to accelerate by using previous TV results
    tv_lim - limits for the tv search space.  default of [0.4, 3.0] are obtained
        for the WV-DIAL  other settings may be desirable for other lidar
    N_tv_pts - number of tv points to evalute in the tv_lim space.  Defaults
        to 48 used in WV-DIAL
    """ 
    
    import ptv.hsrl.denoise as denoise
    from ptv.estimators.poissonnoise import poissonmodel0

    ProfDenoise = ProfRaw.copy()
#    ProfDenoise.slice_time([start_time,end_time])
#    ProfDenoise.slice_range(range_lim=[0,MaxAlt]) 
    range_index = np.argmin(np.abs(MaxAlt-ProfDenoise.range_array))
    ProfDenoise.label = 'Denoised ' + ProfRaw.label
    ProfDenoise.descript = 'Total Variation Denoised in Time\n' + ProfRaw.descript
#    MolDenoise.profile_type = '$m^{-3}$'
        
    # check if background has been subtracted.  If so we need to add it back in
    # before denoising.
    if any('Background Subtracted' in s for s in ProfRaw.ProcessingStatus):
        BG_Sub_Flag = True
        ProfDenoise.profile = ProfDenoise.profile+ProfDenoise.bg[:,np.newaxis]
    else:
        BG_Sub_Flag = False
    
    if n  > ProfRaw.range_array.size:
        n = ProfRaw.range_array.size
        
    tune_list = []
    
    for i_prof in range(np.ceil(range_index*1.0/n).astype(np.int)):
#    for i_prof in range(np.ceil(ProfDenoise.range_array.size*1.0/n).astype(np.int)):
    
        istart = i_prof*n
        iend = np.min(np.array([istart + n,ProfRaw.range_array.size]))

#        if BG_Sub_Flag:
#            ProfFit = ((ProfRaw.profile[:,istart:iend]+ProfRaw.bg[:,np.newaxis])*ProfRaw.NumProfList[:,np.newaxis]).astype (np.int)
#        else:
#            ProfFit = (ProfRaw.profile[:,istart:iend]*ProfRaw.NumProfList[:,np.newaxis]).astype (np.int)
        ProfFit = (ProfDenoise.profile[:,istart:iend]*ProfDenoise.NumProfList[:,np.newaxis]).astype (np.int)
        NumProf = ProfDenoise.NumProfList[:,np.newaxis]
#        NumProf = ProfDenoise.NumProfList[istart:iend,np.newaxis]


        
        # Create the Poisson thin object so that we can do cross-validation
        poisson_thn_obj = denoise.poissonthin (ProfFit, p_trn_flt = 0.5, p_vld_flt = 0.5)
        
        # define coefficients in fit
        A_arr = NumProf.astype(np.double) #ProfFit.astype(np.double) #NumProf
        A_arr[np.nonzero(A_arr==0)] = 1e-10
        A_arr[np.nonzero(np.isnan(A_arr))] = 1e-10

        sparsa_cfg_obj = denoise.sparsaconf (eps_flt = eps_val, verbose_int = 1e6)
        
        # check if the fit data is 1D or 2D.  1D can be run faster.
        if ProfFit.shape[1] == 1:
            # Use for 1D denoising
            est_obj = poissonmodel0 (poisson_thn_obj, A_arr = A_arr, log_model_bl = True, penalty_str = 'condatTV', 
                sparsaconf_obj = sparsa_cfg_obj)    # b_arr=Mol_BG[np.newaxis]
        else:
            # Use for 2D denoising
            est_obj = poissonmodel0 (poisson_thn_obj, A_arr = A_arr, log_model_bl = True, penalty_str = 'TV', 
                sparsaconf_obj = sparsa_cfg_obj)    #, b_arr=Mol_BG[np.newaxis]
            
        # Create the denoiser object
        if ProfFit.shape[1] == 1:
            # Use for 1D denoising:  
            # Defaults: log10_reg_lst = [-2, 2], nr_reg_int = 48
        
            if accel and i_prof > 0:        
                tv_reg = [log_tune[np.argmin(valid_val)]*0.8, log_tune[np.argmin(valid_val)]*1.2]  # log of range of TV values to test
                nr_int = 5  # number of TV values to test
            else:
                tv_reg = [0.4, 5.0]  # log of range of TV values to test
                nr_int = N_tv_pts   # number of TV values to test
                
            denoise_cnf_obj = denoise.denoiseconf (log10_reg_lst = tv_reg, nr_reg_int =nr_int, 
                pen_type_str = 'condatTV', verbose_bl = verbose)
        else:
            # Use for 2D denoising
            denoise_cnf_obj = denoise.denoiseconf (log10_reg_lst = [-2, 2], nr_reg_int = 48, 
                pen_type_str = 'TV', verbose_bl = verbose)
            
        denoiser_obj = denoise.denoisepoisson (est_obj, denoise_cnf_obj)
        # Start the denoising
        denoiser_obj.denoise ()
        
        ProfDenoise.profile[:,istart:iend] = denoiser_obj.getdenoised()/NumProf      
        
        log_tune,valid_val = denoiser_obj.get_validation_loss()
        tune_list.extend([[log_tune,valid_val]])  # store the results from the tuning parameters
    
    if BG_Sub_Flag:
#        ProfDenoise.bg = ProfDenoise.bg
        ProfDenoise.profile = ProfDenoise.profile-ProfDenoise.bg[:,np.newaxis]
    
    ProfDenoise.ProcessingStatus.extend(['Applied time PTV denoising up to %.1f km'%(MaxAlt/1e3)])
    
    return ProfDenoise,tune_list


def DenoiseBG(ProfRaw,bg_index,verbose=False,plot_sol = False, tv_lim =[0.4, 1.8],N_tv_pts=48):
    """
    Use Willem Marais' functions to estimate the background in a lidar signal
    using PTV.
    ProfRaw - raw profile
    bg_index - index to start treating data as background
    verbose - set to True to have text output from optimizer routine
    plot_sol - Plot the denoising result for evaluation
    """ 

    import ptv.hsrl.denoise as denoise
    from ptv.estimators.poissonnoise import poissonmodel0

    ProfFit = ((np.nansum(ProfRaw.profile[:,bg_index:],axis=1)*ProfRaw.NumProfList)[:,np.newaxis]).astype (np.int)
    NumProf = ProfRaw.NumProfList[:,np.newaxis]
#        NumProf = ProfDenoise.NumProfList[istart:iend,np.newaxis]


    
    # Create the Poisson thin object so that we can do cross-validation
    poisson_thn_obj = denoise.poissonthin (ProfFit, p_trn_flt = 0.5, p_vld_flt = 0.5)
    
    # define coefficients in fit
    A_arr = NumProf.astype(np.double) #ProfFit.astype(np.double) #NumProf
    A_arr[np.nonzero(A_arr==0)] = 1e-10
    A_arr[np.nonzero(np.isnan(A_arr))] = 1e-10

    sparsa_cfg_obj = denoise.sparsaconf (eps_flt = 1e-5, verbose_int = 1e6)
    
    # check if the fit data is 1D or 2D.  1D can be run faster.
    if ProfFit.shape[1] == 1:
        # Use for 1D denoising
        est_obj = poissonmodel0 (poisson_thn_obj, A_arr = A_arr, log_model_bl = True, penalty_str = 'condatTV', 
            sparsaconf_obj = sparsa_cfg_obj)    # b_arr=Mol_BG[np.newaxis]
    else:
        # Use for 2D denoising
        est_obj = poissonmodel0 (poisson_thn_obj, A_arr = A_arr, log_model_bl = True, penalty_str = 'TV', 
            sparsaconf_obj = sparsa_cfg_obj)    #, b_arr=Mol_BG[np.newaxis]
        
    # Create the denoiser object 
    tv_reg = tv_lim  # log of range of TV values to test
    nr_int = N_tv_pts   # number of TV values to test
        
    denoise_cnf_obj = denoise.denoiseconf (log10_reg_lst = tv_reg, nr_reg_int =nr_int, 
    pen_type_str = 'condatTV', verbose_bl = verbose)

        
    denoiser_obj = denoise.denoisepoisson (est_obj, denoise_cnf_obj)
    # Start the denoising
    denoiser_obj.denoise ()
    
    ProfRaw.bg = (denoiser_obj.getdenoised()/NumProf/ProfRaw.profile[:,bg_index:].shape[1]).flatten()
    ProfRaw.bg_var = np.nansum(ProfRaw.profile_variance[:,bg_index:],axis=1)/(np.shape(ProfRaw.profile_variance[:,bg_index:])[1])**2
    ProfRaw.profile = ProfRaw.profile-ProfRaw.bg[:,np.newaxis]
    ProfRaw.profile_variance = ProfRaw.profile_variance+ProfRaw.bg_var[:,np.newaxis]
    ProfRaw.ProcessingStatus.extend(['PTV Background Subtracted over [%.2f, %.2f] m'%(ProfRaw.range_array[bg_index],ProfRaw.range_array[-1])])
    
    ProfRaw.profile = ProfRaw.profile-ProfRaw.bg[:,np.newaxis]  
    
    log_tune,valid_val = denoiser_obj.get_validation_loss()
    
    if plot_sol:
        plt.figure();
        plt.plot(ProfRaw.time/3600,ProfFit.flatten())
        plt.plot(ProfRaw.time/3600,ProfRaw.bg*NumProf.flatten()*ProfRaw.profile[:,bg_index:].shape[1])     
        plt.grid(b=True)
        plt.xlabel('Time [h-UTC]')
        plt.ylabel('Photon Counts')
        plt.legend(('Measured','PTV Estimated'))
        
        plt.figure();
        plt.plot(log_tune,valid_val)
        plt.xlabel('log penalty factor')
        plt.ylabel('Fit Error')
        plt.grid(b=True)
    
    return [log_tune,valid_val]

def ProfilesTotalxvalid_2D(x,xvalid,tdim,mol_bs_coeff,Const,Mprof_bg=0,Cprof_bg=0,dt=np.array([1.0])):
    """
    dt sets the adjustment factor to convert counts to count rate needed in
    deadtime correction
    pdim - profile time dimension
    """    
    ix = 5  # sets the profile offset
    x2D = x[ix:].reshape((tdim,2*xvalid.size))
    N = mol_bs_coeff.shape[1]  # length of profile    
    Nx = xvalid.size
    Cam = x[0]
    Gm = x[1]  # molecular gain
    Gc = x[2]   # combinded gain
    sLR = np.zeros((tdim,N))
    sLR[:,xvalid] = np.exp(x2D[:,:Nx])  # lidar ratio terms
    deadtimeMol=x[3]
    deadtimeComb=x[4]
#    bgMol = x[5]
#    bgComb = x[6]
    
    Baer = np.zeros((tdim,N))
    Baer[:,xvalid] = np.exp(x2D[:,Nx:])  # aerosol backscatter terms   
    
    Molmodel = Gm*(mol_bs_coeff+Cam*Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer,axis=1))+Mprof_bg[:,np.newaxis]
    Combmodel = Gc*(mol_bs_coeff+Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer,axis=1))+Cprof_bg[:,np.newaxis]
    
#    print(dt)    
    
    Molmodel = Molmodel*dt[:,np.newaxis]/(dt[:,np.newaxis]+Molmodel*deadtimeMol)   
    Combmodel = Combmodel*dt[:,np.newaxis]/(dt[:,np.newaxis]+Combmodel*deadtimeComb) 
    
    return Molmodel,Combmodel


def LREstimate_buildConst_2D(geo_correct,MolScale,beta_m,range_array,dt):
    geofun = 1/np.interp(range_array,geo_correct['geo_prof'][:,0],geo_correct['geo_prof'][:,1])
    ConstTerms = 1.0/geo_correct['Nprof']*MolScale*beta_m*geofun*np.exp(-2.0*np.cumsum(8*np.pi/3*beta_m,axis=1)*(range_array[1]-range_array[0]))/range_array[np.newaxis,:]**2
#     dt/geo_correct['Nprof']/geo_correct['tres']
    return ConstTerms
    
def LREstimateTotalxvalid_2D(x,xvalid,Mprof,Cprof,mol_bs_coeff,Const,lam,Mprof_bg=0,Cprof_bg=0,dt=1.0):
    
    N = Mprof.shape[1]  # length of profile    
    tdim = Mprof.shape[0]
    ix = 5  # sets the profile offset
    Nx = xvalid.size
    
    x2D = x[ix:].reshape((tdim,2*xvalid.size))

    sLR = np.zeros((tdim,N))
    sLR[:,xvalid] = np.exp(x2D[:,:Nx])  # lidar ratio terms
    Baer = np.zeros((tdim,N))
    Baer[:,xvalid] = np.exp(x2D[:,Nx:])  # aerosol backscatter terms
    BaerExp = np.zeros((tdim,N))
    BaerExp[:,xvalid] = x2D[:,Nx:]  # aerosol backscatter exponent terms

    Molmodel,Combmodel = ProfilesTotalxvalid_2D(x,xvalid,tdim,mol_bs_coeff,Const,Mprof_bg=Mprof_bg,Cprof_bg=Cprof_bg,dt=dt)
    dxvalid = np.diff(xvalid)[np.newaxis,:]
    deriv_t = np.nansum(np.abs(np.diff(sLR[:,xvalid],axis=0)))*lam[0][0] + np.nansum(np.abs(np.diff(BaerExp[:,xvalid],axis=0)))*lam[1][0]
    deriv_r = np.nansum(np.abs(np.diff(sLR[:,xvalid],axis=1)/dxvalid))*lam[0][1] + np.nansum(np.abs(np.diff(BaerExp[:,xvalid],axis=1)/dxvalid))*lam[1][1]
    ErrRet = np.nansum((Molmodel-Mprof*np.log(Molmodel)))+np.nansum((Combmodel-Cprof*np.log(Combmodel)))+deriv_t+deriv_r
    return ErrRet

def LREstimateTotalxvalid_2D_prime(x,xvalid,Mprof,Cprof,mol_bs_coeff,Const,lam,Mprof_bg=0,Cprof_bg=0,dt=1.0):

    N = Mprof.shape[1]  # length of profile    
    tdim = Mprof.shape[0]
    ix = 5  # sets the profile offset
    Nx = xvalid.size
    
    x2D = x[ix:].reshape((tdim,2*xvalid.size)) 
    
#    ix = 7  # sets the profile offset
#    N = Mprof.size  # length of profile    
#    Nx = xvalid.size
    Cam = x[0]
    Gm = x[1]  # molecular gain 
    Gc = x[2]  # combined gain
    
    deadtimeMol=x[3]
    deadtimeComb=x[4]
#    bgMol = x[5]
#    bgComb = x[6]
    
    sLR = np.zeros((tdim,N))
    sLR[:,xvalid] = np.exp(x2D[:,:Nx])  # lidar ratio terms
    Baer = np.zeros((tdim,N))
    Baer[:,xvalid] = np.exp(x2D[:,Nx:])  # aerosol backscatter terms
    BaerExp = np.zeros((tdim,N))
    BaerExp[:,xvalid] = x2D[:,Nx:]  # aerosol backscatter exponent terms

    # obtain models including nonlinear responde
    Molmodel,Combmodel = ProfilesTotalxvalid_2D(x,xvalid,tdim,mol_bs_coeff,Const,Mprof_bg=Mprof_bg,Cprof_bg=Cprof_bg,dt=dt)
    
    # obtain models without nonlinear responde but including background
    xlin = x.copy()
    xlin[3] = 0
    xlin[4] = 0
    Molmodel1,Combmodel1 = ProfilesTotalxvalid_2D(xlin,xvalid,tdim,mol_bs_coeff,Const,Mprof_bg=Mprof_bg,Cprof_bg=Cprof_bg)
    
    #obtain models without nonlinear response or background
    Molmodel0 = Molmodel1-Mprof_bg[:,np.newaxis]
    Combmodel0 = Combmodel1-Cprof_bg[:,np.newaxis]


    
    # useful definitions for gradient calculations
    e0m = (1-Mprof/Molmodel)
    e0c =(1-Cprof/Combmodel)
    e_dtm = dt[:,np.newaxis]**2/(dt[:,np.newaxis]+Molmodel1*deadtimeMol)**2
    e_dtc = dt[:,np.newaxis]**2/(dt[:,np.newaxis]+Combmodel1*deadtimeComb)**2
    grad0m = np.sum(e0m*e_dtm*Molmodel0,axis=1)[:,np.newaxis]-np.cumsum(e0m*e_dtm*Molmodel0,axis=1)
    grad0c = np.sum(e0c*e_dtc*Combmodel0,axis=1)[:,np.newaxis]-np.cumsum(e0c*e_dtc*Combmodel0,axis=1)
    
    # lidar ratio gradient terms:
    gradErrS = -2*Baer*sLR*(grad0m+grad0c)
    gradErrS[np.nonzero(np.isnan(gradErrS))] = 0

    # backscatter cross section gradient terms:
    gradErrB = -2*sLR*Baer*(grad0m+grad0c)+Baer*e0m*Gm*Const*Cam*np.exp(-2*np.cumsum(sLR*Baer,axis=1))+Baer*e0c*Const*Gc*np.exp(-2*np.cumsum(sLR*Baer,axis=1))
    gradErrB[np.nonzero(np.isnan(gradErrB))] = 0

    # cross talk gradient term
    gradErrCam = np.nansum(e0m*e_dtm*Gm*Const*Baer*np.exp(-2*np.cumsum(sLR*Baer,axis=1)))

    # combined gain gradient term
    gradErrGc = np.nansum(e0c*e_dtc*(mol_bs_coeff+Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer,axis=1)))
    
    # molecular gain gradient term
    gradErrGm = np.nansum(e0m*e_dtm*(mol_bs_coeff+Cam*Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer,axis=1)))
    
    # molecular dead time gradient term
    gradErrDTm = np.nansum(-e0m*dt[:,np.newaxis]*Molmodel1**2/(dt[:,np.newaxis]+Molmodel1*deadtimeMol))
    
    # combined dead time gradient term
    gradErrDTc = np.nansum(-e0c*dt[:,np.newaxis]*Combmodel1**2/(dt[:,np.newaxis]+Combmodel1*deadtimeComb))
    
#    # molecular background adjustment term
#    gradErrBGm = np.nansum(e0m*e_dtm*Mprof_bg)
#    
#    # combined background adjustment term
#    gradErrBGc = np.nansum(e0c*e_dtc*Cprof_bg)

    # total variance gradient terms
#    gradErr[np.nonzero(np.isnan(gradErr))] = 0
#    gradErr = gradErr[xvalid]
    dxvalid = np.diff(xvalid)[np.newaxis,:]
    gradErrStv = np.zeros((tdim,Nx))
    gradErrBtv = np.zeros((tdim,Nx))
    
    # range derivative for lidar ratio
    gradpenS = lam[0][1]*np.sign(np.diff(sLR[:,xvalid],axis=1))/dxvalid
    gradpenS[np.nonzero(np.isnan(gradpenS))] = 0
    gradErrStv[:,:-1] = gradErrStv[:,:-1]-gradpenS
    gradErrStv[:,1:] = gradErrStv[:,1:]+gradpenS
    
    # time derivative for lidar ratio
    gradpenS = lam[0][0]*np.sign(np.diff(sLR[:,xvalid],axis=0))
    gradpenS[np.nonzero(np.isnan(gradpenS))] = 0
    gradErrStv[:-1,:] = gradErrStv[:-1,:]-gradpenS
    gradErrStv[1:,:] = gradErrStv[1:,:]+gradpenS
    
    # range derivative for aerosol backscatter
    gradpenB = lam[1][1]*np.sign(np.diff(BaerExp[:,xvalid],axis=1))/dxvalid
    gradpenB[np.nonzero(np.isnan(gradpenB))] = 0
    gradErrBtv[:,:-1] = gradErrBtv[:,:-1]-gradpenB
    gradErrBtv[:,1:] = gradErrBtv[:,1:]+gradpenB
    
    # time derivative for aerosol backscatter
    gradpenB = lam[1][0]*np.sign(np.diff(BaerExp[:,xvalid],axis=0))
    gradpenB[np.nonzero(np.isnan(gradpenB))] = 0
    gradErrBtv[:-1,:] = gradErrBtv[:-1,:]-gradpenB
    gradErrBtv[1:,:] = gradErrBtv[1:,:]+gradpenB
    
    gradErr = np.zeros(ix+2*Nx*tdim)
    gradErr2D = np.zeros((tdim,2*Nx))
    gradErr[0] = gradErrCam
    gradErr[1] = gradErrGm
    gradErr[2] = gradErrGc
    gradErr[3] = gradErrDTm
    gradErr[4] = gradErrDTc
#    gradErr[5] = gradErrBGm
#    gradErr[6] = gradErrBGc
    gradErr2D[:,:Nx] = gradErrS[:,xvalid]+gradErrStv
    gradErr2D[:,Nx:] = gradErrB[:,xvalid]+gradErrBtv
    
    gradErr[ix:] = gradErr2D.flatten()
    
    return gradErr

def MLE_Cals_2D(MolRaw,CombRaw,beta_aer,surf_temp,surf_pres,geo_data,minSNR=0.5,\
    t_lim=np.array([np.nan,np.nan]),verify=False,verbose=False,print_sol=True,\
    plotfigs=True,lam_array=np.array([np.nan]),Nmax=2000):
    """
    Runs a maximum likelihood estimator to correct for
        Channel gain mismatch, 
        Aerosol to Molecular Channel Crosstalk
        Detector Dead Time
    and obtain estimates of
        Backscatter coefficient
        Extinction coefficient
        Lidar Ratio
        Each Channel's gain
        Aerosol to Molecular Channel Crosstalk
        Detector Dead Time
        
    Inputs:
        MolRaw - Raw Molecular Profile
        CombRaw - Raw Combined Profile
        beta_aer - derived estimate of aerosol backscatter coefficent
        surf_temp - temperature data from the surface station
        surf_pres - pressure data from the surface station
        geo_data - data file from loading the geometric overlap function
        minSNR - minimum SNR required to be included as containing possible aerosols
        t_lim - currently disabled.  when added, will allow us to select only a segment to operate on.
        verify - if True it will use Poisson Thinning to verify TV solution
        verbose - if True it will output the fit result of each TV iteration
        print_sol - print the solved calibration parameters
        plotfigs - plot the TV verification error
        lam_array - array containing TV values to be evaluated
        Nmax - maximum optimizer iterations
        
    Substitutions:
        beta_aer for aer_beta_dlb
        beta_aer_E for aer_beta_E
    """
    
    

    t_lim[0] = np.nanmax(np.array([t_lim[0],beta_aer.time[0]]))
    t_lim[1] = np.nanmin(np.array([t_lim[1],beta_aer.time[-1]]))

#    b = np.argmin(np.abs(beta_aer.time/3600-t_lim[0]))  # 6.3, 15
#    Interval = np.argmin(np.abs(beta_aer.time/3600-t_lim[0])) - b + 1
    
    # force processing of the entire profile    
    b = 0
    Interval = MolRaw.time.size  # 
#    minSNR = 2.0 # min aerosol snr to include as an optimization parameter.  Typically 2.0

#    sLRinitial = 35  # initial assumed lidar ratio if no prior data exists


#    dG = 0.04  # allowed change in channel gain between data points
#    use_mask = False

    ix = 5  # number of header variables    
    
    #lam = np.array([4,0.000001])  # Penalty for [Lidar ratio, Backscatter Coefficient]  [0.0005,0.0005]
    #lam = np.array([4.0,0.0000001])
    lam = [[0.01,0.01],[0.01,0.01]]
    
#    dt0 = time()
    
    # copy the raw profiles to avoid modifying them
    MolRawE = MolRaw.copy()
    CombRawE = CombRaw.copy()
    
    MolRawE.profile = MolRawE.profile*MolRawE.NumProfList[:,np.newaxis]
    MolRawE.profile_variance = MolRawE.profile_variance*MolRawE.NumProfList[:,np.newaxis]**2
    CombRawE.profile = CombRawE.profile*CombRawE.NumProfList[:,np.newaxis]
    CombRawE.profile_variance = CombRawE.profile_variance*CombRawE.NumProfList[:,np.newaxis]**2
    
    if verify:
        [MolRawF,MolRawV] = MolRawE.p_thin()
        [CombRawF,CombRawV] = CombRawE.p_thin()
        thin_adj = 0.5  # adjust gain on estimator if profiles have been poisson thinned
    else:
        thin_adj = 1.0  
#        MolRawF = MolRawE.copy()
#        MolRawV = MolRawE
#        CombRawF = CombRawE.copy()
#        CombRawV = CombRawE
          
    
    # lower resolution profiles to obtain better cloud detection
    beta_aer_E = beta_aer.copy()
    beta_aer_E.conv(4,2)
#    MolE = Molecular.copy()
#    MolE.conv(4,2)
#    CombE = CombHi.copy()
#    CombE.conv(4,2)
#    
#    aer_beta_E = lp.AerosolBackscatter(MolE,CombE,beta_mol_sonde)
    
    #MolRawE.nonlinear_correct(38e-9);
    #CombRawE.nonlinear_correct(29.4e-9);
    
    Nprof = MolRawE.NumProfList[b:b+Interval]
#    FitMol = Nprof[:,np.newaxis]*MolRawE.profile[b:b+Interval,:]
#    FitMol_bg = Nprof*np.nanmean(MolRawE.profile[b:b+Interval,-50:],axis=1)
#    
#    FitComb = Nprof[:,np.newaxis]*CombRawE.profile[b:b+Interval,:]
#    FitComb_bg = Nprof*np.nanmean(CombRawE.profile[b:b+Interval,-50:],axis=1)
    
    if verify:
        FitMol = MolRawF.profile[b:b+Interval,:]
        FitMol_bg = np.nanmean(MolRawF.profile[b:b+Interval,-50:],axis=1)
        
        FitComb = CombRawF.profile[b:b+Interval,:]
        FitComb_bg = np.nanmean(CombRawF.profile[b:b+Interval,-50:],axis=1)        
        
        ValidMol = MolRawV.profile[b:b+Interval,:]
        ValidComb = CombRawV.profile[b:b+Interval,:] 
    else:
        FitMol = MolRawE.profile[b:b+Interval,:]
        FitMol_bg = np.nanmean(MolRawE.profile[b:b+Interval,-50:],axis=1)
        
        FitComb = CombRawE.profile[b:b+Interval,:]
        FitComb_bg = np.nanmean(CombRawE.profile[b:b+Interval,-50:],axis=1)
    
        ValidMol = MolRawE.profile[b:b+Interval,:]
        ValidComb = CombRawE.profile[b:b+Interval,:]
        
    
    FitAer = beta_aer_E.profile[b:b+Interval,:]
    FitAer = np.hstack((FitAer,np.zeros((FitAer.shape[0],FitMol.shape[1]-FitAer.shape[1]))))
    FitAer[np.nonzero(np.isnan(FitAer))[0]]=0


#    if model_atm: 
    beta_mol,temp,pres = lp.get_beta_m_model(MolRawE,surf_temp,surf_pres,returnTP=True)
    pres.descript = 'Ideal Atmosphere Pressure in atm'
#    else:
#        beta_m_sonde,sonde_time,sonde_index_prof,temp,pres,sonde_index = lp.get_beta_m_sonde(MolRawE,Years,Months,Days,sonde_path,interp=True,returnTP=True)
#        pres.descript = 'Sonde Measured Pressure in atm'



    dR = MolRaw.mean_dR
#    
    rate_adj = 1.0/(MolRawE.shot_count*MolRawE.NumProfList*MolRawE.binwidth_ns*1e-9)
    
    tdim = Interval #MolRaw.time.size
    
    prefactor = 1.0# 1e-7
    
    #LREstimate_buildConst_2D(geo_correct,MolScale,beta_m,range_array):
    ConstTerms = thin_adj*Nprof[:,np.newaxis]*LREstimate_buildConst_2D(geo_data,1,beta_mol.profile,MolRawE.range_array,MolRawE.mean_dt)
    ConstTerms = ConstTerms/beta_mol.profile
    

    
    xvalid2D = np.zeros(FitComb.shape)
#    CamList = np.zeros(Interval)
#    GmList = np.zeros(Interval)
#    GcList = np.zeros(Interval)
#    Dtmol = np.zeros(Interval)
#    Dtcomb = np.zeros(Interval)
#    fbgmol = np.zeros(Interval)
#    fbgcomb = np.zeros(Interval)
    sLR2D = np.zeros(FitAer.shape)
    beta_a_2D = np.zeros(FitAer.shape)
    alpha_a_2D = np.zeros(FitAer.shape)
    fit_mol_2D = np.zeros(FitAer.shape)
    fit_comb_2D = np.zeros(FitAer.shape)
    
#    fxVal = np.zeros(Interval)
    opt_iterations = np.zeros(Interval)
    opt_exit_mode = np.zeros(Interval)
#    str_exit_mode_list = []
    
    # Find altitudes where there is aerosol data to be estimated.  Only include those altitudes (but across all time) in the estimation
    xvalid = np.nonzero(np.sum(beta_aer_E.SNR()>minSNR,axis=0))[0]
    xvalid2D[:,xvalid] = 1    
    
    
    if verify:
        if np.isnan(lam_array).any():
            lam_array = np.logspace(-5,-2,47)  
    else:
        lam_array = np.array([0])
    fitErrors = np.zeros(lam_array.size)
    sol_List = []
    out_cond_array = np.zeros(lam_array.size)    
    
    ### Optimization Routine ###
    for i_lam in range(lam_array.size):
        if verify:
            lam = np.array([[lam_array[i_lam]]*2,[lam_array[i_lam]]*2])
        else:
            lam = np.array([[0.01,0.01],[0.01,0.01]])  # Penalty for [Lidar ratio, Backscatter Coefficient]  [4,0.000001]
    
        #nonlinear
        FitProfMol = lambda x: prefactor*LREstimateTotalxvalid_2D(x,xvalid,FitMol,FitComb,beta_mol.profile,ConstTerms,lam,Mprof_bg=FitMol_bg,Cprof_bg=FitComb_bg,dt=rate_adj)
        FitProfMolDeriv = lambda x: prefactor*LREstimateTotalxvalid_2D_prime(x,xvalid,FitMol,FitComb,beta_mol.profile,ConstTerms,lam,Mprof_bg=FitMol_bg,Cprof_bg=FitComb_bg,dt=rate_adj)
        
        
        bndsP = np.zeros((ix+2*xvalid.size*tdim,2))
        bndsP[0,0] = 0.0
        bndsP[0,1] = 0.1
        bndsP[1,0] = 0.5
        bndsP[1,1] = 1.1
        bndsP[2,0] = 0.7
        bndsP[2,1] = 1.5
        
        bndsP[3,0] = 0.0  # molecular deadtime
        bndsP[3,1] = 100e-9
        bndsP[4,0] = 0.0 # combined deeadtime
        bndsP[4,1] = 100e-9
        
        bnds2D = np.zeros((tdim,2*xvalid.size))
        bnds2D[:,:xvalid.size] = np.log(1.0*dR)  # lidar ratio lower limit
        bnds2D[:,xvalid.size:] = np.log(1e-12)   # aerosol coefficient lower limit
        bndsP[ix:,0] = bnds2D.flatten()
        
        bnds2D = np.zeros((tdim,2*xvalid.size))
        bnds2D[:,:xvalid.size] = np.log(2e2*dR)  # lidar ratio upper limit
        bnds2D[:,xvalid.size:] = np.log(1e-2)   # aerosol coefficient upper limit
        bndsP[ix:,1] = bnds2D.flatten()
        
        
        x0 = np.zeros((ix+2*xvalid.size*tdim))
        x0[0] = 0.003  # Cam
        x0[1] = 0.7545  # Gm
        x0[2] = 1.0904
        x0[3] = 10e-9  # molecular deadtime
        x0[4] = 20e-9  # combined deadtime
        
        x02D = np.zeros((tdim,2*xvalid.size))
        x02D[:,:xvalid.size] = np.log((-8.5*np.log10(FitAer[:,xvalid])-24.5)*dR)  # lidar ratio
        x02D[np.nonzero(np.isnan(x02D))] = np.log(1.0*dR)  # get rid of invalid numbers
    #    x02D[:,:xvalid.size] = sLRinitial
        x02D[:,xvalid.size:] = np.log(beta_aer.profile[b:b+Interval,xvalid])  # aerosol backscatter
        x02D[np.nonzero(np.isnan(x02D))] = np.log(1e-12)    # get rid of invalid numbers
        x0[ix:] = x02D.flatten()
        
        
        #    wMol,fxVal[pI],opt_iterations[pI],opt_exit_mode[pI],str_exit_mode = \
        #        scipy.optimize.fmin_slsqp(FitProfMol,x0,bounds=bndsP,fprime=FitProfMolDeriv,disp=0,iter=1000,acc=1e-14,full_output=True) # fprime=FitProfMolDeriv disp=0,
        #    str_exit_mode_list.extend(str_exit_mode)
        
        sol1D,opt_iterations,opt_exit_mode = scipy.optimize.fmin_tnc(FitProfMol,x0,bounds=bndsP,fprime=FitProfMolDeriv,maxfun=2000,eta=1e-5,disp=0)    
        
        fit_mol_2D,fit_comb_2D = ProfilesTotalxvalid_2D(sol1D,xvalid,tdim,beta_mol.profile,ConstTerms,Mprof_bg=FitMol_bg,Cprof_bg=FitComb_bg,dt=rate_adj)
        #    ProfileRMSError = np.sqrt(np.sum((FitComb-fit_comb_2D)**2+(FitMol-fit_mol_2D)**2,axis=1))
        
        ProfileLogError = np.nansum(fit_comb_2D-ValidComb*np.log(fit_comb_2D),axis=1) + np.nansum(fit_mol_2D-ValidMol*np.log(fit_mol_2D),axis=1)
    #    ProfileLogError = np.nansum(fit_comb_2D-FitComb*np.log(fit_comb_2D),axis=1) + np.nansum(fit_mol_2D-FitMol*np.log(fit_mol_2D),axis=1)
        
        fitErrors[i_lam] = np.nansum(ProfileLogError)
        sol_List.extend([sol1D])
        out_cond_array[i_lam] = opt_exit_mode           
        if verbose:
            print('Log Error: %f'%fitErrors[i_lam])
            print('MLE Output Flag: %s'%scipy.optimize.tnc.RCSTRINGS[opt_exit_mode])
    
    ### End Optimization Routine ###

    isol = np.argmin(fitErrors)
        
    fit_mol_2D,fit_comb_2D = ProfilesTotalxvalid_2D(sol_List[isol],xvalid,tdim,beta_mol.profile,ConstTerms,Mprof_bg=FitMol_bg,Cprof_bg=FitComb_bg,dt=rate_adj)

    Cam = sol_List[isol][0]
    Gm = sol_List[isol][1]
    Gc = sol_List[isol][2]      
    if print_sol and verify:
        print('Final Solution:')
        print('lam: %f\nCam = %f\nGm = %f\nGc = %f\nMol DT = %f ns\nComb DT = %f ns\nFitError:%f'%(lam_array[isol],Cam,Gm,Gc,sol_List[isol][3]*1e9,sol_List[isol][4]*1e9,fitErrors[isol]))
        print('Output Flag: %d'%out_cond_array[isol])
        print('Output Flag Definition: %s'%scipy.optimize.tnc.RCSTRINGS[out_cond_array[isol]])
    
    if plotfigs:
        if verify:
            plt.figure()
            plt.semilogx(lam_array,fitErrors)
            plt.ylabel('Log Error')
            plt.xlabel('Fit Index')       

    sol2D = sol_List[isol][ix:].reshape((tdim,2*xvalid.size))
    beta_a_2D[:,xvalid] = np.exp(sol2D[:,xvalid.size:])
    sLR2D[:,xvalid] = np.exp(sol2D[:,:xvalid.size])/dR

#    Cam = sol1D[0]
#    Gm = sol1D[1]
#    Gc = sol1D[2]
#    DTmol = sol1D[3]
#    DTcomb = sol1D[4]
    
    cal_params = {'Cam':sol_List[isol][0],'Gm':sol_List[isol][1],'Gc':sol_List[isol][2],'DTmol':sol_List[isol][3],'DTcomb':sol_List[isol][4]}
    
    
    #    fit_mol_2D[pI,:],fit_comb_2D[pI,:] = ProfilesTotalxvalid(wMol,xvalid,beta_m_sonde.profile[pI,:],ConstTerms[pI,:],Mprof_bg=FitMol_bg[pI],Cprof_bg=FitComb_bg[pI])
    #ProfilesTotalxvalid_2D(x,xvalid,tdim,mol_bs_coeff,Const,Mprof_bg=0,Cprof_bg=0,dt=np.array([1.0]))
    
    
    alpha_a_2D = beta_a_2D*sLR2D
    

    
#    # attempt to remove profiles where the fit was bad
#    pbad = np.nonzero(np.abs(np.diff(ProfileError)/ProfileError[:-1])>3)[0]+1
#    xvalid2D[pbad,:]= 0
#    pbad = np.nonzero(np.abs(np.diff(ProfileError)/ProfileError[1:])>3)[0]
#    xvalid2D[pbad,:]= 0


    # Merge with original aerosol data set
    beta_merge = beta_aer.copy()
    beta_a_2D_adj = beta_a_2D[:,:beta_aer.profile.shape[1]]
    iMLE = np.nonzero(xvalid2D)
    beta_merge.profile[iMLE] = beta_a_2D_adj[iMLE]
    beta_merge.descript = 'Maximum Likelihood Estimate of Aerosol Backscatter Coefficient in m^-1 sr^-1'
    
    sLR_mle = beta_aer.copy()
    sLR_mle.profile = sLR2D[:,:beta_aer.profile.shape[1]].copy()
    sLR_mle.descript = 'Maximum Likelihood Estimate of Aerosol Lidar Ratio sr'
    sLR_mle.label = 'Aerosol Lidar Ratio'
    sLR_mle.profile_type = '$sr$'
    sLR_mle.profile_variance = sLR_mle.profile_variance*0.0
    
    beta_a_mle = beta_aer.copy()
    beta_a_mle.profile = beta_a_2D[:,:beta_aer.profile.shape[1]].copy()
    beta_a_mle.descript = 'Maximum Likelihood Estimate of Aerosol Backscatter Coefficient in m^-1 sr^-1'
    beta_a_mle.label = 'Aerosol Backscatter Coefficient'
    beta_a_mle.profile_type = '$m^{-1}sr^{-1}$'
    beta_a_mle.profile_variance = beta_a_mle.profile_variance*0.0
    
    alpha_a_mle = beta_aer.copy()
    alpha_a_mle.profile = alpha_a_2D[:,:beta_aer.profile.shape[1]].copy()
    alpha_a_mle.descript = 'Maximum Likelihood Estimate of Aerosol Extinction Coefficient in m^-1'
    alpha_a_mle.label = 'Aerosol Extinction Coefficient'
    alpha_a_mle.profile_type = '$m^{-1}$'
    alpha_a_mle.profile_variance = alpha_a_mle.profile_variance*0.0
    
    fit_mol_mle = MolRawE.copy()
    fit_mol_mle.profile = fit_mol_2D.copy()
    fit_mol_mle.slice_range_index(range_lim=[0,beta_aer.profile.shape[1]])
    fit_mol_mle.descript = 'Maximum Likelihood Estimate of ' + fit_mol_mle.descript
    fit_comb_mle = CombRawE.copy()
    fit_comb_mle.profile = fit_comb_2D.copy()
    fit_comb_mle.slice_range_index(range_lim=[0,beta_aer.profile.shape[1]])
    fit_comb_mle.descript = 'Maximum Likelihood Estimate of ' + fit_comb_mle.descript
    
    xvalid_mle = beta_aer.copy()
    xvalid_mle.profile = xvalid2D[:,:beta_aer.profile.shape[1]].copy()
    xvalid_mle.descript = 'Maximum Likelihood Estimated Data Points'
    xvalid_mle.label = 'MLE Data Point Mask'
    xvalid_mle.profile_type = 'MLE Data Point Mask'
    xvalid_mle.profile_variance = xvalid_mle.profile*0
    
    
#    if use_mask:
#        NanMask = np.logical_or(Molecular.profile < 4.0,CombHi.profile < 4.0)
#        beta_merge.profile = np.ma.array(beta_merge,mask=NanMask)


    return beta_merge,beta_a_mle,sLR_mle,alpha_a_mle,fit_mol_mle,fit_comb_mle,xvalid_mle,ProfileLogError,cal_params
    
    
    
def LREstimate_buildConst(geo_correct,MolScale,Cam,beta_m,beta_a,range_array,dt):
    geofun = 1/np.interp(range_array,geo_correct['geo_prof'][:,0],geo_correct['geo_prof'][:,1])
    ConstTerms = 1.0/geo_correct['Nprof']*MolScale*(beta_m+Cam*beta_a)*geofun*np.exp(-2.0*np.cumsum(8*np.pi/3*beta_m)*(range_array[1]-range_array[0]))/range_array**2
    #dt/geo_correct['tres']
    return ConstTerms
    
def ProfilesTotalxvalid(x,xvalid,mol_bs_coeff,Const,Mprof_bg=0,Cprof_bg=0,ix=3):
    """
    ix is the number of additional fit variables (e.g. gain, cross talk)
        held in the optimization variable.
    """
    N = mol_bs_coeff.size  # length of profile    
    Nx = xvalid.size
    Cam = x[0]
    Gm = x[1]  # molecular gain
    Gc = x[2]   # combinded gain
    sLR = np.zeros(N)
    sLR[xvalid] = np.exp(x[ix:Nx+ix])  # lidar ratio terms
    Baer = np.zeros(N)
    Baer[xvalid] = np.exp(x[Nx+ix:])  # aerosol backscatter terms
    Molmodel = Gm*(mol_bs_coeff+Cam*Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer))+Mprof_bg
    Combmodel = Gc*(mol_bs_coeff+Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer))+Cprof_bg
    return Molmodel,Combmodel
    
def ProfilesTotalxvalid2(x,xvalid,mol_bs_coeff,Const,Mprof_bg=0,Cprof_bg=0,dt=1.0):
    """
    dt sets the adjustment factor to convert counts to count rate needed in
    deadtime correction
    """
    ix = 7  # sets the profile offset
    N = mol_bs_coeff.size  # length of profile    
    Nx = xvalid.size
    Cam = x[0]
    Gm = x[1]  # molecular gain
    Gc = x[2]   # combinded gain
    sLR = np.zeros(N)
    sLR[xvalid] = np.exp(x[ix:Nx+ix])  # lidar ratio terms
    deadtimeMol=x[3]
    deadtimeComb=x[4]
    bgMol = x[5]
    bgComb = x[6]
    Baer = np.zeros(N)
    Baer[xvalid] = np.exp(x[Nx+ix:])  # aerosol backscatter terms
    Molmodel = Gm*(mol_bs_coeff+Cam*Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer))+Mprof_bg*bgMol
    Combmodel = Gc*(mol_bs_coeff+Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer))+Cprof_bg*bgComb
    Molmodel = Molmodel*dt/(dt+Molmodel*deadtimeMol)   
    Combmodel = Combmodel*dt/(dt+Combmodel*deadtimeComb) 
    
    return Molmodel,Combmodel

def LREstimateTotalxvalid2(x,xvalid,Mprof,Cprof,mol_bs_coeff,Const,lam,Mprof_bg=0,Cprof_bg=0,dt=1.0,weights=np.array([1])):
    N = Mprof.size  # length of profile    
    ix = 7  # sets the profile offset
    Nx = xvalid.size

    sLR = np.zeros(N)
    sLR[xvalid] = np.exp(x[ix:Nx+ix])  # lidar ratio terms
    Baer = np.zeros(N)
    Baer[xvalid] = np.exp(x[Nx+ix:])  # aerosol backscatter terms
    BaerExp = np.zeros(N)
    BaerExp[xvalid] = x[Nx+ix:]  # aerosol backscatter exponent terms

    Molmodel,Combmodel = ProfilesTotalxvalid2(x,xvalid,mol_bs_coeff,Const,Mprof_bg=Mprof_bg,Cprof_bg=Cprof_bg,dt=dt)
    dxvalid = np.diff(xvalid)
    deriv = np.nansum(np.abs(np.diff(sLR[xvalid])/dxvalid))*lam[0] + np.nansum(np.abs(np.diff(BaerExp[xvalid])/dxvalid))*lam[1]
    ErrRet = np.nansum(weights*(Molmodel-Mprof*np.log(Molmodel)))+np.nansum(weights*(Combmodel-Cprof*np.log(Combmodel)))+deriv
    return ErrRet
    
def LREstimateTotalxvalid2_prime(x,xvalid,Mprof,Cprof,mol_bs_coeff,Const,lam,Mprof_bg=0,Cprof_bg=0,dt=1.0,weights=np.array([1])):
    ix = 7  # sets the profile offset
    N = Mprof.size  # length of profile    
    Nx = xvalid.size
    Cam = x[0]
    Gm = x[1]  # molecular gain 
    Gc = x[2]  # combined gain
    
    deadtimeMol=x[3]
    deadtimeComb=x[4]
    bgMol = x[5]
    bgComb = x[6]
    
    sLR = np.zeros(N)
    sLR[xvalid] = np.exp(x[ix:Nx+ix])  # lidar ratio terms
    Baer = np.zeros(N)
    Baer[xvalid] = np.exp(x[Nx+ix:])  # aerosol backscatter terms
    BaerExp = np.zeros(N)
    BaerExp[xvalid] = x[Nx+ix:]  # aerosol backscatter exponent terms

    # obtain models including nonlinear response
    Molmodel,Combmodel = ProfilesTotalxvalid2(x,xvalid,mol_bs_coeff,Const,Mprof_bg=Mprof_bg,Cprof_bg=Cprof_bg,dt=dt)
    
    # obtain models without nonlinear response but including background
    Molmodel1,Combmodel1 = ProfilesTotalxvalid(x,xvalid,mol_bs_coeff,Const,Mprof_bg=Mprof_bg*bgMol,Cprof_bg=Cprof_bg*bgComb,ix=ix)
    
    #obtain models without nonlinear response or background
    Molmodel0 = Molmodel1-Mprof_bg*bgMol
    Combmodel0 = Combmodel1-Cprof_bg*bgComb


    
    # useful definitions for gradient calculations
    e0m = weights*(1-Mprof/Molmodel)
    e0c = weights*(1-Cprof/Combmodel)
    e_dtm = dt**2/(dt+Molmodel1*deadtimeMol)**2
    e_dtc = dt**2/(dt+Combmodel1*deadtimeComb)**2
    grad0m = np.sum(e0m*e_dtm*Molmodel0)-np.cumsum(e0m*e_dtm*Molmodel0)
    grad0c = np.sum(e0c*e_dtc*Combmodel0)-np.cumsum(e0c*e_dtc*Combmodel0)
    
    # lidar ratio gradient terms:
    gradErrS = -2*Baer*sLR*(grad0m+grad0c)
    gradErrS[np.nonzero(np.isnan(gradErrS))] = 0

    # backscatter cross section gradient terms:
    gradErrB = -2*sLR*Baer*(grad0m+grad0c)+Baer*e0m*Gm*Const*Cam*np.exp(-2*np.cumsum(sLR*Baer))+Baer*e0c*Const*Gc*np.exp(-2*np.cumsum(sLR*Baer))
    gradErrB[np.nonzero(np.isnan(gradErrB))] = 0

    # cross talk gradient term
    gradErrCam = np.nansum(e0m*e_dtm*Gm*Const*Baer*np.exp(-2*np.cumsum(sLR*Baer)))

    # combined gain gradient term
    gradErrGc = np.nansum(e0c*e_dtc*(mol_bs_coeff+Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer)))
    
    # molecular gain gradient term
    gradErrGm = np.nansum(e0m*e_dtm*(mol_bs_coeff+Cam*Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer)))
    
    # molecular dead time gradient term
    gradErrDTm = np.nansum(-e0m*dt*Molmodel1**2/(dt+Molmodel1*deadtimeMol))
    
    # combined dead time gradient term
    gradErrDTc = np.nansum(-e0c*dt*Combmodel1**2/(dt+Combmodel1*deadtimeComb))
    
    # molecular background adjustment term
    gradErrBGm = np.nansum(e0m*e_dtm*Mprof_bg)
    
    # combined background adjustment term
    gradErrBGc = np.nansum(e0c*e_dtc*Cprof_bg)

    # total variance gradient terms
#    gradErr[np.nonzero(np.isnan(gradErr))] = 0
#    gradErr = gradErr[xvalid]
    dxvalid = np.diff(xvalid)
    gradErrStv = np.zeros(Nx)
    gradErrBtv = np.zeros(Nx)
    gradpenS = lam[0]*np.sign(np.diff(sLR[xvalid]))/dxvalid
    gradpenS[np.nonzero(np.isnan(gradpenS))] = 0
    gradErrStv[:-1] = gradErrStv[:-1]-gradpenS
    gradErrStv[1:] = gradErrStv[1:]+gradpenS
    
    gradpenB = lam[1]*np.sign(np.diff(BaerExp[xvalid]))/dxvalid
    gradpenB[np.nonzero(np.isnan(gradpenS))] = 0
    gradErrBtv[:-1] = gradErrBtv[:-1]-gradpenB
    gradErrBtv[1:] = gradErrBtv[1:]+gradpenB
    
    gradErr = np.zeros(ix+2*Nx)
    gradErr[0] = gradErrCam
    gradErr[1] = gradErrGm
    gradErr[2] = gradErrGc
    gradErr[3] = gradErrDTm
    gradErr[4] = gradErrDTc
    gradErr[5] = gradErrBGm
    gradErr[6] = gradErrBGc
    gradErr[ix:Nx+ix] = gradErrS[xvalid]+gradErrStv
    gradErr[Nx+ix:] = gradErrB[xvalid]+gradErrBtv
    
    return gradErr.flatten()

def MLE_Cals_1D(prof_time,MolRaw,CombRaw,beta_aer,surf_temp,surf_pres,geo_data,\
    Nmax=1000,minSNR=0.5,plotfigs=True,print_sol=True,verify=True,lam_array=np.array([np.nan]),verbose=False):
        
    """
    Calculates the MLE for aerosol backscatter coefficient, extinction and
        lidar ratio of a single profile occuring at the specified time.
        
    Inputs:
        prof_time - time in hours of the desired profile
        MolRaw - Raw molecular data
        CombRaw - Raw combined data
        beta_aer - aerosol backscatter coefficient from direct retrieval
        surf_temp - surface temperature array (in time)
        surf_pres - surface pressure array (in time)
        geo_data - data loaded from geo overlap file
        Nmax - maximum number of iterations on the optimizor (default 1000)
        minSNR - minimum aerosol SNR to be included in retrieval (default 0.5)
        plotfigs - plots results if set to True (default True)
        print_sol - prints the calibrations for the solution if True (default True)
        verify - uses poisson thinning to verify solution (default True)
        lam_array - array of TV penalty values to be evaluated (default 1e-5-1e-2)
        verbose - outputs results of each TV evaluation if True (default False)
    
    Outputs:
        aer_bs_sol - array of retrieved aerosol backscatter coefficeint
        sLRsol - array of retrieved lidar ratio
        xvalid1 - array of points included in optimization
    
    """

    b = np.argmin(np.abs(beta_aer.time/3600-prof_time))  # 6.3, 15 9.58 -water cloud
    Interval = 1  # going much bigger than this will use too much memory on Burble
#    minSNR = 0.5 # min aerosol snr to include as an optimization parameter.  Typically 2.0
    
    MolRawE = MolRaw.copy()
    CombRawE = CombRaw.copy()
    
    MolRawE.profile = MolRawE.profile*MolRawE.NumProfList[:,np.newaxis]
    MolRawE.profile_variance = MolRawE.profile_variance*MolRawE.NumProfList[:,np.newaxis]**2
    CombRawE.profile = CombRawE.profile*CombRawE.NumProfList[:,np.newaxis]
    CombRawE.profile_variance = CombRawE.profile_variance*CombRawE.NumProfList[:,np.newaxis]**2
    
    
    
    if verify:
        [MolRawF,MolRawV] = MolRawE.p_thin()
        [CombRawF,CombRawV] = CombRawE.p_thin()
        thin_adj = 0.5  # adjust gain on estimator if profiles have been poisson thinned
    else:
        MolRawF = MolRawE.copy()
        MolRawV = MolRawE
        CombRawF = CombRawE.copy()
        CombRawV = CombRawE
        thin_adj = 1.0
    
    # lower resolution profiles to obtain better cloud detection
    beta_aer_E = beta_aer.copy()
    beta_aer_E.conv(4,2)
    
    
    Nprof = MolRawE.NumProfList[b:b+Interval]
    FitMol1 = MolRawF.profile[b:b+Interval,:].flatten()
    FitMol_bg1 = np.nanmean(MolRawF.profile[b:b+Interval,-50:],axis=1)
    
    FitComb1 = CombRawF.profile[b:b+Interval,:].flatten()
    FitComb_bg1 = np.nanmean(CombRawF.profile[b:b+Interval,-50:],axis=1)
    
    ValidMol1 = MolRawV.profile[b:b+Interval,:].flatten()
    ValidComb1 = CombRawV.profile[b:b+Interval,:].flatten()    
    
    dR = MolRawE.mean_dR
    
    FitAer = beta_aer_E.profile[b:b+Interval,:]
    if minSNR > 0:
        xvalid1 = np.nonzero((beta_aer_E.profile[b:b+Interval,:]/np.sqrt(beta_aer_E.profile_variance[b:b+Interval,:])).flatten()>minSNR)[0]
    else:
        if hasattr(beta_aer.profile,'mask'):
            xvalid1 = np.arange(beta_aer.range_array.size)[np.logical_not(beta_aer.profile.mask[b,:])]
        else:
            xvalid1 = np.arange(beta_aer.range_array.size)
    FitAer = np.concatenate((FitAer.flatten(),np.zeros(FitMol1.size-FitAer.size)))
    FitAer[np.nonzero(np.isnan(FitAer))[0]]=0
    #
    FitAerFilt = np.zeros(FitAer.shape)
    FitAerFilt[xvalid1] = FitAer[xvalid1]
    
    #stopIndex = MolRaw.range_array.size
    
    #stopIndex = 150
    #
    #FitMol = Molecular.profile[b,:stopIndex]+Molecular.bg[b]*1.398230
    #FitMol_bg = Molecular.bg[b]*1.398230
    #
    #FitComb = CombHi.profile[b,:stopIndex]+CombHi.bg[b]
    #FitComb_bg = CombHi.bg[b]
    
    lam = np.array([0.10,0.01])  # Penalty for [Lidar ratio, Backscatter Coefficient]  [4,0.000001]
    
    if verify:
        if np.isnan(lam_array).any():
            lam_array = np.logspace(-5,-2,47)  
    else:
        lam_array = np.array([0])
    fitErrors = np.zeros(lam_array.size)
    sol_List = []
    out_cond_array = np.zeros(lam_array.size)
    
    #lam = np.array([4.0,0.0000001])
    #lamCom = np.array([0.1,10e-3])
    sLRinitial = 35
    
    #beta_m,sonde_time,sonde_index_prof = lp.get_beta_m_sonde(MolRawE,Years,Months,Days,sonde_path,interp=True)
    #beta_m_sonde1 = beta_m.profile[b,:]
    
    #if model_atm: 
    beta_m,temp,pres = lp.get_beta_m_model(MolRawE,surf_temp,surf_pres,returnTP=True)
    pres.descript = 'Ideal Atmosphere Pressure in atm'
    #else:
    #    beta_m,sonde_time,sonde_index_prof,temp,pres,sonde_index = lp.get_beta_m_sonde(MolRawE,Years,Months,Days,sonde_path,interp=True,returnTP=True)
    #    pres.descript = 'Sonde Measured Pressure in atm'
    ## convert pressure from Pa to atm.
    pres.gain_scale(9.86923e-6)  
    pres.profile_type = '$atm.$'
    beta_m_sonde1 = beta_m.profile[b,:]

    #Tsonde = np.interp(MolRawE.range_array,SondeAlt[sonde_index,:]-StatElev[sonde_index],TempDat[sonde_index,:])
    #Psonde = np.interp(MolRawE.range_array,SondeAlt[sonde_index,:]-StatElev[sonde_index],PresDat[sonde_index,:])
    #beta_m_sonde = 5.45*(550.0/780.24)**4*1e-32*Psonde/(Tsonde*lp.kB)
    
    ##Mol_Beta_Scale = 2.7e16
    ##Comb_Beta_Scale = 5.3e6
    #Mol_Beta_Scale = 1/1.5
    #Comb_Beta_Scale = Mol_Beta_Scale/MolGain
    #Cam = 0.1  #0.12, (0.033,0.08)


#    geo_cor1 = geo_data['geo_prof']
#    geofun = 1/np.interp(MolRawE.range_array,geo_cor1[:,0],geo_cor1[:,1])

    # omit molecular and aerosol backscatter coefficeint terms
    # omit cross talk and molecular gain terms
    # LREstimate_buildConst(geo_correct,MolScale,Cam,beta_m,beta_a,range_array):
    # LREstimate_buildConst(geo_correct,MolScale,Cam,beta_m,beta_a,range_array,dt):
    ConstTerms1 = thin_adj*Nprof*LREstimate_buildConst(geo_data,1.0,0.0,beta_m_sonde1,np.zeros(FitComb1.size),MolRawE.range_array,MolRaw.mean_dt)
    ConstTerms1 = ConstTerms1/beta_m_sonde1

    #(NumProf*MolGain_Adj*beta_m_2D*geo_est[np.newaxis,:]/MolRaw.range_array**2).T; MolRaw.mean_dt/geo_data['Nprof']/geo_data['tres']*geofun*
    if plotfigs:
        plt.figure(); plt.semilogy(0.75*ConstTerms1*beta_m_sonde1); plt.semilogy(FitMol1.flatten()-FitMol_bg1)
        plt.semilogy(0.9*ConstTerms1*(FitAerFilt+beta_m_sonde1)*np.exp(-2*np.cumsum(sLRinitial*FitAerFilt)))
        plt.semilogy(FitComb1-FitComb_bg1)
#

    #weightfun = np.ones(FitAerFilt.size) 
    #weightfun = 0.001/(0.001+(FitAerFilt/np.mean(FitAerFilt))**2)
#    weightfun = np.ones(FitComb1.size) #np.exp(-(FitAerFilt/1e-6)**2)

    # adjustment factor to obtain count rate from photon counts
    rate_adj1 = 1.0/(MolRawE.shot_count[b]*MolRawE.NumProfList[b]*MolRawE.binwidth_ns*1e-9)

    ##LREstimateTotal(x,Mprof,Cprof,mol_bs_coeff,Const,lam,Mprof_bg=0,Cprof_bg=0,weights=np.array([1])):
    #FitProfMol = lambda x: 1e-3*LREstimateTotalxvalid(x,xvalid,FitMol,FitComb,beta_m_sonde,ConstTerms,lam,Mprof_bg=FitMol_bg,Cprof_bg=FitComb_bg,weights=weightfun)
    #FitProfMolDeriv = lambda x: 1e-3*LREstimateTotalxvalid_prime(x,xvalid,FitMol,FitComb,beta_m_sonde,ConstTerms,lam,Mprof_bg=FitMol_bg,Cprof_bg=FitComb_bg,weights=weightfun)

    for i_lam in range(lam_array.size):
        if verify:
            lam = np.array([lam_array[i_lam]]*2)
        else:
            lam = np.array([0.10,0.01])  # Penalty for [Lidar ratio, Backscatter Coefficient]  [4,0.000001]
#        lam = np.array([lam_array[i_lam],0.01])
        prefactor = 1.0# 1e-7
        # Function includes nonlinear count correction and background adjustement
        # LREstimateTotalxvalid2(x,xvalid,Mprof,Cprof,mol_bs_coeff,Const,lam,Mprof_bg=0,Cprof_bg=0,dt=1.0,weights=np.array([1])):
        FitProfMol1 = lambda x: prefactor*LREstimateTotalxvalid2(x,xvalid1,FitMol1,FitComb1,beta_m_sonde1,ConstTerms1,lam,Mprof_bg=FitMol_bg1,Cprof_bg=FitComb_bg1,dt=rate_adj1)
        FitProfMolDeriv1 = lambda x: prefactor*LREstimateTotalxvalid2_prime(x,xvalid1,FitMol1,FitComb1,beta_m_sonde1,ConstTerms1,lam,Mprof_bg=FitMol_bg1,Cprof_bg=FitComb_bg1,dt=rate_adj1)
    
    
        # example code for testing gradient calculation:
        #gradTest = FitProfMolDeriv(x0)
        #gradTestNum = Num_Gradient(FitProfMol,x0)
        #plt.figure(); plt.plot(gradTest); plt.plot(gradTestNum);
        
        
        #bndsP = np.ones((3+2*xvalid.size,2))
        #bndsP[3:(3+xvalid.size),1] = 1e5
        #bndsP[(3+xvalid.size):,0] = 0.0
        #bndsP[(3+xvalid.size):,1] = 1e-2
        #bndsP[0,0] = 0.0
        #bndsP[0,1] = 1.0
        #bndsP[1,0] = 0.01
        #bndsP[1,1] = 5
        #bndsP[2,0] = 0.01
        #bndsP[2,1] = 5
        
        
        ix = 7  # index where profiles start
        bndsP = np.zeros((ix+2*xvalid1.size,2))
        bndsP[ix:(ix+xvalid1.size),1] = np.log(1e6)
        bndsP[(ix+xvalid1.size):,0] = np.log(1e-12)
        bndsP[(ix+xvalid1.size):,1] = np.log(1e-1)
        bndsP[0,0] = 0.0  # cross talk
        bndsP[0,1] = 0.10
        bndsP[1,0] = 0.01  # molecular gain
        bndsP[1,1] = 5
        bndsP[2,0] = 0.01  # combined gain
        bndsP[2,1] = 5
        
        bndsP[3,0] = 0.0  # molecular deadtime
        bndsP[3,1] = 0.1e-6
        bndsP[4,0] = 0.0 # combined deeadtime
        bndsP[4,1] = 0.1e-6
        
        bndsP[5,0] = 0.9 # molecular background factor
        bndsP[5,1] = 1.1
        bndsP[6,0] = 0.9 # combined background factor
        bndsP[6,1] = 1.1
    
        #FitProfMol = lambda x: 1000*LREstimateExp(x,FitMol,FitAer,beta_m_sonde,ConstTerms,lamMol,xvalid,Mprof_bg=FitMol_bg,weights=weightfun)
        #FitProfMolDeriv = lambda x: 1000*LREstimateExp_prime(x,FitMol,FitAer,beta_m_sonde,ConstTerms,lamMol,xvalid,Mprof_bg=FitMol_bg,weights=weightfun)
        
        #bndsP = np.zeros((xvalid.size,2))
        #bndsP[:,1] = 12
        
        x0 = np.log(sLRinitial*dR)*np.ones(bndsP.shape[0]); #np.random.rand(xvalid.size)   #np.random.rand(FitMol.size)*4000  #1200.0*np.ones(FitMol.shape)
        x0[ix:(ix+xvalid1.size)] = np.log((-8.5*np.log10(FitAer[xvalid1])-24.5)*dR)
        #x0[(ix+xvalid1.size):] = np.log(FitAer[xvalid1])  # aerosol backscatter
        x0[(ix+xvalid1.size):] = np.log(beta_aer.profile[b,xvalid1])  # aerosol backscatter
        x0[0] = 0.03  # Cam
        x0[1] = 0.75  # Gm
        x0[2] = 1.00  # Gt
        x0[3] = 48e-9  # molecular deadtime
        x0[4] = 10e-9  # combined deadtime
        x0[5] = 1.0  # molecular gain adjustment
        x0[6] = 1.0 # combined gain adjustment
    
        #x0[np.nonzero(x0 < bndsP[:,0])] = bndsP[np.nonzero(x0 < bndsP[:,0]),0]
        #x0[np.nonzero(x0 > bndsP[:,1])] = bndsP[np.nonzero(x0 > bndsP[:,1]),1]
        
        #Mol0 = x0[1]*(beta_m_sonde+x0[0]*x0[(3+FitComb.size):])*ConstTerms*np.exp(-2*np.cumsum(x0[3:(3+FitComb.size)]*x0[(3+FitComb.size):]))+FitMol_bg
        #Comb0 = x0[2]*(beta_m_sonde+x0[(3+FitComb.size):])*ConstTerms*np.exp(-2*np.cumsum(x0[3:(3+FitComb.size)]*x0[(3+FitComb.size):]))+FitComb_bg
        
        # no nonlinear correction
        #Mol0,Comb0 = ProfilesTotalxvalid(x0,xvalid,beta_m_sonde,ConstTerms,Mprof_bg=FitMol_bg,Cprof_bg=FitComb_bg)
        ## nonlinear correction
        Mol0,Comb0 = ProfilesTotalxvalid2(x0,xvalid1,beta_m_sonde1,ConstTerms1,Mprof_bg=FitMol_bg1,Cprof_bg=FitComb_bg1,dt=rate_adj1)
        
        #plt.figure();
        #plt.semilogy(FitMol)
        #plt.semilogy(Mol0)
        #
        #plt.figure();
        #plt.semilogy(FitComb)
        #plt.semilogy(Comb0)
    #    dt0 = time()
        #wMol = scipy.optimize.fmin_slsqp(FitProfMol1,x0,bounds=bndsP,fprime=FitProfMolDeriv1 ,iter=1000,acc=1e-14) # fprime=FitProfMolDeriv 
        wMol,fx_iter,out_cond = scipy.optimize.fmin_tnc(FitProfMol1,x0,bounds=bndsP,fprime=FitProfMolDeriv1 ,maxfun=Nmax,eta=1e-7) # fprime=FitProfMolDeriv 
        #wMol,fxEr,out_cond = scipy.optimize.fmin_l_bfgs_b(FitProfMol1,x0,bounds=bndsP,fprime=FitProfMolDeriv1,maxiter=1000,factr=10.0,pgtol=1e-5) # fprime=FitProfMolDeriv 
    #    dt1 = time()-dt0
        
        aer_bs_sol = np.zeros(FitComb1.size)
        aer_bs_sol[xvalid1] = np.exp(wMol[(ix+xvalid1.size):])
        sLRsol = np.zeros(FitComb1.size)
        sLRsol[xvalid1] = np.exp(wMol[ix:(ix+xvalid1.size)])
        Cam = wMol[0]
        Gm = wMol[1]
        Gc = wMol[2]
    
    
        #fitMolsol,fitCombsol = ProfilesTotalxvalid(wMol,xvalid,beta_m_sonde,ConstTerms,Mprof_bg=FitMol_bg,Cprof_bg=FitComb_bg)
        fitMolsol,fitCombsol = ProfilesTotalxvalid2(wMol,xvalid1,beta_m_sonde1,ConstTerms1,Mprof_bg=FitMol_bg1,Cprof_bg=FitComb_bg1,dt=rate_adj1)
    
        #fitMolsol = Gm*ConstTerms*(beta_m_sonde+Cam*aer_bs_sol)*np.exp(-2*np.cumsum(sLRsol*aer_bs_sol))+FitMol_bg
        #fitCombsol = Gc*ConstTerms*(beta_m_sonde+aer_bs_sol)*np.exp(-2*np.cumsum(sLRsol*aer_bs_sol))+FitComb_bg
        
    #    fitMolErr = np.sqrt(np.sum((fitMolsol-FitMol1)**2))
    #    fitCombErr = np.sqrt(np.sum((fitCombsol-FitComb1)**2))
        
    #    ProfileError = np.sqrt(np.sum((FitComb1-fitCombsol)**2+(FitMol1-fitMolsol)**2))
        ProfileLogError = np.nansum(fitCombsol-ValidComb1*np.log(fitCombsol)) + np.nansum(fitMolsol-ValidMol1*np.log(fitMolsol))

        fitErrors[i_lam] = ProfileLogError
        sol_List.extend([wMol])
        out_cond_array[i_lam] = out_cond
        
        #gradient = LREstimateTotalxvalid_prime(wMol,xvalid,FitMol,FitComb,beta_m_sonde,ConstTerms,np.array([0,0]),Mprof_bg=FitMol_bg,Cprof_bg=FitComb_bg)
        #Jac_sLR_m,Jac_sLR_c,Jac_Baer_m,Jac_Baer_c,Jac_Cam,Jac_Gc,Jac_Gm = \
        #    LREstimateTotalxvalid_Jacobian(wMol,xvalid,beta_m_sonde,ConstTerms)
        #
        #J_sLR_m_inv = np.linalg.pinv(Jac_sLR_m)
        #ErrMol = np.diag(np.matrix(J_sLR_m_inv)*np.matrix(np.diag(FitMol))*np.matrix(J_sLR_m_inv).T)
        #
        #J_sLR_c_inv = np.linalg.pinv(Jac_sLR_c)
        #ErrComb = np.diag(np.matrix(J_sLR_c_inv)*np.matrix(np.diag(FitComb))*np.matrix(J_sLR_c_inv).T)
        #
        #sLRsol_u = np.zeros(FitComb.size)
        #sLRsol_u[xvalid] = np.sqrt(ErrComb+ErrMol)
        ##sLRsol_u[xvalid] = np.exp(wMol[3:(3+xvalid.size)]+np.sqrt(ErrComb+ErrMol))
        ##
        ##sLRsol_l = np.zeros(FitComb.size)
        ##sLRsol_l[xvalid] = np.exp(wMol[3:(3+xvalid.size)]-np.sqrt(ErrComb+ErrMol))
        
        #gradsLR = np.zeros(FitComb.size)
        #gradsLR[xvalid] = 1.0/gradient[3:(3+xvalid.size)]/np.sum(1.0/gradient)
        if print_sol and verbose:
            print('lam: %f\nCam = %f\nGm = %f\nGc = %f\nMol DT = %f ns\nComb DT = %f ns\nFitError:%f'%(lam[0],Cam,Gm,Gc,wMol[3]*1e9,wMol[4]*1e9,ProfileLogError))
            print('Output Flag: %d'%out_cond)
            print('Output Flag Definition: %s'%scipy.optimize.tnc.RCSTRINGS[out_cond])
    
    
    isol = np.argmin(fitErrors)
        
    fitMolsol,fitCombsol = ProfilesTotalxvalid2(sol_List[isol],xvalid1,beta_m_sonde1,ConstTerms1,Mprof_bg=FitMol_bg1,Cprof_bg=FitComb_bg1,dt=rate_adj1)
    aer_bs_sol = np.zeros(FitComb1.size)
    aer_bs_sol[xvalid1] = np.exp(sol_List[isol][(ix+xvalid1.size):])
    sLRsol = np.zeros(FitComb1.size)
    sLRsol[xvalid1] = np.exp(sol_List[isol][ix:(ix+xvalid1.size)])
    Cam = sol_List[isol][0]
    Gm = sol_List[isol][1]
    Gc = sol_List[isol][2]      
    if print_sol and verify:
        print('Final Solution:')
        print('lam: %f\nCam = %f\nGm = %f\nGc = %f\nMol DT = %f ns\nComb DT = %f ns\nFitError:%f'%(lam_array[isol],Cam,Gm,Gc,sol_List[isol][3]*1e9,sol_List[isol][4]*1e9,fitErrors[isol]))
        print('Output Flag: %d'%out_cond_array[isol])
        print('Output Flag Definition: %s'%scipy.optimize.tnc.RCSTRINGS[out_cond_array[isol]])
    
    if plotfigs:
        if verify:
            plt.figure()
            plt.semilogx(lam_array,fitErrors)
            plt.ylabel('Log Error')
            plt.xlabel('Fit Index')       
              
        
        plt.figure();
        plt.subplot(1,2,1)
        plt.semilogx(ValidMol1,MolRawE.range_array*1e-3)
        plt.semilogx(fitMolsol,MolRawE.range_array*1e-3)
        plt.grid(b=True)
        plt.title('Molecular')
        plt.xlabel('Photon Counts')
        plt.ylabel('Altitude [km]')
        plt.legend(('Data','Fit'))
        plt.subplot(1,2,2)
        plt.semilogx(ValidComb1,MolRawE.range_array*1e-3)
        plt.semilogx(fitCombsol,MolRawE.range_array*1e-3)
        plt.grid(b=True)
        plt.title('Combined')
        plt.xlabel('Photon Counts')
        
        plt.figure()
        plt.subplot(1,3,1)
        plt.semilogx(aer_bs_sol,MolRawE.range_array*1e-3)
        plt.semilogx(beta_aer.profile[b,:],beta_aer.range_array*1e-3)
        plt.semilogx(beta_m_sonde1/Cam,MolRawE.range_array*1e-3,'k--')
        plt.title('Backscatter')
        plt.ylim([0,15])
        plt.xlim([1e-9,2e-4])
        plt.ylabel('Altitude [km]')
        plt.xlabel(r'$\beta_a$ [$m^{-1}sr^{-1}$]')
        plt.grid(b=True)
        plt.legend(('MLE','Direct','Maximum'))
        plt.subplot(1,3,2)
        plt.semilogx(FitAer[xvalid1]*(-8.5*np.log10(FitAer[xvalid1])-24.5), MolRawE.range_array[xvalid1]*1e-3,'rx')
        plt.semilogx(sLRsol*aer_bs_sol/MolRawE.mean_dR,MolRawE.range_array*1e-3,'b-')
        #plt.semilogx(Extinction.profile[b,:],Extinction.range_array*1e-3,'g-')
        plt.title('Extinction')
        plt.xlabel(r'$\alpha_a$ [$m^{-1}$]')
        plt.ylim([0,15])
        plt.xlim([1e-6,2e-2])
        plt.grid(b=True)
        plt.legend(('Initial','MLE','Direct'))
        plt.subplot(1,3,3)
        plt.plot((-8.5*np.log10(FitAer[xvalid1])-24.5), MolRawE.range_array[xvalid1]*1e-3,'rx')
        plt.plot(sLRsol/MolRawE.mean_dR,MolRawE.range_array*1e-3,'b-')
        plt.ylim([0,15])
        plt.title('Lidar Ratio')
        plt.xlabel(r'$s$ [$sr$]')
        plt.grid(b=True)
        plt.legend(('Initial','MLE'))

    return aer_bs_sol,sLRsol,xvalid1
    
    
    
    
def Build_GVHSRL_Profiles(x,Const,dt=1.0,ix=8,dR=7.5,return_params=False,params={}):
    """
    dt sets the adjustment factor to convert counts to count rate needed in
    deadtime correction
    
    if return_params=True, returns the profiles of the optical parameters
    """
    
    try:
        beta_aer = params['Backscatter_Coefficient']
        p_aer = params['Polarization']
        sLR = params['Lidar_Ratio']
        Tatm = params['Tatm']
    except KeyError:
        var0 = list(Const.keys())[0]
        N = Const[var0]['mol'].shape[1]
        tdim = Const[var0]['mol'].shape[0]
        
        x2D = x[ix:].reshape((tdim,3*N))    
        
        beta_aer = np.exp(x2D[:,N:2*N])
        p_aer = np.arctan(x2D[:,2*N:])/np.pi+0.5
        sLR = np.exp(x2D[:,:N])+1
    
        Tatm = np.exp(-2*np.cumsum(beta_aer*sLR,axis=1)*dR)
        Tatm[:,1:] = Tatm[:,:-1]
        Tatm[:,0] = 0
    
    forward_profs = {}
    for var in Const.keys():
        if 'Molecular' in var:
            forward_profs[var] = x[0]*Const[var]['mult']*(beta_aer*(Const[var]['pol'][0]+p_aer*Const[var]['pol'][1])+Const[var]['mol'])*Tatm+Const[var]['bg']
            forward_profs[var] = forward_profs[var]*dt[:,np.newaxis]/(dt[:,np.newaxis]+forward_profs[var]*np.exp(x[4]))   
        elif 'High' in var:
            forward_profs[var] = x[1]*Const[var]['mult']*(beta_aer*(Const[var]['pol'][0]+p_aer*Const[var]['pol'][1])+Const[var]['mol'])*Tatm+Const[var]['bg']
            forward_profs[var] = forward_profs[var]*dt[:,np.newaxis]/(dt[:,np.newaxis]+forward_profs[var]*np.exp(x[5]))  
        elif 'Low' in var:
            forward_profs[var] = x[2]*Const[var]['mult']*(beta_aer*(Const[var]['pol'][0]+p_aer*Const[var]['pol'][1])+Const[var]['mol'])*Tatm+Const[var]['bg']
            forward_profs[var] = forward_profs[var]*dt[:,np.newaxis]/(dt[:,np.newaxis]+forward_profs[var]*np.exp(x[6]))   
        elif 'Cross' in var:
            forward_profs[var] = x[3]*Const[var]['mult']*(beta_aer*(Const[var]['pol'][0]+p_aer*Const[var]['pol'][1])+Const[var]['mol'])*Tatm+Const[var]['bg']
            forward_profs[var] = forward_profs[var]*dt[:,np.newaxis]/(dt[:,np.newaxis]+forward_profs[var]*np.exp(x[7]))
    
    if return_params:
        forward_profs['Backscatter_Coefficient'] = beta_aer.copy()
        forward_profs['Lidar_Ratio'] = sLR.copy()
        forward_profs['Polarization'] = p_aer.copy()
        forward_profs['Tatm'] = Tatm.copy()
        forward_profs['xB'] = x2D[:,N:2*N]
        forward_profs['xSLR'] = x2D[:,:N]
        forward_profs['xPol'] = x2D[:,2*N:]
        
    
    return forward_profs
    
def built_tv_mesh(l_time,l_pos_x,l_pos_y,l_pos_z):
    """
    Builds a temporal mesh for evaluating TV error on airborne platforms
    Requires matrix of 
        l_time - time associated with each data point
        l_pos_x - x position of each data point
        l_pos_y - y position of each data point
        l_pos_z - z position of each data point
        
    returns a dict containing the indices to the nearest data point for 
        time after the given data point ('pos_h') and 
        time before a given data point ('neg_h') as well as 
        the distance between the data points ('d')
    """
    mesh_pts = {'pos_h':{'i':[np.zeros(l_time.shape,dtype=np.int),np.zeros(l_time.shape,dtype=np.int)],'d':np.ones(l_time.shape)},
                'neg_h':{'i':[np.zeros(l_time.shape,dtype=np.int),np.zeros(l_time.shape,dtype=np.int)],'d':np.ones(l_time.shape)}}
    
    for piT in range(l_time.shape[0]):
        for piR in range(l_time.shape[1]):
            # find the next closest past point
            if piT == 0:
                # if its the first time point, there is no previous point
                # to connect to
                # set to index to itself
                mesh_pts['neg_h']['i'][0][piT,piR] = piT
                mesh_pts['neg_h']['i'][1][piT,piR] = piR
            else:
                dpt = np.sqrt((l_pos_x[piT,piR]-l_pos_x[:piT,:])**2+(l_pos_y[piT,piR]-l_pos_y[:piT,:])**2+(l_pos_z[piT,piR]-l_pos_z[:piT,:])**2)
                i_min = np.unravel_index(dpt.argmin(),dpt.shape)
                mesh_pts['neg_h']['d'][piT,piR] = dpt[i_min]  # store the distance between points
                mesh_pts['neg_h']['i'][0][piT,piR] = i_min[0]
                mesh_pts['neg_h']['i'][1][piT,piR] = i_min[1]
                
            
            # find the next closest future point
            if piT == l_time.shape[0]-1:
                # if its the last time point, there is no future point
                # to connect to
                # set to index to itself
                mesh_pts['pos_h']['i'][0][piT,piR] = piT
                mesh_pts['pos_h']['i'][1][piT,piR] = piR
            else:
                dpt = np.sqrt((l_pos_x[piT,piR]-l_pos_x[piT+1:,:])**2+(l_pos_y[piT,piR]-l_pos_y[piT+1:,:])**2+(l_pos_z[piT,piR]-l_pos_z[piT+1:,:])**2)
                i_min = np.unravel_index(dpt.argmin(),dpt.shape)
                mesh_pts['pos_h']['d'][piT,piR] = dpt[i_min]
                i_min = (piT+1+i_min[0],i_min[1])            
                mesh_pts['pos_h']['i'][0][piT,piR] = i_min[0]
                mesh_pts['pos_h']['i'][1][piT,piR] = i_min[1]
    return mesh_pts

def GVHSRL_FitError(x,fit_profs,Const,lam,meshTV,dt=1.0,ix=8,weights=np.array([1])):
    
    """
    PTV Error of GV-HSRL profiles
    """

    dR = fit_profs['Raw_Molecular_Backscatter_Channel'].mean_dR
    forward_profs = Build_GVHSRL_Profiles(x,Const,dt=dt,ix=ix,dR=dR,return_params=True)
    
    # TV penalty for lidar ratio
    deriv = lam[0][1]*np.nansum(np.abs(np.diff(forward_profs['Lidar_Ratio'],axis=1)/dR))
    deriv = deriv + lam[0][0]*np.nansum(np.abs(forward_profs['Lidar_Ratio'].flatten()-forward_profs['Lidar_Ratio'][meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()])/meshTV['pos_h']['d'].flatten())
    deriv = deriv + lam[0][0]*np.nansum(np.abs(forward_profs['Lidar_Ratio'].flatten()-forward_profs['Lidar_Ratio'][meshTV['neg_h']['i'][0].flatten(),meshTV['neg_h']['i'][1].flatten()])/meshTV['neg_h']['d'].flatten())    
    
    # perform backscatter TV penalty in log space
    deriv = deriv + lam[1][1]*np.nansum(np.abs(np.diff(forward_profs['xB'],axis=1)/dR))
    deriv = deriv + lam[1][0]*np.nansum(np.abs(forward_profs['xB'].flatten()-forward_profs['xB'][meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()])/meshTV['pos_h']['d'].flatten())
    deriv = deriv + lam[1][0]*np.nansum(np.abs(forward_profs['xB'].flatten()-forward_profs['xB'][meshTV['neg_h']['i'][0].flatten(),meshTV['neg_h']['i'][1].flatten()])/meshTV['neg_h']['d'].flatten())

    # TV penalty for depolarization
    deriv = deriv + lam[2][1]*np.nansum(np.abs(np.diff(forward_profs['Polarization'],axis=1)/dR))
    deriv = deriv + lam[2][0]*np.nansum(np.abs(forward_profs['Polarization'].flatten()-forward_profs['Polarization'][meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()])/meshTV['pos_h']['d'].flatten()) 
    deriv = deriv + lam[2][0]*np.nansum(np.abs(forward_profs['Polarization'].flatten()-forward_profs['Polarization'][meshTV['neg_h']['i'][0].flatten(),meshTV['neg_h']['i'][1].flatten()])/meshTV['neg_h']['d'].flatten())     
    
    ErrRet = deriv
    for var in fit_profs:
#        print(var)
        ErrRet = ErrRet + np.nansum(weights*(forward_profs[var]-fit_profs[var].profile*np.log(forward_profs[var])))
        
    return ErrRet

def GVHSRL_FitError_Gradient(x,fit_profs,Const,lam,meshTV,dt=1.0,ix=8,weights=np.array([1])):
    """
    Analytical gradient of GVHSRL_FitError()
    """
    
    dR = fit_profs['Raw_Molecular_Backscatter_Channel'].mean_dR
    forward_profs = Build_GVHSRL_Profiles(x,Const,dt=dt,ix=ix,dR=dR,return_params=True)
#    N = forward_profs['Backscatter_Coefficient'].size
    tdim = forward_profs['Backscatter_Coefficient'].shape[0]
    N = forward_profs['Backscatter_Coefficient'].shape[1]
    
    # obtain models without nonlinear responde but including background
    xlin = x.copy()
    xlin[4] = 0
    xlin[5] = 0
    xlin[6] = 0
    xlin[7] = 0
    lin_profs = Build_GVHSRL_Profiles(xlin,Const,dt=dt,ix=ix,dR=dR,params=forward_profs)
    

    #obtain models without nonlinear response or background
    sig_profs = {}
    e0 = {}
    e_dt = {}
    grad0 = {}
    
    # gradient components of each atmospheric variable
    gradErrS = np.zeros((tdim,N)) # lidar ratio
    gradErrB = np.zeros((tdim,N)) # backscatter coefficient
    gradErrP = np.zeros((tdim,N)) # polarization = (1-d)
    
    gradErr = np.zeros(ix+3*N*tdim)
    gradErr2D = np.zeros((tdim,3*N))
    
    for var in fit_profs.keys():
        sig_profs[var] = lin_profs[var]-Const[var]['bg']
        
        if 'Molecular' in var:
            deadtime = np.exp(x[4])
            Gain = x[0]           
            
        elif 'High' in var:
            deadtime = np.exp(x[5])
            Gain = x[1]
            
        elif 'Low' in var:
            deadtime = np.exp(x[6])
            Gain = x[2]
            
        elif 'Cross' in var:
            deadtime = np.exp(x[7])
            Gain = x[3]   
            
        # useful definitions for gradient calculations
        e0[var] = (1-fit_profs[var].profile/forward_profs[var])  # error function derivative
        e_dt[var] = dt[:,np.newaxis]**2/(dt[:,np.newaxis]+lin_profs[var]*deadtime)**2  # dead time derivative
        grad0[var] = dR*(np.sum(e0[var]*e_dt[var]*sig_profs[var],axis=1)[:,np.newaxis]-np.cumsum(e0[var]*e_dt[var]*sig_profs[var],axis=1))
    
        if 'Molecular' in var:
            # molecular gain gradient term
            gradErr[0] = np.nansum(e0[var]*e_dt[var]*sig_profs[var]/Gain)
    
            # molecular dead time gradient term
            gradErr[4] = np.nansum(-e0[var]*dt[:,np.newaxis]*lin_profs[var]**2/(dt[:,np.newaxis]+lin_profs[var]*deadtime)**2)*deadtime           
            
        elif 'High' in var:
            # comb hi gain gradient term
            gradErr[1] = np.nansum(e0[var]*e_dt[var]*sig_profs[var]/Gain)
    
            # comb hi dead time gradient term
            gradErr[5] = np.nansum(-e0[var]*dt[:,np.newaxis]*lin_profs[var]**2/(dt[:,np.newaxis]+lin_profs[var]*deadtime)**2)*deadtime
            
        elif 'Low' in var:
            # comb lo gain gradient term
            gradErr[2] = np.nansum(e0[var]*e_dt[var]*sig_profs[var]/Gain)
    
            # comb lo dead time gradient term
            gradErr[6] = np.nansum(-e0[var]*dt[:,np.newaxis]*lin_profs[var]**2/(dt[:,np.newaxis]+lin_profs[var]*deadtime)**2)*deadtime       
            
        elif 'Cross' in var:
            # cross gain gradient term
            gradErr[3] = np.nansum(e0[var]*e_dt[var]*sig_profs[var]/Gain)
    
            # cross dead time gradient term
            gradErr[7] = np.nansum(-e0[var]*dt[:,np.newaxis]*lin_profs[var]**2/(dt[:,np.newaxis]+lin_profs[var]*deadtime)**2)*deadtime
        
        gradErrS = gradErrS -2*forward_profs['Backscatter_Coefficient']*forward_profs['Lidar_Ratio']*grad0[var]
        
        gradErrB = gradErrB-2*forward_profs['Backscatter_Coefficient']*forward_profs['Lidar_Ratio']*grad0[var] \
            +e0[var]*e_dt[var]*Gain*Const[var]['mult']*forward_profs['Backscatter_Coefficient']*(Const[var]['pol'][0]+forward_profs['Polarization']*Const[var]['pol'][1])*forward_profs['Tatm']
        
        gradErrP = gradErrP + e0[var]*e_dt[var]*Const[var]['mult']*forward_profs['Backscatter_Coefficient']*Const[var]['pol'][1]*forward_profs['Tatm']/(np.pi*(1+forward_profs['xPol']**2))

    """
    # TV penalty for lidar ratio
    deriv = lam[0][1]*np.nansum(np.abs(np.diff(forward_profs['Lidar_Ratio'],axis=1)/dR))
    deriv = deriv + lam[0][0]*np.nansum(np.abs(forward_profs['Lidar_Ratio'].flatten()-forward_profs['Lidar_Ratio'][meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()])/meshTV['pos_h']['d'].flatten())    
    
    # perform backscatter TV penalty in log space
    deriv = deriv + lam[1][1]*np.nansum(np.abs(np.diff(forward_profs['xB'],axis=1)/dR))
    deriv = deriv + lam[1][0]*np.nansum(np.abs(forward_profs['xB'].flatten()-forward_profs['xB'][meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()])/meshTV['pos_h']['d'].flatten())

    # TV penalty for depolarization
    deriv = deriv + lam[2][1]*np.nansum(np.abs(np.diff(forward_profs['Polarization'],axis=1)/dR))
    deriv = deriv + lam[2][0]*np.nansum(np.abs(forward_profs['Polarization'].flatten()-forward_profs['Polarization'][meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()])/meshTV['pos_h']['d'].flatten()) 
    """

    gradErrStv = np.zeros((tdim,N))
    gradErrBtv = np.zeros((tdim,N))
    gradErrPtv = np.zeros((tdim,N))
    
    # range derivative for lidar ratio
    gradpenS = lam[0][1]*np.sign(np.diff(forward_profs['Lidar_Ratio'],axis=1))/dR
    gradpenS[np.nonzero(np.isnan(gradpenS))] = 0
    gradErrStv[:,:-1] = gradErrStv[:,:-1]-gradpenS*forward_profs['Lidar_Ratio'][:,:-1]
    gradErrStv[:,1:] = gradErrStv[:,1:]+gradpenS*forward_profs['Lidar_Ratio'][:,1:]
    
#    gradErrStv = gradErrStv.flatten()
    
    # time derivative for lidar ratio
    # positive time
    gradpenS = lam[0][0]*np.sign(np.abs(forward_profs['Lidar_Ratio'].flatten()-forward_profs['Lidar_Ratio'][meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()]))/meshTV['pos_h']['d'].flatten()
    gradpenS[np.nonzero(np.isnan(gradpenS))] = 0
    gradErrStv = gradErrStv-(gradpenS*forward_profs['Lidar_Ratio'][meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()]).reshape(gradErrStv.shape)
    gradErrStv[meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()] = gradErrStv[meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()] \
        +gradpenS*forward_profs['Lidar_Ratio'].flatten()
    # negative time
    gradpenS = lam[0][0]*np.sign(np.abs(forward_profs['Lidar_Ratio'].flatten()-forward_profs['Lidar_Ratio'][meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()]))/meshTV['pos_h']['d'].flatten()
    gradpenS[np.nonzero(np.isnan(gradpenS))] = 0
    gradErrStv = gradErrStv-(gradpenS*forward_profs['Lidar_Ratio'][meshTV['neg_h']['i'][0].flatten(),meshTV['neg_h']['i'][1].flatten()]).reshape(gradErrStv.shape)
    gradErrStv[meshTV['neg_h']['i'][0].flatten(),meshTV['neg_h']['i'][1].flatten()] = gradErrStv[meshTV['neg_h']['i'][0].flatten(),meshTV['neg_h']['i'][1].flatten()] \
        +gradpenS*forward_profs['Lidar_Ratio'].flatten()   
    
    # range derivative for aerosol backscatter
    gradpenB = lam[1][1]*np.sign(np.diff(forward_profs['xB'],axis=1))/dR
    gradpenB[np.nonzero(np.isnan(gradpenB))] = 0
    gradErrBtv[:,:-1] = gradErrBtv[:,:-1]-gradpenB
    gradErrBtv[:,1:] = gradErrBtv[:,1:]+gradpenB
    
    # time derivative for aerosol backscatter
    # positive time
    gradpenB = lam[1][0]*np.sign(np.abs(forward_profs['xB'].flatten()-forward_profs['xB'][meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()]))/meshTV['pos_h']['d'].flatten()
    gradpenB[np.nonzero(np.isnan(gradpenB))] = 0
    gradErrBtv = gradErrBtv-gradpenB.reshape(gradErrBtv.shape)
    gradErrBtv[meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()] = gradErrBtv[meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()]+gradpenB
    # negative time
    gradpenB = lam[1][0]*np.sign(np.abs(forward_profs['xB'].flatten()-forward_profs['xB'][meshTV['neg_h']['i'][0].flatten(),meshTV['neg_h']['i'][1].flatten()]))/meshTV['neg_h']['d'].flatten()
    gradpenB[np.nonzero(np.isnan(gradpenB))] = 0
    gradErrBtv = gradErrBtv-gradpenB.reshape(gradErrBtv.shape)
    gradErrBtv[meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()] = gradErrBtv[meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()]+gradpenB
    
    
    # range derivative for polarization
    diff_pol = 1.0/(np.pi*(1+forward_profs['xPol']**2))  # derivative of polariation with respect to the optimization variable
    gradpenP = lam[2][1]*np.sign(np.diff(forward_profs['Polarization'],axis=1))/dR
    gradpenP[np.nonzero(np.isnan(gradpenP))] = 0
    gradErrPtv[:,:-1] = gradErrPtv[:,:-1]-gradpenP*diff_pol[:,:-1]
    gradErrPtv[:,1:] = gradErrPtv[:,1:]+gradpenP*diff_pol[:,1:]
    
    # time derivative for polarization
    # positive time
    gradpenP = lam[2][0]*np.sign(np.abs(forward_profs['Polarization'].flatten()-forward_profs['Polarization'][meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()]))/meshTV['pos_h']['d'].flatten()
    gradpenP[np.nonzero(np.isnan(gradpenP))] = 0
    gradErrPtv = gradErrPtv-(gradpenP*diff_pol[meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()]).reshape(gradErrPtv.shape)
    gradErrPtv[meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()] = gradErrPtv[meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()] \
        +gradpenP*diff_pol.flatten()
    # negative time
    gradpenP = lam[2][0]*np.sign(np.abs(forward_profs['Polarization'].flatten()-forward_profs['Polarization'][meshTV['pos_h']['i'][0].flatten(),meshTV['pos_h']['i'][1].flatten()]))/meshTV['pos_h']['d'].flatten()
    gradpenP[np.nonzero(np.isnan(gradpenP))] = 0
    gradErrPtv = gradErrPtv-(gradpenP*diff_pol[meshTV['neg_h']['i'][0].flatten(),meshTV['neg_h']['i'][1].flatten()]).reshape(gradErrPtv.shape)
    gradErrPtv[meshTV['neg_h']['i'][0].flatten(),meshTV['neg_h']['i'][1].flatten()] = gradErrPtv[meshTV['neg_h']['i'][0].flatten(),meshTV['neg_h']['i'][1].flatten()] \
        +gradpenP*diff_pol.flatten()
    
    gradErr2D[:,:N] = gradErrS+gradErrStv
    gradErr2D[:,N:2*N] = gradErrB+gradErrBtv
    gradErr2D[:,2*N:] = gradErrP+gradErrPtv
    
#    gradErr2D[:,:N] = gradErrStv
#    gradErr2D[:,N:2*N] = gradErrBtv
#    gradErr2D[:,2*N:] = gradErrPtv
    
    gradErr[ix:] = gradErr2D.flatten()
    
    return gradErr

def solve_Condat_1D_subproblem(y,lam):
    """
    Solve_Condat_1D_Subproblem(y,lam)
    solves the problem of estimating a noisy signal
    y with a regularizer lam
    returns the denoised signal x.
    
    Unlike FISTA, this has a closed form solution but it
    can only be applied to 1D data
    
    Algorithm is from
    Condat, L, 
     "A Direct Algorithm for 1-D Total Variation Denoising",
     IEEE SIGNAL PROCESSING LETTERS, VOL. 20, NO. 11, 2013
     DOI: 10.1109/LSP.2013.2278339
    """
    
    
    #  Initialization (Step 1)
    k = 0
    k0 = 0
    kplus = 0
    kminus = 0
    
    vmin = y[0]-lam
    vmax = y[0]+lam
    
    umin = lam
    umax = -lam
    
    x = np.zeros(y.size)  # estimated signal
    
    run_loop1 = True
    
    
    while run_loop1:
        
        # Step 2 check for termination condition 1
        if k >= x.size-1:
            x[-1] = vmin+umin
            run_loop1 = False 
        else:
            run_loop2 = True
            while run_loop2:
                # Step 3
                if y[k+1]+umin < vmin-lam:
                    x[k0:kminus+1] = vmin
                    k = np.minimum(x.size-1,kminus+1)
                    k0 = np.minimum(x.size-1,kminus+1)
                    kplus = np.minimum(x.size-1,kminus+1)
                    kminus = np.minimum(x.size-1,kminus+1)
                    vmin = y[k]
                    vmax = y[k]+2*lam
                    umin = lam
                    umax = -lam
                
                # Step 4
                elif y[k+1]+umax > vmax+lam:
                    x[k0:kplus+1] = vmax
                    k = np.minimum(x.size-1,kplus+1)
                    k0 = np.minimum(x.size-1,kplus+1)
                    kplus = np.minimum(x.size-1,kplus+1)
                    kminus = np.minimum(x.size-1,kplus+1)
                    vmin = y[k]-2*lam
                    vmax = y[k]
                    umin = lam
                    umax = -lam
                    
                # Step 5
                else:
                    k = k+1
                    umin += y[k]-vmin
                    umax += y[k]-vmax
                    
                    # Step 6
                    if umin >= lam:
                        vmin+=np.float(umin-lam)/(k-k0+1)
                        umin = lam
                        kminus=k
                        
                    if umax <= -lam:
                        vmax+=np.float(umax+lam)/(k-k0+1)
                        umax=-lam
                        kplus=k
                    
                # Step 7
                if k < x.size-1:
                    # goto step 3
                    pass
                # Step 8
                elif umin < 0:
                    x[k0:kminus+1]=vmin
                    k = np.minimum(x.size-1,kminus+1)
                    k0 = np.minimum(x.size-1,kminus+1)
                    kminus = np.minimum(x.size-1,kminus+1)
                    vmin = y[k]
                    umin = lam
                    umax = y[k]+lam-vmax
                    run_loop2=False
                # Step 9
                elif umax > 0:
                    x[k0:kplus+1] = vmax
                    k = np.minimum(x.size-1,kplus+1)
                    k0 = np.minimum(x.size-1,kplus+1)
                    kplus = np.minimum(x.size-1,kplus+1)
                    vmax = y[k]
                    umax = -lam
                    umin = y[k]-lam-vmin
                    run_loop2=False
                # Step 10
                else:
                    x[k0:]=vmin+np.float(umax)/(k-k0+1)
                    run_loop2=False
                    run_loop1=False
    return x
    
def solve_FISTA_subproblem(b,lam,max_iter = 10,count_lim = 5,eps=1e-2,bnds=None):
    """
    Solves optimization subproblem by using FISTA [1].
    For most MLE applications 
    b = x_k - 1/alpha * grad(f)
        where x_k is the current state vector, 1/alpha is the step size,
        f - is the error function so grad(f) is the gradient of f with respect to x
        
    lam - TV penalty function is typically
        lam = tau/alpha, where tau is the TV penalty coefficient (often times also lambda)
    
    max_iter - maximum number of loop iterations to estimate the minimum of the subproblem
    
    count_lim - number of successful minimization steps needed in a row before exiting the subproblem
    
    eps - improvement in step needed to count a step as successful
    
    returns x - the solution to the subproblem
    
    Solving the sub problem should be done separately for each separable variable.
        for example, this function should be run once for backscatter coefficient, lidar ratio and depolarization each.
        For non-TV variables (gain, deadtime), don't use this function call.  Just run steepest descent:
        x_{k+1} = x_k - 1/alpha * grad(x)
        
    This function expects rectangular arrays.  Mapping non-rectangular spaces (e.g. altitude varying range data)
    needs to happen prior to calling this function.
    """
    if bnds is None:
        bnds = [-np.inf,np.inf]
    
    # initialize differential state variables with their history
    # k - this iteration, kp1=k+1, km1 = k-1
    r_k = np.zeros((b.shape[0]-1,b.shape[1]))
    s_k = np.zeros((b.shape[0],b.shape[1]-1))
    p_k = np.zeros((b.shape[0]-1,b.shape[1]))
    q_k = np.zeros((b.shape[0],b.shape[1]-1))
    
    p_km1 = np.zeros((b.shape[0]-1,b.shape[1]))
    p_kp1 = np.zeros((b.shape[0]-1,b.shape[1]))
    q_km1 = np.zeros((b.shape[0],b.shape[1]-1))   
    q_kp1 = np.zeros((b.shape[0],b.shape[1]-1))
    
    t_k = 1
    
    grad_a_km1 = np.zeros(b.shape)
    
    cnt_iter = 0
    cnt_eps = 0
    
    while(cnt_iter < max_iter and cnt_eps < count_lim):    
        grad_a = b - lam*map_d2x(r_k,s_k) # compute the FISTA gradient
        grad_a = np.maximum(grad_a,bnds[0])
        grad_a = np.minimum(grad_a,bnds[1])
        grad_ad = map_x2d(grad_a)  # map the gradient to differential space
        
        # update estimates of differential variables
        p_kp1= p_k + 1.0/(8*lam)*grad_ad[0]
        q_kp1= q_k + 1.0/(8*lam)*grad_ad[1]
        
        t_kp1 = (1 + np.sqrt(1+4*t_k**2))/2.0
        
        # Willem performs this normalization step in his FISTA code but 
        # it does not appear in [1]
        # Preform the projection step
        p_kp1 = p_kp1 / np.maximum (np.abs (p_kp1), 1.0)
        q_kp1 = q_kp1 / np.maximum (np.abs (q_kp1), 1.0)
        
        # update normalized differential variables
        r_k = p_k+(t_k-1)/t_kp1*(p_k-p_km1)
        s_k = q_k+(t_k-1)/t_kp1*(q_k-q_km1)
        
        # update variable history before next iteration
        p_km1 = p_k.copy()
        p_k = p_kp1.copy()
        q_km1 = q_k.copy()
        q_k = q_kp1.copy()
        
        t_k = t_kp1
        
        # check for exit criteria
        # Compute the relative step size
        rel_step_num = np.linalg.norm (np.ravel (grad_a) - np.ravel (grad_a_km1))
        rel_step_dem = np.linalg.norm (np.ravel (grad_a))
        
        grad_a_km1 = grad_a.copy()
        
        # re_flt = np.linalg.norm (D_mat - D_prev_mat, "fro") / np.linalg.norm (D_mat, "fro")
        if rel_step_num < (eps * rel_step_dem):
            cnt_eps += 1
        else:
            cnt_eps = 0
        cnt_iter += 1
        
#        print('FISTA cnt_iter: %d'%cnt_iter)
#        print('FISTA cnt_eps:  %d'%cnt_eps)
#        print('FISTA numerator error: %e'%(rel_step_num))
#        print('FISTA denominator error: %e'%(rel_step_dem))

    return grad_a
    

def map_d2x(p,q):
    """
    Maps differential variables p and q (or r and s) to pixel space description
    of a 2D parameter space, x.
    This is script L in [1]
    """    
    
    x = np.zeros((q.shape[0],p.shape[1]))
    x[:-1,:] = p
    x[:,:-1] = x[:,:-1] + q
    x[1:,:] = x[1:,:] - p
    x[:,1:] = x[:,1:] - q
    
    return x
    
    
def map_x2d(x):
    """
    Maps pixel description of the image, x, to a differential description
    p and q (or r and s)
    """
    
    p = np.zeros((x.shape[0]-1,x.shape[1]))
    q = np.zeros((x.shape[0],x.shape[1]-1))
    
    p = x[:-1,:] - x[1:,:]
    q = x[:,:-1] - x[:,1:]
    
    return p,q


def GVHSRL_sparsa_optimizor(F,dF,x0,lam,sub_eps=1e-5,step_eps=1e-5,opt_cnt_max=100,opt_cnt_min=10,cnt_alpha_max=200,sigma=0.1,verbose=False,alpha_verbose=False,alpha=1e5,bnds=None,disable=None,xhist=False,hist_sum=None):
    """
    Sparsa/Spiral TV MLE using FISTA to solve the subproblem
    
    F - function call that gives F(x) = Fit error in the observations
    dF - function call that gives dF(x) = the gradient of x
        
    
    x0 - dictionary for the initial condition of the state variable x
        For GV-HSRL it is assumed to contain
        'xB' - backscatter state vector/matrix
        'xS' - lidar ratio state vector/matrix
        'xP' - polarization (1-d) state vector/matrix
        'xG' - gain for each channel
        'xDT' - dead time for each channel
        
    disable:  list of fit profiles and/or retrieved variables to be disabled in the estimator
    
    hist_sum: index into consideration acceptance criteria (index M for evaluating phi in SPIRAL)
        this should be a negative integer
    """
    # choose alpha (should we use a different alpha for different variables?)
#    alpha = 1e5
    
    if disable is None:
        disable=[]
    if bnds is None:
        bnds = {}
    
    # alpha multiplier when optimization step fails
    eta = 2.0
    
    # maximum iterations through adjusting alpha
#    cnt_alpha_max = 200
    alpha_max = 1e40    
    alpha_min = 1e-1
    
    # adjust for really low precision requirements
    if step_eps < 1e-12:
        step_mult = 1e-12/step_eps
        step_eps = 1e-12
    else:
        step_mult = 1.0
    
    error_hist = [F(x0)]
    step_hist = []
    x_hist = []
    
    # initialize state variable(s)
    x = copy.deepcopy(x0)
    F_k = F(x0)

#    alpha_km1 = alpha
#    alpha_BB_km1 = 0
#    try_BB = True
    
    # Initialize SPARSA loop variables        
    no_min = True    
    cnt_opt = 0 # counts the number of optimization steps with no effective movement in the state vector
    cnt_iter = 0 # counts the number of iteration loops
    
    # Initialize values of state variables, Fit error and gradient
    x_kp1 = copy.deepcopy(x)
    dF_k = dF(x)
    x_k_vec = np.array([])
    dF_k_vec = np.array([])
    for xvar in sorted(x.keys()):
        if not xvar in disable:
            x_k_vec = np.concatenate((x_k_vec,x[xvar].flatten()))
            dF_k_vec = np.concatenate((dF_k_vec,dF_k[xvar].flatten()))
        
    while no_min:

        adj_alpha = True  # flag for exiting the alpha selection loop
        cnt_alpha = 0
        x_kp1_vec = np.array([]) # should this move to inside the while adj_alpha loop?

        while adj_alpha:
            for xvar in sorted(x.keys()):
                if not xvar in disable:
                    if xvar in lam.keys() and x[xvar].size > 1:
                        if xvar in bnds.keys():
                            fista_bounds = bnds[xvar]
                        else:
                            fista_bounds = [-np.inf,np.inf]
                            
                        if x[xvar].ndim == 1 or (1 in x[xvar].shape):
                            # use Condat for 1D TV controlled variables
                            xshape = x[xvar].shape
                            x_kp1[xvar] = solve_Condat_1D_subproblem(x[xvar].flatten()-1/alpha*dF_k[xvar].flatten(),lam[xvar]/alpha).reshape(xshape)
                            x_kp1[xvar][np.isnan(x_kp1[xvar])]=fista_bounds[0]
                            x_kp1[xvar] = np.maximum(fista_bounds[0],np.minimum(fista_bounds[1],x_kp1[xvar]))
                        else:
                            # use FISTA for 2D TV controlled variables
                            x_kp1[xvar] = solve_FISTA_subproblem(x[xvar]-1/alpha*dF_k[xvar],lam[xvar]/alpha,max_iter = 40,count_lim = 5,eps=sub_eps,bnds=fista_bounds)
                    else:
                        # use standard steepest descent for global constants
                        # e.g. Gain and dead time
                        x_kp1[xvar] = x[xvar]-1.0/alpha*dF_k[xvar]
                        if xvar in bnds.keys():
                            x_kp1[xvar] = np.maximum(bnds[xvar][0],np.minimum(bnds[xvar][1],x_kp1[xvar]))
                    x_kp1_vec = np.concatenate((x_kp1_vec,x_kp1[xvar].flatten()))
    #                dF_kvec = np.concatenate((dF_kvec,dF_k[xvar].flatten()))
            
            # calculate new gradient for evaluting acceptance criteria
            dF_kp1 = dF(x_kp1)
            dF_kp1_vec = np.array([])
            for xvar in sorted(dF_kp1.keys()):
                dF_kp1_vec = np.concatenate((dF_kp1_vec,dF_kp1[xvar].flatten()))
                
#            print('x nan counts')
#            print(np.nansum(x_kp1_vec))
#            print(np.nansum(x_k_vec))
            x_diff = x_kp1_vec-x_k_vec
            F_kp1 = F(x_kp1)
#            diff_F_kp1 = GVHSRL_sparsa_Error_difference
            if hist_sum is None:
                acc_criteria = np.nanmax(error_hist)-sigma/2*alpha*np.nansum((x_diff)**2)
            else:
                acc_criteria = np.nanmax(error_hist[hist_sum:])-sigma/2*alpha*np.nansum((x_diff)**2)
#            acc_criteria0 = F_kp1-np.nanmax(error_hist)
#            acc_criteria1 = -sigma/2*alpha*np.sum((x_diff)**2)
            if verbose and alpha_verbose:
                print('cnt_alpha %d'%cnt_alpha)
                print('accept_criteria: %e'%acc_criteria)
#                print('accept criteria0: %e'%acc_criteria0)
#                print('accept criteria1: %e'%(acc_criteria1))
                print('fit error: %e'%F_kp1)
                print('alpha: %e'%alpha)
                
            # test acceptance criteria
            if F_kp1 <= acc_criteria or np.abs(alpha) >= alpha_max or cnt_alpha > cnt_alpha_max:
#            if acc_criteria0 < acc_criteria1 or cnt_alpha > cnt_alpha_max:
#                adj_alpha = False              
#                print('accepted')
                
                
                dF_diff = dF_kp1_vec - dF_k_vec
                if (x_diff != 0).any() and (dF_diff!=0).any():
                    # estimate next alpha
                    alpha = np.sum(x_diff*dF_diff)/np.sum(x_diff**2)
                else:
                    alpha = alpha*eta
                
                if alpha < 0 and verbose and alpha_verbose:
                    print('Warning in Sparsa Optimizer:' )
                    print('  alpha less than zero may indicate TV regularizer is too large')
                    print('  lam['+xvar+'] = %e'%lam[xvar] )
                    print('  alpha = %e'%alpha)                
                
                alpha = np.nanmin([alpha,alpha_max])
                alpha = np.nanmax([alpha,alpha_min])
                
                # check if the fit error went down.  If not, don't save the
                # result
                if F_kp1 < F_k: 
                        
                    adj_alpha = False 
                    # save new terms for next step
                    error_hist.extend([F_kp1])
                    x_k_vec = x_kp1_vec.copy()
                    dF_k_vec = dF_kp1_vec.copy()
                    dF_k = copy.deepcopy(dF_kp1)
                    x = copy.deepcopy(x_kp1)
                    F_k = F_kp1
                    min_success = True
                
                else:
                    alpha*=(eta**cnt_alpha)
                    if cnt_alpha > cnt_alpha_max or np.abs(alpha) > alpha_max or np.isinf(alpha):
                        adj_alpha = False 
                        min_success = False
                        if np.isinf(alpha):
                            alpha=1e10
                        else:
                            alpha*=eta
                        if verbose and alpha_verbose:
                            print('alpha search failed')
                            print('SPARSA alpha: %e'%alpha)
                            print('SPARSA cnt_alpha: %d'%cnt_alpha)
                    else:
                        x_kp1_vec = np.array([])
                        cnt_alpha += 1
                    
           
            else:
                alpha = alpha*eta
                cnt_alpha += 1
                x_kp1_vec = np.array([])
        
        
        if min_success:
            # check for convergence using step size criteria
            step_eval = np.sqrt(np.sum((x_diff)**2))*step_mult/np.sqrt(np.sum((x_kp1_vec)**2))
            if verbose:
                print('SPARSA cnt_alpha: %d'%cnt_alpha)
                print('SPARSA alpha: %e'%alpha)
                print('Step evaluation: %e  (theshold: %e)'%(step_eval,step_eps))
        else:
            # no step was taken
            step_eval = 0
            
        step_hist.extend([step_eval])
        if xhist:
            x_hist.extend([copy.deepcopy(x)])
        
        if cnt_iter > opt_cnt_max:
            no_min = False
            print('SPARSA max iterations (%d) exceeded'%opt_cnt_max)
            print('%d iterations, %e Error'%(cnt_iter,F_k))
            
        if step_eval < step_eps :  # and cnt_opt >= opt_cnt_min) or cnt_opt >= opt_cnt_max
            cnt_opt+=1
            
            # require failed reduction opt_cnt_min times before terminating
            if cnt_opt >= opt_cnt_min:
                # terminate the loop
                no_min = False
                if cnt_opt < opt_cnt_max and verbose:
                    print('SPARSA found minimum')
                else:
                    if verbose:
                        print('SPARSA max iterations (%d) exceeded'%opt_cnt_max)
                        print('%d iterations, %e Error'%(cnt_iter,F_k))
            
            elif verbose:
                print('SPARSA iteration %d, fit error: %f'%(cnt_iter,error_hist[-1]))
                
        else:
            cnt_opt = 0
            if verbose:
                print('SPARSA iteration %d, fit error: %f'%(cnt_iter,error_hist[-1]))
        cnt_iter += 1        
            
    return x,[error_hist,step_hist,x_hist]


def Sparsa_optimizor_alpha(F,dF,x0,lam,sub_eps=1e-5,step_eps=1e-5,opt_cnt_max=100,opt_cnt_min=10,cnt_alpha_max=10,sigma=0.1,verbose=False,alpha0=1e5,bnds={}):
    """
    Sparsa/Spiral TV MLE using FISTA to solve the subproblem
    This version estimates alpha separately for each variable
    
    F - function call that gives F(x) = Fit error in the observations
    dF - function call that gives dF(x) = the gradient of x
        
    
    x0 - dictionary for the initial condition of the state variable x
        For GV-HSRL it is assumed to contain
        'xB' - backscatter state vector/matrix
        'xS' - lidar ratio state vector/matrix
        'xP' - polarization (1-d) state vector/matrix
        'xG' - gain for each channel
        'xDT' - dead time for each channel
    """
    # choose alpha (should we use a different alpha for different variables?)
#    alpha = 1e5
    
    # alpha multiplier when optimization step fails
    eta = 2.0
    
    # maximum iterations through adjusting alpha
    cnt_alpha_max = 200
    alpha_max = 1e20    
    alpha_min = 1e3
    
    # adjust for really low precision requirements
    if step_eps < 1e-12:
        step_mult = 1e-12/step_eps
        step_eps = 1e-12
    else:
        step_mult = 1.0
    
    error_hist = [F(x0)]
    step_hist = []
    
    # initialize state variable(s)
    x = copy.deepcopy(x0)
    F_k = F(x0)

#    alpha_km1 = alpha
#    alpha_BB_km1 = 0
#    try_BB = True
    
    # Initialize SPARSA loop variables        
    no_min = True    
    cnt_opt = 0 # counts the number of optimization steps with no effective movement in the state vector
    cnt_iter = 0 # counts the number of iteration loops
    
    # Initialize values of state variables, Fit error and gradient
    x_kp1 = copy.deepcopy(x)
    dF_k = dF(x)
#    for xvar in sorted(x.keys()):
#        x_k_vec = np.concatenate((x_k_vec,x[xvar].flatten()))
#        dF_k_vec = np.concatenate((dF_k_vec,dF_k[xvar].flatten()))
    
    alpha = {}
    for xvar in x.keys():
        alpha[xvar] = alpha0
    
    while no_min:
        x_k = copy.deepcopy(x)
        for xvar in sorted(x.keys()):
            
            adj_alpha = True  # flag for exiting the alpha selection loop
            cnt_alpha = 0
            x_kp1_vec = np.array([])
    
            while adj_alpha:
                if xvar in lam.keys(): 
                    if xvar in bnds.keys():
                        fista_bounds = bnds[xvar]
                    else:
                        fista_bounds = [-np.inf,np.inf]
                    # use FISTA for TV controlled variables
                    x_kp1[xvar] = solve_FISTA_subproblem(x[xvar]-1/alpha[xvar]*dF_k[xvar],lam[xvar]/alpha[xvar],max_iter = 40,count_lim = 5,eps=sub_eps,bnds=fista_bounds)
                else:
                    # use standard steepest descent for global constants
                    # e.g. Gain and dead time
                    x_kp1[xvar] = x[xvar]-1.0/alpha[xvar]*dF_k[xvar]

            
                dF_kp1 = dF(x_kp1)
                x_diff = x_kp1[xvar]-x[xvar]
                F_kp1 = F(x_kp1)
                
                acc_criteria = np.nanmax(error_hist)-sigma/2*alpha[xvar]*np.nansum((x_diff)**2)
                if verbose:
                    print(xvar)
                    print('cnt_alpha %d'%cnt_alpha)
                    print('accept_criteria: %e'%acc_criteria)
    #                print('accept criteria0: %e'%acc_criteria0)
    #                print('accept criteria1: %e'%(acc_criteria1))
                    print('fit error: %e'%F_kp1)
                    print('alpha: %e'%alpha[xvar])
                    
                # test acceptance criteria
                if F_kp1 <= acc_criteria or np.abs(alpha[xvar]) >= alpha_max or cnt_alpha > cnt_alpha_max:
                    dF_diff = dF_kp1[xvar] - dF_k[xvar]
                    if (x_diff != 0).any() and (dF_diff!=0).any():
                        # estimate next alpha
                        alpha[xvar] = np.sum(x_diff*dF_diff)/np.sum(x_diff**2)
                    else:
                        alpha[xvar] = alpha[xvar]*eta
                    
                    if alpha[xvar] < 0 and verbose:
                        print('Warning in Sparsa Optimizer:' )
                        if xvar in lam.keys():
                            print('  alpha less than zero may indicate TV regularizer is too large')
                            print('  lam['+xvar+'] = %e'%lam[xvar] )
                        print('  alpha = %e'%alpha[xvar])                
                    
                    alpha[xvar] = np.nanmin([alpha[xvar],alpha_max])
                    alpha[xvar] = np.nanmax([alpha[xvar],alpha_min])
                    
                    # check if the fit error went down.  If not, don't save the
                    # result
                    if F_kp1 < F_k: 
                            
                        adj_alpha = False 
                        # save new terms for next step
                        error_hist.extend([F_kp1])
                        dF_k = copy.deepcopy(dF_kp1)
                        x = copy.deepcopy(x_kp1)
                        F_k = F_kp1
                        min_success = True
                        
                        if verbose:
                            print(xvar)
                            print('SPARSA cnt_alpha: %d'%cnt_alpha)
                            print('SPARSA alpha: %e'%alpha[xvar])
                    
                    else:
                        alpha[xvar]*=(eta**cnt_alpha)
                        if cnt_alpha > cnt_alpha_max or np.abs(alpha[xvar]) > alpha_max or np.isinf(alpha[xvar]):
                            adj_alpha = False 
                            min_success = False
                            if np.isinf(alpha[xvar]):
                                alpha[xvar]=1e10
                            else:
                                alpha[xvar]*=eta
                            if verbose:
                                print('alpha search failed')
                                print('SPARSA alpha: %e'%alpha[xvar])
                                print('SPARSA cnt_alpha: %d'%cnt_alpha)
                        else:
                            cnt_alpha += 1
                        
               
                else:
                    alpha[xvar] = alpha[xvar]*eta
                    cnt_alpha += 1
            
            
            
        if min_success:
            x_diff_tot = np.array([])
            x_kp1_vec = np.array([])
            for xvar in sorted(x.keys()):
                x_diff_tot = np.concatenate((x_diff_tot,x[xvar].flatten()-x_k[xvar].flatten()))
                x_kp1_vec = np.concatenate((x_kp1_vec,x[xvar].flatten()))
            # check for convergence using step size criteria
            step_eval = np.sqrt(np.sum((x_diff)**2))*step_mult/np.sqrt(np.sum((x_kp1_vec)**2))
            if verbose:
#                print('SPARSA cnt_alpha: %d'%cnt_alpha)
#                print('SPARSA alpha: %e'%alpha)
                print('Step evaluation: %e  (theshold: %e)'%(step_eval,step_eps))
        else:
            # no step was taken
            step_eval = 0
            
        step_hist.extend([step_eval])
        
        if cnt_iter > opt_cnt_max:
            no_min = False
            print('SPARSA max iterations (%d) exceeded'%opt_cnt_max)
            print('%d iterations, %e Error'%(cnt_iter,F_k))
            
        if step_eval < step_eps :  # and cnt_opt >= opt_cnt_min) or cnt_opt >= opt_cnt_max
            cnt_opt+=1
            
            # require failed reduction opt_cnt_min times before terminating
            if cnt_opt >= opt_cnt_min:
                # terminate the loop
                no_min = False
                if cnt_opt < opt_cnt_max and verbose:
                    print('SPARSA found minimum')
                else:
                    if verbose:
                        print('SPARSA max iterations (%d) exceeded'%opt_cnt_max)
                        print('%d iterations, %e Error'%(cnt_iter,F_k))
            
            elif verbose:
                print('SPARSA iteration %d, fit error: %f'%(cnt_iter,error_hist[-1]))
                
        else:
            cnt_opt = 0
            if verbose:
                print('SPARSA iteration %d, fit error: %f'%(cnt_iter,error_hist[-1]))
        cnt_iter += 1        
            
    return x,[error_hist,step_hist]

    
def GVHSRL_sparsa_Error(x,fit_profs,Const,lam,dt=1.0,weights=np.array([1]),cond_fun=cond_fun_default_gv_hsrl):  
    
    """
    PTV Error of GV-HSRL profiles
    scale={'xB':1,'xS':1,'xP':1} is deprecated
    """

    dR = fit_profs['Raw_Molecular_Backscatter_Channel'].mean_dR
    forward_profs = Build_GVHSRL_sparsa_Profiles(x,Const,dt=dt,dR=dR,return_params=True,cond_fun=cond_fun)
    
    ErrRet = 0    
    
    for var in lam.keys():
        deriv = lam[var]*np.nansum(np.abs(np.diff(x[var],axis=1)))+lam[var]*np.nansum(np.abs(np.diff(x[var],axis=0)))
        ErrRet = ErrRet + deriv

    for var in fit_profs:
#        print(var)
        ErrRet = ErrRet + np.nansum(weights*(forward_profs[var]-fit_profs[var].profile*np.log(forward_profs[var])))
        
    return ErrRet
    
    
def Build_GVHSRL_sparsa_Profiles(x,Const,dt=1.0,dR=7.5,return_params=False,params={},cond_fun=cond_fun_default_gv_hsrl):
    """
    dt sets the adjustment factor to convert counts to count rate needed in
    deadtime correction
    
    if return_params=True, returns the profiles of the optical parameters
    
    cond_fun - dict of conditioning functions for each variable the user wants to condition.
        a condition function should take arguments to provide an inverse and a derivative (df/dx)
        It should have the form
        f(x,opt) where opt is a string input accepting
            'inverse' - to provide an inverse operation
            'derivative' - to provide the derivative of the function
            'norm' or any other argument - results in normal conditioning operation
        if cond_fun is not provided for a variable, it will be treated as a pass through
            and the parameter will be set equal to the state variable
            
        the general format for a condition function is
        cond_function(x,*args):
            if args[0] == 'inverse':
                return np.sqrt( (x-3)/4 )
            if args[0] == 'derivative':
                return 8*x
            else:
                # actual function conditioning
                return 4*x**2 + 3
        this would be passed in for a particular state variable xA by creating
            a dict that points to the condition function and passsing that to
            this, the error function and the gradient function
            
            cond_fun['xA'] = cond_function
        
    """
    
    try:
        beta_aer = params['Backscatter_Coefficient']
        p_aer = params['Polarization']
        sLR = params['Lidar_Ratio']
        Tatm = params['Tatm']
    except KeyError:    
        beta_aer = cond_fun['xB'](x['xB'],'normal') #  np.exp(scale['xB']*x['xB'])
        p_aer = cond_fun['xP'](x['xP'],'normal') # np.arctan(scale['xP']*x['xP'])/np.pi+0.5
        sLR = cond_fun['xS'](x['xS'],'normal') # np.exp(scale['xS']*x['xS'])+1
    
        Tatm = np.exp(-2*np.cumsum(beta_aer*sLR,axis=1)*dR)
        Tatm[:,1:] = Tatm[:,:-1]
        Tatm[:,0] = 0
    
    forward_profs = {}
    for var in Const.keys():
        if 'Molecular' in var:
            forward_profs[var] = np.exp(x['xG'][0])*Const[var]['mult']*(beta_aer*(Const[var]['pol'][0]+p_aer*Const[var]['pol'][1])+Const[var]['mol'])*Tatm+Const[var]['bg']
            if not np.isnan(x['xDT'][0]):
                forward_profs[var] = forward_profs[var]*dt[:,np.newaxis]/(dt[:,np.newaxis]+forward_profs[var]*np.exp(x['xDT'][0]))   
        elif 'High' in var:
            forward_profs[var] = np.exp(x['xG'][1])*Const[var]['mult']*(beta_aer*(Const[var]['pol'][0]+p_aer*Const[var]['pol'][1])+Const[var]['mol'])*Tatm+Const[var]['bg']
            if not np.isnan(x['xDT'][1]):
                forward_profs[var] = forward_profs[var]*dt[:,np.newaxis]/(dt[:,np.newaxis]+forward_profs[var]*np.exp(x['xDT'][1]))  
        elif 'Low' in var:
            forward_profs[var] = np.exp(x['xG'][2])*Const[var]['mult']*(beta_aer*(Const[var]['pol'][0]+p_aer*Const[var]['pol'][1])+Const[var]['mol'])*Tatm+Const[var]['bg']
            if not np.isnan(x['xDT'][2]):
                forward_profs[var] = forward_profs[var]*dt[:,np.newaxis]/(dt[:,np.newaxis]+forward_profs[var]*np.exp(x['xDT'][2]))   
        elif 'Cross' in var:
            forward_profs[var] = np.exp(x['xG'][3])*Const[var]['mult']*(beta_aer*(Const[var]['pol'][0]+p_aer*Const[var]['pol'][1])+Const[var]['mol'])*Tatm+Const[var]['bg']
            if not np.isnan(x['xDT'][3]):
                forward_profs[var] = forward_profs[var]*dt[:,np.newaxis]/(dt[:,np.newaxis]+forward_profs[var]*np.exp(x['xDT'][3]))
    
    if return_params:
        forward_profs['Backscatter_Coefficient'] = beta_aer.copy()
        forward_profs['Lidar_Ratio'] = sLR.copy()
        forward_profs['Polarization'] = p_aer.copy()
        forward_profs['Tatm'] = Tatm.copy()
        
    
    return forward_profs
    
def GVHSRL_sparsa_Error_Gradient(x,fit_profs,Const,lam,dt=1.0,weights=np.array([1]),cond_fun=cond_fun_default_gv_hsrl):
    """
    Analytical gradient of GVHSRL_sparsa_Error()
    """
    
    dR = fit_profs['Raw_Molecular_Backscatter_Channel'].mean_dR
    forward_profs = Build_GVHSRL_sparsa_Profiles(x,Const,dt=dt,dR=dR,return_params=True,cond_fun=cond_fun)

    
    # obtain models without nonlinear responde but including background
    xlin = copy.deepcopy(x)
    xlin['xDT'][0] = np.nan
    xlin['xDT'][1] = np.nan
    xlin['xDT'][2] = np.nan
    xlin['xDT'][3] = np.nan
    lin_profs = Build_GVHSRL_sparsa_Profiles(xlin,Const,dt=dt,dR=dR,params=forward_profs,cond_fun=cond_fun)
    

    #obtain models without nonlinear response or background
    sig_profs = {}
    e0 = {}
    e_dt = {}
    grad0 = {}
    
    # gradient components of each atmospheric variable
    gradErr = {}
    for var in x.keys():
        gradErr[var] = np.zeros(x[var].shape)
    
    for var in fit_profs.keys():
        sig_profs[var] = lin_profs[var]-Const[var]['bg']
        
        if 'Molecular' in var:
            deadtime = np.exp(x['xDT'][0])
            Gain = np.exp(x['xG'][0])        
            
        elif 'High' in var:
            deadtime = np.exp(x['xDT'][1])
            Gain = np.exp(x['xG'][1])
            
        elif 'Low' in var:
            deadtime = np.exp(x['xDT'][2])
            Gain = np.exp(x['xG'][2])
            
        elif 'Cross' in var:
            deadtime = np.exp(x['xDT'][3])
            Gain = np.exp(x['xG'][3])
            
        # useful definitions for gradient calculations
        e0[var] = (1-fit_profs[var].profile/forward_profs[var])  # error function derivative
        e_dt[var] = dt[:,np.newaxis]**2/(dt[:,np.newaxis]+lin_profs[var]*deadtime)**2  # dead time derivative
        grad0[var] = dR*(np.sum(e0[var]*e_dt[var]*sig_profs[var],axis=1)[:,np.newaxis]-np.cumsum(e0[var]*e_dt[var]*sig_profs[var],axis=1))
    
        if 'Molecular' in var:
            # molecular gain gradient term
            gradErr['xG'][0] = np.nansum(e0[var]*e_dt[var]*sig_profs[var])
    
            # molecular dead time gradient term
            gradErr['xDT'][0] = np.nansum(-e0[var]*dt[:,np.newaxis]*lin_profs[var]**2/(dt[:,np.newaxis]+lin_profs[var]*deadtime)**2)*deadtime           
            
        elif 'High' in var:
            # comb hi gain gradient term
            gradErr['xG'][1] = np.nansum(e0[var]*e_dt[var]*sig_profs[var])
    
            # comb hi dead time gradient term
            gradErr['xDT'][1] = np.nansum(-e0[var]*dt[:,np.newaxis]*lin_profs[var]**2/(dt[:,np.newaxis]+lin_profs[var]*deadtime)**2)*deadtime
            
        elif 'Low' in var:
            # comb lo gain gradient term
            gradErr['xG'][2] = np.nansum(e0[var]*e_dt[var]*sig_profs[var])
    
            # comb lo dead time gradient term
            gradErr['xDT'][2] = np.nansum(-e0[var]*dt[:,np.newaxis]*lin_profs[var]**2/(dt[:,np.newaxis]+lin_profs[var]*deadtime)**2)*deadtime       
            
        elif 'Cross' in var:
            # cross gain gradient term
            gradErr['xG'][3] = np.nansum(e0[var]*e_dt[var]*sig_profs[var])
    
            # cross dead time gradient term
            gradErr['xDT'][3] = np.nansum(-e0[var]*dt[:,np.newaxis]*lin_profs[var]**2/(dt[:,np.newaxis]+lin_profs[var]*deadtime)**2)*deadtime
        
        gradErr['xS'] = gradErr['xS'] -2*forward_profs['Backscatter_Coefficient']*grad0[var]*cond_fun['xS'](x['xS'],'derivative') # -2*forward_profs['Backscatter_Coefficient']*forward_profs['Lidar_Ratio']*grad0[var]*scale['xS']
        
        gradErr['xB'] = gradErr['xB']+(-2*forward_profs['Lidar_Ratio']*grad0[var] \
            +e0[var]*e_dt[var]*Gain*Const[var]['mult']*(Const[var]['pol'][0]+forward_profs['Polarization']*Const[var]['pol'][1])*forward_profs['Tatm'])*cond_fun['xB'](x['xB'],'derivative')
        
        gradErr['xP'] = gradErr['xP'] + e0[var]*e_dt[var]*Const[var]['mult']*Gain*forward_profs['Backscatter_Coefficient']*Const[var]['pol'][1]*forward_profs['Tatm']*cond_fun['xP'](x['xP'],'derivative')  #e0[var]*e_dt[var]*Const[var]['mult']*Gain*forward_profs['Backscatter_Coefficient']*Const[var]['pol'][1]*forward_profs['Tatm']*scale['xP']/(np.pi*(1+(scale['xP']*x['xP'])**2))

    
    return gradErr
    
def Build_WVDIAL_sparsa_Profiles(x,Const,return_params=False,params=None,conv=True,cond_fun=None):
    """
    Build_WVDIAL_sparsa_Profiles(x,Const,return_params=False,params=None,conv=True,scale=None)
    
    Builds WV-DIAL profiles for forward inversion.
    x - state variable of observation.  It expects:
        'xPhi' - common parameters in the two observation channels
        'xN' - number density of water vapor
        'xG' - ln array containing online and offline gain coefficients (in that order)
        'xDT' - ln array containing deadtimes of online and offline channels (in that order)
        
    Const - constant terms used throughout the routine for each channel.  It expects
        a dict entry for 'Online' and 'Offline' each containing
        'mult' - constant multipliers.  At minimum this should include a 1/R^2 term  
            but a geo overlap and and molecular backscatter term can also be included
        'bg' - background counts
        'sigma' - the extinction cross section of the WV line at the channel's wavelength
        
    return_params - return parameters calculated for the profile including
        water vapor number density (nWV)
        and common terms (phi)
    
    params - pass in parameters needed to calculate the profile to save time
    
    conv - use the convolution kernel in Const['kconv'] on the profiles
    scale - dictionary with the scaling of the raw variables xN and xPhi
    """
    
    if params is None:
        params={}
    if cond_fun is None:
        cond_fun = {}
        for var in x:
            cond_fun[var] = cond_fun_default_wv_dial
#    if dt is None:
#        dt = {}
    
    try:
        nWV = params['nWV']
        phi = params['phi']
    except KeyError:    
        nWV = cond_fun['xN'](x['xN'],'normal')
        phi = cond_fun['xPhi'](x['xPhi'],'normal')
        
#        nWV = scale['xN']*x['xN'] # np.exp(scale['xN']*x['xN'])  or scale['xN']*x['xN']
#        phi = np.exp(scale['xPhi']*x['xPhi'])
        
    if 'kconv' in Const.keys() and conv:
        kconv = Const['kconv']
    else:
        kconv= np.ones((1,1))
    
    dR = Const['dR']
    
    forward_profs = {}
    for var in Const.keys():
        
            
        if ('Online' in var) or ('Offline' in var):    
            # get index into the gain and deadtime arrays
            if 'Online' in var:
    #            iConst = 0
                GainLabel = 'xGon'
                
            elif 'Offline' in var:
    #            iConst = 1
                GainLabel = 'xGoff'
            Gain = cond_fun[GainLabel](x[GainLabel],'normal')
        
#            forward_profs[var] = np.exp(x['xG'][iConst])*Const[var]['mult']*phi*np.exp(-2*dR*np.cumsum(nWV*Const[var]['sigma'],axis=1))+Const[var]['bg']
            forward_profs[var] = Gain*Const[var]['mult']*phi*np.exp(-2*dR*np.cumsum(nWV*Const[var]['sigma'],axis=1))
            if conv:
                forward_profs[var] = lp.conv2d(forward_profs[var],kconv,keep_mask=False)
                if Const['M'] > 1:
                    # sub sample to observation resolution if performing
                    # super resolution retrievals
                    forward_profs[var] = forward_profs[var][:,Const['M0']::Const['M'],...]
    #                forward_profs[var] = np.convolve(forward_profs[var],kconv,'same')
            forward_profs[var] += Const[var]['bg']
    
                
    if return_params:
        forward_profs['nWV'] = nWV.copy()
        forward_profs['phi'] = phi.copy()
        if kconv.size > 1:
            forward_profs['kconv'] = kconv.copy()
        
    
    return forward_profs

def WVDIAL_sparsa_Error(x,fit_profs,Const,lam,weights=None,cond_fun=None):
    
    """
    WVDIAL_sparsa_Error(x,fit_profs,Const,lam,weights=None,scale=None)
    
    PTV Error of WV-DIAL profiles
    
    x - dictionary of state variabls
        'xN'  - water vapor number density
        'xPhi'  - common terms
        'xGon'  - online channel gain
        'xGoff' - offline channel gain
        'xDT' - array of dead times [online, offline]
        
    fit_profs - dictonary of lidar photon count observations by channel
    
    Const - constant terms used to forward model the profiles
    
    lam - dictionary of regularizer values applied to x variables
    
    weights - weight applied to inverse log-likelihood calculation of a pixel
            defaults to 1.0
            
    scale - dictionary of multipliers used to obtain  
        actual atmospheric parameters from state paremters in x
    
    """
    if cond_fun is None:
        cond_fun = {}
        for var in x:
            cond_fun[var] = cond_fun_default_wv_dial
            
    if weights is None:
        weights=np.array([1])
#    if dt is None:
#        dt={}  # old default was 1.0
    
#    dR = Const['dR']
    forward_profs = Build_WVDIAL_sparsa_Profiles(x,Const,return_params=True,cond_fun=cond_fun)
    
    ErrRet = 0    
    
    for var in lam.keys():
        deriv = lam[var]*np.nansum(np.abs(np.diff(x[var],axis=1)))+lam[var]*np.nansum(np.abs(np.diff(x[var],axis=0)))
        ErrRet = ErrRet + deriv

    DTarray = cond_fun['xDT'](x['xDT'],'normal')
    for var in fit_profs:
        if 'Online' in var:
            DT_index = 0
        elif 'Offline' in var:
            DT_index = 1

            
        if not np.isnan(x['xDT'][DT_index]):
            corrected_fit_profs = deadtime_correct(fit_profs[var].profile,DTarray[DT_index],dt=Const[var].get('rate_adj',1.0))
#            ErrRet += np.nansum(Const[var]['weights']*Const['conv_mask']*(forward_profs[var]-corrected_fit_profs*np.log(forward_profs[var])))
            ErrRet += np.nansum(Const[var]['weights']*Const['conv_mask']*(forward_profs[var]-corrected_fit_profs*np.log(forward_profs[var])))
        else:
#            ErrRet += np.nansum(Const[var]['weights']*Const['conv_mask']*(forward_profs[var]-fit_profs[var].profile*np.log(forward_profs[var])))
            ErrRet += np.nansum(Const[var]['weights']*Const['conv_mask']*(forward_profs[var]-fit_profs[var].profile*np.log(forward_profs[var])))
        
#        ErrRet = ErrRet + np.nansum(weights*Const['conv_mask']*(forward_profs[var]-fit_profs[var].profile*np.log(forward_profs[var])))
        
    return ErrRet
    
def WVDIAL_sparsa_Error_Gradient(x,fit_profs,Const,lam,weights=None,n_conv=0,cond_fun=None):
    """
    Analytical gradient of WVDIAL_sparsa_Error_Gradient(x,fit_profs,Const,lam,weights=None,n_conv=0,scale=None)
    """
    
    if weights is None:
        weights = np.array([1])
        
    if cond_fun is None:
        cond_fun = {}
        for var in x:
            cond_fun[var] = cond_fun_default_wv_dial
    
    
    if 'kconv' in Const.keys():
        kconv = Const['kconv']
        ikconv = Const['ikconv']
    elif n_conv > 0:
        kconv = np.ones((1,n_conv),dtype=np.float)/n_conv  # create convolution kernel for laser pulse
        ikconv = np.arange(kconv.size,dtype=np.int)[np.newaxis,:] - np.int(n_conv/2)
    else:
        kconv= np.ones((1,1))   
        ikconv = np.zeros((0,0),dtype=np.int)
    
    dR = Const['dR']
    # calculate the final profiles
    forward_profs = Build_WVDIAL_sparsa_Profiles(x,Const,return_params=True,cond_fun=cond_fun)
    
    # calculate profiles without the convolution kernel for full parameter
    # convolution terms (xN and xPhi)
    sig_profs = Build_WVDIAL_sparsa_Profiles(x,Const,return_params=True,cond_fun=cond_fun,conv=False)  
    

    #obtain models without nonlinear response or background
    e0 = {}

    
    # gradient components of each atmospheric variable
    gradErr = {}
    for var in x.keys():
        gradErr[var] = np.zeros(x[var].shape)
        
    deadtime_set = cond_fun['xDT'](x['xDT'],'normal')
    ddeadtime_set = cond_fun['xDT'](x['xDT'],'derivative')
    
    for var in fit_profs.keys():

        # signal profiles are the range resolved signals without the 
        # convolution kernel applied
        sig_profs[var] = sig_profs[var]-Const[var]['bg']
        
        if 'Online' in var:
            iConst = 0
            GainLabel = 'xGon'
        elif 'Offline' in var:
            iConst = 1
            GainLabel = 'xGoff'
        
        deadtime = deadtime_set[iConst] #np.exp(x['xDT'][iConst])
        ddeadtime = ddeadtime_set[iConst]

        dt_rate = Const[var].get('rate_adj',1.0)  #dt.get(var,1.0)
        
        # use the corrected photon counts if deadtime corrections are being applied
        if not np.isnan(x['xDT'][iConst]):
            corrected_fit_profs = deadtime_correct(fit_profs[var].profile,deadtime_set[iConst],dt=dt_rate)
        else:
            corrected_fit_profs = fit_profs[var].profile.copy()
        
            
        e0[var] = Const[var]['weights']*(1-corrected_fit_profs/forward_profs[var])  # error function derivative

        GainDeriv = cond_fun[GainLabel](x[GainLabel],'derivative')/cond_fun[GainLabel](x[GainLabel],'normal')
        if x[GainLabel].size > 1:
            gradErr[GainLabel] = np.nansum(Const['conv_mask']*e0[var]*(forward_profs[var]-Const[var]['bg'])*GainDeriv,axis=1)
        else:
            gradErr[GainLabel] = np.nansum(Const['conv_mask']*e0[var]*(forward_profs[var]-Const[var]['bg'])*GainDeriv)
        

        # dead time gradient term
        gradErr['xDT'][iConst] = -np.nansum(Const[var]['weights']*dt_rate*corrected_fit_profs**2/(dt_rate-corrected_fit_profs*deadtime)**2*ddeadtime*np.log(forward_profs[var]))

#        gradErr['xN'] += -2*dR*scale['xN']*Const[var]['sigma']*conv_grad_step(Const['conv_mask']*e0[var],sig_profs[var],kconv.flatten(),hindex=ikconv.flatten())
#
#        gradErr['xPhi'] += scale['xPhi']*conv_grad_delta(Const['conv_mask']*e0[var],sig_profs[var],kconv.flatten(),hindex=ikconv.flatten())
        
        gradErr['xN'] += -2*dR*Const[var]['sigma']*conv_grad_step_superres(Const['conv_mask']*e0[var],sig_profs[var]*cond_fun['xN'](x['xN'],'derivative'),kconv.flatten(),hindex=ikconv.flatten(),M=Const['M'],M0=Const['M0'])

        gradErr['xPhi'] += conv_grad_delta_superres(Const['conv_mask']*e0[var],sig_profs[var]*cond_fun['xPhi'](x['xPhi'],'derivative')/cond_fun['xPhi'](x['xPhi'],'normal'),kconv.flatten(),hindex=ikconv.flatten(),M=Const['M'],M0=Const['M0'])

    
    return gradErr
    

def DLBHSRL_sparsa_Error(x,fit_profs,Const,lam,dt=1.0,weights=np.array([1]),cond_fun=cond_fun_default_gv_hsrl):  
    
    """
    PTV Error of GV-HSRL profiles
    scale={'xB':1,'xS':1,'xP':1} is deprecated
    """

    dR = fit_profs[list(fit_profs.keys())[0]].mean_dR
    forward_profs = Build_DLBHSRL_sparsa_Profiles(x,Const,dt=dt,dR=dR,return_params=True,cond_fun=cond_fun)
    
    ErrRet = 0    
    
    for var in lam.keys():
        deriv = lam[var]*np.nansum(np.abs(np.diff(x[var],axis=1)))+lam[var]*np.nansum(np.abs(np.diff(x[var],axis=0)))
        ErrRet = ErrRet + deriv

    for var in fit_profs:
#        print(var)
        if 'Molecular' in var:
            DT_index = 0
        elif 'Combined' in var:
            DT_index = 1
        
        if not np.isnan(x['xDT'][DT_index]):
            corrected_fit_profs = deadtime_correct(fit_profs[var].profile,np.exp(x['xDT'][DT_index]),dt=dt[:,np.newaxis])
            ErrRet = ErrRet + np.nansum(weights*(forward_profs[var]-corrected_fit_profs*np.log(forward_profs[var])))
        else:
            ErrRet = ErrRet + np.nansum(weights*(forward_profs[var]-fit_profs[var].profile*np.log(forward_profs[var])))
        
    return ErrRet
    
    
def Build_DLBHSRL_sparsa_Profiles(x,Const,dt=1.0,dR=37.5,return_params=False,params={},n_conv=0,cond_fun=cond_fun_default_dlb_hsrl):
    """
    dt sets the adjustment factor to convert counts to count rate needed in
    deadtime correction
    
    if return_params=True, returns the profiles of the optical parameters
    
    cond_fun - dict of conditioning functions for each variable the user wants to condition.
        a condition function should take arguments to provide an inverse and a derivative (df/dx)
        It should have the form
        f(x,opt) where opt is a string input accepting
            'inverse' - to provide an inverse operation
            'derivative' - to provide the derivative of the function
            'norm' or any other argument - results in normal conditioning operation
        if cond_fun is not provided for a variable, it will be treated as a pass through
            and the parameter will be set equal to the state variable
            
        the general format for a condition function is
        cond_function(x,*args):
            if args[0] == 'inverse':
                return np.sqrt( (x-3)/4 )
            if args[0] == 'derivative':
                return 8*x
            else:
                # actual function conditioning
                return 4*x**2 + 3
        this would be passed in for a particular state variable xA by creating
            a dict that points to the condition function and passsing that to
            this, the error function and the gradient function
            
            cond_fun['xA'] = cond_function
        
    """
    
    try:
        beta_aer = params['Backscatter_Coefficient']
        sLR = params['Lidar_Ratio']
        Tatm = params['Tatm']
    except KeyError:    
        beta_aer = cond_fun['xB'](x['xB'],'normal') #  np.exp(scale['xB']*x['xB'])
        sLR = cond_fun['xS'](x['xS'],'normal') # np.exp(scale['xS']*x['xS'])+1
    
        Tatm = np.exp(-2*np.cumsum(beta_aer*sLR,axis=1)*dR)
        
        Tatm[:,1:] = Tatm[:,:-1]
        Tatm[:,0] = 0
    
    if 'kconv' in Const.keys():
        kconv = Const['kconv']
    elif n_conv > 0:
        kconv = np.ones((1,n_conv),dtype=np.float)/n_conv  # create convolution kernel for laser pulse
    else:
        kconv= np.ones((1,1))    
    
    if 'xCam' in x.keys():
        Cam = np.exp(x['xCam'][0])
    elif 'Cam' in Const.keys():
        Cam = Const['Cam']
    else:
        Cam = 1e-5
    
    forward_profs = {}
    for var in Const.keys():
        if 'Molecular' in var:
            forward_profs[var] = np.exp(x['xG'][0])*Const[var]['mult']*(beta_aer*Cam+Const[var]['mol'])*Tatm+Const[var]['bg']
            if kconv.size > 1:
                forward_profs[var] = lp.conv2d(forward_profs[var],kconv,keep_mask=False)

        elif 'Combined' in var:
            forward_profs[var] = np.exp(x['xG'][1])*Const[var]['mult']*(beta_aer+Const[var]['mol'])*Tatm+Const[var]['bg']
            if kconv.size > 1:
                forward_profs[var] = lp.conv2d(forward_profs[var],kconv,keep_mask=False)
            
    
    if return_params:
        forward_profs['Backscatter_Coefficient'] = beta_aer.copy()
        forward_profs['Lidar_Ratio'] = sLR.copy()
        forward_profs['Tatm'] = Tatm.copy()
        if kconv.size > 1:
            forward_profs['kconv'] = kconv.copy()
        
    
    return forward_profs
    
def DLBHSRL_sparsa_Error_Gradient(x,fit_profs,Const,lam,dt=1.0,weights=np.array([1]),n_conv=0,cond_fun=cond_fun_default_dlb_hsrl):
    """
    Analytical gradient of DLBHSRL_sparsa_Error()
    """
    
    dR = fit_profs['Molecular'].mean_dR
    forward_profs = Build_DLBHSRL_sparsa_Profiles(x,Const,dt=dt,dR=dR,return_params=True,cond_fun=cond_fun)
    
    if 'kconv' in Const.keys():
        kconv = Const['kconv']
    elif n_conv > 0:
        kconv = np.ones((1,n_conv),dtype=np.float)/n_conv  # create convolution kernel for laser pulse
    else:
        kconv= np.ones((1,1)) 
    

    #obtain models without nonlinear response or background
    sig_profs = {}
    e0 = {}
#    e_dt = {}
    grad0 = {}
    
    # gradient components of each atmospheric variable
    gradErr = {}
    for var in x.keys():
        gradErr[var] = np.zeros(x[var].shape)
    
    for var in fit_profs.keys():
        sig_profs[var] = forward_profs[var]-Const[var]['bg']
        
        if 'Molecular' in var:
            deadtime = np.exp(x['xDT'][0])
            Gain = np.exp(x['xG'][0])        
            
        elif 'Combined' in var:
            deadtime = np.exp(x['xDT'][1])
            Gain = np.exp(x['xG'][1])
       
        if not np.isnan(deadtime):
            corrected_fit_profs = deadtime_correct(fit_profs[var].profile,deadtime,dt=dt[:,np.newaxis])
        else:
            corrected_fit_profs = fit_profs[var].profile.copy()
            
        # useful definitions for gradient calculations
        e0[var] = (1-corrected_fit_profs/forward_profs[var])  # error function derivative
#        e_dt[var] = dt[:,np.newaxis]**2/(dt[:,np.newaxis]+lin_profs[var]*deadtime)**2  # dead time derivative
        grad0[var] = dR*(np.sum(e0[var]*sig_profs[var],axis=1)[:,np.newaxis]-np.cumsum(e0[var]*sig_profs[var],axis=1))
    
        if 'Molecular' in var:
            # molecular gain gradient term
            if 'xG' in gradErr.keys():
                gradErr['xG'][0] = np.nansum(e0[var]*sig_profs[var])
    
            # molecular dead time gradient term
            if 'xDT' in gradErr.keys():
                gradErr['xDT'][0] = -np.nansum(dt[:,np.newaxis]*fit_profs[var].profile**2/(dt[:,np.newaxis]-fit_profs[var].profile*deadtime)**2*deadtime *np.log(forward_profs[var]))
            
            Ta = np.exp(x['xCam'][0])    # aerosol transmission into molecular channel          
            if 'xCam' in gradErr.keys():
                gradErr['xCam'][0] = np.nansum(e0[var]*Gain*Const[var]['mult']*forward_profs['Backscatter_Coefficient']*forward_profs['Tatm']*Ta)       
            
        elif 'Combined' in var:
            # comb hi gain gradient term
            if 'xG' in gradErr.keys():
                gradErr['xG'][1] = np.nansum(e0[var]*sig_profs[var])
    
            # comb hi dead time gradient term
            if 'xDT' in gradErr.keys():
                gradErr['xDT'][1] = -np.nansum(dt[:,np.newaxis]*fit_profs[var].profile**2/(dt[:,np.newaxis]-fit_profs[var].profile*deadtime)**2*deadtime *np.log(forward_profs[var]))
            
            Ta = 1.0 # aerosol transmission into aerosol channel
        
        if 'xS' in gradErr.keys():
            gradErr['xS'] = gradErr['xS'] -2*forward_profs['Backscatter_Coefficient']*grad0[var]*cond_fun['xS'](x['xS'],'derivative') # -2*forward_profs['Backscatter_Coefficient']*forward_profs['Lidar_Ratio']*grad0[var]*scale['xS']
        if 'xB' in gradErr.keys():
            gradErr['xB'] = gradErr['xB']+(-2*forward_profs['Lidar_Ratio']*grad0[var] \
                +e0[var]*Gain*Const[var]['mult']*forward_profs['Tatm']*Ta)*cond_fun['xB'](x['xB'],'derivative')

    if kconv.size > 1:
        if 'xS' in gradErr.keys():
            gradErr['xS'] = lp.conv2d(gradErr['xS'],kconv,keep_mask=False)
        if 'xB' in gradErr.keys():
            gradErr['xB'] = lp.conv2d(gradErr['xB'],kconv,keep_mask=False)
    
    return gradErr
    




def DLBHSRL_sparsa_Error_step1(x,fit_profs,Const,lam,dt=1.0,weights=np.array([1]),cond_fun=cond_fun_default_gv_hsrl):  
    
    """
    PTV Error of DLB-HSRL profiles for the two step retrieval
    where phi accounts for both extinction and errors in the geometric overlap
    estimate
    """

    dR = fit_profs[list(fit_profs.keys())[0]].mean_dR
    forward_profs = Build_DLBHSRL_sparsa_Profiles_step1(x,Const,dt=dt,dR=dR,return_params=True,cond_fun=cond_fun)
    
    ErrRet = 0    
    
    for var in lam.keys():
        deriv = lam[var]*np.nansum(np.abs(np.diff(x[var],axis=1)))+lam[var]*np.nansum(np.abs(np.diff(x[var],axis=0)))
        ErrRet = ErrRet + deriv

    for var in fit_profs:
#        print(var)
        if 'Molecular' in var:
            DT_index = 0
        elif 'Combined' in var:
            DT_index = 1
        
        if not np.isnan(x['xDT'][DT_index]):
            corrected_fit_profs = deadtime_correct(fit_profs[var].profile,np.exp(x['xDT'][DT_index]),dt=dt[:,np.newaxis])
            ErrRet = ErrRet + np.nansum(weights*(forward_profs[var]-corrected_fit_profs*np.log(forward_profs[var])))
        else:
            ErrRet = ErrRet + np.nansum(weights*(forward_profs[var]-fit_profs[var].profile*np.log(forward_profs[var])))
        
    return ErrRet
    
    
def Build_DLBHSRL_sparsa_Profiles_step1(x,Const,dt=1.0,dR=37.5,return_params=False,params={},n_conv=0,cond_fun=cond_fun_default_dlb_hsrl):
    """
    dt sets the adjustment factor to convert counts to count rate needed in
    deadtime correction
    
    if return_params=True, returns the profiles of the optical parameters
    
    cond_fun - dict of conditioning functions for each variable the user wants to condition.
        a condition function should take arguments to provide an inverse and a derivative (df/dx)
        It should have the form
        f(x,opt) where opt is a string input accepting
            'inverse' - to provide an inverse operation
            'derivative' - to provide the derivative of the function
            'norm' or any other argument - results in normal conditioning operation
        if cond_fun is not provided for a variable, it will be treated as a pass through
            and the parameter will be set equal to the state variable
            
        the general format for a condition function is
        cond_function(x,*args):
            if args[0] == 'inverse':
                return np.sqrt( (x-3)/4 )
            if args[0] == 'derivative':
                return 8*x
            else:
                # actual function conditioning
                return 4*x**2 + 3
        this would be passed in for a particular state variable xA by creating
            a dict that points to the condition function and passsing that to
            this, the error function and the gradient function
            
            cond_fun['xA'] = cond_function
        
    """
    
    try:
        beta_aer = params['Backscatter_Coefficient']
        phi = params['phi']
        Tatm = params['Tatm']
    except KeyError:    
        beta_aer = cond_fun['xB'](x['xB'],'normal') #  np.exp(scale['xB']*x['xB'])
        phi = cond_fun['xPhi'](x['xPhi'],'normal') # np.exp(scale['xS']*x['xS'])+1    
        Tatm = np.exp(phi)

    
    if 'kconv' in Const.keys():
        kconv = Const['kconv']
    elif n_conv > 0:
        kconv = np.ones((1,n_conv),dtype=np.float)/n_conv  # create convolution kernel for laser pulse
    else:
        kconv= np.ones((1,1))    
    
    if 'xCam' in x.keys():
        Cam = np.exp(x['xCam'][0])
    elif 'Cam' in Const.keys():
        Cam = Const['Cam']
    else:
        Cam = 2e-4
    
    forward_profs = {}
    for var in Const.keys():
        if 'Molecular' in var:
            forward_profs[var] = np.exp(x['xG'][0])*Const[var]['mult']*(beta_aer*Cam+Const[var]['mol'])*Tatm+Const[var]['bg']
            if kconv.size > 1:
                forward_profs[var] = lp.conv2d(forward_profs[var],kconv,keep_mask=False)

        elif 'Combined' in var:
            forward_profs[var] = np.exp(x['xG'][1])*Const[var]['mult']*(beta_aer+Const[var]['mol'])*Tatm+Const[var]['bg']
            if kconv.size > 1:
                forward_profs[var] = lp.conv2d(forward_profs[var],kconv,keep_mask=False)
            
    
    if return_params:
        forward_profs['Backscatter_Coefficient'] = beta_aer.copy()
        forward_profs['phi'] = phi.copy()
        forward_profs['Tatm'] = Tatm.copy()
        if kconv.size > 1:
            forward_profs['kconv'] = kconv.copy()
        
    
    return forward_profs
    
def DLBHSRL_sparsa_Error_Gradient_step1(x,fit_profs,Const,lam,dt=1.0,weights=np.array([1]),n_conv=0,cond_fun=cond_fun_default_gv_hsrl):
    """
    Analytical gradient of DLBHSRL_sparsa_Error()
    """
    
    dR = fit_profs['Molecular'].mean_dR
    forward_profs = Build_DLBHSRL_sparsa_Profiles_step1(x,Const,dt=dt,dR=dR,return_params=True,cond_fun=cond_fun)
    
    if 'kconv' in Const.keys():
        kconv = Const['kconv']
    elif n_conv > 0:
        kconv = np.ones((1,n_conv),dtype=np.float)/n_conv  # create convolution kernel for laser pulse
    else:
        kconv= np.ones((1,1)) 
    

    #obtain models without nonlinear response or background
    sig_profs = {}
    e0 = {}
#    e_dt = {}
    grad0 = {}
    
    # gradient components of each atmospheric variable
    gradErr = {}
    for var in x.keys():
        gradErr[var] = np.zeros(x[var].shape)
    
    for var in fit_profs.keys():
        sig_profs[var] = forward_profs[var]-Const[var]['bg']
        
        if 'Molecular' in var:
            deadtime = np.exp(x['xDT'][0])
            Gain = np.exp(x['xG'][0])        
            
        elif 'Combined' in var:
            deadtime = np.exp(x['xDT'][1])
            Gain = np.exp(x['xG'][1])
       
        if not np.isnan(deadtime):
            corrected_fit_profs = deadtime_correct(fit_profs[var].profile,deadtime,dt=dt[:,np.newaxis])
        else:
            corrected_fit_profs = fit_profs[var].profile.copy()
            
        # useful definitions for gradient calculations
        e0[var] = (1-corrected_fit_profs/forward_profs[var])  # error function derivative
#        e_dt[var] = dt[:,np.newaxis]**2/(dt[:,np.newaxis]+lin_profs[var]*deadtime)**2  # dead time derivative
        grad0[var] = dR*(np.sum(e0[var]*sig_profs[var],axis=1)[:,np.newaxis]-np.cumsum(e0[var]*sig_profs[var],axis=1))
    
        if 'Molecular' in var:
            # molecular gain gradient term
            if 'xG' in gradErr.keys():
                gradErr['xG'][0] = np.nansum(e0[var]*sig_profs[var])
    
            # molecular dead time gradient term
            if 'xDT' in gradErr.keys():
                gradErr['xDT'][0] = -np.nansum(dt[:,np.newaxis]*fit_profs[var].profile**2/(dt[:,np.newaxis]-fit_profs[var].profile*deadtime)**2*deadtime *np.log(forward_profs[var]))
            
            Ta = np.exp(x['xCam'][0])    # aerosol transmission into molecular channel          
            if 'xCam' in gradErr.keys():
                gradErr['xCam'][0] = np.nansum(e0[var]*Gain*Const[var]['mult']*forward_profs['Backscatter_Coefficient']*forward_profs['Tatm']*Ta)       
            
        elif 'Combined' in var:
            # comb hi gain gradient term
            if 'xG' in gradErr.keys():
                gradErr['xG'][1] = np.nansum(e0[var]*sig_profs[var])
    
            # comb hi dead time gradient term
            if 'xDT' in gradErr.keys():
                gradErr['xDT'][1] = -np.nansum(dt[:,np.newaxis]*fit_profs[var].profile**2/(dt[:,np.newaxis]-fit_profs[var].profile*deadtime)**2*deadtime *np.log(forward_profs[var]))
            
            Ta = 1.0 # aerosol transmission into aerosol channel
        
        if 'xPhi' in gradErr.keys():
            gradErr['xPhi'] = gradErr['xPhi']+e0[var]*sig_profs[var]*cond_fun['xPhi'](x['xPhi'],'derivative')
        if 'xB' in gradErr.keys():
            gradErr['xB'] = gradErr['xB']+e0[var]*Gain*Const[var]['mult']*forward_profs['Tatm']*Ta*cond_fun['xB'](x['xB'],'derivative')

    if kconv.size > 1:
        if 'xPhi' in gradErr.keys():
            gradErr['xPhi'] = lp.conv2d(gradErr['xPhi'],kconv,keep_mask=False)
        if 'xB' in gradErr.keys():
            gradErr['xB'] = lp.conv2d(gradErr['xB'],kconv,keep_mask=False)
    
    return gradErr




def deadtime_correct(profile,deadtime,dt=1):
    """
    Applied deadtime correction to a profile of photon counts prior to any
    other processing
    profile - lidar profile of photon counts
    deadtime - detector dead time after counting a pulse
    dt - adjustment factor to obtain photon count rate from photon counts
    """
    
    return profile*dt/(dt-deadtime*profile)

def cond_linear(x,scale,offset,operation ='normal'):
    """
    Conditioning linear function for an optimization routine.
    returns:
    (x+offset)*scale
    
    inputs:
    x - input state parameter
    scale - scale argument
    offset - offset argument
    operation - 'normal' evaluate the condition function
                'inverse' return the inverse of the function
                'derivative' return the derivative of the function
                
    setup for use in an optimizer
    cond_dict = {}
    cond_dict['xA'] = lambda x,y: cond_linear(x,scale_setting,offset_setting,y)
    """
    
    if operation == 'derivative':
        return scale*np.ones(x.shape)
    elif operation == 'inverse':
        return x/scale-offset
    else:
        return (x+offset)*scale
    
def conv_grad_step(x,a,h,hindex=None):
    """
    Calculates range convolution component for gradient calculations 
    where the signal derivative is proportional to a step function
    due to the accumulation term in the exponent (e.g. extinction)
    
    sum_k x_k sum_{i=k-n}^{k+n} a_i h_{i-k} 1_{ji}
    where x_k is the gradient of the Inv-LL with respect to the signal
    and a_i is the non-integral component of the gradient function
    and h is the convolution kernel
    hindex is the element index for each value of h where typically 0 is the
        middle value.  
    """
    
    if h.size > 1:
        if hindex is None:
            hindex = np.arange(h.size)-np.int(h.size/2)
            
        sum_val = np.zeros(a.shape)
        for ai,hi in enumerate(hindex):
            shiftval = [0]*x.ndim
            shiftval[1] = hi
            xroll = scipy.ndimage.interpolation.shift(x,shiftval,cval=0,order=0)

#            sum_val += h[ai]*a*np.cumsum((xroll)[:,::-1,...],axis=1)[:,::-1,...]
            sum_val += h[ai]*np.cumsum((xroll*a)[:,::-1,...],axis=1)[:,::-1,...]

    else:
#        sum_val = a*np.cumsum((x)[:,::-1,...],axis=1)[:,::-1,...]
        sum_val = np.cumsum((x*a)[:,::-1,...],axis=1)[:,::-1,...]
    return sum_val

def conv_grad_delta(x,a,h,hindex=None):
    """
    Calculates range convolution component for gradient calculations 
    where the signal derivative is proportional to a delta function
    (e.g. backscatter)
    
    sum_k x_k sum_{i=k-n}^{k+n} a_i h_{i-k} delta_{ji}
    where x_k is the gradient of the Inv-LL with respect to the signal
    and a_i is the signal derivative
    and h is the convolution kernel
    hindex is the element index for each value of h where typically 0 is the
        middle value.  
    """
    
    if h.size > 1:
        if hindex is None:
            hindex = np.arange(h.size)-np.int(h.size/2)
            
        sum_val = np.zeros(a.shape)
        for ai,hi in enumerate(hindex):
            shiftval = [0]*x.ndim
            shiftval[1] = hi
            xroll = scipy.ndimage.interpolation.shift(x,shiftval,cval=0,order=0)
            
#            sum_val += h[ai]*xroll*a
            sum_val += h[ai]*xroll
            
        sum_val = a*sum_val
    else:
        sum_val = x*a
    return sum_val

def conv_grad_step_superres(x,a,h,hindex=None,M=None,M0=None,weight=None):
    """
    conv_grad_step_superres(x,a,h,hindex=None,M=None,M0=None,weight=None)
    
    Compute a gradient for the case where the resolution
    of the retrieval space is higher than the observations
    
    x - low resolution observations.  Typically the error term
    a - high resolution retrieval derivative
    h - convolution kernel
    hindex - element number of h typically an array from -h.size to h.size
    M - super sample resolution.  integer of a.shape[1]/x.shape[1]
    M0 - index offset that directly maps elements of x to elements of a
        (e.g. a[M0::M]->x )
    weight-array matching the size of a that can be used to adjust weights
        of the convolution kernel to reduce edge effects
    
    """
    
    if hindex is None:
        hindex = np.arange(h.size,dtype=np.int)-np.int(h.size/2)
        
    if M is None:
        M = a.shape[1]/x.shape[1]
    
    if M0 is None:
        M0 = np.int(M/2)
        
    if h.size > 1:    
        if hindex is None:
            hindex = np.arange(h.size,dtype=np.int)-np.int(h.size/2)
        sum_val = np.zeros(a.shape)
        
        for ai,hi in enumerate(hindex.flatten()):
            imin = hi+M0
            dindmin = np.int(np.minimum(np.floor((M0+hi)/M),0))
            imin = imin-M*dindmin  # lowest index into a
            klo = -dindmin  # lowest index into x
            
            imax = (x.shape[1]-1)*M+hi+M0  # highest index into a
            dindmax = np.int(np.maximum(np.ceil((imax-a.shape[1]+1)/M),0))
            imax = imax-M*dindmax
            khi = x.shape[1]-1-dindmax  # highest index into x
            
            
            xrev = (x[:,klo:khi+1,...])[:,::-1,...]
            arev = (a[:,imin:imax+M:M,...])[:,::-1,...]
            
            xa = np.cumsum(xrev*arev,axis=1)[:,::-1,...].repeat(M,axis=1)[:,-imax:,...]  # reverse cumsum and repeat results
        
            if imax+1 > xa.shape[1]:
                xa=np.concatenate((xa[:,0:1,...].repeat(imax+1-xa.shape[1],axis=1),xa),axis=1)
                
            sum_val[:,:imax+1,...]+=h[ai]*xa
    else:
        sum_val = np.cumsum((x*a)[:,::-1,...],axis=1)[:,::-1,...]
        
        
    return sum_val

def conv_grad_delta_superres(x,a,h,hindex=None,M=None,M0=None):
    """
    conv_grad_delta_superres(x,a,h,hindex=None,M=None,M0=None)
    
    Compute a gradient for the case where the resolution
    of the retrieval space is higher than the observations
    
    x - low resolution observations.  Typically the error term
    a - high resolution retrieval derivative
    h - convolution kernel
    hindex - element number of h typically an array from -h.size to h.size
    M - super sample resolution.  integer of a.shape[1]/x.shape[1]
    M0 - index offset that directly maps elements of x to elements of a
        (e.g. a[M0::M]->x )
    """
    
    if hindex is None:
        hindex = np.arange(h.size,dtype=np.int)-np.int(h.size/2)
        
    if M is None:
        M = a.shape[1]/x.shape[1]
    
    if M0 is None:
        M0 = np.int(M/2)
        
    if h.size > 1: 
        if hindex is None:
            hindex = np.arange(h.size,dtype=np.nit)-np.int(h.size/2)
    
        sum_val = np.zeros(a.shape)
        for ai,hi in enumerate(hindex.flatten()):    
            imin = hi+M0
            dindmin = np.int(np.minimum(np.floor((M0+hi)/M),0))
            imin = imin-M*dindmin  # lowest index into a
            klo = -dindmin  # lowest index into x
            
            imax = (x.shape[1]-1)*M+hi+M0  # highest index into a
            dindmax = np.int(np.maximum(np.ceil((imax-a.shape[1]+1)/M),0))
            imax = imax-M*dindmax
            khi = x.shape[1]-1-dindmax  # highest index into x
    
              
            sum_val[:,imin:imax+M:M,...]+=h[ai]*x[:,klo:khi+1,...]*a[:,imin:imax+M:M,...] 
            
    else:
        sum_val = x*a
        
    return sum_val

def cond_exp(x,scale,offset,operation ='normal'):
    """
    Conditioning exponential function for an optimization routine.
    returns:
    np.exp((x+offset)*scale)
    
    inputs:
    x - input state parameter
    scale - scale argument
    offset - offset argument
    operation - 'normal' evaluate the condition function
                'inverse' return the inverse of the function
                'derivative' return the derivative of the function
                
    setup for use in an optimizer
    cond_dict = {}
    cond_dict['xA'] = lambda x,y: cond_exp(x,scale_setting,offset_setting,y)
    """
    
    if operation == 'derivative':
        return np.exp((x+offset)*scale)*scale
    elif operation == 'inverse':
        return np.log(x)/scale-offset
    else:
        return np.exp((x+offset)*scale)
        
def cond_arctan(x,scale,offset,bnd_lo,bnd_up,operation ='normal'): 
    """
    Conditioning arctan function for an optimization routine.
    returns:
    np.arctan((x+offset)*scale)/np.pi
    
    inputs:
    x - input state parameter
    scale - scale argument
    offset - offset argument
    bnd_lo - lower bound on output
    bnd_up - upper bound on output
    operation - 'normal' evaluate the condition function
                'inverse' return the inverse of the function
                'derivative' return the derivative of the function
                
    setup for use in an optimizer
    cond_dict = {}
    cond_dict['xA'] = lambda x,y: cond_arctan(x,scale_setting,offset_setting,y)
    """
    
    if operation == 'derivative':
        return (bnd_up-bnd_lo)*(2.0/np.pi)*scale/(1+((x+offset)*scale)**2)
    elif operation == 'inverse':
        return np.tan((x-0.5*(bnd_lo+bnd_up))/((bnd_up-bnd_lo)*2.0/np.pi))/scale-offset
    else:
        return 0.5*(bnd_lo+bnd_up)+(bnd_up-bnd_lo)*2.0/np.pi*np.arctan((x+offset)*scale)

def cond_pass(x,operation='normal'):
    """
    Conditioning passthrough function for an optimization routine.
    returns:
    x
    
    inputs:
    x - input state parameter

    operation - 'normal' evaluate the condition function
                'inverse' return the inverse of the function
                'derivative' return the derivative of the function
                
    setup for use in an optimizer
    cond_dict = {}
    cond_dict['xA'] = lambda x,y: cond_pass(x,y)
    """
    if operation == 'derivative':
        return np.ones(x.shape)
    else:
        return x

def square_laser_pulse(T,dt,norm=True):
    """
    square_laser_pulse(T,dt)
    Calculate the symmetric convolution kernel for a square pulse
    with
    T = pulse length
    dt = time bin width
    
    returns
    pulse definition in bins
    the pulse index array
    
    """
    
    N = np.ceil(0.5*(T/dt-1))
    pulse_bins = np.arange(-N,N+1,dtype=np.int)
    pulse = np.minimum(0.5*T/(0.5*dt+dt*np.abs(pulse_bins)),1)
    if norm:
        pulse=pulse/np.sum(pulse)
    
    pulse_bins=pulse_bins[np.newaxis,:]
    pulse = pulse[np.newaxis,:]
    
    return pulse,pulse_bins