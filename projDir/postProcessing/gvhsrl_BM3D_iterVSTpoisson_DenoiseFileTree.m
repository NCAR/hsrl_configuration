% add the path the the BM3D denoising routine
%addpath(genpath('/h/eol/mhayman/MatlabCode/iterVSTpoisson_STANDALONE'));
addpath(genpath('./bm3d_scripts/'));
%addpath('/h/eol/mhayman/PythonScripts/HSRL_Processing/libraries/matlab_code')  % add path to matlab libraries used in this script
% Trace the HSRL file structure and apply BM3D denoising to all raw data
% found there.

base_path = '/scr/snow2/rsfdata/projects/cset/hsrl/raw/raw_denoised/2015/07/01';

data_name = {'molecular','combined_hi','combined_lo','cross'};

poisson_thin = true;

file_list = trace_file_tree(base_path,'.nc',20e6);

denoise_var = true;

for fi =1:length(file_list)
    file_name = file_list{fi};
    disp(file_name)
    fprintf('file %i of %i\n',fi,length(file_list))
    
    for ai = 1:length(data_name)
        
        disp(data_name{ai})

%         only run denoising if the denoised data does not already exist
        ncid = netcdf.open(file_name,'nowrite');
        try
            ID = netcdf.inqVarID(ncid,[data_name{ai},'_BM3D']);
%             netcdf.close(ncid)
            disp(['Skipping ',data_name{ai},'.  It is already denoised.'])
            run_main = false;
        catch exception
            run_main = true;
        end
        netcdf.close(ncid)
        if poisson_thin
            ncid = netcdf.open(file_name,'nowrite');
            try
                ID = netcdf.inqVarID(ncid,[data_name{ai},'_pthin_ver_BM3D']);
%                 netcdf.close(ncid)
                disp(['Skipping ',data_name{ai},' thinned data.  It is already denoised.'])
                run_thineed = false;
            catch exception
                run_thinned = true;
            end
            netcdf.close(ncid)
        end
        if run_main || (run_thinned && poisson_thin)
            ncid = netcdf.open(file_name,'nowrite');
%             netcdf.close(ncid)
            try
                % check if the variable to be denoised exists in the netcdf
                ID = netcdf.inqVarID(ncid,data_name{ai});
                denoise_var = true;
            catch exception
                disp([data_name{ai}, ' not found in ', file_name])
                denoise_var = false;
            end
            netcdf.close(ncid)
            
            if denoise_var
                Photon_Counts  = double(ncread(file_name,data_name{ai}));
                
                if poisson_thin && run_thinned
                    disp('Poisson thinning the profile')
                    pcounts_fit = binornd(Photon_Counts,0.5);
                    pcounts_ver = Photon_Counts - pcounts_fit;
                    
                    disp('Denoising Poisson thinned fit profile')
                    pfit_Est = iterVSTpoisson(pcounts_fit);
                    
                    try
                        nccreate(file_name,[data_name{ai},'_pthin_fit_BM3D'],'Dimensions',{'bincount','time'})
                        ncwrite(file_name,[data_name{ai},'_pthin_fit_BM3D'],pfit_Est)
                    catch exception
                        disp(['failed to write ',[data_name{ai},'_pthin_fit_BM3D'],' to ',file_name])
                    end
                    
                    disp('Denoising Poisson thinned verification profile')
                    pver_Est = iterVSTpoisson(pcounts_ver);
                    
                    try
                        nccreate(file_name,[data_name{ai},'_pthin_ver_BM3D'],'Dimensions',{'bincount','time'})
                        ncwrite(file_name,[data_name{ai},'_pthin_ver_BM3D'],pver_Est)
                    catch exception
                        disp(['failed to write ',[data_name{ai},'_pthin_ver_BM3D'],' to ',file_name])
                    end
                    
                end
                if run_main
                    disp('Denoising profile')
                    Photon_Est = iterVSTpoisson(Photon_Counts);

                    try
                        nccreate(file_name,[data_name{ai},'_BM3D'],'Dimensions',{'bincount','time'})
                        ncwrite(file_name,[data_name{ai},'_BM3D'],Photon_Est)
                    catch exception
                        disp(['failed to write ',[data_name{ai},'_BM3D'],' to ',file_name])
                    end
                end
            end

        end

    end

end