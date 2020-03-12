% add model data to cfradial files
clear all;
close all;

project='cset'; % socrates, cset, aristo, otrec
quality='qc2'; % field, qc1, qc2
freqData='2hz'; % 10hz, 100hz, or 2hz
whichModel='era5'; % ecmwf or era5

modeldir='/scr/snow2/rsfdata/projects/cset/hsrl/model/ERA5interp/';

caseList=load('~/git/hsrl_configuration/projDir/postProcessing/addModel/inFiles/flights_cset.txt');

indir='/scr/snow2/rsfdata/projects/cset/hsrl/raw/raw_denoised_tempPress/';

%% Run processing

% Go through flights
for ii=15:size(caseList,1)
    
    disp(['Flight ',num2str(ii)]);
    
    startTime=datetime([caseList(ii,1:4),0,0]);
    endTime=datetime([caseList(ii,5:8),0,0]);
   
    fileList=makeFileListHSRL(indir,startTime,endTime,'xxxxxxx20YYMMDDxhhmmss');
   
    if ~isempty(fileList)
        
        % Get model data
        model.p=[];
        model.temp=[];
        
        model=read_model_HSRL(model,modeldir,startTime,endTime);
        timeModelNum=datenum(model.time);
                
        %% Loop through hsrl data files
        for jj=1:length(fileList)
            infile=fileList{jj};
            
            disp(infile);
            
            intime=ncread(infile,'DATA_time');
            milSec=double(intime(7,:))+double(intime(8,:))*1e-3;
            
            timeHSRL=datetime(intime(1,:),intime(2,:),intime(3,:),intime(4,:),...
                intime(5,:),intime(6,:),milSec);
            
            timeHSRLNum=datenum(timeHSRL);
            
            [C,ia,ib] = intersect(timeHSRLNum,timeModelNum);
            
            if length(timeHSRL)~=length(ib)
                warning('Times do not match up. Skipping file.')
                continue
            end
            
            % Write output
            fillVal=-999999;
            
            modVars=fields(model);
            
            for kk=1:length(modVars)
                if ~strcmp((modVars{kk}),'time') & ~strcmp((modVars{kk}),'asl')
                    modOut.(modVars{kk})=model.(modVars{kk})(:,ib);
                    modOut.(modVars{kk})(isnan(modOut.(modVars{kk})))=fillVal;
                end
            end
            
            % Open file
            ncid = netcdf.open(infile,'WRITE');
            netcdf.setFill(ncid,'FILL');
            
            % Get dimensions
            dimtime = netcdf.inqDimID(ncid,'time');
            dimbincount = netcdf.inqDimID(ncid,'bincount');
            
            % Define variables
            netcdf.reDef(ncid);
            varidP = netcdf.defVar(ncid,'PRESS','NC_FLOAT',[dimbincount dimtime]);
            netcdf.defVarFill(ncid,varidP,false,fillVal);
            varidT = netcdf.defVar(ncid,'TEMP','NC_FLOAT',[dimbincount dimtime]);
            netcdf.defVarFill(ncid,varidT,false,fillVal);
            netcdf.endDef(ncid);
            
            % Write variables
            netcdf.putVar(ncid,varidP,modOut.p);
            netcdf.putVar(ncid,varidT,modOut.temp);
                        
            netcdf.close(ncid);
            
            clear ncid
            
            % Write attributes
            ncwriteatt(infile,'PRESS','long_name','pressure');
            ncwriteatt(infile,'PRESS','standard_name','air_pressure');
            ncwriteatt(infile,'PRESS','units','Pa');
            ncwriteatt(infile,'PRESS','grid_mapping','grid_mapping');
            ncwriteatt(infile,'PRESS','coordinates','time bincount');
            
            ncwriteatt(infile,'TEMP','long_name','temperature');
            ncwriteatt(infile,'TEMP','standard_name','air_temperature');
            ncwriteatt(infile,'TEMP','units','K');
            ncwriteatt(infile,'TEMP','grid_mapping','grid_mapping');
            ncwriteatt(infile,'TEMP','coordinates','time bincount');
                        
            clear ncid
        end
    end
end