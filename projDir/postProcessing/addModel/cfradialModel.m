% add model data to cfradial files
clear all;
close all;

addpath(genpath('~/git/HCR_configuration/projDir/qc/dataProcessing/'));

project='otrec'; % socrates, cset, aristo, otrec
quality='qc1'; % field, qc1, qc2
freqData='10hz';
whichModel='era5';

formatOut = 'yyyymmdd';

infile=['~/git/HCR_configuration/projDir/qc/dataProcessing/scriptsFiles/flights_',project,'_data.txt'];

caseList = table2array(readtable(infile));

indir=HCRdir(project,quality,freqData);

[~,modeldir]=modelDir(project,whichModel,freqData);

%% Run processing

% Go through flights
for ii=1:size(caseList,1)
    
    disp(['Flight ',num2str(ii)]);
    
    startTime=datetime(caseList(ii,1:6));
    endTime=datetime(caseList(ii,7:12));
    
    fileList=makeFileList(indir,startTime,endTime,'xxxxxx20YYMMDDxhhmmss',1);
    
    if ~isempty(fileList)
        
        % Get model data
        model.p=[];
        model.temp=[];
        %model.asl=[];
        model.sst=[];
        model.topo=[];
        model.uSurf=[];
        model.vSurf=[];
        model.rh=[];
        
        model=read_model(model,modeldir,startTime,endTime);
        timeModelNum=datenum(model.time);
        
        %model.sst(model.topo>0)=nan;
        
        %% Loop through HCR data files
        for jj=1:length(fileList)
            infile=fileList{jj};
            
            disp(infile);
            
            % Find times that are equal
            startTimeIn=ncread(infile,'time_coverage_start')';
            startTimeFile=datetime(str2num(startTimeIn(1:4)),str2num(startTimeIn(6:7)),str2num(startTimeIn(9:10)),...
                str2num(startTimeIn(12:13)),str2num(startTimeIn(15:16)),str2num(startTimeIn(18:19)));
            timeRead=ncread(infile,'time')';
            timeHCR=startTimeFile+seconds(timeRead);
            
            timeHcrNum=datenum(timeHCR);
            
            [C,ia,ib] = intersect(timeHcrNum,timeModelNum);
            
            if length(timeHCR)~=length(ib)
                warning('Times do not match up. Skipping file.')
                continue
            end
            
            % Write output
            fillVal=-9999;
            
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
            dimrange = netcdf.inqDimID(ncid,'range');
            
            % Define variables
            netcdf.reDef(ncid);
            varidP = netcdf.defVar(ncid,'PRESS','NC_FLOAT',[dimrange dimtime]);
            netcdf.defVarFill(ncid,varidP,false,fillVal);
            varidT = netcdf.defVar(ncid,'TEMP','NC_FLOAT',[dimrange dimtime]);
            netcdf.defVarFill(ncid,varidT,false,fillVal);
            varidRH = netcdf.defVar(ncid,'RH','NC_FLOAT',[dimrange dimtime]);
            netcdf.defVarFill(ncid,varidRH,false,fillVal);
            varidSST = netcdf.defVar(ncid,'SST','NC_FLOAT',[dimtime]);
            netcdf.defVarFill(ncid,varidSST,false,fillVal);
            varidTOPO = netcdf.defVar(ncid,'TOPO','NC_FLOAT',[dimtime]);
            netcdf.defVarFill(ncid,varidTOPO,false,fillVal);
            varidUSURF = netcdf.defVar(ncid,'U_SURF','NC_FLOAT',[dimtime]);
            netcdf.defVarFill(ncid,varidUSURF,false,fillVal);
            varidVSURF = netcdf.defVar(ncid,'V_SURF','NC_FLOAT',[dimtime]);
            netcdf.defVarFill(ncid,varidVSURF,false,fillVal);
            netcdf.endDef(ncid);
            
            % Write variables
            netcdf.putVar(ncid,varidP,modOut.p);
            netcdf.putVar(ncid,varidT,modOut.temp);
            netcdf.putVar(ncid,varidRH,modOut.rh);
            netcdf.putVar(ncid,varidSST,modOut.sst);
            netcdf.putVar(ncid,varidTOPO,modOut.topo);
            netcdf.putVar(ncid,varidUSURF,modOut.uSurf);
            netcdf.putVar(ncid,varidVSURF,modOut.vSurf);
            
            netcdf.close(ncid);
            
            % Write attributes
            ncwriteatt(infile,'PRESS','long_name','pressure');
            ncwriteatt(infile,'PRESS','standard_name','air_pressure');
            ncwriteatt(infile,'PRESS','units','hPa');
            ncwriteatt(infile,'PRESS','grid_mapping','grid_mapping');
            ncwriteatt(infile,'PRESS','coordinates','time range');
            
            ncwriteatt(infile,'TEMP','long_name','temperature');
            ncwriteatt(infile,'TEMP','standard_name','air_temperature');
            ncwriteatt(infile,'TEMP','units','degC');
            ncwriteatt(infile,'TEMP','grid_mapping','grid_mapping');
            ncwriteatt(infile,'TEMP','coordinates','time range');
            
            ncwriteatt(infile,'RH','long_name','relative_humidity');
            ncwriteatt(infile,'RH','standard_name','relative_humidity');
            ncwriteatt(infile,'RH','units','%');
            ncwriteatt(infile,'RH','grid_mapping','grid_mapping');
            ncwriteatt(infile,'RH','coordinates','time range');
            
            ncwriteatt(infile,'SST','long_name','sea_surface_temperature');
            ncwriteatt(infile,'SST','standard_name','sea_surface_temperature');
            ncwriteatt(infile,'SST','units','degC');
            ncwriteatt(infile,'SST','coordinates','time');
            
            ncwriteatt(infile,'TOPO','long_name','terrain_height_above_mean_sea_level');
            ncwriteatt(infile,'TOPO','units','m');
            ncwriteatt(infile,'TOPO','coordinates','time');
            
            ncwriteatt(infile,'U_SURF','long_name','u_wind_velocity');
            ncwriteatt(infile,'U_SURF','standard_name','eastward_wind');
            ncwriteatt(infile,'U_SURF','units','m/s');
            ncwriteatt(infile,'U_SURF','coordinates','time');
            
            ncwriteatt(infile,'V_SURF','long_name','v_wind_velocity');
            ncwriteatt(infile,'V_SURF','standard_name','northward_wind');
            ncwriteatt(infile,'V_SURF','units','m/s');
            ncwriteatt(infile,'V_SURF','coordinates','time');
            
        end
    end
end