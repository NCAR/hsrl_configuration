% find minimum reflectivity values
clear all;
close all;

project='cset'; % socrates, cset, aristo, otrec
quality='qc2'; % field, qc1, qc2
freqData='2hz'; % 10hz, 100hz, or 2hz
whichModel='era5'; % ecmwf or era5

formatOut = 'yyyymmdd_HHMM';

[modeldir ~]=modelDir(project,whichModel,freqData);
outdir='/scr/snow2/rsfdata/projects/cset/hsrl/model/ERA5interp/';

topodir=topoDir(project);

indir='/scr/snow2/rsfdata/projects/cset/hsrl/raw/raw_denoised_tempPress/';

planeDir='/scr/snow2/rsfdata/projects/cset/GV/gv_data/';

flightTimes=load('~/git/hsrl_configuration/projDir/postProcessing/addModel/inFiles/flights_cset.txt');

c=299792458;
binwidth_ns=50;
bin0=37;
lidar_tilt=[0,4];

%% Go through flights
for ii=1:size(flightTimes,1)
    disp(['Flight ',num2str(ii)]);
    
    % Load HSRL data
    disp('Getting HSRL data ...');
    
    startTime=datetime([flightTimes(ii,1:4),0,0]);
    endTime=datetime([flightTimes(ii,5:8),0,0]);
    
    fileList=makeFileListHSRL(indir,startTime,endTime,'xxxxxxx20YYMMDDxhhmmss');
    
    data.time=[];
    data.longitude=[];
    data.latitude=[];
    forDim=[];
    telDir=[];
    
    for jj=1:size(fileList,2)
        infile=fileList{jj};
        
        intime=ncread(infile,'DATA_time');
        milSec=double(intime(7,:))+double(intime(8,:))*1e-3;
        
        data.time=cat(2,data.time,datetime(intime(1,:),intime(2,:),intime(3,:),intime(4,:),...
            intime(5,:),intime(6,:),milSec));
        
        data.longitude=cat(2,data.longitude,ncread(infile,'iwg1_Lon')');
        data.latitude=cat(2,data.latitude,ncread(infile,'iwg1_Lat')');
        telDir=cat(2,telDir,double(ncread(infile,'TelescopeDirection')'));
        
        forDim=cat(2,forDim,double(ncread(infile,'molecular')));
    end
    
    disp(['Processing ',datestr(data.time(1),'yyyy-mm-dd HH:MM:SS'),' to ',...
        datestr(data.time(end),'yyyy-mm-dd HH:MM:SS')]);
    
    %% Load plane data
    disp('Getting GV data ...');
    
    if ii<10
        planeFileIn=dir([planeDir,'CSETrf0',num2str(ii),'.nc']);
    else
        planeFileIn=dir([planeDir,'CSETrf',num2str(ii),'.nc']);
    end
    
    planeFile=[planeFileIn.folder,'/',planeFileIn.name];
    ROLLin=ncread(planeFile,'ROLL');
    PITCHin=ncread(planeFile,'PITCH');
    ALTin=ncread(planeFile,'GGALT');
    planeTimeIn=ncread(planeFile,'Time');
    refTimePlane=datetime(year(data.time(1)),month(data.time(1)),day(data.time(1)),0,0,0);
    planeTime=refTimePlane+seconds(planeTimeIn);
    
    % Synchronize
    hsrlTT=timetable(data.time',data.longitude');
    gvTT=timetable(planeTime,ROLLin,PITCHin,ALTin);
    
    ttsync=synchronize(hsrlTT,gvTT,'first','linear');
    
    %% Range and altitude
    rangeDim=0:size(forDim,1)-1;
    range_array = (rangeDim-bin0)*binwidth_ns*1e-9*c/2;
    data.range=repmat(range_array',1,length(data.time));
    
    % Altitude
    telDir(telDir==0)=-1;
    data.asl=(data.range.*telDir.*cos((ttsync.ROLLin'-lidar_tilt(2).*telDir)*pi/180)...
        .*cos((ttsync.PITCHin'+lidar_tilt(1))*pi/180)+ttsync.ALTin');
        
    %% Model data
    disp('Getting model data ...');
    if strcmp(whichModel,'era5')
        modelData=read_era5_hsrl(modeldir,data.time(1),data.time(end),1);
    elseif strcmp(whichModel,'ecmwf')
        modelData=read_ecmwf_hsrl(modeldir,data.time(1),data.time(end),1);
    end
    
    %% Topo data
    [modelData.topo modelData.topolon modelData.topolat]=read_gtopo30(topodir,modelData.lon,modelData.lat);
    
    %% Remove sst data that is over land
    lonMat=double(repmat(modelData.lon,1,size(modelData.z,2),size(modelData.z,4)));
    latMat=double(repmat(fliplr(modelData.lat'),size(modelData.z,1),1,size(modelData.z,4)));
    timeMat=repmat(datenum(modelData.time),size(modelData.z,1),1,size(modelData.z,2));
    timeMat=permute(timeMat,[1,3,2]);
    
    % Interpolate topo data to model grid
    topoModel=interpn(modelData.topolon',modelData.topolat',modelData.topo',...
        lonMat(:,:,1),latMat(:,:,1));
    
    if strcmp(whichModel,'ecmwf')
        for ll=1:size(modelData.sstSurf,3)
            tempSST=modelData.sstSurf(:,:,ll);
            tempSST(topoModel>0)=nan;
            modelData.sstSurf(:,:,ll)=tempSST;
        end
    end
    %% Interpolate
    disp('Interpolating to HSRL track ...');
    
    % Make thinned out time vector
    % Get 10 second interval
    %     startMinute=datetime(year(data.time(1)),month(data.time(1)),day(data.time(1)),...
    %         hour(data.time(1)),minute(data.time(1)),0);
    %     indInt=find(data.time==startMinute+minutes(1)+seconds(10))-find(data.time==startMinute+minutes(1));
    if strcmp(freqData,'10hz')
        indInt=100;
    elseif strcmp(freqData,'100hz')
        indInt=1000;
    elseif strcmp(freqData,'2hz')
        indInt=20;
    end
    timeInd=1:indInt:length(data.time);
    
    % 3D variables
    int.tempHSRL=[];
    int.zHSRL=[];
    int.pHSRL=[];
    
    for jj=1:size(modelData.z,3)
        Vq = interpn(lonMat,latMat,timeMat,squeeze(modelData.Temperature(:,:,jj,:)),...
            wrapTo360(data.longitude(timeInd)),data.latitude(timeInd),datenum(data.time(timeInd)));
        int.tempHSRL=cat(1,int.tempHSRL,Vq);
        Vq = interpn(lonMat,latMat,timeMat,squeeze(modelData.z(:,:,jj,:)),...
            wrapTo360(data.longitude(timeInd)),data.latitude(timeInd),datenum(data.time(timeInd)));
        int.zHSRL=cat(1,int.zHSRL,Vq);
        Vq = interpn(lonMat,latMat,timeMat,squeeze(modelData.p(:,:,jj,:)),...
            wrapTo360(data.longitude(timeInd)),data.latitude(timeInd),datenum(data.time(timeInd)));
        int.pHSRL=cat(1,int.pHSRL,Vq);
    end
    
    % 2D variables
    surfData.pHSRL = interpn(lonMat,latMat,timeMat,modelData.pSurf,...
        wrapTo360(data.longitude),data.latitude,datenum(data.time));
    surfData.tempHSRL = interpn(lonMat,latMat,timeMat,modelData.tSurf,...
        wrapTo360(data.longitude),data.latitude,datenum(data.time));
    
    % Topo
    surfData.zHSRL=interpn(modelData.topolon',modelData.topolat',modelData.topo',...
        wrapTo360(data.longitude),data.latitude);
    
    intFields=fields(int);
    
    % Remove nans
    for ll=1:length(intFields)
        int.(intFields{ll})= int.(intFields{ll})(any(~isnan(int.(intFields{ll})),2),:);
    end
    
    surfatInds=surfData.zHSRL(timeInd);
    
    % Replace or add surface values
    for mm=1:length(data.time(timeInd))
        zLevels=int.zHSRL(:,mm);
        zSurf=surfatInds(mm);
        zLevels(zLevels<=zSurf)=nan;
        nanInds=find(isnan(zLevels));
        if isempty(nanInds)
            surfInd=length(zLevels);
        else
            surfInd=(min(nanInds));
        end
        for ll=1:length(intFields);
            int.(intFields{ll})(nanInds,mm)=nan;
            int.(intFields{ll})(surfInd,mm)=surfData.(intFields{ll})(timeInd(mm));
        end
    end
    
    %% Interpolate to HSRL grid
    
    disp('Interpolating to HSRL grid ...');
    
    timeMatModel=repmat(datenum(data.time(timeInd)),size(int.zHSRL,1),1);
    
    % Remove data that is too far below surface or too far out
    aslGood=data.asl;
    outInds=find(aslGood<-200 | aslGood>15000);
    aslGood(outInds)=nan;
    keepInds=find(~isnan(aslGood));
    
    % Output coordinates
    timeMatHSRL=repmat(datenum(data.time),size(data.range,1),1);
    xq=[timeMatHSRL(:) aslGood(:)];
    nanXq=find(any(isnan(xq),2));
    xq(nanXq,:)=[];
    
    newGrid=(0:10:15000);
    [X Y]=meshgrid(datenum(data.time(timeInd)),newGrid);
    newTimeGrid=repmat(datenum(data.time(timeInd)),length(newGrid),1);
    
    for ll=1:length(intFields)
        if ~strcmp(intFields{ll},'zHSRL')
            disp(intFields{ll});
            % First make 1d interpolation to a regular grid
            v=int.(intFields{ll});
            vq=[];
            for mm=1:size(timeMatModel,2)
                x=int.zHSRL(:,mm);
                y=v(:,mm);
                xy=cat(2,x,y);
                nanXY=find(any(isnan(xy),2));
                xy(nanXY,:)=[];
                if ~isempty(xy)
                    vq1 = interp1(xy(:,1),xy(:,2),newGrid);
                    vq=cat(2,vq,vq1');
                else
                    vq=cat(2,vq,nan(length(newGrid),1));
                end
            end
            % Then grab the data points at the HSRL grid
            Vq = interp2(X,Y,vq,xq(:,1),xq(:,2));
            modelvar=nan(size(data.range));
            modelvar(keepInds)=Vq;
            %                         %surf(data.time(105000:150000),data.asl(:,105000:150000),modelvar(:,105000:150000),'edgecolor','none');
            %                         surf(data.time,data.asl,modelvar,'edgecolor','none');
            %             surf(data.time(1:5000),data.asl(:,1:5000),modelvar(:,1:5000),'edgecolor','none');
            %                         view(2);
            
            % Fill in nans with last good value
            for kk=1:size(modelvar,2)
                modRay=modelvar(:,kk);
                lastInd=max(find(~isnan(modRay)));
                modelvar(lastInd+1:end,kk)=modRay(lastInd);
            end
            disp(['Saving ',intFields{ll},' data ...']);
            if strcmp(intFields{ll},'tempHSRL')
                tempHSRL=modelvar;
                tempHSRL=tempHSRL+273.15;
                save([outdir,whichModel,'.',intFields{ll},'.',datestr(data.time(1),'YYYYmmDD_HHMMSS'),'_to_',...
                    datestr(data.time(end),'YYYYmmDD_HHMMSS'),'.Flight',num2str(ii),'.mat'],'tempHSRL');
            elseif strcmp(intFields{ll},'pHSRL')
                pHSRL=modelvar;
                pHSRL=pHSRL*100;
                save([outdir,whichModel,'.',intFields{ll},'.',datestr(data.time(1),'YYYYmmDD_HHMMSS'),'_to_',...
                    datestr(data.time(end),'YYYYmmDD_HHMMSS'),'.Flight',num2str(ii),'.mat'],'pHSRL');
            end
        end
    end
    
    disp(['Saving surface data ...']);
    timeHSRL=data.time;
    save([outdir,whichModel,'.time.',datestr(data.time(1),'YYYYmmDD_HHMMSS'),'_to_',...
        datestr(data.time(end),'YYYYmmDD_HHMMSS'),'.Flight',num2str(ii),'.mat'],'timeHSRL');
    topo=surfData.zHSRL;
    save([outdir,whichModel,'.topo.',datestr(data.time(1),'YYYYmmDD_HHMMSS'),'_to_',...
        datestr(data.time(end),'YYYYmmDD_HHMMSS'),'.Flight',num2str(ii),'.mat'],'topo');
    aslHSRL=data.asl;
    save([outdir,whichModel,'.asl.',datestr(data.time(1),'YYYYmmDD_HHMMSS'),'_to_',...
        datestr(data.time(end),'YYYYmmDD_HHMMSS'),'.Flight',num2str(ii),'.mat'],'aslHSRL');
end

