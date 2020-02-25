function ecmwfData = read_ecmwf_hsrl(indir,startTime,endTime,SSTyes)
% Find the right time span for era5 data and read them in

%refTime=datetime(1900,1,1,0,0,0);

%Initialize output
ecmwfData=[];

% Get intime hours
firstHour=hour(startTime);
threeHour=floor(firstHour/3)*3;

startHour=datetime(year(startTime),month(startTime),day(startTime),threeHour,0,0);
endHour=datetime(year(endTime),month(endTime),day(endTime),hour(endTime)+3,0,0);

inhours=startHour:hours(3):endHour;


pHours=[];
tHours=[];
rhHours=[];
zHours=[];
uHours=[];
vHours=[];
pSurf=[];
tSurf=[];
rhSurf=[];
zSurf=[];
uSurf=[];
vSurf=[];
if SSTyes
    sstHours=[];
    sstSurf=[];
end

modelFiles=dir([indir,'*.nc']);
fileNames=cell2mat({modelFiles.name}');
fileTime=datetime(year(startTime),str2num(fileNames(:,12:13)),str2num(fileNames(:,14:15)),...
    str2num(fileNames(:,16:17)),0,0);
runTime=datetime(year(startTime),str2num(fileNames(:,4:5)),str2num(fileNames(:,6:7)),...
    str2num(fileNames(:,8:9)),0,0);

% Find correct model run. (The first two files in each model run don't have
% the surface data so we go to the previous one.)
fileTime(runTime>inhours(end))=[];
runTime(runTime>inhours(end))=[];
lastRunHour=hour(runTime(end));
if lastRunHour==0 & (any(hour(inhours)==0) | any(hour(inhours)==3))
    modelRunTime=runTime(end)-hours(12);
elseif lastRunHour==0 & ~(any(hour(inhours)==0) | any(hour(inhours)==3))
    modelRunTime=runTime(end);
elseif lastRunHour==12 & (any(hour(inhours)==12) | any(hour(inhours)==15))
    modelRunTime=runTime(end)-hours(12);
elseif lastRunHour==12 & ~(any(hour(inhours)==12) | any(hour(inhours)==15))
    modelRunTime=runTime(end);
end

rightFiles=[];
for kk=1:length(inhours) %Loop through all hours
    fileInd=find(fileTime==inhours(kk) & runTime==modelRunTime);
    rightFiles=cat(1,rightFiles,[modelFiles(fileInd).folder,'/',modelFiles(fileInd).name]);
    disp(modelFiles(fileInd).name);
end
if size(rightFiles,1)==0
    disp('No model data found.');
    return
end

% Because variables are not always called the same we need to find the
% right namds
info1=ncinfo(rightFiles(1,:));
varNames={info1.Variables.Name};
lonName=varNames{find(contains(varNames,'lon_'))};
latName=varNames{find(contains(varNames,'lat_'))};
levName=varNames{find(contains(varNames,'lv_'))};

lonRean=ncread(rightFiles(1,:),lonName);
lonRean=wrapTo360(lonRean);
latRean=ncread(rightFiles(1,:),latName);
levRean=ncread(rightFiles(1,:),levName);

%%%%%%%%%%%%%%%%%%% start here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ii=1:size(rightFiles,1)
    t=nan(length(lonRean),length(latRean),length(levRean));
    rh=nan(length(lonRean),length(latRean),length(levRean));
    z=nan(length(lonRean),length(latRean),length(levRean));
    p=nan(length(lonRean),length(latRean),length(levRean));
    
    info2=ncinfo(rightFiles(ii,:));
    varNames2={info2.Variables.Name};
    tName=varNames2{find(contains(varNames2,'T_'))};
    rhName=varNames2{find(contains(varNames2,'R_'))};
    zName=varNames2{find(contains(varNames2,'GH_'))};
    for jj=1:length(levRean)
        p(:,:,jj)=levRean(jj);
        t(:,:,jj)=fliplr(ncread(rightFiles(ii,:),tName,[1,1,jj],[inf,inf,1])-273.15);
        rh(:,:,jj)=fliplr(ncread(rightFiles(ii,:),rhName,[1,1,jj],[inf,inf,1]));
        z(:,:,jj)=fliplr(ncread(rightFiles(ii,:),zName,[1,1,jj],[inf,inf,1]));
    end
        
    pHours=cat(4,pHours,p);
    tHours=cat(4,tHours,t);
    rhHours=cat(4,rhHours,rh);
    zHours=cat(4,zHours,z);
    
    % Surface data
    psName=varNames2{find(contains(varNames2,'MSL_'))};
    tsName=varNames2{find(contains(varNames2,'2T_'))};
    tdName=varNames2{find(contains(varNames2,'2D_'))};
    uName=varNames2{find(contains(varNames2,'10U_'))};
    vName=varNames2{find(contains(varNames2,'10V_'))};
    
    pS=fliplr(ncread(rightFiles(ii,:),psName)./100);
    tS=fliplr(ncread(rightFiles(ii,:),tsName)-273.15);
    td=fliplr(ncread(rightFiles(ii,:),tdName)-273.15);
    rhS=100*(exp((17.625*td)./(243.04+td))./exp((17.625*t(:,:,end))./(243.04+t(:,:,end))));
    sfcU=fliplr(ncread(rightFiles(ii,:),uName));
    sfcV=fliplr(ncread(rightFiles(ii,:),vName));
    
    pSurf=cat(3,pSurf,pS);
    tSurf=cat(3,tSurf,tS);
    rhSurf=cat(3,rhSurf,rhS);
    uSurf=cat(3,uSurf,sfcU);
    vSurf=cat(3,vSurf,sfcV);
    
    %% SST
    if SSTyes
        sstName=varNames2{find(contains(varNames2,'_'))};
        
        sst=fliplr(ncread(rightFiles(ii,:),sstName)-273.15);
        sstSurf=cat(3,sstSurf,sst);
    end
end

ecmwfData.lat=latRean;
ecmwfData.lon=lonRean;
ecmwfData.Temperature=tHours;
ecmwfData.rh=rhHours;
ecmwfData.z=zHours;
ecmwfData.p=pHours;
ecmwfData.pSurf=pSurf;
ecmwfData.tSurf=tSurf;
ecmwfData.rhSurf=rhSurf;
ecmwfData.uSurf=uSurf;
ecmwfData.vSurf=vSurf;
ecmwfData.time=inhours;
if SSTyes
    ecmwfData.sstSurf=sstSurf;
end

end

