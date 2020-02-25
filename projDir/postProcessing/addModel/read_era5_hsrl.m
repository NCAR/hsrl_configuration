function era5data = read_era5_hsrl(indir,startTime,endTime,SSTyes)
% Find the right time span for era5 data and read them in

refTime=datetime(1900,1,1,0,0,0);

%Initialize output
era5data=[];

startHour=datetime(year(startTime),month(startTime),day(startTime),hour(startTime),0,0);
endHour=datetime(year(endTime),month(endTime),day(endTime),hour(endTime)+1,0,0);

inhours=startHour:hours(1):endHour;

pHours=[];
tHours=[];
zHours=[];
pSurf=[];
tSurf=[];
zSurf=[];

for kk=1:length(inhours) %Loop through all hours
    %% Pressure level data
    % Find ecmwf files
    roundTime=inhours(kk);
    dayStr=datestr(roundTime,'yyyymmdd');
    
    tFiles=dir([indir,'T.*',dayStr,'00_',dayStr,'23.nc']);
    zFiles=dir([indir,'Z.*',dayStr,'00_',dayStr,'23.nc']);
            
    if size(zFiles,1)==0 | size(tFiles,1)==0
        disp('No model data found.');
        return
    end
    
    %read in time, lat and lon data
    timeRean=ncread([tFiles(1).folder,'/',tFiles(1).name],'time');
    timeActual=refTime+hours(timeRean);
    timeInd=find(timeActual==roundTime);
    
    lonRean=ncread([tFiles(1).folder,'/',tFiles(1).name],'longitude');
    latRean=ncread([tFiles(1).folder,'/',tFiles(1).name],'latitude');
    
    t=nan(length(lonRean),length(latRean),length(tFiles));
    z=nan(length(lonRean),length(latRean),length(tFiles));
    p=nan(length(lonRean),length(latRean),length(tFiles));
    
    for jj=1:size(tFiles,1)
        p(:,:,jj)=fliplr(ncread([tFiles(jj).folder,'/',tFiles(jj).name],'level'));
        t(:,:,jj)=fliplr(squeeze(ncread([tFiles(jj).folder,'/',tFiles(jj).name],'T',[1,1,1,timeInd],[inf,inf,inf,1]))-273.15);
        z(:,:,jj)=fliplr(squeeze(ncread([zFiles(jj).folder,'/',zFiles(jj).name],'Z',[1,1,1,timeInd],[inf,inf,inf,1])));
    end
    
    z=z./9.806;
    
    pHours=cat(4,pHours,p);
    tHours=cat(4,tHours,t);
    zHours=cat(4,zHours,z);
    
    %% Surface data
    % Find ecmwf files
    monthStr=datestr(roundTime,'yyyymm');
    
    tsFiles=dir([indir,'VAR_2T.*',monthStr,'0100_*.nc']);
    pFiles=dir([indir,'SP.*',monthStr,'0100_*.nc']);
    
    info=ncinfo([tsFiles.folder,'/',tsFiles.name]);
    
    timeReanS=ncread([tsFiles.folder,'/',tsFiles.name],'time');
    timeActualS=refTime+hours(timeReanS);
    timeIndS=find(timeActualS==roundTime);
    pS=fliplr(ncread([pFiles.folder,'/',pFiles.name],'SP',[1,1,timeIndS],[inf,inf,1])./100);
    tS=fliplr(ncread([tsFiles.folder,'/',tsFiles.name],'VAR_2T',[1,1,timeIndS],[inf,inf,1])-273.15);
    
    pSurf=cat(3,pSurf,pS);
    tSurf=cat(3,tSurf,tS);

end
era5data.lat=latRean;
era5data.lon=lonRean;
era5data.Temperature=tHours;
era5data.z=zHours;
era5data.p=pHours;
era5data.pSurf=pSurf;
era5data.tSurf=tSurf;
era5data.time=inhours;
end

