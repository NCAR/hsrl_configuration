function [topoOut topolon topolat] = read_gtopo30(indir,inlon,inlat)
% Find right gtopo30 files and read them

topoOut=[];
topolon=[];
topolat=[];

latlim=double([min(inlat),max(inlat)]);
lonlim=double([min(wrapTo180(inlon)),max(wrapTo180(inlon))]);

[topoOut,refvec] = gtopo30(indir,1,latlim,lonlim);
topoOut(isnan(topoOut))=0;

topolonV=refvec(3):1/120:refvec(3)+size(topoOut,2)/120;
topolonV=topolonV(1:end-1);

topolatV=refvec(2)-size(topoOut,1)/120:1/120:refvec(2);
topolatV=topolatV(1:end-1);

topolon=wrapTo360(repmat(topolonV,size(topoOut,1),1));
topolat=repmat(topolatV',1,size(topoOut,2));
end

