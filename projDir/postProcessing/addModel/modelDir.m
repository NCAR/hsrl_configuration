function [rawDir interp] = modelDir(project,model,freq)
% Find model directory
if strcmp(project,'socrates')
            baseDir='/scr/snow2/rsfdata/projects/socrates/';
elseif strcmp(project,'cset')
    baseDir='/scr/snow2/rsfdata/projects/cset/';
elseif strcmp(project,'otrec')
 baseDir='/scr/snow1/rsfdata/projects/otrec/';
end

rawDir=[baseDir,'model/',model,'/'];
interp=[baseDir,'model/',model,'interp/',freq,'/'];
end