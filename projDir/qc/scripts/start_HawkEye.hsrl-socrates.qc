#! /bin/csh

rehash

setenv PROJ_DIR /opt/local/git/hsrl_configuration/projDir
cd $PROJ_DIR/qc/params

if ($#argv < 6) then
  /opt/local/lrose/bin/HawkEye -params HawkEye.hsrl-socrates.qc -start_time "2018 01 16 00 20 00" >& /dev/null &
else
  /opt/local/lrose/bin/HawkEye -params HawkEye.hsrl-socrates.qc -start_time "$1 $2 $3 $4 $5 $6" >& /dev/null &
endif

