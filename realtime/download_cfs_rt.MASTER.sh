#!/bin/bash

#######  Download real time version of CMORPH.

ftpdir=https://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs

## parent directory directory where data will be downloaded.
download_parent_dir=/path/to/keep/your/data

ENS=01  #Ensemble number. 01 is control. 02, 03, and 04 are perturbed.

#######################################################################

today=`date -u +%Y%m%d`

for days_back in {0..6}
do

  ymd=`date --date=${today}-${days_back}day  +%Y%m%d`
  yyyy=`date --date=${today}-${days_back}day +%Y`
  mm=`date --date=${today}-${days_back}day   +%m`

  HHinit=00

  filewanted=$ftpdir/cfs.$ymd/$HHinit/time_grib_$ENS/prate.$ENS.$ymd$HHinit.daily.grb2
  echo $filewanted
  /usr/bin/wget -q -nc -x -nH --cut-dirs=7 -P $download_parent_dir $filewanted
  if [ $? -eq 0 ]
  then
    echo Success!
  else
    echo Failed! File may not be on the server yet.
  fi

done


exit 0
