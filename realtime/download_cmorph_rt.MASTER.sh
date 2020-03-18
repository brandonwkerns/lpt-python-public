#!/bin/bash

#######  Download real time version of CMORPH.

ftpsite=ftp://ftp.cpc.ncep.noaa.gov/precip/CMORPH_RT/GLOBE/data

## Working directory is the root directory where data will be downloaded.
workdir=/path/to/keep/your/data

#######################################################################


cd $workdir

## Give input as YYYYMMDD, or it will get today's date using the Linux date command.
if [ -z $1 ]
then
  today=`/bin/date -u +%Y%m%d`
else
  today=$1
fi

yyyy=`/bin/date --date=$today +%Y`
mm=`/bin/date --date=$today +%m`


for hh in {00..23}
do

  filewanted=CMORPH_V0.x_RT_8km-30min_$today$hh

  if [ -e rt/$yyyy/$mm/$today/$filewanted ]
  then
    echo I already have ${filewanted}.
  else
    echo Downloading ${filewanted}.

    /usr/bin/wget -q $ftpsite/$yyyy/$yyyy$mm/$filewanted.gz

    if [ -e $filewanted.gz ]
    then

      mkdir -p rt/$yyyy/$mm/$today
      mv $filewanted.gz rt/$yyyy/$mm/$today
      /bin/gunzip -f  rt/$yyyy/$mm/$today/$filewanted.gz
      echo Success!
    else
      echo Failed! File may not be on the server yet.
    fi
  fi

done


yesterday=`date --date=${today}-1day  +%Y%m%d`
yyyy=`date --date=${today}-1day +%Y`
mm=`date --date=${today}-1day +%m`


for hh in {00..23}
do

  filewanted=CMORPH_V0.x_RT_8km-30min_$yesterday$hh


  if [ -e rt/$yyyy/$mm/$yesterday/$filewanted ]
  then
    echo I already have ${filewanted}.
  else
    echo Downloading ${filewanted}.

    /usr/bin/wget -q $ftpsite/$yyyy/$yyyy$mm/$filewanted.gz

    if [ -e $filewanted.gz ]
    then

      mkdir -p rt/$yyyy/$mm/$yesterday
      mv $filewanted.gz rt/$yyyy/$mm/$yesterday
      /bin/gunzip -f  rt/$yyyy/$mm/$yesterday/$filewanted.gz
      echo Success!
    else
      echo Failed! File may not be on the server yet.
    fi

  fi

done

echo Done.
exit 0
