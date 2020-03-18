#!/bin/bash

CMORPH_DAYS=45
CFS_FCST_DAYS=45
WORKDIR=/path/to/this/realtime/script/directory/
ANACONDA_DIR=/path/to/anaconda3
################################################################################
####### End basic edits ########################################################
################################################################################


## Give input as YYYYMMDD, or it will get today's date using the Linux date command.
if [ -z $1 ]
then
  today=`/bin/date -u +%Y%m%d`
else
  today=$1
fi
echo Updating LPT real time for ${today}.

yyyy=`/bin/date --date=$today +%Y`
mm=`/bin/date --date=$today +%m`
hh=00


## Get in to script directory.
cd $WORKDIR
mkdir -p logs

## Activate the Anaconda Python module with all the dependencies.
source $ANACONDA_DIR/bin/activate meteo

## Call the Python driver scripts.
echo CFS Forecast
YMDH1=${today}$hh
YMDH2=`/bin/date --date="$today + $CFS_FCST_DAYS days" +%Y%m%d`$hh
python $WORKDIR/lpt_run_cfs_fcst.py $YMDH1 $YMDH2 >& $WORKDIR/logs/log.rt.cfs.${YMDH1}_${YMDH2}

echo CMORPH
YMDH2=${today}$hh
YMDH1=`/bin/date --date="$today - $CMORPH_DAYS days" +%Y%m%d`$hh
python $WORKDIR/lpt_run_cmorph.py $YMDH1 $YMDH2 >& $WORKDIR/logs/log.rt.cmorph.${YMDH1}_${YMDH2}



echo Done.

exit 0
