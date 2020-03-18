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



##
## Update the animations.
##
# echo Updating Animations.
# cd /home/orca/bkerns/public_html/realtime_mjo_tracking/lpt/images
# rm -f *.png *.gif

# ln -s `find cfs/objects/*/*/* | grep .png | tail -168 ` .
# /usr/bin/convert -delay 15 *.png lp_objects_cfs_FCST45DAYS.gif
# rm *.png
#
# ln -s `find cmorph/objects/*/*/* \( -name "*00.png" -or -name "*06.png" -or -name "*12.png" -or -name "*18.png" \)  | tail -241` .
# /usr/bin/convert -delay 15 *.png lp_objects_cmorph_rt_LAST45DAYS.gif
# rm *.png
#
# ln -s `find tmpa/objects/*/*/* \( -name "*.png" \)  | tail -481` .
# /usr/bin/convert -delay 15 *.png lp_objects_tmpa_rt_LAST45DAYS.gif
# rm *.png
#
#
# ## Make pause at the beginning and end of animation repeats.
# for ff in *.gif
# do
#   convert $ff \( +clone -set delay 100 \) +swap +delete $ff
# done
#
# ## Get latest time-longitude plots.
# ln -sf `find tmpa/systems/*/*/* | grep .png | tail -1 ` lpt_time_lon_tmpa_LATEST.png
# ln -sf `find cmorph/systems/*/*/* | grep .png | tail -1 ` lpt_time_lon_cmorph_LATEST.png
# ln -sf `find cfs/systems/*/*/* | grep .png | tail -1 ` lpt_time_lon_cfs_LATEST.png
#

echo Done.

exit 0
