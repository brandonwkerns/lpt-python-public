import matplotlib; matplotlib.use('agg')
import numpy as np
from context import lpt
import matplotlib.pylab as plt
import datetime as dt
import sys
import os
import matplotlib.colors as colors
import scipy.ndimage
from lpt_generic_netcdf_data_functions import *

################################################################################
##
##       +++ Preserve the MASTER script (for Git)   +++
##       +++ make local copies) and edit/run those. +++
##
## This driver script is for any generic NetCDF data in the proper format
## Using the data for the two times specified on command line.
## The file name convention of the NetCDF data is: gridded_rain_rates_YYYYMMDDHH.nc
## (or, modify dataset['file_name_format'] to change this.)
## The NetCDF must be in the following format template to work with this script:
##
## netcdf gridded_rain_rates_YYYYMMDDHH {
## dimensions:
## 	  lon = NNN ;
## 	  lat = MMM ;
## 	  time = UNLIMITED ; // (1 currently)
## variables:
## 	  double lon(lon) ;
## 	  double lat(lat) ;
## 	  double time(time) ;
## 	   	  time:units = "hours since 1970-1-1 0:0:0" ;
## 	  double rain(time, lat, lon) ;
## 		  rain:units = "mm h-1" ;
##
## Where NNN and MMM are the longitude and latitude array sizes.
##
##       +++ If lon is -180 to 180, it will be converted to 0 - 360. +++
##
################################################################################

"""
Dataset Case Settings
"""
dataset={}
#dataset['label'] = 's2s'
#dataset['raw_data_parent_dir'] = '/home/orca/bkerns/lib/lpt/lpt-matlab/data/s2s_ctl_init2013122500/interim/gridded_rain_rates'
dataset['label'] = 's2s_combine_37'
dataset['raw_data_parent_dir'] = '/home/disk/orca/zyw3/research/lpt-matlab-bkern-20181129-025/lpt-matlab/data/s2s_combine_37/interim/gridded_rain_rates/'
dataset['file_name_format'] = 'gridded_rain_rates_%Y%m%d%H.nc'
dataset['data_time_interval'] = 6           # Time resolution of the data in hours.
dataset['verbose'] = True

"""
Main settings for lpt
"""
## Plot settings.
plotting = {}
plotting['do_plotting'] = True               # True or False -- Should I make plots?
plotting['plot_area'] = [0, 360, -50, 50]    # Plotting area for LPO maps. (Does NOT affect tracking)
plotting['time_lon_range'] = [40, 200]       # Longitude Range for time-longitude plots. (Does NOT affect tracking)

## High level output directories. Images and data will go in here.
output={}
output['img_dir'] = '/home/orca/bkerns/public_html/realtime_mjo_tracking/lpt/images'
output['data_dir'] = '/home/orca/bkerns/public_html/realtime_mjo_tracking/lpt/data'
output['sub_directory_format'] = '%Y/%m/%Y%m%d'

##
## LP Object settings
##
lpo_options={}
lpo_options['do_lpo_calc'] = True
#lpo_options['do_lpo_calc'] = False
lpo_options['thresh'] = 12.0                # LP Objects threshold (mm/day)
lpo_options['accumulation_hours'] = 72      # Accumulation period for LP objects (hours).
lpo_options['filter_stdev'] = 20             # Gaussian filter stdev, in terms of grid points.
lpo_options['filter_n_stdev_width'] = 3     # Gaussian filter width, how many stdevs to go out?
lpo_options['min_points'] = 400             # Throw away LP objects smaller than this.
# If COLD_START_MODE is specified, assume there is no rain data before time zero.
#   Calculate the accumulation as follows:
#   For the first COLD_START_CONST_PERIOD, use the average rain rate during
#    the period, scaled to a 3 day accumulation, for the period.
#   After the COLD_START_CONST_PERIOD, use the accumulation up to that point,
#   scaled to a 3 day accumulation.
#
# !!! If COLD_START_MODE is set to False, there should be gridded rain files for
# !!! the ACCUMULATION_PERIOD time prior to the initial time.  !!!
lpo_options['cold_start_mode'] = False
lpo_options['cold_start_const_period'] = 24.0  # hours

##
## LPT Settings
##
lpt_options={}
#lpt_options['do_lpt_calc'] = False
lpt_options['do_lpt_calc'] = True
lpt_options['min_overlap_points'] = 1600      # LP object connectivity is based on either points
lpt_options['min_overlap_frac'] = 0.5         # -- OR fraction of either LP object.
lpt_options['min_lp_objects_points'] = 400    # Disregard LP objects smaller than this.
lpt_options['min_lpt_duration_hours'] = 7*24  # Minumum duration to keep it as an LPT (hours)
lpt_options['center_jump_max_hours'] = 3*24   # How long to allow center jumps (hours)

## Merging/Splitting settings
merge_split_options={}
merge_split_options['allow_merge_split'] = True
#merge_split_options['allow_merge_split'] = False
merge_split_options['split_merger_min_hours'] = 72     # Min duration of a split/merging track to separate it.

"""
Call the driver function.
"""

lpt_driver(dataset,plotting,output,lpo_options,lpt_options, merge_split_options, sys.argv)
