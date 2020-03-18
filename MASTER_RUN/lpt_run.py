import matplotlib; matplotlib.use('agg')
import numpy as np
from context import lpt
import matplotlib.pylab as plt
import datetime as dt
import sys
import os
import matplotlib.colors as colors
import scipy.ndimage
from lpt.lpt_generic_netcdf_data_functions import *
from lpt.default_options import *
import warnings; warnings.simplefilter('ignore')

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
dataset['raw_data_format'] = 'generic_netcdf'
dataset['file_name_format'] = 'gridded_rain_rates_%Y%m%d%H.nc'
dataset['data_time_interval'] = 6           # Time resolution of the data in hours.
dataset['verbose'] = True
dataset['longitude_variable_name'] = 'lon'
dataset['latitude_variable_name'] = 'lat'
dataset['field_variable_name'] = 'rain'
dataset['field_units'] = 'mm h-1'

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
output['img_dir'] = './images'
output['data_dir'] = './data/processed'
output['sub_directory_format'] = '%Y/%m/%Y%m%d'

##
## LP Object settings
##
lpo_options={}
lpo_options['do_lpo_calc'] = True
#lpo_options['do_lpo_calc'] = False
lpo_options['multiply_factor'] = 24.0       # e.g., 24.0 for mm/h to mm/day.
lpo_options['field_units'] = 'mm d-1'
lpo_options['thresh'] = 12.0                # LP Objects threshold (in units above)
lpo_options['accumulation_hours'] = 72      # Accumulation period for LP objects (hours).
lpo_options['filter_stdev'] = 20            # Gaussian filter stdev, in terms of grid points.
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
lpo_options['overwrite_existing_files'] = False

## LPO Mask Settings.
lpo_options['do_lpo_mask'] = True                         # Whether to generate LPO mask file. Does not require lpo_options['do_lpo_calc'] = True
lpo_options['mask_calc_volrain'] = True                   # Whether to calculate a volumetric rain and include with mask files.
lpo_options['mask_calc_with_filter_radius'] = True        # Whether to calculate the mask with filter variables. (Takes much longer to run)
lpo_options['mask_calc_with_accumulation_period'] = True  # Whether to calculate the mask with filter variables. (Takes much longer to run)
lpo_options['target_memory_for_writing_masks_MB'] = 1000  # Target to limit memory demand from writing masks to files. The more, the faster it can run.

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
lpt_options['fall_below_threshold_max_hours'] = 3*24   # How long to allow center jumps (hours)

## LPT Mask Settings.
lpt_options['do_lpt_individual_masks'] = True # Whether to generate mask files for each LPT system.
lpt_options['individual_masks_begin_lptid'] = 0 # LPT ID to start with
lpt_options['individual_masks_end_lptid'] = 20 # LPT ID to end with
lpt_options['do_lpt_composite_mask'] = True   # Whether to generate mask file for all LPT systems combined.
lpt_options['do_mjo_lpt_composite_mask'] = True   # Whether to generate mask file for all MJO LPT systems combined.
lpt_options['do_non_mjo_lpt_composite_mask'] = True   # Whether to generate mask file for all non MJO LPT systems combined.
lpt_options['mask_calc_volrain'] = True       # Whether to calculate a volumetric rain and include with mask files.
lpt_options['mask_calc_with_filter_radius'] = True        # Whether to calculate the mask with filter variables. (Takes much longer to run)
lpt_options['mask_calc_with_accumulation_period'] = True  # Whether to calculate the mask with filter variables. (Takes much longer to run)
lpt_options['target_memory_for_writing_masks_MB'] = 1000  # Target to limit memory demand from writing masks to files. The more, the faster it can run.

##
## Merging/Splitting settings
##
merge_split_options={}
merge_split_options['allow_merge_split'] = True
#merge_split_options['allow_merge_split'] = False
merge_split_options['split_merger_min_hours'] = 72     # Min duration of a split/merging track to separate it.

##
## MJO LPT Identification settings
##
mjo_id_options = {}
mjo_id_options['do_mjo_id'] = True
mjo_id_options['do_plotting'] = True
mjo_id_options['min_zonal_speed'] = -999.0   # full LPT track net speed, in m/s.
mjo_id_options['min_lpt_duration']    = 7.0*24.0      # In hours. Does NOT include accumulation period.
mjo_id_options['min_eastward_prop_zonal_speed'] = 0.0  # Eastward propagation portion, in m/s.
mjo_id_options['min_eastward_prop_duration'] = 7.0*24.0  # In hours. Doesn't include 3-Day accumulation period.
mjo_id_options['min_eastward_prop_duration_in_lat_band'] = 7.0*24.0  # In hours. Doesn't include 3-Day accumulation period.
mjo_id_options['min_total_eastward_lon_propagation'] = 10.0 # in deg. longitude.
mjo_id_options['max_abs_latitude'] = 15.0 # in deg. latitude. Eastward propagation period must get this close to the Equator at some point.
## These settings are for the eaat/west propagation portions ("divide and conquer").
mjo_id_options['duration_to_avoid_being_conquered'] = 7.0*24.0  # In hours.
mjo_id_options['lon_prop_to_avoid_being_conquered'] = 20.0      # In degrees longitude
mjo_id_options['backtrack_allowance'] = 5.0                     # In degrees longitude

"""
Call the driver function.
"""

lpt_driver(dataset,plotting,output,lpo_options,lpt_options, merge_split_options, mjo_id_options, sys.argv)
