import matplotlib; matplotlib.use('agg')
import numpy as np
from context import lpt
import matplotlib.pylab as plt
import datetime as dt
import sys
import os
import matplotlib.colors as colors
import scipy.ndimage
from lpt.lpt_driver import *
from lpt.default_options import *
import warnings; warnings.simplefilter('ignore')

"""
################################################################################
################################################################################
################################################################################

      +++ Preserve the MASTER run directory (for Git)   +++
      +++ make local copies) and edit/run the files in the copy. +++

Usage: python lpt_run.py YYYYMMDDHH YYYYMMDDHH
(Specify the starting and ending dates and times on command line as YYYYMMDDHH)

This driver script serves as the "namelist" for running LPT.
You can manually change the settings in each section, and the script will
pass these settings to the main driver function lpt/lpt_driver.py

This driver script is by default for any generic NetCDF data in the proper format
Using the data for the two times specified on command line.
The file name convention of the NetCDF data is: gridded_rain_rates_YYYYMMDDHH.nc
(or, modify dataset['file_name_format'] to change this.)
The NetCDF must be in the following format template to work with this script:

netcdf gridded_rain_rates_YYYYMMDDHH {
dimensions:
	  lon = NNN ;
	  lat = MMM ;
	  time = UNLIMITED ; // (1 currently)
variables:
	  double lon(lon) ;
	  double lat(lat) ;
	  double time(time) ;
	   	  time:units = "hours since 1970-1-1 0:0:0" ;
	  double rain(time, lat, lon) ;
		  rain:units = "mm h-1" ;

Where NNN and MMM are the longitude and latitude array sizes.

      +++ If lon is -180 to 180, it will be converted to 0 - 360. +++

Any dataset can be used if the files are first converted to NetCDF.
However, some other data formats are supported natively:
- CMORPH binary data.
- CFS forecasts, GRIB data.
Additional datasets can be implemented natively by adding functions to lpt/readdata.py.
################################################################################
################################################################################
################################################################################
"""

"""
Dataset Case Settings
"""
dataset['label'] = 'trmm'
dataset['raw_data_parent_dir'] = '/gridded/netcdf/data/dir'
dataset['raw_data_format'] = 'generic_netcdf'
dataset['file_name_format'] = 'gridded_rain_rates_%Y%m%d%H.nc'
dataset['data_time_interval'] = 3           # Time resolution of the data in hours.
dataset['verbose'] = True
dataset['longitude_variable_name'] = 'lon'
dataset['latitude_variable_name'] = 'lat'
dataset['field_variable_name'] = 'rain'
dataset['field_units'] = 'mm h-1' ## This is used only for plotting and NetCDF attributes.
"""
Set calendar below. Use a valid cftime calendar.
Valid calendars are currently:
  standard, gregorian, proleptic_gregorian, noleap,
  365_day, 360_day, julian, all_leap, 366_day.
"""
dataset['calendar'] = 'standard'

"""
Main settings for lpt
"""
## Plot settings.
plotting['do_plotting'] = False               # True or False -- Should I make plots?
plotting['plot_area'] = [0, 360, -50, 50]     # Plotting area for LPO maps. (Does NOT affect tracking)
plotting['time_lon_range'] = [40, 200]        # Longitude Range for time-longitude plots. (Does NOT affect tracking)

## High level output directories. Images and data will go in here.
output['img_dir'] = './images'
output['data_dir'] = './data'
output['sub_directory_format'] = '%Y/%m/%Y%m%d'

"""
LPT Steps.

Specify which LPT step(s) should be carried out when you call the script.
Feel free to run the steps one at a time with multiple Python calls,
or turn multiple ones on to do them in one go if you're confident.
The steps are numbered in prerequisite order, e.g., step 2 requires the output
from step 1. The decimal steps are optional, e.g., LPO mask 1.1 depends on LPO 1.0.
"""

"""
################################################################################
## 1.0 LP Objects
## Feature Identification
################################################################################
"""
## Whether to run the LPO step.

#lpo_options['do_lpo_calc'] = True
lpo_options['do_lpo_calc'] = False


## Options for the LPO Step. Only used if lpo_options['do_lpo_calc'] = True
lpo_options['multiply_factor'] = 24.0       # e.g., 24.0 for mm/h to mm/day, 1.0 if you already have mm/day.
lpo_options['field_units'] = 'mm d-1'       # This is used for plotting and NetCDF output.
lpo_options['thresh'] = 12.0                # LP Objects threshold (in units above)
lpo_options['accumulation_hours'] = 72      # Accumulation period for LP objects (hours).
lpo_options['filter_stdev'] = 20            # Gaussian filter stdev, in terms of grid points.
lpo_options['filter_n_stdev_width'] = 3     # Gaussian filter width, how many stdevs to go out?
lpo_options['min_points'] = 400             # Throw away LP objects smaller than this.
# "Cold start mode"
#
# Useful for model runs.
#
# If lpo_options['cold_start_mode'] is True:
#   Assume there is no rain data before time zero.
#   Calculate the accumulation as follows:
#   For the first lpo_options['cold_start_const_period'] hours,
#   use the average rain rate during this initial period,
#   for the entire period. E.g., for all times within the first 24 hours, use the first 24 h mean.
#   After lpo_options['cold_start_const_period'] hours, use the accumulation up to that point,
#   e.g., for 36 and 48 h, use the first 36 and 48 h mean, respectively.
#
# !!! If lpo_options['cold_start_mode'] = False, there MUST be gridded rain files for  !!!
# !!! the entire lpo_options['accumulation_hours'] hours prior to the initial time.    !!!
lpo_options['cold_start_mode'] = False
lpo_options['cold_start_const_period'] = 24.0  # hours


"""
################################################################################
## 1.1 LPO Mask output.
## Mask output is in the form of a 3-D array (time, lat, lon) of 0s and 1s,
## where 1s are within an LPO.
################################################################################
"""

## Whether to generate the LPO mask output.

#lpo_options['do_lpo_mask'] = True
lpo_options['do_lpo_mask'] = False


## Options for the LPO mask output.
lpo_options['mask_calc_volrain'] = True                    # Whether to calculate a volumetric rain and include with mask files.
lpo_options['mask_calc_with_filter_radius'] = True         # Whether to calculate the mask with filter variables. (See coarse grid factor option if this takes too long to run.)
lpo_options['mask_calc_with_accumulation_period'] = True   # Whether to calculate the mask with filter variables.
lpo_options['mask_coarse_grid_factor'] = 0                 # If > 0, it will use a coarsened grid to calculate masks. Good for high res data.
lpo_options['target_memory_for_writing_masks_MB'] = 1000   # Target to limit memory demand from writing masks to files. The more, the faster it can run.
lpo_options['mask_n_cores'] = 1                            # How many processors to use for LPO mask calculations.


"""
################################################################################
## 2.0 LPT Step
## Tracking in time.
################################################################################
"""

## Whether to run the LPT step.

#lpt_options['do_lpt_calc'] = True
lpt_options['do_lpt_calc'] = False

## Options for the LPT step.
lpt_options['min_overlap_points'] = 1600      # LP object connectivity is based on either points
lpt_options['min_overlap_frac'] = 0.5         # -- OR fraction of either LP object.
lpt_options['min_lp_objects_points'] = 400    # Disregard LP objects smaller than this.
lpt_options['min_lpt_duration_hours'] = 7*24  # Minumum duration to keep it as an LPT (hours)
lpt_options['fall_below_threshold_max_hours'] = 3*24   # How long to allow center jumps (hours)

## Merging/Splitting settings for the LPT step.
merge_split_options['allow_merge_split'] = True
merge_split_options['split_merger_min_hours'] = 72  # Min duration of a split/merging track to separate it.

"""
################################################################################
## 3.0 MJO LPT Identification
## Identify which LPT Systems are MJO LPT Systems
################################################################################
"""

## Whether to do the MJO LPT identification step.

#mjo_id_options['do_mjo_id'] = True
mjo_id_options['do_mjo_id'] = False

## Settings for the MJO LPT identification step.

mjo_id_options['do_plotting'] = False                     # Whether to generate a set of diagnostic plots
mjo_id_options['min_zonal_speed'] = -999.0                # full LPT track net speed, in m/s.
mjo_id_options['min_lpt_duration']    = 7.0*24.0          # In hours. Does NOT include accumulation period.
mjo_id_options['min_eastward_prop_zonal_speed'] = 0.0     # Eastward propagation portion, in m/s.
mjo_id_options['min_eastward_prop_duration'] = 7.0*24.0                # In hours. Doesn't include 3-Day accumulation period.
mjo_id_options['min_eastward_prop_duration_in_lat_band'] = 7.0*24.0    # In hours. Doesn't include 3-Day accumulation period.
mjo_id_options['min_total_eastward_lon_propagation'] = 10.0            # in deg. longitude.
mjo_id_options['max_abs_latitude'] = 15.0                              # in deg. latitude. Eastward propagation period must get this close to the Equator at some point.

## These settings are for the east/west propagation portions ("divide and conquer").
mjo_id_options['duration_to_avoid_being_conquered'] = 7.0*24.0  # In hours.
mjo_id_options['lon_prop_to_avoid_being_conquered'] = 20.0      # In degrees longitude
mjo_id_options['backtrack_allowance'] = 5.0                     # In degrees longitude


"""
################################################################################
## 2.1 LPT mask output -- Individual LPTs and composite of all LPTs
## 3.1 MJO LPT specific mask output
## Mask output is in the form of a 3-D array (time, lat, lon) of 0s and 1s,
## where 1s are within an LPT System.
## This can be run for:
##    - Separate files for each individual LPT System (2.1, requires LPO and LPT steps)
##    - Composite file with all of the LPT systems (2.1, requires LPO and LPT steps)
##    - Composite file with only the MJO LPT systems (3.1, also requires MJO Identification step)
################################################################################
"""

## Whether to generate the LPO systems mask output, and which ones.
lpt_options['do_lpt_individual_masks'] = False          # Whether to generate mask files for each LPT system.
lpt_options['do_lpt_composite_mask'] = False            # Whether to generate mask file for all LPT systems combined.
lpt_options['do_mjo_lpt_composite_mask'] = False        # Whether to generate mask file for all MJO LPT systems combined.
lpt_options['do_non_mjo_lpt_composite_mask'] = False    # Whether to generate mask file for all non MJO LPT systems combined.

## Options for the LPT systems mask output. Only used if one or more of the above "do" masks options are True
lpt_options['individual_masks_begin_lptid'] = 0           # LPT ID to start with
lpt_options['individual_masks_end_lptid'] = 1             # LPT ID to end with
lpt_options['mask_calc_volrain'] = True                   # Whether to calculate a volumetric rain and include with mask files.
lpt_options['mask_calc_with_filter_radius'] = True        # Whether to calculate the mask with filter variables. (Takes much longer to run)
lpt_options['mask_calc_with_accumulation_period'] = True  # Whether to calculate the mask with filter variables. (Takes much longer to run)
lpt_options['target_memory_for_writing_masks_MB'] = 1000  # Target to limit memory demand from writing masks to files. The more, the faster it can run.
lpt_options['mask_n_cores'] = 1                           # How many processors to use for LPT system mask calculations.


"""
Call the driver function.
"""

lpt_driver(dataset,plotting,output,lpo_options,lpt_options, merge_split_options, mjo_id_options, sys.argv)
