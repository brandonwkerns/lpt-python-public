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
dataset['label'] = 'cmorph'
dataset['raw_data_parent_dir'] = '/path/to/cmorph/rt/'
dataset['raw_data_format'] = 'cmorph'
dataset['file_name_format'] = '%Y/%m/%Y%m%d/CMORPH_V0.x_RT_8km-30min_%Y%m%d%H'
dataset['data_time_interval'] = 1           # Time resolution of the data in hours.
dataset['verbose'] = True
dataset['field_units'] = 'mm h-1'
dataset['area'] = [40, 240, -50, 50]   # Geographical area of data to use.

## High level output directories. Images and data will go in here.
output['img_dir'] = '/path/to/this/realtime/script/directory/images'
output['data_dir'] = '/path/to/this/realtime/script/directory/data'
output['sub_directory_format'] = '%Y/%m/%Y%m%d'  # This applies to LP objects files only.


"""
Which steps to do?
"""
# Plotting
plotting['do_plotting'] = True               # True or False -- Should I make plots?

# LPO
lpo_options['do_lpo_calc'] = True
lpo_options['overwrite_existing_files'] = False

# LPT
lpt_options['do_lpt_calc'] = True

# MJO ID
mjo_id_options['do_mjo_id'] = True

##----------------------------
## LPT Mask Settings.
##----------------------------
# The mask steps can take a long time and computational resources.
# It may be good to limit them for real time purposes.
# These steps run much faster with:
# 1) lpt_options['mask_calc_with_filter_radius']  = False (No "mask with filter" variables.)
# 2) lpt_options['mask_calc_volrain']  = False (No volumetric rain calculations.)

lpt_options['do_lpt_individual_masks']       = True       # Whether to generate mask files for each LPT system.
lpt_options['individual_masks_mjo_only']     = True       # If True, then only do MJO LPTs only.

lpt_options['do_lpt_composite_mask']         = False      # Whether to generate mask file for all LPT systems combined.
lpt_options['do_mjo_lpt_composite_mask']     = True       # Whether to generate mask file for MJO LPT systems combined.
lpt_options['do_non_mjo_lpt_composite_mask'] = False      # Whether to generate mask file for all non MJO LPT systems combined.

lpt_options['mask_calc_volrain']             = True       # Whether to calculate a volumetric rain and include with mask files.
lpt_options['mask_calc_with_filter_radius']  = False       # Whether to calculate the mask with filter variables. (Takes much longer to run)
lpt_options['mask_coarse_grid_factor']       = 7          # If > 0, it will use a coarsened grid to calculate masks. Good for high res data.
lpt_options['mask_calc_with_accumulation_period'] = True  # Whether to calculate the mask with filter variables. (Takes much longer to run)
lpt_options['target_memory_for_writing_masks_MB'] = 1000   # Target to limit memory demand from writing masks to files. The more, the faster it can run.

################################################################################


"""
Main settings for lpt
"""
## Plot settings.
plotting['plot_area'] = [40, 240, -50, 50]   # Plotting area for maps.
plotting['time_lon_range'] = [40, 240]       # Longitude Range for time-longitude plots. (Does NOT affect tracking)


##
## LP Object settings
##
lpo_options['thresh'] = 12.0                 # LP Objects threshold
lpo_options['accumulation_hours'] = 72       # Accumulation period for LP objects.
lpo_options['filter_stdev'] = 70             # Gaussian filter width, in terms of grid points.
lpo_options['min_points'] = 4800

## LPO Mask Settings.
lpo_options['do_lpo_mask'] = False                     # Whether to generate LPO mask file. Does not require lpo_options['do_lpo_calc'] = True
lpo_options['mask_coarse_grid_factor'] = 7                 # If > 0, it will use a coarsened grid to calculate masks. Good for high res data.
lpo_options['mask_calc_volrain'] = True                   # Whether to calculate a volumetric rain and include with mask files.
lpo_options['mask_calc_with_filter_radius'] = True        # Whether to calculate the mask with filter variables. (Takes much longer to run)
lpo_options['mask_calc_with_accumulation_period'] = True  # Whether to calculate the mask with filter variables. (Takes much longer to run)
lpo_options['target_memory_for_writing_masks_MB'] = 1000  # Target to limit memory demand from writing masks to files. The more, the faster it can run.

##
## LPT Settings
##
lpt_options['min_overlap_points'] = 19000              # LP object connectivity is based on either points
lpt_options['min_overlap_frac'] = 0.5                  # -- OR fraction of either LP object.
lpt_options['min_lp_objects_points'] = 4800            # Disregard LP objects smaller than this.

##
## MJO LPT Identification settings
##
# Using defaults.

##
## LPT Mask Settings.
##


##
## Merging/Splitting settings
##
# Using defaults.

"""
Call the driver function.
"""

lpt_driver(dataset,plotting,output,lpo_options,lpt_options, merge_split_options, mjo_id_options, sys.argv)
