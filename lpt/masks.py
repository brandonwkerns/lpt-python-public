## This file contains the content for LPO, LPT, and LPT "grand master" masks as functions, to be used with lpt-python-public.

import numpy as np
import xarray as xr
import scipy.ndimage
from scipy.interpolate import RegularGridInterpolator
import contourpy
from shapely.geometry import LinearRing, Polygon

import matplotlib.pyplot as plt
from netCDF4 import Dataset
import datetime as dt
import cftime
from dateutil.relativedelta import relativedelta
from context import lpt
import os
import sys
from scipy.sparse import dok_matrix, csr_matrix, find, SparseEfficiencyWarning
from multiprocessing import Pool, RLock, freeze_support
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)



##
## Initialize mask function
##
def initialize_mask_arrays(ny, nx, nt, detailed_output,
                           calc_with_accumulation_period, accumulation_hours,
                           calc_with_filter_radius):
    """
    initialize_mask_arrays(ny, nx, nt, detailed_output,
                            calc_with_accumulation_period, accumulation_hours,
                            calc_with_filter_radius)

    Initialize sparse matrices (csr_matrix) for the spatio-temporal masks.

    Each mask variable is returned as a list of 2-D sparse matrices
    of size (ny, nx), in which each entry of the list corresponds
    with a time stamp. Number of time stamps is nt.

    Which mask arrays are created depends on the other parameters:
    - If detailed_output is False, only "mask" is created. "Mask" has values of
      0 for outside, 1 for inside the filter + accumulation zone, 2 for
      within the LPO (i.e., a "core region.")
    - If detailed_output is True, up to four mask variables are created:
      + "mask_at_end_time", which is just the LPO, is always created.
      + "mask_with_accumulation" if there is an accumulation/averaging
        period > 0 and calc_with_accumulation_period is True.
      + "mask_with_filter_at_end_time" if there is a filter radius > 0
        and if calc_with_filter_radius is True.
      + "mask_with_filter_and_accumulation" the above two conditions
        both apply.
    """

    mask_arrays_shape2d = (ny, nx)

    mask_arrays = {}

    if detailed_output:
        mask_arrays['mask_at_end_time'] = [csr_matrix(mask_arrays_shape2d, dtype=np.bool_) for x in range(nt)]
        if accumulation_hours > 0 and calc_with_accumulation_period:
            mask_arrays['mask_with_accumulation'] = [csr_matrix(mask_arrays_shape2d, dtype=np.bool_) for x in range(nt)]
        if calc_with_filter_radius:
            mask_arrays['mask_with_filter_at_end_time'] = [csr_matrix(mask_arrays_shape2d, dtype=np.bool_) for x in range(nt)]
            if accumulation_hours > 0 and calc_with_accumulation_period:
                mask_arrays['mask_with_filter_and_accumulation'] = [csr_matrix(mask_arrays_shape2d, dtype=np.bool_) for x in range(nt)]
    else:
        mask_arrays['mask'] = [csr_matrix(mask_arrays_shape2d, dtype=np.uint8) for x in range(nt)]

    return mask_arrays


##
## feature spread function -- used for all of the mask functions.
##


def feature_spread_2d(array_2d, npoints, spread_value=None):

    array_2d_new = array_2d.copy()

    if type(npoints) is list:
        npx = npoints[0]
        npy = npoints[1]
    else:
        npx = 1*npoints
        npy = 1*npoints

    [circle_array_x, circle_array_y] = np.meshgrid(np.arange(-1*npx,npx+1), np.arange(-1*npy,npy+1))
    circle_array_dist = np.sqrt(np.power(circle_array_x,2) + np.power(circle_array_y * (npx/npy),2))
    circle_array_mask = (circle_array_dist < (npx + 0.1)).astype(np.double)
    circle_array_mask = circle_array_mask / np.sum(circle_array_mask)

    ## Loop over the times.
    ## For each time, use the convolution to "spread out" the effect of each time's field.
    ## (I tried using 3-D convolution here, but it took almost twice as much memory
    ##  and was slightly SLOWER than this method.)
    unique_values = np.unique(array_2d)
    unique_values = unique_values[unique_values > 0]  #take out zero -- it is not a feature.
    for this_value in unique_values:
        starting_mask = (array_2d == this_value).astype(np.double)
        binary_dilation_mask = array_2d < 0.1 #starting_mask < 0.1
        starting_mask_spread = scipy.ndimage.binary_dilation(starting_mask,structure=circle_array_mask, iterations=1, mask=binary_dilation_mask)
        new_feature_mask = np.logical_and(starting_mask_spread > 0.001, binary_dilation_mask)
        if spread_value is None:
            array_2d_new[new_feature_mask] = this_value
        else:
            array_2d_new[new_feature_mask] = spread_value

    return array_2d_new


def feature_spread(data, npoints, spread_value=None, nproc=1):

    ## Use the binary dilation technique to expand the mask "array_in" a radius of np points.
    ## For this purpose, it takes a 3-D array with the first entry being time.

    with Pool(nproc) as p:
        r = p.starmap(
            feature_spread_2d,
            tqdm([(x.toarray(), npoints, spread_value) for x in data]),
            chunksize=1
            )

    data_new = [csr_matrix(x) for x in r]

    return data_new


def back_to_orig_res(array_2d_reduced, S_orig, reduce_res_factor):
    """
    array_2d_reduced: The coarsened grid data. 2-d Numpy array.
    S_orig: The size of the original grid.
    reduce_res_factor: The factor used for reducing the resolution.
    """

    S = array_2d_reduced.shape

    ## Array of Indices
    x = np.repeat(np.arange(S[1], dtype=np.int32), reduce_res_factor)
    y = np.repeat(np.arange(S[0], dtype=np.int32), reduce_res_factor)
    X, Y = np.meshgrid(x, y)

    ## Apply array of indices.
    array_2d_new0 = array_2d_reduced[Y, X]

    ## Keep only the size I need.
    array_2d_new = array_2d_new0[0:S_orig[0], 0:S_orig[1]]

    return array_2d_new


def feature_spread_reduce_res(data, npoints, spread_value=None, reduce_res_factor=5, nproc=1):
    ## Use the binary dilation technique to expand the mask "array_in" a radius of np points.
    ## For this purpose, it takes a 3-D array with the first entry being time.
    ##
    ## In this version of the feature_spread function,
    ## use a reduced resolution grid then interp back to the original resolution.

    print('Feature spread with reduce_res_factor = {}'.format(reduce_res_factor))

    start_idx = max(0, int(reduce_res_factor/2)-1) # Try to get near the middle of reduce_res_factor

    with Pool(nproc) as p:
        r = p.starmap(
            feature_spread_2d,
            tqdm(
                [(x.toarray()[start_idx::reduce_res_factor,start_idx::reduce_res_factor], int(npoints/reduce_res_factor), spread_value) for x in data]
                ),
            chunksize=1
            )

    ## Interpolating the coarsened data to the original resolution grid.
    print('Interpolate back to original grid.')
    S = data[0].shape

    with Pool(nproc) as p2:
        r2 = p2.starmap(back_to_orig_res, tqdm([(x2, S, reduce_res_factor) for x2 in r]))

    data_new = [csr_matrix(x3) for x3 in r2]

    return data_new




def get_mask_type_list(mask_arrays):

    mask_type_list = [*mask_arrays]
    mask_type_list = [x for x in mask_type_list if 'mask' in x]
    return mask_type_list


def get_masked_rain_at_time(this_dt, this_mask_array, multiply_factor, dataset_dict):

    #### Fill in values by time. Multiply rain field by applicable mask (0 and 1 values).
    precip = np.nan_to_num(
        lpt.readdata.readdata(this_dt,
                              dataset_dict,
                              verbose=False)['data'][:],   # Override verbose
                              nan=0.0) * multiply_factor
    precip[precip < -0.01] = 0.0
    precip[this_mask_array.toarray() < 0.5] = 0.0

    return csr_matrix(precip)


def add_masked_rain_rates(mask_array, mask_times, multiply_factor, dataset_dict, nproc=1):

    ## Parallelize in time.
    with Pool(nproc) as p:
        mask_array_new = p.starmap(get_masked_rain_at_time, tqdm([(mask_times[tt],
                                    mask_array[tt],
                                    multiply_factor,
                                    dataset_dict) for tt in range(len(mask_times))]))

    return mask_array_new


def get_volrain_at_time(this_dt, this_mask_array, multiply_factor, AREA, dataset_dict):

    ## Initialize
    this_volrain = {}

    mask_type_list = get_mask_type_list(this_mask_array)

    ## Get rain
    try:
        RAIN = lpt.readdata.readdata(this_dt, dataset_dict, verbose=False) # Override verbose

        precip = RAIN['data'][:] * multiply_factor / 24.0  # Assuming LPT analysis is in mm/day, get mm/h.
        precip[~np.isfinite(precip)] = 0.0
        precip[precip < -0.01] = 0.0

        
        ## Global
        this_volrain['volrain_global_tser'] = np.sum(precip * AREA)

        ## Masked
        for field in mask_type_list:
            this_mask = this_mask_array[field].toarray()
            this_mask[this_mask > 0] = 1.0
            precip_masked = precip * this_mask
            this_volrain[field.replace('mask','volrain')+'_tser'] = np.sum(precip_masked * AREA)

    ## If there is a problem reading the rain file (missint or corrupt), return NaN.
    except:
        this_volrain['volrain_global_tser'] = np.nan
        for field in mask_type_list:
            this_volrain[field.replace('mask','volrain')+'_tser'] = np.nan
            
    return this_volrain


def mask_calc_volrain(mask_times,interval_hours,multiply_factor,AREA,mask_arrays, dataset_dict, nproc=1):

    this_volrain = {}
    mask_type_list = [*mask_arrays]
    mask_type_list = [x for x in mask_type_list if 'mask' in x]

    #### Initialize
    VOLRAIN = {}

    ## Global
    VOLRAIN['volrain_global'] = 0.0
    VOLRAIN['volrain_global_tser'] = np.nan * np.zeros(len(mask_times))

    #### Fill in values by time. Multiply rain field by applicable mask (0 and 1 values).
    with Pool(nproc) as p:
        r = p.starmap(get_volrain_at_time,
                      tqdm([(mask_times[tt], 
                             {key:value[tt] for (key,value) in mask_arrays.items()}, 
                             multiply_factor, AREA, dataset_dict) for tt in range(len(mask_times))]))

    ## Put the outputs into lists for output.
    VOLRAIN['volrain_global_tser'] = [x['volrain_global_tser'] for x in r]
    VOLRAIN['volrain_global'] = np.nansum(VOLRAIN['volrain_global_tser'])*interval_hours

    ## Masked
    for field in mask_type_list:
        if not 'with_rain' in field:   # Skip the ones that are masked rain rates.
            VOLRAIN[field.replace('mask','volrain')+'_tser'] = [x[field.replace('mask','volrain')+'_tser'] for x in r]
            VOLRAIN[field.replace('mask','volrain')] = np.nansum(VOLRAIN[field.replace('mask','volrain')+'_tser'])*interval_hours

    return VOLRAIN


def add_mask_var_to_netcdf(fn, mask_var_name, data, memory_target_mb = 1000, data_mask = None):
    """
    Append a 3-D mask variable to a NetCDF file.
    FILE MUST EXIST AND HAVE mask_var DEFINED!!!
    add_mask_var_to_netcdf(DS, mask_var, data, dims=('time','lat','lon'))

    For very large mask arrays, writing all at once may consume all the system memory!
    Therefore, we write the data in batches. The batch size is determined based on
    the specified memory target for writing.

    Inputs:
        fn        is a netCDF4.Dataset file object.
        mask_var  is a string for the variable name.
        data      is the data to use.
        memory_target_mb  is memory target for writing to file.
    Outputs:
        -- None --
    """

    ## Figure out how many times I can write at once to honor memory target.
    F = data[0].toarray()
    memory_of_dense_single_time = F.size * F.itemsize / (1024*1024)

    ## Use a factor of two here as memory is used both when:
    ## -- Constructing the dense array from sparse arrays.
    ## -- Writing to the NetCDF file.
    batch_n_time = max(1,int(np.floor(memory_target_mb/(2*memory_of_dense_single_time))))

    for tt1 in range(0,len(data),batch_n_time):
        with Dataset(fn, 'r+') as DS:
            tt2 = min(tt1 + batch_n_time, len(data))
            if 'with_rain' in mask_var_name:
                if data_mask == None:
                    data_to_write = np.array([data[tt].toarray() for tt in range(tt1,tt2)], dtype=np.float32)
                else:
                    data_to_write = np.array([data[tt].toarray() for tt in range(tt1,tt2)], dtype=np.float32)
                    data_to_write[np.logical_not(np.array([data_mask[tt].toarray() for tt in range(tt1,tt2)], dtype=np.float32))] = np.nan
                DS[mask_var_name][tt1:tt2,:,:] = data_to_write

            else:
                if mask_var_name == 'mask':
                    DS[mask_var_name][tt1:tt2,:,:] = np.array([data[tt].toarray() for tt in range(tt1,tt2)], dtype='int')
                else:
                    DS[mask_var_name][tt1:tt2,:,:] = np.array([data[tt].toarray() for tt in range(tt1,tt2)], dtype='bool_')


def add_volrain_to_netcdf(DS, volrain_var_sum, data_sum
        , volrain_var_tser, data_tser, sum_dims=('n',), tser_dims=('time',)
        , fill_value = -999):
    """
    Add sum and time series volrain to a NetCDF file.
    add_volrain_to_netcdf(DS, volrain_var_sum, data_sum
        , volrain_var_tser, data_tser, tser_dims=('time',))

    Inputs:
        DS                is a netCDF4.Dataset file object.
        volrain_var_sum   is a string for the volrain sum variable name.
        data_sum          is the volrain sum data to use.
        sum_dims          is a tuple of dimension names for sum data. Default: ('n',)
        volrain_var_tser  is a string for the volrain time series variable name.
        data_tser         is the volrain time series data to use.
        tser_dims         is a tuple of dimension names for time series. Default: ('time',)
    Outputs:
        -- None --
    """


    ## Sum
    DS.createVariable(volrain_var_sum,'f4',sum_dims,fill_value=fill_value)
    DS[volrain_var_sum][:] = data_sum
    DS[volrain_var_sum].setncatts({'units':'mm - km2','description':'Time Integrated Volumetric Rain','description':'Volumetric rain (sum of area * raw rain rate).'})
    ## Time Series
    DS.createVariable(volrain_var_tser,'f4',tser_dims,fill_value=fill_value)
    DS[volrain_var_tser][:] = data_tser
    DS[volrain_var_tser].setncatts({'units':'mm - km2','description':'Time Series of Volumetric Rain','description':'Volumetric rain (sum of area * raw rain rate).'})


################################################################################
################################################################################
################################################################################
###########  calc_lpo_mask #####################################################
################################################################################
################################################################################
################################################################################

def calc_lpo_mask(dt_begin, dt_end, interval_hours, accumulation_hours = 0, filter_stdev = 0
    , lp_objects_dir = '.', lp_objects_fn_format='objects_%Y%m%d%H.nc', mask_output_dir = '.'
    , detailed_output = False
    , include_rain_rates=False, do_volrain=False, dataset_dict = {}
    , calc_with_filter_radius = True
    , calc_with_accumulation_period = True
    , cold_start_mode = False
    , coarse_grid_factor = 0
    , multiply_factor = 1.0
    , units = '1'
    , memory_target_mb = 1000
    , nproc = 1):

    MISSING = -999.0
    FILL_VALUE = MISSING

    """
    dt_begin, dt_end: datetime objects for the first and last times. These are END of accumulation times!
    """

    YMDH1_YMDH2 = (dt_begin.strftime('%Y%m%d%H') + '_' + dt_end.strftime('%Y%m%d%H'))

    dt1 = dt_end
    if accumulation_hours > 0 and calc_with_accumulation_period and not cold_start_mode: #Cold start mode doesn't have data before init time.
        dt0 = dt_begin - dt.timedelta(hours=accumulation_hours)
        dt_idx0 = int(accumulation_hours/interval_hours)
    else:
        dt0 = dt_begin
        dt_idx0 = 0
    duration_hours = int((dt1 - dt0).total_seconds()/3600)
    mask_times = [dt0 + dt.timedelta(hours=x) for x in range(0,duration_hours+interval_hours,interval_hours)]
    mask_arrays = None

    for dt_idx in range(dt_idx0, len(mask_times)):
        this_dt = mask_times[dt_idx]

        if dt_idx == dt_idx0 or dt_idx == len(mask_times)-1 or np.mod(dt_idx, 100) == 0:
            print(('LPO Mask: ' + this_dt.strftime('%Y%m%d%H') + ' ('+str(dt_idx)+' of '+str(len(mask_times)-1)+')'), flush=True)

        fn = (lp_objects_dir + '/' + this_dt.strftime(lp_objects_fn_format)).replace('///','/').replace('//','/')
        try:
            DS=Dataset(fn)
        except:
            print('WARNING: For LPO mask calculation, Objects file not found: ' + fn)
            continue


        ## Initialize the mask arrays dictionary if this is the first LP object.
        ## First, I need the grid information. Get this from the first LP object.
        if mask_arrays is None:
            mask_lon = DS['grid_lon'][:]
            mask_lat = DS['grid_lat'][:]
            AREA = DS['grid_area'][:]

            mask_arrays = initialize_mask_arrays(
                len(mask_lat), len(mask_lon), len(mask_times),
                detailed_output,
                calc_with_accumulation_period, accumulation_hours,
                calc_with_filter_radius)

        ##
        ## Get LP Object pixel information.
        ##
        try:
            iii = DS['pixels_x'][:].compressed()
            jjj = DS['pixels_y'][:].compressed()
            DS.close()

            ##
            ## Fill in the mask information.
            ##

            if detailed_output:
                ## For mask_at_end_time, just use the mask from the objects file.
                mask_arrays['mask_at_end_time'][dt_idx][jjj, iii] = 1

                ## For the mask with accumulation, go backwards and fill in ones.
                if accumulation_hours > 0 and calc_with_accumulation_period:
                    n_back = int(accumulation_hours/interval_hours)
                    for ttt in range(dt_idx - n_back, dt_idx+1):
                        mask_arrays['mask_with_accumulation'][ttt][jjj, iii] = 1

            else:
                # For mask > 1, apply accumulation hours, if specified.
                # This region is expanded below in "filter width spreading"
                # For the consolidated "mask" variable, I need to make sure
                # I don't set values of "2" to be "1" in future loop iterations.
                if accumulation_hours > 0:
                    n_back = int(accumulation_hours/interval_hours)
                    for ttt in range(dt_idx - n_back, dt_idx+1):
                        dummy = mask_arrays['mask'][ttt].copy()
                        dummy[jjj, iii] = 1
                        mask_arrays['mask'][ttt] = mask_arrays['mask'][ttt].maximum(dummy)

                # Set the "inner core" of the mask to 2.
                # (or 1, if no filter or accumulation)
                if accumulation_hours < 0.01 and filter_stdev < 0.01:
                    mask_arrays['mask'][dt_idx][jjj, iii] = 1
                else:
                    mask_arrays['mask'][dt_idx][jjj, iii] = 2

        except:
            DS.close()

    ##
    ## Do filter width spreading.
    ##

    do_filter = False
    if type(filter_stdev) is list:
        if filter_stdev[0] > 0 and (calc_with_filter_radius or detailed_output):
            do_filter = True
    else:
        if filter_stdev > 0 and (calc_with_filter_radius or detailed_output):
            do_filter = True
    if do_filter:
        print('Filter width spreading...this may take awhile.', flush=True)

        if detailed_output:

            if coarse_grid_factor > 1:
                mask_arrays['mask_with_filter_at_end_time'] = feature_spread_reduce_res(mask_arrays['mask_at_end_time'], filter_stdev, coarse_grid_factor, nproc=nproc)
            else:
                mask_arrays['mask_with_filter_at_end_time'] = feature_spread(mask_arrays['mask_at_end_time'], filter_stdev, nproc=nproc)
            if accumulation_hours > 0 and calc_with_accumulation_period:
                if coarse_grid_factor > 1:
                    mask_arrays['mask_with_filter_and_accumulation'] = feature_spread_reduce_res(mask_arrays['mask_with_accumulation'], filter_stdev, coarse_grid_factor, nproc=nproc)
                else:
                    mask_arrays['mask_with_filter_and_accumulation'] = feature_spread(mask_arrays['mask_with_accumulation'], filter_stdev, nproc=nproc)

        else:
            # Only spread the "1" region. Not "2."
            if coarse_grid_factor > 1:
                mask_arrays['mask'] = feature_spread_reduce_res(
                    mask_arrays['mask'], filter_stdev,
                    reduce_res_factor=coarse_grid_factor,
                    spread_value=1, nproc=nproc)
            else:
                mask_arrays['mask'] = feature_spread(
                    mask_arrays['mask'], filter_stdev,
                    spread_value=1, nproc=nproc)

    ## Do volumetric rain.
    if do_volrain:
        print('Now calculating the volumetric rain.', flush=True)
        VOLRAIN = mask_calc_volrain(mask_times,interval_hours,multiply_factor,AREA,mask_arrays,dataset_dict,nproc=nproc)

    ##
    ## Output.
    ##

    ## Set coordinates
    coords_dict = {}
    coords_dict['n'] = (['n',], [1,])
    coords_dict['time'] = (['time',], mask_times)
    coords_dict['lon'] = (['lon',], mask_lon, {'units':'degrees_east'})
    coords_dict['lat'] = (['lat',], mask_lat, {'units':'degrees_north'})

    ## Set Data
    data_dict = {}
    data_dict['grid_area'] = (['lat','lon',], AREA, {'units':'km2','description':'Area of each grid cell.'})

    # Volumetric Rain, if specified.
    if do_volrain:
        fields = [*VOLRAIN]
        for field in fields:
            if 'tser' in field:
                data_dict[field] = (['time',], VOLRAIN[field])
            else:
                data_dict[field] = (['n',], [VOLRAIN[field],])
    
    ## Create XArray Dataset
    DS = xr.Dataset(data_vars=data_dict, coords=coords_dict)

    ## Write the data to NetCDF
    fn_out = (mask_output_dir + '/lp_objects_mask_' + YMDH1_YMDH2 + '.nc').replace('///','/').replace('//','/')
    os.makedirs(mask_output_dir, exist_ok=True)

    print('Writing to: ' + fn_out, flush=True)
    DS.to_netcdf(path=fn_out, mode='w', unlimited_dims=['time',], encoding={'time': {'dtype': 'i'}, 'n': {'dtype': 'i'}})

    ## Define the mask variables here.
    ## But don't assign them yet.
    DSnew = Dataset(fn_out, 'a')
    fields = [*mask_arrays]
    fields = [x for x in fields if 'mask' in x]
    for field in fields:
        dtype = 'i1'
        this_units = '1'
        DSnew.createVariable(field,dtype,('time','lat','lon'),zlib=True,complevel=4)
        DSnew[field].setncattr('units',this_units)
    DSnew.close()

    ## Writing mask variables.
    ## Do them one at a time so it uses less memory while writing them.
    for field in fields:
        print('- ' + field)
        add_mask_var_to_netcdf(fn_out, field, mask_arrays[field]
                            , memory_target_mb = memory_target_mb)

    ## Include masked rain rates, if specified.
    if include_rain_rates:
        print('Adding masked rainfall.', flush=True)
        fields = [*mask_arrays]
        fields = [x for x in fields if 'mask' in x]
        dtype = 'float32'
        this_units = units

        for field in fields:
            new_field = field + '_with_rain'
            print(new_field)
            mask_array_with_rain = add_masked_rain_rates(mask_arrays[field], mask_times, multiply_factor, dataset_dict, nproc=nproc)

            with Dataset(fn_out, 'a') as DSnew:
                DSnew.createVariable(new_field,dtype,('time','lat','lon'),zlib=True,complevel=4)
                DSnew[new_field].setncattr('units',this_units)
            add_mask_var_to_netcdf(fn_out, new_field, mask_array_with_rain
                                , memory_target_mb = memory_target_mb, data_mask = mask_arrays[field])

        print('Done adding masked rainfall.')


################################################################################
################################################################################
################################################################################
###########  calc_individual_lpt_masks #########################################
################################################################################
################################################################################
################################################################################


def get_lpo_grid_coords(lp_object_id, lp_objects_dir, lp_objects_fn_format):

    dt_this0 = lpt.helpers.get_objid_datetime(lp_object_id)
    dt_this = cftime.datetime(dt_this0.year,dt_this0.month,dt_this0.day,dt_this0.hour)

    fn = (lp_objects_dir + '/' + dt_this.strftime(lp_objects_fn_format))
    
    with Dataset(fn) as ds:
        grid_lon = ds['grid_lon'][:]
        grid_lat = ds['grid_lat'][:]
        grid_area = ds['grid_area'][:]

    return (grid_lon, grid_lat, grid_area)


def get_lpo_grid_points(lp_object_id, mask_times, lp_objects_dir, lp_objects_fn_format):

    nnnn = int(str(int(lp_object_id))[-4:])

    dt_this0 = lpt.helpers.get_objid_datetime(lp_object_id)
    dt_this = cftime.datetime(dt_this0.year,dt_this0.month,dt_this0.day,dt_this0.hour) #,calendar=mask_times[0].calendar)
    dt_idx = [tt for tt in range(len(mask_times)) if dt_this == mask_times[tt]][0]

    fn = (lp_objects_dir + '/' + dt_this.strftime(lp_objects_fn_format))
    
    with Dataset(fn) as ds:

        ##
        ## Get LP Object pixel information.
        ##
        iii = ds['pixels_x'][nnnn,:].compressed().tolist()
        jjj = ds['pixels_y'][nnnn,:].compressed().tolist()

    return (dt_idx, iii, jjj)


def calc_individual_lpt_masks(dt_begin, dt_end, interval_hours, prod='trmm'
    ,accumulation_hours = 0, filter_stdev = 0
    , lp_objects_dir = '.', lp_objects_fn_format='objects_%Y%m%d%H.nc'
    , lpt_systems_dir = '.'
    , mask_output_dir = '.', verbose=False
    , do_volrain=False, include_rain_rates = False
    , dataset_dict = {}
    , detailed_output = False
    , calc_with_filter_radius = True
    , calc_with_accumulation_period = True
    , cold_start_mode = False
    , begin_lptid = 0, end_lptid = 10000, mjo_only = False
    , multiply_factor = 1.0
    , units = '1'
    , coarse_grid_factor = 0
    , memory_target_mb = 1000
    , nproc = 1):

    """
    dt_begin, dt_end: datetime objects for the first and last times. These are END of accumulation times!
    """
    YMDH1_YMDH2 = (dt_begin.strftime('%Y%m%d%H') + '_' + dt_end.strftime('%Y%m%d%H'))

    lpt_systems_file = (lpt_systems_dir + '/lpt_systems_'+prod+'_'+YMDH1_YMDH2+'.nc').replace('///','/').replace('//','/')
    # lpt_group_file = (lpt_systems_dir + '/lpt_systems_'+prod+'_'+YMDH1_YMDH2+'.group_array.txt').replace('///','/').replace('//','/')

    MISSING = -999.0
    FILL_VALUE = MISSING

    ## Read Stitched data.
    TC = lpt.lptio.read_lpt_systems_netcdf(lpt_systems_file)
    unique_lpt_ids = np.unique(TC['lptid'])

    if mjo_only:
        lpt_list_file = (lpt_systems_dir + '/mjo_lpt_list_'+prod+'_'+YMDH1_YMDH2+'.txt')
        try:
            lpt_list = np.loadtxt(lpt_list_file,skiprows=1)
            if len(lpt_list.shape) == 1:
                mjo_lpt_list = np.unique(lpt_list[2])
            else:
                mjo_lpt_list = np.unique(lpt_list[:,2])
        except:
            mjo_lpt_list = np.array([-999]) #If trouble reading, probably no MJOs in the file.
    else:
        mjo_lpt_list = np.array([-999])

    for this_lpt_idx, this_lpt_id in enumerate(TC['lptid']):
        if int(np.floor(this_lpt_id)) < int(np.floor(begin_lptid)) or int(np.floor(this_lpt_id)) > int(np.floor(end_lptid)):
            continue
        if mjo_only and np.min(np.abs(mjo_lpt_list - this_lpt_id)) > 0.0001:
            continue
        print('Calculating LPT system mask for lptid = ' + str(this_lpt_id) + ' of time period ' + YMDH1_YMDH2 + '.')

        ## Get list of LP Objects for this LPT system.
        lp_object_id_list = TC['objid'][this_lpt_idx,0:int(TC['num_objects'][this_lpt_idx])]

        if accumulation_hours > 0 and calc_with_accumulation_period and not cold_start_mode: #Cold start mode doesn't have data before init time.
            dt00 = dt.datetime.strptime(str(int(np.min(lp_object_id_list))).zfill(14)[0:10],'%Y%m%d%H') - dt.timedelta(hours=accumulation_hours)
        else:
            dt00 = dt.datetime.strptime(str(int(np.min(lp_object_id_list))).zfill(14)[0:10],'%Y%m%d%H')
        dt0 = cftime.datetime(dt00.year,dt00.month,dt00.day,dt00.hour,calendar=TC['datetime'][0].calendar)
        dt11 = dt.datetime.strptime(str(int(np.max(lp_object_id_list))).zfill(14)[0:10],'%Y%m%d%H')
        dt1 = cftime.datetime(dt11.year,dt11.month,dt11.day,dt11.hour,calendar=TC['datetime'][0].calendar)
        duration_hours = int((dt1 - dt0).total_seconds()/3600)
        mask_times = [dt0 + dt.timedelta(hours=x) for x in range(0,duration_hours+interval_hours,interval_hours)]
        # mask_arrays = None


        ## Initialize the mask arrays dictionary if this is the first LP object.
        ## First, I need the grid information. Get this from the first LP object.
        mask_lon, mask_lat, AREA = get_lpo_grid_coords(
            lp_object_id_list[0], lp_objects_dir, lp_objects_fn_format
        )
        
        mask_arrays = initialize_mask_arrays(
            len(mask_lat), len(mask_lon), len(mask_times),
            detailed_output,
            calc_with_accumulation_period, accumulation_hours,
            calc_with_filter_radius
        )

        # Get iii, jjj grid points for each LPO.
        # points_collect = []
        # for lp_object_id in lp_object_id_list:
        #     points_collect += [get_lpo_grid_points(lp_object_id, mask_times,
        #         lp_objects_dir, lp_objects_fn_format)]

        with Pool(32) as p:
            points_collect = p.starmap(
                get_lpo_grid_points,
                [(x, mask_times, lp_objects_dir, lp_objects_fn_format) for x in lp_object_id_list],
                chunksize=1
            )

        for points in points_collect:
            dt_idx = points[0]
            iii = points[1]
            jjj = points[2]
            ##
            ## Fill in the mask information.
            ##
            if detailed_output:
                ## For mask_at_end_time, just use the mask from the objects file.
                mask_arrays['mask_at_end_time'][dt_idx][jjj, iii] = 1

                ## For the mask with accumulation, go backwards and fill in ones.
                if accumulation_hours > 0 and calc_with_accumulation_period:
                    n_back = int(accumulation_hours/interval_hours)
                    for ttt in range(dt_idx - n_back, dt_idx+1):
                        mask_arrays['mask_with_accumulation'][ttt][jjj, iii] = 1

            else:
                # For mask > 1, apply accumulation hours, if specified.
                # This region is expanded below in "filter width spreading"
                # For the consolidated "mask" variable, I need to make sure
                # I don't set values of "2" to be "1" in future loop iterations.
                if accumulation_hours > 0:
                    n_back = int(accumulation_hours/interval_hours)
                    for ttt in range(dt_idx - n_back, dt_idx+1):
                        dummy = mask_arrays['mask'][ttt].copy()
                        dummy[jjj, iii] = 1
                        mask_arrays['mask'][ttt] = mask_arrays['mask'][ttt].maximum(dummy)

                # Set the "inner core" of the mask to 2.
                # (or 1, if no filter or accumulation)
                if accumulation_hours < 0.01 and filter_stdev < 0.01:
                    mask_arrays['mask'][dt_idx][jjj, iii] = 1
                else:
                    mask_arrays['mask'][dt_idx][jjj, iii] = 2


        ##
        ## Get contours from the grid.
        ##

        ntimes = len(TC['timestamp_stitched']) #len(mask_times)
        max_len = 1   # 2000
        max_len_core = 1 # 4000

        with xr.open_dataset(lpt_systems_file) as ds:
            if 'mask_contour_lon' in ds:

                F_lon = ds['mask_contour_lon'].data
                F_lat = ds['mask_contour_lat'].data
                F_lon_core = ds['mask_contour_core_lon'].data
                F_lat_core = ds['mask_contour_core_lat'].data
                max_len = F_lat.shape[1]
                max_len_core = F_lat_core.shape[1]

            else:

                F_lon = np.full([ntimes, max_len], np.nan)
                F_lat = np.full([ntimes, max_len], np.nan)
                F_lon_core = np.full([ntimes, max_len_core], np.nan)
                F_lat_core = np.full([ntimes, max_len_core], np.nan)

            mask_contour_lon = [[] for x in range(len(mask_times))]
            mask_contour_lat = [[] for x in range(len(mask_times))]
            mask_contour_lon_core = [[] for x in range(len(mask_times))]
            mask_contour_lat_core = [[] for x in range(len(mask_times))]

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # NOTE: 
            # I need to add the "spin-up" time (e.g., 3 days) to the stitched variables.
            # Right now, the consecutive LPT systems are running in to eachother.
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            for dt_idx, dt_this in enumerate(mask_times):

                # Just in case the mask hits the north or south pole,
                # I need it to have a closed contour.
                # So set the mask values at the north and south poles
                # to zero for this step.
                this_mask = mask_arrays['mask'][dt_idx].todense()
                this_mask[0,:] = 0
                this_mask[-1,:] = 0
                # In rare cases, it may wrap all the way around! Then it
                # would not be a closed contour, and not have an exterior.
                # Set the first column to zero.
                if np.nanmin(np.nanmax(this_mask, axis=0)) > 0.5:
                    this_mask[:,0] = 0
    
                cg = contourpy.contour_generator(
                    mask_lon, mask_lat, this_mask)

                contours = cg.lines(0.5)
                contours_core = cg.lines(1.5)

                for this_contour in contours_core:
                    mask_contour_lon_core[dt_idx] += [np.nan,]
                    mask_contour_lat_core[dt_idx] += [np.nan,]
                    mask_contour_lon_core[dt_idx] += this_contour[:,0].flatten().tolist()
                    mask_contour_lat_core[dt_idx] += this_contour[:,1].flatten().tolist()

                # Filter width spreading. (Only for the outer contour)
                for this_contour in contours:
                    res = mask_lon[1] - mask_lon[0]
                    s = Polygon(LinearRing(this_contour))
                    print(dt_this)
                    t2 = Polygon(s.buffer(filter_stdev*res).exterior)
                    t2_coords = np.array(
                        [t2.exterior.coords[x] for x in range(len(t2.exterior.coords))])

                    mask_contour_lon[dt_idx] += [np.nan,]
                    mask_contour_lat[dt_idx] += [np.nan,]
                    mask_contour_lon[dt_idx] += t2_coords[:,0].flatten().tolist()
                    mask_contour_lat[dt_idx] += t2_coords[:,1].flatten().tolist()

            for dt_idx, dt_this in enumerate(mask_times):

                if dt_idx < 24:
                    continue

                timestamp_stitched_idx = np.argwhere(
                    np.logical_and(
                        TC['lptid_stitched'] == this_lpt_id, 
                        TC['timestamp_stitched'] == dt_this)) 
                # Check if I need to expand the arrays.
                if len(mask_contour_lon[dt_idx]) > max_len:
                    print('Expand max_lon to: ', len(mask_contour_lon[dt_idx]))
                    array_to_stack = np.full([ntimes, len(mask_contour_lon[dt_idx]) - max_len], np.nan)
                    F_lon = np.column_stack((F_lon, array_to_stack))
                    F_lat = np.column_stack((F_lat, array_to_stack))
                    max_len = len(mask_contour_lon[dt_idx])
                if len(mask_contour_lon_core[dt_idx]) > max_len_core:
                    print('Expand max_lon to: ', len(mask_contour_lon_core[dt_idx]))
                    array_to_stack = np.full([ntimes, len(mask_contour_lon_core[dt_idx]) - max_len_core], np.nan)
                    F_lon_core = np.column_stack((F_lon_core, array_to_stack))
                    F_lat_core = np.column_stack((F_lat_core, array_to_stack))
                    max_len_core = len(mask_contour_lon_core[dt_idx])

                F_lon[timestamp_stitched_idx, 0:len(mask_contour_lon[dt_idx])] = mask_contour_lon[dt_idx]
                F_lat[timestamp_stitched_idx, 0:len(mask_contour_lat[dt_idx])] = mask_contour_lat[dt_idx]
                F_lon_core[timestamp_stitched_idx, 0:len(mask_contour_lon_core[dt_idx])] = mask_contour_lon_core[dt_idx]
                F_lat_core[timestamp_stitched_idx, 0:len(mask_contour_lat_core[dt_idx])] = mask_contour_lat_core[dt_idx]


            ## Write mask contour coordinates to LPT systems file.
            ## First, start a new set of variables if it does not yet exist.
            ## Otherwise modify the existing variables.

            with xr.open_dataset(lpt_systems_file) as ds:

                new_coords = {'mask_contour_npts': (['mask_contour_npts',], range(max_len)),
                                'mask_contour_core_npts': (['mask_contour_core_npts',], range(max_len_core))}

                desc = ('The mask_contour_core is the contour of the LPOs which make up the LPT system.'
                    + ' The mask_contour includes the entire LPO area swept out during the accumulation period'
                    + ' as well as a buffer of one standard deviation.'
                    + ' The mask_contour contains the full extent of rainfall contributing to the LPT system,'
                    + ' whereas the mask_contour_core is regarded as the core region of the LPT system.')

                new_data_vars = {
                    'mask_contour_lon': (['nstitch','mask_contour_npts'], F_lon, {'units':'degrees_east','description':desc}),
                    'mask_contour_lat': (['nstitch','mask_contour_npts'], F_lat, {'units':'degrees_north','description':desc}),
                    'mask_contour_core_lon': (['nstitch','mask_contour_core_npts'], F_lon_core, {'units':'degrees_east','description':desc}),
                    'mask_contour_core_lat': (['nstitch','mask_contour_core_npts'], F_lat_core, {'units':'degrees_north','description':desc})}

                ds2 = ds.copy()
                if 'mask_contour_lon' in ds2.variables:
                    ds2 = ds2.drop_vars(['mask_contour_lon','mask_contour_lat','mask_contour_core_lon','mask_contour_core_lat'])
                ds2 = ds2.assign_coords(new_coords)
                ds2 = ds2.assign(new_data_vars)

            encoding = {
                'nlpt': {'dtype': 'i'}, 'nstitch': {'dtype': 'i'},
                'nobj': {'dtype': 'i'}, 'nobj_stitched': {'dtype': 'i'},
                'num_objects': {'dtype': 'i'},
                'mask_contour_npts': {'dtype': 'i'},
                'mask_contour_core_npts': {'dtype': 'i'},
                'is_mjo': {'dtype': 'bool'}, 'is_mjo_stitched': {'dtype': 'bool'},
                'is_mjo_eprop_stitched': {'dtype': 'bool'},
                'mask_contour_lon': {'zlib': True},
                'mask_contour_lat': {'zlib': True},
                'mask_contour_core_lon': {'zlib': True},
                'mask_contour_core_lat': {'zlib': True},
                }

            print(f'Update mask contour coordinates in {lpt_systems_file}')
            fn_temp = f'temp.{os.getpid()}.nc'
            ds2.to_netcdf(path=fn_temp, mode='w', encoding=encoding)
            os.rename(fn_temp, lpt_systems_file)

        ##
        ## Do filter width spreading.
        ##

        do_filter = False
        if type(filter_stdev) is list:
            if filter_stdev[0] > 0 and (calc_with_filter_radius or detailed_output):
                do_filter = True
        else:
            if filter_stdev > 0 and (calc_with_filter_radius or detailed_output):
                do_filter = True
        if do_filter:
            print('Filter width spreading...this may take awhile.', flush=True)

            if detailed_output:

                if coarse_grid_factor > 1:
                    mask_arrays['mask_with_filter_at_end_time'] = feature_spread_reduce_res(mask_arrays['mask_at_end_time'], filter_stdev, coarse_grid_factor, nproc=nproc)
                else:
                    mask_arrays['mask_with_filter_at_end_time'] = feature_spread(mask_arrays['mask_at_end_time'], filter_stdev, nproc=nproc)
                if accumulation_hours > 0 and calc_with_accumulation_period:
                    if coarse_grid_factor > 1:
                        mask_arrays['mask_with_filter_and_accumulation'] = feature_spread_reduce_res(mask_arrays['mask_with_accumulation'], filter_stdev, coarse_grid_factor, nproc=nproc)
                    else:
                        mask_arrays['mask_with_filter_and_accumulation'] = feature_spread(mask_arrays['mask_with_accumulation'], filter_stdev, nproc=nproc)

            else:
                # Only spread the "1" region. Not "2."
                if coarse_grid_factor > 1:
                    mask_arrays['mask'] = feature_spread_reduce_res(
                        mask_arrays['mask'], filter_stdev,
                        reduce_res_factor=coarse_grid_factor,
                        spread_value=1, nproc=nproc)
                else:
                    mask_arrays['mask'] = feature_spread(
                        mask_arrays['mask'], filter_stdev,
                        spread_value=1, nproc=nproc)

        ## Do volumetric rain.
        if do_volrain:
            print('Now calculating the volumetric rain.', flush=True)
            VOLRAIN = mask_calc_volrain(mask_times,interval_hours,multiply_factor,AREA,mask_arrays,dataset_dict,nproc=nproc)

            # Save the volrain to the LPT systems file.
            with xr.open_dataset(lpt_systems_file) as ds:
                if 'volrain' in ds:

                    volrain = ds['volrain'].data
                    volrain_global = ds['volrain_global'].data
                    maxvolrain = ds['maxvolrain'].data
                    maxvolrain_global = ds['maxvolrain_global'].data
                    volrain_tser = ds['volrain_stitched'].data
                    volrain_global_tser = ds['volrain_global_stitched'].data

                else:

                    volrain = np.full([len(TC['lptid']),], np.nan)
                    volrain_global = np.full([len(TC['lptid']),], np.nan)
                    maxvolrain = np.full([len(TC['lptid']),], np.nan)
                    maxvolrain_global = np.full([len(TC['lptid']),], np.nan)
                    volrain_tser = np.full([ntimes,], np.nan)
                    volrain_global_tser = np.full([ntimes,], np.nan)

            volrain[this_lpt_idx] = VOLRAIN['volrain']
            volrain_global[this_lpt_idx] = VOLRAIN['volrain_global']

            for dt_idx, dt_this in enumerate(mask_times):

                if dt_idx < 24:
                    continue

                timestamp_stitched_idx = np.argwhere(
                    np.logical_and(
                        TC['lptid_stitched'] == this_lpt_id, 
                        TC['timestamp_stitched'] == dt_this)) 

                volrain_tser[timestamp_stitched_idx] = VOLRAIN['volrain_tser'][dt_idx]
                volrain_global_tser[timestamp_stitched_idx] = VOLRAIN['volrain_global_tser'][dt_idx]

            maxvolrain[this_lpt_idx] = np.nanmax(volrain_tser)
            maxvolrain_global[this_lpt_idx] = np.nanmax(volrain_global_tser)

            # Add the volrain variables to the output NetCDF file.
            print(f'Adding volrain variables to: {lpt_systems_file}')

            volrain_atts1 = {
                'units':'mm h-1 km2',
                'description':'Instantaneous volumetric rain flux of the LPT system.',
                'note':'Units assumes rain is converted to mm d-1 for LPT analysis.'}

            volrain_atts2 = {
                'units':'mm h-1 km2',
                'description':'Instantaneous volumetric rain flux: Entire area with rainfall data.',
                'note':'Units assumes rain is converted to mm d-1 for LPT analysis.'}

            volrain_atts3 = {
                'units':'mm h-1 km2',
                'description':'Max of volrain over the LPT system lifetime.',
                'note':'Units assumes rain is converted to mm d-1 for LPT analysis.'}

            volrain_atts4 = {
                'units':'mm h-1 km2',
                'description':'Max of volrain_global over the LPT system lifetime.',
                'note':'Units assumes rain is converted to mm d-1 for LPT analysis.'}

            volrain_atts5 = {
                'units':'mm km2',
                'description':'Time integral of volrain over the LPT system lifetime.',
                'note':'Units assumes rain is converted to mm d-1 for LPT analysis.'}

            volrain_atts6 = {
                'units':'mm km2',
                'description':'Time integral of volrain_global over the LPT system lifetime.',
                'note':'Units assumes rain is converted to mm d-1 for LPT analysis.'}


            data_vars = {
                'volrain_stitched': (['nstitch',], volrain_tser, volrain_atts1),
                'volrain_global_stitched': (['nstitch',], volrain_global_tser, volrain_atts2),
                'maxvolrain': (['nlpt',], maxvolrain, volrain_atts3),
                'maxvolrain_global': (['nlpt',], maxvolrain_global, volrain_atts4),
                'volrain': (['nlpt',], volrain, volrain_atts5),
                'volrain_global': (['nlpt',], volrain_global, volrain_atts6),
            }

            with xr.open_dataset(lpt_systems_file) as ds:
                ds2 = ds.copy().assign(data_vars)


            encoding = {
                'nlpt': {'dtype': 'i'}, 'nstitch': {'dtype': 'i'},
                'nobj': {'dtype': 'i'}, 'nobj_stitched': {'dtype': 'i'},
                'num_objects': {'dtype': 'i'},
                'mask_contour_npts': {'dtype': 'i'},
                'mask_contour_core_npts': {'dtype': 'i'},
                'is_mjo': {'dtype': 'bool'}, 'is_mjo_stitched': {'dtype': 'bool'},
                'is_mjo_eprop_stitched': {'dtype': 'bool'},
                'mask_contour_lon': {'zlib': True},
                'mask_contour_lat': {'zlib': True},
                'mask_contour_core_lon': {'zlib': True},
                'mask_contour_core_lat': {'zlib': True},
                }

            ds2.to_netcdf(path='./temp.nc', mode='w', encoding=encoding)
            os.rename('./temp.nc', lpt_systems_file)

        ##########################################################
        ## Include some basic LPT info for user friendliness.  ###
        ##########################################################
        lptidx = [ii for ii in range(len(TC['lptid'])) if this_lpt_id == TC['lptid'][ii]][0]

        basic_lpt_info_field_list = ['centroid_lon','centroid_lat','area'
                    ,'largest_object_centroid_lon','largest_object_centroid_lat'
                    ,'max_filtered_running_field','max_running_field','max_inst_field'
                    ,'min_filtered_running_field','min_running_field','min_inst_field'
                    ,'amean_filtered_running_field','amean_running_field','amean_inst_field']

        # Time varying properties
        for var in basic_lpt_info_field_list:
            mask_arrays[var] = MISSING * np.ones(len(mask_times))

        for ttt in range(TC['i1'][lptidx],TC['i2'][lptidx]+1):
            this_time_indx = [ii for ii in range(len(mask_times)) if TC['datetime'][ttt] == mask_times[ii]]
            if len(this_time_indx) > 0:
                for var in basic_lpt_info_field_list:
                    mask_arrays[var][this_time_indx] = TC[var][ttt]

        # Bulk properties
        bulk_lpt_info_field_list = ['lptid','duration','maxarea',
            'zonal_propagation_speed','meridional_propagation_speed']
        for var in bulk_lpt_info_field_list:
            mask_arrays[var] = TC[var][this_lpt_idx]
        ##########################################################


        ##
        ## Output.
        ##

        ## Set coordinates
        coords_dict = {}
        coords_dict['n'] = (['n',], [1,])
        coords_dict['time'] = (['time',], mask_times)
        coords_dict['lon'] = (['lon',], mask_lon, {'units':'degrees_east'})
        coords_dict['lat'] = (['lat',], mask_lat, {'units':'degrees_north'})

        ## Set Data
        data_dict = {}
        data_dict['grid_area'] = (['lat','lon',], AREA, {'units':'km2','description':'Area of each grid cell.'})

        ## Basic LPT system track data for convenience
        for mask_var in basic_lpt_info_field_list:
            data_dict[mask_var] = (['time',], mask_arrays[mask_var])

        # Volumetric Rain, if specified.
        if do_volrain:
            fields = [*VOLRAIN]
            for field in fields:
                if 'tser' in field:
                    data_dict[field] = (['time',], VOLRAIN[field])
                else:
                    data_dict[field] = (['n',], [VOLRAIN[field],])

        for mask_var in bulk_lpt_info_field_list:
            print(mask_var, mask_arrays[mask_var])
            data_dict[mask_var] = (['n',], [mask_arrays[mask_var],])

        ## Create XArray Dataset
        DS = xr.Dataset(data_vars=data_dict, coords=coords_dict)
        DS.centroid_lon.attrs = {'units':'degrees_east','long_name':'centroid longitude (0-360)','standard_name':'longitude','note':'Time is end of running mean time.'}
        DS.centroid_lat.attrs = {'units':'degrees_north','long_name':'centroid latitude (-90-00)','standard_name':'latitude','note':'Time is end of running mean time.'}
        DS.largest_object_centroid_lon.attrs = {'units':'degrees_east','long_name':'centroid longitude (0-360)','standard_name':'longitude','note':'Time is end of running mean time.'}
        DS.largest_object_centroid_lat.attrs = {'units':'degrees_east','long_name':'centroid latitude (-90-00)','standard_name':'latitude','note':'Time is end of running mean time.'}
        DS.area.attrs = {'units':'km2','long_name':'LPT System enclosed area','note':'Time is end of running mean time.'}

        # Time varying fields
        for var in ['max_filtered_running_field','max_running_field','max_inst_field'
                    ,'min_filtered_running_field','min_running_field','min_inst_field'
                    ,'amean_filtered_running_field','amean_running_field','amean_inst_field']:
            DS[var].attrs = {'units':'mm day-1',
                    'long_name':'LP object running mean rain rate (at end of accum time).',
                    'note':'Time is end of running mean time. Based on mask_at_end_time'}

        DS.maxarea.attrs = {'units':'km2','long_name':'LPT System enclosed area','note':'This is the max over the LPT life time.'}
        DS.duration.attrs = {'units':'h','long_name':'LPT System duration'}
        DS.zonal_propagation_speed.attrs = {'units':'m s-1','long_name':'Zonal popagation speed','description':'Zonal popagation speed of the entire LPT system -- based on least squares fit of lon(time).'}
        DS.meridional_propagation_speed.attrs = {'units':'m s-1','long_name':'meridional popagation speed','description':'Meridional popagation speed of the entire LPT system -- based on least squares fit of lon(time).'}



        ## Write the data to NetCDF
        fn_out = (mask_output_dir + '/' + YMDH1_YMDH2 + '/lpt_system_mask_'+prod+'.lptid{0:010.4f}.nc'.format(this_lpt_id))
        os.makedirs(mask_output_dir + '/' + YMDH1_YMDH2, exist_ok=True)

        print('Writing to: ' + fn_out, flush=True)
        DS.to_netcdf(path=fn_out, mode='w', unlimited_dims=['time',], encoding={'time': {'dtype': 'i'}, 'n': {'dtype': 'i'}})
        

        ## Define the mask variables here.
        ## But don't assign them yet.
        with Dataset(fn_out, 'a') as DSnew:
            fields = [*mask_arrays]
            fields = [x for x in fields if 'mask' in x]
            for field in fields:
                dtype = 'i1'
                this_units = '1'
                DSnew.createVariable(field,dtype,('time','lat','lon'),zlib=True,complevel=4)
                DSnew[field].setncattr('units',this_units)

        ## Writing mask variables.
        ## Do them one at a time so it uses less memory while writing them.
        for field in fields:
            print('- ' + field)
            add_mask_var_to_netcdf(fn_out, field, mask_arrays[field]
                                , memory_target_mb = memory_target_mb)

        ## Include masked rain rates, if specified.
        if include_rain_rates:
            print('Adding masked rainfall.', flush=True)
            fields = [*mask_arrays]
            fields = [x for x in fields if 'mask' in x]
            dtype = 'float32'
            this_units = units

            for field in fields:
                new_field = field + '_with_rain'
                print(new_field)
                mask_array_with_rain = add_masked_rain_rates(mask_arrays[field], mask_times, multiply_factor, dataset_dict, nproc=nproc)

                with Dataset(fn_out, 'a') as DSnew:
                    DSnew.createVariable(new_field,dtype,('time','lat','lon'),zlib=True,complevel=4)
                    DSnew[new_field].setncattr('units',this_units)
                add_mask_var_to_netcdf(fn_out, new_field, mask_array_with_rain
                                    , memory_target_mb = memory_target_mb, data_mask = mask_arrays[field])

            print('Done adding masked rainfall.')

        # Add the volrain variables to the output NetCDF file.
        if do_volrain:
            print(f'Adding volrain to: {lpt_systems_file}')

            fields = [*VOLRAIN]
            for field in fields:
                if not 'tser' in field:
                    with xr.open_dataset(lpt_systems_file) as ds:

                        # Which indices do I use for this LPT system?
                        this_lpt = (np.abs(ds['lptid'].data - this_lpt_id) < 0.0001)
                        this_lpt_stitched = (np.abs(ds['lptid_stitched'].data - this_lpt_id) < 0.0001)
                        print((np.abs(ds['lptid'].data - this_lpt_id), this_lpt))

                        if not field in ds:
                            volrain_all = np.nan * ds['lptid'].data
                            volrain_all_stitched = np.nan * ds['lptid_stitched'].data
                        else:
                            volrain_all = ds[field].data
                            volrain_all_stitched = ds[field+'_stitched'].data

                        volrain_all[this_lpt] = VOLRAIN[field]
                        # volrain_all_stitched[this_lpt_stitched] = VOLRAIN[field+'_tser'][72:] # Skip accum. period.
                        volrain_all_stitched[this_lpt_stitched] = VOLRAIN[field+'_tser'][-1*len(volrain_all_stitched[this_lpt_stitched]):] # Skip accum. period.

                        data_vars = {
                            field: (['nlpt',], volrain_all),
                            field+'_stitched': (['nstitch',], volrain_all_stitched),
                        }

                        encoding = {
                            'nlpt': {'dtype': 'i'}, 'nstitch': {'dtype': 'i'},
                            'nobj': {'dtype': 'i'}, 'nobj_stitched': {'dtype': 'i'},
                            'num_objects': {'dtype': 'i'},
                            'is_mjo': {'dtype': 'bool'}, 'is_mjo_stitched': {'dtype': 'bool'},
                            'is_mjo_eprop_stitched': {'dtype': 'bool'}}

                        ds2 = ds.copy().assign(data_vars)

                    fn_temp = f'temp.{os.getpid()}.nc'
                    ds2.to_netcdf(path=fn_temp, mode='w', encoding=encoding)
                    os.rename(fn_temp, lpt_systems_file)


def calc_individual_lpt_group_masks(dt_begin, dt_end, interval_hours, prod='trmm'
    ,accumulation_hours = 0, filter_stdev = 0
    , lp_objects_dir = '.', lp_objects_fn_format='objects_%Y%m%d%H.nc'
    , lpt_systems_dir = '.'
    , mask_output_dir = '.', verbose=False
    , do_volrain=False, include_rain_rates = False
    , dataset_dict = {}
    , detailed_output = False
    , calc_with_filter_radius = True
    , calc_with_accumulation_period = True
    , cold_start_mode = False
    , begin_lptid = 0, end_lptid = 10000, mjo_only = False
    , multiply_factor = 1.0
    , units = '1'
    , coarse_grid_factor = 0
    , memory_target_mb = 1000
    , nproc = 1):

    """
    dt_begin, dt_end: datetime objects for the first and last times. These are END of accumulation times!
    """
    YMDH1_YMDH2 = (dt_begin.strftime('%Y%m%d%H') + '_' + dt_end.strftime('%Y%m%d%H'))

    lpt_systems_file = (lpt_systems_dir + '/lpt_systems_'+prod+'_'+YMDH1_YMDH2+'.nc').replace('///','/').replace('//','/')
    lpt_group_file = (lpt_systems_dir + '/lpt_systems_'+prod+'_'+YMDH1_YMDH2+'.group_array.txt').replace('///','/').replace('//','/')

    MISSING = -999.0
    FILL_VALUE = MISSING

    ## Read Stitched data.
    TC = lpt.lptio.read_lpt_systems_netcdf(lpt_systems_file)
    unique_lpt_ids = np.unique(TC['lptid'])
    unique_lpt_groups = np.unique(np.floor(TC['lptid'])).astype('int')
    print(unique_lpt_groups)
    if mjo_only:
        lpt_list_file = (lpt_systems_dir + '/mjo_lpt_list_'+prod+'_'+YMDH1_YMDH2+'.txt')
        try:
            lpt_list = np.loadtxt(lpt_list_file,skiprows=1)
            if len(lpt_list.shape) == 1:
                mjo_lpt_list = np.unique(lpt_list[2])
            else:
                mjo_lpt_list = np.unique(lpt_list[:,2])
        except:
            mjo_lpt_list = np.array([-999]) #If trouble reading, probably no MJOs in the file.
    else:
        mjo_lpt_list = np.array([-999])

    for this_lpt_group_idx, this_lpt_group in enumerate(unique_lpt_groups):

        if int(np.floor(this_lpt_group)) < int(np.floor(begin_lptid)) or int(np.floor(this_lpt_group)) > int(np.floor(end_lptid)):
            continue

        print('Calculating LPT system mask for LPT group = ' + str(this_lpt_group) + ' of time period ' + YMDH1_YMDH2 + '.')

        lp_object_id_list = []
        lpt_id_in_this_group = [x for x in TC['lptid'] if int(np.floor(x)) == this_lpt_group]
        print(lpt_id_in_this_group)

        for this_lpt_id in lpt_id_in_this_group:

            this_lpt_idx = np.argwhere(TC['lptid'] == this_lpt_id).flatten()[0]
            print('Adding LPOs for lptid = ' + str(this_lpt_id) + '.')

            ## Get list of LP Objects for this LPT system.
            lp_object_id_list = np.append(
                lp_object_id_list,
                TC['objid'][this_lpt_idx,0:int(TC['num_objects'][this_lpt_idx])]
                )

        lp_object_id_list = np.unique(lp_object_id_list)

        if accumulation_hours > 0 and calc_with_accumulation_period and not cold_start_mode: #Cold start mode doesn't have data before init time.
            dt00 = dt.datetime.strptime(str(int(np.min(lp_object_id_list))).zfill(14)[0:10],'%Y%m%d%H') - dt.timedelta(hours=accumulation_hours)
        else:
            dt00 = dt.datetime.strptime(str(int(np.min(lp_object_id_list))).zfill(14)[0:10],'%Y%m%d%H')
        dt0 = cftime.datetime(dt00.year,dt00.month,dt00.day,dt00.hour,calendar=TC['datetime'][0].calendar)
        dt11 = dt.datetime.strptime(str(int(np.max(lp_object_id_list))).zfill(14)[0:10],'%Y%m%d%H')
        dt1 = cftime.datetime(dt11.year,dt11.month,dt11.day,dt11.hour,calendar=TC['datetime'][0].calendar)
        duration_hours = int((dt1 - dt0).total_seconds()/3600)
        mask_times = [dt0 + dt.timedelta(hours=x) for x in range(0,duration_hours+interval_hours,interval_hours)]
        mask_arrays = None

        for lp_object_id in lp_object_id_list:
            nnnn = int(str(int(lp_object_id))[-4:])
            try:
                dt_this0 = lpt.helpers.get_objid_datetime(lp_object_id)
                dt_this = cftime.datetime(dt_this0.year,dt_this0.month,dt_this0.day,dt_this0.hour,calendar=TC['datetime'][0].calendar)
                dt_idx = [tt for tt in range(len(mask_times)) if dt_this == mask_times[tt]]
            except:
                continue


            if len(dt_idx) < 0:
                print('This time not found in mask time list. Skipping LP object id: ' + str(int(lp_object_id)))
                continue
            elif len(dt_idx) > 1:
                print('Found more than one mask time for this LP object. This should not happen! Skipping it.')
                continue
            else:
                dt_idx = dt_idx[0]

            fn = (lp_objects_dir + '/' + dt_this.strftime(lp_objects_fn_format))
            if verbose:
                print(fn)
            DS=Dataset(fn)

            ## Initialize the mask arrays dictionary if this is the first LP object.
            ## First, I need the grid information. Get this from the first LP object.
            if mask_arrays is None:
                mask_lon = DS['grid_lon'][:]
                mask_lat = DS['grid_lat'][:]
                AREA = DS['grid_area'][:]

                mask_arrays = initialize_mask_arrays(
                    len(mask_lat), len(mask_lon), len(mask_times),
                    detailed_output,
                    calc_with_accumulation_period, accumulation_hours,
                    calc_with_filter_radius)

            ##
            ## Get LP Object pixel information.
            ##
            try:
                iii = DS['pixels_x'][nnnn,:].compressed()
                jjj = DS['pixels_y'][nnnn,:].compressed()

            except:
                DS.close()
                continue

            DS.close()

            ##
            ## Fill in the mask information.
            ##

            ## For mask_at_end_time, just use the mask from the objects file.
            if detailed_output:
                mask_arrays['mask_at_end_time'][dt_idx][jjj, iii] = 1

                ## For the mask with accumulation, go backwards and fill in ones.
                if accumulation_hours > 0 and calc_with_accumulation_period:
                    n_back = int(accumulation_hours/interval_hours)
                    for ttt in range(dt_idx - n_back, dt_idx+1):
                        mask_arrays['mask_with_accumulation'][ttt][jjj, iii] = 1

            else:
                # For mask > 1, apply accumulation hours, if specified.
                # This region is expanded below in "filter width spreading"
                # For the consolidated "mask" variable, I need to make sure
                # I don't set values of "2" to be "1" in future loop iterations.
                if accumulation_hours > 0:
                    n_back = int(accumulation_hours/interval_hours)
                    for ttt in range(dt_idx - n_back, dt_idx+1):
                        dummy = mask_arrays['mask'][ttt].copy()
                        dummy[jjj, iii] = 1
                        mask_arrays['mask'][ttt] = mask_arrays['mask'][ttt].maximum(dummy)

                # Set the "inner core" of the mask to 2.
                # (or 1, if no filter or accumulation)
                if accumulation_hours < 0.01 and filter_stdev < 0.01:
                    mask_arrays['mask'][dt_idx][jjj, iii] = 1
                else:
                    mask_arrays['mask'][dt_idx][jjj, iii] = 2

        ##
        ## Do filter width spreading.
        ##

        do_filter = False
        if type(filter_stdev) is list:
            if filter_stdev[0] > 0 and (calc_with_filter_radius or detailed_output):
                do_filter = True
        else:
            if filter_stdev > 0 and (calc_with_filter_radius or detailed_output):
                do_filter = True
        if do_filter:
            print('Filter width spreading...this may take awhile.', flush=True)

            if detailed_output:

                if coarse_grid_factor > 1:
                    mask_arrays['mask_with_filter_at_end_time'] = feature_spread_reduce_res(mask_arrays['mask_at_end_time'], filter_stdev, coarse_grid_factor, nproc=nproc)
                else:
                    mask_arrays['mask_with_filter_at_end_time'] = feature_spread(mask_arrays['mask_at_end_time'], filter_stdev, nproc=nproc)
                if accumulation_hours > 0 and calc_with_accumulation_period:
                    if coarse_grid_factor > 1:
                        mask_arrays['mask_with_filter_and_accumulation'] = feature_spread_reduce_res(mask_arrays['mask_with_accumulation'], filter_stdev, coarse_grid_factor, nproc=nproc)
                    else:
                        mask_arrays['mask_with_filter_and_accumulation'] = feature_spread(mask_arrays['mask_with_accumulation'], filter_stdev, nproc=nproc)

            else:
                # Only spread the "1" region. Not "2."
                if coarse_grid_factor > 1:
                    mask_arrays['mask'] = feature_spread_reduce_res(
                        mask_arrays['mask'], filter_stdev,
                        reduce_res_factor=coarse_grid_factor,
                        spread_value=1, nproc=nproc)
                else:
                    mask_arrays['mask'] = feature_spread(
                        mask_arrays['mask'], filter_stdev,
                        spread_value=1, nproc=nproc)


        ## Do volumetric rain.
        if do_volrain:
            print('Now calculating the volumetric rain.', flush=True)
            VOLRAIN = mask_calc_volrain(mask_times,interval_hours,multiply_factor,AREA,mask_arrays,dataset_dict,nproc=nproc)


        ##########################################################
        ## Include some basic LPT info for user friendliness.  ###
        ##########################################################
        lptidx = [ii for ii in range(len(TC['lptid'])) if this_lpt_id == TC['lptid'][ii]][0]

        basic_lpt_info_field_list = ['centroid_lon','centroid_lat','area'
                    ,'largest_object_centroid_lon','largest_object_centroid_lat'
                    ,'max_filtered_running_field','max_running_field','max_inst_field'
                    ,'min_filtered_running_field','min_running_field','min_inst_field'
                    ,'amean_filtered_running_field','amean_running_field','amean_inst_field']

        # Time varying properties
        for var in basic_lpt_info_field_list:
            mask_arrays[var] = MISSING * np.ones(len(mask_times))

        for ttt in range(TC['i1'][lptidx],TC['i2'][lptidx]+1):
            this_time_indx = [ii for ii in range(len(mask_times)) if TC['datetime'][ttt] == mask_times[ii]]
            if len(this_time_indx) > 0:
                for var in basic_lpt_info_field_list:
                    mask_arrays[var][this_time_indx] = TC[var][ttt]

        # Bulk properties
        bulk_lpt_info_field_list = ['group','duration','maxarea',
            'zonal_propagation_speed','meridional_propagation_speed']
        for var in bulk_lpt_info_field_list:
            if var == 'group':
                mask_arrays[var] = this_lpt_group
            else:
                mask_arrays[var] = TC[var][this_lpt_idx]
        ##########################################################


        ##
        ## Output.
        ##

        ## Set coordinates
        coords_dict = {}
        coords_dict['n'] = (['n',], [1,])
        coords_dict['time'] = (['time',], mask_times)
        coords_dict['lon'] = (['lon',], mask_lon, {'units':'degrees_east'})
        coords_dict['lat'] = (['lat',], mask_lat, {'units':'degrees_north'})

        ## Set Data
        data_dict = {}
        data_dict['grid_area'] = (['lat','lon',], AREA, {'units':'km2','description':'Area of each grid cell.'})

        ## Basic LPT system track data for convenience
        for mask_var in basic_lpt_info_field_list:
            data_dict[mask_var] = (['time',], mask_arrays[mask_var])

        # Volumetric Rain, if specified.
        if do_volrain:
            fields = [*VOLRAIN]
            for field in fields:
                if 'tser' in field:
                    data_dict[field] = (['time',], VOLRAIN[field])
                else:
                    data_dict[field] = (['n',], [VOLRAIN[field],])

        for mask_var in bulk_lpt_info_field_list:
            print(mask_var, mask_arrays[mask_var])
            data_dict[mask_var] = (['n',], [mask_arrays[mask_var],])

        ## Create XArray Dataset
        DS = xr.Dataset(data_vars=data_dict, coords=coords_dict)
        DS.centroid_lon.attrs = {'units':'degrees_east','long_name':'centroid longitude (0-360)','standard_name':'longitude','note':'Time is end of running mean time.'}
        DS.centroid_lat.attrs = {'units':'degrees_north','long_name':'centroid latitude (-90-00)','standard_name':'latitude','note':'Time is end of running mean time.'}
        DS.largest_object_centroid_lon.attrs = {'units':'degrees_east','long_name':'centroid longitude (0-360)','standard_name':'longitude','note':'Time is end of running mean time.'}
        DS.largest_object_centroid_lat.attrs = {'units':'degrees_east','long_name':'centroid latitude (-90-00)','standard_name':'latitude','note':'Time is end of running mean time.'}
        DS.area.attrs = {'units':'km2','long_name':'LPT System enclosed area','note':'Time is end of running mean time.'}

        # Time varying fields
        for var in ['max_filtered_running_field','max_running_field','max_inst_field'
                    ,'min_filtered_running_field','min_running_field','min_inst_field'
                    ,'amean_filtered_running_field','amean_running_field','amean_inst_field']:
            DS[var].attrs = {'units':'mm day-1',
                    'long_name':'LP object running mean rain rate (at end of accum time).',
                    'note':'Time is end of running mean time. Based on mask_at_end_time'}

        DS.maxarea.attrs = {'units':'km2','long_name':'LPT System enclosed area','note':'This is the max over the LPT life time.'}
        DS.duration.attrs = {'units':'h','long_name':'LPT System duration'}
        DS.zonal_propagation_speed.attrs = {'units':'m s-1','long_name':'Zonal popagation speed','description':'Zonal popagation speed of the entire LPT system -- based on least squares fit of lon(time).'}
        DS.meridional_propagation_speed.attrs = {'units':'m s-1','long_name':'meridional popagation speed','description':'Meridional popagation speed of the entire LPT system -- based on least squares fit of lon(time).'}



        ## Write the data to NetCDF
        fn_out = (mask_output_dir + '/' + YMDH1_YMDH2 + '/lpt_system_mask_'+prod+'.lptid{0:05d}.group.nc'.format(this_lpt_group))
        os.makedirs(mask_output_dir + '/' + YMDH1_YMDH2, exist_ok=True)

        print('Writing to: ' + fn_out, flush=True)
        DS.to_netcdf(path=fn_out, mode='w', unlimited_dims=['time',], encoding={'time': {'dtype': 'i'}, 'n': {'dtype': 'i'}})
        

        ## Define the mask variables here.
        ## But don't assign them yet.
        with Dataset(fn_out, 'a') as DSnew:
            fields = [*mask_arrays]
            fields = [x for x in fields if 'mask' in x]
            for field in fields:
                dtype = 'i1'
                this_units = '1'
                DSnew.createVariable(field,dtype,('time','lat','lon'),zlib=True,complevel=4)
                DSnew[field].setncattr('units',this_units)

        ## Writing mask variables.
        ## Do them one at a time so it uses less memory while writing them.
        for field in fields:
            print('- ' + field)
            add_mask_var_to_netcdf(fn_out, field, mask_arrays[field]
                                , memory_target_mb = memory_target_mb)

        ## Include masked rain rates, if specified.
        if include_rain_rates:
            print('Adding masked rainfall.', flush=True)
            fields = [*mask_arrays]
            fields = [x for x in fields if 'mask' in x]
            dtype = 'float32'
            this_units = units

            for field in fields:
                new_field = field + '_with_rain'
                print(new_field)
                mask_array_with_rain = add_masked_rain_rates(mask_arrays[field], mask_times, multiply_factor, dataset_dict, nproc=nproc)

                with Dataset(fn_out, 'a') as DSnew:
                    DSnew.createVariable(new_field,dtype,('time','lat','lon'),zlib=True,complevel=4)
                    DSnew[new_field].setncattr('units',this_units)
                add_mask_var_to_netcdf(fn_out, new_field, mask_array_with_rain
                                    , memory_target_mb = memory_target_mb, data_mask = mask_arrays[field])

            print('Done adding masked rainfall.')





################################################################################
################################################################################
################################################################################
###########  calc_composite_lpt_mask ###########################################
################################################################################
################################################################################
################################################################################

def calc_composite_lpt_mask(dt_begin, dt_end, interval_hours, prod='trmm'
    , accumulation_hours = 0, filter_stdev = 0
    , lp_objects_dir = '.', lp_objects_fn_format='%Y/%m/%Y%m%d/objects_%Y%m%d%H.nc'
    , lpt_systems_dir = '.'
    , mask_output_dir = '.', verbose=True
    , do_volrain=False, include_rain_rates = False
    , dataset_dict = {}
    , detailed_output = False
    , cold_start_mode = False
    , calc_with_filter_radius = True
    , calc_with_accumulation_period = True
    , multiply_factor = 1.0
    , units = '1'
    , coarse_grid_factor = 0
    , subset='all'
    , memory_target_mb = 1000
    , nproc = 1):

    """
    dt_begin, dt_end: datetime objects for the first and last times. These are END of accumulation times!
    """


    def rain_read_function(dt, verbose=False):
        return lpt.readdata.read_generic_netcdf_at_datetime(dt, data_dir = rain_dir, verbose=verbose)

    YMDH1_YMDH2 = (dt_begin.strftime('%Y%m%d%H') + '_' + dt_end.strftime('%Y%m%d%H'))

    lpt_systems_file = (lpt_systems_dir + '/lpt_systems_'+prod+'_'+YMDH1_YMDH2+'.nc')
    lpt_group_file = (lpt_systems_dir + '/lpt_systems_'+prod+'_'+YMDH1_YMDH2+'.group_array.txt')


    MISSING = -999.0
    FILL_VALUE = MISSING


    dt_hours = interval_hours
    if accumulation_hours > 0 and calc_with_accumulation_period and not cold_start_mode: #Cold start mode doesn't have data before init time.
        dt00 = dt_begin - dt.timedelta(hours=accumulation_hours) #(dt_begin - dt.datetime(1970,1,1,0,0,0)).total_seconds()/3600.0 - accumulation_hours
    else:
        dt00 = dt_begin #(dt_begin - dt.datetime(1970,1,1,0,0,0)).total_seconds()/3600.0
    grand_mask_times = lpt.helpers.dtrange(dt00, dt_end + dt.timedelta(hours=int(dt_hours)), dt_hours)   # [dt.datetime(1970,1,1,0,0,0) + dt.timedelta(hours=x) for x in grand_mask_timestamps]


    ## Initialize the mask arrays dictionary if this is the first LP object.
    ## First, I need the grid information. Get this from the first LP object.
    fn = (lp_objects_dir + '/' + dt_begin.strftime(lp_objects_fn_format))
    with Dataset(fn) as DS:
        grand_mask_lon = DS['grid_lon'][:]
        grand_mask_lat = DS['grid_lat'][:]
        AREA = DS['grid_area'][:]

    mask_arrays = initialize_mask_arrays(
        len(grand_mask_lat), len(grand_mask_lon), len(grand_mask_times),
        detailed_output,
        calc_with_accumulation_period, accumulation_hours,
        calc_with_filter_radius)

    ## Read Stitched NetCDF data.
    TC = lpt.lptio.read_lpt_systems_netcdf(lpt_systems_file)
    
    unique_lpt_ids = np.unique(TC['lptid'])

    ############################################################################
    ## Get LPT list if it's MJO (subset='mjo') or non MJO (subset='non_mjo') case.
    ## Over-ride unique_lpt_ids so it only includes those LPTs.
    if subset == 'mjo':
        lpt_list_file = (lpt_systems_dir + '/mjo_lpt_list_'+prod+'_'+YMDH1_YMDH2+'.txt')
        try:
            lpt_list = np.loadtxt(lpt_list_file,skiprows=1)
            if len(lpt_list.shape) == 1:
                unique_lpt_ids = np.unique(lpt_list[2])
            else:
                unique_lpt_ids = np.unique(lpt_list[:,2])
        except:
            unique_lpt_ids = np.array([])

    if subset == 'non_mjo':
        lpt_list_file = (lpt_systems_dir + '/non_mjo_lpt_list_'+prod+'_'+YMDH1_YMDH2+'.txt')
        try:
            lpt_list = np.loadtxt(lpt_list_file,skiprows=1)
            if len(lpt_list.shape) == 1:
                unique_lpt_ids = np.unique(lpt_list[2])
            else:
                unique_lpt_ids = np.unique(lpt_list[:,2])
        except:
            unique_lpt_ids = np.array([])

    ############################################################################


    for this_lpt_id in unique_lpt_ids:
        print('Adding LPT system mask for lptid = ' + str(this_lpt_id) + ' (max = ' + str(np.max(unique_lpt_ids)) + ') of time period ' + YMDH1_YMDH2 + '.')

        ## Get list of LP Objects for this LPT system.
        this_lpt_idx = np.argwhere(np.round(10000*TC['lptid']) == np.round(10000*this_lpt_id))[0][0]
        lp_object_id_list = TC['objid'][this_lpt_idx,0:int(TC['num_objects'][this_lpt_idx])]

        dt00 = dt.datetime.strptime(str(int(np.min(lp_object_id_list))).zfill(14)[0:10],'%Y%m%d%H')
        dt0 = cftime.datetime(dt00.year,dt00.month,dt00.day,dt00.hour,calendar=TC['datetime'][0].calendar)
        dt11 = dt.datetime.strptime(str(int(np.max(lp_object_id_list))).zfill(14)[0:10],'%Y%m%d%H')
        dt1 = cftime.datetime(dt11.year,dt11.month,dt11.day,dt11.hour,calendar=TC['datetime'][0].calendar)
        duration_hours = int((dt1 - dt0).total_seconds()/3600)

        for lp_object_id in lp_object_id_list:

            nnnn = int(str(int(lp_object_id))[-4:])
            try:
                dt_this0 = lpt.helpers.get_objid_datetime(lp_object_id)
                dt_this = cftime.datetime(dt_this0.year,dt_this0.month,dt_this0.day,dt_this0.hour,calendar=TC['datetime'][0].calendar)
                dt_idx = [tt for tt in range(len(grand_mask_times)) if dt_this == grand_mask_times[tt]]
            except:
                continue

            if len(dt_idx) < 0:
                print('This time not found in mask time list. Skipping LP object id: ' + str(int(lp_object_id)))
                continue
            elif len(dt_idx) > 1:
                print('Found more than one mask time for this LP object. This should not happen! Skipping it.')
                continue
            else:
                dt_idx = dt_idx[0]

            fn = (lp_objects_dir + '/' + dt_this.strftime(lp_objects_fn_format))
            DS=Dataset(fn)

            ##
            ## Get LP Object pixel information.
            ##
            try:
                iii = DS['pixels_x'][nnnn,:].compressed()
                jjj = DS['pixels_y'][nnnn,:].compressed()
            except:
                DS.close()
                continue

            DS.close()

            ##
            ## Fill in the mask information.
            ##

            if detailed_output:
                ## For mask_at_end_time, just use the mask from the objects file.
                mask_arrays['mask_at_end_time'][dt_idx][jjj, iii] = 1

                ## For the mask with accumulation, go backwards and fill in ones.
                if accumulation_hours > 0 and calc_with_accumulation_period:
                    n_back = int(accumulation_hours/interval_hours)
                    for ttt in range(dt_idx - n_back, dt_idx+1):
                        mask_arrays['mask_with_accumulation'][ttt][jjj, iii] = 1

            else:
                # For mask > 1, apply accumulation hours, if specified.
                # This region is expanded below in "filter width spreading"
                # For the consolidated "mask" variable, I need to make sure
                # I don't set values of "2" to be "1" in future loop iterations.
                if accumulation_hours > 0:
                    n_back = int(accumulation_hours/interval_hours)
                    for ttt in range(dt_idx - n_back, dt_idx+1):
                        dummy = mask_arrays['mask'][ttt].copy()
                        dummy[jjj, iii] = 1
                        mask_arrays['mask'][ttt] = mask_arrays['mask'][ttt].maximum(dummy)

                # Set the "inner core" of the mask to 2.
                # (or 1, if no filter or accumulation)
                if accumulation_hours < 0.01 and filter_stdev < 0.01:
                    mask_arrays['mask'][dt_idx][jjj, iii] = 1
                else:
                    mask_arrays['mask'][dt_idx][jjj, iii] = 2


    ##
    ## Do filter width spreading.
    ##

    do_filter = False
    if type(filter_stdev) is list:
        if filter_stdev[0] > 0 and (calc_with_filter_radius or detailed_output):
            do_filter = True
    else:
        if filter_stdev > 0 and (calc_with_filter_radius or detailed_output):
            do_filter = True
    if do_filter:
        print('Filter width spreading...this may take awhile.', flush=True)

        if detailed_output:

            if coarse_grid_factor > 1:
                mask_arrays['mask_with_filter_at_end_time'] = feature_spread_reduce_res(mask_arrays['mask_at_end_time'], filter_stdev, coarse_grid_factor, nproc=nproc)
            else:
                mask_arrays['mask_with_filter_at_end_time'] = feature_spread(mask_arrays['mask_at_end_time'], filter_stdev, nproc=nproc)
            if accumulation_hours > 0 and calc_with_accumulation_period:
                if coarse_grid_factor > 1:
                    mask_arrays['mask_with_filter_and_accumulation'] = feature_spread_reduce_res(mask_arrays['mask_with_accumulation'], filter_stdev, coarse_grid_factor, nproc=nproc)
                else:
                    mask_arrays['mask_with_filter_and_accumulation'] = feature_spread(mask_arrays['mask_with_accumulation'], filter_stdev, nproc=nproc)

        else:
            # Only spread the "1" region. Not "2."
            if coarse_grid_factor > 1:
                mask_arrays['mask'] = feature_spread_reduce_res(
                    mask_arrays['mask'], filter_stdev,
                    reduce_res_factor=coarse_grid_factor,
                    spread_value=1, nproc=nproc)
            else:
                mask_arrays['mask'] = feature_spread(
                    mask_arrays['mask'], filter_stdev,
                    spread_value=1, nproc=nproc)

    ## Do volumetric rain.
    if do_volrain:
        print('Now calculating the volumetric rain.', flush=True)
        VOLRAIN = mask_calc_volrain(grand_mask_times,interval_hours,multiply_factor,AREA,mask_arrays,dataset_dict,nproc=nproc)


    ##
    ## Output.
    ##
    if subset == 'mjo':
        fn_out = (mask_output_dir+'/lpt_composite_mask_'+YMDH1_YMDH2+'_mjo_lpt.nc')
    elif subset == 'non_mjo':
        fn_out = (mask_output_dir+'/lpt_composite_mask_'+YMDH1_YMDH2+'_non_mjo_lpt.nc')
    else:
        fn_out = (mask_output_dir+'/lpt_composite_mask_'+YMDH1_YMDH2+'.nc')


    ## Set coordinates
    coords_dict = {}
    coords_dict['n'] = (['n',], [1,])
    coords_dict['time'] = (['time',], grand_mask_times)
    coords_dict['lon'] = (['lon',], grand_mask_lon, {'units':'degrees_east'})
    coords_dict['lat'] = (['lat',], grand_mask_lat, {'units':'degrees_north'})

    ## Set Data
    data_dict = {}
    data_dict['grid_area'] = (['lat','lon',], AREA, {'units':'km2','description':'Area of each grid cell.'})

    # Volumetric Rain, if specified.
    if do_volrain:
        fields = [*VOLRAIN]
        for field in fields:
            if 'tser' in field:
                data_dict[field] = (['time',], VOLRAIN[field])
            else:
                data_dict[field] = (['n',], [VOLRAIN[field],])

    DS = xr.Dataset(data_vars=data_dict, coords=coords_dict)

    ## Write the data to NetCDF
    os.makedirs(mask_output_dir + '/' + YMDH1_YMDH2, exist_ok=True)

    print('Writing to: ' + fn_out, flush=True)
    DS.to_netcdf(path=fn_out, mode='w', unlimited_dims=['time',], encoding={'time': {'dtype': 'i'}, 'n': {'dtype': 'i'}})
    

    ## Define the mask variables here.
    ## But don't assign them yet.
    with Dataset(fn_out, 'a') as DSnew:
        fields = [*mask_arrays]
        fields = [x for x in fields if 'mask' in x]
        for field in fields:
            dtype = 'i1'
            this_units = '1'
            DSnew.createVariable(field,dtype,('time','lat','lon'),zlib=True,complevel=4)
            DSnew[field].setncattr('units',this_units)

    ## Writing mask variables.
    ## Do them one at a time so it uses less memory while writing them.
    for field in fields:
        print('- ' + field)
        add_mask_var_to_netcdf(fn_out, field, mask_arrays[field]
                            , memory_target_mb = memory_target_mb)

    ## Include masked rain rates, if specified.
    if include_rain_rates:
        print('Adding masked rainfall.', flush=True)
        fields = [*mask_arrays]
        fields = [x for x in fields if 'mask' in x]
        dtype = 'float32'
        this_units = units

        for field in fields:
            new_field = field + '_with_rain'
            print(new_field)
            mask_array_with_rain = add_masked_rain_rates(mask_arrays[field], grand_mask_times, multiply_factor, dataset_dict, nproc=nproc)

            with Dataset(fn_out, 'a') as DSnew:
                DSnew.createVariable(new_field,dtype,('time','lat','lon'),zlib=True,complevel=4)
                DSnew[new_field].setncattr('units',this_units)
            add_mask_var_to_netcdf(fn_out, new_field, mask_array_with_rain
                                , memory_target_mb = memory_target_mb, data_mask = mask_arrays[field])

        print('Done adding masked rainfall.')

