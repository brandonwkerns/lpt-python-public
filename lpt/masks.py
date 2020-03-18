## This file contains the content for LPO, LPT, and LPT "grand master" masks as functions, to be used with lpt-python-public.

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import datetime as dt
from dateutil.relativedelta import relativedelta
from context import lpt
import os
import sys
from scipy.sparse import dok_matrix, csr_matrix, find, SparseEfficiencyWarning
import warnings
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


##
## feature spread function -- used for all of the mask functions.
##

def feature_spread_reduce_res(data, npoints, reduce_res_factor=5):
    from scipy.interpolate import RegularGridInterpolator
    ## Use the binary dilation technique to expand the mask "array_in" a radius of np points.
    ## For this purpose, it takes a 3-D array with the first entry being time.
    ##
    ## In this version of the feature_spread function,
    ## use a reduced resolution grid then interp back to the original resolution.

    print('Feature spread with reduce_res_factor = {}'.format(reduce_res_factor))

    data_new = data.copy()

    start_idx = int(reduce_res_factor/2) # Try to get near the middle of reduce_res_factor
    S = data_new[0].shape
    X = np.arange(S[1])
    Y = np.arange(S[0])
    X2,Y2 = np.meshgrid(X,Y)
    pts = np.transpose([Y2.flatten().tolist(),X2.flatten().tolist()])
    X_reduced = X[start_idx::reduce_res_factor]
    Y_reduced = Y[start_idx::reduce_res_factor]

    if type(npoints) is list:
        npx = int(npoints[0]/reduce_res_factor)
        npy = int(npoints[1]/reduce_res_factor)
    else:
        npx = int(npoints/reduce_res_factor)
        npy = int(npoints/reduce_res_factor)

    [circle_array_x, circle_array_y] = np.meshgrid(np.arange(-1*npx,npx+1), np.arange(-1*npy,npy+1))
    circle_array_dist = np.sqrt(np.power(circle_array_x,2) + np.power(circle_array_y * (npx/npy),2))
    circle_array_mask = (circle_array_dist < (npx + 0.1)).astype(np.double)
    circle_array_mask = circle_array_mask / np.sum(circle_array_mask)


    for tt in range(len(data)):

        if tt % 100 == 0:
            print(('Feature Spread: ' + str(tt) + ' of max ' + str(len(data)) + '.'),flush=True)

        array_2d = data_new[tt].toarray()
        array_2d_new = array_2d.copy()
        array_2d_reduced = array_2d[start_idx::reduce_res_factor,start_idx::reduce_res_factor]
        array_2d_new_reduced = array_2d_reduced.copy()

        ## Loop over the times.
        ## For each time, use the convolution to "spread out" the effect of each time's field.
        ## (I tried using 3-D convolution here, but it took almost twice as much memory
        ##  and was slightly SLOWER than this method.)
        unique_values = np.unique(array_2d_reduced)
        unique_values = unique_values[unique_values > 0]  #take out zero -- it is not a feature.
        for this_value in unique_values:
            starting_mask = (array_2d_reduced == this_value).astype(np.double)
            starting_mask_spread = scipy.ndimage.binary_dilation(starting_mask,structure=circle_array_mask, iterations=1, mask=starting_mask < 0.1)
            F = RegularGridInterpolator((Y_reduced, X_reduced,), starting_mask_spread, method='nearest',bounds_error=False)
            starting_mask_spread2 = F(pts).reshape(S)
            array_2d_new[starting_mask_spread2 > 0.0001] = this_value

        data_new[tt] = csr_matrix(array_2d_new)

    return data_new



def feature_spread(data, npoints):

    ## Use the binary dilation technique to expand the mask "array_in" a radius of np points.
    ## For this purpose, it takes a 3-D array with the first entry being time.

    data_new = data.copy()

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


    for tt in range(len(data)):

        if tt % 100 == 0:
            print(('Feature Spread: ' + str(tt) + ' of max ' + str(len(data)) + '.'),flush=True)

        array_2d = data[tt].toarray()

        array_2d_new = array_2d.copy()

        ## Loop over the times.
        ## For each time, use the convolution to "spread out" the effect of each time's field.
        ## (I tried using 3-D convolution here, but it took almost twice as much memory
        ##  and was slightly SLOWER than this method.)
        unique_values = np.unique(array_2d)
        unique_values = unique_values[unique_values > 0]  #take out zero -- it is not a feature.
        for this_value in unique_values:
            starting_mask = (array_2d == this_value).astype(np.double)
            starting_mask_spread = scipy.ndimage.binary_dilation(starting_mask,structure=circle_array_mask, iterations=1, mask=starting_mask < 0.1)
            array_2d_new[starting_mask_spread > 0.001] = this_value

        data_new[tt] = csr_matrix(array_2d_new)

    return data_new



def mask_calc_volrain(mask_times,interval_hours,AREA,mask_arrays, dataset_dict):

    mask_type_list = [*mask_arrays]
    mask_type_list = [x for x in mask_type_list if 'mask' in x]

    #### Initialize
    VOLRAIN = {}

    ## Global
    VOLRAIN['volrain_global'] = 0.0
    VOLRAIN['volrain_global_tser'] = np.nan * np.zeros(len(mask_times))

    ## Masked
    for field in mask_type_list:
        VOLRAIN[field.replace('mask','volrain')] = 0.0
        VOLRAIN[field.replace('mask','volrain')+'_tser'] = np.nan * np.zeros(len(mask_times))

    #### Fill in values by time. Multiply rain field by applicable mask (0 and 1 values).
    for tt in range(len(mask_times)):
        this_dt = mask_times[tt]

        ## Get rain
        RAIN = lpt.readdata.readdata(this_dt, dataset_dict)

        precip = RAIN['data'][:]
        precip[~np.isfinite(precip)] = 0.0
        precip[precip < -0.01] = 0.0

        ## Global
        VOLRAIN['volrain_global'] += interval_hours * np.sum(precip * AREA)
        VOLRAIN['volrain_global_tser'][tt] = interval_hours * np.sum(precip * AREA)

        ## Masked
        for field in mask_type_list:
            this_mask = mask_arrays[field][tt].toarray()
            this_mask[this_mask > 0] = 1.0
            precip_masked = precip * this_mask
            VOLRAIN[field.replace('mask','volrain')] += interval_hours * np.sum(precip_masked * AREA)
            VOLRAIN[field.replace('mask','volrain')+'_tser'][tt] = interval_hours * np.sum(precip_masked * AREA)

    return VOLRAIN


def add_mask_var_to_netcdf(fn, mask_var, data, memory_target_mb = 1000):
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
        DS = Dataset(fn, 'r+')
        tt2 = min(tt1 + batch_n_time, len(data))
        DS[mask_var][tt1:tt2,:,:] = np.array([data[tt].toarray() for tt in range(tt1,tt2)], dtype='bool_')
        DS.close()


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
    , do_volrain=False, dataset_dict = {}
    , calc_with_filter_radius = True
    , calc_with_accumulation_period = True
    , cold_start_mode = False
    , coarse_grid_factor = 0
    , memory_target_mb = 1000):

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
    mask_times = [dt0 + dt.timedelta(hours=x) for x in range(0,duration_hours+1,interval_hours)]

    mask_arrays={} #Start with empty dictionary

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
        if len(mask_arrays) < 1:
            lon = DS['grid_lon'][:]
            lat = DS['grid_lat'][:]
            AREA = DS['grid_area'][:]
            mask_arrays['lon'] = DS['grid_lon'][:]
            mask_arrays['lat'] = DS['grid_lat'][:]

            mask_arrays_shape2d = (len(lat), len(lon))
            mask_arrays['mask_at_end_time'] = [csr_matrix(mask_arrays_shape2d, dtype=np.bool_) for x in range(len(mask_times))]
            if accumulation_hours > 0 and calc_with_accumulation_period:
                mask_arrays['mask_with_accumulation'] = [csr_matrix(mask_arrays_shape2d, dtype=np.bool_) for x in range(len(mask_times))]
            if calc_with_filter_radius:
                mask_arrays['mask_with_filter_at_end_time'] = [csr_matrix(mask_arrays_shape2d, dtype=np.bool_) for x in range(len(mask_times))]
                if accumulation_hours > 0 and calc_with_accumulation_period:
                    mask_arrays['mask_with_filter_and_accumulation'] = [csr_matrix(mask_arrays_shape2d, dtype=np.bool_) for x in range(len(mask_times))]

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

            ## For mask_at_end_time, just use the mask from the objects file.
            mask_arrays['mask_at_end_time'][dt_idx][jjj, iii] = 1

            ## For the mask with accumulation, go backwards and fill in ones.
            if accumulation_hours > 0 and calc_with_accumulation_period:
                n_back = int(accumulation_hours/interval_hours)
                for ttt in range(dt_idx - n_back, dt_idx+1):
                    mask_arrays['mask_with_accumulation'][ttt][jjj, iii] = 1

        except:
            DS.close()

    ##
    ## Do filter width spreading.
    ##

    do_filter = False
    if type(filter_stdev) is list:
        if filter_stdev[0] > 0 and calc_with_filter_radius:
            do_filter = True
    else:
        if filter_stdev > 0 and calc_with_filter_radius:
            do_filter = True

    if do_filter:
        print('Filter width spreading...this may take awhile.', flush=True)
        if coarse_grid_factor > 1:
            mask_arrays['mask_with_filter_at_end_time'] = feature_spread_reduce_res(mask_arrays['mask_at_end_time'], filter_stdev, coarse_grid_factor)
        else:
            mask_arrays['mask_with_filter_at_end_time'] = feature_spread(mask_arrays['mask_at_end_time'], filter_stdev)
        if accumulation_hours > 0 and calc_with_accumulation_period:
            if coarse_grid_factor > 1:
                mask_arrays['mask_with_filter_and_accumulation'] = feature_spread_reduce_res(mask_arrays['mask_with_accumulation'], filter_stdev, coarse_grid_factor)
            else:
                mask_arrays['mask_with_filter_and_accumulation'] = feature_spread(mask_arrays['mask_with_accumulation'], filter_stdev)

    ## Do volumetric rain.
    if do_volrain:
        print('Now calculating the volumetric rain.', flush=True)
        VOLRAIN = mask_calc_volrain(mask_times,interval_hours,AREA,mask_arrays,dataset_dict)

    ##
    ## Output.
    ##
    fn_out = (mask_output_dir + '/lp_objects_mask_' + YMDH1_YMDH2 + '.nc').replace('///','/').replace('//','/')
    os.makedirs(mask_output_dir, exist_ok=True)

    print('Writing to: ' + fn_out, flush=True)
    DSnew = Dataset(fn_out, 'w')
    DSnew.createDimension('n', 1)
    DSnew.createDimension('time',len(mask_times))
    DSnew.createDimension('lon',len(lon))
    DSnew.createDimension('lat',len(lat))

    DSnew.createVariable('time','d',('time',))
    DSnew.createVariable('lon','d',('lon',))
    DSnew.createVariable('lat','d',('lat',))
    DSnew.createVariable('grid_area','f4',('lat','lon'))

    DSnew['time'][:] = [(x - dt.datetime(1970,1,1,0,0,0)).total_seconds()/3600.0 for x in mask_times]
    DSnew['time'].setncattr('units','hours since 1970-1-1 0:0:0')
    DSnew['lon'][:] = lon
    DSnew['lon'].setncattr('units','degrees_east')
    DSnew['lat'][:] = lat
    DSnew['lat'].setncattr('units','degrees_north')

    DSnew['grid_area'][:] = AREA
    DSnew['grid_area'].setncattr('units','km2')
    DSnew['grid_area'].setncattr('description','Area of each grid cell.')

    ## Write volrain if it was calculated.
    if do_volrain:
        fields = [*VOLRAIN]
        fields = [x for x in fields if not 'tser' in x]
        for field in fields:
            add_volrain_to_netcdf(DSnew, field, VOLRAIN[field]
                    , field+'_tser', VOLRAIN[field+'_tser']
                    , fill_value = FILL_VALUE)

    ## Define the mask variables here.
    ## But don't assign them yet.
    fields = [*mask_arrays]
    fields = [x for x in fields if 'mask' in x]
    for field in fields:
        DSnew.createVariable(field,'i1',('time','lat','lon'),zlib=True,complevel=4)
        DSnew[field].setncattr('units','1')
    DSnew.close()

    ## Writing mask variables.
    ## Do them one at a time so it uses less memory while writing them.
    for field in fields:
        print('- ' + field)
        add_mask_var_to_netcdf(fn_out, field, mask_arrays[field]
                            , memory_target_mb = memory_target_mb)

################################################################################
################################################################################
################################################################################
###########  calc_individual_lpt_masks #########################################
################################################################################
################################################################################
################################################################################

def calc_individual_lpt_masks(dt_begin, dt_end, interval_hours, prod='trmm'
    ,accumulation_hours = 0, filter_stdev = 0
    , lp_objects_dir = '.', lp_objects_fn_format='objects_%Y%m%d%H.nc'
    , lpt_systems_dir = '.'
    , mask_output_dir = '.', verbose=True
    , do_volrain=False, dataset_dict = {}
    , calc_with_filter_radius = True
    , calc_with_accumulation_period = True
    , cold_start_mode = False
    , begin_lptid = 0, end_lptid = 10000, mjo_only = False
    , coarse_grid_factor = 0
    , memory_target_mb = 1000):

    """
    dt_begin, dt_end: datetime objects for the first and last times. These are END of accumulation times!
    """
    YMDH1_YMDH2 = (dt_begin.strftime('%Y%m%d%H') + '_' + dt_end.strftime('%Y%m%d%H'))

    lpt_systems_file = (lpt_systems_dir + '/lpt_systems_'+prod+'_'+YMDH1_YMDH2+'.nc').replace('///','/').replace('//','/')
    lpt_group_file = (lpt_systems_dir + '/lpt_systems_'+prod+'_'+YMDH1_YMDH2+'.group_array.txt').replace('///','/').replace('//','/')

    MISSING = -999.0
    FILL_VALUE = MISSING

    ## Read Stitched data.
    DS = Dataset(lpt_systems_file)
    TC={}
    TC['lptid'] = DS['lptid'][:]
    TC['objid'] = DS['objid'][:]
    TC['num_objects'] = DS['num_objects'][:]
    TC['i1'] = DS['lpt_begin_index'][:]
    TC['i2'] = DS['lpt_end_index'][:]
    TC['timestamp_stitched'] = DS['timestamp_stitched'][:]
    TC['datetime'] = [dt.datetime(1970,1,1,0,0,0) + dt.timedelta(hours=int(x)) if x > 100 else None for x in TC['timestamp_stitched']]
    TC['centroid_lon'] = DS['centroid_lon_stitched'][:]
    TC['centroid_lat'] = DS['centroid_lat_stitched'][:]
    TC['largest_object_centroid_lon'] = DS['largest_object_centroid_lon_stitched'][:]
    TC['largest_object_centroid_lat'] = DS['largest_object_centroid_lat_stitched'][:]
    TC['area'] = DS['area_stitched'][:]
    for var in ['max_filtered_running_field','max_running_field','max_inst_field'
                ,'min_filtered_running_field','min_running_field','min_inst_field'
                ,'amean_filtered_running_field','amean_running_field','amean_inst_field'
                ,'duration','maxarea','zonal_propagation_speed','meridional_propagation_speed']:
        TC[var] = DS[var][:]
    DS.close()


    F = Dataset(lpt_systems_file)
    unique_lpt_ids = np.unique(F['lptid'][:]).data
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

    for this_lpt_id in unique_lpt_ids:
        if int(np.floor(this_lpt_id)) < int(np.floor(begin_lptid)) or int(np.floor(this_lpt_id)) > int(np.floor(end_lptid)):
            continue
        if mjo_only and np.min(np.abs(mjo_lpt_list - this_lpt_id)) > 0.0001:
            continue
        print('Calculating LPT system mask for lptid = ' + str(this_lpt_id) + ' of time period ' + YMDH1_YMDH2 + '.')

        ## Get list of LP Objects for this LPT system.
        this_lpt_idx = np.argwhere(TC['lptid'] == this_lpt_id)[0][0]
        print((this_lpt_idx,TC['num_objects'][this_lpt_idx]))
        lp_object_id_list = TC['objid'][this_lpt_idx,0:int(TC['num_objects'][this_lpt_idx])]

        if accumulation_hours > 0 and calc_with_accumulation_period and not cold_start_mode: #Cold start mode doesn't have data before init time.
            dt0 = dt.datetime.strptime(str(int(np.min(lp_object_id_list)))[0:10],'%Y%m%d%H') - dt.timedelta(hours=accumulation_hours)
        else:
            dt0 = dt.datetime.strptime(str(int(np.min(lp_object_id_list)))[0:10],'%Y%m%d%H')
        dt1 = dt.datetime.strptime(str(int(np.max(lp_object_id_list)))[0:10],'%Y%m%d%H')
        duration_hours = int((dt1 - dt0).total_seconds()/3600)
        mask_times = [dt0 + dt.timedelta(hours=x) for x in range(0,duration_hours+1,interval_hours)]
        mask_arrays={} #Start with empty dictionary


        ## Include some basic LPT info for user friendliness.
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
        for var in ['duration','maxarea','zonal_propagation_speed','meridional_propagation_speed']:
            mask_arrays[var] = TC[var][0]
        mask_arrays['volrain'] = MISSING
        mask_arrays['volrain_global'] = MISSING

        for lp_object_id in lp_object_id_list:

            nnnn = int(str(int(lp_object_id))[-4:])
            try:
                dt_this = lpt.helpers.get_objid_datetime(lp_object_id) # dt.datetime.strptime(str(int(lp_object_id))[0:10],'%Y%m%d%H')
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
            if not 'lon' in mask_arrays:
                lon = DS['grid_lon'][:]
                lat = DS['grid_lat'][:]
                AREA = DS['grid_area'][:]
                mask_arrays['lon'] = DS['grid_lon'][:]
                mask_arrays['lat'] = DS['grid_lat'][:]

                mask_arrays_shape2d = (len(lat), len(lon))
                mask_arrays['mask_at_end_time'] = [csr_matrix(mask_arrays_shape2d, dtype=np.bool_) for x in range(len(mask_times))]
                if accumulation_hours > 0 and calc_with_accumulation_period:
                    mask_arrays['mask_with_accumulation'] = [csr_matrix(mask_arrays_shape2d, dtype=np.bool_) for x in range(len(mask_times))]
                if calc_with_filter_radius:
                    mask_arrays['mask_with_filter_at_end_time'] = [csr_matrix(mask_arrays_shape2d, dtype=np.bool_) for x in range(len(mask_times))]
                    if accumulation_hours > 0 and calc_with_accumulation_period:
                        mask_arrays['mask_with_filter_and_accumulation'] = [csr_matrix(mask_arrays_shape2d, dtype=np.bool_) for x in range(len(mask_times))]

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
            mask_arrays['mask_at_end_time'][dt_idx][jjj, iii] = 1

            ## For the mask with accumulation, go backwards and fill in ones.
            if accumulation_hours > 0 and calc_with_accumulation_period:
                n_back = int(accumulation_hours/interval_hours)
                for ttt in range(dt_idx - n_back, dt_idx+1):
                    mask_arrays['mask_with_accumulation'][ttt][jjj, iii] = 1

        ##
        ## Do filter width spreading.
        ##

        do_filter = False
        if type(filter_stdev) is list:
            if filter_stdev[0] > 0 and calc_with_filter_radius:
                do_filter = True
        else:
            if filter_stdev > 0 and calc_with_filter_radius:
                do_filter = True
        if do_filter:
            print('Filter width spreading...this may take awhile.', flush=True)
            if coarse_grid_factor > 1:
                mask_arrays['mask_with_filter_at_end_time'] = feature_spread_reduce_res(mask_arrays['mask_at_end_time'], filter_stdev, coarse_grid_factor)
            else:
                mask_arrays['mask_with_filter_at_end_time'] = feature_spread(mask_arrays['mask_at_end_time'], filter_stdev)
            if accumulation_hours > 0 and calc_with_accumulation_period:
                if coarse_grid_factor > 1:
                    mask_arrays['mask_with_filter_and_accumulation'] = feature_spread_reduce_res(mask_arrays['mask_with_accumulation'], filter_stdev, coarse_grid_factor)
                else:
                    mask_arrays['mask_with_filter_and_accumulation'] = feature_spread(mask_arrays['mask_with_accumulation'], filter_stdev)

        ## Do volumetric rain.
        if do_volrain:
            print('Now calculating the volumetric rain.', flush=True)
            VOLRAIN = mask_calc_volrain(mask_times,interval_hours,AREA,mask_arrays,dataset_dict)


        ##
        ## Output.
        ##
        os.makedirs(mask_output_dir + '/' +YMDH1_YMDH2, exist_ok=True)
        fn_out = (mask_output_dir + '/' + YMDH1_YMDH2 + '/lpt_system_mask_'+prod+'.lptid{0:010.4f}.nc'.format(this_lpt_id))

        os.remove(fn_out) if os.path.exists(fn_out) else None
        print('Writing to: ' + fn_out, flush=True)
        DSnew = Dataset(fn_out, 'w', data_model='NETCDF4', clobber=True)
        DSnew.createDimension('n', 1)
        DSnew.createDimension('time',len(mask_times))
        DSnew.createDimension('lon',len(lon))
        DSnew.createDimension('lat',len(lat))
        DSnew.createVariable('lon','f4',('lon',))
        DSnew.createVariable('lat','f4',('lat',))
        DSnew.createVariable('grid_area','f4',('lat','lon'))
        DSnew.createVariable('time','d',('time',))
        DSnew.createVariable('centroid_lon','f4',('time',),fill_value=FILL_VALUE)
        DSnew.createVariable('centroid_lat','f4',('time',),fill_value=FILL_VALUE)
        DSnew.createVariable('largest_object_centroid_lon','f4',('time',),fill_value=FILL_VALUE)
        DSnew.createVariable('largest_object_centroid_lat','f4',('time',),fill_value=FILL_VALUE)
        DSnew.createVariable('area','d',('time',),fill_value=FILL_VALUE)

        # Time varying fields
        for var in ['max_filtered_running_field','max_running_field','max_inst_field'
                    ,'min_filtered_running_field','min_running_field','min_inst_field'
                    ,'amean_filtered_running_field','amean_running_field','amean_inst_field']:
            DSnew.createVariable(var,'f4',('time',),fill_value=FILL_VALUE)

        # Bulk fields.
        for var in ['duration','maxarea','zonal_propagation_speed','meridional_propagation_speed']:
            DSnew.createVariable(var,'f4',('n',),fill_value=FILL_VALUE)

        ts = [(x - dt.datetime(1970,1,1,0,0,0)).total_seconds()/3600.0 for x in mask_times]
        DSnew['time'][:] = ts
        DSnew['time'].setncattr('units','hours since 1970-1-1 0:0:0')
        DSnew['lon'][:] = lon
        DSnew['lon'].setncattr('units','degrees_east')
        DSnew['lat'][:] = lat
        DSnew['lat'].setncattr('units','degrees_north')

        for mask_var in basic_lpt_info_field_list:
            DSnew[mask_var][:] = mask_arrays[mask_var]

        for mask_var in ['duration','maxarea','zonal_propagation_speed','meridional_propagation_speed']:
            DSnew[mask_var][:] = mask_arrays[mask_var]

        DSnew['centroid_lon'].setncatts({'units':'degrees_east','long_name':'centroid longitude (0-360)','standard_name':'longitude','note':'Time is end of running mean time.'})
        DSnew['centroid_lat'].setncatts({'units':'degrees_east','long_name':'centroid latitude (-90-00)','standard_name':'latitude','note':'Time is end of running mean time.'})
        DSnew['largest_object_centroid_lon'].setncatts({'units':'degrees_east','long_name':'centroid longitude (0-360)','standard_name':'longitude','note':'Time is end of running mean time.'})
        DSnew['largest_object_centroid_lat'].setncatts({'units':'degrees_east','long_name':'centroid latitude (-90-00)','standard_name':'latitude','note':'Time is end of running mean time.'})
        DSnew['area'].setncatts({'units':'km2','long_name':'LPT System enclosed area','note':'Time is end of running mean time.'})
        DSnew['maxarea'].setncatts({'units':'km2','long_name':'LPT System enclosed area','note':'This is the max over the LPT life time.'})
        DSnew['duration'].setncatts({'units':'h','long_name':'LPT System duration'})
        DSnew['zonal_propagation_speed'].setncatts({'units':'m s-1','long_name':'Zonal popagation speed','description':'Zonal popagation speed of the entire LPT system -- based on least squares fit of lon(time).'})
        DSnew['meridional_propagation_speed'].setncatts({'units':'m s-1','long_name':'meridional popagation speed','description':'Meridional popagation speed of the entire LPT system -- based on least squares fit of lon(time).'})

        for var in ['max_filtered_running_field','max_running_field','max_inst_field'
                    ,'min_filtered_running_field','min_running_field','min_inst_field'
                    ,'amean_filtered_running_field','amean_running_field','amean_inst_field']:
            DSnew[var].setncatts({'units':'mm day-1','long_name':'LP object running mean rain rate (at end of accum time).','note':'Time is end of running mean time. Based on mask_at_end_time'})

        DSnew['grid_area'][:] = AREA
        DSnew['grid_area'].setncattr('units','km2')
        DSnew['grid_area'].setncattr('description','Area of each grid cell.')

        ## Write volrain if it was calculated.
        if do_volrain:
            fields = [*VOLRAIN]
            fields = [x for x in fields if not 'tser' in x]
            for field in fields:
                add_volrain_to_netcdf(DSnew, field, VOLRAIN[field]
                        , field+'_tser', VOLRAIN[field+'_tser']
                        , fill_value = FILL_VALUE)

        ## Define the mask variables here.
        ## But don't assign them yet.
        fields = [*mask_arrays]
        fields = [x for x in fields if 'mask' in x]
        for field in fields:
            DSnew.createVariable(field,'i1',('time','lat','lon'),zlib=True,complevel=4)
            DSnew[field].setncattr('units','1')
        DSnew.close()

        ## Writing mask variables.
        ## Do them one at a time so it uses less memory while writing them.
        for field in fields:
            print('- ' + field)
            add_mask_var_to_netcdf(fn_out, field, mask_arrays[field]
                                , memory_target_mb = memory_target_mb)


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
    , do_volrain=False, dataset_dict = {}
    , cold_start_mode = False
    , calc_with_filter_radius = True
    , calc_with_accumulation_period = True
    , coarse_grid_factor = 0
    , subset='all', memory_target_mb = 1000):

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
        grand_mask_timestamps0 = (dt_begin - dt.datetime(1970,1,1,0,0,0)).total_seconds()/3600.0 - accumulation_hours
    else:
        grand_mask_timestamps0 = (dt_begin - dt.datetime(1970,1,1,0,0,0)).total_seconds()/3600.0
    grand_mask_timestamps1 = (dt_end - dt.datetime(1970,1,1,0,0,0)).total_seconds()/3600.0

    grand_mask_timestamps = np.arange(grand_mask_timestamps0, grand_mask_timestamps1 + dt_hours, dt_hours)
    grand_mask_times = [dt.datetime(1970,1,1,0,0,0) + dt.timedelta(hours=x) for x in grand_mask_timestamps]
    mask_arrays = None

    ## Read Stitched NetCDF data.
    DS = Dataset(lpt_systems_file)
    TC={}
    TC['lptid'] = DS['lptid'][:]
    TC['objid'] = DS['objid'][:]
    TC['num_objects'] = DS['num_objects'][:]
    TC['i1'] = DS['lpt_begin_index'][:]
    TC['i2'] = DS['lpt_end_index'][:]
    TC['timestamp_stitched'] = DS['timestamp_stitched'][:]
    TC['datetime'] = [dt.datetime(1970,1,1,0,0,0) + dt.timedelta(hours=int(x)) if x > 100 else None for x in TC['timestamp_stitched']]
    TC['centroid_lon'] = DS['centroid_lon_stitched'][:]
    TC['centroid_lat'] = DS['centroid_lat_stitched'][:]
    TC['area'] = DS['area_stitched'][:]
    for var in ['max_filtered_running_field','max_running_field','max_inst_field'
                ,'min_filtered_running_field','min_running_field','min_inst_field'
                ,'amean_filtered_running_field','amean_running_field','amean_inst_field'
                ,'duration','maxarea','zonal_propagation_speed','meridional_propagation_speed']:
        TC[var] = DS[var][:]
    DS.close()

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
        this_lpt_idx = np.argwhere(TC['lptid'] == this_lpt_id)[0][0]
        lp_object_id_list = TC['objid'][this_lpt_idx,0:int(TC['num_objects'][this_lpt_idx])]

        dt0 = dt.datetime.strptime(str(int(np.min(lp_object_id_list)))[0:10],'%Y%m%d%H') - dt.timedelta(hours=accumulation_hours)
        dt1 = dt.datetime.strptime(str(int(np.max(lp_object_id_list)))[0:10],'%Y%m%d%H')
        duration_hours = int((dt1 - dt0).total_seconds()/3600)
        mask_times = [dt0 + dt.timedelta(hours=x) for x in range(0,duration_hours+1,interval_hours)]

        for lp_object_id in lp_object_id_list:

            nnnn = int(str(int(lp_object_id))[-4:])
            try:
                dt_this = lpt.helpers.get_objid_datetime(lp_object_id) # dt.datetime.strptime(str(int(lp_object_id))[0:10],'%Y%m%d%H')
                timestamp_this = (dt_this - dt.datetime(1970,1,1,0,0,0)).total_seconds()/3600.0
                dt_idx = [tt for tt in range(len(grand_mask_timestamps)) if timestamp_this == grand_mask_timestamps[tt]]
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

            ## Initialize the mask arrays dictionary if this is the first LP object.
            ## First, I need the grid information. Get this from the first LP object.
            if mask_arrays is None:
                grand_mask_lon = DS['grid_lon'][:]
                grand_mask_lat = DS['grid_lat'][:]
                AREA = DS['grid_area'][:]
                mask_arrays = {}

                mask_arrays_shape2d = (len(grand_mask_lat), len(grand_mask_lon))
                mask_arrays['mask_at_end_time'] = [csr_matrix(mask_arrays_shape2d, dtype=np.bool_) for x in range(len(grand_mask_timestamps))]
                if accumulation_hours > 0 and calc_with_accumulation_period:
                    mask_arrays['mask_with_accumulation'] = [csr_matrix(mask_arrays_shape2d, dtype=np.bool_) for x in range(len(grand_mask_timestamps))]
                if calc_with_filter_radius:
                    mask_arrays['mask_with_filter_at_end_time'] = [csr_matrix(mask_arrays_shape2d, dtype=np.bool_) for x in range(len(grand_mask_timestamps))]
                    if accumulation_hours > 0 and calc_with_accumulation_period:
                        mask_arrays['mask_with_filter_and_accumulation'] = [csr_matrix(mask_arrays_shape2d, dtype=np.bool_) for x in range(len(grand_mask_timestamps))]


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
            mask_arrays['mask_at_end_time'][dt_idx][jjj, iii] = 1

            ## For the mask with accumulation, go backwards and fill in ones.
            if accumulation_hours > 0 and calc_with_accumulation_period:
                n_back = int(accumulation_hours/interval_hours)
                for ttt in range(dt_idx - n_back, dt_idx+1):
                    mask_arrays['mask_with_accumulation'][ttt][jjj, iii] = 1

    ##
    ## Do filter width spreading.
    ##

    do_filter = False
    if type(filter_stdev) is list:
        if filter_stdev[0] > 0 and calc_with_filter_radius:
            do_filter = True
    else:
        if filter_stdev > 0 and calc_with_filter_radius:
            do_filter = True
    if do_filter:
        print('Filter width spreading...this may take awhile.', flush=True)
        if coarse_grid_factor > 1:
            mask_arrays['mask_with_filter_at_end_time'] = feature_spread_reduce_res(mask_arrays['mask_at_end_time'], filter_stdev, coarse_grid_factor)
        else:
            mask_arrays['mask_with_filter_at_end_time'] = feature_spread(mask_arrays['mask_at_end_time'], filter_stdev)
        if accumulation_hours > 0 and calc_with_accumulation_period:
            if coarse_grid_factor > 1:
                mask_arrays['mask_with_filter_and_accumulation'] = feature_spread_reduce_res(mask_arrays['mask_with_accumulation'], filter_stdev, coarse_grid_factor)
            else:
                mask_arrays['mask_with_filter_and_accumulation'] = feature_spread(mask_arrays['mask_with_accumulation'], filter_stdev)

    ## Do volumetric rain.
    if do_volrain:
        print('Now calculating the volumetric rain.', flush=True)
        VOLRAIN = mask_calc_volrain(grand_mask_times,interval_hours,AREA,mask_arrays,dataset_dict)

    ##
    ## Output.
    ##
    if subset == 'mjo':
        fn_out = (mask_output_dir+'/lpt_composite_mask_'+YMDH1_YMDH2+'_mjo_lpt.nc')
    elif subset == 'non_mjo':
        fn_out = (mask_output_dir+'/lpt_composite_mask_'+YMDH1_YMDH2+'_non_mjo_lpt.nc')
    else:
        fn_out = (mask_output_dir+'/lpt_composite_mask_'+YMDH1_YMDH2+'.nc')

    os.remove(fn_out) if os.path.exists(fn_out) else None
    print('Writing to: ' + fn_out, flush=True)
    DSnew = Dataset(fn_out, 'w', data_model='NETCDF4', clobber=True)
    DSnew.createDimension('n', 1)
    DSnew.createDimension('time',len(grand_mask_timestamps))
    DSnew.createDimension('lon',len(grand_mask_lon))
    DSnew.createDimension('lat',len(grand_mask_lat))
    DSnew.createVariable('lon','f4',('lon',))
    DSnew.createVariable('lat','f4',('lat',))
    DSnew.createVariable('grid_area','f4',('lat','lon'))
    DSnew.createVariable('time','d',('time',))

    DSnew['time'][:] = grand_mask_timestamps
    DSnew['time'].setncattr('units','hours since 1970-1-1 0:0:0')
    DSnew['lon'][:] = grand_mask_lon
    DSnew['lon'].setncattr('units','degrees_east')
    DSnew['lat'][:] = grand_mask_lat
    DSnew['lat'].setncattr('units','degrees_north')

    DSnew['grid_area'][:] = AREA
    DSnew['grid_area'].setncattr('units','km2')
    DSnew['grid_area'].setncattr('description','Area of each grid cell.')

    ## Write volrain if it was calculated.
    if do_volrain:
        fields = [*VOLRAIN]
        fields = [x for x in fields if not 'tser' in x]
        for field in fields:
            add_volrain_to_netcdf(DSnew, field, VOLRAIN[field]
                    , field+'_tser', VOLRAIN[field+'_tser']
                    , fill_value = FILL_VALUE)

    ## Define the mask variables here.
    ## But don't assign them yet.
    fields = [*mask_arrays]
    fields = [x for x in fields if 'mask' in x]
    for field in fields:
        DSnew.createVariable(field,'i1',('time','lat','lon'),zlib=True,complevel=4)
        DSnew[field].setncattr('units','1')
    DSnew.close()

    ## Writing mask variables.
    ## Do them one at a time so it uses less memory while writing them.
    for field in fields:
        print('- ' + field)
        add_mask_var_to_netcdf(fn_out, field, mask_arrays[field]
                            , memory_target_mb = memory_target_mb)
