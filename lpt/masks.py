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


##
## feature spread function -- used for all of the mask functions.
##
def feature_spread(array_in, npoints):

    ## Use convolution to expand the mask "array_in" a radius of np points.
    ## For this purpose, it takes a 3-D array with the first entry being time.

    array_out = array_in.copy()
    s = array_in.shape

    if type(npoints) is list:
        npx = npoints[0]
        npy = npoints[1]
    else:
        npx = 1*npoints
        npy = 1*npoints

    nt = s[0]

    [circle_array_x, circle_array_y] = np.meshgrid(np.arange(-1*npx,npx+1), np.arange(-1*npy,npy+1))
    circle_array_dist = np.sqrt(np.power(circle_array_x,2) + np.power(circle_array_y * (npx/npy),2))
    circle_array_mask = (circle_array_dist < (npx + 0.1)).astype(np.double)
    circle_array_mask = circle_array_mask / np.sum(circle_array_mask)

    ## Loop over the times.
    ## For each time, use the convolution to "spread out" the effect of each time's field.
    ## (I tried using 3-D convolution here, but it took almost twice as much memory
    ##  and was slightly SLOWER than this method.)
    for tt in range(s[0]):
        array_2d = array_in[tt,:,:]
        array_2d_new = array_2d.copy()
        unique_values = np.unique(array_2d)
        unique_values = unique_values[unique_values > 0]  #take out zero -- it is not a feature.
        for this_value in unique_values:
            starting_mask = (array_2d == this_value).astype(np.double)
            starting_mask_spread = scipy.ndimage.convolve(starting_mask, circle_array_mask, mode='constant')
            array_2d_new[starting_mask_spread > 0.001] = this_value

        array_out[tt,:,:] = array_2d_new

    return array_out


def add_mask_var_to_netcdf(DS, mask_var, data, dims=('time','lat','lon')):
    """
    Add a 3-D mask variable to a NetCDF file.
    add_mask_var_to_netcdf(DS, mask_var, data, dims=('time','lat','lon'))

    Inputs:
        DS        is a netCDF4.Dataset file object.
        mask_var  is a string for the variable name.
        data      is the data to use. Should be integers.
        dims      is a tuple of dimension names. Default: ('time','lat','lon')
    Outputs:
        -- None --
    """
    DS.createVariable(mask_var,'i',dims,zlib=True,complevel=4)
    DS[mask_var][:] = data
    DS[mask_var].setncattr('units','1')


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
    , calc_with_filter_radius = True
    , calc_with_accumulation_period = True):

    """
    dt_begin, dt_end: datetime objects for the first and last times. These are END of accumulation times!
    """

    YMDH1_YMDH2 = (dt_begin.strftime('%Y%m%d%H') + '_' + dt_end.strftime('%Y%m%d%H'))


    # These times are for the END of accumulation time.
    """
    total_hours0 = (dt_end - dt_begin).total_seconds()/3600.0
    mask_times0 = [dt_begin + dt.timedelta(hours=x) for x in np.arange(0,total_hours0+1,interval_hours)]

    # These times include the accumulation period leading up to the first END of accumulation time.
    if accumulation_hours > 0 and calc_with_accumulation_period:
        total_hours = (dt_end - dt_begin).total_seconds()/3600.0  + accumulation_hours
        mask_times = [dt_begin - dt.timedelta(hours=accumulation_hours) + dt.timedelta(hours=x) for x in np.arange(0,total_hours+1,interval_hours)]
    else:
        total_hours = (dt_end - dt_begin).total_seconds()/3600.0
        mask_times = [dt_begin + dt.timedelta(hours=x) for x in np.arange(0,total_hours+1,interval_hours)]
    """

    dt1 = dt_end
    if accumulation_hours > 0 and calc_with_accumulation_period:
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

        fn = (lp_objects_dir + '/' + this_dt.strftime(lp_objects_fn_format))
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
            mask_arrays['lon'] = DS['grid_lon'][:]
            mask_arrays['lat'] = DS['grid_lat'][:]
            mask_arrays_shape = [len(mask_times), len(lat), len(lon)]
            mask_arrays['mask_at_end_time'] = np.zeros(mask_arrays_shape)
            if accumulation_hours > 0 and calc_with_accumulation_period:
                mask_arrays['mask_with_accumulation'] = np.zeros(mask_arrays_shape)
            if calc_with_filter_radius:
                mask_arrays['mask_with_filter_at_end_time'] = np.zeros(mask_arrays_shape)
                if accumulation_hours > 0 and calc_with_accumulation_period:
                    mask_arrays['mask_with_filter_and_accumulation'] = np.zeros(mask_arrays_shape)

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
            mask_arrays['mask_at_end_time'][dt_idx, jjj, iii] = 1

            ## For the mask with accumulation, go backwards and fill in ones.
            if accumulation_hours > 0 and calc_with_accumulation_period:
                n_back = int(accumulation_hours/interval_hours)
                for ttt in range(dt_idx - n_back, dt_idx+1):
                    mask_arrays['mask_with_accumulation'][ttt, jjj, iii] = 1

        except:
            DS.close()

    ##
    ## Do filter width spreading.
    ##

    if filter_stdev > 0 and calc_with_filter_radius:
        print('Filter width spreading...this may take awhile.', flush=True)
        mask_arrays['mask_with_filter_at_end_time'] = feature_spread(mask_arrays['mask_at_end_time'], filter_stdev)
        if accumulation_hours > 0 and calc_with_accumulation_period:
            mask_arrays['mask_with_filter_and_accumulation'] = feature_spread(mask_arrays['mask_with_accumulation'], filter_stdev)

    ##
    ## Output.
    ##
    fn_out = (mask_output_dir + '/lp_objects_mask_' + YMDH1_YMDH2 + '.nc')
    os.makedirs(mask_output_dir, exist_ok=True)

    print('Writing to: ' + fn_out, flush=True)
    DSnew = Dataset(fn_out, 'w')
    DSnew.createDimension('time',len(mask_times))
    DSnew.createDimension('lon',len(lon))
    DSnew.createDimension('lat',len(lat))

    DSnew.createVariable('time','d',('time',))
    DSnew.createVariable('lon','d',('lon',))
    DSnew.createVariable('lat','d',('lat',))

    DSnew['time'][:] = [(x - dt.datetime(1970,1,1,0,0,0)).total_seconds()/3600.0 for x in mask_times]
    DSnew['time'].setncattr('units','hours since 1970-1-1 0:0:0')
    DSnew['lon'][:] = lon
    DSnew['lon'].setncattr('units','degrees_east')
    DSnew['lat'][:] = lat
    DSnew['lat'].setncattr('units','degrees_north')

    add_mask_var_to_netcdf(DSnew, 'mask_at_end_time', mask_arrays['mask_at_end_time'])
    if accumulation_hours > 0 and accumulation_hours > 0 and calc_with_accumulation_period:
        add_mask_var_to_netcdf(DSnew, 'mask_with_accumulation', mask_arrays['mask_with_accumulation'])
    if filter_stdev > 0 and calc_with_filter_radius:
        add_mask_var_to_netcdf(DSnew, 'mask_with_filter_at_end_time', mask_arrays['mask_with_filter_at_end_time'])
        if accumulation_hours > 0 and accumulation_hours > 0 and calc_with_accumulation_period:
            add_mask_var_to_netcdf(DSnew, 'mask_with_filter_and_accumulation', mask_arrays['mask_with_filter_and_accumulation'])

    DSnew.close()


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
    , do_volrain=False, rain_dir = '.'
    , calc_with_filter_radius = True
    , calc_with_accumulation_period = True):

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


    ## Read Stitched data.
    DS = Dataset(lpt_systems_file)
    TC={}
    TC['lptid'] = DS['lptid'][:]
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


    LPT, BRANCHES = lpt.lptio.read_lpt_systems_group_array(lpt_group_file)

    F = Dataset(lpt_systems_file)
    unique_lpt_ids = np.unique(F['lptid'][:])

    for this_lpt_id in unique_lpt_ids:
        print('Calculating LPT system mask for lptid = ' + str(this_lpt_id) + ' of time period ' + YMDH1_YMDH2 + '.')

        this_group = np.floor(this_lpt_id)
        this_group_lptid_list = sorted([x for x in unique_lpt_ids if np.floor(x) == this_group])

        if np.round( 100.0 * (this_group_lptid_list[0] - this_group)) > 0:
            this_branch = int(2**(np.round( 100.0 * (this_lpt_id - this_group)) - 1))
        else:
            this_branch = int(2**(np.round( 1000.0 * (this_lpt_id - this_group)) - 1))

        print((this_lpt_id, this_branch))
        if this_branch > 0:
            this_branch_idx = [x for x in range(len(BRANCHES)) if LPT[x,2]==this_group and (BRANCHES[x] & this_branch) > 0] # bitwise and
        else:
            this_branch_idx = [x for x in range(len(BRANCHES)) if LPT[x,2]==this_group]

        lp_object_id_list = LPT[this_branch_idx,1]

        if accumulation_hours > 0 and calc_with_accumulation_period:
            dt0 = dt.datetime.strptime(str(int(np.min(lp_object_id_list)))[0:10],'%Y%m%d%H') - dt.timedelta(hours=accumulation_hours)
        else:
            dt0 = dt.datetime.strptime(str(int(np.min(lp_object_id_list)))[0:10],'%Y%m%d%H')
        dt1 = dt.datetime.strptime(str(int(np.max(lp_object_id_list)))[0:10],'%Y%m%d%H')
        duration_hours = int((dt1 - dt0).total_seconds()/3600)
        mask_times = [dt0 + dt.timedelta(hours=x) for x in range(0,duration_hours+1,interval_hours)]
        mask_arrays={} #Start with empty dictionary


        ## Include some basic LPT info for user friendliness.
        lptidx = [ii for ii in range(len(TC['lptid'])) if this_lpt_id == TC['lptid'][ii]][0]

        # Time varying properties
        for var in ['centroid_lon','centroid_lat','area'
                    ,'max_filtered_running_field','max_running_field','max_inst_field'
                    ,'min_filtered_running_field','min_running_field','min_inst_field'
                    ,'amean_filtered_running_field','amean_running_field','amean_inst_field']:
            mask_arrays[var] = MISSING * np.ones(len(mask_times))

        for ttt in range(TC['i1'][lptidx],TC['i2'][lptidx]+1):
            this_time_indx = [ii for ii in range(len(mask_times)) if TC['datetime'][ttt] == mask_times[ii]]
            if len(this_time_indx) > 0:
                for var in ['centroid_lon','centroid_lat','area'
                    ,'max_filtered_running_field','max_running_field','max_inst_field'
                    ,'min_filtered_running_field','min_running_field','min_inst_field'
                    ,'amean_filtered_running_field','amean_running_field','amean_inst_field']:
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
                mask_arrays_shape = [len(mask_times), len(lat), len(lon)]
                mask_arrays['mask_at_end_time'] = np.zeros(mask_arrays_shape)
                if accumulation_hours > 0 and calc_with_accumulation_period:
                    mask_arrays['mask_with_accumulation'] = np.zeros(mask_arrays_shape)
                if filter_stdev > 0 and calc_with_filter_radius:
                    mask_arrays['mask_with_filter_at_end_time'] = np.zeros(mask_arrays_shape)
                    if accumulation_hours > 0 and calc_with_accumulation_period:
                        mask_arrays['mask_with_filter_and_accumulation'] = np.zeros(mask_arrays_shape)


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
            mask_arrays['mask_at_end_time'][dt_idx, jjj, iii] = 1

            ## For the mask with accumulation, go backwards and fill in ones.
            if accumulation_hours > 0 and calc_with_accumulation_period:
                n_back = int(accumulation_hours/interval_hours)
                for ttt in range(dt_idx - n_back, dt_idx+1):
                    mask_arrays['mask_with_accumulation'][ttt, jjj, iii] = 1

        ##
        ## Do filter width spreading.
        ##

        if filter_stdev > 0 and calc_with_filter_radius:
            print('Filter width spreading...this may take awhile.', flush=True)
            mask_arrays['mask_with_filter_at_end_time'] = feature_spread(mask_arrays['mask_at_end_time'], filter_stdev)
            if accumulation_hours > 0 and calc_with_accumulation_period:
                mask_arrays['mask_with_filter_and_accumulation'] = feature_spread(mask_arrays['mask_with_accumulation'], filter_stdev)

        ## Do volumetric rain.
        if do_volrain:
            print('Now calculating the volumetric rain.', flush=True)

            #### Initialize
            VOLRAIN = {}

            ## Global
            VOLRAIN['volrain_global'] = 0.0
            VOLRAIN['volrain_global_tser'] = np.nan * np.zeros(len(mask_times))

            ## This LPT
            VOLRAIN['volrain_at_end_time'] = 0.0
            VOLRAIN['volrain_at_end_time_tser'] = np.nan * np.zeros(len(mask_times))
            if accumulation_hours > 0 and calc_with_accumulation_period:
                VOLRAIN['volrain_with_accumulation'] = 0.0
                VOLRAIN['volrain_with_accumulation_tser'] = np.nan * np.zeros(len(mask_times))
            if filter_stdev > 0 and calc_with_filter_radius:
                VOLRAIN['volrain_with_filter_at_end_time'] = 0.0
                VOLRAIN['volrain_with_filter_at_end_time_tser'] = np.nan * np.zeros(len(mask_times))
                if accumulation_hours > 0 and calc_with_accumulation_period:
                    VOLRAIN['volrain_with_filter_and_accumulation'] = 0.0
                    VOLRAIN['volrain_with_filter_and_accumulation_tser'] = np.nan * np.zeros(len(mask_times))

            #### Fill in values by time. Multiply rain field by applicable mask (0 and 1 values).
            for tt in range(len(mask_times)):
                this_dt = mask_times[tt]

                ## Get rain.
                RAIN = rain_read_function(this_dt, verbose=False)
                precip = RAIN['data'][:]
                precip[~np.isfinite(precip)] = 0.0
                precip[precip < -0.01] = 0.0

                ## Global
                VOLRAIN['volrain_global'] += interval_hours * np.sum(precip * AREA)
                VOLRAIN['volrain_global_tser'][tt] = interval_hours * np.sum(precip * AREA)

                ## This LPT
                this_mask = mask_arrays['mask_at_end_time'][tt]
                this_mask[this_mask > 0] = 1.0
                precip_masked = precip * this_mask
                VOLRAIN['volrain_at_end_time'] += interval_hours * np.sum(precip_masked * AREA)
                VOLRAIN['volrain_at_end_time_tser'][tt] = interval_hours * np.sum(precip_masked * AREA)
                if accumulation_hours > 0 and calc_with_accumulation_period:
                    this_mask = mask_arrays['mask_with_accumulation'][tt]
                    this_mask[this_mask > 0] = 1.0
                    precip_masked = precip * this_mask
                    VOLRAIN['volrain_with_accumulation'] += interval_hours * np.sum(precip_masked * AREA)
                    VOLRAIN['volrain_with_accumulation_tser'][tt] = interval_hours * np.sum(precip_masked * AREA)
                if filter_stdev > 0 and calc_with_filter_radius:
                    this_mask = mask_arrays['mask_with_filter_at_end_time'][tt]
                    this_mask[this_mask > 0] = 1.0
                    precip_masked = precip * this_mask
                    VOLRAIN['volrain_with_filter_at_end_time'] += interval_hours * np.sum(precip_masked * AREA)
                    VOLRAIN['volrain_with_filter_at_end_time_tser'][tt] = interval_hours * np.sum(precip_masked * AREA)
                    if accumulation_hours > 0 and calc_with_accumulation_period:
                        this_mask = mask_arrays['mask_with_filter_and_accumulation'][tt]
                        this_mask[this_mask > 0] = 1.0
                        precip_masked = precip * this_mask
                        VOLRAIN['volrain_with_filter_and_accumulation'] += interval_hours * np.sum(precip_masked * AREA)
                        VOLRAIN['volrain_with_filter_and_accumulation_tser'][tt] = interval_hours * np.sum(precip_masked * AREA)

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

        for mask_var in ['centroid_lon','centroid_lat','area'
                    ,'max_filtered_running_field','max_running_field','max_inst_field'
                    ,'min_filtered_running_field','min_running_field','min_inst_field'
                    ,'amean_filtered_running_field','amean_running_field','amean_inst_field'
                    ,'duration','maxarea','zonal_propagation_speed','meridional_propagation_speed']:
            DSnew[mask_var][:] = mask_arrays[mask_var]

        DSnew['centroid_lon'].setncatts({'units':'degrees_east','long_name':'centroid longitude (0-360)','standard_name':'longitude','note':'Time is end of running mean time.'})
        DSnew['centroid_lat'].setncatts({'units':'degrees_east','long_name':'centroid latitude (-90-00)','standard_name':'latitude','note':'Time is end of running mean time.'})
        DSnew['area'].setncatts({'units':'km2','long_name':'LPT System enclosed area','note':'Time is end of running mean time.'})
        DSnew['maxarea'].setncatts({'units':'km2','long_name':'LPT System enclosed area','note':'This is the max over the LPT life time.'})
        DSnew['duration'].setncatts({'units':'h','long_name':'LPT System duration'})
        DSnew['zonal_propagation_speed'].setncatts({'units':'m s-1','long_name':'Zonal popagation speed','description':'Zonal popagation speed of the entire LPT system -- based on least squares fit of lon(time).'})
        DSnew['meridional_propagation_speed'].setncatts({'units':'m s-1','long_name':'meridional popagation speed','description':'Meridional popagation speed of the entire LPT system -- based on least squares fit of lon(time).'})

        for var in ['max_filtered_running_field','max_running_field','max_inst_field'
                    ,'min_filtered_running_field','min_running_field','min_inst_field'
                    ,'amean_filtered_running_field','amean_running_field','amean_inst_field']:
            DSnew[var].setncatts({'units':'mm day-1','long_name':'LP object running mean rain rate (at end of accum time).','note':'Time is end of running mean time. Based on mask_at_end_time'})

        add_mask_var_to_netcdf(DSnew, 'mask_at_end_time', mask_arrays['mask_at_end_time'])
        if accumulation_hours > 0 and calc_with_accumulation_period:
            add_mask_var_to_netcdf(DSnew, 'mask_with_accumulation', mask_arrays['mask_with_accumulation'])
        if filter_stdev > 0 and calc_with_filter_radius:
            add_mask_var_to_netcdf(DSnew, 'mask_with_filter_at_end_time', mask_arrays['mask_with_filter_at_end_time'])
            if accumulation_hours > 0 and calc_with_accumulation_period:
                add_mask_var_to_netcdf(DSnew, 'mask_with_filter_and_accumulation', mask_arrays['mask_with_filter_and_accumulation'])

        if do_volrain:
            add_volrain_to_netcdf(DSnew, 'volrain_global', VOLRAIN['volrain_global']
                    , 'volrain_global_tser', VOLRAIN['volrain_global_tser']
                    , fill_value = FILL_VALUE)
            add_volrain_to_netcdf(DSnew, 'volrain_at_end_time', VOLRAIN['volrain_at_end_time']
                    , 'volrain_at_end_time_tser', VOLRAIN['volrain_at_end_time_tser']
                    , fill_value = FILL_VALUE)
            if accumulation_hours > 0 and calc_with_accumulation_period:
                add_volrain_to_netcdf(DSnew, 'volrain_with_accumulation', VOLRAIN['volrain_with_accumulation']
                        , 'volrain_with_accumulation_tser', VOLRAIN['volrain_with_accumulation_tser']
                        , fill_value = FILL_VALUE)
            if filter_stdev > 0 and calc_with_filter_radius:
                add_volrain_to_netcdf(DSnew, 'volrain_with_filter_at_end_time', VOLRAIN['volrain_with_filter_at_end_time']
                        , 'volrain_with_filter_at_end_time_tser', VOLRAIN['volrain_with_filter_at_end_time_tser']
                        , fill_value = FILL_VALUE)
                if accumulation_hours > 0 and calc_with_accumulation_period:
                    add_volrain_to_netcdf(DSnew, 'volrain_with_filter_and_accumulation', VOLRAIN['volrain_with_filter_and_accumulation']
                            , 'volrain_with_filter_and_accumulation_tser', VOLRAIN['volrain_with_filter_and_accumulation_tser']
                            , fill_value = FILL_VALUE)

        DSnew['grid_area'][:] = AREA
        DSnew['grid_area'].setncattr('units','km2')
        DSnew['grid_area'].setncattr('description','Area of each grid cell.')

        DSnew.close()




################################################################################
################################################################################
################################################################################
###########  calc_composite_lpt_mask ###########################################
################################################################################
################################################################################
################################################################################

def calc_composite_lpt_mask(dt_begin, dt_end, interval_hours, prod='trmm'
    ,accumulation_hours = 0, filter_stdev = 0
    , lp_objects_dir = '.', lp_objects_fn_format='%Y/%m/%Y%m%d/objects_%Y%m%d%H.nc'
    , lpt_systems_dir = '.'
    , mask_output_dir = '.', verbose=True
    , calc_with_filter_radius = True
    , calc_with_accumulation_period = True
    , subset='all'):

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
    if accumulation_hours > 0 and calc_with_accumulation_period:
        grand_mask_timestamps0 = (dt_begin - dt.datetime(1970,1,1,0,0,0)).total_seconds()/3600.0 - accumulation_hours
    else:
        grand_mask_timestamps0 = (dt_begin - dt.datetime(1970,1,1,0,0,0)).total_seconds()/3600.0
    grand_mask_timestamps1 = (dt_end - dt.datetime(1970,1,1,0,0,0)).total_seconds()/3600.0

    grand_mask_timestamps = np.arange(grand_mask_timestamps0, grand_mask_timestamps1 + dt_hours, dt_hours)
    mask_arrays = None


    ## Read Stitched NetCDF data.
    DS = Dataset(lpt_systems_file)
    TC={}
    TC['lptid'] = DS['lptid'][:]
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

    LPT, BRANCHES = lpt.lptio.read_lpt_systems_group_array(lpt_group_file)


    ## Get LPT list if it's MJO (subset='mjo') or non MJO (subset='non_mjo') case.
    ## Over-ride unique_lpt_ids so it only includes those LPTs.
    if subset == 'mjo':
        lpt_list_file = (lpt_systems_dir + '/mjo_lpt_list_'+prod+'_'+YMDH1_YMDH2+'.txt')
        lpt_list = np.loadtxt(lpt_list_file,skiprows=1)
        unique_lpt_ids = np.unique(lpt_list[:,2])
    if subset == 'non_mjo':
        lpt_list_file = (lpt_systems_dir + '/non_mjo_lpt_list_'+prod+'_'+YMDH1_YMDH2+'.txt')
        lpt_list = np.loadtxt(lpt_list_file,skiprows=1)
        unique_lpt_ids = np.unique(lpt_list[:,2])



    for this_lpt_id in unique_lpt_ids:
        print('Adding LPT system mask for lptid = ' + str(this_lpt_id) + ' (max = ' + str(np.max(unique_lpt_ids)) + ') of time period ' + YMDH1_YMDH2 + '.')

        this_group = np.floor(this_lpt_id)
        this_group_lptid_list = sorted([x for x in unique_lpt_ids if np.floor(x) == this_group])

        if np.round( 100.0 * (this_group_lptid_list[0] - this_group)) > 0:
            this_branch = int(2**(np.round( 100.0 * (this_lpt_id - this_group)) - 1))
        else:
            this_branch = int(2**(np.round( 1000.0 * (this_lpt_id - this_group)) - 1))

        if this_branch > 0:
            this_branch_idx = [x for x in range(len(BRANCHES)) if LPT[x,2]==this_group and (BRANCHES[x] & this_branch) > 0] # bitwise and
        else:
            this_branch_idx = [x for x in range(len(BRANCHES)) if LPT[x,2]==this_group]

        lp_object_id_list = LPT[this_branch_idx,1]


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
                mask_arrays_shape = (len(grand_mask_timestamps), len(grand_mask_lat), len(grand_mask_lon))
                mask_arrays['mask_at_end_time'] = np.zeros(mask_arrays_shape)
                if accumulation_hours > 0 and calc_with_accumulation_period:
                    mask_arrays['mask_with_accumulation'] = np.zeros(mask_arrays_shape)
                if calc_with_filter_radius:
                    mask_arrays['mask_with_filter_at_end_time'] = np.zeros(mask_arrays_shape)
                    if accumulation_hours > 0 and calc_with_accumulation_period:
                        mask_arrays['mask_with_filter_and_accumulation'] = np.zeros(mask_arrays_shape)

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
            mask_arrays['mask_at_end_time'][dt_idx, jjj, iii] = 1

            ## For the mask with accumulation, go backwards and fill in ones.
            if accumulation_hours > 0 and calc_with_accumulation_period:
                n_back = int(accumulation_hours/interval_hours)
                for ttt in range(dt_idx - n_back, dt_idx+1):
                    mask_arrays['mask_with_accumulation'][ttt, jjj, iii] = 1

    ##
    ## Do filter width spreading.
    ##

    if filter_stdev > 0 and calc_with_filter_radius:
        print('Filter width spreading...this may take awhile.', flush=True)
        mask_arrays['mask_with_filter_at_end_time'] = feature_spread(mask_arrays['mask_at_end_time'], filter_stdev)
        if accumulation_hours > 0 and calc_with_accumulation_period:
            mask_arrays['mask_with_filter_and_accumulation'] = feature_spread(mask_arrays['mask_with_accumulation'], filter_stdev)


    ##
    ## Output.
    ##
    if subset == 'mjo':
        fn_out = (lpt_systems_dir+'/lpt_composite_mask_'+YMDH1_YMDH2+'_mjo_lpt.nc')
    elif subset == 'non_mjo':
        fn_out = (lpt_systems_dir+'/lpt_composite_mask_'+YMDH1_YMDH2+'_non_mjo_lpt.nc')
    else:
        fn_out = (lpt_systems_dir+'/lpt_composite_mask_'+YMDH1_YMDH2+'.nc')

    os.remove(fn_out) if os.path.exists(fn_out) else None
    print('Writing to: ' + fn_out, flush=True)
    DSnew = Dataset(fn_out, 'w', data_model='NETCDF4', clobber=True)
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

    add_mask_var_to_netcdf(DSnew, 'mask_at_end_time', mask_arrays['mask_at_end_time'])
    if accumulation_hours > 0 and accumulation_hours > 0 and calc_with_accumulation_period:
        add_mask_var_to_netcdf(DSnew, 'mask_with_accumulation', mask_arrays['mask_with_accumulation'])
    if filter_stdev > 0 and calc_with_filter_radius:
        add_mask_var_to_netcdf(DSnew, 'mask_with_filter_at_end_time', mask_arrays['mask_with_filter_at_end_time'])
        if accumulation_hours > 0 and accumulation_hours > 0 and calc_with_accumulation_period:
            add_mask_var_to_netcdf(DSnew, 'mask_with_filter_and_accumulation', mask_arrays['mask_with_filter_and_accumulation'])

    DSnew['grid_area'][:] = AREA
    DSnew['grid_area'].setncattr('units','km2')
    DSnew['grid_area'].setncattr('description','Area of each grid cell.')

    DSnew.close()
