import matplotlib; matplotlib.use('agg')
import numpy as np
from context import lpt
import matplotlib.pyplot as plt
import datetime as dt
import sys
import os
import matplotlib.colors as colors
import scipy.ndimage
from netCDF4 import Dataset

################################################################################
## These functions are used by lpt_generic_netcdf_data_driver.py
################################################################################

def filter_str(stdev):
    if type(stdev) == int:
        strout = 'g' + str(int(stdev))
    elif type(stdev) == list:
        strout = 'g' + str(int(stdev[0])) + 'x' + str(int(stdev[1]))
    else:
        print('Warning: Wrong data type!')
        strout = None
    return strout


def read_generic_netcdf(fn):
    """
    DATA = read_generic_netcdf(fn)

    output is like this:
    list(DATA)
    Out[12]: ['lon', 'lat', 'precip']
    In [21]: DATA['lon'].shape
    Out[21]: (1440,)
    In [22]: DATA['lat'].shape
    Out[22]: (400,)
    In [23]: DATA['precip'].shape
    Out[23]: (400, 1440)
    """

    DS = Dataset(fn)
    DATA={}
    DATA['lon'] = DS['lon'][:]
    DATA['lat'] = DS['lat'][:]
    DATA['precip'] = DS['rain'][:][0]
    DS.close()

    ## Need to get from (-180, 180) to (0, 360) longitude.
    lon_lt_0, = np.where(DATA['lon'] < -0.0001)
    lon_ge_0, = np.where(DATA['lon'] > -0.0001)
    if len(lon_lt_0) > 0:
        DATA['lon'][lon_lt_0] += 360.0
        DATA['lon'] = np.concatenate((DATA['lon'][lon_ge_0], DATA['lon'][lon_lt_0]))
        DATA['precip'] = np.concatenate((DATA['precip'][:,lon_ge_0], DATA['precip'][:,lon_lt_0]), axis=1)

    return DATA


def read_generic_netcdf_at_datetime(dt, data_dir='.', fmt='gridded_rain_rates_%Y%m%d%H.nc', verbose=False):

    fn = (data_dir + '/' + dt.strftime(fmt))
    DATA=None

    if not os.path.exists(fn):
        print('File not found: ', fn)
    else:
        if verbose:
            print(fn)
        DATA=read_generic_netcdf(fn)

    return DATA



def lpt_driver(dataset,plotting,output,lpo_options,lpt_options,merge_split_options,argv):

    ## Get begin and end time from command line.
    ## Give warning message if it has not been specified.

    if len(argv) < 3:
        print('Specify begin and end time on command line, format YYYYMMDDHH.')
        print('Example: python lpt_generic_netcdf_data_driver 2011060100 2012063021')
        return

    begin_time = dt.datetime.strptime(str(argv[1]), '%Y%m%d%H') # command line arg #1 format: YYYYMMDDHH
    end_time = dt.datetime.strptime(str(argv[2]), '%Y%m%d%H') # command line arg #1 format: YYYYMMDDHH

    hours_list = np.arange(0.0, 0.1 +(end_time-begin_time).total_seconds()/3600.0, dataset['data_time_interval'])
    time_list = [begin_time + dt.timedelta(hours=x) for x in hours_list]

    if plotting['do_plotting']:
        fig1 = plt.figure(1, figsize = (8.5,4))
        fig2 = plt.figure(2, figsize = (8.5,11))

    if lpo_options['do_lpo_calc']:

        for end_of_accumulation_time in time_list:

            try:

                YMDH = end_of_accumulation_time.strftime('%Y%m%d%H')
                YMDH_fancy = end_of_accumulation_time.strftime('%Y-%m-%d %H:00 UTC')

                beginning_of_accumulation_time = end_of_accumulation_time - dt.timedelta(hours=lpo_options['accumulation_hours'])
                print((beginning_of_accumulation_time, end_of_accumulation_time))
                dt_list = [beginning_of_accumulation_time
                    + dt.timedelta(hours=x) for x in np.double(np.arange(0,lpo_options['accumulation_hours']
                                                      + dataset['data_time_interval'],dataset['data_time_interval']))]

                ## Get accumulated rain.
                data_collect = []
                count = 0

                for this_dt in reversed(dt_list):
                    DATA_RAW = read_generic_netcdf_at_datetime(this_dt, data_dir=dataset['raw_data_parent_dir'], fmt=dataset['file_name_format'], verbose=dataset['verbose'])
                    DATA_RAW['precip'][DATA_RAW['precip'] < -0.01] = 0.0
                    if count < 1:
                        data_collect = np.array(DATA_RAW['precip'])
                    else:
                        data_collect += np.array(DATA_RAW['precip'])
                    count += 1

                DATA_RUNNING = (data_collect/count) * 24.0 # Get the mean in mm/day.
                print('Running mean done.',flush=True)

                ## Filter the data
                DATA_FILTERED = scipy.ndimage.gaussian_filter(DATA_RUNNING, lpo_options['filter_stdev']
                    , order=0, output=None, mode='reflect', cval=0.0, truncate=lpo_options['filter_n_stdev_width'])
                print('filter done.',flush=True)

                ## Get LP objects.
                label_im = lpt.helpers.identify_lp_objects(DATA_FILTERED, lpo_options['thresh'], min_points=lpo_options['min_points'], verbose=dataset['verbose'])
                #print(label_im)
                OBJ = lpt.helpers.calculate_lp_object_properties(DATA_RAW['lon'], DATA_RAW['lat']
                            , DATA_RAW['precip'], DATA_RUNNING, DATA_FILTERED, label_im, 0
                            , end_of_accumulation_time, verbose=True)
                OBJ['units_inst'] = 'mm h-1'
                OBJ['units_running'] = 'mm day-1'
                OBJ['units_filtered'] = 'mm day-1'

                print('objects properties.',flush=True)

                """
                Object Output files
                """
                objects_dir = (output['data_dir'] + '/' + dataset['label']
                                + '/' + filter_str(lpo_options['filter_stdev'])
                                + '_' + str(int(lpo_options['accumulation_hours'])) + 'h'
                                + '/thresh' + str(int(lpo_options['thresh']))
                                + '/objects/'
                                + end_of_accumulation_time.strftime(output['sub_directory_format']))

                os.makedirs(objects_dir, exist_ok = True)
                objects_fn = (objects_dir + '/objects_' + YMDH)
                lpt.lptio.lp_objects_output_ascii(objects_fn, OBJ)
                lpt.lptio.lp_objects_output_netcdf(objects_fn + '.nc', OBJ)

                """
                Object Plot
                """
                if plotting['do_plotting']:
                    plt.figure(1)
                    fig1.clf()
                    ax1 = fig1.add_subplot(111)
                    lpt.plotting.plot_rain_map_with_filtered_contour(ax1
                            , DATA_RUNNING, OBJ
                            , plot_area = plotting['plot_area'])
                    ax1.set_title((dataset['label'].upper()
                                    + str(lpo_options['accumulation_hours'])
                                    + '-h Rain Rate and LP Objects\nEnding ' + YMDH_fancy))

                    img_dir1 = (output['img_dir'] + '/' + dataset['label']
                                    + '/' + filter_str(lpo_options['filter_stdev'])
                                    + '_' + str(int(lpo_options['accumulation_hours'])) + 'h'
                                    + '/thresh' + str(int(lpo_options['thresh']))
                                    + '/objects/'
                                    + end_of_accumulation_time.strftime(output['sub_directory_format']))

                    os.makedirs(img_dir1, exist_ok = True)
                    file_out_base = (img_dir1 + '/lp_objects_' + dataset['label'] + '_' + YMDH)
                    lpt.plotting.print_and_save(file_out_base)
                    fig1.clf()

            except FileNotFoundError:
                print('Data not yet available up to this point. Skipping.')


    """
    LPT Tracking Calculations (i.e., connect LP Objects in time)
    """
    options = lpt_options
    options['objdir'] = (output['data_dir'] + '/' + dataset['label']
                    + '/' + filter_str(lpo_options['filter_stdev'])
                    + '_' + str(int(lpo_options['accumulation_hours'])) + 'h'
                    + '/thresh' + str(int(lpo_options['thresh'])) + '/objects')
    options['outdir'] = (output['data_dir'] + '/' + dataset['label']
                    + '/' + filter_str(lpo_options['filter_stdev'])
                    + '_' + str(int(lpo_options['accumulation_hours'])) + 'h'
                    + '/thresh' + str(int(lpo_options['thresh'])) + '/systems')
    #print(options)
    #print(lpo_options)

    if options['do_lpt_calc']:

        begin_tracking_time = begin_time
        latest_lp_object_time = end_time

        YMDHb = begin_time.strftime('%Y%m%d%H')
        YMDHb_fancy = begin_time.strftime('%Y-%m-%d %H:00 UTC')

        YMDH = end_time.strftime('%Y%m%d%H')
        YMDH_fancy = end_time.strftime('%Y-%m-%d %H:00 UTC')


        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(('Doing LPT tracking for: '
                + begin_tracking_time.strftime('%Y%m%d%H')
                + ' to ' + latest_lp_object_time.strftime('%Y%m%d%H')))
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        dt_list = time_list # [begin_tracking_time + dt.timedelta(hours=x) for x in range(0, 24*lpt_options['lpt_history_days']+1, dataset['data_time_interval'])]

        ## Initialize LPT
        LPT0, BRANCHES0 = lpt.helpers.init_lpt_group_array(dt_list, options['objdir'])

        ## Remove small LPOs
        LPT0, BRANCHES0 = lpt.helpers.lpt_group_array_remove_small_objects(LPT0, BRANCHES0, options)

        ## Connect objects
        print('Connecting objects...')
        LPTfb, BRANCHESfb = lpt.helpers.calc_lpt_group_array3(LPT0, BRANCHES0, options, min_points = lpt_options['min_lp_objects_points'], verbose=True)
        #LPTfb, BRANCHESfb = lpt.helpers.calc_lpt_group_array2(LPT0, BRANCHES0, options, min_points = lpt_options['min_lp_objects_points'], verbose=True)

        ## Allow center jumps.
        print(('Allow center jumps up to ' + str(options['center_jump_max_hours']) + ' hours.'))
        LPT_center_jumps, BRANCHES_center_jumps = lpt.helpers.lpt_group_array_allow_center_jumps2(LPTfb, BRANCHESfb, options)
        #LPT_center_jumps = LPTfb.copy()
        #BRANCHES_center_jumps = BRANCHESfb.copy()

        ## Eliminate short duration systems.
        print(('Remove LPT shorter than ' + str(options['min_lpt_duration_hours']) + ' hours.'))
        LPT_remove_short, BRANCHES_remove_short = lpt.helpers.remove_short_lived_systems(LPT_center_jumps, BRANCHES_center_jumps, options['min_lpt_duration_hours']
                                , latest_datetime = latest_lp_object_time)

        ## Handle splitting and merging, if specified.
        if merge_split_options['allow_merge_split']:
            LPT, BRANCHES = lpt.helpers.lpt_split_and_merge(LPT_remove_short, BRANCHES_remove_short, merge_split_options, options)
            #LPT = LPT_remove_short.copy()
            #BRANCHES = BRANCHES_remove_short.copy()
        else:
            LPT = LPT_remove_short.copy()
            BRANCHES = BRANCHES_remove_short.copy()

        ## Re-order LPT system branches.
        print('Re-ordering LPT branches.')
        BRANCHES = lpt.helpers.reorder_LPT_branches(LPT, BRANCHES)

        ## Get "timeclusters" tracks.
        print('Calculating LPT properties.')
        if merge_split_options['allow_merge_split']:
            TIMECLUSTERS = lpt.helpers.calc_lpt_system_group_properties_with_branches(LPT, BRANCHES, options)
        else:
            TIMECLUSTERS = lpt.helpers.calc_lpt_system_group_properties(LPT, options)

        fn_tc_base = (options['outdir'] #+ '/' + end_time.strftime(output['sub_directory_format'])
                         + '/lpt_systems_' + dataset['label'] + '_' + YMDHb + '_' + YMDH)
        lpt.lptio.lpt_system_tracks_output_ascii(fn_tc_base + '.txt', TIMECLUSTERS)
        lpt.lptio.lpt_systems_group_array_output_ascii(fn_tc_base + '.group_array.txt', LPT, BRANCHES)
        lpt.lptio.lpt_system_tracks_output_netcdf(fn_tc_base + '.nc', TIMECLUSTERS)


        """
        LPT Plotting
        """

        if plotting['do_plotting']:
            plt.figure(2)
            fig2.clf()
            ax2 = fig2.add_subplot(111)

            timelon_rain = []
            for this_dt in dt_list:
                if 'sub_area' in dataset.keys():
                    DATA_RAW = read_generic_netcdf_at_datetime(this_dt, data_dir=dataset['raw_data_parent_dir'], fmt=dataset['file_name_format'], verbose=dataset['verbose'], area=dataset['sub_area'])
                else:
                    DATA_RAW = read_generic_netcdf_at_datetime(this_dt, data_dir=dataset['raw_data_parent_dir'], fmt=dataset['file_name_format'], verbose=dataset['verbose'])

                lat_idx, = np.where(np.logical_and(DATA_RAW['lat'] > -15.0, DATA_RAW['lat'] < 15.0))
                timelon_rain.append(np.mean(np.array(DATA_RAW['precip'][lat_idx,:]), axis=0))


            lpt.plotting.plot_timelon_with_lpt(ax2, dt_list, DATA_RAW['lon']
                    , timelon_rain, TIMECLUSTERS, plotting['time_lon_range']
                    , accum_time_hours = lpo_options['accumulation_hours'])

            ax2.set_title((dataset['label'].upper()
                            + ' Rain Rate (15$\degree$S-15$\degree$N) and LPTs\n' + YMDHb_fancy + ' to ' + YMDH_fancy))

            ax2.text(0.87,1.02,'(<15$\degree$S, >15$\degree$N Dashed)', transform=ax2.transAxes)

            img_dir2 = (output['img_dir'] + '/' + dataset['label'] + '/systems/')
            #                + end_time.strftime(output['sub_directory_format']))

            os.makedirs(img_dir2, exist_ok = True)
            file_out_base = (img_dir2 + '/lpt_time_lon_' + dataset['label'] + '_' + YMDHb + '_' + YMDH)
            lpt.plotting.print_and_save(file_out_base)
            fig2.clf()
