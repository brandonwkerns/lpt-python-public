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
import networkx as nx

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

def handle_variable_names(dataset):
    ## Extract variable names from the dataset dict if specified
    ## Otherwise, return a default tuple of variable names.
    if (('longitude_variable_name' in dataset) and
        ('latitude_variable_name' in dataset) and
        ('field_variable_name' in dataset)):

        variable_names = (dataset['longitude_variable_name']
                , dataset['latitude_variable_name']
                , dataset['field_variable_name'])
    else:
        variable_names = ('lon','lat','rain')
    return variable_names


def lpt_driver(dataset,plotting,output,lpo_options,lpt_options
                ,merge_split_options,mjo_id_options,argv):

    variable_names = handle_variable_names(dataset)

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
                    ## Read in data according to specified raw_data_format.
                    if dataset['raw_data_format'] == 'generic_netcdf':
                        DATA_RAW = lpt.readdata.read_generic_netcdf_at_datetime(this_dt
                                , variable_names = variable_names
                                , data_dir=dataset['raw_data_parent_dir']
                                , fmt=dataset['file_name_format']
                                , verbose=dataset['verbose'])
                    elif dataset['raw_data_format'] == 'cmorph':
                        DATA_RAW = lpt.readdata.read_cmorph_at_datetime(this_dt, verbose=dataset['verbose'])
                        DATA_RAW['data'] = np.ma.masked_array(DATA_RAW['precip'])
                    else:
                        print((dataset['raw_data_format'] + ' is not a valid raw_data_format!'))

                    DATA_RAW['data'] = np.array(DATA_RAW['data'].filled(fill_value=0.0))
                    DATA_RAW['data'][~np.isfinite(DATA_RAW['data'])] = 0.0
                    if count < 1:
                        data_collect = DATA_RAW['data'].copy()
                    else:
                        data_collect += DATA_RAW['data']
                    count += 1
                    print(np.max(data_collect))

                DATA_RUNNING = (data_collect/count) * lpo_options['multiply_factor'] # Get to the units you want for objects.
                print('Running mean done.',flush=True)

                ## Filter the data
                DATA_FILTERED = scipy.ndimage.gaussian_filter(DATA_RUNNING, lpo_options['filter_stdev']
                    , order=0, output=None, mode='reflect', cval=0.0, truncate=lpo_options['filter_n_stdev_width'])
                print('filter done.',flush=True)

                ## Get LP objects.
                label_im = lpt.helpers.identify_lp_objects(DATA_FILTERED, lpo_options['thresh'], min_points=lpo_options['min_points'], verbose=dataset['verbose'])
                OBJ = lpt.helpers.calculate_lp_object_properties(DATA_RAW['lon'], DATA_RAW['lat']
                            , DATA_RAW['data'], DATA_RUNNING, DATA_FILTERED, label_im, 0
                            , end_of_accumulation_time, verbose=True)
                OBJ['units_inst'] = dataset['field_units']
                OBJ['units_running'] = lpo_options['field_units']
                OBJ['units_filtered'] = lpo_options['field_units']

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

    ##
    ## Do LPO mask, if specified.
    ##
    if lpo_options['do_lpo_mask']:

        ## In case LPO wasn't calculated, make sure the relevant stuff is defined.
        objects_dir = (output['data_dir'] + '/' + dataset['label']
                        + '/' + filter_str(lpo_options['filter_stdev'])
                        + '_' + str(int(lpo_options['accumulation_hours'])) + 'h'
                        + '/thresh' + str(int(lpo_options['thresh']))
                        + '/objects/')

        lpt.masks.calc_lpo_mask(begin_time, end_time, dataset['data_time_interval']
            , accumulation_hours = lpo_options['accumulation_hours'], filter_stdev = lpo_options['filter_stdev']
            , lp_objects_dir=objects_dir, lp_objects_fn_format=(output['sub_directory_format']+'/objects_%Y%m%d%H.nc')
            , mask_output_dir=objects_dir
            , do_volrain = lpo_options['mask_calc_volrain']
            , dataset_dict = dataset
            , calc_with_filter_radius = lpo_options['mask_calc_with_filter_radius']
            , calc_with_accumulation_period = lpo_options['mask_calc_with_accumulation_period']
            , memory_target_mb = lpo_options['target_memory_for_writing_masks_MB'])


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

        dt_list = time_list

        ## Initialize LPT
        G = lpt.helpers.init_lpt_graph(dt_list, options['objdir']
            , min_points=options['min_lp_objects_points'], fmt="/%Y/%m/%Y%m%d/objects_%Y%m%d%H.nc")

        ## Connect objects
        ## This is the "meat" of the LPT method: The connection in time step.
        print('Connecting objects...', flush=True)
        G = lpt.helpers.connect_lpt_graph(G, options, min_points = lpt_options['min_lp_objects_points'], verbose=True)
        print((str(nx.number_connected_components(nx.to_undirected(G)))+ ' LPT groups found.'), flush=True)
        sys.stdout.flush()

        ## Allow for falling below the threshold, if specified..
        if options['fall_below_threshold_max_hours'] > 0:
            print(('Allow for falling below the threshold up to ' + str(options['fall_below_threshold_max_hours']) + ' hours.'),flush=True)
            G = lpt.helpers.lpt_graph_allow_falling_below_threshold(G, options, verbose=True)
            print((str(nx.number_connected_components(nx.to_undirected(G)))+ ' LPT groups left.'), flush=True)

        ## Eliminate short duration systems.
        if options['min_lpt_duration_hours'] > 0.0:
            print(('Remove LPT shorter than ' + str(options['min_lpt_duration_hours']) + ' hours.'),flush=True)
            G = lpt.helpers.lpt_graph_remove_short_duration_systems(G, options['min_lpt_duration_hours']
                                    , latest_datetime = latest_lp_object_time)
            print((str(nx.number_connected_components(nx.to_undirected(G)))+ ' LPT groups left.'), flush=True)


        if merge_split_options['allow_merge_split']:
            print('Will split groups in to separate overlapping LPTs.',flush=True)
            if merge_split_options['split_merger_min_hours'] > 0:
                print('Remove splits and mergers < '+str(merge_split_options['split_merger_min_hours'])+' h.', flush=True)
                G = lpt.helpers.lpt_graph_remove_short_ends(G
                    , merge_split_options['split_merger_min_hours']
                      - dataset['data_time_interval'])

            print('--- Calculating LPT System Properties. ---', flush=True)
            print('    !!! This step may take a while !!!', flush=True)
            TIMECLUSTERS = lpt.helpers.calc_lpt_properties_with_branches(G, options)

        else:
            print('Splits and mergers retained as the same LPT system.', flush=True)
            print('--- Calculating LPT System Properties. ---', flush=True)
            TIMECLUSTERS = lpt.helpers.calc_lpt_properties_without_branches(G, options)

        ## Output
        print('--- Writing output. ---',flush=True)

        fn_tc_base = (options['outdir']
                         + '/lpt_systems_' + dataset['label'] + '_' + YMDHb + '_' + YMDH)
        lpt.lptio.lpt_system_tracks_output_ascii(fn_tc_base + '.txt', TIMECLUSTERS)
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
                DATA_RAW = lpt.readdata.read_generic_netcdf_at_datetime(this_dt
                        , variable_names = variable_names
                        , data_dir=dataset['raw_data_parent_dir']
                        , fmt=dataset['file_name_format']
                        , verbose=dataset['verbose'])

                lat_idx, = np.where(np.logical_and(DATA_RAW['lat'] > -15.0, DATA_RAW['lat'] < 15.0))
                timelon_rain.append(np.mean(np.array(DATA_RAW['data'][lat_idx,:]), axis=0))


            lpt.plotting.plot_timelon_with_lpt(ax2, dt_list, DATA_RAW['lon']
                    , timelon_rain, TIMECLUSTERS, plotting['time_lon_range']
                    , accum_time_hours = lpo_options['accumulation_hours'])

            ax2.set_title((dataset['label'].upper()
                            + ' Rain Rate (15$\degree$S-15$\degree$N) and LPTs\n' + YMDHb_fancy + ' to ' + YMDH_fancy))

            ax2.text(0.87,1.02,'(<15$\degree$S, >15$\degree$N Dashed)', transform=ax2.transAxes)

            img_dir2 = (output['img_dir'] + '/' + dataset['label'] + '/systems/')
            os.makedirs(img_dir2, exist_ok = True)
            file_out_base = (img_dir2 + '/lpt_time_lon_' + dataset['label'] + '_' + YMDHb + '_' + YMDH)
            lpt.plotting.print_and_save(file_out_base)
            fig2.clf()


    if mjo_id_options['do_mjo_id']:

        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(('Doing MJO Identification for: '
                + begin_time.strftime('%Y%m%d%H')
                + ' to ' + end_time.strftime('%Y%m%d%H')))
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        ## In case LPO wasn't calculated, make sure the relevant stuff is defined.
        objects_dir = (output['data_dir'] + '/' + dataset['label']
                        + '/' + filter_str(lpo_options['filter_stdev'])
                        + '_' + str(int(lpo_options['accumulation_hours'])) + 'h'
                        + '/thresh' + str(int(lpo_options['thresh']))
                        + '/objects/')

        lpt.mjo_id.do_mjo_id(begin_time, end_time, dataset['data_time_interval']
            , mjo_id_options, prod = dataset['label']
            , accumulation_hours = lpo_options['accumulation_hours'], filter_stdev = lpo_options['filter_stdev']
            , lp_objects_dir=objects_dir, lp_objects_fn_format=(output['sub_directory_format']+'/objects_%Y%m%d%H.nc')
            , lpt_systems_dir=options['outdir'])



    ##
    ## Do LPT mask, if specified.
    ##
    if lpt_options['do_lpt_individual_masks']:

        ## In case LPO wasn't calculated, make sure the relevant stuff is defined.
        objects_dir = (output['data_dir'] + '/' + dataset['label']
                        + '/' + filter_str(lpo_options['filter_stdev'])
                        + '_' + str(int(lpo_options['accumulation_hours'])) + 'h'
                        + '/thresh' + str(int(lpo_options['thresh']))
                        + '/objects/')

        lpt.masks.calc_individual_lpt_masks(begin_time, end_time, dataset['data_time_interval']
            , prod = dataset['label']
            , accumulation_hours = lpo_options['accumulation_hours'], filter_stdev = lpo_options['filter_stdev']
            , lp_objects_dir=objects_dir, lp_objects_fn_format=(output['sub_directory_format']+'/objects_%Y%m%d%H.nc')
            , lpt_systems_dir=options['outdir']
            , mask_output_dir=options['outdir']
            , do_volrain = lpt_options['mask_calc_volrain']
            , dataset_dict = dataset
            , calc_with_filter_radius = lpt_options['mask_calc_with_filter_radius']
            , calc_with_accumulation_period = lpt_options['mask_calc_with_accumulation_period']
            , begin_lptid = lpt_options['individual_masks_begin_lptid']
            , end_lptid = lpt_options['individual_masks_end_lptid']
            , memory_target_mb = lpt_options['target_memory_for_writing_masks_MB'])


    if lpt_options['do_lpt_composite_mask']:

        ## In case LPO wasn't calculated, make sure the relevant stuff is defined.
        objects_dir = (output['data_dir'] + '/' + dataset['label']
                        + '/' + filter_str(lpo_options['filter_stdev'])
                        + '_' + str(int(lpo_options['accumulation_hours'])) + 'h'
                        + '/thresh' + str(int(lpo_options['thresh']))
                        + '/objects/')

        lpt.masks.calc_composite_lpt_mask(begin_time, end_time, dataset['data_time_interval']
            , prod = dataset['label']
            , accumulation_hours = lpo_options['accumulation_hours'], filter_stdev = lpo_options['filter_stdev']
            , lp_objects_dir=objects_dir, lp_objects_fn_format=(output['sub_directory_format']+'/objects_%Y%m%d%H.nc')
            , lpt_systems_dir=options['outdir']
            , mask_output_dir=options['outdir']
            , do_volrain = lpt_options['mask_calc_volrain']
            , dataset_dict = dataset
            , calc_with_filter_radius = lpt_options['mask_calc_with_filter_radius']
            , calc_with_accumulation_period = lpt_options['mask_calc_with_accumulation_period']
            , memory_target_mb = lpt_options['target_memory_for_writing_masks_MB'])


    if lpt_options['do_mjo_lpt_composite_mask']:

        ## In case LPO wasn't calculated, make sure the relevant stuff is defined.
        objects_dir = (output['data_dir'] + '/' + dataset['label']
                        + '/' + filter_str(lpo_options['filter_stdev'])
                        + '_' + str(int(lpo_options['accumulation_hours'])) + 'h'
                        + '/thresh' + str(int(lpo_options['thresh']))
                        + '/objects/')

        lpt.masks.calc_composite_lpt_mask(begin_time, end_time, dataset['data_time_interval']
            , prod = dataset['label']
            , accumulation_hours = lpo_options['accumulation_hours'], filter_stdev = lpo_options['filter_stdev']
            , lp_objects_dir=objects_dir, lp_objects_fn_format=(output['sub_directory_format']+'/objects_%Y%m%d%H.nc')
            , lpt_systems_dir=options['outdir']
            , mask_output_dir=options['outdir']
            , do_volrain = lpt_options['mask_calc_volrain']
            , dataset_dict = dataset
            , calc_with_filter_radius = lpt_options['mask_calc_with_filter_radius']
            , calc_with_accumulation_period = lpt_options['mask_calc_with_accumulation_period']
            , memory_target_mb = lpt_options['target_memory_for_writing_masks_MB']
            , subset='mjo')

    if lpt_options['do_non_mjo_lpt_composite_mask']:

        ## In case LPO wasn't calculated, make sure the relevant stuff is defined.
        objects_dir = (output['data_dir'] + '/' + dataset['label']
                        + '/' + filter_str(lpo_options['filter_stdev'])
                        + '_' + str(int(lpo_options['accumulation_hours'])) + 'h'
                        + '/thresh' + str(int(lpo_options['thresh']))
                        + '/objects/')

        lpt.masks.calc_composite_lpt_mask(begin_time, end_time, dataset['data_time_interval']
            , prod = dataset['label']
            , accumulation_hours = lpo_options['accumulation_hours'], filter_stdev = lpo_options['filter_stdev']
            , lp_objects_dir=objects_dir, lp_objects_fn_format=(output['sub_directory_format']+'/objects_%Y%m%d%H.nc')
            , lpt_systems_dir=options['outdir']
            , mask_output_dir=options['outdir']
            , do_volrain = lpt_options['mask_calc_volrain']
            , dataset_dict = dataset
            , calc_with_filter_radius = lpt_options['mask_calc_with_filter_radius']
            , calc_with_accumulation_period = lpt_options['mask_calc_with_accumulation_period']
            , memory_target_mb = lpt_options['target_memory_for_writing_masks_MB']
            , subset='non_mjo')
