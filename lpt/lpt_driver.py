import matplotlib; matplotlib.use('agg')
import numpy as np
from context import lpt
import matplotlib.pyplot as plt
import datetime as dt
import cftime
import sys
import os
import matplotlib.colors as colors
from netCDF4 import Dataset
import networkx as nx
from multiprocessing import Pool

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


def lpt_driver(dataset,plotting,output,lpo_options,lpt_options
                ,merge_split_options,mjo_id_options,argv):

    #variable_names = handle_variable_names(dataset)

    ## Get begin and end time from command line.
    ## Give warning message if it has not been specified.
    if len(argv) < 3:
        print('Specify begin and end time on command line, format YYYYMMDDHH.')
        print('Example: python lpt_generic_netcdf_data_driver 2011060100 2012063021')
        return

    begin_time = lpt.helpers.str2cftime(str(argv[1]), '%Y%m%d%H', dataset['calendar'])  # command line arg #1 format: YYYYMMDDHH
    end_time =   lpt.helpers.str2cftime(str(argv[2]), '%Y%m%d%H', dataset['calendar'])  # command line arg #1 format: YYYYMMDDHH
    time_list =  lpt.helpers.dtrange(begin_time, end_time + dt.timedelta(hours=dataset['data_time_interval']), dataset['data_time_interval'])

    if plotting['do_plotting']:
        fig2 = plt.figure(2, figsize = (8.5,11))


    if lpo_options['do_lpo_calc']:

        with Pool(lpo_options['lpo_calc_n_cores']) as p:
            p.starmap(lpt.helpers.do_lpo_calc,
                [(x, begin_time, dataset, lpo_options, output, plotting) for x in time_list],
                chunksize=1)

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
            , include_rain_rates = lpo_options['mask_include_rain_rates']
            , do_volrain = lpo_options['mask_calc_volrain']
            , dataset_dict = dataset
            , calc_with_filter_radius = lpo_options['mask_calc_with_filter_radius']
            , cold_start_mode = lpo_options['cold_start_mode']
            , multiply_factor = lpo_options['multiply_factor']
            , calc_with_accumulation_period = lpo_options['mask_calc_with_accumulation_period']
            , coarse_grid_factor = lpo_options['mask_coarse_grid_factor']
            , memory_target_mb = lpo_options['target_memory_for_writing_masks_MB']
            , nproc = lpo_options['mask_n_cores'])


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
    options['calendar'] = dataset['calendar']

    if options['do_lpt_calc']:

        begin_tracking_time = begin_time
        latest_lp_object_time = end_time

        YMDHb = begin_time.strftime('%Y%m%d%H')
        YMDHb_fancy = begin_time.strftime('%Y-%m-%d %H:00 UTC')

        YMDH = end_time.strftime('%Y%m%d%H')
        YMDH_fancy = end_time.strftime('%Y-%m-%d %H:00 UTC')


        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(('Doing LPT tracking for: ' + YMDHb + ' to ' + YMDH + '.'))
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


        ## Initialize LPT
        G = lpt.helpers.init_lpt_graph(time_list, options['objdir']
            , min_points=options['min_lp_objects_points'], fmt=output['sub_directory_format']+"/objects_%Y%m%d%H.nc")

        ## Connect objects
        ## This is the "meat" of the LPT method: The connection in time step.
        print('Connecting objects...', flush=True)
        G = lpt.helpers.connect_lpt_graph(G, options, min_points = lpt_options['min_lp_objects_points']
            , fmt=output['sub_directory_format']+"/objects_%Y%m%d%H.nc", verbose=True)
        print((str(nx.number_connected_components(nx.to_undirected(G)))+ ' initial LPT groups found.'), flush=True)
        sys.stdout.flush()

        ## Allow for falling below the threshold, if specified..
        if options['fall_below_threshold_max_hours'] > 0:
            print(('Allow for falling below the threshold up to ' + str(options['fall_below_threshold_max_hours']) + ' hours.'),flush=True)
            G = lpt.helpers.lpt_graph_allow_falling_below_threshold(G, options, fmt=output['sub_directory_format']+"/objects_%Y%m%d%H.nc", verbose=True)
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
            TIMECLUSTERS = lpt.helpers.calc_lpt_properties_with_branches(G, options, fmt=output['sub_directory_format']+"/objects_%Y%m%d%H.nc")

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

            ## matplotlib needs datetime.datetime to plot. Can't use cftime.datetime.
            dt_list = [dt.datetime(x.year,x.month,x.day,x.hour,x.minute,x.second) for x in time_list]

            timelon_rain = []
            for this_dt in time_list:
                DATA_RAW = lpt.readdata.readdata(this_dt, dataset)
                lat_idx, = np.where(np.logical_and(DATA_RAW['lat'] > -15.0, DATA_RAW['lat'] < 15.0))
                timelon_rain.append(np.mean(np.array(DATA_RAW['data'][lat_idx,:]), axis=0))

            timelon_rain = np.array(timelon_rain)
            timelon_rain *= lpo_options['multiply_factor'] / 24.0

            lpt.plotting.plot_timelon_with_lpt(ax2, dt_list, DATA_RAW['lon']
                    , timelon_rain, TIMECLUSTERS, plotting['time_lon_range']
                    , accum_time_hours = lpo_options['accumulation_hours'])

            ax2.set_title((dataset['label'].upper()
                            + ' Rain Rate (15$\degree$S-15$\degree$N) and LPTs\n' + YMDHb_fancy + ' to ' + YMDH_fancy))

            ax2.text(0.87,1.02,'(<15$\degree$S, >15$\degree$N Dashed)', transform=ax2.transAxes)

            img_dir2 = (output['img_dir'] + '/' + dataset['label']
                        + '/' + filter_str(lpo_options['filter_stdev'])
                        + '_' + str(int(lpo_options['accumulation_hours'])) + 'h'
                        + '/thresh' + str(int(lpo_options['thresh'])) + '/systems')

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
            , include_rain_rates = lpt_options['mask_include_rain_rates']
            , do_volrain = lpt_options['mask_calc_volrain']
            , dataset_dict = dataset
            , calc_with_filter_radius = lpt_options['mask_calc_with_filter_radius']
            , calc_with_accumulation_period = lpt_options['mask_calc_with_accumulation_period']
            , cold_start_mode = lpo_options['cold_start_mode']
            , multiply_factor = lpo_options['multiply_factor']
            , begin_lptid = lpt_options['individual_masks_begin_lptid']
            , end_lptid = lpt_options['individual_masks_end_lptid']
            , mjo_only = lpt_options['individual_masks_mjo_only']
            , coarse_grid_factor = lpt_options['mask_coarse_grid_factor']
            , memory_target_mb = lpt_options['target_memory_for_writing_masks_MB']
            , nproc = lpt_options['mask_n_cores'])


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
            , include_rain_rates = lpt_options['mask_include_rain_rates']
            , do_volrain = lpt_options['mask_calc_volrain']
            , dataset_dict = dataset
            , calc_with_filter_radius = lpt_options['mask_calc_with_filter_radius']
            , calc_with_accumulation_period = lpt_options['mask_calc_with_accumulation_period']
            , cold_start_mode = lpo_options['cold_start_mode']
            , multiply_factor = lpo_options['multiply_factor']
            , coarse_grid_factor = lpt_options['mask_coarse_grid_factor']
            , memory_target_mb = lpt_options['target_memory_for_writing_masks_MB']
            , nproc = lpt_options['mask_n_cores'])


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
            , include_rain_rates = lpt_options['mask_include_rain_rates']
            , do_volrain = lpt_options['mask_calc_volrain']
            , dataset_dict = dataset
            , calc_with_filter_radius = lpt_options['mask_calc_with_filter_radius']
            , calc_with_accumulation_period = lpt_options['mask_calc_with_accumulation_period']
            , cold_start_mode = lpo_options['cold_start_mode']
            , multiply_factor = lpo_options['multiply_factor']
            , coarse_grid_factor = lpt_options['mask_coarse_grid_factor']
            , memory_target_mb = lpt_options['target_memory_for_writing_masks_MB']
            , nproc = lpt_options['mask_n_cores']
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
            , include_rain_rates = lpt_options['mask_include_rain_rates']
            , do_volrain = lpt_options['mask_calc_volrain']
            , dataset_dict = dataset
            , calc_with_filter_radius = lpt_options['mask_calc_with_filter_radius']
            , calc_with_accumulation_period = lpt_options['mask_calc_with_accumulation_period']
            , cold_start_mode = lpo_options['cold_start_mode']
            , coarse_grid_factor = lpt_options['mask_coarse_grid_factor']
            , memory_target_mb = lpt_options['target_memory_for_writing_masks_MB']
            , nproc = lpt_options['mask_n_cores']
            , subset='non_mjo')
