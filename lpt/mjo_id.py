import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import linregress
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import datetime as dt
try:
    from context import lpt
except:
    pass
import os
import glob
import sys

################################################################################
##  The helper functions.
################################################################################

def get_delta_t_hours(datetime_list):
    ## Figure out the hours between times.
    return np.nanmin([(datetime_list[x] - datetime_list[x-1]).total_seconds()/3600.0 for x in range(1,len(datetime_list))])

def fill_in_center_jumps(datetime_list, lon, delta_t_hours):
    ## Fill in gaps from center jumps, if any.
    hours_since_beginning = [(datetime_list[x] - datetime_list[0]).total_seconds()/3600.0 for x in range(len(datetime_list))]
    hours_since_beginning_filled = np.arange(0,hours_since_beginning[-1] + delta_t_hours,delta_t_hours)
    lon_filled = np.interp(hours_since_beginning_filled, hours_since_beginning, lon)
    return (hours_since_beginning_filled, lon_filled)

def calc_centered_diff_speed(lon, delta_t_hours):
    ## Calculate the zonal speed time series.
    ## Speed is centered difference except forward (backward) diff at beginning (end).
    ## Output units should be in m/s.
    lon = np.array(lon)
    delta_t_seconds = delta_t_hours * 3600.0
    spd_raw = 0.0 * lon;
    spd_raw[1:-1] = (lon[2:] - lon[0:-2]) * 110000.0 / (2.0*(delta_t_seconds))
    spd_raw[0] = (lon[1] - lon[0]) * 110000.0 / delta_t_seconds
    spd_raw[-1] = (lon[-1] - lon[-2]) * 110000.0 / delta_t_seconds
    return spd_raw

def median_filter(data):
    data = np.array(data)
    data_filtered = 0.0 * data
    data_filtered[0] = np.median(data[0:3])
    data_filtered[-1] = np.median(data[-3:])
    for ii in range(1,len(data)):
        data_filtered[ii] = np.median(data[ii-1:ii+2])
    return data_filtered

def get_mask_period_info(hours_since_beginning, lon, mask, lat=None, verbose=False):
    ## Get Pandas data frame of east propagation periods.
    ## If lat is set to None, don't calculate latitude related stuff.
    label_im, nE = ndimage.label(mask)
    if verbose:
        print(f'Found {str(nE)} propagation periods.')

    props = {}
    begin_hours,end_hours,indx1, indx2 = ndimage.extrema(hours_since_beginning, label_im, range(1,nE+1))

    props['duration'] = (np.array(end_hours) - np.array(begin_hours))
    props['begin_indx'] = np.array(indx1)[:,0].astype(int)
    props['end_indx'] = np.array(indx2)[:,0].astype(int)
    props['begin_lon'] = lon[props['begin_indx']]
    props['end_lon'] = lon[props['end_indx']]
    #print(props['begin_indx'])
    #props['min_lon'] = np.array([np.nanmin(lon[props['begin_indx'][x]:props['end_indx'][x]+1]) for x in range(len(props['begin_indx'])) ])
    #props['max_lon'] = np.array([np.nanmax(lon[props['begin_indx'][x]:props['end_indx'][x]+1]) for x in range(len(props['begin_indx'])) ])
    props['lon_propagation'] = lon[props['end_indx']] - lon[props['begin_indx']]
    props['total_zonal_spd'] = linregress(hours_since_beginning,lon)[0] * (111000 / 3600.0 )

    min_lon,max_lon,indx1, indx2 = ndimage.extrema(lon, label_im, range(1,nE+1))
    props['min_lon'] = min_lon
    props['max_lon'] = max_lon


    props['segment_zonal_spd'] = 0.0 * np.arange(nE)
    for iii in range(nE):
        ii1 = props['begin_indx'][iii]
        ii2 = props['end_indx'][iii]+1
        props['segment_zonal_spd'][iii] = linregress(hours_since_beginning[ii1:ii2],lon[ii1:ii2])[0] * (111000 / 3600.0 )

    if not lat is None:
        props['begin_lat'] = lat[props['begin_indx']]
        props['end_lat'] = lat[props['end_indx']]
        props['lat_propagation'] = lat[props['end_indx']] - lat[props['begin_indx']]
        min_lat,max_lat,indx1, indx2 = ndimage.extrema(lat, label_im, range(1,nE+1))
        props['min_lat'] = min_lat
        props['max_lat'] = max_lat
        props['total_meridional_spd'] = linregress(hours_since_beginning,lat)[0] * (111000 / 3600.0 )

        props['segment_meridional_spd'] = 0.0 * np.arange(nE)
        for iii in range(nE):
            ii1 = props['begin_indx'][iii]
            ii2 = props['end_indx'][iii]+1
            props['segment_meridional_spd'][iii] = linregress(hours_since_beginning[ii1:ii2],lat[ii1:ii2])[0] * (111000 / 3600.0 )

    ## Put in to Pandas DataFrame
    F = pd.DataFrame.from_dict(props)

    return F

def datetime2dateint(x):
    return int(1e6*x.year + 1e4*x.month + 1e2*x.day + x.hour)

def duration_in_hours(x):
    return (x[-1] - x[0]).total_seconds()/3600.0


################################################################################
################################################################################
################################################################################
########### Function for separating east and west propatation periods. #########
################################################################################
################################################################################
################################################################################


def west_east_divide_and_conquer(datetime_list, lon, opts, do_plotting=False, plot_path='east_west_propagation_division.png', plot_suptitle=''):
    """
    [mask_net_eastward_propagation, spd_raw] = west_east_divide_and_conquer(G, year1, ii, do_plotting)

    datetime_list is a list of Python datetime objects. It can be irregular, e.g., if there are center jumps in the track.
    lon is a list or Numpy array of longitudes. Must be same size as datetime_list.
    """
    lon = np.array(lon)  # Force to be numpy array

    delta_t_hours = get_delta_t_hours(datetime_list)
    hours_since_beginning = [(x - datetime_list[0]).total_seconds()/3600.0 for x in datetime_list]
    hours_since_beginning_filled, lon_filled = fill_in_center_jumps(datetime_list, lon, delta_t_hours)
    spd_raw = calc_centered_diff_speed(lon_filled, delta_t_hours)

    ## plot speed, if plots are specified.
    #########################################

    if do_plotting:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        plt.close('all')

        os.makedirs('plots', exist_ok=True)

        fig = plt.figure(figsize=(6,10))
        gs = gridspec.GridSpec(ncols=1, nrows=6)

        ax0 = fig.add_subplot(gs[0:3,0])

        ax0.plot(hours_since_beginning_filled, lon_filled,'b-')
        ax0.plot(hours_since_beginning, lon, 'bo')

        ax1 = fig.add_subplot(gs[3,0])
        ax1.plot(hours_since_beginning_filled, spd_raw, 'b',linewidth = 1.0)
        ax1.plot(hours_since_beginning_filled, 0.0*spd_raw, 'k--',linewidth = 0.5)
        ax1.set_title('Zonal Propagation Speed')
        ax1.set_ylabel('[m/s]')
        ax1.set_ylim([-10.0, 10.0])

    #########################################

    """
    Group the contiguous periods of > 0.
    This is an iterative process!
    For the first step, eat all cases that are one single
    3h period of eastward (westward) propagation surrounded by
    westward (eastward) propagation. Use the median filter for this.
    """

    mask_net_eastward_propagation = [spd_raw > 0.0][0].astype(int)

    #########################################
    if do_plotting:
        ax2 = fig.add_subplot(gs[4,0])
        plt.bar(hours_since_beginning_filled,2*mask_net_eastward_propagation-1
                ,width=delta_t_hours,edgecolor='none', linewidth=0)
        ax2.plot(hours_since_beginning_filled, 0.0*spd_raw, 'k--',linewidth = 0.5)
        ax2.set_ylim([-1,1])
        ax2.set_yticks([-1,1])
        ax2.set_yticklabels(['West','East'])
        ax2.set_title('Zonal Propagation Direction: Raw')
    #########################################

    ## Keep going UNLESS it is all easterly or westerly.
    if len(np.unique(mask_net_eastward_propagation)) > 1:
        keep_going = True
    else:
        keep_going = False ## Skip if it has no east or west prop portions in RAW data.
    niter = 0;
    maxiter = 100;
    while keep_going:

        keep_going = False
        niter = niter + 1;

        if niter > maxiter:
            keep_going = False
            print('WARNING! Reached max iterations: ' + str(maxiter))
            break

        old_mask_net_eastward_propagation = mask_net_eastward_propagation.copy()

        ## %%%%%%%%%%%%%%%%%%%%%%%%%%%
        ## Sort the eastward and westward propagation periods.
        ## %%%%%%%%%%%%%%%%%%%%%%%%%%%

        statsE = get_mask_period_info(hours_since_beginning_filled,lon_filled,mask_net_eastward_propagation)
        statsE['east_prop'] = np.full(len(statsE.index), True)
        statsW = get_mask_period_info(hours_since_beginning_filled,lon_filled,1 - mask_net_eastward_propagation)
        statsW['east_prop'] = np.full(len(statsW.index), False)

        if ( len(statsE.index) < 1 or len(statsW.index) < 1):
            break

        statsEW = statsE.append(statsW)
        statsEW.sort_values('begin_indx',inplace=True)
        statsEW.index = range(len(statsEW.index))

        sort_indices = statsEW['duration'].values.argsort()

        for ii in sort_indices:
            conquer = False  # Start by assuming no conquer.
            ii1 = statsEW['begin_indx'].values[ii]
            ii2 = statsEW['end_indx'].values[ii]+1

            if statsEW['duration'].values[ii] < 0.001:
                conquer = True
            ## First, check whether it is long enough duration and/or has enough
            ## longitude propagation to stand on its own and not get conquered.
            if statsEW['duration'].values[ii] < opts['duration_to_avoid_being_conquered']:
                if abs(statsEW['lon_propagation'].values[ii]) < opts['lon_prop_to_avoid_being_conquered']:

                    ## OK, it cannot stand on it's own by default.
                    ## To determine whether it gets conquered, use the neighboring periods.
                    if ii == 0:  # On left side, compare the time period to the right.
                        continue

                    elif ii == len(sort_indices)-1:  # On right side, compare the time period to the left.
                        continue

                    else:  # Somewhere in interior. Compare the time periods to the left and right.
                        if statsEW['duration'].values[ii] <= statsEW['duration'].values[ii+1] and statsEW['duration'].values[ii] < statsEW['duration'].values[ii-1]:
                            conquer = True
                        if abs(statsEW['lon_propagation'].values[ii]) <= abs(statsEW['lon_propagation'].values[ii+1]) and abs(statsEW['lon_propagation'].values[ii]) < abs(statsEW['lon_propagation'].values[ii-1]):
                            conquer = True

                        ## Check "backtrack allowance"
                        if statsEW['east_prop'].values[ii]:
                            ## Check for net westward progression
                            if (statsEW['min_lon'].values[ii+1] - statsEW['min_lon'].values[ii-1]) > opts['backtrack_allowance']:
                                conquer = False
                                print('Prevent conquer due to lon propagation.')
                        else:
                            ## Check for net eastward progression
                            if (statsEW['max_lon'].values[ii+1] - statsEW['max_lon'].values[ii-1]) < -1*opts['backtrack_allowance']:
                                conquer = False

            if conquer:
                ## If the condition is satisfied to conquer it, then
                old_values = mask_net_eastward_propagation[ii1:ii2]
                mask_net_eastward_propagation[ii1:ii2] = 1 - old_values

                ## Keep going UNLESS it is all easterly or westerly.
                if len(np.unique(mask_net_eastward_propagation)) > 1:
                    keep_going = True

                break


    print('Finished after ' + str(niter) + ' iterations.')


    #########################################
    if do_plotting:
        #ax3 = fig.add_subplot(3,1,3)
        ax3 = fig.add_subplot(gs[5,0])

        plt.bar(hours_since_beginning_filled,2*mask_net_eastward_propagation-1
                ,width=delta_t_hours, edgecolor='none', linewidth=0)
        ax3.plot(hours_since_beginning_filled, 0.0*spd_raw, 'k--',linewidth = 0.5)
        ax3.set_ylim([-1,1])
        ax3.set_yticks([-1,1])
        ax3.set_yticklabels(['West','East'])
        ax3.set_title('Zonal Propagation Direction: Divide and Conquer')
        ax3.set_xlabel('Time Since Initiation [h]')

        plt.suptitle(plot_suptitle, y=1.02)
        plt.tight_layout()
        print(plot_path)
        plt.savefig(plot_path,dpi=100,bbox_inches='tight')
    #########################################


    ## Get back to the original times provided (e.g., center jumps are not filled in)
    keep_indices = [x for x in range(len(hours_since_beginning_filled)) if hours_since_beginning_filled[x] in hours_since_beginning]

    return (mask_net_eastward_propagation[keep_indices], spd_raw[keep_indices])

################################################################################
################################################################################
################################################################################
########### The main driver script for MJO Identificaiton. #####################
################################################################################
################################################################################
################################################################################

def do_mjo_id(dt_begin, dt_end, interval_hours, opts, prod='trmm'
    ,accumulation_hours = 0, filter_stdev = 0
    , lp_objects_dir = '.', lp_objects_fn_format='objects_%Y%m%d%H.nc'
    , lpt_systems_dir = '.', verbose=True):

    do_plotting = opts['do_plotting']
    YMDH1_YMDH2 = (dt_begin.strftime('%Y%m%d%H') + '_' + dt_end.strftime('%Y%m%d%H'))
    dateint1 = datetime2dateint(dt_begin)
    dateint2 = datetime2dateint(dt_end)

    ##
    ## Initialize
    ##
    count_mjo=0    ## Keep count of east propagating systems.
    count_in_mjo_group=0    ## Keep count of systems in the same group as east propagating systems.
    count_non_mjo=0   ## Keep count of non MJO systems

    ## These arrays will get filled in throughout the function, and written at the end.
    FOUT_mjo_lpt = [] ## This will hold the MJO LPT info.
    FOUT_mjo_group_extra_lpt = [] ## This will hold the LPTs in an MJO LPT group but not selected as MJO LPT.
    FOUT_non_mjo_lpt = [] ## This will hold the MJO LPT info.

    ##
    ## Read in LPT stuff.
    ##
    lpt_systems_file = (lpt_systems_dir + '/lpt_systems_'+prod+'_'+YMDH1_YMDH2+'.nc')

    ds = Dataset(lpt_systems_file)
    f = {}
    f['lptid'] = ds['lptid'][:]
    f['group'] = np.floor(f['lptid'][:])
    f['i1'] = ds['lpt_begin_index'][:]
    f['i2'] = ds['lpt_end_index'][:]
    f['lon'] = ds['centroid_lon_stitched'][:]
    f['lat'] = ds['centroid_lat_stitched'][:]
    f['area'] = ds['area_stitched'][:]
    f['duration'] = ds['duration'][:]
    ts = ds['timestamp_stitched'][:]
    ds.close()


    ## Search for an MJO within each group.
    #for this_group in [149]:
    for this_group in sorted(np.unique(f['group'])):

        print(('----------- Group #' + str(this_group) + ' -----------'), flush=True)

        lptid_for_this_clump = np.array(sorted([f['lptid'][x] for x in range(len(f['lptid'])) if f['group'][x] == this_group]))

        if len(lptid_for_this_clump) < 1:
            continue

        eastward_propagation_metric = 0.0 * lptid_for_this_clump
        eastward_propagation_metric2 = 0.0 * lptid_for_this_clump
        longest_east_prop_zonal_speed_all = 0.0 * lptid_for_this_clump
        longest_east_prop_duration_all = 0.0 * lptid_for_this_clump
        longest_east_prop_duration_in_lat_band_all = 0.0 * lptid_for_this_clump
        idx_begin_east_prop_all = 0.0 * lptid_for_this_clump
        idx_end_east_prop_all = 0.0 * lptid_for_this_clump

        meets_mjo_criteria = 0.0 * lptid_for_this_clump

        n = 0

        mjo_candidates_list = []
        mjo_candidates_list_count = 0

        east_prop_group_df = None

        for this_lptid in lptid_for_this_clump:

            print('LPT ID: ' + str(this_lptid))
            iii = np.where(f['lptid'] == this_lptid)[0][0]
            i1 = f['i1'][iii]
            i2 = f['i2'][iii]

            this_lpt_time = [dt.datetime(1970,1,1,0,0,0) + dt.timedelta(hours=int(x)) for x in ts[i1:i2+1]]
            hours_since_beginning = [(x - this_lpt_time[0]).total_seconds()/3600.0 for x in this_lpt_time]
            this_duration = f['duration'][iii]
            this_lpt_lon = f['lon'][i1:i2+1]
            this_lpt_lat = f['lat'][i1:i2+1]
            this_lpt_area = f['area'][i1:i2+1]  ## Only used in case of "tie breaker" for dominant MJO east prop segment.

            ## "bulk" properties of the LPT as a whole.
            duration = duration_in_hours(this_lpt_time)
            total_lon_propagation = max(this_lpt_lon) - min(this_lpt_lon)
            net_lon_propagation = this_lpt_lon[-1] - this_lpt_lon[1]

            plot_dir = ('mjo_id_plots/' + YMDH1_YMDH2)
            if do_plotting:
                os.makedirs(plot_dir, exist_ok=True)
            plot_file = (plot_dir + '/east_west_propagation_division_'+prod+'_'+YMDH1_YMDH2+ '.lptid{0:010.4f}.png'.format(this_lptid))
            [mask_net_eastward_propagation, spd_raw] = west_east_divide_and_conquer(this_lpt_time, this_lpt_lon, opts
                , do_plotting=do_plotting, plot_path=plot_file, plot_suptitle='LPT ID: {0:010.4f}'.format(this_lptid))
            ## Get the data frame with the characteristics of each of the periods
            ## of eastward propagation.
            east_prop_df = {}

            ## If there are NO eastward propagation periods, hack it with west propagation period
            ## HOWEVER, make sure meets_mjo_criteria is always FALSE.
            if 1 in mask_net_eastward_propagation:
                east_prop_df = get_mask_period_info(hours_since_beginning,this_lpt_lon,mask_net_eastward_propagation,lat=this_lpt_lat)
                east_prop_df['area_times_speed'] = np.nansum(this_lpt_area * spd_raw)
                east_prop_df['lptid'] = this_lptid
                east_prop_df['total_duration'] = this_duration
                east_prop_df['year_begin'] = this_lpt_time[0].year
                east_prop_df['month_begin'] = this_lpt_time[0].month
                east_prop_df['day_begin'] = this_lpt_time[0].day
                east_prop_df['hour_begin'] = this_lpt_time[0].hour
                east_prop_df['year_end'] = this_lpt_time[-1].year
                east_prop_df['month_end'] = this_lpt_time[-1].month
                east_prop_df['day_end'] = this_lpt_time[-1].day
                east_prop_df['hour_end'] = this_lpt_time[-1].hour

                ## Fill in 'meets_mjo_criteria' field. Check each for MJO criteria.
                east_prop_df['meets_mjo_criteria'] = False
                for iiii in range(len(east_prop_df.index)):
                    ii1 = east_prop_df['begin_indx'][iiii]
                    ii2 = east_prop_df['end_indx'][iiii]

                    if (east_prop_df['duration'][iiii] > (opts['min_eastward_prop_duration'] - 0.001)
                            and east_prop_df['lon_propagation'][iiii] > (opts['min_total_eastward_lon_propagation'] - 0.001)):

                        hours_in_lat = 3.0*np.sum(np.abs(this_lpt_lat[ii1:ii2+1]) <= opts['max_abs_latitude'])
                        if hours_in_lat > opts['min_eastward_prop_duration_in_lat_band'] - 0.001:
                            east_prop_df['meets_mjo_criteria'][iiii] = True

                if east_prop_group_df is None:
                    east_prop_group_df = east_prop_df
                else:
                    east_prop_group_df = east_prop_group_df.append(east_prop_df)

        if east_prop_group_df is None:

            # If I'm here, there were no east propagation periods in this LPT.
            ## - Write to the non MJO LPT file.
            east_prop_df = get_mask_period_info(hours_since_beginning,this_lpt_lon,1-mask_net_eastward_propagation,lat=this_lpt_lat)
            east_prop_df['area_times_speed'] = np.nansum(this_lpt_area * spd_raw)
            east_prop_df['lptid'] = this_lptid
            east_prop_df['total_duration'] = this_duration
            east_prop_df['year_begin'] = this_lpt_time[0].year
            east_prop_df['month_begin'] = this_lpt_time[0].month
            east_prop_df['day_begin'] = this_lpt_time[0].day
            east_prop_df['hour_begin'] = this_lpt_time[0].hour
            east_prop_df['year_end'] = this_lpt_time[-1].year
            east_prop_df['month_end'] = this_lpt_time[-1].month
            east_prop_df['day_end'] = this_lpt_time[-1].day
            east_prop_df['hour_end'] = this_lpt_time[-1].hour

            east_prop_group_df = east_prop_df

            for jj in range(len(east_prop_group_df)):

                year11 = east_prop_group_df['year_begin'].values[jj]
                month11 = east_prop_group_df['month_begin'].values[jj]
                day11 = east_prop_group_df['day_begin'].values[jj]
                hour11 = east_prop_group_df['hour_begin'].values[jj]
                year22 = east_prop_group_df['year_end'].values[jj]
                month22 = east_prop_group_df['month_end'].values[jj]
                day22 = east_prop_group_df['day_end'].values[jj]
                hour22 = east_prop_group_df['hour_end'].values[jj]

                ## Check whether this one has already been added.
                if len(FOUT_non_mjo_lpt) > 0:
                    if east_prop_group_df['lptid'].values[jj] in np.array(FOUT_non_mjo_lpt)[:,2]:
                        continue

                FOUT_non_mjo_lpt += [ [dateint1, dateint2, east_prop_group_df['lptid'].values[jj], this_group
                                        , east_prop_group_df['total_duration'].values[jj]
                                        , east_prop_group_df['total_zonal_spd'].values[jj]
                                        , year11, month11, day11, hour11
                                        , year22, month22, day22, hour22
                                        , -999, -999, -999, -999
                                        , 9999,99,99,99
                                        , 9999,99,99,99
                                        , -999, -999
                                    ] ]

        else:

            ## If I am here, there were east propagation periods.
            ## If any of them qualified as an MJO:
            ## - Figure out which one is the domimant east propagation period
            ## - Write to MJO LPT file OR to MJO group "extras".
            ## Otherwise:
            ## - Write to the non MJO LPT file.

            if east_prop_group_df['meets_mjo_criteria'].any():

                lpts_in_group_with_mjo_eprop = east_prop_group_df[east_prop_group_df['meets_mjo_criteria']]['lptid'].values
                print(lpts_in_group_with_mjo_eprop)
                ## TODO: Allow multiple MJO eprop per one LPT. Allow multiple LPT in group of LPTs.

                ## Sort the data frame to figure out which should be used for the MJO.
                east_prop_group_df_sort = east_prop_group_df.sort_values(['meets_mjo_criteria','duration','area_times_speed'],ascending=False)


                for jj in range(len(east_prop_group_df_sort.index)):
                    ## Check whether it is already represented.
                    ## It is already represented if:
                    ##    The identical east propagation period has already been used as part of another LPT.
                    already_used = False
                    if jj > 0:

                        jjj = np.where(f['lptid'] == east_prop_group_df_sort['lptid'].values[jj])[0][0]
                        i1 = f['i1'][jjj]+east_prop_group_df_sort['begin_indx'].values[jj]
                        i2 = f['i1'][jjj]+east_prop_group_df_sort['end_indx'].values[jj]
                        points1 =  pd.DataFrame.from_dict({'ts':ts[i1:i2+1]
                                                , 'lon':f['lon'][i1:i2+1]
                                                , 'lat':f['lat'][i1:i2+1]})

                        for jj0 in range(0,jj):
                            jjj = np.where(f['lptid'] == east_prop_group_df_sort['lptid'].values[jj0])[0][0]
                            i1 = f['i1'][jjj]+east_prop_group_df_sort['begin_indx'].values[jj0]
                            i2 = f['i1'][jjj]+east_prop_group_df_sort['end_indx'].values[jj0]
                            points0 =  pd.DataFrame.from_dict({'ts':ts[i1:i2+1]
                                                    , 'lon':f['lon'][i1:i2+1]
                                                    , 'lat':f['lat'][i1:i2+1]})


                            points_merge = pd.merge(points0, points1
                                                        , how='inner'
                                                        , on=['ts','lon','lat'])
                            print(points_merge)
                            if len(points_merge.index) > 0:
                                already_used = True
                                break

                    if not already_used and east_prop_group_df_sort['meets_mjo_criteria'].values[jj]:

                        print('LPT ID ' + str(east_prop_group_df_sort['lptid'].values[jj]) + ' is selected as an MJO event.')
                        jjj = np.where(f['lptid'] == east_prop_group_df_sort['lptid'].values[jj])[0][0]
                        i1 = f['i1'][jjj]
                        i2 = f['i2'][jjj]
                        this_lpt_time = [dt.datetime(1970,1,1,0,0,0) + dt.timedelta(hours=int(x)) for x in ts[i1:i2+1]]

                        ii1 = east_prop_group_df_sort['begin_indx'].values[jj]
                        ii2 = east_prop_group_df_sort['end_indx'].values[jj]

                        year11 = east_prop_group_df_sort['year_begin'].values[jj]
                        month11 = east_prop_group_df_sort['month_begin'].values[jj]
                        day11 = east_prop_group_df_sort['day_begin'].values[jj]
                        hour11 = east_prop_group_df_sort['hour_begin'].values[jj]
                        year22 = east_prop_group_df_sort['year_end'].values[jj]
                        month22 = east_prop_group_df_sort['month_end'].values[jj]
                        day22 = east_prop_group_df_sort['day_end'].values[jj]
                        hour22 = east_prop_group_df_sort['hour_end'].values[jj]

                        FOUT_mjo_lpt += [ [dateint1, dateint2, east_prop_group_df_sort['lptid'].values[jj], this_group
                                            , east_prop_group_df_sort['total_duration'].values[jj]
                                            , east_prop_group_df_sort['total_zonal_spd'].values[jj]
                                            , year11, month11, day11, hour11
                                            , year22, month22, day22, hour22
                                            , east_prop_group_df_sort['begin_indx'].values[jj]
                                            , east_prop_group_df_sort['end_indx'].values[jj]
                                            , east_prop_group_df_sort['segment_zonal_spd'].values[jj]
                                            , east_prop_group_df_sort['duration'].values[jj]
                                            , this_lpt_time[ii1].year, this_lpt_time[ii1].month, this_lpt_time[ii1].day, this_lpt_time[ii1].hour
                                            , this_lpt_time[ii2].year, this_lpt_time[ii2].month, this_lpt_time[ii2].day, this_lpt_time[ii2].hour
                                            , east_prop_group_df_sort['begin_lon'].values[jj]
                                            , east_prop_group_df_sort['end_lon'].values[jj]
                                        ] ]

                    else:

                        year11 = east_prop_group_df_sort['year_begin'].values[jj]
                        month11 = east_prop_group_df_sort['month_begin'].values[jj]
                        day11 = east_prop_group_df_sort['day_begin'].values[jj]
                        hour11 = east_prop_group_df_sort['hour_begin'].values[jj]

                        year22 = east_prop_group_df_sort['year_end'].values[jj]
                        month22 = east_prop_group_df_sort['month_end'].values[jj]
                        day22 = east_prop_group_df_sort['day_end'].values[jj]
                        hour22 = east_prop_group_df_sort['hour_end'].values[jj]

                        ## Check whether this one has already been added.
                        if len(FOUT_mjo_group_extra_lpt) > 0:
                            if east_prop_group_df_sort['lptid'].values[jj] in np.array(FOUT_mjo_group_extra_lpt)[:,2]:
                                continue
                        if len(FOUT_mjo_lpt) > 0:
                            if east_prop_group_df_sort['lptid'].values[jj] in np.array(FOUT_mjo_lpt)[:,2]:
                                continue

                        FOUT_mjo_group_extra_lpt += [ [dateint1, dateint2, east_prop_group_df_sort['lptid'].values[jj], this_group
                                                , east_prop_group_df_sort['total_duration'].values[jj]
                                                , east_prop_group_df_sort['total_zonal_spd'].values[jj]
                                                , year11, month11, day11, hour11
                                                , year22, month22, day22, hour22
                                                , -999, -999, -999, -999
                                                , 9999,99,99,99
                                                , 9999,99,99,99
                                                , -999, -999
                                            ] ]


            else:

                for jj in range(len(east_prop_group_df)):

                    year11 = east_prop_group_df['year_begin'].values[jj]
                    month11 = east_prop_group_df['month_begin'].values[jj]
                    day11 = east_prop_group_df['day_begin'].values[jj]
                    hour11 = east_prop_group_df['hour_begin'].values[jj]

                    year22 = east_prop_group_df['year_end'].values[jj]
                    month22 = east_prop_group_df['month_end'].values[jj]
                    day22 = east_prop_group_df['day_end'].values[jj]
                    hour22 = east_prop_group_df['hour_end'].values[jj]

                    ## Check whether this one has already been added.
                    if len(FOUT_non_mjo_lpt) > 0:
                        if east_prop_group_df['lptid'].values[jj] in np.array(FOUT_non_mjo_lpt)[:,2]:
                            continue

                    FOUT_non_mjo_lpt += [ [dateint1, dateint2, east_prop_group_df['lptid'].values[jj], this_group
                                            , east_prop_group_df['total_duration'].values[jj]
                                            , east_prop_group_df['total_zonal_spd'].values[jj]
                                            , year11, month11, day11, hour11
                                            , year22, month22, day22, hour22
                                            , -999, -999, -999, -999
                                            , 9999,99,99,99
                                            , 9999,99,99,99
                                            , -999, -999
                                        ] ]

    ## Output
    ## For output table files.
    FMT=('%14d%14d%10.4f%10d%10.2f%16.2f  %4d%0.2d%0.2d%0.2d  %4d%0.2d%0.2d%0.2d '
            + '%20d%15d%11.2f%11.2f  %4d%0.2d%0.2d%0.2d  %4d%0.2d%0.2d%0.2d%20.1f%15.1f')

    header = 'begin_tracking  end_tracking     lptid  lptgroup  duration  mean_zonal_spd   lpt_begin     lpt_end      eprop_begin_idx  eprop_end_idx  eprop_spd  eprop_dur eprop_begin   eprop_end     eprop_lon_begin  eprop_lon_end'

    mjo_lpt_file = (lpt_systems_dir + '/mjo_lpt_list_'+prod+'_'+YMDH1_YMDH2+'.txt')
    if len(FOUT_mjo_lpt) > 0:
        print(mjo_lpt_file)
        np.savetxt(mjo_lpt_file, FOUT_mjo_lpt, fmt=FMT,header=header,comments='')

    mjo_lpt_group_extras_file = (lpt_systems_dir + '/mjo_group_extra_lpt_list_'+prod+'_'+YMDH1_YMDH2+'.txt')
    if len(FOUT_mjo_group_extra_lpt) > 0:
        print(mjo_lpt_group_extras_file)
        np.savetxt(mjo_lpt_group_extras_file, FOUT_mjo_group_extra_lpt, fmt=FMT,header=header,comments='')

    non_mjo_lpt_file = (lpt_systems_dir + '/non_mjo_lpt_list_'+prod+'_'+YMDH1_YMDH2+'.txt')
    if len(FOUT_non_mjo_lpt) > 0:
        print(non_mjo_lpt_file)
        np.savetxt(non_mjo_lpt_file, FOUT_non_mjo_lpt, fmt=FMT,header=header,comments='')
