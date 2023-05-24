import matplotlib; matplotlib.use('agg')
import numpy as np
import datetime as dt
import cftime
import scipy.ndimage
from scipy.signal import convolve2d
from scipy import ndimage
import matplotlib.pylab as plt
from netCDF4 import Dataset
import glob
import networkx as nx
import sys
import os

import lpt.readdata

###################################################################
#####################  General Use Functions  #####################
###################################################################


def str2cftime(date_string, fmt, calendar):
    """
    Example usage: cftime_datetime = str2cftime('2021010100', '%Y%m%d%H', 'standard')
    Converts string time (date_string) in a strftime format (fmt) to a cftime datetime,
    using the calendar specified.
    
    Unfortunately, cftime does not have a strptime function. This function uses
    datetime's strptime function as an intermediate step.
    """
    this_dt = dt.datetime.strptime(date_string, fmt)
    return cftime.datetime(this_dt.year,this_dt.month,this_dt.day,
            this_dt.hour,this_dt.minute,this_dt.second, calendar=calendar)


def dtrange(begin_datetime, ending_datetime, interval_hours):
    """
    Example Usage: dt_list = dtrange(cftime.datetime(2021,1,1,0), cftime.datetime(2021,2,1,0), 3)
    3-hourly times from 00Z 2021-1-1 to 21Z 2021-1-31 (excludes ending_datetime)

    Analagous to range and Numpy's arange function, but for datetime objects.

    Returns a list of datetime objects.
    
    Works for both datetime.datetime and cftime.datetime.
    """
    total_sec = (ending_datetime - begin_datetime).total_seconds()
    return [begin_datetime + dt.timedelta(seconds=int(x)) for x in np.arange(0, total_sec, 3600*interval_hours)]


###################################################################
######################  LP Object Functions  ######################
###################################################################


def calc_scaled_average(data_in_accumulation_period, factor):
    """
    accumulated_data = calc_accumulation(data_in accumulation_period, factor)

    Calculate the sum and multiply by the data time interval to get the accumulation.
    -- data_in_accumulation_period[t,y,x] is a 3D array.
    -- factor gets multiplied by the mean. E.g., if the data is rain rate in mm/h,
       using factor of 24 would be in mm/day.
    """

    return factor * np.nanmean(data_in_accumulation_period, axis=0)


def identify_lp_objects(field, threshold, min_points=1
                        , object_is_gt_threshold=True, verbose=False):

    """
    label_im = identify_lp_objects(lon, lat, field, threshold
                            , object_minimum_gridpoints=0
                            , object_is_gt_threshold=True)

    Given an input data field (e.g., already accumulated and filtered),
    identify the LP Objects in that field. Return an array the same size
    as field, but with values indexed by object IDs.
    """

    field_bw = 0 * field
    if object_is_gt_threshold:
        field_bw[(field > threshold)] = 1
    else:
        field_bw[(field < threshold)] = 1

    label_im, nb_labels = ndimage.label(field_bw)
    if verbose:
        print('Found '+str(nb_labels)+' objects.', flush=True) # how many regions?

    label_points = ndimage.sum(1, label_im, range(nb_labels+1))

    throw_away = [x for x in range(1, nb_labels+1) if label_points[x] < min_points]
    if len(throw_away) > 0:
        if verbose:
            if str(len(throw_away)) == 1:
                print('Discarding ' + str(len(throw_away)) + ' feature that was < ' + str(min_points) + ' points.',flush=True)
            else:
                print('Discarding ' + str(len(throw_away)) + ' features that were < ' + str(min_points) + ' points.',flush=True)
        for nn in throw_away:
            label_im[label_im == nn] = 0

        ## Re-order LP Object IDs.
        label_im_old = label_im.copy()
        id_list = sorted(np.unique(label_im_old))
        for nn in range(len(id_list)):
            label_im[label_im_old == id_list[nn]] = nn

    return label_im


def calc_grid_cell_area(lon, lat):

    """
    area = calc_grid_cell_area(lon, lat)

    Given lon and lat arrays, calculate the area of each grid cell.
    - lon and lat don't need to be a uniform grid, but they need to be increasing
      in both the x and y direction for this function to work.
    - If 1-D arrays are given, they will be converted to 2D using np.meshgrid.
    """

    area = None
    if lon.ndim == 1:
        print('ERROR: lon and lat must be 2D arrays for function calc_grid_cell_area.', flush=True)
    else:
        ny,nx = lon.shape
        dlon = 0.0*lon
        dlat = 0.0*lat

        dlon[:,1:nx-1] = abs(0.5*(lon[:,1:nx-1] + lon[:,2:nx]) - 0.5*(lon[:,0:nx-2] + lon[:,1:nx-1]))
        dlon[:,0] = dlon[:,1]
        dlon[:,nx-1] = dlon[:,nx-2]
        dlat[1:ny-1,:] = abs(0.5*(lat[1:ny-1,:] + lat[2:ny,:]) - 0.5*(lat[0:ny-2,:] + lat[1:ny-1,:]))
        dlat[0,:] = dlat[1,:]
        dlat[ny-1,:] = dlat[ny-2,:]

        area = (dlat*111.195) * (dlon*111.195*np.cos(np.pi*lat/180.0))

    return area


def filter_str(stdev):
    if type(stdev) == int:
        strout = 'g' + str(int(stdev))
    elif type(stdev) == list:
        strout = 'g' + str(int(stdev[0])) + 'x' + str(int(stdev[1]))
    else:
        print('Warning: Wrong data type!')
        strout = None
    return strout



def do_lpo_calc(end_of_accumulation_time0, begin_time, dataset, lpo_options, output, plotting):


    YMDH = end_of_accumulation_time0.strftime('%Y%m%d%H')
    YMDH_fancy = end_of_accumulation_time0.strftime('%Y-%m-%d %H:00 UTC')

    """
    Check whether the output file already exists.
    If it does and lpo_options['overwrite_existing_files'] is set to False,
    Skip processing this time.
    """
    objects_dir = (output['data_dir'] + '/' + dataset['label']
                    + '/' + filter_str(lpo_options['filter_stdev'])
                    + '_' + str(int(lpo_options['accumulation_hours'])) + 'h'
                    + '/thresh' + str(int(lpo_options['thresh']))
                    + '/objects/'
                    + end_of_accumulation_time0.strftime(output['sub_directory_format']))
    objects_fn = (objects_dir + '/objects_' + YMDH)
    if not lpo_options['overwrite_existing_files'] and os.path.exists(objects_fn):
        print('-- This time already has LPO step done. Skipping.')

    else:
        ## NOTE: In cold start mode, the begin_time is assumed to be the model initiation time!
        hours_since_init = (end_of_accumulation_time0 - begin_time).total_seconds()/3600
        if (hours_since_init < lpo_options['cold_start_const_period'] and lpo_options['cold_start_mode']):
            beginning_of_accumulation_time = begin_time
            end_of_accumulation_time = beginning_of_accumulation_time + dt.timedelta(hours=24)
        elif (hours_since_init >= lpo_options['cold_start_const_period']
                and hours_since_init <= lpo_options['accumulation_hours'] and lpo_options['cold_start_mode']):
            beginning_of_accumulation_time = begin_time
            end_of_accumulation_time = end_of_accumulation_time0
        else:
            end_of_accumulation_time = end_of_accumulation_time0
            beginning_of_accumulation_time = end_of_accumulation_time0 - dt.timedelta(hours=lpo_options['accumulation_hours'])

        hours_to_divide = (end_of_accumulation_time - beginning_of_accumulation_time).total_seconds()/3600.0


        #beginning_of_accumulation_time = end_of_accumulation_time - dt.timedelta(hours=lpo_options['accumulation_hours'])
        print(('LPO time period: ' + beginning_of_accumulation_time.strftime('%Y-%m-%d %H:00 UTC') + ' to '
                + end_of_accumulation_time.strftime('%Y-%m-%d %H:00 UTC') + '.'), flush=True)

        try:

            accumulation_hours = int((end_of_accumulation_time - beginning_of_accumulation_time).total_seconds()/3600.0)
            dt_list = [beginning_of_accumulation_time
                + dt.timedelta(hours=x) for x in np.arange(0,accumulation_hours
                                                    + dataset['data_time_interval'],dataset['data_time_interval']).astype('double')]

            ## Get accumulated rain. # So far used only for model run, e.g., CFS.
            data_collect = []
            count = 0

            dataset['datetime_init'] = begin_time
            for this_dt in reversed(dt_list):
                DATA_RAW = lpt.readdata.readdata(this_dt, dataset)
                DATA_RAW['data'] = np.array(DATA_RAW['data'].filled(fill_value=0.0))
                DATA_RAW['data'][~np.isfinite(DATA_RAW['data'])] = 0.0
                if count < 1:
                    data_collect = DATA_RAW['data'].copy()
                else:
                    data_collect += DATA_RAW['data']
                count += 1

            DATA_RUNNING = (data_collect/count) * lpo_options['multiply_factor'] # Get to the units you want for objects.
            print('Running mean done.',flush=True)

            ## Filter the data
            DATA_FILTERED = scipy.ndimage.gaussian_filter(DATA_RUNNING, lpo_options['filter_stdev']
                , order=0, output=None, mode='reflect', cval=0.0, truncate=lpo_options['filter_n_stdev_width'])
            print('filter done.',flush=True)

            ## Get LP objects.
            label_im = identify_lp_objects(DATA_FILTERED, lpo_options['thresh'], min_points=lpo_options['min_points'], verbose=dataset['verbose'])
            OBJ = calculate_lp_object_properties(DATA_RAW['lon'], DATA_RAW['lat']
                        , DATA_RAW['data'], DATA_RUNNING, DATA_FILTERED, label_im, 0
                        , end_of_accumulation_time0, verbose=True)
            OBJ['units_inst'] = dataset['field_units']
            OBJ['units_running'] = lpo_options['field_units']
            OBJ['units_filtered'] = lpo_options['field_units']

            print('objects properties.',flush=True)

            """
            Object Output files
            """

            os.makedirs(objects_dir, exist_ok = True)
            lpt.lptio.lp_objects_output_ascii(objects_fn, OBJ)
            lpt.lptio.lp_objects_output_netcdf(objects_fn + '.nc', OBJ)

            """
            Object Plot
            """
            if plotting['do_plotting']:
                fig1 = plt.figure(1, figsize = (8.5,4))
                # plt.figure(1)
                # fig1.clf()
                ax1 = fig1.add_subplot(111)
                lpt.plotting.plot_rain_map_with_filtered_contour(ax1
                        , DATA_RUNNING, OBJ
                        , plotting, lpo_options)
                ax1.set_title((dataset['label'].upper()
                                + str(lpo_options['accumulation_hours'])
                                + '-h Rain Rate and LP Objects\nEnding ' + YMDH_fancy))

                img_dir1 = (output['img_dir'] + '/' + dataset['label']
                                + '/' + filter_str(lpo_options['filter_stdev'])
                                + '_' + str(int(lpo_options['accumulation_hours'])) + 'h'
                                + '/thresh' + str(int(lpo_options['thresh']))
                                + '/objects/'
                                + end_of_accumulation_time0.strftime(output['sub_directory_format']))

                os.makedirs(img_dir1, exist_ok = True)
                file_out_base = (img_dir1 + '/lp_objects_' + dataset['label'] + '_' + YMDH)
                lpt.plotting.print_and_save(file_out_base)
                fig1.clf()

        except FileNotFoundError:
            print('Data not yet available up to this point. Skipping.')





def calculate_lp_object_properties(lon, lat, field, field_running, field_filtered, label_im
                        , object_minimum_gridpoints, end_of_accumulation_time
                        , verbose=False):

    nb_labels = np.max(label_im)
    mask = 1*label_im
    mask[label_im > 0] = 1

    ## If lon and lat not in 2d arrays, put them through np.meshgrid.
    if lon.ndim == 1:
        if verbose:
            print('Detected 1-D lat/lon. Using np.meshgrid to get 2d lat/lon.', flush=True)
        lon2, lat2 = np.meshgrid(lon, lat)
    else:
        lon2 = lon
        lat2 = lat

    X2, Y2 = np.meshgrid(np.arange(lon2.shape[1]), np.arange(lon2.shape[0]))

    area2d = calc_grid_cell_area(lon2, lat2)

    sizes = ndimage.sum(mask, label_im, range(1, nb_labels + 1))

    amean_instantaneous_field = ndimage.sum(field*area2d, label_im, range(1, nb_labels + 1)) / ndimage.sum(area2d, label_im, range(1, nb_labels + 1))
    amean_running_field = ndimage.sum(field_running*area2d, label_im, range(1, nb_labels + 1)) / ndimage.sum(area2d, label_im, range(1, nb_labels + 1))
    amean_filtered_running_field = ndimage.sum(field_filtered*area2d, label_im, range(1, nb_labels + 1)) / ndimage.sum(area2d, label_im, range(1, nb_labels + 1))
    min_instantaneous_field = ndimage.minimum(field, label_im, range(1, nb_labels + 1))
    min_running_field = ndimage.minimum(field_running, label_im, range(1, nb_labels + 1))
    min_filtered_running_field = ndimage.minimum(field_filtered, label_im, range(1, nb_labels + 1))
    max_instantaneous_field = ndimage.maximum(field, label_im, range(1, nb_labels + 1))
    max_running_field = ndimage.maximum(field_running, label_im, range(1, nb_labels + 1))
    max_filtered_running_field = ndimage.maximum(field_filtered, label_im, range(1, nb_labels + 1))

    centroid_lon = ndimage.sum(lon2*area2d, label_im, range(1, nb_labels + 1)) / ndimage.sum(area2d, label_im, range(1, nb_labels + 1))
    centroid_lat = ndimage.sum(lat2*area2d, label_im, range(1, nb_labels + 1)) / ndimage.sum(area2d, label_im, range(1, nb_labels + 1))
    centroid_x = ndimage.sum(X2*area2d, label_im, range(1, nb_labels + 1)) / ndimage.sum(area2d, label_im, range(1, nb_labels + 1))
    centroid_y = ndimage.sum(Y2*area2d, label_im, range(1, nb_labels + 1)) / ndimage.sum(area2d, label_im, range(1, nb_labels + 1))
    area = ndimage.sum(area2d, label_im, range(1, nb_labels + 1))
    max_lon = ndimage.maximum(lon2, label_im, range(1, nb_labels + 1))
    min_lon = ndimage.minimum(lon2, label_im, range(1, nb_labels + 1))
    max_lat = ndimage.maximum(lat2, label_im, range(1, nb_labels + 1))
    min_lat = ndimage.minimum(lat2, label_im, range(1, nb_labels + 1))

    ## Assign LPT IDs. Order is by longitude. Use zero-base indexing.
    id0 = 1e10 * end_of_accumulation_time.year + 1e8 * end_of_accumulation_time.month + 1e6 * end_of_accumulation_time.day + 1e4 * end_of_accumulation_time.hour

    id = id0 + np.arange(len(centroid_lon))

    ## Prepare output dict.
    OBJ={}
    OBJ['id'] = id
    OBJ['label_im'] = label_im
    OBJ['lon'] = centroid_lon
    OBJ['lat'] = centroid_lat
    OBJ['min_lon'] = min_lon
    OBJ['min_lat'] = min_lat
    OBJ['max_lon'] = max_lon
    OBJ['max_lat'] = max_lat
    OBJ['x'] = centroid_x
    OBJ['y'] = centroid_y
    OBJ['n_points'] = sizes
    OBJ['area'] = area

    OBJ['amean_inst_field'] = amean_instantaneous_field
    OBJ['amean_running_field'] = amean_running_field
    OBJ['amean_filtered_running_field'] = amean_filtered_running_field
    OBJ['min_inst_field'] = min_instantaneous_field
    OBJ['min_running_field'] = min_running_field
    OBJ['min_filtered_running_field'] = min_filtered_running_field
    OBJ['max_inst_field'] = max_instantaneous_field
    OBJ['max_running_field'] = max_running_field
    OBJ['max_filtered_running_field'] = max_filtered_running_field

    ## Edge points to go along with the min_lon, max_lon, min_lat, max_lat.
    w_edge_lat = 0.0 * OBJ['lon']
    e_edge_lat = 0.0 * OBJ['lon']
    s_edge_lon = 0.0 * OBJ['lon']
    n_edge_lon = 0.0 * OBJ['lon']
    for ii in range(len(OBJ['lon'])):
        ypoints, xpoints = np.where(OBJ['label_im'] == ii+1)
        lon_points = np.array([lon2[ypoints[x], xpoints[x]] for x in range(len(xpoints))])
        lat_points = np.array([lat2[ypoints[x], xpoints[x]] for x in range(len(xpoints))])
        w_edge_lat[ii] = np.mean(lat_points[lon_points < np.nanmin(lon_points) + 0.001])
        e_edge_lat[ii] = np.mean(lat_points[lon_points > np.nanmax(lon_points) - 0.001])
        s_edge_lon[ii] = np.mean(lon_points[lat_points < np.nanmin(lat_points) + 0.001])
        n_edge_lon[ii] = np.mean(lon_points[lat_points > np.nanmax(lat_points) - 0.001])

    OBJ['westmost_lat'] = w_edge_lat
    OBJ['eastmost_lat'] = e_edge_lat
    OBJ['southmost_lon'] = s_edge_lon
    OBJ['northmost_lon'] = n_edge_lon

    # Grid stuff.
    OBJ['grid'] = {}
    OBJ['grid']['lon'] = lon
    OBJ['grid']['lat'] = lat
    OBJ['grid']['area'] = area2d

    return OBJ


def get_objid_datetime(objid,calendar='standard'):
    """
    usge: this_datetime = get_objid_datetime(this_objid, calendar)

    Get the datetime from an objid of form YYYYMMDDHHnnnn.
    """
    ymdh_int = int(np.floor(objid/1e4))
    ymdh_str = str(ymdh_int).zfill(10)
    return str2cftime(ymdh_str, '%Y%m%d%H', calendar)


def read_lp_object_properties(objid, objdir, property_list, verbose=False, fmt="/%Y/%m/%Y%m%d/objects_%Y%m%d%H.nc"):

    dt1 = get_objid_datetime(objid)
    fn1 = (objdir + '/' + dt1.strftime(fmt)).replace('///','/').replace('//','/')

    if verbose:
        print(fn1)

    DS1 = Dataset(fn1)
    id1 = DS1['objid'][:]
    idx1, = np.where(np.abs(id1 - objid) < 0.1)

    out_dict = {}
    for property in property_list:
        out_dict[property] = to1d(DS1[property][:][idx1])
    DS1.close()

    return out_dict


def get_latest_lp_object_time(objdir, level=3):
    obj_file_list = sorted(glob.glob((objdir + "/*"*level + "/*.nc")))
    last_obj_file = obj_file_list[-1]
    return dt.datetime.strptime(last_obj_file[-13:-3], "%Y%m%d%H")


##################################################################
######################  Tracking Functions  ######################
##################################################################

def to1d(ndarray_or_ma):
    try:
        fout = ndarray_or_ma.compressed()
    except:
        fout = ndarray_or_ma.flatten()
    return fout


def calc_overlapping_points(objid1, objid2, objdir, fmt="/%Y/%m/%Y%m%d/objects_%Y%m%d%H.nc"):

    dt1 = get_objid_datetime(objid1)
    dt2 = get_objid_datetime(objid2)

    fn1 = (objdir + '/' + dt1.strftime(fmt)).replace('///','/').replace('//','/')
    fn2 = (objdir + '/' + dt2.strftime(fmt)).replace('///','/').replace('//','/')

    DS1 = Dataset(fn1)
    id1 = DS1['objid'][:]
    idx1, = np.where(id1 == objid1)
    x1 = to1d(DS1['pixels_x'][:][idx1])
    y1 = to1d(DS1['pixels_y'][:][idx1])
    DS1.close()


    DS2 = Dataset(fn2)
    id2 = DS2['objid'][:]
    idx2, = np.where(id2 == objid2)
    x2 = to1d(DS2['pixels_x'][:][idx2])
    y2 = to1d(DS2['pixels_y'][:][idx2])
    DS2.close()

    xy1 = set(zip(x1,y1))
    xy2 = set(zip(x2,y2))

    overlap = [x in xy2 for x in xy1]

    OUT = (len(x1), len(x2), np.sum(overlap))
    del x1
    del y1
    del x2
    del y2
    del xy1
    del xy2

    return OUT



def init_lpt_graph(dt_list, objdir, min_points = 1, fmt = "/%Y/%m/%Y%m%d/objects_%Y%m%d%H.nc"):

    G = nx.DiGraph() # Empty graph
    REFTIME = cftime.datetime(1970,1,1,0,0,0,calendar=dt_list[0].calendar) ## Only used internally.

    for this_dt in dt_list:

        print(this_dt)
        fn = (objdir + '/' + this_dt.strftime(fmt)).replace('///','/').replace('//','/')
        print(fn)
        try:
            DS = Dataset(fn)
            try:
                id_list = DS['objid'][:]
                lon = DS['centroid_lon'][:]
                lat = DS['centroid_lat'][:]
                area = DS['area'][:]
                pixels_x = DS['pixels_x'][:]
            except IndexError:
                print('WARNING: No LPO at this time: ' + str(this_dt),flush=True)
                id_list = [] # In case of no LPOs at this time.
            DS.close()

            for ii in range(len(id_list)):
                npts = pixels_x[ii,:].count()  #ma.count() for number of non masked values.
                if npts >= min_points:
                    G.add_node(int(id_list[ii]), timestamp=(this_dt - REFTIME).total_seconds()
                        , lon = lon[ii], lat=lat[ii], area=area[ii]
                        , pos = (lon[ii], (this_dt - REFTIME).total_seconds()))

        except FileNotFoundError:
            print('WARNING: Missing this file!',flush=True)

    return G

def get_lpo_overlap(dt1, dt2, objdir, min_points=1, fmt="/%Y/%m/%Y%m%d/objects_%Y%m%d%H.nc"):

    ##
    ## Read in LPO masks for current and previous times.
    ##
    fn1 = (objdir + '/' + dt1.strftime(fmt)).replace('///','/').replace('//','/')
    fn2 = (objdir + '/' + dt2.strftime(fmt)).replace('///','/').replace('//','/')

    DS1 = Dataset(fn1, 'r')
    mask1 = DS1['grid_mask'][:]
    objid1 = DS1['objid'][:]
    DS1.close()

    DS2 = Dataset(fn2, 'r')
    mask2 = DS2['grid_mask'][:]
    objid2 = DS2['objid'][:]
    DS2.close()

    ##
    ## Apply minimum size.
    ## Any LPOs smaller than the minimum size get taken out of the mask.
    ## Their mask values get set to zero, and they will not be considered
    ## for overlapping.
    ##
    if min_points > 1:
        sizes = ndimage.sum(1, mask1, range(np.nanmax(mask1)+1))
        for nn in [x for x in range(len(sizes)) if sizes[x] < min_points]:
            mask1[mask1 == nn] = -1

        sizes = ndimage.sum(1, mask2, range(np.nanmax(mask2)+1))
        for nn in [x for x in range(len(sizes)) if sizes[x] < min_points]:
            mask2[mask2 == nn] = -1

    ##
    ## Each overlap must necessarily be one LPO against another single LPO.
    ##
    overlap = np.logical_and(mask1 > -1, mask2 > -1)
    label_im, nb_labels = ndimage.label(overlap)

    ########################################################################
    ## Construct overlapping points "look up table" array.
    ## Then, we will use this array as a look up table for specific LPOs.
    ##   -----------> objid2
    ##   |
    ##   |
    ##   |
    ##   v
    ## objid1
    ########################################################################

    overlapping_npoints = np.zeros([len(objid1), len(objid2)])
    overlapping_frac1 = np.zeros([len(objid1), len(objid2)])
    overlapping_frac2 = np.zeros([len(objid1), len(objid2)])

    for nn in range(1,nb_labels+1):
        ## Figure out which LPOs this represents.
        ii = int(ndimage.maximum(mask1, label_im, nn))
        jj = int(ndimage.maximum(mask2, label_im, nn))

        overlapping_npoints[ii,jj] += ndimage.sum(overlap, label_im, nn)
        overlapping_frac1[ii,jj] = overlapping_npoints[ii,jj] / np.sum(mask1==ii)
        overlapping_frac2[ii,jj] = overlapping_npoints[ii,jj] / np.sum(mask2==jj)

    ## Prepare outputs.
    OVERLAP={}
    OVERLAP['npoints'] = overlapping_npoints
    OVERLAP['frac1'] = overlapping_frac1
    OVERLAP['frac2'] = overlapping_frac2
    return OVERLAP


def connect_lpt_graph(G0, options, min_points=1, verbose=False, fmt="/%Y/%m/%Y%m%d/objects_%Y%m%d%H.nc"):

    """
    usage: LPT = calc_lpt_group_array(LPT, objdir, options)
    Calculate the simple LPT groups.

    options dictionary entries needed:
    options['objdir']
    options['min_overlap_points']
    options['min_overlap_frac']

    "LPT" is a 2-D "group" array (np.int64) with columns: [timestamp, objid, lpt_group_id, begin_point, end_point, split_point]
    -- timestamp = Linux time stamp (e.g., seconds since 00 UTC 1970-1-1)
    -- objid = LP object id (YYYYMMDDHHnnnn)
    -- lpt_group_id = LPT group id, connected LP objects have a common LPT group id.
    -- begin point = 1 if it is the beginning of a track. 0 otherwise.
    -- end point = 1 if no tracks were connected to it, 0 otherwise.
    -- split point = 1 if split detected, 0 otherwise.

    BRANCHES is a 1-D native Python list with native Python int values.
    This is needed because BRANCHES is bitwise, and there can be more than 64 branches in a group.
    -- branches = bitwise binary starts from 1 at each branch. Mergers will have separate branch numbers.
                   overlapping portions will have multiple branch numbers associated with them.
    """

    # Make copies to avoid immutability weirdness.
    Gnew = G0.copy()
    objdir = options['objdir']

    lpo_id_list = list(G0.nodes())
    datetime_list = [get_objid_datetime(x,options['calendar']) for x in lpo_id_list]
    timestamp_list = [int((x - cftime.datetime(1970,1,1,0,0,0,calendar=datetime_list[0].calendar)).total_seconds()) for x in datetime_list]

    ## Now, loop through the times.
    unique_timestamp_list = np.unique(timestamp_list)
    for tt in range(1,len(unique_timestamp_list)):

        ## Datetimes for this time and previous time.
        this_timestamp = unique_timestamp_list[tt]
        prev_timestamp = unique_timestamp_list[tt-1]
        this_dt = cftime.datetime(1970,1,1,0,0,0,calendar=options['calendar']) + dt.timedelta(seconds=int(this_timestamp))
        prev_dt = cftime.datetime(1970,1,1,0,0,0,calendar=options['calendar']) + dt.timedelta(seconds=int(prev_timestamp))
        print(this_dt, flush=True)

        ## Get overlap points.
        OVERLAP = get_lpo_overlap(this_dt, prev_dt, objdir, fmt=fmt, min_points = min_points)
        overlapping_npoints = OVERLAP['npoints']
        overlapping_frac1 = OVERLAP['frac1']
        overlapping_frac2 = OVERLAP['frac2']

        ## The indices (e.g., LPT and BRANCHES array rows) for these times.
        this_time_idx, = np.where(timestamp_list == this_timestamp)
        prev_time_idx, = np.where(timestamp_list == prev_timestamp)

        for ii in this_time_idx:
            this_objid = lpo_id_list[ii]
            idx1 = int(str(this_objid)[-4:])

            ## 1) Figure out which "previous time" LPT indices overlap.
            matches = []
            for jj in prev_time_idx:
                prev_objid = lpo_id_list[jj]
                idx2 = int(str(prev_objid)[-4:])

                n_overlap = overlapping_npoints[idx1,idx2]
                frac1 = overlapping_frac1[idx1,idx2]
                frac2 = overlapping_frac2[idx1,idx2]

                if n_overlap >= options['min_overlap_points']:
                    matches.append(jj)
                elif 1.0*frac1 > options['min_overlap_frac']:
                    matches.append(jj)
                elif 1.0*frac2 > options['min_overlap_frac']:
                    matches.append(jj)
            if verbose:
                print(str(this_objid),flush=True)

            ## 2) Link the previous LPT Indices to the group.
            ##    If no matches, it will skip this loop.
            for match in matches:
                if verbose:
                    print(' --> with: ' + str(lpo_id_list[match]),flush=True)
                ## Add it as a graph edge.
                Gnew.add_edge(lpo_id_list[match],this_objid)

    return Gnew


def lpt_graph_allow_falling_below_threshold(G, options, min_points=1, fmt="/%Y/%m/%Y%m%d/objects_%Y%m%d%H.nc", verbose=False):
    """
    Check duration of "leaf" (e.g., "this") to "root" nodes of other DAGs, and connect if less than
    center_jump_max_hours.
    """

    objdir=options['objdir']
    # Get connected components of graph.
    CC = list(nx.connected_components(nx.to_undirected(G)))
    SG = [G.subgraph(CC[x]).copy() for x in range(len(CC))]

    for kk in range(len(SG)):

        end_nodes = [x for x in SG[kk].nodes() if SG[kk].out_degree(x)==0 and SG[kk].in_degree(x)>=1]
        if len(end_nodes) < 0:
            continue

        for ll in range(len(SG)):
            if ll == kk:
                continue

            begin_nodes = [x for x in SG[ll].nodes() if SG[ll].out_degree(x)>=1 and SG[ll].in_degree(x)==0]
            if len(begin_nodes) < 0:
                continue

            for kkkk in end_nodes:
                kkkk_idx = kkkk - int(1000*np.floor(kkkk/1000))
                for llll in begin_nodes:

                    # When end nodes are linked to begin nodes, they are
                    # removed from the list of end/begin nodes.
                    # Therefore, check whether these have already been removed.
                    # If they were already removed, it means the edge was already
                    # added to the graph, and we want to avoid the end node from
                    # potentially being linked to another begin node further upstream.
                    # When this happened, I was getting duplicate LPTs.
                    if not kkkk in end_nodes or not llll in begin_nodes:
                        continue

                    llll_idx = llll - int(1000*np.floor(llll/1000))
                    hours_diff = (get_objid_datetime(llll)-get_objid_datetime(kkkk)).total_seconds()/3600.0
                    if hours_diff > 0.1 and hours_diff < options['fall_below_threshold_max_hours']+0.1:

                        begin_dt = get_objid_datetime(llll)
                        end_dt = get_objid_datetime(kkkk)

                        OVERLAP = get_lpo_overlap(end_dt, begin_dt, objdir, fmt=fmt, min_points = min_points)
                        overlapping_npoints = OVERLAP['npoints']
                        overlapping_frac1 = OVERLAP['frac1']
                        overlapping_frac2 = OVERLAP['frac2']

                        n_overlap = overlapping_npoints[kkkk_idx, llll_idx]
                        frac1 = overlapping_frac1[kkkk_idx, llll_idx]
                        frac2 = overlapping_frac2[kkkk_idx, llll_idx]

                        if n_overlap >= options['min_overlap_points']:
                            print('Overlap: '+ str(kkkk) + ' --> ' + str(llll) + '!', flush=True)
                            G.add_edge(kkkk,llll)
                            end_nodes.remove(kkkk)
                            begin_nodes.remove(llll)
                        elif 1.0*frac1 > options['min_overlap_frac']:
                            print('Overlap: '+ str(kkkk) + ' --> ' + str(llll) + '!', flush=True)
                            G.add_edge(kkkk,llll)
                            end_nodes.remove(kkkk)
                            begin_nodes.remove(llll)
                        elif 1.0*frac2 > options['min_overlap_frac']:
                            print('Overlap: '+ str(kkkk) + ' --> ' + str(llll) + '!', flush=True)
                            G.add_edge(kkkk,llll)
                            end_nodes.remove(kkkk)
                            begin_nodes.remove(llll)

    return G


def lpt_graph_remove_short_duration_systems(G, min_duration
                        , latest_datetime = dt.datetime(3000,1,1,0,0,0)):

    nodes_master_list = list(G.nodes())
    Gundirected = nx.to_undirected(G)

    while(len(nodes_master_list) > 0):
        print(('N = ' + str(len(nodes_master_list)) + ' nodes left to check.'), flush=True)
        nodes_component_list = list(nx.node_connected_component(Gundirected, nodes_master_list[0]))

        min_lpo_id = np.nanmin(nodes_component_list)
        max_lpo_id = np.nanmax(nodes_component_list)
        min_dt = get_objid_datetime(min_lpo_id)
        max_dt = get_objid_datetime(max_lpo_id)
        duration = (max_dt - min_dt).total_seconds()/3600.0
        if duration < min_duration - 0.1:
            G.remove_nodes_from(nodes_component_list)

        for x in nodes_component_list:
            nodes_master_list.remove(x)

    return G


def get_short_ends(G):

    Grev = G.reverse() # Reversed graph is used for splits below.

    ## Break in to individual paths (e.g., LPT branches).
    roots = []
    leaves = []
    for node in G.nodes:
        if G.in_degree(node) == 0: # it's a root
            roots.append(node)
        elif G.out_degree(node) == 0: # it's a leaf
            leaves.append(node)

    ## Root short ends -- mergers.
    Plist_mergers = []
    for root in roots:
        this_short_end = [root]
        this_node = root
        more_to_do = True
        while more_to_do:
            more_to_do = False
            next_node = list(G[this_node])[0]
            this_short_end.append(next_node)
            if G.in_degree(next_node) == 1 and G.out_degree(next_node) == 1:
                this_node = next_node
                more_to_do = True
        Plist_mergers.append(this_short_end)

    ## Leaf short ends -- splits.
    Plist_splits = []
    for leaf in leaves:
        this_short_end = [leaf]
        this_node = leaf
        more_to_do = True
        while more_to_do:
            more_to_do = False
            next_node = list(Grev[this_node])[0]
            this_short_end.append(next_node)
            if Grev.in_degree(next_node) == 1 and Grev.out_degree(next_node) == 1:
                this_node = next_node
                more_to_do = True
        Plist_splits.append(this_short_end)

    return (Plist_mergers, Plist_splits)


def lpt_graph_remove_short_ends(G, min_duration_to_keep):

    ## Work on each connected component (DAG, directed acyclical graph) separately.
    ## First, I get a list of DAG sub groups to loop through.
    CC = list(nx.connected_components(nx.to_undirected(G)))
    SG = [G.subgraph(CC[x]).copy() for x in range(len(CC))]

    ## Loop over each DAG (LPG Group)
    for kk in range(len(SG)):
        more_to_do = True
        print('--> LPT group ' + str(kk+1) + ' of ' + str(len(SG)),flush=True)
        niter = 0
        while more_to_do:
            niter += 1
            more_to_do = False
            areas = nx.get_node_attributes(SG[kk],'area') # used for tie breaker if same duration

            Plist_mergers, Plist_splits = get_short_ends(SG[kk])
            print('----> Iteration #'+str(niter)+': Found '+str(len(Plist_mergers))+' merge ends and '+str(len(Plist_splits))+' split ends.',flush=True)

            nodes_to_remove = []
            ## Handle mergers.
            if len(Plist_mergers) > 1: # Don't bother if only one root short end.

                merger_datetimes = [get_objid_datetime(x[-1]) for x in Plist_mergers]
                merger_timestamps = np.array([(x - cftime.datetime(1970,1,1,0,0,0,calendar=merger_datetimes[0].calendar)).total_seconds()/3600 for x in merger_datetimes])
                for iiii in range(len(Plist_mergers)):
                    path1 = Plist_mergers[iiii]
                    # Don't use the last node, as it intersects the paths I want to keep.
                    dur1 = (get_objid_datetime(path1[-2]) - get_objid_datetime(path1[0])).total_seconds()/3600.0

                    ## Check whether intersections with any others
                    override_removal = False
                    others = list(range(len(Plist_mergers)))
                    others.remove(iiii)
                    found_intersecting_short_end = False
                    for jjjj in others:
                        path2 = Plist_mergers[jjjj]
                        if path1[-1] == path2[-1]: #Make sure I am comparing short ends that TOUCH.
                            found_intersecting_short_end = True
                            dur2 = (get_objid_datetime(path2[-2]) - get_objid_datetime(path2[0])).total_seconds()/3600.0
                            if dur1 > dur2:
                                override_removal = True
                            elif dur1 == dur2:
                                ## Tiebreaker is integrated area in time.
                                integrate_area1 = np.nansum([areas[x] for x in path1])
                                integrate_area2 = np.nansum([areas[x] for x in path2])
                                if integrate_area1 >= integrate_area2:
                                    override_removal = True
                    if not found_intersecting_short_end:
                        # Check if it is the earliest merger time.
                        if merger_timestamps[iiii] == np.min(merger_timestamps):
                            override_removal = True

                    if dur1 < min_duration_to_keep - 0.01 and not override_removal:
                        ## NOTE: KC20 code used min_duration_to_keep PLUS 0.1 above.
                        ##       Hence, the parameter wasn't strictly the *minimum*
                        ##       duration to keep a branch, since short ends
                        ##       with duration *exactly* equal to min_duration_to_keep
                        ##       would have been discarded.

                        ## Make sure I wouldn't remove any parts of the cycles
                        nodes_to_remove += path1[:-1] # Don't remove the last one. It intersects the paths I want to keep.

            ## Handle splits. NOTE: The ordering here is REVERSED in time.
            if len(Plist_splits) > 1: # Don't bother if only one leaf short end.

                split_datetimes = [get_objid_datetime(x[-1]) for x in Plist_splits]
                split_timestamps = np.array([(x - cftime.datetime(1970,1,1,0,0,0,calendar=split_datetimes[0].calendar)).total_seconds()/3600 for x in split_datetimes])

                for iiii in range(len(Plist_splits)):
                    path1 = Plist_splits[iiii]
                    # Don't use the last node, as it intersects the paths I want to keep.
                    dur1 = (get_objid_datetime(path1[0]) - get_objid_datetime(path1[-2])).total_seconds()/3600.0

                    ## Check whether intersections with any others
                    override_removal = False
                    others = list(range(len(Plist_splits)))
                    others.remove(iiii)
                    found_intersecting_short_end = False
                    for jjjj in others:
                        path2 = Plist_splits[jjjj]
                        if path1[-1] == path2[-1]:  #Make sure I am comparing short ends that TOUCH.
                                                    # using index [-1] works here because order is reversed, from get_short_ends
                            found_intersecting_short_end = True
                            dur2 = (get_objid_datetime(path2[0]) - get_objid_datetime(path2[-2])).total_seconds()/3600.0
                            if dur1 > dur2:
                                override_removal = True
                            elif dur1 == dur2:
                                ## Tiebreaker is integrated area in time.
                                integrate_area1 = np.nansum([areas[x] for x in path1])
                                integrate_area2 = np.nansum([areas[x] for x in path2])
                                if integrate_area1 >= integrate_area2:
                                    override_removal = True
                    if not found_intersecting_short_end:
                        # Check if it is the latest split time.
                        if split_timestamps[iiii] == np.max(split_timestamps):
                            override_removal = True


                    if dur1 < min_duration_to_keep + 0.1 and not override_removal:
                        ## Make sure I wouldn't remove any parts of the cycles
                        nodes_to_remove += path1[:-1] # Don't remove the last one. It intersects the paths I want to keep.

            if len(nodes_to_remove) > 0:
                G.remove_nodes_from(nodes_to_remove)
                SG[kk].remove_nodes_from(nodes_to_remove)

                ## In some situations, the branch removal process leaves behind "isolates"
                ##  -- Nodes without any neighbors! This has occurred in the following situation:
                ##                   *
                ##                   *
                ##                   *     ^
                ##                   *    * *
                ##                   *   *   *
                ##                 *   *
                ##               *
                ##             *
                ##             *
                ## Where ^ is the node that was left "stranded", e.g, the isolate.
                ## To handle this, remove any isolates from main graph and current sub graph.
                isolates_list = list(nx.isolates(SG[kk]))
                if len(isolates_list) > 0:
                    G.remove_nodes_from(isolates_list)
                    SG[kk].remove_nodes_from(isolates_list)

                more_to_do = True

    return G


def initialize_time_cluster_fields(TC, length):

    ## Fields initialized to zero.
    for field in ['nobj','area','centroid_lon','centroid_lat'
                ,'largest_object_centroid_lon','largest_object_centroid_lat'
                ,'amean_inst_field','amean_running_field','amean_filtered_running_field']:
        TC[field] = np.zeros(length)

    ## Fields initialized to 999.0.
    for field in ['min_lon','min_lat','westmost_lat', 'southmost_lon'
                ,'min_inst_field','min_running_field'
                ,'min_filtered_running_field']:
        TC[field] =  999.0 * np.ones(length)

    ## Fields initialized to -999.0.
    for field in ['max_lon','max_lat','eastmost_lat', 'northmost_lon'
                ,'max_inst_field','max_running_field'
                ,'max_filtered_running_field']:
        TC[field] = -999.0 * np.ones(length)

    return TC


def calc_lpt_properties_without_branches(G, options, fmt="/%Y/%m/%Y%m%d/objects_%Y%m%d%H.nc"):
    ## The branch nodes of G have the properties of timestamp, lon, lat, and area.
    TC_all = []

    ## First, I get a list of DAG sub groups to loop through.
    CC = list(nx.connected_components(nx.to_undirected(G)))
    SG = [G.subgraph(CC[x]).copy() for x in range(len(CC))]

    ## Loop over each DAG
    for kk in range(len(SG)):
        print('--> LPT group ' + str(kk+1) + ' of ' + str(len(SG)),flush=True)

        TC_this = {}
        TC_this['lpt_group_id'] = kk
        TC_this['lpt_id'] = 1.0*kk

        TC_this['objid'] = sorted(list(SG[kk].nodes()))
        ts=nx.get_node_attributes(SG[kk],'timestamp')
        timestamp_all = [ts[x] for x in TC_this['objid']]
        TC_this['timestamp'] = np.unique(timestamp_all)
        TC_this['datetime'] = [cftime.datetime(1970,1,1,0,0,0,calendar=options['calendar']) + dt.timedelta(seconds=int(x)) for x in TC_this['timestamp']]

        ##
        ## Sum/average the LPTs to get bulk/mean properties at each time.
        ##

        ## Initialize
        TC_this = initialize_time_cluster_fields(TC_this, len(TC_this['timestamp']))

        ## Loop over unique time stamps.
        for tt in range(len(TC_this['timestamp'])):
            max_area_already_used = -999.0
            this_objid_list = [TC_this['objid'][x] for x in range(len(TC_this['objid'])) if timestamp_all[x] == TC_this['timestamp'][tt]]
            for this_objid in this_objid_list:

                OBJ = read_lp_object_properties(this_objid, options['objdir']
                        , ['centroid_lon','centroid_lat','area','pixels_x','pixels_y'
                        ,'min_lon','max_lon','min_lat','max_lat'
                        ,'westmost_lat','eastmost_lat'
                        ,'southmost_lon','northmost_lon'
                        ,'amean_inst_field','amean_running_field','max_inst_field','max_running_field'
                        ,'min_inst_field','min_running_field','min_filtered_running_field'
                        ,'amean_filtered_running_field','max_filtered_running_field'], fmt=fmt)

                TC_this['nobj'][tt] += 1
                TC_this['area'][tt] += OBJ['area']
                TC_this['centroid_lon'][tt] += OBJ['centroid_lon'] * OBJ['area']
                TC_this['centroid_lat'][tt] += OBJ['centroid_lat'] * OBJ['area']
                if OBJ['area'] > max_area_already_used:
                    TC_this['largest_object_centroid_lon'][tt] = 1.0*OBJ['centroid_lon']
                    TC_this['largest_object_centroid_lat'][tt] = 1.0*OBJ['centroid_lat']
                    max_area_already_used = 1.0*OBJ['area']

                if OBJ['min_lon'] < TC_this['min_lon'][tt]:
                    TC_this['westmost_lat'][tt] = OBJ['westmost_lat']
                if OBJ['max_lon'] > TC_this['max_lon'][tt]:
                    TC_this['eastmost_lat'][tt] = OBJ['eastmost_lat']
                if OBJ['min_lat'] < TC_this['min_lat'][tt]:
                    TC_this['southmost_lon'][tt] = OBJ['southmost_lon']
                if OBJ['max_lat'] > TC_this['max_lat'][tt]:
                    TC_this['northmost_lon'][tt] = OBJ['northmost_lon']

                TC_this['min_lon'][tt] = min((TC_this['min_lon'][tt], OBJ['min_lon']))
                TC_this['min_lat'][tt] = min((TC_this['min_lat'][tt], OBJ['min_lat']))
                TC_this['max_lon'][tt] = max((TC_this['max_lon'][tt], OBJ['max_lon']))
                TC_this['max_lat'][tt] = max((TC_this['max_lat'][tt], OBJ['max_lat']))

                TC_this['amean_inst_field'][tt] += OBJ['amean_inst_field'] * OBJ['area']
                TC_this['amean_running_field'][tt] += OBJ['amean_running_field'] * OBJ['area']
                TC_this['amean_filtered_running_field'][tt] += OBJ['amean_filtered_running_field'] * OBJ['area']
                TC_this['min_inst_field'][tt] = min((TC_this['min_inst_field'][tt], OBJ['min_inst_field']))
                TC_this['min_running_field'][tt] = min((TC_this['min_running_field'][tt], OBJ['min_running_field']))
                TC_this['min_filtered_running_field'][tt] = min((TC_this['min_filtered_running_field'][tt], OBJ['min_filtered_running_field']))
                TC_this['max_inst_field'][tt] = max((TC_this['max_inst_field'][tt], OBJ['max_inst_field']))
                TC_this['max_running_field'][tt] = max((TC_this['max_running_field'][tt], OBJ['max_running_field']))
                TC_this['max_filtered_running_field'][tt] = max((TC_this['max_filtered_running_field'][tt], OBJ['max_filtered_running_field']))

            TC_this['centroid_lon'][tt] /= TC_this['area'][tt]
            TC_this['centroid_lat'][tt] /= TC_this['area'][tt]

            TC_this['amean_inst_field'][tt] /= TC_this['area'][tt]
            TC_this['amean_running_field'][tt] /= TC_this['area'][tt]
            TC_this['amean_filtered_running_field'][tt] /= TC_this['area'][tt]

        ## Least squares linear fit for propagation speed.
        Pzonal = np.polyfit(TC_this['timestamp'],TC_this['centroid_lon'],1)
        TC_this['zonal_propagation_speed'] = Pzonal[0] * 111000.0  # deg / s --> m / s

        Pmeridional = np.polyfit(TC_this['timestamp'],TC_this['centroid_lat'],1)
        TC_this['meridional_propagation_speed'] = Pmeridional[0] * 111000.0  # deg / s --> m / s

        TC_all.append(TC_this)

    return TC_all



def get_list_of_path_graphs(G):

    Plist=[] # initialize empty list.

    ## Break in to individual paths (e.g., LPT branches).
    roots = []
    leaves = []
    for node in G.nodes:
        if G.in_degree(node) == 0: # it's a root
            roots.append(node)
        elif G.out_degree(node) == 0: # it's a leaf
            leaves.append(node)

    iii=-1
    for root in roots:
        iii+=1
        print(('    root ' + str(iii) + ' of max ' + str(len(roots)-1) + '.'), flush=True)
        for leaf in leaves:
            for path in nx.all_simple_paths(G, source=root, target=leaf):
                Plist.append(G.subgraph(path).copy())  # Add to list.

    return Plist


def get_list_of_path_graphs_rejoin_cycles(G):

    #########################################################
    ## 1. For any path intersecting with a cycle, add all nodes of the cycle.
    ## 2. Remove duplicate paths.

    cycles = nx.cycle_basis(nx.to_undirected(G))
    if len(cycles) > 1:
        ## In some cases, there are cycles that touch eathother. Cycles that toush are treated
        ##   together all at once, and hence need to be combined.
        for cc in range(len(cycles)):
            for cccc in range(len(cycles)):
                if cccc == cc:
                    continue
                if len(set(cycles[cc]).intersection(set(cycles[cccc]))) > 0:
                    cycles[cc] = list(set(cycles[cc]).union(set(cycles[cccc])))
                    cycles[cccc] = list(set(cycles[cc]).union(set(cycles[cccc])))

    Plist = get_list_of_path_graphs(G)

    for ii in range(len(Plist)):
        for C in cycles:
            intersection_nodes = set(C).intersection(set(Plist[ii].nodes()))
            intersecton_times = [get_objid_datetime(x) for x in intersection_nodes]
            if len(intersection_nodes) > 0:
                Cadd = []
                for this_node in C:
                    if get_objid_datetime(this_node) in intersecton_times:
                        Cadd.append(this_node)
                Plist[ii].add_nodes_from(Cadd)  #Only nodes are copied, not edges. But that's all I need.

    ## This step eliminates duplicates.
    for ii in range(len(Plist)):
        if ii == 0:
            Plist_new = [Plist[ii]]
        else:
            ## Check whether it is already in Plist_new.
            ##  Use XOR on the set of nodes.
            include_it = True
            for P in Plist_new:
                if len(set(Plist[ii].nodes()) ^ set(P.nodes())) == 0:
                    include_it = False
                    break
            if include_it:
                Plist_new.append(Plist[ii])

    return Plist_new


def calc_lpt_properties_with_branches(G, options, fmt="/%Y/%m/%Y%m%d/objects_%Y%m%d%H.nc"):
    ## The branch nodes of G have the properties of timestamp, lon, lat, and area.
    TC_all = []

    ## First, I get a list of DAG sub groups to loop through.
    CC = list(nx.connected_components(nx.to_undirected(G)))
    SG = [G.subgraph(CC[x]).copy() for x in range(len(CC))]

    ## Loop over each DAG
    for kk in range(len(SG)):
        print('--> LPT group ' + str(kk+1) + ' of ' + str(len(SG)),flush=True)

        #Plist = get_list_of_path_graphs(SG[kk])
        Plist = get_list_of_path_graphs_rejoin_cycles(SG[kk])

        if len(Plist) == 1:
            print('----> Found '+str(len(Plist))+' LPT system.',flush=True)
        else:
            print('----> Found '+str(len(Plist))+' LPT systems.',flush=True)


        ## Get "timeclusters" for each branch.
        for iiii in range(len(Plist)):
            path1 = Plist[iiii]

            PG = SG[kk].subgraph(path1).copy()

            TC_this = {}
            TC_this['lpt_group_id'] = kk
            TC_this['lpt_id'] = 1.0*kk + (iiii+1)/max(10.0,np.power(10,np.ceil(np.log10(len(Plist)))))
                                            ## ^ I should probably account for possible cycles here.
            TC_this['objid'] = sorted(list(PG.nodes()))
            ts=nx.get_node_attributes(PG,'timestamp')
            timestamp_all = [ts[x] for x in TC_this['objid']]
            TC_this['timestamp'] = np.unique(timestamp_all)
            TC_this['datetime'] = [cftime.datetime(1970,1,1,0,0,0,calendar=options['calendar']) + dt.timedelta(seconds=int(x)) for x in TC_this['timestamp']]

            ##
            ## Sum/average the LPTs to get bulk/mean properties at each time.
            ##

            ## Initialize
            TC_this = initialize_time_cluster_fields(TC_this, len(TC_this['timestamp']))

            ## Loop over unique time stamps.
            for tt in range(len(TC_this['timestamp'])):
                max_area_already_used = -999.0
                this_objid_list = [TC_this['objid'][x] for x in range(len(TC_this['objid'])) if timestamp_all[x] == TC_this['timestamp'][tt]]
                for this_objid in this_objid_list:

                    OBJ = read_lp_object_properties(this_objid, options['objdir']
                            , ['centroid_lon','centroid_lat','area','pixels_x','pixels_y'
                            ,'min_lon','max_lon','min_lat','max_lat'
                            ,'westmost_lat', 'eastmost_lat', 'southmost_lon', 'northmost_lon'
                            ,'amean_inst_field','amean_running_field','max_inst_field','max_running_field'
                            ,'min_inst_field','min_running_field','min_filtered_running_field'
                            ,'amean_filtered_running_field','max_filtered_running_field'], fmt=fmt)

                    TC_this['nobj'][tt] += 1
                    TC_this['area'][tt] += OBJ['area']
                    TC_this['centroid_lon'][tt] += OBJ['centroid_lon'] * OBJ['area']
                    TC_this['centroid_lat'][tt] += OBJ['centroid_lat'] * OBJ['area']
                    if OBJ['area'] > max_area_already_used:
                        TC_this['largest_object_centroid_lon'][tt] = 1.0*OBJ['centroid_lon']
                        TC_this['largest_object_centroid_lat'][tt] = 1.0*OBJ['centroid_lat']
                        max_area_already_used = 1.0*OBJ['area']

                    if OBJ['min_lon'] < TC_this['min_lon'][tt]:
                        TC_this['westmost_lat'][tt] = OBJ['westmost_lat']
                    if OBJ['max_lon'] > TC_this['max_lon'][tt]:
                        TC_this['eastmost_lat'][tt] = OBJ['eastmost_lat']
                    if OBJ['min_lat'] < TC_this['min_lat'][tt]:
                        TC_this['southmost_lon'][tt] = OBJ['southmost_lon']
                    if OBJ['max_lat'] > TC_this['max_lat'][tt]:
                        TC_this['northmost_lon'][tt] = OBJ['northmost_lon']

                    TC_this['min_lon'][tt] = min((TC_this['min_lon'][tt], OBJ['min_lon']))
                    TC_this['min_lat'][tt] = min((TC_this['min_lat'][tt], OBJ['min_lat']))
                    TC_this['max_lon'][tt] = max((TC_this['max_lon'][tt], OBJ['max_lon']))
                    TC_this['max_lat'][tt] = max((TC_this['max_lat'][tt], OBJ['max_lat']))


                    TC_this['amean_inst_field'][tt] += OBJ['amean_inst_field'] * OBJ['area']
                    TC_this['amean_running_field'][tt] += OBJ['amean_running_field'] * OBJ['area']
                    TC_this['amean_filtered_running_field'][tt] += OBJ['amean_filtered_running_field'] * OBJ['area']
                    TC_this['min_inst_field'][tt] = min((TC_this['min_inst_field'][tt], OBJ['min_inst_field']))
                    TC_this['min_running_field'][tt] = min((TC_this['min_running_field'][tt], OBJ['min_running_field']))
                    TC_this['min_filtered_running_field'][tt] = min((TC_this['min_filtered_running_field'][tt], OBJ['min_filtered_running_field']))
                    TC_this['max_inst_field'][tt] = max((TC_this['max_inst_field'][tt], OBJ['max_inst_field']))
                    TC_this['max_running_field'][tt] = max((TC_this['max_running_field'][tt], OBJ['max_running_field']))
                    TC_this['max_filtered_running_field'][tt] = max((TC_this['max_filtered_running_field'][tt], OBJ['max_filtered_running_field']))

                TC_this['centroid_lon'][tt] /= TC_this['area'][tt]
                TC_this['centroid_lat'][tt] /= TC_this['area'][tt]

                TC_this['amean_inst_field'][tt] /= TC_this['area'][tt]
                TC_this['amean_running_field'][tt] /= TC_this['area'][tt]
                TC_this['amean_filtered_running_field'][tt] /= TC_this['area'][tt]

            ## Least squares linear fit for propagation speed.
            Pzonal = np.polyfit(TC_this['timestamp'],TC_this['centroid_lon'],1)
            TC_this['zonal_propagation_speed'] = Pzonal[0] * 111000.0  # deg / s --> m / s

            Pmeridional = np.polyfit(TC_this['timestamp'],TC_this['centroid_lat'],1)
            TC_this['meridional_propagation_speed'] = Pmeridional[0] * 111000.0  # deg / s --> m / s

            TC_all.append(TC_this)

    return TC_all


###################################################
### Other processing functions. ###########################
###################################################


def get_lpo_mask(objid, objdir):

    dt1 = get_objid_datetime(objid)

    fmt = ("/%Y/%m/%Y%m%d/objects_%Y%m%d%H.nc")
    fn1 = (objdir + '/' + dt1.strftime(fmt)).replace('///','/').replace('//','/')

    DS1 = Dataset(fn1)
    id1 = DS1['objid'][:]
    idx1, = np.where(np.abs(id1 - objid) < 0.1)

    x1 = DS1['pixels_x'][:][idx1].compressed()
    y1 = DS1['pixels_y'][:][idx1].compressed()
    lon = DS1['grid_lon'][:]
    lat = DS1['grid_lat'][:]

    DS1.close()

    mask = np.zeros([len(lat), len(lon)])
    mask[y1,x1] = 1

    return (lon, lat, mask)

def plot_lpt_groups_time_lon_text(ax, LPT, BRANCHES, options, text_color='k'):

    objdir = options['objdir']
    dt_min = dt.datetime(1970,1,1,0,0,0) + dt.timedelta(seconds=int(np.min(LPT[:,0])))
    dt_max = dt.datetime(1970,1,1,0,0,0) + dt.timedelta(seconds=int(np.max(LPT[:,0])))

    for ii in range(len(LPT[:,0])):
        objid = LPT[ii,1]
        dt1 = get_objid_datetime(objid)
        fmt = ("/%Y/%m/%Y%m%d/objects_%Y%m%d%H.nc")
        fn1 = (objdir + '/' + dt1.strftime(fmt)).replace('///','/').replace('//','/')

        DS1 = Dataset(fn1)
        id1 = DS1['objid'][:]
        idx1, = np.where(np.abs(id1 - objid) < 0.1)
        lon = DS1['centroid_lon'][:][idx1]
        DS1.close()

        this_text_color = text_color
        this_zorder = 10
        if (LPT[ii,3] == 1):
            this_text_color = 'b'
            this_zorder = 20
        if (LPT[ii,4] == 1):
            this_text_color = 'm'
            this_zorder = 20
        if (LPT[ii,5] == 1):
            this_text_color = 'g'
            this_zorder = 20

        plt.text(lon, dt1, (str(LPT[ii,2]) + ": " + branches_binary_str4(BRANCHES[ii]))
                  , color=this_text_color, zorder=this_zorder, fontsize=6, clip_on=True)

    ax.set_xlim([0.0, 360.0])
    ax.set_ylim([dt_min, dt_max + dt.timedelta(hours=3)])



def float_lpt_id(group, branch):
    """
    Branch is a decimal tacked on to the group ID.
    group 7, branch #1 is 7.01
    group 20, branch 10 is 20.10
    Branch > 100 will give an error message and return np.nan.
    """
    if branch > 99:
        print('ERROR! Branch number > 99.')
        float_id = np.nan
    else:
        float_id = group + branch / 100.0

    return float_id


def plot_timeclusters_time_lon(ax, TIMECLUSTERS, linewidth=2.0):

    for ii in range(len(TIMECLUSTERS)):
        x = TIMECLUSTERS[ii]['centroid_lon']
        y = TIMECLUSTERS[ii]['datetime']
        ax.plot(x, y, 'k', linewidth=linewidth)

        plt.text(x[0], y[0], str(int(ii)), fontweight='bold', color='red', clip_on=True)
        plt.text(x[-1], y[-1], str(int(ii)), fontweight='bold', color='red', clip_on=True)
