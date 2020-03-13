import matplotlib; matplotlib.use('agg')
import numpy as np
import datetime as dt
from scipy.signal import convolve2d
from scipy import ndimage
import matplotlib.pylab as plt
from netCDF4 import Dataset
import glob
import networkx as nx
import sys


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

    amean_instantaneous_field = ndimage.mean(field, label_im, range(1, nb_labels + 1))
    amean_running_field = ndimage.mean(field_running, label_im, range(1, nb_labels + 1))
    amean_filtered_running_field = ndimage.mean(field_filtered, label_im, range(1, nb_labels + 1))
    min_instantaneous_field = ndimage.minimum(field, label_im, range(1, nb_labels + 1))
    min_running_field = ndimage.minimum(field_running, label_im, range(1, nb_labels + 1))
    min_filtered_running_field = ndimage.minimum(field_filtered, label_im, range(1, nb_labels + 1))
    max_instantaneous_field = ndimage.maximum(field, label_im, range(1, nb_labels + 1))
    max_running_field = ndimage.maximum(field_running, label_im, range(1, nb_labels + 1))
    max_filtered_running_field = ndimage.maximum(field_filtered, label_im, range(1, nb_labels + 1))

    centroid_lon = ndimage.mean(lon2, label_im, range(1, nb_labels + 1))
    centroid_lat = ndimage.mean(lat2, label_im, range(1, nb_labels + 1))
    centroid_x = ndimage.mean(X2, label_im, range(1, nb_labels + 1))
    centroid_y = ndimage.mean(Y2, label_im, range(1, nb_labels + 1))
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

    # Grid stuff.
    OBJ['grid'] = {}
    OBJ['grid']['lon'] = lon
    OBJ['grid']['lat'] = lat
    OBJ['grid']['area'] = area2d

    return OBJ


def get_objid_datetime(objid):
    """
    usge: this_datetime = get_objid_datetime(this_objid)

    Get the datetime from an objid of form YYYYMMDDHHnnnn.
    """
    ymdh_int = int(np.floor(objid/1e4))
    ymdh_str = str(ymdh_int)
    return dt.datetime.strptime(ymdh_str, "%Y%m%d%H")


def read_lp_object_properties(objid, objdir, property_list, verbose=False, fmt="/%Y/%m/%Y%m%d/objects_%Y%m%d%H.nc"):

    dt1 = get_objid_datetime(objid)
    fn1 = (objdir + dt1.strftime(fmt))

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

    fn1 = (objdir + dt1.strftime(fmt))
    fn2 = (objdir + dt2.strftime(fmt))

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

    for this_dt in dt_list:

        print(this_dt)
        fn = (objdir + this_dt.strftime(fmt))
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
                    G.add_node(int(id_list[ii]), timestamp=(this_dt - dt.datetime(1970,1,1,0,0,0)).total_seconds()
                        , lon = lon[ii], lat=lat[ii], area=area[ii]
                        , pos = (lon[ii], (this_dt - dt.datetime(1970,1,1,0,0,0)).total_seconds()))

        except FileNotFoundError:
            print('WARNING: Missing this file!',flush=True)

    return G

def get_lpo_overlap(dt1, dt2, objdir, min_points=1, fmt="/%Y/%m/%Y%m%d/objects_%Y%m%d%H.nc"):

    ##
    ## Read in LPO masks for current and previous times.
    ##
    fn1 = objdir + '/' + dt1.strftime(fmt)
    fn2 = objdir + '/' + dt2.strftime(fmt)

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
    datetime_list = [get_objid_datetime(x) for x in lpo_id_list]
    timestamp_list = [int((x - dt.datetime(1970,1,1,0,0,0)).total_seconds()) for x in datetime_list]

    ## Now, loop through the times.
    unique_timestamp_list = np.unique(timestamp_list)
    for tt in range(1,len(unique_timestamp_list)):

        ## Datetimes for this time and previous time.
        this_timestamp = unique_timestamp_list[tt]
        prev_timestamp = unique_timestamp_list[tt-1]
        this_dt = dt.datetime(1970,1,1,0,0,0) + dt.timedelta(seconds=int(this_timestamp))
        prev_dt = dt.datetime(1970,1,1,0,0,0) + dt.timedelta(seconds=int(prev_timestamp))
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
                        elif 1.0*frac1 > options['min_overlap_frac']:
                            print('Overlap: '+ str(kkkk) + ' --> ' + str(llll) + '!', flush=True)
                            G.add_edge(kkkk,llll)
                        elif 1.0*frac2 > options['min_overlap_frac']:
                            print('Overlap: '+ str(kkkk) + ' --> ' + str(llll) + '!', flush=True)
                            G.add_edge(kkkk,llll)

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
                merger_timestamps = np.array([(x - dt.datetime(1970,1,1,0,0,0)).total_seconds()/3600 for x in merger_datetimes])
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
                split_timestamps = np.array([(x - dt.datetime(1970,1,1,0,0,0)).total_seconds()/3600 for x in split_datetimes])

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
                more_to_do = True

    return G


def initialize_time_cluster_fields(TC, length):

    ## Fields initialized to zero.
    for field in ['nobj','area','centroid_lon','centroid_lat'
                ,'largest_object_centroid_lon','largest_object_centroid_lat'
                ,'amean_inst_field','amean_running_field','amean_filtered_running_field']:
        TC[field] = np.zeros(length)

    ## Fields initialized to 999.0.
    for field in ['min_lon','min_lat','min_inst_field','min_running_field'
                ,'min_filtered_running_field']:
        TC[field] =  999.0 * np.ones(length)

    ## Fields initialized to -999.0.
    for field in ['max_lon','max_lat','max_inst_field','max_running_field'
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
        TC_this['datetime'] = [dt.datetime(1970,1,1,0,0,0) + dt.timedelta(seconds=int(x)) for x in TC_this['timestamp']]

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
    Plist = get_list_of_path_graphs(G)

    for ii in range(len(Plist)):
        for C in cycles:
            if len(set(C).intersection(set(Plist[ii].nodes()))) > 0:
                Plist[ii].add_nodes_from(C)  #Only nodes are copied. But that's all I need.

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
            TC_this['datetime'] = [dt.datetime(1970,1,1,0,0,0) + dt.timedelta(seconds=int(x)) for x in TC_this['timestamp']]

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
    fn1 = (objdir + dt1.strftime(fmt))

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
        fn1 = (objdir + dt1.strftime(fmt))

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
