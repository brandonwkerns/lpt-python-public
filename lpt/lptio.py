import matplotlib; matplotlib.use('agg')
import numpy as np
import numpy.ma as ma
from context import lpt
from netCDF4 import Dataset
import os
import os.path
import csv
import datetime as dt

###################################################
### Output functions
###################################################
def lp_objects_output_ascii(fn, OBJ):
    """
    This function outputs the "bulk" LP object properties (centroid, date, area)
    to an ascii file.
    """
    print('Writing LP object ASCII output to: ' + fn)
    fmt = '%7.2f%8.2f%7.1f%7.1f%20.1f   %16d\n'
    file = open(fn, 'w')

    file.write(' lat.__  lon.__    y._    x._         area[km2]._     YYYYMMDDHHnnnn\n') # Header line.
    for ii in range(len(OBJ['lon'])):

        print(fmt % (OBJ['lat'][ii], OBJ['lon'][ii],
                OBJ['y'][ii], OBJ['x'][ii],
                OBJ['area'][ii], OBJ['id'][ii]))

        file.write(fmt % (OBJ['lat'][ii], OBJ['lon'][ii],
                OBJ['y'][ii], OBJ['x'][ii],
                OBJ['area'][ii], OBJ['id'][ii]))

    file.close()


def lp_objects_output_netcdf(fn, OBJ):
    """
    This function outputs the "bulk" LP object properties (centroid, date, area)
    Plus the pixel information to a compressed netcdf file.
    """

    if not 'units_inst' in OBJ:
        print('WARNING: units_inst, units_running, and units_filtered not specified in dict OBJ. Netcdf file will have empty units.')
        OBJ['units_inst'] = ''
        OBJ['units_running'] = ''
        OBJ['units_filtered'] = ''

    print('Writing LP object NetCDF output to: ' + fn)

    os.remove(fn) if os.path.exists(fn) else None
    DS = Dataset(fn, 'w', format='NETCDF4_CLASSIC', clobber=True)
    DS.description = ("LP Objects NetCDF file. Time stamp is for the END of running mean time. nobj is the number of objects (each one has an objid), "
        + "and npoints is the max pixels in any object. "
        + "Parameters for the LP objects as a whole: centroid_lon, centroid_lat, and area. "
        + "To see the pixels for each LP object, either use (pixels_x and pixels_y), "
        + "or (grid_lon, grid_lat, grid_mask.). To plot the contour, best to use the grid_mask. "
        + "The values in grid_mask are: -1 for no LPT, or the nnnn part of the LP object otherwise. "
        + "NOTE: If no LP objects, nobj (npoints) dimension will be 0 (1).")
    DS.N = len(OBJ['n_points'])

    ##
    ## Dimensions
    ##
    DS.createDimension('nobj', 0)  # Unlimited demension.

    ## Grid stuff.
    if OBJ['grid']['lon'].ndim == 1:
        DS.createDimension('grid_x', len(OBJ['grid']['lon']))
        DS.createDimension('grid_y', len(OBJ['grid']['lat']))
    else:
        ny,nx=OBJ['grid']['lon'].shape
        DS.createDimension('grid_x', nx)
        DS.createDimension('grid_y', ny)

    ##
    ## Variables
    ##

    ## LP Object "bulk" properties.
    var_objid = DS.createVariable('objid','d',('nobj',))
    var_centroid_lon = DS.createVariable('centroid_lon','f4',('nobj',))
    var_centroid_lat = DS.createVariable('centroid_lat','f4',('nobj',))
    var_centroid_x = DS.createVariable('centroid_x','f4',('nobj',))
    var_centroid_y = DS.createVariable('centroid_y','f4',('nobj',))
    var_max_lon = DS.createVariable('max_lon','f4',('nobj',))
    var_max_lat = DS.createVariable('max_lat','f4',('nobj',))
    var_min_lon = DS.createVariable('min_lon','f4',('nobj',))
    var_min_lat = DS.createVariable('min_lat','f4',('nobj',))
    var_area = DS.createVariable('area','f4',('nobj',))

    var_amean_inst_field = DS.createVariable('amean_inst_field','f4',('nobj',))
    var_amean_running_field = DS.createVariable('amean_running_field','f4',('nobj',))
    var_amean_filtered_running_field = DS.createVariable('amean_filtered_running_field','f4',('nobj',))
    var_min_inst_field = DS.createVariable('min_inst_field','f4',('nobj',))
    var_min_running_field = DS.createVariable('min_running_field','f4',('nobj',))
    var_min_filtered_running_field = DS.createVariable('min_filtered_running_field','f4',('nobj',))
    var_max_inst_field = DS.createVariable('max_inst_field','f4',('nobj',))
    var_max_running_field = DS.createVariable('max_running_field','f4',('nobj',))
    var_max_filtered_running_field = DS.createVariable('max_filtered_running_field','f4',('nobj',))

    ## Pixels information.
    if len(OBJ['n_points']) > 0:

        max_points = np.max(OBJ['n_points'])
        DS.createDimension('npoints', max_points)

        var_pixels_x = DS.createVariable('pixels_x','i4',('nobj','npoints',), zlib=True)
        var_pixels_y = DS.createVariable('pixels_y','i4',('nobj','npoints',), zlib=True)

        ##
        ## Values
        ##
        var_objid[:] = OBJ['id']
        var_centroid_lon[:] = OBJ['lon']
        var_centroid_lat[:] = OBJ['lat']
        var_centroid_x[:] = OBJ['y']
        var_centroid_y[:] = OBJ['x']
        var_max_lon[:] = OBJ['max_lon']
        var_max_lat[:] = OBJ['max_lat']
        var_min_lon[:] = OBJ['min_lon']
        var_min_lat[:] = OBJ['min_lat']
        var_area[:] = OBJ['area']

        var_amean_inst_field[:] = OBJ['amean_inst_field']
        var_amean_running_field[:] = OBJ['amean_running_field']
        var_amean_filtered_running_field[:] = OBJ['amean_filtered_running_field']
        var_min_inst_field[:] = OBJ['min_inst_field']
        var_min_running_field[:] = OBJ['min_running_field']
        var_min_filtered_running_field[:] = OBJ['min_filtered_running_field']
        var_max_inst_field[:] = OBJ['max_inst_field']
        var_max_running_field[:] = OBJ['max_running_field']
        var_max_filtered_running_field[:] = OBJ['max_filtered_running_field']

        for ii in range(len(OBJ['lon'])):
            ypoints, xpoints = np.where(OBJ['label_im'] == ii+1)
            var_pixels_x[ii,:len(xpoints)] = xpoints
            var_pixels_y[ii,:len(ypoints)] = ypoints

    else:
        ## If there are no LP Objects, keep it as "missing values".
        DS.createDimension('npoints', 1)
        var_pixels_x = DS.createVariable('pixels_x','i4',('nobj','npoints',), zlib=True)
        var_pixels_y = DS.createVariable('pixels_y','i4',('nobj','npoints',), zlib=True)


    ## Grid variables.
    if OBJ['grid']['lon'].ndim == 1:
        var_grid_lon = DS.createVariable('grid_lon','f4',('grid_x',))
        var_grid_lat = DS.createVariable('grid_lat','f4',('grid_y',))
    else:
        var_grid_lon = DS.createVariable('grid_lon','f4',('grid_y','grid_x'))
        var_grid_lat = DS.createVariable('grid_lat','f4',('grid_y','grid_x'))

    var_grid_area = DS.createVariable('grid_area','f4',('grid_y','grid_x',), zlib=True)
    var_grid_mask = DS.createVariable('grid_mask','i4',('grid_y','grid_x',), zlib=True, fill_value=-1)

    var_grid_lon[:] = OBJ['grid']['lon']
    var_grid_lat[:] = OBJ['grid']['lat']
    var_grid_area[:] = OBJ['grid']['area']
    mask = OBJ['label_im'] - 1
    mask = ma.masked_array(mask, mask = (mask < -0.5))
    var_grid_mask[:] = mask


    ##
    ## Attributes/Metadata
    ##
    var_objid.setncatts({'units':'0','long_name':'LP Object ID'
        ,'description':'A unique ID for each LP object. Convention is YYYYMMDDHHnnnn where nnnn starts at 0000. YYYYMMDDHH is the END of running mean time.'})

    var_centroid_lon.setncatts({'units':'degrees_east','long_name':'centroid longitude (0-360)'})
    var_centroid_lat.setncatts({'units':'degrees_north','long_name':'centroid latitude (-90-90)'})
    var_centroid_x.setncatts({'units':'1.0','long_name':'centroid x grid point (0 to NX-1)'})
    var_centroid_y.setncatts({'units':'1.0','long_name':'centroid y grid point (0 to NY-1)'})
    var_max_lon.setncatts({'units':'degrees_east','long_name':'max (eastmost) longitude (0-360)'})
    var_max_lat.setncatts({'units':'degrees_north','long_name':'max (northmost) latitude (-90-90)'})
    var_min_lon.setncatts({'units':'degrees_east','long_name':'min (westmost) longitude (0-360)'})
    var_min_lat.setncatts({'units':'degrees_north','long_name':'min (southmost) latitude (-90-90)'})
    var_area.setncatts({'units':'km2','long_name':'LP object enclosed area'})

    var_amean_inst_field.setncatts({'units':OBJ['units_inst'],'long_name':'LP object area mean of instantaneous field', 'note': 'end of running mean time'})
    var_amean_running_field.setncatts({'units':OBJ['units_running'],'long_name':'LP object area mean of running mean field', 'note': 'end of running mean time'})
    var_amean_filtered_running_field.setncatts({'units':OBJ['units_filtered'],'long_name':'LP object area mean of filtered running mean field', 'note': 'end of running mean time'})
    var_min_inst_field.setncatts({'units':OBJ['units_inst'],'long_name':'LP object area min of instantaneous field', 'note': 'end of running mean time'})
    var_min_running_field.setncatts({'units':OBJ['units_running'],'long_name':'LP objedt area min of running mean field', 'note': 'end of running mean time'})
    var_min_filtered_running_field.setncatts({'units':OBJ['units_filtered'],'long_name':'LP object area min of filtered running mean field', 'note': 'end of running mean time'})
    var_max_inst_field.setncatts({'units':OBJ['units_inst'],'long_name':'LP object area max of instantaneous field', 'note': 'end of running mean time'})
    var_max_running_field.setncatts({'units':OBJ['units_running'],'long_name':'LP object area max of running mean field', 'note': 'end of running mean time'})
    var_max_filtered_running_field.setncatts({'units':OBJ['units_filtered'],'long_name':'LP object area max of filtered running mean field', 'note': 'end of running mean time'})

    var_grid_lon.setncatts({'units':'degrees_east','long_name':'grid longitude (0-360)','standard_name':'longitude','axis':'X'})
    var_grid_lat.setncatts({'units':'degrees_north','long_name':'grid latitude (-90-90)','standard_name':'latitude','axis':'Y'})
    var_grid_area.setncatts({'units':'km2','long_name':'area of each grid point'})
    var_grid_mask.setncatts({'units':'0','long_name':'mask by nnnn part of LP Object ID.', 'note':'-1 for no LP Object.'})
    var_pixels_x.setncatts({'units':'0','long_name':'grid point pixel indices in the x direction','note':'zero based (Python convention)'})
    var_pixels_y.setncatts({'units':'0','long_name':'grid point pixel indices in the y direction','note':'zero based (Python convention)'})

    DS.close()


def lpt_system_tracks_output_ascii(fn, TIMECLUSTERS):
    """
    This function outputs the "bulk" LPT system properties (centroid, date, area)
    to an ascii file. Does NOT give LPO list.
    """
    print('Writing LPT system track ASCII output to: ' + fn)
    fmt='        %4d%02d%02d%02d %8d %10.2f %10.2f %2d\n'

    os.makedirs(os.path.dirname(fn), exist_ok=True) # Make directory if needed.
    file = open(fn, 'w')

    ## Header
    file.write("LPT nnnnn.nnnn\n")
    file.write("        YYYYMMDDHH _A_[km2] cen_lat.__ cen_lon.__ Nobj\n")

    ## Data
    for ii in range(len(TIMECLUSTERS)):
        file.write("LPT %10.4f\n" % (TIMECLUSTERS[ii]['lpt_id'],))

        for tt in range(len(TIMECLUSTERS[ii]['datetime'])):
            year,month,day,hour = TIMECLUSTERS[ii]['datetime'][tt].timetuple()[0:4]
            file.write(fmt % (year,month,day,hour
                                , TIMECLUSTERS[ii]['area'][tt]
                                , TIMECLUSTERS[ii]['centroid_lat'][tt]
                                , TIMECLUSTERS[ii]['centroid_lon'][tt]
                                , TIMECLUSTERS[ii]['nobj'][tt]))

    file.close()


def lpt_systems_group_array_output_ascii(fn, LPT, BRANCHES):

    print('Writing LPT system group info (including LP object IDs) to: ' + fn)
    header='time_stamp__, YYYYMMDDHHnnnn, lpt_sys_id, B, E, S, branches\n'

    OUT = []
    for jj in range(len(BRANCHES)):
        OUT.append(list(LPT[jj,:]) + [lpt.helpers.branches_binary_str4(BRANCHES[jj])])

    fid = open(fn,'w')
    fid.write(header)
    for this_line in OUT:
        fid.write(("{:12d}, {:14d}, {:10d}, {:1d}, {:1d}, {:1d}, {:s}\n").format(*this_line))

    fid.close()


def read_lpt_systems_group_array(fn):

    fid = open(fn,'r')
    LPT = []
    BRANCHES = []

    for this_line in fid.readlines()[1:]:

        entry_list = this_line.split(',')
        this_lpt_row = [int(x) for x in entry_list[0:6]]
        LPT.append(this_lpt_row)
        BRANCHES.append(int(entry_list[6].replace(" ", ""), 2))

    fid.close()
    LPT = np.array(LPT)

    return (LPT, BRANCHES)




def lpt_system_tracks_output_netcdf(fn, TIMECLUSTERS, units={}):
    """
    This function outputs the "bulk" LPT system properties (centroid, date, area)
    plus the LP Objects belonging to each "TIMECLUSTER" to a netcdf file.
    """
    print('Writing LPT system track NetCDF output to: ' + fn)

    if not 'units_inst' in units:
        print('WARNING: units_inst, units_running, and units_filtered not specified in dict "units". Netcdf file will have empty units.')
        units['units_inst'] = ''
        units['units_running'] = ''
        units['units_filtered'] = ''



    os.makedirs(os.path.dirname(fn), exist_ok=True) # Make directory if needed.

    DS = Dataset(fn, 'w', format='NETCDF4', clobber=True)
    DS.description = 'LPT Systems "timeclusters" NetCDF file.'

    MISSING = -999.0
    FILL_VALUE = MISSING

    ##
    ## Initialize stitched variables.
    ##
    lptid_collect = np.array([MISSING])
    timestamp_collect = np.double([MISSING])
    nobj_collect = np.double([MISSING])
    centroid_lon_collect = np.array([MISSING])
    centroid_lat_collect = np.array([MISSING])
    largest_object_centroid_lon_collect = np.array([MISSING])
    largest_object_centroid_lat_collect = np.array([MISSING])
    max_lon_collect = np.array([MISSING])
    min_lon_collect = np.array([MISSING])
    max_lat_collect = np.array([MISSING])
    min_lat_collect = np.array([MISSING])
    area_collect = np.array([MISSING])

    mean_inst_collect = np.array([MISSING])
    mean_running_collect = np.array([MISSING])
    mean_filtered_running_collect = np.array([MISSING])
    min_inst_collect = np.array([MISSING])
    min_running_collect = np.array([MISSING])
    min_filtered_running_collect = np.array([MISSING])
    max_inst_collect = np.array([MISSING])
    max_running_collect = np.array([MISSING])
    max_filtered_running_collect = np.array([MISSING])

    lpt_begin_index = []
    lpt_end_index = []
    time_step_hours = 999

    ##
    ## Fill in stitched variables.
    ##
    max_lpo = 1
    for ii in range(len(TIMECLUSTERS)):

        max_lpo = max(max_lpo, len(TIMECLUSTERS[ii]['objid'])) # will be used below.

        lpt_begin_index += [len(lptid_collect)] # zero based, so next index is the length.

        lptid_collect = np.append(np.append(lptid_collect, np.ones(len(TIMECLUSTERS[ii]['datetime']))*TIMECLUSTERS[ii]['lpt_id']),MISSING)
        this_timestamp = [(TIMECLUSTERS[ii]['datetime'][x] - dt.datetime(1970,1,1,0,0,0)).total_seconds()/3600.0 for x in range(len(TIMECLUSTERS[ii]['datetime']))]
        timestamp_collect = np.append(np.append(timestamp_collect, this_timestamp), MISSING)
        nobj_collect = np.append(np.append(nobj_collect, TIMECLUSTERS[ii]['nobj']),MISSING)
        centroid_lon_collect = np.append(np.append(centroid_lon_collect, TIMECLUSTERS[ii]['centroid_lon']),MISSING)
        centroid_lat_collect = np.append(np.append(centroid_lat_collect, TIMECLUSTERS[ii]['centroid_lat']),MISSING)
        largest_object_centroid_lon_collect = np.append(np.append(largest_object_centroid_lon_collect, TIMECLUSTERS[ii]['largest_object_centroid_lon']),MISSING)
        largest_object_centroid_lat_collect = np.append(np.append(largest_object_centroid_lat_collect, TIMECLUSTERS[ii]['largest_object_centroid_lat']),MISSING)
        area_collect = np.append(np.append(area_collect, TIMECLUSTERS[ii]['area']),MISSING)
        max_lon_collect = np.append(np.append(max_lon_collect, TIMECLUSTERS[ii]['max_lon']),MISSING)
        max_lat_collect = np.append(np.append(max_lat_collect, TIMECLUSTERS[ii]['max_lat']),MISSING)
        min_lon_collect = np.append(np.append(min_lon_collect, TIMECLUSTERS[ii]['min_lon']),MISSING)
        min_lat_collect = np.append(np.append(min_lat_collect, TIMECLUSTERS[ii]['min_lat']),MISSING)

        mean_inst_collect = np.append(np.append(mean_inst_collect, TIMECLUSTERS[ii]['amean_inst_field']),MISSING)
        mean_running_collect = np.append(np.append(mean_running_collect, TIMECLUSTERS[ii]['amean_running_field']),MISSING)
        mean_filtered_running_collect = np.append(np.append(mean_filtered_running_collect, TIMECLUSTERS[ii]['amean_filtered_running_field']),MISSING)
        min_inst_collect = np.append(np.append(min_inst_collect, TIMECLUSTERS[ii]['min_inst_field']),MISSING)
        min_running_collect = np.append(np.append(min_running_collect, TIMECLUSTERS[ii]['min_running_field']),MISSING)
        min_filtered_running_collect = np.append(np.append(min_filtered_running_collect, TIMECLUSTERS[ii]['min_filtered_running_field']),MISSING)
        max_inst_collect = np.append(np.append(max_inst_collect, TIMECLUSTERS[ii]['max_inst_field']),MISSING)
        max_running_collect = np.append(np.append(max_running_collect, TIMECLUSTERS[ii]['max_running_field']),MISSING)
        max_filtered_running_collect = np.append(np.append(max_filtered_running_collect, TIMECLUSTERS[ii]['max_filtered_running_field']),MISSING)

        lpt_end_index += [len(lptid_collect)-2] # zero based, and I added a NaN, so end index is the length.


    ##
    ## Initialize LPO variables.
    ##
    lpo_objid = MISSING * np.ones([len(TIMECLUSTERS), max_lpo])

    ##
    ## Fill in LPO variables
    ##

    for ii in range(len(TIMECLUSTERS)):
        lpo_objid[ii,0:len(TIMECLUSTERS[ii]['objid'])] = TIMECLUSTERS[ii]['objid']



    ##
    ## Dimensions
    ##
    DS.createDimension('nlpt', len(TIMECLUSTERS))
    DS.createDimension('nall', len(timestamp_collect))
    DS.createDimension('obj', max_lpo)

    ##
    ## Variables
    ##

    ## Single value "bulk" variables.
    DS.createVariable('lptid', 'f4', ('nlpt',),fill_value=FILL_VALUE)
    DS.createVariable('lpt_begin_index', 'i', ('nlpt',),fill_value=FILL_VALUE)
    DS.createVariable('lpt_end_index', 'i', ('nlpt',),fill_value=FILL_VALUE)
    DS.createVariable('duration', 'f4', ('nlpt',),fill_value=FILL_VALUE)
    DS.createVariable('maxarea', 'd', ('nlpt',),fill_value=FILL_VALUE)
    DS.createVariable('zonal_propagation_speed','f4',('nlpt',),fill_value=FILL_VALUE)
    DS.createVariable('meridional_propagation_speed','f4',('nlpt',),fill_value=FILL_VALUE)

    ## LP Objects variables.
    DS.createVariable('num_objects', 'i', ('nlpt',),fill_value=FILL_VALUE)
    DS.createVariable('objid', 'i8', ('nlpt','obj'),fill_value=FILL_VALUE)

    ## Stitched "bulk" variables.
    var_timestamp_all = DS.createVariable('timestamp_stitched','u4',('nall',),fill_value=int(FILL_VALUE))
    var_lptid_all = DS.createVariable('lptid_stitched','f4',('nall',),fill_value=FILL_VALUE)
    var_nobj_all = DS.createVariable('nobj_stitched','i',('nall',),fill_value=FILL_VALUE)
    var_centroid_lon_all = DS.createVariable('centroid_lon_stitched','f4',('nall',),fill_value=FILL_VALUE)
    var_centroid_lat_all = DS.createVariable('centroid_lat_stitched','f4',('nall',),fill_value=FILL_VALUE)
    var_largest_object_centroid_lon_all = DS.createVariable('largest_object_centroid_lon_stitched','f4',('nall',),fill_value=FILL_VALUE)
    var_largest_object_centroid_lat_all = DS.createVariable('largest_object_centroid_lat_stitched','f4',('nall',),fill_value=FILL_VALUE)
    var_area_all = DS.createVariable('area_stitched','d',('nall',),fill_value=FILL_VALUE)
    var_max_lon_all = DS.createVariable('max_lon_stitched','f4',('nall',),fill_value=FILL_VALUE)
    var_max_lat_all = DS.createVariable('max_lat_stitched','f4',('nall',),fill_value=FILL_VALUE)
    var_min_lon_all = DS.createVariable('min_lon_stitched','f4',('nall',),fill_value=FILL_VALUE)
    var_min_lat_all = DS.createVariable('min_lat_stitched','f4',('nall',),fill_value=FILL_VALUE)
    DS.createVariable('amean_inst_field', 'f4', ('nall',),fill_value=FILL_VALUE)
    DS.createVariable('amean_running_field', 'f4', ('nall',),fill_value=FILL_VALUE)
    DS.createVariable('amean_filtered_running_field', 'f4', ('nall',),fill_value=FILL_VALUE)
    DS.createVariable('min_inst_field', 'f4', ('nall',),fill_value=FILL_VALUE)
    DS.createVariable('min_running_field', 'f4', ('nall',),fill_value=FILL_VALUE)
    DS.createVariable('min_filtered_running_field', 'f4', ('nall',),fill_value=FILL_VALUE)
    DS.createVariable('max_inst_field', 'f4', ('nall',),fill_value=FILL_VALUE)
    DS.createVariable('max_running_field', 'f4', ('nall',),fill_value=FILL_VALUE)
    DS.createVariable('max_filtered_running_field', 'f4', ('nall',),fill_value=FILL_VALUE)


    ##
    ## Values
    ##
    DS['lptid'][:] = [TIMECLUSTERS[ii]['lpt_id'] for ii in range(len(TIMECLUSTERS))]
    DS['lpt_begin_index'][:] = lpt_begin_index
    DS['lpt_end_index'][:] = lpt_end_index
    DS['duration'][:] = [(TIMECLUSTERS[ii]['datetime'][-1] - TIMECLUSTERS[ii]['datetime'][0]).total_seconds()/3600.0 for ii in range(len(TIMECLUSTERS))]
    DS['maxarea'][:] = [np.max(TIMECLUSTERS[ii]['area']) for ii in range(len(TIMECLUSTERS))]
    DS['zonal_propagation_speed'][:] = [TIMECLUSTERS[ii]['zonal_propagation_speed'] for ii in range(len(TIMECLUSTERS))]
    DS['meridional_propagation_speed'][:] = [TIMECLUSTERS[ii]['meridional_propagation_speed'] for ii in range(len(TIMECLUSTERS))]

    DS['num_objects'][:] = [len(TIMECLUSTERS[ii]['objid']) for ii in range(len(TIMECLUSTERS))]
    DS['objid'][:] = lpo_objid

    var_timestamp_all[:] = timestamp_collect
    var_lptid_all[:] = lptid_collect
    var_nobj_all[:] = nobj_collect
    var_centroid_lon_all[:] = centroid_lon_collect
    var_centroid_lat_all[:] = centroid_lat_collect
    var_largest_object_centroid_lon_all[:] = largest_object_centroid_lon_collect
    var_largest_object_centroid_lat_all[:] = largest_object_centroid_lat_collect
    var_area_all[:] = area_collect
    var_max_lon_all[:] = max_lon_collect
    var_max_lat_all[:] = max_lat_collect
    var_min_lon_all[:] = min_lon_collect
    var_min_lat_all[:] = min_lat_collect

    DS['amean_inst_field'][:] = mean_inst_collect
    DS['amean_running_field'][:] = mean_running_collect
    DS['amean_filtered_running_field'][:] = mean_filtered_running_collect
    DS['min_inst_field'][:] = min_inst_collect
    DS['min_running_field'][:] = min_running_collect
    DS['min_filtered_running_field'][:] = min_filtered_running_collect
    DS['max_inst_field'][:] = max_inst_collect
    DS['max_running_field'][:] = max_running_collect
    DS['max_filtered_running_field'][:] = max_filtered_running_collect


    ##
    ## Attributes/Metadata
    ##
    DS['lptid'].setncatts({'units':'1.0','long_name':'LPT System id'})
    DS['lpt_begin_index'].setncatts({'units':'1','long_name':'LPT System beginning index (zero-based, Python convention)'})
    DS['lpt_end_index'].setncatts({'units':'1','long_name':'LPT System beginning index (zero-based, Python convention)'})
    DS['duration'].setncatts({'units':'h','long_name':'LPT System Duration'})
    DS['maxarea'].setncatts({'units':'km2','long_name':'LPT System area at time of largest extent'})
    DS['zonal_propagation_speed'].setncatts({'units':'m s-1','long_name':'Centroid Zonal Propagation Speed from least squares regression.'})
    DS['meridional_propagation_speed'].setncatts({'units':'m s-1','long_name':'Centroid Meridional Propagation Speed from least squares regression.'})
    var_timestamp_all.setncatts({'units':'hours since 1970-1-1 0:0','long_name':'LPT System time stamp -- stitched'})
    var_lptid_all.setncatts({'units':'1.0','long_name':'LPT System id -- stitched'})
    var_centroid_lon_all.setncatts({'units':'degrees_east','long_name':'centroid longitude, may be inbetween objects (0-360) -- stitched','standard_name':'longitude'})
    var_centroid_lat_all.setncatts({'units':'degrees_north','long_name':'centroid latitude, may be inbetween objects (-90-90) -- stitched','standard_name':'latitude'})
    var_largest_object_centroid_lon_all.setncatts({'units':'degrees_east','long_name':'centroid longitude of the largest contiguous object (0-360) -- stitched','standard_name':'longitude'})
    var_largest_object_centroid_lat_all.setncatts({'units':'degrees_north','long_name':'centroid latitude of the largest contiguous object (-90-90) -- stitched','standard_name':'latitude'})
    var_area_all.setncatts({'units':'km2','long_name':'LPT System enclosed area -- stitched'})
    var_max_lon_all.setncatts({'units':'degrees_east','long_name':'max (eastmost) longitude (0-360) -- stitched','standard_name':'longitude'})
    var_max_lat_all.setncatts({'units':'degrees_north','long_name':'max (northmost) latitude (-90-90) -- stitched','standard_name':'longitude'})
    var_min_lon_all.setncatts({'units':'degrees_east','long_name':'min (westmost) longitude (0-360) -- stitched','standard_name':'longitude'})
    var_min_lat_all.setncatts({'units':'degrees_north','long_name':'min (southmost) latitude (-90-90) -- stitched','standard_name':'longitude'})

    DS['amean_inst_field'].setncatts({'units':'mm h-1','long_name':'LP object mean instantaneous rain rate (at end of running time).'})
    DS['amean_running_field'].setncatts({'units':'mm day-1','long_name':'LP object running mean, mean rain rate (at end of running time).'})
    DS['amean_filtered_running_field'].setncatts({'units':'mm day-1','long_name':'LP object filtered running mean, mean rain rate (at end of running time).'})
    DS['min_inst_field'].setncatts({'units':'mm h-1','long_name':'LP object min instantaneous rain rate (at end of running time).'})
    DS['min_running_field'].setncatts({'units':'mm day-1','long_name':'LP object running mean, min rain rate (at end of running time).'})
    DS['min_filtered_running_field'].setncatts({'units':'mm day-1','long_name':'LP object filtered running mean, min rain rate (at end of running time).'})
    DS['max_inst_field'].setncatts({'units':'mm h-1','long_name':'LP object max instantaneous rain rate (at end of running time).'})
    DS['max_running_field'].setncatts({'units':'mm day-1','long_name':'LP object running mean, max rain rate (at end of running time).'})
    DS['max_filtered_running_field'].setncatts({'units':'mm day-1','long_name':'LP object filtered running mean, max rain rate (at end of running time).'})


    DS.close()
