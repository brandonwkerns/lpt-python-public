import numpy as np
from numpy import ma
import xarray as xr
from netCDF4 import Dataset
import struct
import sys
import os
import datetime as dt
import glob

"""
This module contains functions for reading external data
to use with LPT.

The data_read_function is called at various points in other LPT functions.

To add a new data set, do the following:
1) Write a read function similar to read_generic_netcdf below.
2) Add an "elif" option that calls that function in readdata
"""


################################################################################

def readdata(datetime_to_read, dataset_options_dict, verbose=None):
    """
    Main data read function. Get data at datetime datetime_to_read.
    Based on the oprions in dataset_options_dict, it will look in the data directory
    and use the rain function specified below.

    To add a dataset type, add an elif block to this function.

    The function is expected to return a dictionary with keys 'lon', 'lat', and 'data'

    Verbose option (new 05/2023):
    - If set to None (default), it will use the verbose option from dataset_options_dict.
    - Otherwise, the value will be used *instead of* dataset_options_dict.
      This allows a function call to override the setting in dataset_options_dict.
    """

    ## Manage verbose
    if verbose is None:
        verbose_actual = dataset_options_dict['verbose']
    else:
        verbose_actual = verbose

    if dataset_options_dict['raw_data_format'] == 'generic_netcdf':
        variable_names = (dataset_options_dict['longitude_variable_name']
                , dataset_options_dict['latitude_variable_name']
                , dataset_options_dict['field_variable_name'])

        DATA = read_generic_netcdf_at_datetime(datetime_to_read
                , variable_names = variable_names
                , data_dir = dataset_options_dict['raw_data_parent_dir']
                , fmt = dataset_options_dict['file_name_format']
                , verbose = verbose_actual)

    if dataset_options_dict['raw_data_format'] == 'generic_netcdf_with_multiple_times':
        variable_names = (dataset_options_dict['longitude_variable_name']
                , dataset_options_dict['latitude_variable_name']
                , dataset_options_dict['time_variable_name']
                , dataset_options_dict['field_variable_name'])

        DATA = read_generic_netcdf_at_datetime(datetime_to_read
                , variable_names = variable_names
                , dt_to_use = datetime_to_read
                , data_dir = dataset_options_dict['raw_data_parent_dir']
                , fmt = dataset_options_dict['file_name_format']
                , verbose = verbose_actual)


    elif dataset_options_dict['raw_data_format'] == 'cmorph':
        DATA = read_cmorph_at_datetime(datetime_to_read
                , area = dataset_options_dict['area']
                , data_dir = dataset_options_dict['raw_data_parent_dir']
                , fmt = dataset_options_dict['file_name_format']
                , verbose = verbose_actual)

    elif dataset_options_dict['raw_data_format'] == 'imerg_hdf5':
        DATA = read_imerg_hdf5_at_datetime(datetime_to_read
                , area = dataset_options_dict['area']
                , data_dir = dataset_options_dict['raw_data_parent_dir']
                , fmt = dataset_options_dict['file_name_format']
                , verbose = verbose_actual)

    elif dataset_options_dict['raw_data_format'] == 'cfs_forecast':
        fcst_hour = int((datetime_to_read - dataset_options_dict['datetime_init']).total_seconds()/3600)
        fcst_resolution_hours = dataset_options_dict['data_time_interval']
        if fcst_hour < 1: # There is no data in the file for fcst = 0. Use 6h fcst values.
            records = [1,]
        else:
            records = [int(fcst_hour/fcst_resolution_hours),]

        DATA = read_cfs_rt_at_datetime(dataset_options_dict['datetime_init'] # datetime_to_read
                , data_dir = dataset_options_dict['raw_data_parent_dir']
                , fmt = dataset_options_dict['file_name_format']
                , records = records
                , verbose = verbose_actual)
        DATA['data'] = ma.masked_array(DATA['precip'][0])

    ## -- Add an elif block here for new datasets. --

    else:
        print(('ERROR! '+dataset['raw_data_format'] + ' is not a valid raw_data_format!'), flush=True)
        DATA = None

    return DATA

################################################################################
## Read functions for generic NetCDF data.
################################################################################

def read_generic_netcdf(fn, variable_names=('lon','lat','rain'), dt_to_use=None):
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

    DATA = {}
    with xr.open_dataset(fn) as DS:
        DATA['lon'] = DS[variable_names[0]].values
        DATA['lat'] = DS[variable_names[1]].values
        ## If no time variable, just retrieve the 2-D data as it is.
        if not dt_to_use is None: #'time' in list(DS.variables):
            DATA['data'] = DS.sel({variable_names[2]:str(dt_to_use)},method='nearest')[variable_names[3]].values
        else:
            DATA['data'] = DS[variable_names[2]].values

    DATA['data'] = np.ma.masked_array(DATA['data'], mask=np.isnan(DATA['data']))

    ## Need to get from (-180, 180) to (0, 360) longitude.
    lon_lt_0, = np.where(DATA['lon'] < -0.0001)
    lon_ge_0, = np.where(DATA['lon'] > -0.0001)
    if len(lon_lt_0) > 0:
        DATA['lon'][lon_lt_0] += 360.0
        DATA['lon'] = np.concatenate((DATA['lon'][lon_ge_0], DATA['lon'][lon_lt_0]))
        DATA['data'] = np.concatenate((DATA['data'][:,lon_ge_0], DATA['data'][:,lon_lt_0]), axis=1)

    return DATA


def read_generic_netcdf_at_datetime(dt, data_dir='.'
        , variable_names=('lon','lat','rain'), dt_to_use=None, fmt='gridded_rain_rates_%Y%m%d%H.nc'
        , verbose=False):

    fn = (data_dir + '/' + dt.strftime(fmt))
    DATA=None

    if not os.path.exists(fn):
        print('File not found: ', fn)
    else:
        if verbose:
            print(fn)
        DATA=read_generic_netcdf(fn,
            variable_names = variable_names,
            dt_to_use = dt_to_use)

    return DATA


################################################################################
## Read functions for specific datasets.
################################################################################

"""
CMORPH reading functions.
"""
def read_cmorph_rt_bin(fn, area=[0,360,-90,90]):

    """
    DATA = read_cmorph_rt_bin(fn)
    DATA is a dict with keys lon, lat, and precip.

    CMORPH RT files are binary.
    The GrADS control file below is used as the basis for this function:

    DSET ^../%y4/%y4%m2/CMORPH_V0.x_RT_8km-30min_%y4%m2%d2%h2
    OPTIONS little_endian template
    UNDEF -999.0
    TITLE CMORPH Rain Rate (Real-Time Version)
    XDEF  4948 LINEAR   0.0363783345 0.072756669
    YDEF  1649 LINEAR -59.963614312  0.072771376
    ZDEF     1 LEVELS   1
    TDEF 99999 LINEAR 00:00z01Jan2017 30mn
    VARS 1
    cmorph  1  99  CMORPH Rain Rate [mm/hr]
    ENDVARS
    """

    dtype=np.dtype([('field1', '<i2')])
    DATA={}
    DATA['lon'] = np.arange(0.0363783345, 360.0, 0.072756669)
    DATA['lat'] = np.arange(-59.963614312, 60.0, 0.072771376)
    fid = open(fn,'rb')

    ## GrADS uses FORTRAN REAL values, which is np.float32 for Python.
    DATA['data'] = np.fromfile(fid, dtype=np.float32, count=2*4948*1649)
    if sys.byteorder == 'big': # Data is little endian.
        DATA['data'] = DATA['data'].byteswap()

    ## Shape and scale the data.
    DATA['data'] = np.reshape(np.double(DATA['data']), [2, 1649, 4948])
    DATA['data'][DATA['data'] < -0.001] = 0.0 # Usually, missing high latitude data.
    fid.close()

    ## Cut out area.
    keep_lon, = np.where(np.logical_and(DATA['lon'] > area[0], DATA['lon'] < area[1]))
    keep_lat, = np.where(np.logical_and(DATA['lat'] > area[2], DATA['lat'] < area[3]))

    DATA['lon'] = DATA['lon'][keep_lon[0]:keep_lon[-1]+1]
    DATA['lat'] = DATA['lat'][keep_lat[0]:keep_lat[-1]+1]
    DATA['data'] = DATA['data'][:, keep_lat[0]:keep_lat[-1]+1, keep_lon[0]:keep_lon[-1]+1]
    DATA['data'] = 0.5*(DATA['data'][0,:,:] + DATA['data'][1,:,:])

    return DATA



def read_cmorph_at_datetime(dt_this, force_rt=False, data_dir='.'
        , fmt='CMORPH_V0.x_RT_8km-30min_%Y%m%d%H'
        , verbose=False, area=[0,360,-90,90]):

    """
    DATA = read_cmorph_at_datetime(dt, force_rt=False, verbose=False)

    DATA is a dict with keys lon, lat, and precip.

    Based on the provided datetime dt, read in the CMORPH data.

    By default, it will first check for the research product,
    and use the realtime product if the research product was not found.
    However, if force_rt = True, it just uses the realtime product.
    """

    ## First try research product
    fn = (data_dir + '/' + dt_this.strftime(fmt))
    if verbose:
        print(fn)
    DATA = read_cmorph_rt_bin(fn, area=area)

    DATA['data'] = ma.masked_array(DATA['data'])
    return DATA


def read_imerg_hdf5_at_datetime(dt_this, force_rt=False, data_dir='.'
        , fmt='%Y/%m/%d/3B-HHR.MS.MRG.3IMERG.%Y%m%d-S%H*.HDF5'
        , verbose=False, area=[0,360,-90,90]):

    """
    DATA = read_imerg_hdf5_at_datetime(dt_this, force_rt=False, data_dir='.'
        , fmt='%Y/%m/%d/3B-HHR.MS.MRG.3IMERG.%Y%m%d-S%H*.HDF5'
        , verbose=False, area=[0,360,-90,90])

    DATA is a dict with keys lon, lat, and precip.

    Based on the provided datetime dt, read in the IMERG HDF data.

    By default, it will first check for the final product,
    and use the "late" realtime product if the final product was not found.
    However, if force_rt = True, it just uses the "late" realtime product.

    (It will search for a filename with modified fmt to check for "late" product
    - append 'late/' to the front of the directory path.
    - replace '3B-HHR' with '3B-HHR-L').
    """

    fn_list = sorted(glob.glob(data_dir + '/' + dt_this.strftime(fmt)))
    if len(fn_list) < 1:
        if not force_rt:
            ## Try "late" realtime data.
            print('Final data version not found. Trying to use late realtime data instead.')
            fmt_rt = 'late/' + fmt.replace('3B-HHR','3B-HHR-L')
            fn_list = sorted(glob.glob(data_dir + '/' + dt_this.strftime(fmt_rt)))

    if len(fn_list) < 1:
        print('WARNING: No input data found.')

    fn = fn_list[0]
    if verbose:
        print(fn)

    with Dataset(fn) as DS:
        lon_rain = DS['Grid']['lon'][:]
        lat_rain = DS['Grid']['lat'][:]
        rain = DS['Grid']['precipitationCal'][:][0].T

    if len(fn_list) > 1:
        fn = fn_list[1]
        if verbose:
            print(fn)

        with Dataset(fn) as DS:
            rain30 = DS['Grid']['precipitationCal'][:][0].T

        rain = 0.5 * (rain + rain30)

    ## lon -180:0 --> 180:360
    idx_neg_lon = [x for x in range(len(lon_rain)) if lon_rain[x] < -0.0001]
    idx_pos_lon = [x for x in range(len(lon_rain)) if lon_rain[x] > -0.0001]

    lon_rain = np.append(lon_rain[idx_pos_lon[0]:idx_pos_lon[-1]+1], 360.0 + lon_rain[idx_neg_lon[0]:idx_neg_lon[-1]+1], axis=0)
    rain = np.append(rain[:,idx_pos_lon[0]:idx_pos_lon[-1]+1], rain[:,idx_neg_lon[0]:idx_neg_lon[-1]+1], axis=1)
    
    DATA={}
    DATA['lon'] = lon_rain
    DATA['lat'] = lat_rain
    DATA['data'] = ma.masked_array(rain)

    ## Cut out area.
    keep_lon, = np.where(np.logical_and(DATA['lon'] > area[0], DATA['lon'] < area[1]))
    keep_lat, = np.where(np.logical_and(DATA['lat'] > area[2], DATA['lat'] < area[3]))

    DATA['lon'] = DATA['lon'][keep_lon[0]:keep_lon[-1]+1]
    DATA['lat'] = DATA['lat'][keep_lat[0]:keep_lat[-1]+1]
    DATA['data'] = DATA['data'][keep_lat[0]:keep_lat[-1]+1, keep_lon[0]:keep_lon[-1]+1]

    return DATA


################################################################################
################################################################################
################################################################################


"""
CFS Grib2 reading function
"""

def read_cfs_rt_at_datetime(dt_this, data_dir = './'
                , fmt = 'cfs.%Y%m%d/%H/time_grib_01/prate.01.%Y%m%d%H.daily.grb2'
                , records=range(1,45*4+1), verbose=False):

    fn = (data_dir + '/' + dt_this.strftime(fmt))
    if verbose:
        print(fn, flush=True)

    return read_cfs_rt_grib2(fn, records=records, verbose=verbose)


def read_cfs_rt_grib2(fn, records=range(1,45*4+1), verbose=False):
    """
    RT = read_cfs_rt_grib2(fn, records=N)

    N is the list of records to get.
    By default, get the first 45 days, 6 hourly intervals.

    example output:
    In [23]: RT['lon'].shape
    Out[23]: (384,)

    In [24]: RT['lat'].shape
    Out[24]: (190,)

    In [25]: RT['precip'].shape
    Out[25]: (180, 190, 384)
    """
    
    import gdal  # Import gdal if dealing with grib data.

    DS = gdal.Open(fn, gdal.GA_ReadOnly)
    width = DS.RasterXSize
    height = DS.RasterYSize
    lon =  np.arange(0.0,359.062 + 0.5,0.938)
    ## grid file with Gaussian latitude was obtained from wgrib2 like this:
    ## wgrib2 -d 1 -gridout grid.txt /home/orca/data/model_fcst_grib/cfs/cfs.20190508/00/time_grib_01/prate.01.2019050800.daily.grb2
    ## awk -F, '{print $3}' grid.txt | uniq | tr "\n" ", "
    lat = np.flip(np.array([-89.277, -88.340, -87.397, -86.454, -85.509
                , -84.565, -83.620, -82.676, -81.731, -80.786
                , -79.841, -78.897, -77.952, -77.007, -76.062
                , -75.117, -74.173, -73.228, -72.283, -71.338
                , -70.393, -69.448, -68.503, -67.559, -66.614
                , -65.669, -64.724, -63.779, -62.834, -61.889
                , -60.945, -60.000, -59.055, -58.110, -57.165
                , -56.220, -55.275, -54.330, -53.386, -52.441
                , -51.496, -50.551, -49.606, -48.661, -47.716
                , -46.771, -45.827, -44.882, -43.937, -42.992
                , -42.047, -41.102, -40.157, -39.212, -38.268
                , -37.323, -36.378, -35.433, -34.488, -33.543
                , -32.598, -31.653, -30.709, -29.764, -28.819
                , -27.874, -26.929, -25.984, -25.039, -24.094
                , -23.150, -22.205, -21.260, -20.315, -19.370
                , -18.425, -17.480, -16.535, -15.590, -14.646
                , -13.701, -12.756, -11.811, -10.866, -9.921
                , -8.976, -8.031, -7.087, -6.142, -5.197
                , -4.252, -3.307, -2.362, -1.417, -0.472
                , 0.472, 1.417, 2.362, 3.307, 4.252
                , 5.197, 6.142, 7.087, 8.031, 8.976
                , 9.921, 10.866, 11.811, 12.756, 13.701
                , 14.646, 15.590, 16.535, 17.480, 18.425
                , 19.370, 20.315, 21.260, 22.205, 23.150
                , 24.094, 25.039, 25.984, 26.929, 27.874
                , 28.819, 29.764, 30.709, 31.653, 32.598
                , 33.543, 34.488, 35.433, 36.378, 37.323
                , 38.268, 39.212, 40.157, 41.102, 42.047
                , 42.992, 43.937, 44.882, 45.827, 46.771
                , 47.716, 48.661, 49.606, 50.551, 51.496
                , 52.441, 53.386, 54.330, 55.275, 56.220
                , 57.165, 58.110, 59.055, 60.000, 60.945
                , 61.889, 62.834, 63.779, 64.724, 65.669
                , 66.614, 67.559, 68.503, 69.448, 70.393
                , 71.338, 72.283, 73.228, 74.173, 75.117
                , 76.062, 77.007, 77.952, 78.897, 79.841
                , 80.786, 81.731, 82.676, 83.620, 84.565
                , 85.509, 86.454, 87.397, 88.340, 89.277]), axis=0)


    num_list = []
    for band in records:
        if verbose:
            print('Record #' + str(band), flush=True)
        data_array = DS.GetRasterBand(band).ReadAsArray()
        for row in data_array:
            for value in row:
                num_list.append(value*3600.0) # kg/m2/sec --> mm/h

    DS = None # Close the file.

    precip = np.array(num_list).reshape([len(records), len(lat), len(lon)])

    DATA={}
    DATA['lon'] = lon
    DATA['lat'] = lat
    DATA['precip'] = precip

    return DATA


def read_cfsr_grib2(fn, band_list=None, verbose=False):
    """
    RT = read_cfsr_grib2(fn)

    example output:
    In [23]: RT['lon'].shape
    Out[23]: (384,)

    In [24]: RT['lat'].shape
    Out[24]: (190,)

    In [25]: RT['precip'].shape
    Out[25]: (180, 190, 384)
    """

    DS = gdal.Open(fn, gdal.GA_ReadOnly)
    width = DS.RasterXSize
    height = DS.RasterYSize
    lon =  np.arange(0.0,359.51,0.5)
    lat =  np.arange(90.0,-90.01,-0.5)
    n_records = DS.RasterCount

    num_list = []

    if band_list is None:
        band_list = range(1, n_records+1)

    for band in band_list:
        if verbose:
            print((str(band) + ' of ' + str(n_records)))
        data_array = DS.GetRasterBand(band).ReadAsArray()
        for row in data_array:
            for value in row:
                num_list.append(value)

    DS = None # Close the file.

    precip = np.array(num_list).reshape([int(len(band_list)/6), 6, len(lat), len(lon)])
    #precip /= 1e6  # Values in file are multiplied by 1e6.
                   # kg/m2 in 1h is equivalent to mm/h.

    DATA={}
    DATA['lon'] = lon
    DATA['lat'] = lat
    DATA['precip'] = precip

    return DATA

def get_cfsr_6h_rain(dt_ending, verbose=False):

    """
    Read in the rainfall using read_cfs_historical_grib2(fn)
    Then calculate the 6 hourly rain rate (mm/h) and return it.

    CFSR rain is stored in monthly files. It it initialized every 6 h,
    and the data provide hourly accumulations (in kg/m^2, equivalent to mm) like this:

    1:0:d=2011120100:APCP:surface:0-1 hour acc fcst:
    2:94325:d=2011120100:APCP:surface:1-2 hour acc fcst:
    3:193206:d=2011120100:APCP:surface:2-3 hour acc fcst:
    4:309596:d=2011120100:APCP:surface:3-4 hour acc fcst:
    5:421187:d=2011120100:APCP:surface:4-5 hour acc fcst:
    6:537704:d=2011120100:APCP:surface:5-6 hour acc fcst:

    To get the 6 hourly accumulation, all 6 of these need to be added.
    Then take the mean (e.g., divide by 6h) to get mm/h.
    """

    dt_beginning = dt_ending - dt.timedelta(hours=6)

    if dt_beginning < dt.datetime(2011,3,31,23,59,0):
        fn_beginning = ('/home/orca/data/model_anal/cfsr/rain_accum/' + dt_beginning.strftime('%Y')
            + '/apcp.gdas.' + dt_beginning.strftime('%Y%m') + '.grb2')
    else:
        fn_beginning = ('/home/orca/data/model_anal/cfsr/rain_accum/' + dt_beginning.strftime('%Y')
            + '/apcp.cdas1.' + dt_beginning.strftime('%Y%m') + '.grb2')

    if verbose:
        print(fn_beginning, flush=True)

    rec_num = 1 + int((dt_beginning - dt.datetime(dt_beginning.year, dt_beginning.month,1,0,0,0)).total_seconds()/3600.0)
    F = read_cfsr_grib2(fn_beginning, band_list=range(rec_num,rec_num+6,1), verbose=verbose)

    precip6hr = np.nanmean(F['precip'], axis=1)[0]

    DATA={}
    DATA['lon'] = F['lon']
    DATA['lat'] = F['lat']
    DATA['precip'] = precip6hr

    return DATA
