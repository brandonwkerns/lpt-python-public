import numpy as np
from netCDF4 import Dataset
import struct
import sys
import os
import gdal
import datetime as dt
import glob

"""
This module contains functions for reading external data
to use with LPT.
"""

################################################################################
################################################################################
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
    DATA['precip'] = np.fromfile(fid, dtype=np.float32, count=2*4948*1649)
    if sys.byteorder == 'big': # Data is little endian.
        DATA['precip'] = DATA['precip'].byteswap()

    ## Shape and scale the data.
    DATA['precip'] = np.reshape(np.double(DATA['precip']), [2, 1649, 4948])
    DATA['precip'][DATA['precip'] < -0.001] = 0.0 # Usually, missing high latitude data.
    fid.close()

    ## Cut out area.
    keep_lon, = np.where(np.logical_and(DATA['lon'] > area[0], DATA['lon'] < area[1]))
    keep_lat, = np.where(np.logical_and(DATA['lat'] > area[2], DATA['lat'] < area[3]))

    DATA['lon'] = DATA['lon'][keep_lon[0]:keep_lon[-1]]
    DATA['lat'] = DATA['lat'][keep_lat[0]:keep_lat[-1]]
    DATA['precip'] = DATA['precip'][:, keep_lat[0]:keep_lat[-1], keep_lon[0]:keep_lon[-1]]
    DATA['precip'] = 0.5*(DATA['precip'][0,:,:] + DATA['precip'][1,:,:])

    return DATA



def read_cmorph_at_datetime(dt, force_rt=False, verbose=False, area=[0,360,-90,90]):

    """
    DATA = read_cmorph_at_datetime(dt, force_rt=False, verbose=False)

    DATA is a dict with keys lon, lat, and precip.

    Based on the provided datetime dt, read in the CMORPH data.
    By default, it will first check for the research product,
    and use the realtime product if the research product was not found.
    However, if force_rt = True, it just uses the realtime product.
    """

    YYYY = dt.strftime("%Y")
    MM = dt.strftime("%m")
    DD = dt.strftime("%d")
    HH = dt.strftime("%H")
    YMD = YYYY + MM + DD

    ## First try research product
    fn = ('/home/orca/data/satellite/cmorph/'
       + YYYY+'/'+MM+'/'+YMD+'/3B42.'+YMD+'.'+HH+'.7.HDF') #TODO: Update this for CMORPH final product.

    DATA=None
    if os.path.exists(fn) and not force_rt:
        if verbose:
            print(fn)
        DATA=read_tmpa_hdf(fn)
    else:
        ## If no research grade, use the research product
        fn = ('/home/orca/data/satellite/cmorph/rt/'
           + YYYY+'/'+MM+'/'+YMD+'/CMORPH_V0.x_RT_8km-30min_'+YMD+HH)

        if verbose:
            print(fn)
        DATA=read_cmorph_rt_bin(fn)
    return DATA


################################################################################
################################################################################
################################################################################


"""
TRMM 3B42/TMPA reading functions.
"""

def read_tmpa_hdf(fn):
    """
    DATA = read_tmpa_hdf(fn)

    output:
    list(DATA)
    Out[12]: ['lon', 'lat', 'precip']
    In [21]: DATA['lon'].shape
    Out[21]: (1440,)
    In [22]: DATA['lat'].shape
    Out[22]: (400,)
    In [23]: DATA['precip'].shape
    Out[23]: (400, 1440)
    """

    ## The TMPA HDF files can be read using NetCDF4 Dataset.
    DS = Dataset(fn)
    DATA={}
    DATA['lon'] = np.arange(-179.875, 180.0, 0.25)
    DATA['lat'] = np.arange(-49.875, 50.0, 0.25)
    DATA['precip'] = DS['precipitation'][:].T
    DS.close()

    ## Need to get from (-180, 180) to (0, 360) longitude.
    lon_lt_0, = np.where(DATA['lon'] < -0.0001)
    lon_ge_0, = np.where(DATA['lon'] > -0.0001)
    DATA['lon'][lon_lt_0] += 360.0
    DATA['lon'] = np.concatenate((DATA['lon'][lon_ge_0], DATA['lon'][lon_lt_0]))
    DATA['precip'] = np.concatenate((DATA['precip'][:,lon_ge_0], DATA['precip'][:,lon_lt_0]), axis=1)

    return DATA


def read_tmpa_rt_bin(fn):
    """
    RT = read_tmpa_rt_bin(fn)

    output:
    In [24]: list(RT)
    Out[24]: ['lon', 'lat', 'precip']
    In [25]: RT['lon'].shape
    Out[25]: (1440,)
    In [26]: RT['lat'].shape
    Out[26]: (480,)
    In [27]: RT['precip'].shape
    Out[27]: (480, 1440)

    missing values (stored as -31999) are set to np.NaN.
    """

    ## TMPA RT files are binary.
    dtype=np.dtype([('field1', '<i2')])
    DATA={}
    DATA['lon'] = np.arange(0.125, 360.0, 0.25)
    DATA['lat'] = np.arange(-59.875, 60.0, 0.25)
    fid = open(fn,'rb')

    ## skip header
    fid.seek(2880)
    DATA['precip'] = np.fromfile(fid, dtype=np.int16, count=691200)
    if sys.byteorder == 'little':
        DATA['precip'] = DATA['precip'].byteswap()

    ## Shape and scale the data.
    DATA['precip'] = np.flip(np.reshape(np.double(DATA['precip']) / 100.0, [480, 1440]), axis=0)
    DATA['precip'][DATA['precip'] < -0.001] = 0.0 # Usually, missing high latitude data.
    fid.close()

    return DATA


def read_tmpa_at_datetime(dt, force_rt=False, verbose=False):

    """
    DATA = read_tmpa_at_datetime(dt, force_rt=False, verbose=False)

    DATA is a dict with keys lon, lat, and precip.

    Based on the provided datetime dt, read in the TMPA data.
    By default, it will first check for the research product,
    and use the realtime product if the research product was not found.
    However, if force_rt = True, it just uses the realtime product.
    """

    YYYY = dt.strftime("%Y")
    MM = dt.strftime("%m")
    DD = dt.strftime("%d")
    HH = dt.strftime("%H")
    YMD = YYYY + MM + DD

    ## First try research product
    fn = ('/home/orca/data/satellite/trmm_global_rainfall/'
       + YYYY+'/'+MM+'/'+YMD+'/3B42.'+YMD+'.'+HH+'.7.HDF')

    DATA=None

    ## Sometimes, the file name is "7A" instead of "7".
    if not os.path.exists(fn):
        fn = ('/home/orca/data/satellite/trmm_global_rainfall/'
           + YYYY+'/'+MM+'/'+YMD+'/3B42.'+YMD+'.'+HH+'.7A.HDF')

    if os.path.exists(fn) and not force_rt:
        if verbose:
            print(fn)
        DATA=read_tmpa_hdf(fn)
    else:
        ## If no research grade, use the research product
        fn = ('/home/orca/data/satellite/trmm_global_rainfall/rt/'
           + YYYY+'/'+MM+'/'+YMD+'/3B42RT.'+YMD+HH+'.7.bin')

        ## Sometimes, the file name is "7A" instead of "7".
        if not os.path.exists(fn):
            fn = ('/home/orca/data/satellite/trmm_global_rainfall/rt/'
               + YYYY+'/'+MM+'/'+YMD+'/3B42RT.'+YMD+HH+'.7A.bin')

        if verbose:
            print(fn)
        DATA=read_tmpa_rt_bin(fn)
    return DATA

"""
CFS Grib2 reading function
"""

def read_cfs_rt_at_datetime(dt, records=range(1,45*4+1), verbose=False):

    YYYY = dt.strftime("%Y")
    MM = dt.strftime("%m")
    DD = dt.strftime("%d")
    HH = dt.strftime("%H")
    YMD = YYYY + MM + DD

    ## First try research product
    fn = ('/home/orca/data/model_fcst_grib/cfs/cfs.'
       +YMD+'/'+HH+'/time_grib_01/prate.01.'+ YMD + HH + '.daily.grb2')

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


def read_era5_nc(fn, init_time_list=None, fcst_list=range(19), verbose=False):
    """
    RT = read_era5_nc(fn)

    Units are accumulation in meters for a 1 h forecast period.
    (forecast time 0 is "initialization", which seems to [and should be!] always be zero.)
    """
    DS = Dataset(fn)
    lon =  DS['longitude'][:]
    lat =  DS['latitude'][:]

    ## Read precip. Use init forecast time list, if provided. Otherwise, return all initializations.
    if init_time_list is None:
        precip = DS['TP'][:,fcst_list,:,:]
    else:
        precip = DS['TP'][init_time_list,fcst_list,:,:]

    DS.close() # Close the file.

    DATA={}
    DATA['lon'] = lon
    DATA['lat'] = lat
    DATA['precip'] = precip

    return DATA


def get_era5_12h_rain(dt_ending, verbose=False):

    """
    Read in the rainfall using read_era5_nc(fn)
    Then calculate the 12 hourly rain rate (mm/h) and return it.
    """

    dt_beginning = dt_ending - dt.timedelta(hours=18)

    if dt_beginning < dt.datetime(dt_beginning.year, dt_beginning.month,1,5,59,0):
        dt00 = dt_beginning - dt.timedelta(month=1)
        dt0 = dt.datetime(dt00.year, dt00.month,16,6,0,0)
    elif dt_beginning < dt.datetime(dt_beginning.year, dt_beginning.month,16,5,59,0):
        dt0 = dt.datetime(dt_beginning.year, dt_beginning.month,1,6,0,0)
    else:
        dt0 = dt.datetime(dt_beginning.year, dt_beginning.month,16,6,0,0)

    fn = glob.glob('/home/orca/data/model_anal/era5/rain_accum/e5.oper.fc.sfc.accumu.128_228_tp.regn320sc.'+dt0.strftime('%Y%m%d%H')+'*.nc')[0]

    if verbose:
        print(fn, flush=True)


    init_time_indx = int((dt_beginning - dt0).total_seconds()/(12*3600))  # 12h increments from beginning of file.

    F = read_era5_nc(fn, init_time_list=(init_time_indx,), fcst_list=range(6,19), verbose=verbose)

    precip12hr = 1000.0 * np.nanmean(F['precip'], axis=1)[0] # m to mm

    DATA={}
    DATA['lon'] = F['lon']
    DATA['lat'] = F['lat']
    DATA['precip'] = precip12hr

    return DATA


def get_merra2_6h_rain(dt_ending, verbose=False
        , raw_data_parent_dir='/home/orca/asavarin/LPT/MERRA2'):

    """
    Read in the 6 h MERRA2 rainfall from the daily files with hourly data.

    MERRA2 hourly rain rate is from the data increment update (IAU) step
    of the assimilation cycle. It is a six hour analysis cycle, with the IAU period
    +-3h from the analysis point. The valid time for the rain rate is
    on the half hour, e.g., 0:30, 1:30, ect. Therefore, as an example,
    for 6 UTC, I use 3:30, 4:30, 5:30, 6:30, 7:30, and 8:30. These are time
    indices range(2:8) in the daily file.

    Comment: To be most consistent with TRMM, this might use just the 5:30 time.
    However, other model analyis products tend to be for accumulation over 6-12 h.
    Finally, I use the 3 day accumulation for LPT.

    Also: I am using PRECTOT for now. This is the raw output from the IAU step.
    this has known issues over tropics land!
    Potentially, I could use the PRECTOTCORR (corrected to obs over land) instead.
    """

    dt_list = [dt_ending + dt.timedelta(hours=x) for x in range(-3,3,1)]

    lon = None
    lat = None
    precip = None

    for this_dt in dt_list:

        fn = glob.glob(raw_data_parent_dir
            + '/MERRA2_*.tavg1_2d_flx_Nx.'+this_dt.strftime('%Y%m%d')+'.SUB.nc')[0]

        if verbose:
            print(fn)
            print(int(this_dt.hour), flush=True)

        DS = Dataset(fn)

        if lon is None:
            lon = DS['lon'][:]
        if lat is None:
            lat = DS['lat'][:]

        if precip is None:
            precip = DS['PRECTOT'][int(this_dt.hour),:,:]
        else:
            precip += DS['PRECTOT'][int(this_dt.hour),:,:]

    precip = precip/6.0
    precip *= 3600.0   # kg m-2 s-1 = mm s-1 ==? mm h-1

    ## (-180 180) --> (0 360)
    idx_lt_0 = [ii for ii in range(len(lon)) if lon[ii] < -0.001]
    idx_lt_0_max = np.max(idx_lt_0)
    lon = np.append(lon[idx_lt_0_max+1:], 360.0 + lon[:idx_lt_0_max+1])
    precip = np.append(precip[:,idx_lt_0_max+1:], precip[:,:idx_lt_0_max+1], axis=1)

    DATA={}
    DATA['lon'] = lon
    DATA['lat'] = lat
    DATA['precip'] = precip

    return DATA


## WRF Rainfall Data
## First read in d01.
## TO DO: replace grid points with d02, ect., where appropriate.

def get_wrfout_rain(dt_ending, verbose=False, raw_data_parent_dir='./'):

    fmt = '%Y-%m-%d_%H:00:00'
    timestamp_ending = dt_ending.strftime(fmt)
    if verbose:
        print(raw_data_parent_dir + '/wrfout_d01_' + timestamp_ending)
    DS = Dataset(raw_data_parent_dir + '/wrfout_d01_' + timestamp_ending)

    lon = DS['XLONG'][:][0]
    lat = DS['XLAT'][:][0]
    precip = DS['RAINC'][:][0] + DS['RAINNC'][:][0]
    if 'RAINSH' in DS.variables:
        precip += DS['RAINSH'][:][0]

    DS.close()

    DATA={}
    DATA['lon'] = lon
    DATA['lat'] = lat
    DATA['precip'] = precip

    return DATA
