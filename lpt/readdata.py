import numpy as np
from netCDF4 import Dataset
import struct
import sys
import os
import datetime as dt
import glob

"""
This module contains functions for reading external data
to use with LPT.
"""



def read_generic_netcdf(fn, variable_names=('lon','lat','rain')):
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
    DATA['lon'] = DS[variable_names[0]][:]
    DATA['lat'] = DS[variable_names[1]][:]
    DATA['data'] = DS[variable_names[2]][:][0]
    DS.close()

    ## Need to get from (-180, 180) to (0, 360) longitude.
    lon_lt_0, = np.where(DATA['lon'] < -0.0001)
    lon_ge_0, = np.where(DATA['lon'] > -0.0001)
    if len(lon_lt_0) > 0:
        DATA['lon'][lon_lt_0] += 360.0
        DATA['lon'] = np.concatenate((DATA['lon'][lon_ge_0], DATA['lon'][lon_lt_0]))
        DATA['data'] = np.concatenate((DATA['data'][:,lon_ge_0], DATA['data'][:,lon_lt_0]), axis=1)

    return DATA


def read_generic_netcdf_at_datetime(dt, data_dir='.'
        , variable_names=('lon','lat','rain'), fmt='gridded_rain_rates_%Y%m%d%H.nc'
        , verbose=False):

    fn = (data_dir + '/' + dt.strftime(fmt))
    DATA=None

    if not os.path.exists(fn):
        print('File not found: ', fn)
    else:
        if verbose:
            print(fn)
        DATA=read_generic_netcdf(fn, variable_names = variable_names)

    return DATA
