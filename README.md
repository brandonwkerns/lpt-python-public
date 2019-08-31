# lpt-python-public
Python version of Large-Scale Precipitation Tracking (LPT): Public Release.
This version of LPT is to be released with Kerns and Chen (2019), submitted to the Journal of Climate.

This version is set up to run with "generic" NetCDF files with the following format:

```
## netcdf gridded_rain_rates_YYYYMMDDHH {
## dimensions:
##        lon = NNN ;
##        lat = MMM ;
##        time = UNLIMITED ; // (1 currently)
## variables:
##        double lon(lon) ;
##        double lat(lat) ;
##        double time(time) ;
##                time:units = "hours since 1970-1-1 0:0:0" ;
##        double rain(time, lat, lon) ;
##                rain:units = "mm h-1" ;
##
## Where NNN and MMM are the longitude and latitude array sizes.
##
##       +++ If lon is -180 to 180, it will be converted to 0 - 360. +++
## The names of the coordinate variables "lon" and "lat" and data variable "rain"
##    can be set in the driver script.
```

*This version includes splitting up LPT system groups in to track branches.*  
*This version DOES NOT yet include MJO identification.*

## MASTER_RUN directory.
__Please do not modify the files in the MASTER_RUN directory unless you are doing code development.__
They are Github repository files and can be updated with `git pull`. But `git pull` will cause
issues if the files have been previously updated locally!
Instead, copy the directory to a local working directory
 which you can feel free to modify to run your case.


## Python module dependencies (see below for full environment I used):
- numpy
- scipy.signal, scipy.ndimage
- NetCDF4.Dataset
- gdal (for reading grib files)

## Code organization:
- The main Python functions directory is lpt/.
  * Functions for reading data are in lpt/readdata.py.
  * Functions for LP object and LPTs input/output are in lpt/lptio.py.
  * Supporting functions for calculations are in lpt/helpers.py.
  * Example plotting functions are in lpt/plotting.py
- The following generic NetCDF data driver scripts is included in MASTER_RUN/:
  * lpt_generic_netcdf_data_driver.py
- The following example preprocessor scripts are included in MASTER_RUN/:
  * preprocess_tmpa.py
  * preprocess_wrf.py
  * preprocess_global_4km_ir.py
- The following directories are included by default under the MASTER_RUN/ directory
    (these are used by default in the driver script)
  * data/
    + raw/                         Optional: You can link or copy raw data files here.
    + interim/                     Gridded NetCDF files.
    + processed/
      - objects/                   "Objects" snapshots files.
      - systems/                   "Systems" tracked in time files.
  * images
    + objects/
    + systems/

## Setting up LPT on a new system:
1) Clone this repository to your system, or download the zip file format.
2) Copy the MASTER_RUN directory to a new run directory.
3) Edit the lpt_driver.py script as needed.
4) Run the lpt_driver.py script.



## Python environment

I used Anaconda Python 3.6.2 with the following environment:

```
 $ conda list                                                                                                          [16:36:42]
# packages in environment at /home/disk/atmos/bkerns/anaconda3/envs/meteo:
#
asn1crypto                0.24.0                py36_1003    conda-forge
backports                 1.0                      py36_1    conda-forge
backports.functools_lru_cache 1.5                      py36_0    conda-forge
basemap                   1.2.0            py36h673bf1a_2    conda-forge
basemap-data-hires        1.1.0                         0    conda-forge
blas                      1.0                         mkl  
boost-cpp                 1.68.0            h11c811c_1000    conda-forge
bzip2                     1.0.6             h14c3975_1002    conda-forge
ca-certificates           2018.1.18                     0    conda-forge
cairo                     1.14.12              he6fea26_5    conda-forge
cartopy                   0.17.0          py36h0aa2c8f_1004    conda-forge
certifi                   2018.1.18                py36_0    conda-forge
cffi                      1.12.3           py36h8022711_0    conda-forge
cftime                    1.0.3.4         py36h3010b51_1000    conda-forge
chardet                   3.0.4                 py36_1003    conda-forge
cmocean                   2.0                        py_0    conda-forge
cryptography              2.6.1            py36h72c5cf5_0    conda-forge
curl                      7.64.1               hf8cf82a_0    conda-forge
cycler                    0.10.0                   py36_0    conda-forge
dbus                      1.13.0               h3a4f0e9_0    conda-forge
decorator                 4.2.1                    py36_0    conda-forge
expat                     2.2.5                         0    conda-forge
fontconfig                2.13.1               h65d0f4c_0    conda-forge
freetype                  2.9.1                h6debe1e_4    conda-forge
freexl                    1.0.5             h14c3975_1002    conda-forge
gdal                      2.4.1            py36hf242f0b_0    conda-forge
geos                      3.7.1             hf484d3e_1000    conda-forge
geotiff                   1.4.3             h1105359_1000    conda-forge
gettext                   0.19.8.1                      0    conda-forge
giflib                    5.1.7                h516909a_1    conda-forge
glib                      2.55.0                        0    conda-forge
gst-plugins-base          1.12.5               hde13a9d_0    conda-forge
gstreamer                 1.12.5               h61a6719_0    conda-forge
h5py                      2.9.0           nompi_py36hf008753_1102    conda-forge
hdf4                      4.2.13                        0    conda-forge
hdf5                      1.10.4          nompi_h3c11f04_1106    conda-forge
icu                       58.2                          0    conda-forge
idna                      2.8                   py36_1000    conda-forge
intel-openmp              2018.0.0                      8  
ipython                   6.2.1                    py36_1    conda-forge
ipython_genutils          0.2.0                    py36_0    conda-forge
jedi                      0.11.1                   py36_0    conda-forge
jpeg                      9c                   h470a237_1    conda-forge
json-c                    0.13.1            h14c3975_1001    conda-forge
kealib                    1.4.10            h1978553_1003    conda-forge
kiwisolver                1.0.1                    py36_1    conda-forge
krb5                      1.16.3            h05b26f9_1001    conda-forge
libcurl                   7.64.1               hda55be3_0    conda-forge
libdap4                   3.19.1            hd48c02d_1000    conda-forge
libedit                   3.1.20170329      hf8c457e_1001    conda-forge
libffi                    3.2.1                         3    conda-forge
libgcc-ng                 8.2.0                hdf63c60_1  
libgdal                   2.4.1                hdb8f723_0    conda-forge
libgfortran               3.0.0                         1  
libgfortran-ng            7.2.0                hdf63c60_3  
libiconv                  1.15                          0    conda-forge
libkml                    1.3.0             h328b03d_1009    conda-forge
libnetcdf                 4.6.2             hbdf4f91_1001    conda-forge
libpng                    1.6.37               hed695b0_0    conda-forge
libpq                     11.2                 h4770945_0    conda-forge
libspatialite             4.3.0a            hb5ec416_1026    conda-forge
libssh2                   1.8.2                h22169c7_2    conda-forge
libstdcxx-ng              8.2.0                hdf63c60_1  
libtiff                   4.0.9                he6b73bb_2    conda-forge
libuuid                   2.32.1               h470a237_2    conda-forge
libxcb                    1.13                          0    conda-forge
libxml2                   2.9.8                         0    conda-forge
libxslt                   1.1.32            h4785a14_1002    conda-forge
lxml                      4.3.3            py36h7ec2d77_0    conda-forge
matplotlib                3.0.3                    py36_1    conda-forge
matplotlib-base           3.0.3            py36h5f35d83_1    conda-forge
mkl                       2018.0.2                      1  
mkl_fft                   1.0.6                    py36_0    conda-forge
mkl_random                1.0.1                    py36_0    conda-forge
ncurses                   6.1               hf484d3e_1002    conda-forge
netcdf4                   1.5.1.2          py36had58050_0    conda-forge
numpy                     1.13.3           py36hdbf6ddf_4  
olefile                   0.46                       py_0    conda-forge
openblas                  0.2.20                        7    conda-forge
openjpeg                  2.3.1                h58a6597_0    conda-forge
openssl                   1.1.1b               h14c3975_1    conda-forge
owslib                    0.17.1                     py_0    conda-forge
pandas                    0.23.4           py36hf8a1672_0    conda-forge
parso                     0.1.1                      py_0    conda-forge
pcre                      8.41                          1    conda-forge
pexpect                   4.4.0                    py36_0    conda-forge
pickleshare               0.7.4                    py36_0    conda-forge
pillow                    5.2.0            py36hc736899_1    conda-forge
pip                       9.0.1                    py36_1    conda-forge
pixman                    0.34.0            h14c3975_1003    conda-forge
poppler                   0.67.0               h4d7e492_3    conda-forge
poppler-data              0.4.9                         1    conda-forge
postgresql                11.2                 h61314c7_0    conda-forge
proj4                     5.2.0             h14c3975_1001    conda-forge
prompt_toolkit            1.0.15                   py36_0    conda-forge
ptyprocess                0.5.2                    py36_0    conda-forge
pycparser                 2.19                     py36_1    conda-forge
pyepsg                    0.4.0                      py_0    conda-forge
pygments                  2.2.0                    py36_0    conda-forge
pykdtree                  1.3.1           py36h3010b51_1002    conda-forge
pyopenssl                 19.0.0                   py36_0    conda-forge
pyparsing                 2.2.0                    py36_0    conda-forge
pyproj                    1.9.5.1                  py36_0    conda-forge
pyqt                      4.11.4                   py36_3    conda-forge
pyshp                     1.2.12                     py_0    conda-forge
pysocks                   1.6.8                 py36_1002    conda-forge
python                    3.6.7             h381d211_1004    conda-forge
python-dateutil           2.7.2                      py_0    conda-forge
pytz                      2018.3                     py_0    conda-forge
qt                        4.8.7                         2  
readline                  7.0               hf8c457e_1001    conda-forge
requests                  2.21.0                py36_1000    conda-forge
scipy                     1.1.0            py36hfc37229_0  
setuptools                39.0.1                   py36_0    conda-forge
shapely                   1.6.4           py36h2afed24_1004    conda-forge
simplegeneric             0.8.1                    py36_0    conda-forge
sip                       4.18                     py36_1    conda-forge
six                       1.11.0                   py36_1    conda-forge
sqlite                    3.26.0            h67949de_1001    conda-forge
tk                        8.6.9             h84994c4_1001    conda-forge
tornado                   5.0.1                    py36_1    conda-forge
traitlets                 4.3.2                    py36_0    conda-forge
tzcode                    2018g             h14c3975_1001    conda-forge
urllib3                   1.24.2                   py36_0    conda-forge
wcwidth                   0.1.7                    py36_0    conda-forge
wheel                     0.30.0                   py36_2    conda-forge
xarray                    0.11.0                py36_1000    conda-forge
xerces-c                  3.2.2             hac72e42_1001    conda-forge
xorg-kbproto              1.0.7             h14c3975_1002    conda-forge
xorg-libice               1.0.9             h516909a_1004    conda-forge
xorg-libsm                1.2.3             h84519dc_1000    conda-forge
xorg-libx11               1.6.7             h14c3975_1000    conda-forge
xorg-libxau               1.0.8                         3    conda-forge
xorg-libxdmcp             1.1.2                         3    conda-forge
xorg-libxext              1.3.4                h516909a_0    conda-forge
xorg-libxrender           0.9.10            h516909a_1002    conda-forge
xorg-renderproto          0.11.1            h14c3975_1002    conda-forge
xorg-xextproto            7.3.0             h14c3975_1002    conda-forge
xorg-xproto               7.0.31            h14c3975_1007    conda-forge
xz                        5.2.4                h470a237_1    conda-forge
zlib                      1.2.11                        0    conda-forge
```
