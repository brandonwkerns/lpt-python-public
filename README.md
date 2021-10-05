# lpt-python-public
Python version of Large-Scale Precipitation Tracking (LPT): Public Release.
This version of LPT is to be released with Kerns and Chen (2019), submitted to the Journal of Climate.

This version is set up to run with "generic" NetCDF files with the following format:

```
##
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
##
## Note: The names of the coordinate variables "lon" and "lat"
##       and data variable "rain" can be set in the lpt_driver.py script.
##
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

Python module dependencies are documented in the environment.yml file.

To use it to create an Anaconda Python virtual environment:
```conda env create -f environment.yml -p ./env```
Then, to activate the environment:
```conda activate ./env```


## Code organization:
- The main Python functions directory is lpt/.
  * Functions for reading data are in lpt/readdata.py.
  * Functions for LP object and LPTs input/output are in lpt/lptio.py.
  * Supporting functions for calculations are in lpt/helpers.py.
  * Example plotting functions are in lpt/plotting.py
- The following generic NetCDF data driver scripts is included in MASTER_RUN/:
  * lpt_run.py
- The following example preprocessor scripts are included in MASTER_RUN/:
  * preprocess_tmpa.py
  * preprocess_wrf.py
  * preprocess_global_4km_ir.py
- The following directories are included by default under the MASTER_RUN/ directory
    (these are used by default in the driver script)
  * data/                (Digital output data. Organized in sub directories.)
  * images/              (Images produced by the scripts. Organized in sub directories.)

## Setting up LPT on a new system:
1) Clone this repository to your system, or download the zip file format.
2) Copy the MASTER_RUN directory to a new run directory.
3) Edit the lpt_run.py script as needed.
4) Run the lpt_run.py script.
     Usage: `python lpt_run.py YYYYMMDDHH YYYYMMDDHH`
     (Specify start and end times of the tracking)



## Python environment

I used Anaconda Python, with an environment set up the following way:

```
conda create --name lpt numpy pandas matplotlib basemap netCDF4 h5py scipy wrf-python gdal
source activate lpt
```
