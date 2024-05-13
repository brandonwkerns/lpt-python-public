# lpt-python-public
Updated: May 2024

Python version of Large-Scale Precipitation Tracking (LPT): Public Release.
This version of LPT is to be released with Kerns and Chen (2019), submitted to the Journal of Climate.


## MASTER_RUN directory.
__Please do not modify the files in the MASTER_RUN directory unless you are doing code development.__
They are Github repository files and can be updated with `git pull`. But `git pull` will cause
issues if the files have been previously updated locally!
Instead, copy the directory to a local working directory
 which you can feel free to modify to run your case.


## Python dependencies

Python module dependencies are documented in the environment.yml file.

To use it to create an Anaconda Python virtual environment:
```conda env create -f environment.yml -p ./env```
Then, to activate the environment:
```conda activate ./env```


## Code organization:
- The main Python functions directory is **lpt/**.
  * Functions for reading data are in **lpt/readdata.py**. (See the options for the **dataset** dictionary below.) Currently there are functions for:
    + Generic NetCDF data like (lat, lon, precip) or (time, lat, lon, precip). The variable names can be customized.
    + CMORPH
    + IMERG Version 6 (HDF5)
    + CFSR and CFS Forecast (Grib2)
  * Functions for LP object and LPTs input/output are in **lpt/lptio.py**.
  * Supporting functions for calculations are in **lpt/helpers.py**.
  * Example plotting functions are in **lpt/plotting.py**
- The following LPT driver scripts is included in **MASTER_RUN/**:
  * **lpt_run.py**
- The following directories are included by default under the **MASTER_RUN/** directory
    When you copy the directory, these subdirectories are used 
    to write output files, by default.
  * **data/**                (Digital output data. Organized in sub directories.)
  * **images/**              (Images produced by the scripts. Organized in sub directories.)


## Setting up LPT on your system:

### Getting the code

1) Clone this repository to your system, or download the zip file.
2) Copy the MASTER_RUN directory to a new run directory. E.g. `cp -r MASTER_RUN RUN`

### Creating a Python environment

I used Anaconda Python, with an environment set up using the provided environment.yml file:

```
conda env create -f environment.yml -p ./env
```

To activate the environment, you can use
```
conda activate ./env
```


## Running the code

### Overview

There are three main steps for running the LPT code:
1) Get in to the "RUN" directory.
2) Edit the lpt_run.py script as needed (see comments in the file and below).
3) Run the lpt_run.py script: `python lpt_run.py YYYYMMDDHH YYYYMMDDHH`
   (Specify start and end times of the tracking)
4) By default, output will be written under the **data/** directory.

### Default and Custom Settings

Options are defined as Python dictionaries. There are X dictionaries named:
- dataset
- plotting
- output
- lpo_options
- lpt_options
- merge_split_options
- mjo_id_options

The option values are set directly by Python scripts. After the values are set, the main function **lpt_driver.py** is called.
- Default values for all the options are set in **lpt/default_options.py**.
- Many of these options are over-ridden in your **[RUN]/lpt_run.py**.

For a brief overview/reminder of what the settings are,
see the comments in **lpt/default_options.py**, **MASTER_RUN/lpt_run.py** and **[RUN]/lpt_run.py**.
For more details, see below.

#### Dataset options

Table 1. **dataset** dictionary options.
| Option | Data Type | Description |
| - | - | - |
| dataset['label'] | string | Used in the output file names. For example, "imerg" in **lpt_systems_imerg_2023111000_2024020823.nc** |
| dataset['raw_data_parent_dir'] | string | Parent directory, which is in common with all the files, for the input data. It can be a relative path. Subdirectories, such as by date, can be set using the file_name_format option. |
| dataset['raw_data_format'] | string | Controls which **lpt/readdata.py** function gets used to read in the raw data. The value must match a valid data format in the if/elif/else block at the top of **readdata.py**. See the current list of current options in Table 2 below. |
| dataset['file_name_format'] | string | The path for filenames under the raw_data_parent_dir. This is a Python format string such as would be used with datetime.strftime(). For example, for 00 UTC 2024-01-10, "%Y/%m/gridded_rain_rates_%Y%m%d%H.nc" would get converted in to "2024/01/gridded_rain_rates_2024011000.nc". |
| dataset['data_time_interval'] | integer | The time between input files. **Units: Hours**. |
| dataset['verbose'] | True or False | Whether to print more detailed information about the files to the screen. |
| dataset['longitude_variable_name'] | string | Longitude variable name for generic_netcdf. *NOTE: The readdata.py functions are set up to convert -180 to 180 longitude to 0 - 360.* |
| dataset['latitude_variable_name'] | string | Latitude variable name for generic_netcdf |
| dataset['time_variable_name'] | string | Time variable name for generic_netcdf. Ignored if there is no time dimension. |
| dataset['field_variable_name'] | string | Name of the variable to use for feature identification, a Python string (e.g., "rainfall" for LPT). |
| dataset['field_units'] | string | Units of data. This is mainly used for generating plots, not for calculations. It is OK to set it to "" if plots are not being created. |
| dataset['area'] | list of Floats | Geographical area of data to use. A Python list of float values for [lon_begin, lon_end, lat_begin, lat_end], e.g., [0.0, 360.0, -50.0, 50.0]. The input data will be subsetted to this region. *NOTE: The readdata.py functions are set up to convert -180 to 180 longitude to 0 - 360.* |

Table 2. Raw data format options.
| Raw data option value | Description |
| - | - |
| generic_netcdf | NetCDF data. The intended variable must have dimensions (lat, lon) or (time, lat, lon), or similar variables. <br>The specific variable names are set by the dataset dictionary options named like "*_variable_name". NOTE: These options are ignored for the other raw data formats. |
| cmorph | CMORPH data in binary format. NOTE: For NetCDF format data, you can use generic_netcdf instead. |
| imerg_hdf5 | IMERG V6 data in HDF5 format. |
| cfs_forecast | CFS Forecast data in Grib2 format. |

#### Plotting options

Table 3. **plotting** dictionary options.
| Option | Data Type | Description |
| - | - | - |
| plotting['do_plotting'] | True or False | Whether to generate plots. This applies only to the LPO and LPT steps, e.g., lpo_options['do_lpo_calc'] (Map plots of rainfall and LPO) and lpt_options['do_lpt_calc'] (Time-longitude plot). The other plotting options are ignored if this is set to False. <br>NOTE: This is best used as a "gut check" for a short time period to determine whether the code is doing what you expect. If you are running for a long period, this will consume resources, so maybe set it to False for your "production" runs. |
| plotting['plot_area'] | list of Floats | Geographical area of data for map plots. A Python list of float values for [lon_begin, lon_end, lat_begin, lat_end], e.g., [0.0, 360.0, -50.0, 50.0] |
| plotting['time_lon_range'] | list of Floats | Longitude range for time-longitude plots. Does not need to be the same as A Python list of float values for [lon_begin, lon_end], e.g., [40.0, 200.0] |

#### Output options

The output path has several components, depending on the dataset label, accumulation/averaging period, spatial filtering, and threshold value.

The convention for LPO data output, expressed as a Python formatted string, is like this:
```python
fout = (f"{output['data_dir']}"
    + f"/{dataset['label']}"
    + f"/g{lpo_options['filter_stdev']}"
    + f"_{lpo_options['accumulation_hours']}h"
    + f"/thresh{lpo_options['thresh']}"
    + "/objects/"
    + dt_this.strftime(output['sub_directory_format'])
    + "/" + dt_this.strftime('objects_%Y%m%d%H.nc')
```

for example: **./data/imerg/g50_72h/thresh12/objects/2024/01/20240110/objects_2024011000.nc**.
- For images, replace "data" with "images" and ".nc" with ".png".
- For systems, replace "objects" with "systems" and no date-based sub directory (ignore output['sub_directory_format'])


Table 4. **output** dictionary options.

| Option | Data Type | Description |
| - | - | - |
| output['img_dir'] | string | directory for plotting outputs. Can be a relative path. |
| output['data_dir'] | string | directory for data outputs (text/NetCDF). Can be a relative path. |
| output['sub_directory_format'] | string | The subdirectory beneath the img_dir or data_dir. This is a Python format string such as would be used with datetime.strftime(). For example, for 00 UTC 2024-01-10, '%Y/%m/%Y%m%d' is converted in to '2024/01/20240110'. This pertains to LPO output data and LPO map plots. |




