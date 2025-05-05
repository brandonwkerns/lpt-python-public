# lpt-python-public
Updated: May 2025

Python version of Large-Scale Precipitation Tracking (LPT): Public Release.
This version of LPT is to be released with Kerns and Chen (2019), submitted to the Journal of Climate.

This README provides an overview of how to obtain, set up, and run the code. For specifics, consult the Wiki files. The Wiki files are also linked in each section.
- Settings
- Output


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
2) Edit the **lpt_run.py** script as needed. By default, all of the calculation steps are set to False. You need to turn on the ones you want to run. (See below, and comments in **lpt_run.py**).
3) Run **lpt_run.py**: `python lpt_run.py YYYYMMDDHH YYYYMMDDHH` (Specify start and end times of the tracking)
4) By default, output will be written under the **data/** directory.

### Suggested Workflow

+++ These suggestions assume using LPT for MJO +++

A great way to run the code is to do it in four stages. In each step, you edit **lpt_run.py** and run it like `python lpt_run.py YYYYMMDDHH YYYYMMDDHH`.
1. LPO identification: Set lpo_options['do_lpo_calc'] = True (and all other calc steps False). First run it with lpo_options['do_lpo_calc'] = True for a few days when you know there was an MJO event. The quick plots will give you a chance to tune the LPO parameters. Then turn plotting off and run for a longer period, e.g., at least two weeks to get a system long enough to be identified as an MJO LPT.
2. Connecting LPOs as LPTs: set lpt_options['do_lpt_calc'] = True (and all other calc steps False). Run for the entire 2 week period that you used with LPOs in Step 1, with plotting set to True. Check the time-longitude plot. Adjust the lpt_options dictionary options if needed. Verify visually that the LPT system propagated eastward. NOTE: There will likely be other LPT systems tracked that are not MJO.
3. Identifying which LPT systems are MJO LPTs. Set mjo_id_options['do_mjo_id'] = True (other calc steps set to false). To see the details about how the LPT system is being identified as an MJO LPT, also set mjo_id_options['do_plotting'] = True to get plots of eastward and westward propagation periods and how they are combined into an overall eastward (or westward) system motion.
4. Calculate spatio-temporal masks for the two week period. This is done by setting one or more of the "masks" options to True. Preview the NetCDF files with `ncview` and verify that it is doing what you expect.

Once you are satisfied with the output from the above steps, you can run for longer time periods, e.g., 1 year or if you're ambitious and have the computational resources even 10+ years on a relatively coarse dataset.

### Default and Custom Settings

Options are defined as Python dictionaries. There are 7 dictionaries each with options that can be set:
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

For more details about each setting, see the [Settings wiki](https://github.com/brandonwkerns/lpt-python-public/wiki/Settings).


## Output files

LPT digital data output is in text (.txt) and NetCDF (.nc) format.

By default, output data is produced in the **./data** directory. The default naming conventions of the output files is summarized here.


### LPO output files

Example Directory: **./data/imerg/g50_72h/thresh12/objects/2024/01/20240110/**
| Example file name | Description |
| - | - |
| **objects_2024011012** | Text summary of LPOs. Includes: centroid (lat, lon), centroid grid point (x, y), area, and LPO ID (YYYYMMDDHHnnnn where nnnn starts at 0000). |
| **objects_2024011012.nc** | NetCDF file with detailed LPO data. Includes all the information from the text files, plus the grid point information for each LPO.


### LPT system files
These files provide the "bulk" information (e.g., centroid track, area, maximum rainfall) of the LPTs through their life cycles.

Example Directory: **./data/imerg/g50_72h/thresh12/systems/**
| Example file name | Description |
| - | - |
| **lpt_systems_imerg_2023111000_2024020823.txt** | Text summary of LPT systems. Includes: centroid (lat, lon), area, and number of LPOs for each time stamp. |
| **lpt_systems_imerg_2023111000_2024020823.nc** | NetCDF file with detailed LPT data. Includes all the information from the text files, plus various "bulk" details about the LPTs. Variables that apply to the LPTs as a whole use the "nlpt" dimension. Variables that vary through the life cycle are stitched together and use the "nstitch" dimension. The individual systems are stitched together with NaN values inbetween them. <i>Use the **lptid_stitched** variable to separate out individual LPTs.</i> |
| **mjo_lpt_list_imerg_2023111000_2024020823.txt** | Column delimited text file documenting which LPT systems were identified as MJO LPTs. Can be read in using Pandas read_csv. It includes propagation direction and speed for the LPTs as a whole, and for the eastward propagation period(s) which got them classified as an MJO LPT system (eprop variables). |
| **mjo_group_extra_lpt_list_imerg_2023111000_2024020823.txt** | Same as MJO LPT list, but for systems that overlap with MJO LPTs, but were not chosen as MJO LPTs themselves. Variables specific to MJO LPTs are set to -999, 9999999999, and similar. |
| **non_mjo_lpt_list_imerg_2023111000_2024020823.txt** | Similar to MJO LPT list, but for systems that were not MJO LPTs. Variables specific to MJO LPTs are set to -999, 9999999999, and similar. |


### Spatio-temporal mask files

These files document the spatial extent of the LPOs and LPTs over time, as well as the rainfall (or other chosen tracking variable) associated with them. There are separate mask files for:
- Aggregate of all LPOs
- Aggregate of all LPTs
- Individual LPTs

#### Aggregate spatio-temporal mask files

In general, mask variables start with "mask" and have dimensions (time, lat, lon). If the corresponding "detailed_output" setting is True, 4 mask variables plus corresponding mask with rainfall variables are created. Otherwise, and by default, a single "mask" variable and "mask_with_rain" is produced.

Directory for LPOs: **./data/imerg/g50_72h/thresh12/objects**
| Example file name | Description |
| - | - |
| **lp_objects_mask_2023111000_2024020823.nc** | Aggregate spatio-temporal mask for LPOs. |

Directory for LPTs: **./data/imerg/g50_72h/thresh12/systems**
| Example file name | Description |
| - | - |
| **lpt_composite_mask_2023111000_2024020823.nc** | Aggregate spatio-temporal mask for all LPT systems. |
| **lpt_composite_mask_2023111000_2024020823_mjo_lpt.nc** | Same as above, but only including the systems identified as MJO LPTs. |
| **lpt_composite_mask_2023111000_2024020823_mjo_lpt.nc** | Same as above, but only including the systems *not* identified as MJO LPTs. |


#### Individual LPT system mask files

Example Directory: **./data/imerg/g50_72h/thresh12/systems/2023111000_2024020823**
The tracking period is denoted as YYYYMMDDHH_YYYYMMDDHH.

This applies for idividual LPTs and groups of overlapping LPTs.

| Example file name | Description |
| - | - |
| **lpt_system_mask_imerg.lptid00000.1000.nc** | LPT ID is denoted by NNNNN.nnnn. |
| **lpt_system_mask_imerg.lptid00000.group.nc** | LPT group is denoted by NNNNN.group. |


### More details

For more details about the various output files and variables within them, see the [Output wiki](https://github.com/brandonwkerns/lpt-python-public/wiki/Output).

