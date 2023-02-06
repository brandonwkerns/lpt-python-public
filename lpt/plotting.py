import matplotlib; matplotlib.use('agg')
import numpy as np
from context import lpt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import sys
import os
import os.path
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
from netCDF4 import Dataset

"""
This module file contains function for making LPT related plots.
Most of the plotting function take a Matplotlib axis object as input
So create Matlplotlib figures and axes before calling the functions.
"""






def rain_cmap_12():

    cmap = plt.cm.jet
    N_keep = 12
    cmap2 = colors.LinearSegmentedColormap.from_list(name='custom', colors=cmap(range(cmap.N)), N=N_keep)
    color_list1 = cmap2(range(cmap2.N))
    ## Modify individual colors here.
    color_list11 = cmap2(range(cmap2.N))
    color_list11[0] = [1,1,1,1] # Make first color white.
    color_list11[1] = [0.7,0.7,0.7, 1] # Make first color gray.
    color_list11[2] = [0.0, 0.9, 1.0, 1.0]
    color_list11[3] = [0.0/255.0, 156.0/255.0, 255.0/255.0, 1.0]
    color_list11[4] = [0.0/255.0, 58.0/255.0, 255.0/255.0, 1.0]
    color_list11[5] = [0.3, 1.0, 0.3, 1.]
    color_list11[6] = [0.0, 0.8, 0.0, 1.]
    color_list11[9] = color_list1[10]  # Move top to colors of jet color map down.
    color_list11[10] = color_list1[11] # Move top to colors of jet color map down.
    color_list11[11] = [1,0,1,1] # Magenta on top.

    color_list11[:,-1] = np.linspace(0.25, 1, N_keep)

    cmap = colors.LinearSegmentedColormap.from_list(name='custom', colors=color_list11, N=N_keep)

    return cmap


def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return colors.LinearSegmentedColormap('colormap',cdict,1024)


def plot_map_background(plotArea=[0,360,-60,60], lon_labels = [1,0,0,0], lat_labels = [0,0,0,1], res='c', anchor='C'
                        , coast_color = 'k', fontsize=10):
    map = Basemap(projection='cyl',resolution=res, anchor=anchor,
                  llcrnrlat = plotArea[2],
                  urcrnrlat = plotArea[3],
                  llcrnrlon = plotArea[0],
                  urcrnrlon = plotArea[1])

    #map.fillcontinents(color='lightgray', lake_color='lightgray')
    map.drawcoastlines(linewidth=0.5, color=coast_color)
    map.drawparallels(np.arange(-80,81,20), linewidth=0.5, labels = lon_labels, fontsize = fontsize)
    if plotArea[1] - plotArea[0] < 180.1:
        map.drawmeridians(np.arange(0,361,20), linewidth=0.5, labels = lat_labels, fontsize = fontsize)
    else:
        map.drawmeridians(np.arange(0,361,40), linewidth=0.5, labels = lat_labels, fontsize = fontsize)

    return map


def print_and_save(file_out_base):
    os.makedirs(os.path.dirname(file_out_base), exist_ok=True) # Make directory if needed.
    print(file_out_base + '.png')
    plt.savefig(file_out_base + '.png' ,bbox_inches='tight', dpi=150)


"""
####################################################################
## High level plotting functions follow below.
####################################################################
"""

def plot_rain_map_with_filtered_contour(ax, DATA_ACCUM, OBJ, plotting, lpo_options, label_font_size=10):

    plot_area = plotting['plot_area']

    lon = OBJ['grid']['lon']
    lat = OBJ['grid']['lat']

    map1 = plot_map_background(plot_area)
    cmap = cmap_map(lambda x: x/2 + 0.5, plt.cm.jet)
    cmap.set_under(color='white')
    H1 = map1.pcolormesh(lon, lat, DATA_ACCUM, cmap=cmap, vmin=plotting['vmin'], vmax=plotting['vmax'])

    label_im = np.array(OBJ['label_im'])
    label_im[label_im > 0.5] = 1
    Hobj = plt.contour(lon, lat, label_im, [0.5,], colors='k', linewidths=1.0)

    map1.plot(OBJ['lon'], OBJ['lat'], 'kx', markersize=7)
    cax = plt.gcf().add_axes([0.92, 0.2, 0.025, 0.6])
    CB = plt.colorbar(H1, cax=cax)
    CB.set_label(label='[{}]'.format(lpo_options['field_units']), fontsize=label_font_size)
    CB.ax.tick_params(labelsize=label_font_size)
    return (map1, H1, Hobj, CB)


def plot_timelon_with_lpt(ax2, dt_list, lon, timelon_rain, TIMECLUSTERS
        , lon_range, accum_time_hours = 0, label_font_size=10, offset_time=True):

    ## With CFtime, plotting the y axis as cftime.datetime does not work.
    ## Therefore, use time stamps and manually set the y axis labels below.
    timestamp_list = [(x - dt_list[0]).total_seconds() for x in dt_list]

    
    cmap = plt.cm.jet
    N_keep = 12
    cmap2 = colors.LinearSegmentedColormap.from_list(name='custom', colors=cmap(range(cmap.N)), N=N_keep)
    color_list1 = cmap2(range(cmap2.N))
    ## Modify individual colors here.
    color_list11 = cmap2(range(cmap2.N))
    color_list11[0] = [1,1,1,1] # Make first color white.
    color_list11[1] = [0.7,0.7,0.7, 1] # Make first color gray.
    color_list11[2] = [0.0, 0.9, 1.0, 1.0]
    color_list11[3] = [0.0/255.0, 156.0/255.0, 255.0/255.0, 1.0]
    color_list11[4] = [0.0/255.0, 58.0/255.0, 255.0/255.0, 1.0]
    color_list11[5] = [0.3, 1.0, 0.3, 1.]
    color_list11[6] = [0.0, 0.8, 0.0, 1.]
    color_list11[9] = color_list1[10]  # Move top to colors of jet color map down.
    color_list11[10] = color_list1[11] # Move top to colors of jet color map down.
    color_list11[11] = [1,0,1,1] # Magenta on top.

    color_list11[:,-1] = np.linspace(0.25, 1, N_keep)

    cmap = colors.LinearSegmentedColormap.from_list(name='custom', colors=color_list11, N=N_keep)

    
    cmap.set_under(color='white')
    timelon_rain = np.array(timelon_rain)
    timelon_rain[timelon_rain < 0.1] = np.nan
    Hrain = ax2.pcolormesh(lon, timestamp_list, timelon_rain, vmin=0.2, vmax=1.4, cmap=cmap)
    cax = plt.gcf().add_axes([0.95, 0.2, 0.025, 0.6])
    cbar = plt.colorbar(Hrain, cax=cax)
    cbar.set_label(label='Rain Rate [mm/h]', fontsize=label_font_size)
    cbar.ax.tick_params(labelsize=label_font_size)

    lpt_group_colors = plt.cm.Set2(range(plt.cm.Set2.N))

    for ii in range(len(TIMECLUSTERS)):
        x = TIMECLUSTERS[ii]['centroid_lon']
        lat = TIMECLUSTERS[ii]['centroid_lat']
        if offset_time:
            y = [((yy - dt.timedelta(hours=0.5*accum_time_hours)) - dt_list[0]).total_seconds() for yy in TIMECLUSTERS[ii]['datetime']]
        else:
            y = (TIMECLUSTERS[ii]['datetime'] - dt_list[0]).total_seconds()

        this_color_idx = int(np.floor(TIMECLUSTERS[ii]['lpt_id'])) % len(lpt_group_colors[:,0])
        this_color = lpt_group_colors[this_color_idx,:]

        ax2.plot(x, y, '-', color='k', linewidth=4.0)
        ax2.plot(x, y, '--', color=this_color, linewidth=2.0)
        x2 = 1.0*x
        y2 = y.copy()
        x2[abs(lat) > 15.0] = np.nan
        ax2.plot(x2, y2, '-', color=this_color, linewidth=2.0)

        ax2.text(x[0], y[0], str(TIMECLUSTERS[ii]['lpt_id']), fontweight='bold', color='k',clip_on=True, fontsize=14, ha='center', va='top')
        ax2.text(x[-1], y[-1], str(TIMECLUSTERS[ii]['lpt_id']), fontweight='bold', color='k',clip_on=True, fontsize=14, ha='center', va='bottom')

    ax2.set_xlim(lon_range)
    yticks = [x*86400 for x in range(0, (dt_list[-1] - dt_list[0]).days + 1 ,7)]
    yticklabels = [(dt_list[0] + dt.timedelta(seconds=x)).strftime('%m/%d') for x in yticks]
    ax2.set_ylim([timestamp_list[0], timestamp_list[-1]])
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticklabels)
    ax2.grid(linestyle='--', linewidth=0.5, color='k')
    ax2.set_xlabel('Longitude', fontsize=label_font_size)
    plt.xticks(fontsize=label_font_size)
    plt.yticks(fontsize=label_font_size)

    return (Hrain)

################################################################################
###### Functions used for post processing scripts. #############################
################################################################################

def manage_time_lon(year1, parent_dir='.'):

    fn_timelon = (parent_dir + '/time_lon_rain_15S_15N__' + str(year1) + '_' + str(year1+1) + '.nc')
    DATA={}

    if os.path.isfile(fn_timelon):
        print('Reading: ', fn_timelon)
        DS = Dataset(fn_timelon, 'r')
        DATA['lon'] = DS['lon'][:]
        DATA['time'] = [dt.datetime(1970,1,1,0,0,0) + dt.timedelta(hours=int(x)) for x in DS['time'][:]]
        DATA['time3'] = [dt.datetime(1970,1,1,0,0,0) + dt.timedelta(hours=int(x)) for x in DS['time3'][:]]
        DATA['precip'] = DS['precip'][:]
        DATA['precip3'] = DS['precip3'][:]
        DS.close()

    else:

        begin_time = dt.datetime(year1,6,1,0,0,0)
        end_time = dt.datetime(year1+1,6,1,0,0,0)

        hours_list = np.arange(0.0, 0.1 +(end_time-begin_time).total_seconds()/3600.0, interval_hours)
        dt_list = [begin_time + dt.timedelta(hours=x) for x in hours_list]

        timelon_rain = []
        for this_dt in dt_list:
            DATA_RAW = read_function(this_dt, verbose=True)

            lat_idx, = np.where(np.logical_and(DATA_RAW['lat'] > -15.0, DATA_RAW['lat'] < 15.0))
            timelon_rain.append(np.mean(np.array(DATA_RAW['precip'][lat_idx,:]), axis=0))

        lon = DATA_RAW['lon']
        timelon_rain = np.array(timelon_rain)
        dt_list3 = dt_list[0::int(accumulation_hours/interval_hours)]
        timelon_rain3 = []
        for tt in range(0,len(dt_list),int(accumulation_hours/interval_hours)):
            timelon_rain3.append(np.nanmean(timelon_rain[tt:tt+int(accumulation_hours/interval_hours),:], axis=0))
        timelon_rain3 = np.array(timelon_rain3)

        print('Writing: ', fn_timelon)
        os.mkdirs(parent_dir, exist_ok=True)
        DS = Dataset(fn_timelon, 'w')

        DS.createDimension('lon', len(DATA_RAW['lon']))
        DS.createDimension('time', len(dt_list))
        DS.createDimension('time3', len(dt_list3))

        DS.createVariable('lon','d',('lon',))
        DS.createVariable('time','d',('time',))
        DS.createVariable('time3','d',('time3',))
        DS.createVariable('precip','d',('time','lon',))
        DS.createVariable('precip3','d',('time3','lon',))

        DS['lon'][:] = DATA_RAW['lon']
        DS['time'][:] = [(x - dt.datetime(1970,1,1,0,0,0)).total_seconds()/3600.0 for x in dt_list]
        DS['time3'][:] = [(x - dt.datetime(1970,1,1,0,0,0)).total_seconds()/3600.0 for x in dt_list3]
        DS['precip'][:] = timelon_rain
        DS['precip3'][:] = timelon_rain3

        DATA['lon'] = DATA_RAW['lon']
        DATA['time'] = [dt.datetime(1970,1,1,0,0,0) + dt.timedelta(hours=int(x)) for x in dt_list]
        DATA['time3'] = [dt.datetime(1970,1,1,0,0,0) + dt.timedelta(hours=int(x)) for x in dt_list3]
        DATA['precip'] = timelon_rain
        DATA['precip3'] = timelon_rain3

        DS.close()

    ## Data return
    return DATA


def print_and_save(file_out_base, pdf=False):
    if pdf:
        print(file_out_base + '.pdf')
        plt.savefig(file_out_base + '.pdf' ,bbox_inches='tight', dpi=150)
    print(file_out_base + '.png')
    plt.savefig(file_out_base + '.png' ,bbox_inches='tight', dpi=150)
