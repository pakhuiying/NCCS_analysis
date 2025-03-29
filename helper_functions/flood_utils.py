import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import os
import datetime
from osgeo import gdal, ogr
import requests

def get_weather_stns():
    """ returns weather stations in singapore
    Returns:
        dict: where key is the station name and the value is a dict with the station attributes
    """
    weather_stns = pd.read_csv(r"C:\Users\hypak\OneDrive - Singapore Management University\Documents\Data\SG_Climate_data\Weather_stations_drainageMap.csv")
    weather_stns_dict = weather_stns.to_dict('records')
    weather_stns_dict = {d['station_name'].lower().replace(' ',''):
                        {'station_code':d['station_code'],
                        'station_name':d['station_name'],
                        'longitude':d['longitude'],
                        'latitude':d['latitude'],
                        'drainage_map':d['drainage_map']
                        } for d in weather_stns_dict}
    return weather_stns_dict

def get_closest_weather_stn(weather_stns_dict,stn_name):
    """ 
    returns the closest weather station in ascending distance, where the first element is the stn_name itself
    """
    dist_diff = lambda lat1,lon1,lat2,lon2: ((lat2-lat1)**2 + (lon2-lon1)**2)**(1/2)
    stn_name = stn_name.lower().replace(' ','')
    lat1, lon1 = weather_stns_dict[stn_name]['latitude'], weather_stns_dict[stn_name]['longitude']
    diff_dist = [(stn_n,dist_diff(weather_stns_dict[stn_n]['latitude'],weather_stns_dict[stn_n]['longitude'],lat1,lon1)) for stn_n in weather_stns_dict.keys()]
    sorted_dist = list(sorted(diff_dist,key=lambda tup: tup[1]))
    return sorted_dist

def grid_code_drainage_catchment():
    return {1: 'Jurong', 2: 'Kranji', 3: 'Pandan', 4: 'Woodlands', 5: 'Kallang', 6: 'Bukit Timah'
            , 7: 'Stamford Marina', 8: 'Singapore River', 9: 'Punggol', 10: 'Geylang', 11: 'Changi'}

def get_coordinates_from_location(location):
    """returns number of results found, search value, and coordinates given a supplied location 
    Args:
        location (str): a location in singapore
    Returns:
        tuple: strings corresponding to number of results found, search value, lat, and lon
    """
    
    headers = {"Authorization": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZWY3ZDlhYTRkNDIyMWVjYjA2NzE2MTg0Yjc3MmU5ZCIsImlzcyI6Imh0dHA6Ly9pbnRlcm5hbC1hbGItb20tcHJkZXppdC1pdC0xMjIzNjk4OTkyLmFwLXNvdXRoZWFzdC0xLmVsYi5hbWF6b25hd3MuY29tL2FwaS92Mi91c2VyL3Bhc3N3b3JkIiwiaWF0IjoxNzI1Njg0NDE4LCJleHAiOjE3MjU5NDM2MTgsIm5iZiI6MTcyNTY4NDQxOCwianRpIjoiMnZuS0FHQ3FkdzVwRHpFRiIsInVzZXJfaWQiOjQ1NDcsImZvcmV2ZXIiOmZhbHNlfQ.pl1b-XkwgvBjdp-gczsdx17OoSLlGvrsAjfgUTeqY7M"}
    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={location}&returnGeom=Y&getAddrDetails=Y"
        
    response = requests.request("GET", url, headers=headers)
    response = response.json()
    response_found = response['found'] # number of results found
    response_first_result = response['results'][0] # get first item in the list
    searchVal, latitude, longitude = response_first_result['SEARCHVAL'], response_first_result['LATITUDE'], response_first_result['LONGITUDE']

    return response_found, searchVal, latitude, longitude

def get_closest_weather_stn(lat,lon):
    """ returns the closest weather stn given coordinates
    Args:
        lat (float): latitude
        lon (float): longitude
    Returns:
        str: the closest weather station name
    """
    dist_diff = lambda lat1,lon1,lat2,lon2: ((lat2-lat1)**2 + (lon2-lon1)**2)**(1/2)
    # import weather_stns
    weather_stns = get_weather_stns()
    weather_idx = np.argmin([dist_diff(d['latitude'],d['longitude'],lat,lon) for i,d in weather_stns.items()]) # get the smallest distance aka closest
    # print(weather_idx,type(weather_idx))
    return [i['station_name'] for i in weather_stns.values()][weather_idx]

def get_historical_floods_dict(historical_floods_df):
    """ returns a dict with each datetime-specific entry representing the location of each flood events
    Args:
        historical_floods_df (pd.DataFrame): a dataframe of all the flood occurrences with headers 'Date' and 'Location_Road'
    Returns:
        dict: dictionary where keys are unique dates (Date format), and values are a list of attributes
    """
    df = historical_floods_df.copy()
    df['Date'] = pd.to_datetime(historical_floods_df['Date'], format='%d-%b-%y')
    historical_floods_list = df[['Date','Location_Road']].to_dict('records')
    
    historical_floods_dict = dict()
    for item in historical_floods_list:
        date = item['Date']
        locations = item['Location_Road'].split(',') # split dictinct locations by ','
        locations = [i.strip() for i in locations] # remove leading and trailing empty spaces
        searched_location = {l: get_coordinates_from_location(l) for l in locations}
        searched_location = [{'flooded_location':l,
                              'responses_found':i[0], 
                              'matched_location':i[1], 
                              'latitude':i[2], 
                              'longitude':i[3],
                              'closest_weather_stn': get_closest_weather_stn(float(i[2]),float(i[3]))
                              } for l,i in searched_location.items()]
        if item not in list(historical_floods_dict):
            historical_floods_dict[date] = searched_location # intialise an empty list where items of dict can be added
    
    return historical_floods_dict

def sort_closest_weather_stn(lat,lon):
    """ returns the closest weather stn is ascending order of distance (closest to furthest)
    Args:
        lat (float): latitude
        lon (float): longitude
    Returns:
        list of str: the closest weather station name in descending order of distance
    """
    dist_diff = lambda lat1,lon1,lat2,lon2: ((lat2-lat1)**2 + (lon2-lon1)**2)**(1/2)
    # import weather_stns
    weather_stns = get_weather_stns()
    weather_stns_distance = [(d['station_name'],dist_diff(d['latitude'],d['longitude'],lat,lon)) for i,d in weather_stns.items()] # get the smallest distance aka closest
    weather_stns_sorted = sorted(weather_stns_distance,key = lambda x: x[1])
    # print(weather_idx,type(weather_idx))
    return [i[0] for i in weather_stns_sorted]

def extract_precipitation_values(weather_df, flood_dict):
    """ returns the amended historical_floods_dict with the precipitation values
    Args:
        weather_df (pd.DataFrame): dataframe with the historical data by stations and date
        flood_dict (dict): dictionary where keys are unique dates (Date format), and values are a list of attributes
    Returns: 
        dict: dictionary where keys are unique dates (Date format), and values are a list of attributes, including ppt values for each location
    """
    rainfall_columns = [c for c in weather_df.columns.to_list() if 'rainfall' in c]
    average_weather = weather_df.groupby(["Date"]).agg(mean_pr=('daily rainfall total (mm)','mean'),
                                                min_pr=('daily rainfall total (mm)','min'),
                                                max_pr=('daily rainfall total (mm)','max')
                                                ).reset_index()
    # flood_dict_copy = flood_dict.copy() # make a copy of the dictionary
    for dt, att_list in flood_dict.items():
        for d in att_list: # dict
            # print(dt)
            avg_weather = average_weather[average_weather['Date'] == dt]['mean_pr'].values[0]
            d['mean_pr'] = avg_weather
            stn = d['closest_weather_stn']
            # rainfall_list returns list of dict
            rainfall_list = weather_df[(weather_df['Date'] == dt) & (weather_df['station'] == stn)][rainfall_columns].to_dict('records')
            closest_weather_stn = sort_closest_weather_stn(float(d['latitude']),float(d['longitude']))
            stn_count = 1
            while len(rainfall_list) < 1 or np.isnan(list(rainfall_list[0].values())[0]):
                print(f'No data for {dt}, {stn}, {rainfall_list}')
                rainfall_list = weather_df[(weather_df['Date'] == dt) & (weather_df['station'] == closest_weather_stn[stn_count])][rainfall_columns].to_dict('records')
                # print(f'Try from {closest_weather_stn[stn_count]}...')
                print(f'Attempt {stn_count}: {stn} replaced with {closest_weather_stn[stn_count]}:, {rainfall_list}')
                stn_count += 1 # move to next closest weather station

            for rainfall_chr, val in rainfall_list[0].items():
                d[rainfall_chr] = val
        
    return flood_dict

def load_shapefile(fn):
    """ returns a shapefile
    Args:
        fn (str): filename of the shape file
    Returns:
         gdal shapefile layer (Vector)
    """
    with gdal.OpenEx(fn) as ds:
        lyr = ds.GetLayer(0)
        layer_definition = lyr.GetLayerDefn()
        field_names = [layer_definition.GetFieldDefn(i).GetName() for i in range(layer_definition.GetFieldCount())]
        print(f'Number of polygons: {len(lyr)}')
        print (f"Fields: {field_names}")
        # loop through each feature and print field values
        for feature in lyr:
            print(f'Feature ID: {feature.GetFID()}')
            for field_name in field_names:
                field_value = feature.GetField(field_name)
                print(f'{field_name}: {field_value}')                
    
    return lyr

def load_points(lon, lat):
    """ returns gdal geometry (point)
    Args:
        lon (float): longitude
        lat (float): latitiude
    Returns:
        gdal geometry (point)
    """
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(lon, lat)
    return point
# kranji_pt = load_points(103.709614,1.403377)

def check_polygon_contains_point(fn, point):
    """ returns True/False on whether polygon contains point, and the associated polygon attributes
    Args:
        fn (str): filepath to a multipolygon shapefile
        point (shapefile Geometry): point
    Returns:
        dict: whether if a polygon contains a point
    """
    drainage_catchment = grid_code_drainage_catchment()
    with gdal.OpenEx(fn) as ds:
        lyr = ds.GetLayer(0)
        for feature in lyr:
            geom = feature.GetGeometryRef()
            # check if point is within polygon
            if geom.Contains(point):
                
                field_value = feature.GetField('gridcode')
                matched_drainage = drainage_catchment[int(field_value)]
                print(f"drainage_catchment: {matched_drainage}")
                break
    return matched_drainage

def extract_drainage_catchment(drainage_catchment_fn,flood_dict):
    """ from the flood coordinates, extracts the corresponding drainage catchment from the shp file and add to dict
    Args:
        drainage_catchment_fn (str): filepath to a multipolygon shapefile
        flood_dict (dict): where keys represent datetime, values refer to a list of dict that contains the individual flood attributes
    Returns:
        dict: whether if a polygon contains a point
    """
    drainage_catchment = grid_code_drainage_catchment()
    with gdal.OpenEx(drainage_catchment_fn) as ds:
        lyr = ds.GetLayer(0)
        for dt , att_list in flood_dict.items():
            for d in att_list:
                lat = float(d['latitude'])
                lon = float(d['longitude'])
                point = load_points(lon, lat)
                # initialise with values
                found = False
                nearest_distance = np.inf
                nearest_drainage_catchment = None
                for feature in lyr:
                    geom = feature.GetGeometryRef()
                    field_value = feature.GetField('gridcode')
                    # check if point is within polygon
                    if geom.Contains(point):
                        matched_drainage = drainage_catchment[int(field_value)]
                        d['drainage_catchment'] = matched_drainage
                        found = True # update boolean flag to true if point is within polygon
                    # if point is not within the polygon, compute the distance of point to the polygon instead
                    else:
                        distance = point.Distance(geom)
                        if distance < nearest_distance:
                            nearest_distance = distance # update nearest distance by iterating through geom
                            nearest_drainage_catchment = drainage_catchment[int(field_value)]
                
                if not found: # if points are not found within polygon, assign drainage catchment as the nearest drainage catchment
                    d['drainage_catchment'] = nearest_drainage_catchment
                    
    return flood_dict

def dict_to_dataframe(flood_dict,save_dir = None):
    """ returns a df of flood events with their associated characteristics
    Args:
        flood_dict (dict):
        save_dir (str): directory of where to save the df
    Returns:
        pd.DataFrame: details of each flood events
    """
    list_of_dict = [] # to contain list of dict
    for dt, att_list in flood_dict.items():
        for d in att_list:
            d['time'] = dt
            list_of_dict.append(d)
    df = pd.DataFrame(list_of_dict)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        df.to_csv(os.path.join(save_dir,"precipitation_levels_during_flood_events.csv"),index=False)
    return df

def plot_grouped_boxplot(data,ax=None,group_colors=None,subgroup_colors=None,sym="+",
                         subgroup_hatches=[None,'.','o', 'O'],vert=True,cmap='plasma',show_fliers = True,figsize=None):
    """ 
    Args:
        data (dict): data is arranged by each distinct group
            {'A': {'1': [1,2,3], '2':[1,2,3,4], '3': [1,2,3]},
            'B': {'1': [5,2,3], '2':[5,2,3,4], '3': [5,2,3]},
            'C': {'1': [9,2,3], '2':[9,2,3,4], '3': [9,2,3]}
            }
            where groups A,B,C are the tick labels, and A1,A2,A3 are the sub boxplots within A
        ax (mpl.Axes): if no axis is supplied, a new figure is created. Else, artists is drawn on supplied ax
        group_colors (list): list of hex colours for groups e.g. A, B, C have different colors, and subgroups within A have same color
        subgroup_colors (list): list of hex colours e.g. sub groups within A have different colors
        subgroup_hatches (list): list of hatches symbols e.g. '/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'
        vert (bool): whether to plot the boxplot vertically (vert=True) or horizontally (vert=False)
        show_fliers (bool): if True, it will show the fliers (aka outliers). Else, fliers will be hidden. Default is True.
    """

    def set_box_color(bp, colors,hatch = None):
        """ 
        Args:
            bp (Patch): mpl Patch artists, which we can use it to set diff properties
            hatch (symbol): e.g. '/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'
                /   - diagonal hatching
                \   - back diagonal
                |   - vertical
                -   - horizontal
                +   - crossed
                x   - crossed diagonal
                o   - small circle
                O   - large circle
                .   - dots
                *   - stars
        """

        plt.setp(bp['medians'], color='black',linewidth=2)
        for box,color in zip(bp['boxes'],colors):
            # change outline color
            box.set_edgecolor('black')
            # set alpha
            box.set_alpha(0.5)
            # change fill color
            box.set_facecolor(color)
            # set linewidth
            box.set_linewidth(2)
            # change hatch
            if hatch is not None:
                box.set_hatch(hatch)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ticks = list(data) # names of the groups
    # number of observations in each group
    num_obs = {group: len(subgroups[list(subgroups)[0]]) for group,subgroups in data.items() }
    
    # colors = {group: cmap(norm(n)) for group,n in num_obs.items()}
    # hatch for subplots 
    # hatches = [None,'.','o', 'O']
    # prepare data for subplots
    subplot_data = {k:[data[t][k] for t in ticks] for k in list(data[ticks[0]])} # where keys are the subgroup names, and 
    # subplot names
    subplot_names = list(subplot_data)

    # plot each subgroup
    for i,(subgroups, subgroup_data) in enumerate(subplot_data.items()):
        if show_fliers is True:
            bp = ax.boxplot(subgroup_data, sym=sym,positions=np.array(range(len(subgroup_data)))*2.0+i*0.4, 
                            vert=vert,widths=0.3,patch_artist=True)
        else:
            bp = ax.boxplot(subgroup_data, sym=sym,positions=np.array(range(len(subgroup_data)))*2.0+i*0.4, 
                            vert=vert,widths=0.3,patch_artist=True)
        # set_box_color(bp, colors=list(colors.values()),hatch = hatches[i])
        if subgroup_colors is not None:
            # assert len(ticks) == len(group_colors),"length of colours != length of groups or ticks"
            set_box_color(bp, colors=[subgroup_colors[i]]*len(bp['boxes']),
                          hatch = subgroup_hatches[i])
        
        if group_colors is not None:
            # assert len(ticks) == len(group_colors),"length of colours != length of groups or ticks"
            set_box_color(bp, colors=group_colors,hatch = subgroup_hatches[i])

    if vert is False:
        # plot horizontal box plot
        ax.set_yticks(range(0, len(ticks) * 2, 2), [f'{group}\n(N = {obs})' for group,obs in num_obs.items()])
        ax.invert_yaxis() # labels read top-to-bottom
    else:
        # plot vertical box plot
        ax.set_xticks(range(0, len(ticks) * 2, 2), [f'{group}\n(N = {obs})' for group,obs in num_obs.items()])
    
    if ax is None:
        plt.legend()
        plt.tight_layout()
        plt.show()
    return ax

def plot_boxplots_floods_by_drainage(boxplot_data,show_fliers = True,save_dir=None):
    """ returns a grouped boxplot
    Args:
        boxplot_data (dict): {'A': {'1': [12,3,1,2,3], '2':[1,2,3,4], '3': [1,2,3]},
                    'B': {'1': [5,2,3], '2':[5,2,3,4], '3': [5,2,3]},
                    'C': {'1': [2,4,5,53,9,2,3], '2':[9,2,3,4], '3': [9,2,3]}
                    }
        show_fliers (bool): if True, it will show the fliers (aka outliers). Else, fliers will be hidden. Default is True.
        save_dir (str): directory of where to save the plot (Default = None)
    """

    colormap = {'Changi':'#b28efe','Punggol':'#fff98b','Woodlands':'#e5ffba','Kallang':'#c7fcb3',
                'Bukit Timah':'#fcb3f4','Stamford Marina':'#fcd3b3','Singapore River':'#b8fcf0',
                'Kranji':'#fcd2fa','Pandan':'#ccebfc','Jurong':'#fcc0cd','Geylang':'#cfd2fc'}
    
    colors = [colormap[drainage_c] for drainage_c in list(boxplot_data)]
    
    fig, ax = plt.subplots(figsize=(15,5))
    plot_grouped_boxplot(boxplot_data,ax=ax,group_colors=colors,show_fliers = show_fliers)
    ax.set_ylabel('Precipitation (mm)')
    ax.set_xlabel('Drainage catchment')
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=-20,horizontalalignment='left')

    # manually add legends
    hatches = [None,'.','o', 'O']
    legend_names = ['daily rainfall total (mm)','highest 30 min rainfall (mm)','highest 60 min rainfall (mm)','highest 120 min rainfall (mm)']
    handles = [mpatches.Patch(facecolor='white',edgecolor='black',hatch=h, label=name) for name,h in zip(legend_names,hatches)]
    fig.legend(handles=handles, bbox_to_anchor = (0.5,-0.05), loc = 'lower center',ncol = len(handles))
    plt.tight_layout()
    if save_dir is not None and os.path.exists(save_dir):
        # export csv file
        fp_save = os.path.join(save_dir,'Rainfall_flood_drainageCatchment.png')
        plt.savefig(fp_save, bbox_inches = 'tight')
    plt.show()
    return boxplot_data

def get_boxplot_data(flood_df):
    """ returns a dict where keys are drainage catchment names, and values 
        are dict of pr_total and a list of ppt values
    Args:
        flood_df (pd.DataFrame): which has the column names 'drainage_catchment','daily rainfall total (mm)','highest 30 min rainfall (mm)', 'highest 60 min rainfall (mm)',
        'highest 120 min rainfall (mm)'
    Returns:
        dict:  where keys are drainage catchment names, and values are dict of 'drainage_catchment','daily rainfall total (mm)','highest 30 min rainfall (mm)', 'highest 60 min rainfall (mm)',
        'highest 120 min rainfall (mm)' and a list of corresponding ppt values. Note: NA values are removed
    """
    boxplot_data = dict()
    ppt_names = ['daily rainfall total (mm)','highest 30 min rainfall (mm)','highest 60 min rainfall (mm)','highest 120 min rainfall (mm)']
    for entry in flood_df.to_dict('records'):
        drainage_catchment = entry['drainage_catchment']

        if drainage_catchment not in list(boxplot_data):
            boxplot_data[drainage_catchment] = dict()
        
        if len(list(boxplot_data[drainage_catchment])) == 0:
            for i in ['daily rainfall total (mm)','highest 30 min rainfall (mm)','highest 60 min rainfall (mm)','highest 120 min rainfall (mm)']:
                boxplot_data[drainage_catchment][i] = []
        
        for ppt_name in ppt_names:
            ppt_val = entry[ppt_name]
            if not np.isnan(ppt_val): # append to list if value is not NaN
                boxplot_data[drainage_catchment][ppt_name].append(ppt_val)

    return boxplot_data

def get_historical_ppt_percentiles(fp, 
                                   rf_types = ['highest 30 min rainfall (mm)','highest 60 min rainfall (mm)','highest 120 min rainfall (mm)'],
                                   percentiles = [50, 75, 90, 95, 99]):
    """ returns a extreme rainfall values for max 30, 60, 120 mins
    Args:
        fp (str): filepath to the historical weather csv
        rf_types (list of str): type of ppt e.g. max 30mins, max 60mins
        percentiles (list or np.array): percentile values to extract from
    Returns:
        percentiles (list or np.array): percentile values to extract from
        dict: returns the 50, 75, 90, 95, 99th percentile of historical rainfall events for each rainfall type
    """
    weather_df1 = pd.read_csv(fp)
    historical_ppt_percentiles = dict()
    for rf_type in rf_types:
        x = weather_df1[rf_type].values
        x = x[~np.isnan(x)]
        p = np.percentile(x, percentiles) # CCRS deems extreme ppt at 95th and 99th percentile
        historical_ppt_percentiles[rf_type] = p
    return percentiles, historical_ppt_percentiles