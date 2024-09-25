import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

def get_weather_stns():
    weather_stns = pd.read_csv(r"C:\Users\hypak\OneDrive - Singapore Management University\Documents\Data\SG_Climate_data\Weather_stations.csv")
    weather_stns_dict = weather_stns.to_dict('records')
    weather_stns_dict = {d['station_name'].lower().replace(' ',''):
                        {'station_code':d['station_code'],
                        'station_name':d['station_name'],
                        'longitude':d['longitude'],
                        'latitude':d['latitude']
                        } for d in weather_stns_dict}
    return weather_stns_dict

def get_closest_weather_stn(weather_stns_dict,stn_name):
    """ 
    returns the closest weather station
    """
    dist_diff = lambda lat1,lon1,lat2,lon2: ((lat2-lat1)**2 + (lon2-lon1)**2)**(1/2)
    stn_name = stn_name.lower().replace(' ','')
    lat1, lon1 = weather_stns_dict[stn_name]['latitude'], weather_stns_dict[stn_name]['longitude']
    diff_dist = {stn_n: dist_diff(weather_stns_dict[stn_n]['latitude'],weather_stns_dict[stn_n]['longitude'],lat1,lon1) for stn_n in weather_stns_dict.keys() if stn_n != stn_name}
    min_idx = np.argmin(list(diff_dist.values()))
    return list(diff_dist)[min_idx]