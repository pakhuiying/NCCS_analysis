import requests
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import osmnx as ox
import importlib
import LTA_API_key
importlib.reload(LTA_API_key)
import LTA_API_key as apiKeys
from datetime import datetime

def generate_OneMap_token():
    """ generates new one map token """
    onemapKey = apiKeys.get_OneMap_token()
    headers = {"Authorization": onemapKey}
    return headers

def get_OneMap_response(url,headers):
    """ 
    Args:
        url (str): url for request
        headers (dict): Authorization headers
    """
    response = requests.request("GET", url, headers=headers)
    response = response.json()
    return response

def get_OneMap_itineraries(headers,start_lat,start_lon,end_lat,end_lon,date,time ,
                           route_type = "pt", mode = 'TRANSIT', maxWalkDistance = 1000, numItineraries = 3):
    """
    Args:
        headers (dict): Authorization headers
        start_lat (float): latitude coordinate of where you start your journey
        start_lon (float): longitude coordinate of where you start your journey
        end_lat (float): latitude coordinate of where you end your journey
        end_lon (float): longitude coordinate of where you end your journey
        route_type (str): "pt" # Route types available walk, drive, pt, and cycle. Only lowercase is allowed.
        date (str):  e.g. '01-13-2025' Date of the selected start point in MM-DD-YYYY.
        time (str): e.g. '07%3A35%3A00' Time of the selected start point in [HH][MM][SS], using the 24-hour clock system. 
        mode (str): e.g. 'TRANSIT'.  Mode of public transport: TRANSIT, BUS, RAIL. Entry must be specified in UPPERCASE
        maxWalkDistance (float): e.g. 1000. The maximum walking distance set by the user in metres.
        numItineraries (int): maximum number if possible itineraries to fetch
    """
    url = f"https://www.onemap.gov.sg/api/public/routingsvc/route?start={start_lat}%2C{start_lon}&end={end_lat}%2C{end_lon}&routeType={route_type}&date={date}&time={time}&mode={mode}&maxWalkDistance={maxWalkDistance}&numItineraries={numItineraries}"

    response = get_OneMap_response(url, headers)
    try:
        itineraries = response['plan']['itineraries']
    except Exception as e:
        print(f'Token may have expired, input new headers: {e}')
    return itineraries


class OneMapItinerary:
    def __init__(self, itinerary):
        """ 
        Args:
            itinerary (dict): an itinerary, where keys are the metadata of the itinerary. Itineraries are obtained from OneMap's routing API
            GTFS (pd.DataFrame): a dataframe created from joining GTFS's stop_times and stops, such that stop sequence, and its stop ID and coordinates are captured
        """
        self.itinerary = itinerary
        self.duration = itinerary['duration'] # in seconds
        self.startTime = itinerary['startTime']
        self.endTime = itinerary['endTime']
        self.transitTime = itinerary['transitTime']
        self.waitingTime = itinerary['waitingTime']
        self.transfers = itinerary['transfers']
        self.n_legs = len(itinerary['legs'])
        self.legs = itinerary['legs']
    
    def get_bus_legs(self):
        """ 
        filter legs where mode == 'BUS'
        """
        return [l for l in self.legs if l['mode']=='BUS']
    
    def get_bus_routes(self):
        """ 
        concatenate all bus legs to get the intermediate bus stops
        Returns:
            list: list of dataframes, where each dataframe represent the bus routes
        """
        bus_legs = self.get_bus_legs()
        busLegs = [BusLeg(b) for b in bus_legs]
        # add to attribute
        self.busLegs = busLegs
        busLegs_dfs = [B.get_stops_data() for B in busLegs]
        # busLeg_df = pd.concat(busLegs_dfs)
        return busLegs_dfs
    

    def get_total_distance(self):
        """ 
        must have already ran `get_bus_routes`
        calculate total distance for all the bus legs
        """
        return sum([B.distance for B in self.busLegs])
    
    def get_total_duration(self):
        """ 
        must have already ran `get_bus_routes`
        calculate total duration for all the bus legs
        """
        return sum([B.duration for B in self.busLegs])

class BusLeg:
    def __init__(self, busLeg):
        """ 
        Args:
            busLeg (dict): a bus leg, where keys are meta data for the bus leg
            GTFS (pd.DataFrame): dataframe of the GTFS's shapes file
        """
        self.busLeg = busLeg
        self.duration = busLeg['duration'] # in seconds
        self.distance = busLeg['distance'] # in metres
        self.startTime = busLeg['startTime']
        self.endTime = busLeg['endTime']
        
        self.mode = busLeg['mode']
        self.routeId = busLeg['routeId'] # bus number

    def get_stops_data(self):
        """ 
        Args:
            stops (dict): with keys: 'arrival','departure','lat','lon','name','stopCode','stopId','stopIndex','stopSequence','vertexType'
        Returns:
            pd.DataFrame: shows a sequence of bus stops at 'from' bus stop to 'to' bus stop
        """
        stops_list = [self.busLeg['from']] + self.busLeg['intermediateStops'] + [self.busLeg['to']]
        stops_df = pd.DataFrame(stops_list)
        
        return stops_df

def get_bus_lims(lat,lon):
    """ 
    Args:
        lat (Iterable, array, pd.Geoseries): latitude coords
        lon (Iterable, array, pd.Geoseries): longitude coords
    Returns:
        returns lat extent, lat_delta, lon extent, lon_delta
    """
    min_lat, max_lat = lat.min(),lat.max()
    delta_lat = max_lat - min_lat
    min_lon, max_lon = lon.min(),lon.max()
    delta_lon = max_lon - min_lon
    return min_lat, max_lat, delta_lat, min_lon, max_lon, delta_lon

def plot_bus_edges(G, gtfs, busLeg_df,ax=None, xlim_factor = 0.2,ylim_factor = 0.5):
    """ 
    plot edges where bus stops/shapes fall on
    Args:
        G (G): driving route
        gtfs (pd.DataFrame): GTFS shape df that shows the bus stop coords
        busLeg_df (pd.DataFrame): dataframe represent the bus routes obtained from OneMapItinerary or BusLeg
        ax (mpl.Axes): if None, plot on a new figure, else, plot on supplied ax
        xlim_factor (float): expand plot limits based on coordinates limits
    Returns:
        fig, ax: plotting params
    """
    # edges return a list of tuples (u,v,k)
    edges = ox.distance.nearest_edges(G,X = busLeg_df['lon'], Y = busLeg_df['lat'])
    edges_GTFS = ox.distance.nearest_edges(G,X = gtfs['shape_pt_lon'], Y = gtfs['shape_pt_lat'])
    # plot edges
    edges = list(edges)
    edges_GTFS = list(edges_GTFS)
    ec = dict()
    ew = dict()
    for e in G.edges(keys=True):
        if e in edges:
            ec[e] = "yellow"
            ew[e] = 2
        elif e in edges_GTFS:
            ec[e] = "green"
            ew[e] = 2
        else:
            ec[e] = "white"
            ew[e] = 0.2
    
    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_color = list(ec.values()),
        edge_linewidth=list(ew.values()),
        edge_alpha = 0.5,
        ax=ax,
        show = False,
        close = False
    )
    # get graph limits
    min_lat,max_lat,delta_lat,min_lon,max_lon,delta_lon = get_bus_lims(busLeg_df['lat'],busLeg_df['lon'])
    # set graph limits
    ax.set_ylim(min_lat-ylim_factor*delta_lat,max_lat+ylim_factor*delta_lat)
    ax.set_xlim(min_lon-xlim_factor*delta_lon,max_lon+xlim_factor*delta_lon)
    return fig, ax

def plot_bus_nodes(G, gtfs, busLeg_df,ax=None, xlim_factor = 0.2,ylim_factor = 0.5):
    """ 
    plot nearest nodes based on gtfs and buslegs coords
    Args:
        G (G): driving route
        gtfs (pd.DataFrame): GTFS shape df that shows the bus stop coords
        busLeg_df (pd.DataFrame): dataframe represent the bus routes obtained from OneMapItinerary or BusLeg
        ax (mpl.Axes): if None, plot on a new figure, else, plot on supplied ax
        xlim_factor (float): expand plot limits based on coordinates limits
    Returns:
        fig, ax: plotting params
    """
    nodes = ox.distance.nearest_nodes(G,X = busLeg_df['lon'], Y = busLeg_df['lat'])
    nc = ["yellow" if node in nodes else "white" for node in G.nodes()]
    ns = [15 if node in nodes else 0 for node in G.nodes()]
    fig, ax = ox.plot_graph(G,ax=ax,
                            node_color=nc,node_size=ns,edge_linewidth=0.2,
                            show=False,close=False)
    for ix, rows in busLeg_df.iterrows():
        ax.plot(rows['lon'],rows['lat'],'ro',alpha=0.7)
    ax.plot(gtfs['shape_pt_lon'],gtfs['shape_pt_lat'],'go',alpha=0.7)
    # get graph limits
    min_lat,max_lat,delta_lat,min_lon,max_lon,delta_lon = get_bus_lims(busLeg_df['lat'],busLeg_df['lon'])
    # set graph limits
    ax.set_ylim(min_lat-ylim_factor*delta_lat,max_lat+ylim_factor*delta_lat)
    ax.set_xlim(min_lon-xlim_factor*delta_lon,max_lon+xlim_factor*delta_lon)
    return fig, ax
        
        