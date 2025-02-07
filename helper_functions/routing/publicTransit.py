import requests
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
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
        self.tripId = busLeg['tripId']
        self.mode = busLeg['mode']
        self.routeId = busLeg['routeId'] # bus number
        self.legGeometry = busLeg['legGeometry']

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

class BusStopRoutes:
    def __init__(self, G, gtfs):
        """
        Args:
            G (MultiDiGraph): graph of drive network
            gtfs (pd.DataFrame): gtfs shapes df of a selected bus service and route from gtfs_shape
        """
        self.G = G
        self.gtfs = gtfs

    def get_bus_edges_nodes(self):
        """ 
        get edges where bus stops lie on top/closest to the road
        from the edges, derive the nodes of the edges
        Returns:
            list: list of candidate node IDs that make up the route
        """
        edges_GTFS = ox.distance.nearest_edges(self.G,X = self.gtfs['shape_pt_lon'], Y = self.gtfs['shape_pt_lat'])
        # find the shortest path between nodes, minimizing travel time
        routes = []
        for e in range(len(edges_GTFS)-1):
            # get the shortest path within an identified edge
            route1 = [edges_GTFS[e][0], edges_GTFS[e][1]]
            # get the shortest path from the end of an edge to the start of the next edge
            route2 = ox.shortest_path(self.G, edges_GTFS[e][1], edges_GTFS[e+1][0], weight="travel_time")
            route = route1 + route2
            for r in route:
                # append the nodes visited by shortest path in sequential order
                routes.append(r)
        # append the last node of the last edge
        routes.append(edges_GTFS[-1][1])
        # print('edges_GTFS: ',edges_GTFS)
        return routes


    def get_route_graph(self, routes: list, plot = True):
        """ 
        The goal is to build a graph from the sequence of nodes, and locate if there is a connected shortestest path from the candidates nodes
        Args:
            routes (list): list of sequence of nodes in order from the origin to destination
            plot (bool): If true, plot a simple network graph on mpl
        Return:
            G (nx.DiGraph)
        """
        G = nx.DiGraph() # instatiate an empty directed graph
        G.add_node(routes[0]) # instatiate origin node as the origin
        for i in range(len(routes)-1): #
            # we want to ensure that the nodes are only added downstream from the O to D
            if (routes[i] != routes[i+1]) and (not G.has_node(routes[i+1])): # make sure there is no self-loop, and the children node is not the parent node
                G.add_edge(routes[i],routes[i+1])

        if plot:
            nodes = list(G)
            labels = {n:f'{i}: {n}' for i,n in enumerate(nodes)}
            if routes[0] in nodes:
                labels[routes[0]] = f"O: {routes[0]}"
            if routes[-1] in nodes:
                labels[routes[-1]] = f"D: {routes[-1]}"
            nx.draw(G,labels=labels,font_size=6)
            
        return G

    def check_shortest_path(self, G, origin_node, destination_node):
        """ 
        if there is a connected path from origin to destination_node, return the shortest OD path
        Args:
            G (nx.DiGraph): graph generated from `get_route_graph`
            origin_node (int): node ID of origin
            destination_node (int): node ID of destination
        """
        return nx.shortest_path(G, origin_node, destination_node)

    def busRoute_shortestPath(self, plot = True):
        """ 
        get edges where bus stops lie on top/closest to the road
        from the edges, derive the nodes of the edges
        Args:
            G (MultiDiGraph): graph of drive network
            gtfs (pd.DataFrame): gtfs shapes df
        Returns:
            list: list of candidate node IDs that make up the route
        """
        # get nodes from edges
        routes = self.get_bus_edges_nodes()
        sub_G = self.get_route_graph(routes,plot=False)
        try:
            updated_nodes = self.check_shortest_path(sub_G, routes[0], routes[-1])
        except:
            updated_nodes = [] # if there is an error, yield 0 nodes
        
        if plot:
            try:
                fig,ax = ox.plot_graph_route(self.G, updated_nodes, node_size=0,show=False,close=False)
                min_lat, max_lat, delta_lat, min_lon, max_lon, delta_lon = get_bus_lims(self.gtfs['shape_pt_lat'],self.gtfs['shape_pt_lon'])
                ax.set_ylim(min_lat-0.5*delta_lat,max_lat+0.5*delta_lat)
                ax.set_xlim(min_lon-0.5*delta_lon,max_lon+0.5*delta_lon)
                plt.show()

            except:
                plot_routes(self.G,updated_nodes,self.gtfs)

        return updated_nodes

# def identify_duplicated_node(nodes: list):
#     """ 
#     identifies duplicated nodes and returns the indices of the duplicated nodes
#     Args:
#         nodes (list): list of node IDs
#     Returns:
#         dict: keys are node IDs, values are list of indices where the node is duplicated
#     """
#     route_dict = dict()
#     for i,n in enumerate(nodes):
#         if n not in route_dict:
#             route_dict[n] = [i]
#         else:
#             route_dict[n].append(i)
#     return route_dict

# def busRoute_shortestPath(G,gtfs, plot = True):
#     """ 
#     get edges where bus stops lie on top/closest to the road
#     from the edges, derive the nodes of the edges
#     Args:
#         G (MultiDiGraph): graph of drive network
#         gtfs (pd.DataFrame): gtfs shapes df
#     Returns:
#         list: list of candidate node IDs that make up the route
#     """
#     # get nodes from edges
#     routes = get_bus_edges_nodes(G,gtfs)
#     # identify which nodes get duplicated
#     duplicated_nodes = identify_duplicated_node(routes) # returns a dict where keys = nodes, values = list of indices of the duplicated nodes
#     updated_nodes = check_connectivity(G,duplicated_nodes)
    
#     if plot:
#         try:
#             fig,ax = ox.plot_graph_route(G, updated_nodes, node_size=0,show=False,close=False)
#             min_lat, max_lat, delta_lat, min_lon, max_lon, delta_lon = publicTransit.get_bus_lims(gtfs['shape_pt_lat'],gtfs['shape_pt_lon'])
#             ax.set_ylim(min_lat-0.5*delta_lat,max_lat+0.5*delta_lat)
#             ax.set_xlim(min_lon-0.5*delta_lon,max_lon+0.5*delta_lon)
#             plt.show()

#         except:
#             publicTransit.plot_routes(G,updated_nodes,gtfs)

#     return updated_nodes

# def check_connectivity(G,nodes_dict):
#     """ 
#     checks connectivity between the candidate nodes, and removes any disjointe paths
#     Args:
#         G (MultiDiGraph): graph of drive network
#         nodes_dict (dict): dict: keys are node IDs, values are list of indices where the node is duplicated
#     Returns:
#         list: node IDs which shows the connected route
#     """
#     candidate_nodes = list(nodes_dict)
#     nodes_last_idx = [i[-1] for i in list(nodes_dict.values())]
#     # if the route progresses naturally without hiccups, it should only traverse each node once
#     # if there is a disjoint/disruption in the path, then the routes will not be strictly increasing
#     # a disjoint manifest in a negative value when we take the diff
#     negative_diff = np.diff(nodes_last_idx) # diff[i] = x[i+1] - x[i]
#     negative_ix = np.where(negative_diff<0)[0] + 1 # returns the index where diff is negative
#     negative_ix = negative_ix.tolist()
#     # find the shortest path between the 2 nodes prior and after the negative_ix node
#     updated_nodes = []
#     counter_nix = 0
#     current_ix = negative_ix[counter_nix]
#     for i in range(len(candidate_nodes)):
#         if i != current_ix:
#             updated_nodes.append(candidate_nodes[i])
#         else:
#             if (i > 0) and (i < len(candidate_nodes) - 1):
#                 nix1, nix2 = current_ix-1,current_ix+1 # the indices before and after the negative_ix nodes
#                 print(f'{candidate_nodes[nix1]} to {candidate_nodes[nix2]}')
#                 rerouted_route = ox.shortest_path(G, candidate_nodes[nix1], candidate_nodes[nix2], weight="travel_time")
#                 updated_nodes.extend(rerouted_route[1:-1])
#             else:
#                 # if the node to remove is at the start or at the end, just skip the node
#                 pass
#             counter_nix += 1
#             if counter_nix < len(negative_ix):
#                 current_ix = negative_ix[counter_nix]
#     return updated_nodes
# =======PLOTTING UTILS===============

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
        xlim_factor (float): expand plot xlimits based on coordinates limits
        ylim_factor (float): expand plot ylimits based on coordinates limits
    Returns:
        fig, ax: plotting params. Yellow edges refer to edges determined by busLeg_df, green edges refer to edges determined by GTFS
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
        xlim_factor (float): expand plot xlimits based on coordinates limits
        ylim_factor (float): expand plot ylimits based on coordinates limits
    Returns:
        fig, ax: plotting params. yellow refer to the nearest nodes from busLeg_df, red points refer to busLeg_df, green points refer to GTFS, 
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
        
def plot_routes(G,routes,gtfs,ax=None, xlim_factor = 0.2,ylim_factor = 0.5):
    """ 
    Plot route for every adjacent node
    Args:
        G (G): driving route
        routes (list): list of candidate nodes that make up the route
        gtfs (pd.DataFrame): GTFS shape df that shows the bus stop coords
        ax (mpl.Axes): if None, plot on a new figure, else, plot on supplied ax
        xlim_factor (float): expand plot xlimits based on coordinates limits
        ylim_factor (float): expand plot ylimits based on coordinates limits
    """
    fig, ax = ox.plot_graph_route(G, [routes[0],routes[1]], node_size=0,
                                route_color="r",
                                ax=ax,show=False,close=False)
    color_cycler = ['r','g','b']
    for i in range(1,len(routes)-1):
        ix_color = i%len(color_cycler)
        try:
            ox.plot_graph_route(G, [routes[i],routes[i+1]], node_size=0,
                                    route_color=color_cycler[ix_color],
                                    ax=ax,show=False,close=False)
        except:
            print(f'{routes[i]} to {routes[i+1]}')
            pass
    
    # get graph limits
    min_lat,max_lat,delta_lat,min_lon,max_lon,delta_lon = get_bus_lims(gtfs['shape_pt_lat'],gtfs['shape_pt_lon'])
    # set graph lims
    ax.set_ylim(min_lat-ylim_factor*delta_lat,max_lat+ylim_factor*delta_lat)
    ax.set_xlim(min_lon-xlim_factor*delta_lon,max_lon+xlim_factor*delta_lon)
    return fig, ax