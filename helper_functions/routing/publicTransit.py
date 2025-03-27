import requests
import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
import importlib
import LTA_API_key
import helper_functions.utils
importlib.reload(LTA_API_key)
importlib.reload(helper_functions.utils)
import LTA_API_key as apiKeys
import helper_functions.utils as utils
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
    Returns:
        list of dict: where each dict represents different itineraries (i.e. different ways from O to D)
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
    def __init__(self, G, gtfs,lat_name='shape_pt_lat',lon_name='shape_pt_lon'):
        """
        Args:
            G (MultiDiGraph): graph of drive network
            gtfs (pd.DataFrame): gtfs shapes df of a selected bus service, route, and bus stop coordinates
            lat_name (str): column name of bus stop latitude
            lon_name (str): column name of bus stop longitude
        """
        self.G = G
        self.gtfs = gtfs
        self.lat_name = lat_name
        self.lon_name = lon_name

    def get_bus_edges_nodes(self):
        """ 
        get edges where bus stops lie on top/closest to the road
        from the edges, derive the nodes of the edges
        Returns:
            list: list of candidate node IDs that make up the route
        """
        edges_GTFS = ox.distance.nearest_edges(self.G,X = self.gtfs[self.lon_name], Y = self.gtfs[self.lat_name])
        # find the shortest path between nodes, minimizing travel time
        routes = []
        for e in range(len(edges_GTFS)-1):
            # get the shortest path within an identified edge
            route1 = [edges_GTFS[e][0], edges_GTFS[e][1]]
            # get the shortest path from the end of an edge to the start of the next edge
            route2 = ox.shortest_path(self.G, edges_GTFS[e][1], edges_GTFS[e+1][0], weight="travel_time")
            if not isinstance(route2,list): # check if route2 returns a list
                print(f"Shortest path route does not return a list: {type(route2)}")
                route2 = []
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

        # get bus stop coordinates
        lat_iterable,lon_iterable = self.gtfs[self.lat_name], self.gtfs[self.lon_name]
        try:
            updated_nodes = self.check_shortest_path(sub_G, routes[0], routes[-1])
        except:
            updated_nodes = [] # if there is an error, yield 0 nodes
        
        if plot:
            try:
                fig,ax = ox.plot_graph_route(self.G, updated_nodes, node_size=0,show=False,close=False)
                min_lat, max_lat, delta_lat, min_lon, max_lon, delta_lon = get_bus_lims(lat_iterable,lon_iterable)
                ax.set_ylim(min_lat-0.5*delta_lat,max_lat+0.5*delta_lat)
                ax.set_xlim(min_lon-0.5*delta_lon,max_lon+0.5*delta_lon)
                plt.show()

            except:
                plot_routes(self.G,updated_nodes,lat_iterable,lon_iterable)

        return updated_nodes

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
        
def plot_routes(G,routes,lat_iterable,lon_iterable,ax=None, xlim_factor = 0.2,ylim_factor = 0.5):
    """ 
    Plot route for every adjacent node
    Args:
        G (G): driving route
        routes (list): list of candidate nodes that make up the route
        lat_iterable (Iterable, np.array, pd.Series): list of lat arrays that bounds the routes
        lon_iterable (Iterable, np.array, pd.Series): list of lon arrays that bounds the routes
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
    min_lat,max_lat,delta_lat,min_lon,max_lon,delta_lon = get_bus_lims(lat_iterable,lon_iterable)
    # set graph lims
    ax.set_ylim(min_lat-ylim_factor*delta_lat,max_lat+ylim_factor*delta_lat)
    ax.set_xlim(min_lon-xlim_factor*delta_lon,max_lon+xlim_factor*delta_lon)
    return fig, ax

def public_transit_routing(origin_coord, origin_key, destination_coord, destination_key,
                           G_bus, GTFS_shapes, headers,
                           route_type = "pt" ,
                            date = '02-05-2025',
                            time = '08%3A00%3A00' ,
                            mode = 'TRANSIT' ,
                            maxWalkDistance = '1000' ,
                            numItineraries = '3',
                            busRouteMode = 'OneMap',
                            plot = True,
                            save_fp = None):
    """ Information captured for an individual OD journey. Obtain routes from GTFS.
    Args:
        origin_coord (tuple): lat,lon
        origin_key (Any): a key to help u identify the origin coord e.g. bus stop code
        destination_coord (tuple): lat,lon
        destination_key (Any): a key to help u identify the destination coord e.g. nodeID in G_bus
        G_bus (MultiDiGraph): graph of drive bus network
        GTFS_shapes (pd.DataFrame): dataframe of the GTFS's shapes file
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
        busRouteMode (str): if bus route is extracted from GTFS, then busRouteMode = 'GTFS'
                            else if bus route is extracted from OneMap API's bus leg, then busRouteMode = 'OneMap' (Default)
        plot (bool): if True, plot shortest bus route
        save_fp (str or None): exports the itinerary as json data
    """
    # initialise OD
    start_lat = origin_coord[0]
    start_lon = origin_coord[1]
    end_lat = destination_coord[0]
    end_lon = destination_coord[1]

    def checkGTFS(gtfs_df):
        """ check if df is of length 0"""
        if len(gtfs_df.index) == 0:
            raise Exception("GTFS df is of zero length!")
        else:
            return gtfs_df
        
    # fetch itinerary from OneMapAPI
    try:
        itineraries = get_OneMap_itineraries(headers=headers,
                                        start_lat = start_lat,
                                        start_lon = start_lon,
                                        end_lat = end_lat,
                                        end_lon= end_lon,
                                        route_type = route_type ,
                                        date = date,
                                        time = time ,
                                        mode = mode ,
                                        maxWalkDistance = maxWalkDistance ,
                                        numItineraries = numItineraries
                                        )
    except:
        # if token expires, generate a new token
        headers = generate_OneMap_token()
        itineraries = get_OneMap_itineraries(headers=headers,
                                        start_lat = start_lat,
                                        start_lon = start_lon,
                                        end_lat = end_lat,
                                        end_lon= end_lon,
                                        route_type = route_type ,
                                        date = date,
                                        time = time ,
                                        mode = mode ,
                                        maxWalkDistance = maxWalkDistance ,
                                        numItineraries = numItineraries
                                        )

    # get bus legs from the FIRST itinerary
    OMI = OneMapItinerary(itinerary=itineraries[0])
    busLeg_dfs = OMI.get_bus_routes()

    # initialise empty dict to store data
    # assumes that route is from any bus stop in SG to a workplace
    save_itinerary = {'busStart':(start_lat,start_lon, origin_key),
                      'workEnd': (end_lat, end_lon, destination_key),
                      'duration': OMI.duration,
                      'startTime': OMI.startTime,
                      'endTime': OMI.endTime,
                      'transitTime': OMI.transitTime,
                      'waitingTime': OMI.waitingTime,
                      'transfers': OMI.transfers,
                      'busLegs': []
                      }
    # iterate across different busLeg
    for busLeg_number in range(len(busLeg_dfs)):
        busLeg = OMI.busLegs[busLeg_number]
        # get bus leg meta data
        duration = busLeg.duration
        distance = busLeg.distance
        startTime = busLeg.startTime
        endTime = busLeg.endTime
        tripId = busLeg.tripId
        tripDirection = int(tripId.split('-')[1]) - 1
        mode = busLeg.mode
        routeId = busLeg.routeId
        legGeometry = busLeg.legGeometry
        # get bus leg df
        busLeg_df = busLeg_dfs[busLeg_number]
        busLeg_dict = busLeg_df.to_dict('records')
        stopSequence = busLeg_df['stopSequence'].to_list()
        # filter gtfs_shapes based on bus's stop sequence
        # check if gtfs has length 0
        try:
            gtfs = checkGTFS(GTFS_shapes[(GTFS_shapes['shape_id'].str.contains(f'^{routeId}:WD:{tripDirection}.*')) & (GTFS_shapes['shape_pt_sequence'].isin(stopSequence))])
        except:
            try:
                gtfs = checkGTFS(GTFS_shapes[(GTFS_shapes['shape_id'].str.contains(f'^{routeId}:SAT:{tripDirection}.*')) & (GTFS_shapes['shape_pt_sequence'].isin(stopSequence))])
            except:
                try:
                    gtfs = checkGTFS(GTFS_shapes[(GTFS_shapes['shape_id'].str.contains(f'^{routeId}:SUN:{tripDirection}.*')) & (GTFS_shapes['shape_pt_sequence'].isin(stopSequence))])
                except:
                    gtfs = None
                    gtfs_dict = None
                    # raise Exception(f'Relevant GTFS data cannot be obtained from the routeId and trip direction.Check if GTFS has the bus service.: {routeId} ')
        # make sure to sort the shape_pt_sequence in ascending order to ensure chronological sequence of bus stops visited along bus route
        if gtfs is not None:
            gtfs = gtfs.sort_values('shape_pt_sequence')
            gtfs_dict = gtfs.to_dict('records')
        # get shortest bus route based on gtfs_shape sequence
        try:
            if busRouteMode == 'OneMap':
                # if bus route is extracted from OneMap API's bus leg, then input for gtfs=busLeg_df
                gtfs = busLeg_df
                lat_name = 'lat'
                lon_name = 'lon'
            else:
                # if bus route is extracted from GTFS, then input for gtfs = gtfs
                lat_name = 'shape_pt_lat'
                lon_name = 'shape_pt_lon'
            BSR = BusStopRoutes(G=G_bus,gtfs=gtfs,lat_name=lat_name,lon_name=lon_name)
            route_nodesID = BSR.busRoute_shortestPath(plot=plot)
        except Exception as e:
            # if there is an error, change the save_fp name so that it's easy to identify the error
            if save_fp is not None:
                save_fp = f'{os.path.splitext(save_fp)[0]}_ERROR'
            route_nodesID = None # save the error msg as nodesID

        save_busLeg = {'leg_number': busLeg_number,
                       'duration': duration,
                       'distance': distance,
                       'startTime': startTime,
                       'endTime': endTime,
                       'tripId': tripId,
                       'tripDirection': tripDirection,
                       'mode': mode,
                       'routeId': routeId,
                       'legGeometry': legGeometry,
                       'busLeg': busLeg_dict,
                       'gtfs': gtfs_dict,
                       'routesNodesID': route_nodesID
                       }
        save_itinerary['busLegs'].append(save_busLeg)
    
    if save_fp is not None:
        # export as json
        try:
            utils.json_data(save_itinerary, save_fp)
        except:
            print('Object is not serializable as JSON')
    return save_itinerary

class TripItinerary:
    """helper functions for processing public transit itinerary json file into a consolidated pd.DataFrame"""
    def __init__(self,G, itinerary):
        """ 
        Args:
            G (G): driving network
            itinerary (dict): itinerary from public transit routing
        """
        self.G = G
        self.itinerary = itinerary

    @classmethod
    def from_file(cls, G, fp):
        """ defines factory method that returns an instance of the class
        Args:
            fp (str): filepath to itinerary json file
        Returns:
            dict: itinerary object
        """
        itinerary = utils.load_json(fp)
        return cls(G, itinerary)

    def bounding_box_coords(self,start_coords,end_coords):
        """ 
        get coordinates bounding box
        """
        min_lat = min([start_coords[0],end_coords[0]])
        max_lat = max([start_coords[0],end_coords[0]])
        min_lon = min([start_coords[1],end_coords[1]])
        max_lon = max([start_coords[1],end_coords[1]])
        delta_lat = max_lat-min_lat
        delta_lon = max_lon-min_lon
        return min_lat,max_lat,delta_lat,min_lon,max_lon,delta_lon

    def get_itinerary_bus_routes(self,color_cycler=["y", "c","purple"]):
        """ 
        Args:
            color_cycler (list of str): list of color for cycling through colors
        Returns:
            tuple: list, list, list
                list: list of routes, where route is a list of nodes
                list: routeID aka service number of buses
                list: list of rgb
        """
        routes = []
        rc = []
        routeId = []
        for i, busLeg in enumerate(self.itinerary['busLegs']):
            ix_color = i%len(color_cycler)
            routes.append(busLeg['routesNodesID'])
            rc.append(color_cycler[ix_color])
            routeId.append(busLeg['routeId'])
        return routes, routeId, rc

    def get_route_time_and_distance(self,route):
        """ get simulated route time and distance via osmnx
        Args:
            route (list): list of candidate nodes that make up the route
        Returns:
            tuple: route_length,route_time
        """
        route_length = int(sum(ox.routing.route_to_gdf(self.G, route, weight="travel_time")["length"]))
        route_time = int(sum(ox.routing.route_to_gdf(self.G, route, weight="travel_time")["travel_time"]))
        return route_length,route_time

    def get_non_bus_duration(self):
        """ get non bus duration from OneMap itinerary e.g. total duration - bus duration
        Returns:
            tuple: total duration, time spend on buses. time spent on non-bus public transit (e.g. walking, mrt travel duration, time spent at bus stops & traffic lights, etc) 
        """
        total_duration = float(self.itinerary['duration'])
        bus_duration = sum([float(busLeg['duration']) for busLeg in self.itinerary['busLegs']])
        return total_duration, bus_duration

    def plot_itinerary(self, 
                    ax = None,
                    xlim_factor = 0.2,ylim_factor = 0.5):
        """ 
        plot itinerary and the bus routes herein
        Args:
            ax (mpl.Axes): if None, plot on a new figure, else, plot on supplied ax
            xlim_factor (float): expand plot xlimits based on coordinates limits
            ylim_factor (float): expand plot ylimits based on coordinates limits
        """
        start_coords = self.itinerary['busStart'][:2]
        end_coords = self.itinerary['workEnd'][:2]
        # get graph limits
        min_lat,max_lat,delta_lat,min_lon,max_lon,delta_lon = self.bounding_box_coords(start_coords,end_coords)
        
        routes, routeId ,rc = self.get_itinerary_bus_routes()
        fig, ax = ox.plot_graph_routes(self.G, routes, route_colors=rc, route_linewidth=6, node_size=0,
                                    ax=ax,show=False,close=False)
        
        # plot orig node
        ax.scatter(start_coords[1],start_coords[0],marker="X",c="g",s=25)
        # plot end node
        ax.scatter(end_coords[1],end_coords[0],marker="X",c="r",s=25)

        # set graph lims
        ax.set_ylim(min_lat-ylim_factor*delta_lat,max_lat+ylim_factor*delta_lat)
        ax.set_xlim(min_lon-xlim_factor*delta_lon,max_lon+xlim_factor*delta_lon)
        return

    def get_itinerary_entry(self):
        """ 
        Returns:
            dict: for an entry of a pandas row, to be consolidated into a df when calling pd.DataFrame.from_records
        """
        start_lat = self.itinerary['busStart'][0]
        start_lon = self.itinerary['busStart'][1]
        end_lat = self.itinerary['workEnd'][0]
        end_lon = self.itinerary['workEnd'][1]
        actual_total_duration, actual_bus_duration = self.get_non_bus_duration()
        non_bus_duration = actual_total_duration - actual_bus_duration
        routes, routeId, _ = self.get_itinerary_bus_routes()
        actual_bus_distance = sum([float(busLeg['distance']) for busLeg in self.itinerary['busLegs']])
        simulated_bus_distance = simulated_bus_duration = 0
        for r in routes:
            route_length,route_time = self.get_route_time_and_distance(r)
            simulated_bus_distance += route_length
            simulated_bus_duration += route_time
        number_of_busroutes = len(self.itinerary['busLegs'])
        simulated_total_duration = non_bus_duration + simulated_bus_duration
        return {'start_lat':start_lat,'start_lon':start_lon,'end_lat':end_lat,'end_lon':end_lon,
                'duration':self.itinerary['duration'],'transitTime':self.itinerary['transitTime'],
                'waitingTime':self.itinerary['waitingTime'],'transfers':self.itinerary['transfers'],
                'actual_bus_duration':actual_bus_duration,'simulated_bus_duration':simulated_bus_duration,
                'actual_bus_distance':actual_bus_distance,'simulated_bus_distance':simulated_bus_distance,
                'actual_total_duration':actual_total_duration,'simulated_total_duration':simulated_total_duration,
                'non_bus_duration':non_bus_duration,'number_of_busroutes':number_of_busroutes, 'routeId':','.join(routeId),
                }
    
