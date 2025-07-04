import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects
import polyline
from shapely.geometry import LineString
import osmnx as ox
import networkx as nx
import importlib
import LTA_API_key
import warnings
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

def plot_bus_routes(G,itinerary_fp,planningArea, shift_x, shift_y,
                    flooded_edges=None,ax=None,
                    cmap="plasma",plot_groundtruth=True,fontsize=15,linewidth=3,title="",
                    save_fp = None):
    """   Plot indivdual bus routes with unique colors, showing bus route ID
    TO decode polyline, can use https://developers.google.com/maps/documentation/utilities/polylineutility
    or python package: https://github.com/frederickjansen/polyline
    Args:
        G (G): driving route
        itinerary_fp (str): filepath of the itinerary
        planningArea (gpd.GeoDataFrame): geopandas df of planning areas of SG
        shift_x (list of float): longitude shifts to plot the arrows
        shift_y (list of float): latitude shifts to plot the arrows
        flooded_edges (list of edges): each item in this list is an edge e.g. (u,v, key)
        ax (mpl.Axes): if None, plot on a new figure, else, plot on supplied ax
        cmap (str): Colors that are assigned to individual bus route shall be sampled from plasma. see mpl colormap_reference
        plot_groundtruth (bool): if True, plot the polyline from OneMap
        fontsize (float): fontsize of bus route service ID
        title (str): title for plot
        save_fp (str): file path to save figure to
    """
    # load itinerary
    itinerary = utils.load_json(itinerary_fp)
    # get colormap
    cmap = mpl.colormaps[cmap]
    # get unique colors based on number of bus legs
    colors = cmap(np.linspace(0,1, len(itinerary['busLegs'])))
    # plot base map
    if ax is None:
        fig, ax = ox.plot_graph(
            G,node_size=0,edge_linewidth=0.5,
            bgcolor="white",
            show = False,
            close = False
        )
    # plot planning area boundary
    planningArea.plot(fc="None",ec="lightgrey",alpha=0.7,ax=ax,linewidth=2)
    # flooded edges gdf
    edges_gdf = ox.graph_to_gdfs(G,nodes=False)
    flooded_edges_gdf = edges_gdf.loc[flooded_edges,:]
    flooded_edges_gdf.plot(ax=ax,color="blue")
    # store the lat and lon to set the xlim and ylim later
    lons = []
    lats = []
    # plot each bus leg
    for ix,busLeg in enumerate(itinerary['busLegs']):
        routesNodesID = busLeg['routesNodesID']
        color = colors[[ix],:]
        ox.plot_graph_route(G, routesNodesID, node_size=0, 
                            ax=ax,show=False,close=False,
                            route_color=color, route_linewidth=linewidth)
        # get groundtruth route
        polyline_str = busLeg['legGeometry']['points']
        route_geojson = polyline.decode(polyline_str,geojson=True)
        line = LineString(route_geojson)
        line_gdf = gpd.GeoSeries(line,crs = 4326)
        if plot_groundtruth:
            line_gdf.plot(color="green",alpha=0.7,ax=ax)
        # add route bounds
        lon_iterable = [r[0] for r in route_geojson]
        lat_iterable = [r[1] for r in route_geojson]
        lons.extend(lon_iterable)
        lats.extend(lat_iterable)
        # bus service
        print(busLeg['routeId'])
        # add annotations
        # text, end_coord (x,y), start_coord (x,y)
        dy = max(lat_iterable) - min(lat_iterable)
        xy_start = (busLeg['busLeg'][0]['lon'], busLeg['busLeg'][0]['lat'])
        shiftx = shift_x[ix]
        shifty = shift_y[ix]
        # shift = dy if bool(ix%2) else -dy # so the y placement of the text alternates
        xy_end = (busLeg['busLeg'][0]['lon'] + shiftx, busLeg['busLeg'][0]['lat'] + shifty)
        txt = ax.annotate(text=busLeg['routeId'], xy=xy_start, xytext=xy_end,xycoords='data',color="k",
                                arrowprops=dict(arrowstyle="->",connectionstyle="angle3", lw=1.5,color="k"),
                                ha="center", fontsize=fontsize,
                                path_effects=[patheffects.withStroke(linewidth=3,
                                                    foreground=color)])
        txt.arrow_patch.set_path_effects([
            patheffects.Stroke(linewidth=3, foreground=color),
            patheffects.Normal()])

    # add black border around axis
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth(2) 
    # set limit
    min_lat, max_lat, delta_lat, min_lon, max_lon, delta_lon = get_bus_lims(np.array(lats),np.array(lons))
    print(delta_lat,delta_lon)
    ax.set_ylim(min_lat-0.5*delta_lat,max_lat+0.5*delta_lat)
    ax.set_xlim(min_lon-0.5*delta_lon,max_lon+0.5*delta_lon)
    ax.set_title(title)
    if save_fp is not None:
        plt.savefig(save_fp, bbox_inches = 'tight')
    if ax is None:
        plt.show()
    return 

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

class PublicTransitRouting:
    def __init__(self, G, itinerary,itinerary_index=0):
        """ 
        Args:
            G (MultiDiGraph): graph of drive bus network
            itinerary (dict): itinerary from OneMap API
            itinerary_index (int): index of the itinerary to extract bus legs from
        """
        self.G = G
        self.itinerary = itinerary
        self.itinerary_index = itinerary_index
    
    @classmethod
    def from_file(cls, G, fp):
        """ defines factory method that returns an instance of the class
        Args:
            G (MultiDiGraph): graph of drive bus network
            fp (str): filepath to itinerary json file
        Returns:
            dict: itinerary object
        """
        itinerary = utils.load_json(fp)
        return cls(G, itinerary)
    
    def get_busLegs(self,plot=True):
        """ get bus legs from the itinerary
        Args:
            itinerary_index (int): index of the itinerary to extract bus legs from
            plot (bool): if True, plot shortest bus route
        Returns:
            list: list of bus legs in the itinerary
            error_flag (bool): if True, there is an error in the vehicle routing
        """
        # intialise flag to identify any errors in the vehicle routing
        error_flag = False
        # get bus legs from the FIRST itinerary
        OMI = OneMapItinerary(itinerary=self.itinerary['plan']['itineraries'][self.itinerary_index])
        # concatenate all bus legs to get the intermediate bus stops
        # list of dataframes, where each dataframe represent the bus routes
        busLeg_dfs = OMI.get_bus_routes()
        # initialise empty list to store data
        busLegs = []
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
            try:
                # try to get routing from osmnx graph
                BSR = BusStopRoutes(G=self.G,gtfs=busLeg_df,lat_name='lat',lon_name='lon')
                route_nodesID = BSR.busRoute_shortestPath(plot=plot)
            except:
                # if there is an error, means not vehicle route can be found, then set route_nodesID to None
                route_nodesID = None
                error_flag = True

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
                       'routesNodesID': route_nodesID
                       }
            busLegs.append(save_busLeg)
        return busLegs, error_flag
    
    def get_itinerary(self,save_fp=None):
        """ 
        Information captured for an individual OD journey.
        Args:
            save_fp (str or None): exports the itinerary as json data
        """
        busLegs, error_flag = self.get_busLegs(plot=False)
        # get itinerary meta data
        fromPlace = self.itinerary['requestParameters']['fromPlace']
        start_lat,start_lon = [float(i) for i in fromPlace.split(",")]
        toPlace = self.itinerary['requestParameters']['toPlace']
        end_lat, end_lon = [float(i) for i in toPlace.split(",")]
        # itinerary
        itinerary = self.itinerary['plan']['itineraries'][self.itinerary_index]

        save_itinerary = {'start_lat':start_lat,'start_lon':start_lon,
                          'end_lat':end_lat,'end_lon':end_lon,
                      'duration': itinerary['duration'], # in seconds
                      'startTime': itinerary['startTime'],
                      'endTime': itinerary['endTime'],
                      'transitTime': itinerary['transitTime'],
                      'waitingTime': itinerary['waitingTime'],
                      'transfers': itinerary['transfers'],
                      'itinerary_index': self.itinerary_index,
                        'error_flag': error_flag,
                      'busLegs': busLegs
                      }
        if save_fp is not None:
            # export as json
            utils.json_data(save_itinerary, save_fp)
        return save_itinerary
    
class TripItinerary:
    """helper functions for processing public transit itinerary json file into a consolidated pd.DataFrame"""
    def __init__(self,G, itinerary, fp):
        """ 
        Args:
            G (G): driving network
            itinerary (dict): itinerary from public transit routing
            fp (str): filepath to the itinerary, or it can be a unique ID corresponding to the itinerary
        """
        self.G = G
        self.itinerary = itinerary
        self.error_flag = self.itinerary['error_flag']
        self.fp = fp

    @classmethod
    def from_file(cls, G, fp):
        """ defines factory method that returns an instance of the class
        Args:
            fp (str): filepath to itinerary json file
        Returns:
            dict: itinerary object
        """
        itinerary = utils.load_json(fp)
        return cls(G, itinerary,fp)

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
        ORIGIN_PT_CODE = [] # store the start bus stop IDs for each bus journey
        DESTINATION_PT_CODE = [] # store the end bus stop IDs for each bus journey
        for i, busLeg in enumerate(self.itinerary['busLegs']):
            ix_color = i%len(color_cycler)
            routes.append(busLeg['routesNodesID'])
            rc.append(color_cycler[ix_color])
            routeId.append(busLeg['routeId'])
            # also append the bus stop ids for start and stop 
            ORIGIN_PT_CODE.append(busLeg['busLeg'][0]['stopCode'])
            DESTINATION_PT_CODE.append(busLeg['busLeg'][-1]['stopCode'])
        return routes, routeId, ORIGIN_PT_CODE, DESTINATION_PT_CODE, rc

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
            tuple: total duration, time spend on buses (list of float), total time spent on buses. time spent on non-bus public transit (e.g. walking, mrt travel duration, time spent at bus stops & traffic lights, etc) 
        """
        total_duration = float(self.itinerary['duration'])
        bus_duration = [float(busLeg['duration']) for busLeg in self.itinerary['busLegs']]
        total_bus_duration = sum(bus_duration)
        return total_duration, bus_duration, total_bus_duration

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
        # start_coords = self.itinerary['busStart'][:2]
        # end_coords = self.itinerary['workEnd'][:2]
        start_coords = self.itinerary['start_lat'],self.itinerary['start_lon']
        end_coords = self.itinerary['end_lat'],self.itinerary['end_lon']
        # get graph limits
        min_lat,max_lat,delta_lat,min_lon,max_lon,delta_lon = self.bounding_box_coords(start_coords,end_coords)
        
        routes, routeId ,ORIGIN_PT_CODE, DESTINATION_PT_CODE,rc = self.get_itinerary_bus_routes()
        fig, ax = ox.plot_graph_routes(self.G, routes, route_colors=rc, route_linewidth=6, node_size=0,
                                    ax=ax,show=False,close=False)
        
        # plot orig node
        # ax.scatter(start_coords[1],start_coords[0],marker="X",c="g",s=25)
        ax.scatter(self.itinerary['start_lon'],self.itinerary['start_lat'],marker="X",c="g",s=25)
        # plot end node
        # ax.scatter(end_coords[1],end_coords[0],marker="X",c="r",s=25)
        ax.scatter(self.itinerary['end_lon'],self.itinerary['end_lat'],marker="X",c="r",s=25)

        # set graph lims
        ax.set_ylim(min_lat-ylim_factor*delta_lat,max_lat+ylim_factor*delta_lat)
        ax.set_xlim(min_lon-xlim_factor*delta_lon,max_lon+xlim_factor*delta_lon)
        return

    def get_itinerary_entry(self):
        """ 
        Returns:
            dict: for an entry of a pandas row, to be consolidated into a df when calling pd.DataFrame.from_records
        """
        # start_lat = self.itinerary['busStart'][0]
        # start_lon = self.itinerary['busStart'][1]
        # end_lat = self.itinerary['workEnd'][0]
        # end_lon = self.itinerary['workEnd'][1]
        if self.error_flag is False:
            start_lat = self.itinerary['start_lat']
            start_lon = self.itinerary['start_lon']
            end_lat = self.itinerary['end_lat']
            end_lon = self.itinerary['end_lon']
            actual_total_duration, actual_bus_duration, total_actual_bus_duration = self.get_non_bus_duration()
            # concatenate actual_bus_duration (list of float) as a string to represent time taken for each bus journey (relevant for itineraries with more than 1 bus journeys)
            actual_bus_duration = ','.join([str(i) for i in actual_bus_duration])
            non_bus_duration = actual_total_duration - total_actual_bus_duration
            routes, routeId, ORIGIN_PT_CODE, DESTINATION_PT_CODE, _ = self.get_itinerary_bus_routes()
            actual_bus_distance = sum([float(busLeg['distance']) for busLeg in self.itinerary['busLegs']])
            total_simulated_bus_distance = total_simulated_bus_duration = 0
            
            # store simulated bus duration as a list to obtain the bus duration for each bus route
            simulated_bus_duration = []
            for r in routes:
                route_length,route_time = self.get_route_time_and_distance(r)
                total_simulated_bus_distance += route_length
                total_simulated_bus_duration += route_time
                simulated_bus_duration.append(route_time)
            # save a list of simulated bus duration as a string to get the breakdown of duration per bus route
            simulated_bus_duration = ','.join([str(d) for d in simulated_bus_duration])
            number_of_busroutes = len(self.itinerary['busLegs']) # includes flooded and non-flooded routes
            simulated_total_duration = non_bus_duration + total_simulated_bus_duration
            return_dict = {'filepath':os.path.basename(self.fp), 'start_lat':start_lat,'start_lon':start_lon,'end_lat':end_lat,'end_lon':end_lon,
                    'duration':self.itinerary['duration'],'transitTime':self.itinerary['transitTime'],
                    'waitingTime':self.itinerary['waitingTime'],'transfers':self.itinerary['transfers'],
                    'actual_bus_duration':actual_bus_duration,'total_actual_bus_duration':total_actual_bus_duration,
                    'simulated_bus_duration':simulated_bus_duration,'total_simulated_bus_duration':total_simulated_bus_duration,
                    'actual_bus_distance':actual_bus_distance,'total_simulated_bus_distance':total_simulated_bus_distance,
                    'actual_total_duration':actual_total_duration,'simulated_total_duration':simulated_total_duration,
                    'non_bus_duration':non_bus_duration,'number_of_busroutes':number_of_busroutes, 
                    'routeId':','.join(routeId),'ORIGIN_PT_CODE':','.join(ORIGIN_PT_CODE),'DESTINATION_PT_CODE':','.join(DESTINATION_PT_CODE)
                    }
            
            return return_dict
        
def itinerary_entry_generator(G_bus,fp_list):
    """
    creates a generator object that yields itinerary entries
    Args:
        G_bus (MultiDiGraph): graph of drive bus network
        fp_list (list): list of filepaths to itinerary json files
    """
    for fp in fp_list:
        TI = TripItinerary.from_file(G_bus, fp=fp)
        if TI.error_flag is False: # yields None if error_flag is True
            try:
                yield TI.get_itinerary_entry()
            except Exception as e:
                # print(f"{fp}: {e}")
                pass    

class BusTrip:
    """
    generates bus_itinerary_df
    """
    def __init__(self, G, fp_list, planningArea):
        """ 
        Args:
            G (MultiDiGraph): graph of drive bus network
            fp_list (list): list of filepaths to itinerary json files e.g. json file that list routes as list of nodes
            planningArea (geopandas.GeoDataFrame): dataframe that shows the planning area that has columns: PLN_AREA_N and REGION_N
        """
        self.G = G
        self.fp_list = fp_list
        self.planningArea = planningArea

    def spatial_join_with_planning_area(self,df,prefix):
        """ 
        Args:
            df (pd.DataFrame): dataframe that shows the coordinates and node IDs of start nodes and end nodes
            prefix (str): prefix to locate and rename columns in the dataframe
        Returns:
            pd.DataFrame: dataframe that shows the coordinates of workplace nodes and nodesId with planning area information
        """
        unique_nodes_gdf = df[[f'{prefix}_nodesID']].drop_duplicates().reset_index(drop=True) # drop=True to avoid adding index column
        # print("Length of unique nodes: ",len(unique_nodes_gdf.index))
        # use coordinates that corresponds to the nodesID in G, so that different latitude and longitude values will not result in the same nodesID
        # do not use the coordinates from df, as they may not correspond to the nodesID in G
        nodes_gdf = ox.graph_to_gdfs(self.G,nodes=True, edges=False)
        nodes_gdf = nodes_gdf[['y','x']].reset_index() # index are osmid
        nodes_gdf = nodes_gdf.rename(columns={"osmid":f"{prefix}_nodesID","y":f"{prefix}_lat","x":f"{prefix}_lon"})
        # filter nodes_gdf to only include nodes that are in unique_nodes_gdf
        nodes_gdf = nodes_gdf[nodes_gdf[f'{prefix}_nodesID'].isin(unique_nodes_gdf[f'{prefix}_nodesID'])]
        # convert into gpd
        nodes_gdf = gpd.GeoDataFrame(
                    nodes_gdf, geometry=gpd.points_from_xy(nodes_gdf[f'{prefix}_lon'], nodes_gdf[f'{prefix}_lat']), crs="EPSG:4326"
                )
        # select only relevant columns
        nodes_gdf = nodes_gdf.sjoin(self.planningArea[['PLN_AREA_N','PLN_AREA_C','REGION_N','REGION_C','geometry']], how="left")
        # print("Length of df: ",len(nodes_gdf.index))
        # rename columns with a prefix of "start_", if columns already have a "start_" prefix, skip renaming
        nodes_gdf = nodes_gdf.rename(columns=lambda x: f"{prefix}_{x}" if not x.startswith(f"{prefix}_") else x)

        # keep column names that contains "PLN_AREA_N" or "REGION_N"
        nodes_gdf = nodes_gdf.loc[:,nodes_gdf.columns.str.contains("nodesID|PLN_AREA|REGION")]

        # merge nodes_gdf with df based on the nodesID, how="inner" preserves only the rows that have matching nodesID in both dataframes
        # validate="many_to_one" check if merge keys are unique in right dataset. if merge keys are not unique in the right dataset, it will throw an error
        # indicator=True adds a column "_merge" to the output DataFrame, which indicates whether each row was found in both DataFrames or only in one of them
        df = df.merge(nodes_gdf, how="inner", left_on=f"{prefix}_nodesID", right_on=f"{prefix}_nodesID",
                      indicator=True, validate="many_to_one")
        return df
    
    def get_itinerary_entry(self,save_fp=None):
        """ 
        generates bus_itinerary_df
        Args:
            save_fp (str or None): if not None, save the bus itinerary df to this filepath
        Returns:
            pd.DataFrame: bus itinerary df
        """
        # concatenate the generator into a single DataFrame
        itinerary_df = pd.DataFrame.from_records(itinerary_entry_generator(self.G,self.fp_list))
        # remove rows with number_of_busroutes == 0
        itinerary_df = itinerary_df[itinerary_df['number_of_busroutes'] > 0]
        # append nodes ID to the start and end coordinates
        itinerary_df['start_nodesID'] = ox.distance.nearest_nodes(self.G,X = itinerary_df['start_lon'], Y = itinerary_df['start_lat'])
        itinerary_df['end_nodesID'] = ox.distance.nearest_nodes(self.G,X = itinerary_df['end_lon'], Y = itinerary_df['end_lat'])
        prev_len = len(itinerary_df)
        # spatial join between start_nodesID with planning area
        itinerary_df = self.spatial_join_with_planning_area(itinerary_df,prefix="start")
        # rename merge column
        itinerary_df = itinerary_df.rename(columns={"_merge":"start_merge"})
        # spatial join between end_nodesID with planning area
        itinerary_df = self.spatial_join_with_planning_area(itinerary_df,prefix="end")
        # rename merge column
        itinerary_df = itinerary_df.rename(columns={"_merge":"end_merge"})
        after_len = len(itinerary_df)
        if prev_len != after_len:
            warnings.warn(f"Length of df before and after spatial join is not the same: {prev_len} vs {after_len}. This may be due to missing planning area information for some nodes.")
        if save_fp is not None:
            # export as csv
            itinerary_df.to_csv(save_fp, index=False)
        return itinerary_df