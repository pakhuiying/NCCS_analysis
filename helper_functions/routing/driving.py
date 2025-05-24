import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import osmnx as ox
import os
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon
import shapely.ops as so
import pandas as pd
import numpy as np

class CarTrip:
    def __init__(self, G,workplace_cluster,planningArea):
        """ 
        Args:
            G (G): driving network
            workplace_cluster (pd.DataFrame): dataframe that shows the coordinates of workplace nodes and nodesId
            planningArea (geopandas.GeoDataFrame): dataframe that shows the planning area that has columns: PLN_AREA_N and REGION_N
        """
        self.G = G
        self.workplace_cluster = workplace_cluster
        self.planningArea = planningArea

    def spatial_join_with_planning_area(self,df,prefix):
        """ 
        Args:
            df (pd.DataFrame): dataframe that shows the coordinates and node IDs of start nodes and end nodes
            prefix (str): prefix to locate and rename columns in the dataframe
        Returns:
            pd.DataFrame: dataframe that shows the coordinates of workplace nodes and nodesId with planning area information
        """
        nodes_gdf = df[[f'{prefix}_nodesID',f'{prefix}_lat',f'{prefix}_lon']].drop_duplicates()
        # print("Length of df: ",len(nodes_gdf.index))
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

        # merge nodes_gdf with df based on the nodesID
        df = df.merge(nodes_gdf, how="left", left_on=f"{prefix}_nodesID", right_on=f"{prefix}_nodesID")
        return df
    
    def get_itinerary_entry(self,cost="travel_time"):
        """ 
        Args:
            cost (str): name of the attribute in edges that define the cost for determining the shortest path
        Returns:
            pd.DataFrame: total travel time (in seconds) from orig to dest along the shortest path 
        """
        # convert nodes into coordinates
        # nodes_gdf represents the coordinates of the start nodes
        nodes_gdf = ox.graph_to_gdfs(self.G,nodes=True, edges=False)
        nodes_gdf = nodes_gdf[['y','x']].reset_index()
        nodes_gdf = nodes_gdf.rename(columns={"osmid":"start_nodesID","y":"start_lat","x":"start_lon"})
        df_list = []
        # simulate vehicle trips from each workplace cluster to all other nodes in the network
        # this is also equivalent to finding the trips from all other nodes to the workplace cluster
        # workplace cluster is the destination (i.e. end node)
        for row_ix,row in self.workplace_cluster.iterrows():
            node_id = row['node_ID']
            end_lat = row['latitude']
            end_lon = row['longitude']
            # PLN_AREA_N = row['PLN_AREA_N']
            # REGION_N = row['REGION_N']
            # returns a dict keyed by target, values are shortest path length from the source to the target
            route_times = nx.shortest_path_length(self.G,source=node_id, target=None, weight=cost)
            df = pd.DataFrame({'start_nodesID': list(route_times),'simulated_total_duration': list(route_times.values())})
            df['end_nodesID'] = node_id
            df['end_lat'] = end_lat
            df['end_lon'] = end_lon
            # df['end_PLN_AREA_N'] = PLN_AREA_N
            # df['end_REGION_N'] = REGION_N
            df_list.append(df)
        itinerary_df = pd.concat(df_list)
        # merge nodes_gdf and itinerary_df
        itinerary_df = itinerary_df.merge(nodes_gdf,how="left",on="start_nodesID")
        # cast as int64
        itinerary_df[['start_nodesID','end_nodesID']] = itinerary_df[['start_nodesID','end_nodesID']].astype(int)

        # spatial join between start_nodesID with planning area
        itinerary_df = self.spatial_join_with_planning_area(itinerary_df,prefix="start")
        # spatial join between end_nodesID with planning area
        itinerary_df = self.spatial_join_with_planning_area(itinerary_df,prefix="end")
        return itinerary_df
    
def get_shortest_path_driving(G, orig, dest=None,
                                cost="travel_time",cmap="plasma",cbar=None,node_size=5,
                                plot = True, ax=None):
    """ 
    Args:
        G (MultiDiGraph): graph of car network
        orig (node or list of node IDs): a node in the car network G
        dest (node or list of node IDs): a node in the car network G. If none, compute shortest paths using all nodes as dest nodes
        cost (str): name of the attribute in edges that define the cost for determining the shortest path
        cmap (str): cmap for colouring the isochrones
        cbar (ScalarMappable or None): if None, use cmap to automatically generate unique colours based on number of nodes. Else, use cbar to map values to colours
        node_size (float): size of notes for plotting
    Returns:
        dict: keys: node IDs, values: total travel time (in seconds) from orig to dest along the shortest path 
    """
    # returns a dict keyed by target, values are shortest path length from the source to the target
    route_times = nx.shortest_path_length(G,source=orig, target=dest, weight=cost)

    if plot:
        # get one color for each isochrone
        if cbar is None:
            iso_colors = ox.plot.get_colors(n=len(route_times), cmap=cmap, start=0)
        else:
            iso_colors = [mpl.colors.rgb2hex(cbar.to_rgba(i),keep_alpha=True) for i in route_times.values()]
        node_colors = {node: nc_ for node, nc_ in zip(list(route_times),iso_colors)}
        nc = [node_colors[node] if node in node_colors else "none" for node in G.nodes() ]
        ns = [node_size if node in node_colors else 0 for node in G.nodes()]
        fig, ax = ox.plot_graph(
            G,
            ax=ax,
            node_color=nc,
            node_size=ns,
            node_alpha=0.8,
            edge_linewidth=0.2,
            edge_color="#999999",
            show = False,
            close = False
        )

    return route_times

def get_nodes_driving_isochrones(G, center_node, trip_times,
                                 cost="travel_time",cmap="plasma",node_size=15,
                                 plot = True, ax = None):
    """ 
    Args:
        G (MultiDiGraph): graph of car network
        center_node (node): a node in the car network G
        trip_times (list of float): thresholds for trip times in seconds
        cost (str): name of the attribute in edges that define the cost for determining the shortest path
        cmap (str): cmap for colouring the isochrones
        node_size (float): size of notes for plotting
    Returns:
        dict: keys: node ID, values: colour
    """
    # get one color for each isochrone
    iso_colors = ox.plot.get_colors(n=len(trip_times), cmap=cmap, start=0)
    # color the nodes according to isochrone then plot the street network
    node_colors = {}
    for trip_time, color in zip(sorted(trip_times, reverse=True), iso_colors):
        # for each travel time, create a sub-graph
        subgraph = nx.ego_graph(G, center_node, radius=trip_time, distance=cost)
        # assign colours to each node within the sub-graph
        for node in subgraph.nodes():
            node_colors[node] = color
    node_colors[center_node] = "red" # mark central node
    nc = [node_colors[node] if node in node_colors else "none" for node in G.nodes()]
    ns = [node_size if node in node_colors else 0 for node in G.nodes()]
    
    if plot:
        fig, ax = ox.plot_graph(
            G,
            ax=ax,
            node_color=nc,
            node_size=ns,
            node_alpha=0.8,
            edge_linewidth=0.2,
            edge_color="#999999",
            show = False,
            close = False
        )
    return node_colors

def get_poly_driving_isochrones(G, center_node, trip_times,
                                cost="travel_time",cmap="plasma",node_size=0,
                                plot = True, ax=None):
    """ 
    Args:
        G (MultiDiGraph): graph of car network
        center_node (node): a node in the car network G
        trip_times (list of float): thresholds for trip times in seconds
        cost (str): name of the attribute in edges that define the cost for determining the shortest path
        cmap (str): cmap for colouring the isochrones
        node_size (float): size of notes for plotting
    Returns:
        gdf: isochrone polygons
    """
    # get one color for each isochrone
    iso_colors = ox.plot.get_colors(n=len(trip_times), cmap=cmap, start=0)
    # make the isochrone polygons
    isochrone_polys = []
    # create a subgraph for each trip time
    for trip_time in sorted(trip_times, reverse=True):
        subgraph = nx.ego_graph(G, center_node, radius=trip_time, distance=cost)
        node_points = [Point((data["x"], data["y"])) for node, data in subgraph.nodes(data=True)]
        bounding_poly = gpd.GeoSeries(node_points).unary_union.convex_hull
        isochrone_polys.append(bounding_poly)
    gdf = gpd.GeoDataFrame(geometry=isochrone_polys)
    if plot:
        # plot the network then add isochrones as colored polygon patches
        fig, ax = ox.plot_graph(
            G, ax=ax, show=False, close=False, edge_color="#000000", edge_alpha=0.2, node_size=node_size
        )
        gdf.plot(ax=ax, color=iso_colors, ec="none", alpha=0.6, zorder=-1)
        
    return gdf

