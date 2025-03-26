import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import osmnx as ox
import os
import pandas as pd
import numpy as np
import copy
import helper_functions.routing.publicTransit as publicTransit
import helper_functions.plot_utils as plot_utils

class UpdateFloodNetwork:
    def __init__(self, G, flooded_maxspeed=20):
        """ Assumes that all flooded roads have a flat reduced speed of flooded_maxspeed
        TODO: change flooded_maxspeed as a function of flood depth
        Args:
            G (G): driving route
            flooded_maxspeed (float): max speed on the road when it is flooded. Default is 20 km/h. This will override the maxspeed attribute in G.
        """
        self.G = G
        self.flooded_maxspeed=flooded_maxspeed

    def identify_flooded_roads(self, historical_floods,rf_value,rf_type='highest 30 min rainfall (mm)', plot = True, ax = None,
                               flooded_edge_color="red",edge_color="white", flooded_edge_linewidth=2):
        """ 
        identify global flooded roads based on historical flooded roads and their associated rainfall value and rainfall type
        Assumes that multiple roads are flooded simultaneously across all Singapore based on historical data
        Args:
            historical_floods (pd.DataFrame): historical floods from 2014 onwards
            rf_value (float): threshold precipitation value to filter the historical_floods df
            rf_type (str): type of ppt e.g. max 30mins, max 60mins
            plot (bool): if plot is True, plot the identified flooded roads
            ax (mpl.Axes): if None, plot on a new figure, else, plot on supplied ax
        Returns:
            list: list of edges representing flooded roads
        """
        hist_flood_filtered = historical_floods[historical_floods[rf_type]<=rf_value]
        flooded_edges = ox.distance.nearest_edges(self.G,X = hist_flood_filtered['longitude'], Y = hist_flood_filtered['latitude'])
        flooded_edges = list(flooded_edges)
        if plot:
            ec = [flooded_edge_color if e in flooded_edges else edge_color for e in self.G.edges(keys=True) ]
            ew = [flooded_edge_linewidth if e in flooded_edges else 0.2 for e in self.G.edges(keys=True) ]
            fig, ax = ox.plot_graph(
                self.G,
                node_size=0,
                edge_color = ec,
                edge_linewidth=ew,
                ax=ax,
                show = False,
                close = False
            )
        return flooded_edges

    def update_flooded_road_network(self, flooded_edges, plot=True,cmap="plasma",**kwargs):
        """ 
        Args:
            flood_edges (list of edges): each item in this list is an edge e.g. (u,v, key)
            plot (bool): If plot is True, visualise the maxspeed on the road
            cmap (str): see mpl colormap_reference
            **kwargs (Any): keyword arguments for ox.plot_graph
        Returns:
            G: graph network with the updated travel time on edges
        """
        G_copy = copy.deepcopy(self.G) # deep copy so it does not change the original
        # convert edges in the graph
        edges = ox.graph_to_gdfs(G_copy, nodes=False)
        # filter edges df by those identified as flooded in flooded_edges, and update the flooded speed
        edges.loc[flooded_edges,'maxspeed'] = self.flooded_maxspeed
        # case maxspeed column to float64
        edges['maxspeed'] = pd.to_numeric(edges.maxspeed, errors='coerce')
        # update maxspeed in G_copy
        for (_, _, _, data),maxspeed in zip(G_copy.edges(data=True, keys=True),edges['maxspeed']):
            data['maxspeed'] = maxspeed
        # impute speed on all edges missing data
        G_copy = ox.add_edge_speeds(G_copy)
        # calculate travel time (seconds) for all edges
        G_copy = ox.add_edge_travel_times(G_copy)
        # if flooded_maxspeed = 0, update travel time as at most 1 hour because that's the longest it takes for flood waters to drain out
        if plot:
            # visualise the maxspeed
            # sort from lowest to highest max speed
            sorted_maxspeed = edges.sort_values('maxspeed')
            # get unique colours based on unique max speed
            iso_colors = ox.plot.get_colors(n=len(sorted_maxspeed.maxspeed.unique()), cmap=cmap, start=0)
            # map colours according to the sorted maxspeed
            speed_color_map = {speed:col for speed,col in zip(sorted_maxspeed.maxspeed.unique(), iso_colors)}
            # map colours according to the G_copy's road's maxspeed
            ec = [speed_color_map[data['maxspeed']] for _, _, _, data in G_copy.edges(data=True, keys=True)]
            # plot the graph with colored edges
            fig, ax = ox.plot_graph(G_copy, edge_color=ec,**kwargs)

        return G_copy
    
    def get_flooded_publicTransit_df(self,publicTransit_fp_list,historical_floods,rf_value,save_fp,error_fp,
                                     rf_type='highest 30 min rainfall (mm)'):
        """ 
        Args:
            publicTransit_fp_list (list of str): list of filepath that has the itinerary information in json
            historical_floods (pd.DataFrame): historical floods from 2014 onwards
            rf_value (float): threshold precipitation value to filter the historical_floods df
            rf_type (str): type of ppt e.g. max 30mins, max 60mins
            save_fp (str): filepath.csv of where to save the itinerary entries 
            error_fp (str): filepath.txt of where to save the error files
        Returns:
            pd.DataFrame: outputs the updated itinerary based on global flooded road conditions
        """
        # identify flooded roads as flooded edges in G
        flooded_edges = self.identify_flooded_roads(historical_floods,rf_value,rf_type=rf_type,plot=False)
        # update travel speed and travel time in G
        G_bus_flooded = self.update_flooded_road_network(flooded_edges,plot=False)

        # rerun bus route time with the updated travel time on flooded roads
        itinerary_entries = []
        for fp in publicTransit_fp_list:
            TI = publicTransit.TripItinerary.from_file(G_bus_flooded, fp=fp)
            try:
                itinerary_entry = TI.get_itinerary_entry()
                itinerary_entries.append(itinerary_entry)
            except:
                if not os.path.exists(error_fp):
                    with open(error_fp, "w") as myfile:
                        myfile.write(f'{fp}\n')

                else:
                    with open(error_fp, "a") as myfile:
                        myfile.write(f'{fp}\n')

        itinerary_df = pd.DataFrame.from_records(itinerary_entries)
        # append notes ID to the start and end coordinates
        itinerary_df['start_nodesID'] = ox.distance.nearest_nodes(G_bus_flooded,X = itinerary_df['start_lon'], Y = itinerary_df['start_lat'])
        itinerary_df['end_nodesID'] = ox.distance.nearest_nodes(G_bus_flooded,X = itinerary_df['end_lon'], Y = itinerary_df['end_lat'])
        # save as csv
        itinerary_df.to_csv(save_fp,index=False)
        return itinerary_df
    

class TravelTimeDelay:
    def __init__(self, flooded_df,dry_df):
        """
        Args:
            flooded_df (pd.DataFrame): dataframe that shows the actual and simulated travel time and distance during dry weather
            dry_df (pd.DataFrame): dataframe that shows the actual and simulated travel time and distance during dry weather
        """
        self.flooded_df = flooded_df
        self.dry_df = dry_df

    def compute_travel_time_delay(self):
        """ 
        Compute travel time delay and append it as a column to flooded_df
        Returns:
            pd.DataFrame: Compute travel time delay and append it as a column to flooded_df
        """
        flooded_df = self.flooded_df.copy()
        flooded_df.set_index(['start_nodesID','end_nodesID'],inplace=True)
        dry_df = self.dry_df.copy()
        dry_df.set_index(['start_nodesID','end_nodesID'],inplace=True)
        travel_time_delay_df = flooded_df.join(dry_df[['simulated_bus_duration']],how='inner',lsuffix='_wet',rsuffix='_dry')
        travel_time_delay_df['travel_time_delay'] = travel_time_delay_df['simulated_bus_duration_wet'] - travel_time_delay_df['simulated_bus_duration_dry']
        return travel_time_delay_df
    
    def get_grouped_travel_time_delay(self, travel_time_delay_df=None):
        """ 
        split df based on end_nodesID
        Args:
            travel_time_delay_df (pd.DataFrame): Compute travel time delay and append it as a column to flooded_df. If None, call compute_travel_time_delay method
        Returns:
            dict: keys are end_nodesID, values are pd.DataFrame
        """
        if travel_time_delay_df is None:
            travel_time_delay_df = self.compute_travel_time_delay()
        
        travel_time_delay_df = travel_time_delay_df.reset_index()
        # split df based on end_nodesID
        return {k:df for k,df in travel_time_delay_df.groupby('end_nodesID')}

    def plot_shortest_path_publicTransit(self,G, itinerary_df,flooded_edges=None,
                                        column_value='travel_time_delay',ax=None,
                                        flooded_edge_color="red",
                                        cmap="plasma",cbar=None,
                                        node_size=5,node_alpha=0.8,
                                        edge_linewidth=0.2,edge_color="#999999"):
        """ 
        plot an isochrone using the simulated total duration
        Args:
            G (MultiDiGraph): graph of drive network
            itinerary_df (pd.DataFrame): df with columns that describes the simulated or actual time
            flooded_edges (list): list of edges in G representing flooded roads
            column_value (str): column in itinerary_df which will determine the plotting of node colors on G
            ax (mpl.Axes): if None, plot on a new figure, else plot on supplied Axes
            cmap (str): cmap for colouring the isochrones
            cbar (ScalarMappable or None): if None, use cmap to automatically generate unique colours based on number of nodes. Else, use cbar to map values to colours
            node_size (float or Iterable): size of nodes for plotting
            node_alpha (float): transparency of nodes
            edge_linewidth (float or Iterable): width of edges for plotting
            edge_color (float or Iterable): colour of edges for plotting
        Returns:
            dict: sorted route times, where key are start_nodesID, and values are route times
        """
        # create a dict where key are start_nodesID, and values are route times
        route_times = itinerary_df[['start_nodesID',column_value]].set_index('start_nodesID')
        # remove route times that are 0, only identify nodes that have non-zero time travel delay
        route_times = route_times[route_times[column_value]>0].to_dict()
        route_times = route_times[column_value]
        # sort dict based on the value i.e. route times
        # keys are start_nodesID
        route_times = {k: v for k, v in sorted(route_times.items(), key=lambda item: item[1])}
        # define colours mapped to route times
        if cbar is None:
            iso_colors = ox.plot.get_colors(n=len(route_times), cmap=cmap, start=0)
        else:
            iso_colors = [mpl.colors.rgb2hex(cbar.to_rgba(i),keep_alpha=True) for i in route_times.values()]
        # map nodes to colours
        node_colors = {node: nc_ for node, nc_ in zip(list(route_times),iso_colors)}
        nc = [node_colors[node] if node in node_colors else "none" for node in G.nodes() ]
        ns = [node_size if node in node_colors else 0 for node in G.nodes()]
        # plot flooded edges, overlay flooded roads
        if flooded_edges is not None:
            edge_color = [flooded_edge_color if e in flooded_edges else "white" for e in G.edges(keys=True) ]
            edge_linewidth = [int(edge_linewidth*10) if e in flooded_edges else edge_linewidth for e in G.edges(keys=True) ]
        fig, ax = ox.plot_graph(
            G,
            ax=ax,
            node_color=nc,
            node_size=ns,
            node_alpha=node_alpha,
            edge_linewidth=edge_linewidth,
            edge_color=edge_color,
            show = False,
            close = False
        )
        return route_times

    def plot_travel_time_delay(self, G, planningArea, workplace_cluster, flooded_edges,
                               cmap="plasma",flooded_edge_color="red",workplace_node_color="red",
                               cbar=None,save_fp=None):
        """ 
        plot gridded isochrones using the simulated total duration
        Args:
            G (MultiDiGraph): graph of car network
            planningArea (gpd.GeoDataFrame): geopandas df of planning areas of SG
            workplace_cluster (pd.DataFrame): df of workplace coords and nodes ID
            flooded_edges (list): list of edges in G representing flooded roads
            cmap (str): cmap for colouring the isochrones
            flooded_edge_color (str): color for showing flooded roads
            workplace_node_color (str): color for showing destination aka workplace node
            cbar (ScalarMappable or None): if None, use cmap to automatically generate unique colours based on number of nodes. Else, use cbar to map values to colours
            save_fp (str): file path to save figure to
            **kwargs:
                node_size (float or Iterable): size of nodes for plotting
                node_alpha (float): transparency of nodes
                edge_linewidth (float or Iterable): width of edges for plotting
                edge_color (float or Iterable): colour of edges for plotting
        Returns:
            dict: sorted route times, where key are start_nodesID, and values are route times
        """
        travel_time_delay_df = self.compute_travel_time_delay()
        # split df based on end_nodesID, where end_nodesID are the work place node id
        itinerary_df_list = self.get_grouped_travel_time_delay(travel_time_delay_df)
        # plot grid
        n_clusters = len(workplace_cluster.index)
        ncols = 3
        nrows = n_clusters//ncols
        if cbar is None:
            # define cbar 
            cbar = plot_utils.get_colorbar(vmin=0,vmax=travel_time_delay_df['travel_time_delay'].max(),cmap=cmap,plot=False)
        # plot
        fig, axes = plt.subplots(nrows, ncols, figsize = (ncols*4,nrows*3))
        for i, ax in enumerate(axes.flatten()):
            # plot planning area boundary
            planningArea.plot(fc='white',ec='k',ax=ax)
            # get attributes
            lat = workplace_cluster.loc[i,"latitude"]
            lon = workplace_cluster.loc[i,"longitude"]
            node_id = workplace_cluster.loc[i,"node_ID"]
            # extract itinerary based on work place node_id
            itinerary_df = itinerary_df_list[node_id]
            # remove itineraries where there are no bus routes
            itinerary_df = itinerary_df[itinerary_df['number_of_busroutes']>0]
            # plot isochrone
            self.plot_shortest_path_publicTransit(G, itinerary_df,flooded_edges,
                                                column_value='travel_time_delay',ax=ax,
                                                cbar=cbar,cmap=cmap,flooded_edge_color=flooded_edge_color
                                                )
            # plot orig node
            ax.scatter(lon,lat,marker="X",c=workplace_node_color,s=25)
        # plt.tight_layout()
        # plot colorbar
        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.15, 0.15, 0.75, 0.01]) # left, bottom, width, height
        fig.colorbar(cbar, cax=cbar_ax, orientation='horizontal', label='Travel time delay (seconds)')
        if save_fp is not None:
            plt.savefig(save_fp, bbox_inches = 'tight')
        plt.show()
        return
    
    def get_planningArea_itinerary(self,planningArea, itinerary_df,colors=None,plot=True):
        """
        spatial joint of planning area and itinerary
        Args:
            planningArea (gpd.GeoDataFrame): geopandas df of planning areas of SG
            itinerary_df (pd.DataFrame): outputs the updated itinerary based on flooded road conditions and travel time delay
            colors (dict): color rgb hex code mapping to the 5 administrative districts
        Returns:
            gpd: spatial joint of planning area and itinerary
        """
        # create point geometry based on coords of nodesID
        nodes_gdf = gpd.GeoDataFrame(
            itinerary_df, geometry=gpd.points_from_xy(itinerary_df.start_lon, itinerary_df.start_lat), crs="EPSG:4326"
        )
        # spatial join to assign node points to REGION_N polygons
        planningArea_nodes = planningArea.sjoin(nodes_gdf,how="inner",predicate='intersects')
        # plot nodes by colors of REGION_N
        if plot:
            planningArea_nodes = planningArea_nodes[~planningArea_nodes['PLN_AREA_N'].str.contains("ISLAND")]
            if colors is None:
                colors = {'EAST REGION':"#dffeb2","WEST REGION": "#ffe7c8","CENTRAL REGION":"#bedcfd",
                    'NORTH REGION':"#e9b3fd",'NORTH-EAST REGION':"#fdb3ba"}
            colormap = pd.DataFrame({'colors':list(colors.values()),'REGION_N':list(colors)})
            planningArea_nodes = planningArea_nodes.merge(colormap,on="REGION_N")
            # plot planning area boundary
            ax = planningArea_nodes.plot(fc="none",ec="k")
            # planningArea_nodes = planningArea_nodes[['colors','start_lat','start_lon']]
            gdf = gpd.GeoDataFrame(
                planningArea_nodes[['colors']], geometry=gpd.points_from_xy(planningArea_nodes.start_lon, planningArea_nodes.start_lat), crs="EPSG:4326"
            )
            # plot node points by color
            gdf.plot(column="colors",color=gdf['colors'],alpha=0.5,ax=ax)
        
        return planningArea_nodes
    
    def get_total_travel_time_delay(self,planningArea):
        """ 
        get total travel time delay by planning area
        Args:
            planningArea (gpd.GeoDataFrame): geopandas df of planning areas of SG
            itinerary_df_list (dict): values are end_nodesID, values are gpd.DataFrame itinerary
        Returns:
            dict: a nested dict where 1st level of keys are end_nodesID, 2nd level of keys are REGION_N (district names), and values are travel time delay
        """
        # remove islands
        planningArea = planningArea[~planningArea['PLN_AREA_N'].str.contains("ISLAND")]
        itinerary_df_list = self.get_grouped_travel_time_delay()
        # iterate through different itineraries
        travelTimeDelay_districts_dict = dict()#{p:[] for p in planningArea['REGION_N'].to_list()}
        for end_nodesID, itinerary_df in itinerary_df_list.items():
            # get total travel time delay by district areas in Singapore (dict)
            travelTimeDelay_districts = self.get_planningArea_itinerary(planningArea, itinerary_df,plot=False)
            # group by REGION_N and sum up total travel time delay by REGION_N
            travelTimeDelay_districts = travelTimeDelay_districts.groupby(['REGION_N'])['travel_time_delay'].sum().to_dict()
            travelTimeDelay_districts_dict[end_nodesID] = travelTimeDelay_districts
        
        return travelTimeDelay_districts_dict

    def plot_total_travel_time_delay(self,planningArea,xlabels=None,colors=None,width=0.5):
        """ 
        Args:
            planningArea (gpd.GeoDataFrame): geopandas df of planning areas of SG
            xlabels (list of str): list of x axis labels
            colors (dict): keys are REGION_N and values are rgb hex codes for each REGION_N
            width (float): width of each bar
        """
        # reorganise dict for plotting
        travelTimeDelay_districts_dict = self.get_total_travel_time_delay(planningArea)
        # x labels are end_nodesID
        if xlabels is None:
            xlabels = [str(i) for i in list(travelTimeDelay_districts_dict)] # list of end_nodesID
        if colors is None:
            colors = {'EAST REGION':"#dffeb2","WEST REGION": "#ffe7c8","CENTRAL REGION":"#bedcfd",
                    'NORTH REGION':"#e9b3fd",'NORTH-EAST REGION':"#fdb3ba"}
        # districts are stacking variable in stacked bar
        plotting_dict = {district: [] for district in list(colors)}
        for nodesID, travel_time_delay_dict in travelTimeDelay_districts_dict.items():
            for district, timeDelay in travel_time_delay_dict.items():
                plotting_dict[district].append(timeDelay)
        # plot
        fig, ax = plt.subplots()
        # start baseline at 0
        bottom = np.zeros(len(xlabels))
        
        # iterate through district, and cummulatively add district values "stacking"
        for district, travel_time_delay_arr in plotting_dict.items():
            p = ax.bar(xlabels,travel_time_delay_arr,width=width,
                    label=district,bottom=bottom,color=colors[district])
            bottom += travel_time_delay_arr
        
        ax.set_title("Total travel time delay from administrative districts to workclusters ID")
        # rotate node ID 45 deg
        ax.set_xticklabels(xlabels,rotation=45,ha='right')
        ax.legend(bbox_to_anchor=(1.05,-0.2),ncols=3)
        plt.show()
        return 