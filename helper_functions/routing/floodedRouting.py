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
import helper_functions.routing.driving as driving
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
            flooded_edges (list of edges): each item in this list is an edge e.g. (u,v, key)
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
    
    def get_flooded_car_df(self,workplace_cluster,flooded_edges,save_fp,cost="travel_time"):
        """ 
        Args:
            workplace_cluster (pd.DataFrame): dataframe that shows the coordinates of workplace nodes and nodesId
            flood_edges (list of edges): each item in this list is an edge e.g. (u,v, key)
            cost (str): name of the attribute in edges that define the cost for determining the shortest path
            save_fp (str): filepath.csv of where to save the itinerary entries 
        Returns:
            pd.DataFrame: outputs the updated itinerary based on global flooded road conditions
        """
        # update travel speed and travel time in G
        G_car_flooded = self.update_flooded_road_network(flooded_edges,plot=False)
        CT = driving.CarTrip(G_car_flooded,workplace_cluster)
        itinerary_df = CT.get_itinerary_entry(cost=cost)
        # save as csv
        itinerary_df.to_csv(save_fp,index=False)
        return itinerary_df
    
    def get_flooded_publicTransit_df(self,publicTransit_fp_list,flooded_edges,
                                     save_fp,error_fp):
        """ 
        Args:
            publicTransit_fp_list (list of str): list of filepath that has the itinerary information in json
            flood_edges (list of edges): each item in this list is an edge e.g. (u,v, key)
            save_fp (str): filepath.csv of where to save the itinerary entries 
            error_fp (str): filepath.txt of where to save the error files
        Returns:
            pd.DataFrame: outputs the updated itinerary based on global flooded road conditions
        """
        # identify flooded roads as flooded edges in G
        # flooded_edges = self.identify_flooded_roads(historical_floods,rf_value,rf_type=rf_type,plot=False)
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
    def __init__(self, flooded_df,dry_df,planningArea,column_value):
        """
        Args:
            flooded_df (pd.DataFrame): dataframe that shows the actual and simulated travel time and distance during dry weather
            dry_df (pd.DataFrame): dataframe that shows the actual and simulated travel time and distance during dry weather
            planningArea (gpd.GeoDataFrame): geopandas df of planning areas of SG
            column_value (str): column name which to compute the travel time delay
        """
        self.flooded_df = flooded_df
        self.dry_df = dry_df
        self.planningArea = planningArea
        self.column_value = column_value

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
        travel_time_delay_df = flooded_df.join(dry_df[[self.column_value]],how='inner',lsuffix='_wet',rsuffix='_dry')
        travel_time_delay_df['travel_time_delay'] = travel_time_delay_df[f'{self.column_value}_wet'] - travel_time_delay_df[f'{self.column_value}_dry']
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
    
    def get_planningArea_itinerary(self, itinerary_df,colors=None,planningArea_column_value="REGION_N",plot=True):
        """
        spatial joint of planning area and itinerary
        Args:
            itinerary_df (pd.DataFrame): an updated itinerary based on flooded road conditions and travel time delay
            colors (dict): color rgb hex code mapping to the 5 administrative districts
            planningArea_column_value (str): column which describes the spatial region of Singapore
        Returns:
            gpd: spatial joint of planning area and itinerary
        """
        # create point geometry based on coords of start_nodesID
        nodes_gdf = gpd.GeoDataFrame(
            itinerary_df, geometry=gpd.points_from_xy(itinerary_df.start_lon, itinerary_df.start_lat), crs="EPSG:4326"
        )
        # spatial join to assign start_nodesID points to shapefile planningArea polygons
        planningArea_nodes = self.planningArea.sjoin(nodes_gdf,how="inner",predicate='intersects')
        # plot nodes by colors of REGION_N
        if plot:
            try:
                planningArea_nodes = planningArea_nodes[~planningArea_nodes['PLN_AREA_N'].str.contains("ISLAND")]
            except:
                pass
            if colors is None:
                colors = {'EAST REGION':"#dffeb2","WEST REGION": "#ffe7c8","CENTRAL REGION":"#bedcfd",
                    'NORTH REGION':"#e9b3fd",'NORTH-EAST REGION':"#fdb3ba"}
            colormap = pd.DataFrame({'colors':list(colors.values()),planningArea_column_value:list(colors)})
            planningArea_nodes = planningArea_nodes.merge(colormap,on=planningArea_column_value)
            # plot planning area boundary
            ax = planningArea_nodes.plot(fc="none",ec="k")
            # planningArea_nodes = planningArea_nodes[['colors','start_lat','start_lon']]
            gdf = gpd.GeoDataFrame(
                planningArea_nodes[['colors']], geometry=gpd.points_from_xy(planningArea_nodes.start_lon, planningArea_nodes.start_lat), crs="EPSG:4326"
            )
            # plot node points by color
            gdf.plot(column="colors",color=gdf['colors'],alpha=0.5,ax=ax)
        
        return planningArea_nodes
    
    def get_total_travel_time_delay(self,planningArea_column_value="REGION_N"):
        """ 
        get total travel time delay by planning area
        Args:
            column_value (str): column name which to compute the travel time delay
            planningArea_column_value (str): column which describes the spatial region of Singapore
        Returns:
            dict: a nested dict where 1st level of keys are end_nodesID, 2nd level of keys are REGION_N (district names), and values are travel time delay
        """
        # remove islands
        # planningArea = self.planningArea[~self.planningArea['PLN_AREA_N'].str.contains("ISLAND")]
        itinerary_df_list = self.get_grouped_travel_time_delay()
        # iterate through different itineraries
        travelTimeDelay_districts_dict = dict()#{p:[] for p in planningArea['REGION_N'].to_list()}
        for end_nodesID, itinerary_df in itinerary_df_list.items():
            # get total travel time delay by district areas in Singapore (dict)
            travelTimeDelay_districts = self.get_planningArea_itinerary(itinerary_df,plot=False)
            # group by REGION_N and sum up total travel time delay by REGION_N
            travelTimeDelay_districts = travelTimeDelay_districts.groupby([planningArea_column_value])['travel_time_delay'].sum().to_dict()
            travelTimeDelay_districts_dict[end_nodesID] = travelTimeDelay_districts
        
        return travelTimeDelay_districts_dict

    def plot_total_travel_time_delay(self,planningArea_column_value="REGION_N",xlabels=None,colors=None,width=0.5,title="",save_fp=None):
        """ plot a horizontal bar chart of total travel time delay per planning area
        Args:
            planningArea_column_value (str): column which describes the spatial region of Singapore
            xlabels (list of str): list of x axis labels
            colors (dict): keys are REGION_N and values are rgb hex codes for each REGION_N
            width (float): width of each bar
            title (str): title for plot
            save_fp (str): file path to save figure to
        """
        # reorganise dict for plotting
        travelTimeDelay_districts_dict = self.get_total_travel_time_delay(planningArea_column_value=planningArea_column_value)
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
            p = ax.barh(xlabels,travel_time_delay_arr,height=width,
                    label=district,left=bottom,color=colors[district])
            bottom += travel_time_delay_arr
        
        ax.set_title("Total travel time delay from administrative districts to workclusters ID")
        # rotate time 45 deg
        # ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha='right')
        # set y axis title
        ax.set_ylabel("Workclusters ID")
        ax.set_xlabel("Total travel time delay (s)")
        ax.set_title(title)
        ax.invert_yaxis() # labels read top-to-bottom
        ax.legend(loc='lower center',bbox_to_anchor=(0.5,-0.3),ncols=3)
        if save_fp is not None:
            plt.savefig(save_fp, bbox_inches = 'tight')
        plt.show()
        return 
    
    def get_stats_travel_time_delay(self,itinerary_df=None,planningArea_column_value="REGION_N"):
        """ 
        Args:
            itinerary_df (pd.DataFrame): spatial joint of planning area and itinerary_df (output from get_planningArea_itinerary)
            planningArea_column_value (str): column which describes the spatial region of Singapore
        Returns:
            dict: 1st level keys are end_nodesID, 2nd level keys are planningArea_column_value, 
                3rd level keys are summary_travel_time_delay (summary of travel time delay) and summary_buses_delayed (keys are bus services and values are count of that affected bus service)
        """
        if itinerary_df is None:
            # compute travel time delay
            travel_time_delay_df = self.compute_travel_time_delay()
            # spatial joint of start_nodesID to their associated planningArea polygons
            itinerary_df = self.get_planningArea_itinerary(travel_time_delay_df,colors=None,planningArea_column_value=planningArea_column_value,plot=False)
        
        itinerary_df = itinerary_df.reset_index()
        grouped_stats = {end_nodesID: {region: region_df for region, region_df in end_nodesID_df.groupby(planningArea_column_value)} 
        for end_nodesID, end_nodesID_df in itinerary_df.groupby('end_nodesID')}
        
        summary_stats = dict()
        for end_nodesID, end_nodesID_df in grouped_stats.items():
            region_stats = dict()
            for region, region_df in end_nodesID_df.items():
                df = region_df[['start_nodesID','end_nodesID','travel_time_delay','routeId',planningArea_column_value]]
                # only filter bus routes where travel time delay is experienced. there could be some routes where there are no time delay
                df = df[df['travel_time_delay']>0]
                travel_time_delay_stats = df['travel_time_delay'].describe().to_dict()
                busServices = [i.split(',') if isinstance(i,str) else i for i in df['routeId'].to_list() ]
                buses = []
                for b in busServices:
                    if isinstance(b,list):
                        for i in b:
                            buses.append(i)
                    else:
                        buses.append(str(b))
                buses_delayed, buses_count = np.unique(buses,return_counts=True)
                summary_buses_delayed = {str(k):v for k,v in zip(buses_delayed,buses_count)}
                region_stats[region] = {'summary_travel_time_delay': travel_time_delay_stats,
                                        'summary_buses_delayed': summary_buses_delayed}
            summary_stats[end_nodesID] = region_stats

        return summary_stats
