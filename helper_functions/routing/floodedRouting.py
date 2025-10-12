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
import re
import warnings
import helper_functions.routing.publicTransit as publicTransit
import helper_functions.routing.driving as driving
import helper_functions.plot_utils as plot_utils

class UpdateFloodNetwork:
    def __init__(self, G, flooded_maxspeed=None, percentage_reduction_maxspeed=10):
        """
        Either use flooded_maxspeed or percentage_reduction_maxspeed to update flood network, one of the params must be None
        TODO: change flooded_maxspeed as a function of flood depth
        Args:
            G (G): driving route
            flooded_maxspeed (float): max speed on the road when it is flooded. Default is 20 km/h. This will override the maxspeed attribute in G.
            percentage_reduction_maxspeed (float, 0 to 100): percentage reduction from max speed allowed on roads
        """
        self.G = G
        self.flooded_maxspeed=flooded_maxspeed
        self.percentage_reduction_maxspeed = percentage_reduction_maxspeed
        # check if either one of the params is None, this is to make sure one param doesnt override the other param
        assert not(all([flooded_maxspeed != None, percentage_reduction_maxspeed != None])), "either flooded_maxspeed or percentage_reduction_maxspeed must be None"
        # to make sure both params are not None
        assert not(all([flooded_maxspeed == None, percentage_reduction_maxspeed == None])), "both flooded_maxspeed and percentage_reduction_maxspeed must not be None"

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

    def get_coastal_flood_coordinates(self,flood_arr,geotransform,flood_depth_thresh=0.15):
        """ Get indices of array where flood depth exceeds 0.15 m
        Args:
            flood_arr (np.ndarray): flood depth array in metres
            geotransform (tuple of float): GetGeoTransform() function returns tuple with 6 values (coordinates of origin, resolution, angle),
            flood_depth_thresh (np.ndarray): threshold (in metres) on the flood_arr, above this threshold, fetch the corresponding coordinates
        Returns:
            list of tuple: list of coordinates (lat,lon)
        """
        flood_idx = np.argwhere(flood_arr > flood_depth_thresh)
        def get_coords(dx,dy):
            """" 
            Args:
                dx (int): column pixel from the origin (upper left corner)
                dy (int): row pixel from the origin (upper left corner)
            """
            # origin
            px = geotransform[0]
            py = geotransform[3]
            # pixel size
            rx = geotransform[1]
            ry = geotransform[5]
            x = dx*rx + px
            y = dy*ry + py
            return y,x

        return [get_coords(idx[1],idx[0]) for idx in flood_idx]

    def identify_coastal_flooded_roads(self,flooded_coordinates, plot=True, ax = None,
                               flooded_edge_color="red",edge_color="white", flooded_edge_linewidth=2):
        """ 
        Add attribute traffic_flow to G. 
        Traffic flow Returns hourly average traffic flow, taken from a representative month of every quarter during 0700-0900 hours.
        Args:
            G (G): driving route
            flooded_coordinates (list of float tuple): [(lat,lon),(lat,lon)]
            plot (bool): if True, plot traffic volume
            ax (mpl.Axes): if None, plot on a new figure, else, plot on supplied ax
        Returns:
            list: list of edges representing flooded roads
        """
        G_copy = copy.deepcopy(self.G) # deep copy so it does not change the original
        # get nearest edges to flooded edges
        flooded_edges = ox.distance.nearest_edges(G_copy,X = [coord[1] for coord in flooded_coordinates], Y = [coord[0] for coord in flooded_coordinates])
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
        if self.flooded_maxspeed is not None:
            # filter edges df by those identified as flooded in flooded_edges, and update the flooded speed
            # replace flooded speed with a flat value
            edges.loc[flooded_edges,'maxspeed'] = self.flooded_maxspeed
        if self.percentage_reduction_maxspeed is not None:
            # filter edges df by those identified as flooded in flooded_edges, and update the flooded speed
            # reduce speed by percentage reduction from the max speed
            speed_reduc_factor = 1 - self.percentage_reduction_maxspeed/100
            edges.loc[flooded_edges,'maxspeed'] = pd.to_numeric(edges.loc[flooded_edges,'maxspeed'],errors='coerce')*speed_reduc_factor

        # case maxspeed column to float64
        edges['maxspeed'] = pd.to_numeric(edges.maxspeed, errors='coerce')
        # check if edges maxspeed has any NAs
        if edges['maxspeed'].isna().any() is np.True_:
            warnings.warn("Edges' maxspeed has NAs")
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
    
    def get_flooded_car_df(self,workplace_cluster,planningArea,flooded_edges,save_fp,cost="travel_time"):
        """ 
        Args:
            workplace_cluster (pd.DataFrame): dataframe that shows the coordinates of workplace nodes and nodesId
            planningArea (gpd.GeoDataFrame): geopandas df of planning areas of SG
            flood_edges (list of edges): each item in this list is an edge e.g. (u,v, key)
            cost (str): name of the attribute in edges that define the cost for determining the shortest path
            save_fp (str): filepath.csv of where to save the itinerary entries 
        Returns:
            pd.DataFrame: outputs the updated itinerary based on global flooded road conditions
        """
        # update travel speed and travel time in G due to flooded roads, and rerun the shortest path algorithm for vehicle routing
        G_car_flooded = self.update_flooded_road_network(flooded_edges,plot=False)
        CT = driving.CarTrip(G_car_flooded,workplace_cluster,planningArea)
        itinerary_df = CT.get_itinerary_entry(cost=cost)
        # save as csv
        itinerary_df.to_csv(save_fp,index=False)
        return itinerary_df
    
    def get_flooded_publicTransit_df(self,publicTransit_fp_list,planningArea,flooded_edges,
                                     save_fp):
        """ 
        Args:
            publicTransit_fp_list (list of str): list of filepath that has the itinerary information in json
            planningArea (gpd.GeoDataFrame): geopandas df of planning areas of SG
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

        # rerun the public transit routing with the updated travel time on flooded roads
        # itinerary_df = pd.DataFrame.from_records(publicTransit.itinerary_entry_generator(G_bus_flooded,publicTransit_fp_list))
        # # remove rows with number_of_busroutes == 0
        # itinerary_df = itinerary_df[itinerary_df['number_of_busroutes'] > 0]
        # # rerun bus route time with the updated travel time on flooded roads
        # itinerary_entries = []
        # for fp in publicTransit_fp_list:
        #     TI = publicTransit.TripItinerary.from_file(G_bus_flooded, fp=fp)
        #     try:
        #         itinerary_entry = TI.get_itinerary_entry()
        #         itinerary_entries.append(itinerary_entry)
        #     except:
        #         if not os.path.exists(error_fp):
        #             with open(error_fp, "w") as myfile:
        #                 myfile.write(f'{fp}\n')

        #         else:
        #             with open(error_fp, "a") as myfile:
        #                 myfile.write(f'{fp}\n')

        # itinerary_df = pd.DataFrame.from_records(itinerary_entries)
        # append notes ID to the start and end coordinates
        # itinerary_df['start_nodesID'] = ox.distance.nearest_nodes(G_bus_flooded,X = itinerary_df['start_lon'], Y = itinerary_df['start_lat'])
        # itinerary_df['end_nodesID'] = ox.distance.nearest_nodes(G_bus_flooded,X = itinerary_df['end_lon'], Y = itinerary_df['end_lat'])
        BT = publicTransit.BusTrip(G_bus_flooded, publicTransit_fp_list, planningArea)
        # save as csv
        itinerary_df = BT.get_itinerary_entry(save_fp)
        
        return itinerary_df
    

class TravelTimeDelay:
    def __init__(self, flooded_df,dry_df,planningArea,column_value,end_groupby="end_PLN_AREA_N",start_groupby="start_REGION_N"):
        """
        Args:
            flooded_df (pd.DataFrame): dataframe that shows the actual and simulated travel time and distance during dry weather
            dry_df (pd.DataFrame): dataframe that shows the actual and simulated travel time and distance during dry weather
            planningArea (gpd.GeoDataFrame): geopandas df of planning areas of SG
            column_value (str): column name which to compute the travel time delay
            end_groupby (str): column which describes the spatial region of Singapore that itinerary ends at
            start_groupby (str): column which describes the spatial region of Singapore that itinerary starts from
        """
        self.flooded_df = flooded_df
        self.dry_df = dry_df
        self.planningArea = planningArea
        self.column_value = column_value
        self.start_groupby = start_groupby
        self.end_groupby = end_groupby
        if len(self.flooded_df.index) != len(self.dry_df.index):
            warnings.warn(f"flooded_df {(len(self.flooded_df.index))} do not have the same length as dry_df ({len(self.dry_df.index)})!")

    def compute_travel_time_delay(self):
        """ 
        Compute travel time delay and append it as a column to flooded_df
        Returns:
            pd.DataFrame: Compute travel time delay and append it as a column to flooded_df
        """
        # check if both column values are same along rows (i.e. on axis=1), and then check if there are any False values
        check_matching_keys = self.flooded_df[["start_nodesID","end_nodesID"]].eq(self.dry_df[["start_nodesID","end_nodesID"]]).all(axis=1).all()
        if (check_matching_keys is np.True_):
            travel_time_delay_df = self.flooded_df.copy()
            travel_time_delay_df = travel_time_delay_df.rename(columns={self.column_value: f'{self.column_value}_wet'})
            # direct assigning is used for public transit because 'start_nodesID','end_nodesID' are not unique keys, and merging will not be 1:1
            travel_time_delay_df[f'{self.column_value}_dry'] = self.dry_df[self.column_value]
        else:
            flooded_df = self.flooded_df.copy()
            flooded_df.set_index(['start_nodesID','end_nodesID'],inplace=True)
            dry_df = self.dry_df.copy()
            dry_df.set_index(['start_nodesID','end_nodesID'],inplace=True)
            # if the indices are all unique, only allow for 1:1 join
            travel_time_delay_df = flooded_df.join(dry_df[[self.column_value]],how='inner',lsuffix='_wet',rsuffix='_dry',validate="1:1")
        # compute travel time delay column
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
        # return {k:df for k,df in travel_time_delay_df.groupby('end_nodesID')}
        return {k:df for k,df in travel_time_delay_df.groupby(self.end_groupby)}
    
    def get_planningArea_itinerary(self, itinerary_df,colors=None,plot=True):
        """
        spatial joint of planning area and itinerary, which assigns the start_nodesID to their associated planningArea polygons
        Args:
            itinerary_df (pd.DataFrame): an updated itinerary based on flooded road conditions and travel time delay
            colors (dict): color rgb hex code mapping to the 5 administrative districts
            start_groupby (str): column which describes the spatial region of Singapore
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
            colormap = pd.DataFrame({'colors':list(colors.values()),self.start_groupby:list(colors)})
            planningArea_nodes = planningArea_nodes.merge(colormap,on=self.start_groupby)
            # plot planning area boundary
            ax = planningArea_nodes.plot(fc="none",ec="k")
            # planningArea_nodes = planningArea_nodes[['colors','start_lat','start_lon']]
            gdf = gpd.GeoDataFrame(
                planningArea_nodes[['colors']], geometry=gpd.points_from_xy(planningArea_nodes.start_lon, planningArea_nodes.start_lat), crs="EPSG:4326"
            )
            # plot node points by color
            gdf.plot(column="colors",color=gdf['colors'],alpha=0.5,ax=ax)
        
        return planningArea_nodes
    
    def get_stats_travel_time_delay(self,itinerary_df=None,save_fp=None):
        """ 
        Args:
            itinerary_df (pd.DataFrame): output from compute_travel_time_delay()
            save_fp (str): file path to save csv to
        Returns:
            dict: keys are a tuple of (end_groupby, start_groupby), corresponding to planning area and region 
                values are descriptive statistics of travel time delay
            pd.DataFrame: travelTimeDelay_districts_df that summarises the statistics of travel time delay by planning area and region.
        """
        if itinerary_df is None:
            # compute travel time delay
            itinerary_df = self.compute_travel_time_delay()
        
        # aggregate travel time delay by plannning area and region and calculate count, sum, mean as separate columns
        def summarise_func(row):
            """ custom function to summarise travel time delay per df row"""
            # initialise dictionary
            d = row['travel_time_delay'].describe().to_dict()
            d['sum'] = row['travel_time_delay'].sum()
            # get routes where travel time delay is greater than 0
            rows_delay = row['travel_time_delay'][row['travel_time_delay'] > 0]
            # number of paths with travel time delay
            rows_delay_describe = rows_delay.describe().to_dict()
            for k,v in rows_delay_describe.items():
                d[k+"_delay"] = v
            # compute IQR for those routes with travel time delay
            IQR = d['75%_delay'] - d['25%_delay']
            d['whislow_delay'] = d['25%_delay'] - 1.5*IQR
            d['whishigh_delay'] = d['75%_delay'] + 1.5*IQR
            # get count of delayed buses to identify how many routes and bus services are affected as additional column
            if 'routeId' in row:
                # remove NA 
                buses = [j for i in row['routeId'] for j in str(i).split(',') if j != "nan"]
                # get count of delayed buses to identify how many routes and bus services are affected
                buses_delayed, buses_count = np.unique(buses,return_counts=True)
                summary_buses_delayed = [f"{str(k)}:{v}" for k,v in zip(buses_delayed,buses_count)]
                d['buses_delayed_id_count'] = ','.join(summary_buses_delayed)

            return pd.Series(d,index=list(d))
        
        # group by end_groupby and start_groupby
        travelTimeDelay_districts_df = itinerary_df.groupby([self.end_groupby,self.start_groupby]).apply(lambda x: summarise_func(x))
        
        travelTimeDelay_districts_df = travelTimeDelay_districts_df.reset_index()
        # save as csv is fp is not None
        if save_fp is not None:
            travelTimeDelay_districts_df.to_csv(save_fp,index=False)
        # convert to dict
        return travelTimeDelay_districts_df
    
    def plot_total_travel_time_delay(self,travelTimeDelay_districts_df = None,stats_param='sum',
                                    selected_planningArea=['TAMPINES','JURONG EAST','WOODLANDS','DOWNTOWN CORE','SELETAR'],
                                    figsize=(6, 9),bbox_to_anchor=(0.4,-0.15),
                                    xlabels=None,colors=None,width=0.6,title="",ax=None,save_fp=None):
        """ plot a horizontal bar chart of total travel time delay per planning area
        Args:
            travelTimeDelay_districts_df (pd.DataFrame): dataframe that summarises the statistics of travel time delay by planning area and region. If None, calculate it using internal method.
            stats_param (str): column name in the stats_travel_time_delay that is used for plotting the sum
            selected_planningArea (list): if None, plot all planning area. Else, list of planning areas to be plotted
            figsize (tuple): figsize for matplotlib figure
            bbox_to_anchor (tuple): matplotlib param for placing legends
            xlabels (list of str): list of x axis labels
            colors (dict): keys are REGION_N and values are rgb hex codes for each REGION_N
            width (float): width of each bar
            title (str): title for plot
            ax (mpl.Axes): if None, plot on a new figure, else, plot on supplied ax
            save_fp (str): file path to save figure to
        """
        # get stats of travel time delay
        if travelTimeDelay_districts_df is None:
            travelTimeDelay_districts_df = self.get_stats_travel_time_delay()
        if selected_planningArea is not None:
            # filter dataframe for selected planning areas
            travelTimeDelay_districts_df = travelTimeDelay_districts_df[travelTimeDelay_districts_df[self.end_groupby].isin(selected_planningArea)]
        # convert df to dict
        travelTimeDelay_districts_dict = travelTimeDelay_districts_df.set_index([self.end_groupby,self.start_groupby]).to_dict(orient='index')
        # reorganise dict for plotting
        # first level of keys are region names, second level of keys are planning area names
        # plotting_dict = dict()
        # for (pln,region) in travelTimeDelay_districts_dict.keys():
        #     plotting_dict[region] = dict()
        # for (pln,region),stats in travelTimeDelay_districts_dict.items():
        #     plotting_dict[region][pln] = stats['sum']
        if selected_planningArea is None:
            # plot all planning areas
            PLN_AREA_N = [p for p in list(sorted(self.planningArea['PLN_AREA_N'].unique())) if 'ISLAND' not in p]
            REGION_N = list(sorted(self.planningArea['REGION_N'].unique()))
        else:
            # plot for selected areas only
            PLN_AREA_N = list(set([pln for (pln,_) in travelTimeDelay_districts_dict.keys()]))
            REGION_N = list(set([region for (_,region) in travelTimeDelay_districts_dict.keys()]))
        # reorganise dict for plotting
        # first level of keys are region names, second level of keys are planning area names
        plotting_dict = {region: {pln: 0 for pln in PLN_AREA_N} for region in REGION_N }
        # iterate across planning area and region
        for (pln, region), stats in travelTimeDelay_districts_dict.items():
            plotting_dict[region][pln] = stats[stats_param]
        xlabels = list(plotting_dict[list(plotting_dict)[0]]) # planning area names
        # start baseline at 0
        bottom = np.zeros(len(xlabels))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # assign colors to each region
        colors = {'EAST REGION':"#dffeb2","WEST REGION": "#ffe7c8","CENTRAL REGION":"#bedcfd",
                    'NORTH REGION':"#e9b3fd",'NORTH-EAST REGION':"#fdb3ba"}
        # iterate through region, and cummulatively "stacking" planning area values region by region
        for region,pln_dict in plotting_dict.items():
            pln_time_delay = list(pln_dict.values())
            p = ax.barh(xlabels,pln_time_delay,height=width,
                label=region,left=bottom,color=colors[region])
            bottom += pln_time_delay
        
        ax.set_ylabel("Planning area")
        ax.set_xlabel("Total travel time delay (s)")
        ax.set_title(title)
        ax.invert_yaxis() # labels read top-to-bottom
        ax.legend(loc='lower center',bbox_to_anchor=bbox_to_anchor,ncols=3)
        if save_fp is not None:
            plt.savefig(save_fp, bbox_inches = 'tight')
        if ax is None:
            plt.show()
        return plotting_dict
    
    def get_potential_car_time_delay(self,trafficVol,travelTimeDelay_districts_df = None):
        """   get potential total travel time delay based on the road with the highest max traffic volume
        calculates the total travel time delay for cars travelling from region to planning area
        Args:
            trafficVol (pd.DataFrame): traffic volume from region to planning area
            travelTimeDelay_districts_df (pd.DataFrame): dataframe that summarises the statistics of travel time delay by planning area and region. If None, calculate it using internal method.
        """
        if travelTimeDelay_districts_df is None:
            # get stats of travel time delay
            travelTimeDelay_districts_df = self.get_stats_travel_time_delay()
        columns_select = [i for i in trafficVol.columns if bool(re.search("^max_traffic.*REGION$",i))]
        # rename columns
        trafficVol_long = trafficVol[['PLN_AREA_N']+columns_select].rename(columns = lambda x: x.replace("max_traffic_",""))
        # cast table from wide to long
        trafficVol_long = pd.melt(trafficVol_long,id_vars=['PLN_AREA_N'],var_name="REGION_N", value_name="max_traffic_vol")
        # merge traffic vol df with travel time delay stats df
        trafficVol_stats = pd.merge(left=trafficVol_long,right=travelTimeDelay_districts_df,
                                    how="inner",left_on=['PLN_AREA_N','REGION_N'], right_on=[self.end_groupby,self.start_groupby])
        
        stats_col = ['mean_delay','min_delay', '25%_delay', '50%_delay', '75%_delay','max_delay']
        for col in stats_col:
            # get probability of routes affected
            prop_routes_affected = trafficVol_stats['count_delay']/trafficVol_stats['count']
            # num of cars affected
            max_traffic_vol = trafficVol_stats['max_traffic_vol']
            trafficVol_stats[f"potential_total_{col}"] = max_traffic_vol*prop_routes_affected*trafficVol_stats[col]

        return trafficVol_stats
    
    def get_publicTransit_volume(self,delay_df,spatialTravelPatterns):
        """ Summarise the delay by planning area and region by aggregating all travel time delay from public transit
        Args:
            delay_df (pd.DataFrame): df that calculates entry-wise total travel time delay from affected bus routes and the corresponding OD bus trips
            spatialTravelPatterns (pd.DataFrame): df that describes the number of commuters travelling from region to planning area via transport modes
        Returns:
            pd.DataFrame: aggregated total travel time for each planning area and region, scaled back by the spatial travel patterns, because the OD trip volume double counts individuals
        """
        time_delay_columns = [c for c in delay_df.columns if re.search("^potential.*|.*traffic_vol|simulated_bus_delay",c)]
        stats_c = ['mean','min', '25%', '50%', '75%','max']
        # sum all travl time delay by end_PLN_AREA_N and start_REGION_N
        delay_df = delay_df.groupby([self.end_groupby,self.start_groupby])[time_delay_columns].sum().reset_index()
        # merge based on end_PLN_AREA_N and start_REGION_N
        delay_df = delay_df.merge(spatialTravelPatterns[[self.end_groupby,self.start_groupby,'Combinations of MRT/LRT or Public Bus']],how="inner")
        # it is likely that trips from OD bus stop IDs are doublecounting the actual trips taken by individuals travelling from O to D
        # because e.g. there are multiple entries from the same O with different D, and double count the same individuals who boarded at O
        # to account for the double counting, we need to scale back the number of trips based on the actual spatial travel patterns
        def scale_travel_time(row,stats_c):
            d = dict()
            for c in stats_c:
                # the problem with this is that if spatialtravelpatterns > traffic_vol, 
                # spatialTravelPatterns/trafficVol>1, total travel time of a lower percentile will have higher total travel time 
                # esp if travel time delay is the same or similar with travel delay from a higher percentle
                # so this method can only be applied if traffic volume is constant, suggest to use max_traffic_vol to scale back
                # which is to scale back entries with unreasonably high traffic vol e.g. max traffic vol
                d[f"potential_total_{c}_delay"] = (row['Combinations of MRT/LRT or Public Bus']/row[f"max_traffic_vol"])*row[f"potential_total_{c}_delay"]

            return pd.Series(d,index=list(d))
        # overwrite the potential total* delay columns
        delay_df[[f"potential_total_{c}_delay" for c in stats_c]] = delay_df.apply(lambda x: scale_travel_time(x,stats_c),axis=1)
        return delay_df
    
    def get_potential_publicTransit_time_delay(self,OD_tripVol, spatialTravelPatterns,travel_time_delay_df=None):
        """   get potential total travel time delay based on the commuter trip volume
        Args:
            OD_tripVol (pd.DataFrame): hourly average trip volume from origin to destination pt codes
            spatialTravelPatterns (pd.DataFrame): df that describes the number of commuters travelling from region to planning area via transport modes
            travel_time_delay_df (pd.DataFrame): travel time delay_df from calling method compute_travel_time_delay()
        """
        if travel_time_delay_df is None:
            # get stats of travel time delay
            travel_time_delay_df = self.compute_travel_time_delay()
        delay_index = travel_time_delay_df[travel_time_delay_df['travel_time_delay']>0].index
        # filter rows to get rows for delayed only
        delayed_df_dict = {df_name: df.loc[delay_index,:] for df, df_name in zip([self.flooded_df, self.dry_df],['flood','dry'])}
        # convert bus duration to numpy array, and calculate travel time delay
        delay_arrays = {df_name: df['simulated_bus_duration'].str.split(',', expand=True).astype(float).to_numpy() for df_name, df in delayed_df_dict.items()}
        # calculate travel time delay (flood will have longer travel time)
        delay_arr = delay_arrays['flood'] - delay_arrays['dry']
        # convert bus codes to arrays
        origin_arr = delayed_df_dict['flood']['ORIGIN_PT_CODE'].str.split(',',expand=True).values
        destination_arr = delayed_df_dict['flood']['DESTINATION_PT_CODE'].str.split(',',expand=True).values
        assert delay_arr.shape==origin_arr.shape==destination_arr.shape, "array shapes of bus routes number and bus stop ids must be the same"
        
        # get index where travel time delay for each individual bus route is > 0
        # use nonzero such that it can be used to index an array, argwhere output cannot be used for indexing
        # identifies which bus route experiences travel time delay and return the corresponding index
        # delay row index corresponds to the row index of the delayed_df
        delay_row_idx, delay_col_idx = np.nonzero(delay_arr>0)#np.argwhere(delay_arr>0)
        # identify the OD bus stop ID for the bus route that experiences delay
        origin_delay_id = origin_arr[delay_row_idx, delay_col_idx]
        destination_delay_id = destination_arr[delay_row_idx, delay_col_idx]
        # identify the travel time delay where values > 0
        delay_arr = delay_arr[delay_row_idx, delay_col_idx]
        assert origin_delay_id.shape == destination_delay_id.shape == delay_arr.shape, "array shapes of origin and destination bus stop ids must be the same"
        # create a df to merge with OD df later
        tripVol = pd.DataFrame({'delay_row_index': delay_row_idx,'simulated_bus_delay':delay_arr,
                                'ORIGIN_PT_CODE':origin_delay_id,'DESTINATION_PT_CODE':destination_delay_id})
        # merge with OD_hourly avg based on ORIGIN and DESTINATION pt code columns
        tripVol = tripVol.merge(OD_tripVol,how="inner") # possible that some rows will not have a corresponding OD trip vol and thus dropped out
        assert len(tripVol) > 0, "tripVol length is 0 after merging. Likely that there are no common rows between tripVol and OD_tripVol"
        stats_c = ['mean','min', '25%', '50%', '75%','max'] # column names that have the trip volume values
        for c in stats_c:
            tripVol[f"{c}_traffic_vol"] = tripVol[c]
            tripVol[f"potential_total_{c}_delay"] = tripVol['simulated_bus_delay']*tripVol[c]
        # group by delay row index to get the sum of total delay for each trip
        tripVol = tripVol.groupby('delay_row_index').sum(numeric_only=True).reset_index().set_index('delay_row_index')
        # merge it with travel time delay
        delay_df = travel_time_delay_df.loc[delay_index,:].reset_index()
        # # include simulated bus delay as a sanity check to see if it matches with travel time delay column
        columns_select = [c for c in tripVol.columns if bool(re.search("^potential.*|.*_traffic_vol|simulated_bus_delay",c))]
        # join by index (inner join by default)
        delay_df = pd.merge(delay_df,tripVol[columns_select], left_index=True,right_index=True)
        assert len(delay_df) > 0, "delay_df length is 0 after merging with tripVol"
        # a sanity check would be to call delay_df[~delay_df['travel_time_delay'].ge(delay_df['simulated_bus_delay'])]
        # to check if travel time delay column matches with simulated bus delay columns
        # scale the calculated total travel time by the actual number of spatial travel patterns from region to planning area
        delay_df = self.get_publicTransit_volume(delay_df,spatialTravelPatterns)
        return delay_df

    def plot_travel_time_delay_stats(self, travelTimeDelay_districts_df = None,
                                    selected_planningArea=['TAMPINES','JURONG EAST','WOODLANDS','DOWNTOWN CORE','SELETAR'],
                                    figsize=(5*4,4),
                                    title = "",
                                    showmeans=True,
                                    showwhiskers=True,
                                    showfliers=False,
                                    showcaps=False,
                                    showbox=True,
                                    save_fp=None):
        """ 
        plot modified boxplot for selected planning areas for routes that experienced travel time delay
        Args:
            travelTimeDelay_districts_df (pd.DataFrame): dataframe that summarises the statistics of travel time delay by planning area and region. If None, calculate it using internal method.
            selected_planningArea (list): if None, plot all planning areas, else, list of planning areas to be plotted
            figsize (tuple): figsize for matplotlib figure
            title (str): title of plot
            showmeans (bool): if True, show mean in the boxplot
            showmedians (bool): if True, show median in the boxplot
            showwhiskers (bool): if True, show whiskers in the boxplot
            showfliers (bool): if True, show outliers in the boxplot
            showcaps (bool): if True, show caps in the boxplot
            showbox (bool): if True, show box in the boxplot
            save_fp (str): file path to save figure to
        """
        if travelTimeDelay_districts_df is None:
            # get stats of travel time delay
            travelTimeDelay_districts_df = self.get_stats_travel_time_delay()
        # assign colors to each region
        colors = {'EAST REGION':"#dffeb2","WEST REGION": "#ffe7c8","CENTRAL REGION":"#bedcfd",
                    'NORTH REGION':"#e9b3fd",'NORTH-EAST REGION':"#fdb3ba"}
        if selected_planningArea is not None:
            # filter dataframe for selected planning areas
            travelTimeDelay_districts_df = travelTimeDelay_districts_df[travelTimeDelay_districts_df[self.end_groupby].isin(selected_planningArea)]
        
        # reorganise dict for plotting
        regions =  travelTimeDelay_districts_df[self.start_groupby].unique()
        # create a dictionary to store the plotting data
        plotting_dict = {region: [] for region in regions}
        travelTimeDelay_districts_dict = travelTimeDelay_districts_df.set_index([self.end_groupby,self.start_groupby]).to_dict(orient='index')

        for (pln,region),stats in travelTimeDelay_districts_dict.items():
            med = stats['50%_delay']
            q1 = stats['25%_delay']
            q3 = stats['75%_delay']
            # modify whisker to be min and max instead of 1.5*IQR
            if showwhiskers:
                whislo = stats['min_delay']
                whishi = stats['max_delay']
            else:
                # if show whiskers is false, show std dev around the mean
                whislo = q1
                whishi = q3
            mean = stats['mean_delay']
            d = {'med': med, 'q1': q1, 'q3': q3, 'whislo': whislo, 'whishi': whishi, 'mean': mean,
                'label': pln, 'color': colors[region]}
            plotting_dict[region].append(d)

        # modify axis title
        ax_text_style = dict(horizontalalignment='center', verticalalignment='center',
                        fontsize=14,weight='bold')
        # # number of columns corresponds to number of regions
        ncols = len(regions)
        
        fig, axes = plt.subplots(1,ncols,figsize=figsize,sharey=True,sharex=True)
        for (region,bxpstats_list),ax in zip(plotting_dict.items(),axes.flatten()):
            ax.bxp(bxpstats_list,orientation='horizontal',patch_artist=True,
                boxprops=dict(facecolor=colors[region], edgecolor='k'),
                medianprops=dict(color='k'),
                meanprops=dict(marker='*', markerfacecolor='yellow', markeredgecolor='black',markersize=10),
                whiskerprops=dict(color='k'),
                showmeans=showmeans,
                showfliers=showfliers,
                showcaps=showcaps,
                showbox=showbox)
            ax.invert_yaxis() # labels read top-to-bottom
            ax.set_title(region,**ax_text_style)
            ax.set_xlabel("Travel time delay (s) for affected routes")
        fig.suptitle(title,**ax_text_style)
        plt.tight_layout()
        if save_fp is not None:
            plt.savefig(save_fp, bbox_inches = 'tight')
        plt.show()
        return plotting_dict
    
    # def get_total_travel_time_delay(self):
    #     """ 
    #     get total travel time delay by planning area
    #     Args:
    #         column_value (str): column name which to compute the travel time delay
    #     Returns:
    #         dict: a nested dict where 1st level of keys are end_nodesID, 2nd level of keys are REGION_N (district names), and values are travel time delay
    #     """
    #     # remove islands
    #     # planningArea = self.planningArea[~self.planningArea['PLN_AREA_N'].str.contains("ISLAND")]
    #     itinerary_df_list = self.get_grouped_travel_time_delay()
    #     # iterate through different itineraries
    #     travelTimeDelay_districts_dict = dict()#{p:[] for p in planningArea['REGION_N'].to_list()}
    #     # iterate through each end_nodesID which each has a unique planning area
    #     for end_nodesID, itinerary_df in itinerary_df_list.items():
    #         # get total travel time delay by district areas in Singapore (dict)
    #         # travelTimeDelay_districts = self.get_planningArea_itinerary(itinerary_df,plot=False)
    #         # group by REGION_N and sum up total travel time delay by REGION_N
    #         # travelTimeDelay_districts = travelTimeDelay_districts.groupby([self.start_groupby])['travel_time_delay'].sum().to_dict()
    #         travelTimeDelay_districts = itinerary_df.groupby([self.start_groupby])['travel_time_delay'].sum().to_dict()
    #         travelTimeDelay_districts_dict[end_nodesID] = travelTimeDelay_districts
        
    #     return travelTimeDelay_districts_dict

    # def plot_total_travel_time_delay(self,xlabels=None,colors=None,width=0.5,title="",ax=None,save_fp=None):
    #     """ plot a horizontal bar chart of total travel time delay per planning area
    #     Args:
    #         xlabels (list of str): list of x axis labels
    #         colors (dict): keys are REGION_N and values are rgb hex codes for each REGION_N
    #         width (float): width of each bar
    #         title (str): title for plot
    #         ax (mpl.Axes): if None, plot on a new figure, else, plot on supplied ax
    #         save_fp (str): file path to save figure to
    #     """
    #     # reorganise dict for plotting
    #     travelTimeDelay_districts_dict = self.get_total_travel_time_delay()
    #     # x labels are end_nodesID
    #     if xlabels is None:
    #         xlabels = [str(i) for i in list(travelTimeDelay_districts_dict)] # list of end_nodesID
    #     if colors is None:
    #         colors = {'EAST REGION':"#dffeb2","WEST REGION": "#ffe7c8","CENTRAL REGION":"#bedcfd",
    #                 'NORTH REGION':"#e9b3fd",'NORTH-EAST REGION':"#fdb3ba"}
    #     # districts are stacking variable in stacked bar
    #     plotting_dict = {district: [] for district in list(colors)}

    #     # iterate through each end_nodesID which each has a unique planning area
    #     for nodesID, travel_time_delay_dict in travelTimeDelay_districts_dict.items():
    #         # iterate through each start REGION_N and append the travel time delay to the plotting dict
    #         for district, timeDelay in travel_time_delay_dict.items():
    #             plotting_dict[district].append(timeDelay)
    #     # plot
    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=(6, 9))
    #     # start baseline at 0
    #     bottom = np.zeros(len(xlabels))
        
    #     # iterate through district, and cummulatively add district values "stacking"
    #     for district, travel_time_delay_arr in plotting_dict.items():
    #         p = ax.barh(xlabels,travel_time_delay_arr,height=width,
    #                 label=district,left=bottom,color=colors[district])
    #         bottom += travel_time_delay_arr
        
    #     ax.set_title("Total travel time delay from administrative districts to work clusters")
    #     # rotate time 45 deg
    #     # ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha='right')
    #     # set y axis title
    #     ax.set_ylabel("Work clusters")
    #     ax.set_xlabel("Total travel time delay (s)")
    #     ax.set_title(title)
    #     ax.invert_yaxis() # labels read top-to-bottom
    #     ax.legend(loc='lower center',bbox_to_anchor=(0.4,-0.2),ncols=3)
    #     if save_fp is not None:
    #         plt.savefig(save_fp, bbox_inches = 'tight')
    #     if ax is None:
    #         plt.show()
    #     return 
    
    # def get_stats_travel_time_delay(self,itinerary_df=None,save_fp=None):
    #     """ 
    #     Args:
    #         itinerary_df (pd.DataFrame): spatial joint of planning area and itinerary_df (output from get_planningArea_itinerary)
    #         save_fp (str): file path to save csv to
    #     Returns:
    #         dict: 1st level keys are end_nodesID, 2nd level keys are start_groupby, 
    #             3rd level keys are summary_travel_time_delay (summary of travel time delay) and summary_buses_delayed (keys are bus services and values are count of that affected bus service)
    #         pd.DataFrame: if save_fp is not None, export it as csv.
    #     """
    #     if itinerary_df is None:
    #         # compute travel time delay
    #         travel_time_delay_df = self.compute_travel_time_delay()
    #         # spatial joint of start_nodesID to their associated planningArea polygons
    #         itinerary_df = self.get_planningArea_itinerary(travel_time_delay_df,colors=None,plot=False)
        
    #     itinerary_df = itinerary_df.reset_index()
    #     grouped_stats = {end_nodesID: {region: region_df for region, region_df in end_nodesID_df.groupby(self.start_groupby)} 
    #     for end_nodesID, end_nodesID_df in itinerary_df.groupby('end_nodesID')}
        
    #     summary_stats = dict()
    #     for end_nodesID, end_nodesID_df in grouped_stats.items():
    #         region_stats = dict()
    #         for region, region_df in end_nodesID_df.items():
    #             N_routes = len(region_df.index) # number of different routes
    #             # column names to filter
    #             columns_filter = ['start_nodesID','end_nodesID','travel_time_delay',self.start_groupby]
    #             if 'routeId' in region_df.columns:
    #                 columns_filter.append('routeId')
    #             df = region_df[columns_filter]
    #             # only filter bus routes where travel time delay is experienced. there could be some routes where there are no time delay
    #             df = df[df['travel_time_delay']>0]
    #             travel_time_delay_stats = df['travel_time_delay'].describe().to_dict()
    #             region_stats[region] = {'N_routes':N_routes,
    #                                     'summary_travel_time_delay': travel_time_delay_stats,
    #                                     }
    #             # (only applicable for bus routes)
    #             if 'routeId' in df.columns:
    #                 # remove NA 
    #                 buses = [j for i in df['routeId'] for j in str(i).split(',') if j != "nan"]
    #                 # get count of delayed buses to identify how many routes and bus services are affected
    #                 buses_delayed, buses_count = np.unique(buses,return_counts=True)
    #                 summary_buses_delayed = {str(k):v for k,v in zip(buses_delayed,buses_count)}
    #                 region_stats[region]['summary_buses_delayed'] = summary_buses_delayed
    #             # region_stats[region] = {'summary_travel_time_delay': travel_time_delay_stats,
    #             #                         'summary_buses_delayed': summary_buses_delayed,
    #             #                         'N_routes':N_routes}
    #         summary_stats[end_nodesID] = region_stats

    #     # export as csv
    #     if save_fp is not None:
    #         # summarise it in a table
    #         summary_travel_time_delay = self.get_table_travel_time_delay(summary_stats)
    #         summary_travel_time_delay.to_csv(save_fp,index=False)

    #     return summary_stats

    # def get_table_travel_time_delay(self,stats_travel_time_delay=None):
    #     """ 
    #     Args:
    #         stats_travel_time_delay (dict): output of get_stats_travel_time_delay
    #     Returns:
    #         pd.DataFrame: a long dataframe with stats, values, region and end_nodesID
    #     """
    #     if stats_travel_time_delay is None:
    #         stats_travel_time_delay = self.get_stats_travel_time_delay()
    #     df_end_nodesID = []
    #     for end_nodesID, end_nodesID_df in stats_travel_time_delay.items():
    #         df_region = []
    #         for region, region_df in end_nodesID_df.items():
    #             df = pd.DataFrame.from_dict({'stats':list(region_df['summary_travel_time_delay']),
    #                                         'values':list(region_df['summary_travel_time_delay'].values())})
    #             df[self.start_groupby] = region
    #             df['end_nodesID'] = end_nodesID
    #             df_region.append(df)
    #         df_region = pd.concat(df_region)
    #         df_end_nodesID.append(df_region)
        
    #     summary_travel_time_delay = pd.concat(df_end_nodesID)
        
    #     return summary_travel_time_delay
    
    def get_all_bus_services(self, travel_time_delay_df=None):
        """ 
        get unique bus services number from all simulated routes
        Args:
            travel_time_delay_df (pd.DataFrame): Compute travel time delay and append it as a column to flooded_df. If None, call compute_travel_time_delay method
        Returns:
            dict: keys are end_nodesID, values are pd.DataFrame
        """
        if travel_time_delay_df is None:
            travel_time_delay_df = self.compute_travel_time_delay()
        # pad strings so that everything is sorted in order when calling set()
        routeId = [j.rjust(4) for i in travel_time_delay_df['routeId'] for j in str(i).split(',') if j != "nan"]
        routeId = list(set(routeId))
        return routeId

    def plot_bus_services_disruption(self, itinerary_df=None,figsize=(7,8),save_fp=None):
        """ 
        plot top 3 bus services affected as a function of OD and chances of the bus service being disrupted
        Args:
            itinerary_df (pd.DataFrame): spatial joint of planning area and itinerary_df (output from get_planningArea_itinerary)
            figsize (tuple): adjust figsize of matplotlib plot
            save_fp (str): filepath to save the figure to
        """
        
        stats_travel_time_delay = self.get_stats_travel_time_delay(itinerary_df)

        ylabel_text_style = dict(horizontalalignment='right', verticalalignment='center',
                  fontsize=8)
        ax_text_style = dict(horizontalalignment='center', verticalalignment='center',
                    fontsize=8,weight='bold')
        bus_text_style = dict(horizontalalignment='center', verticalalignment='center',
                    fontsize=8, fontfamily='monospace',weight='bold')
        freq_text_style = dict(horizontalalignment='center', verticalalignment='center',
                    fontsize=6)
        marker_style = dict(linestyle=':', color='0.8', markersize=15,
                            markerfacecolor="#9eeb34", markeredgecolor="#9eeb34")

        def format_axes(ax):
            ax.margins(0.2)
            ax.set_axis_off()
            ax.invert_yaxis()

        fig, axes = plt.subplots(ncols = 5,figsize=figsize)

        for row_ix, (end_nodesID, end_nodesID_df) in enumerate(stats_travel_time_delay.items()):
            for col_ix, (ax, (region, region_df)) in enumerate(zip(axes,end_nodesID_df.items())):
                buses_disrupted = region_df['summary_buses_delayed']
                # total number of routes simulated (flooded and non-flooded) from a region to end_nodesID
                N_routes = region_df['N_routes']
                # sort bus services by the number of disruptions
                sorted_buses_disrupted = {k: v for k, v in sorted(buses_disrupted.items(), key=lambda item: item[1],reverse=True)}
                # list out top 3 bus services affected
                bus3 = list(sorted_buses_disrupted)[:3]
                # number of unique bus services disrupted
                # n_buses = len(list(buses_disrupted))
                # number of disruptions for the top 3 bus services affected
                nDisruption = list(sorted_buses_disrupted.values())[:3]
                # chance of a bus service being disrupted is calculated as number of disruptions (on diff routes) for a particular bus service divided by number of routes
                chance_of_disruption = [d/N_routes for d in nDisruption]
                # print(nDisruption)
                # row_label = f'{region}' r"$\rightarrow$" f'{end_nodesID}'
                # label rows
                if col_ix == 0:
                    # y axis label, only do it for the first column axes
                    ax.text(-0.5, row_ix, f'{end_nodesID}', **ylabel_text_style)
                # add region name as title
                ax.set_title(region,**ax_text_style)
                # add bus icons as square markers
                ax.plot([row_ix] * 3, marker='s', **marker_style)
                for i in range(3):
                    # add chance of disruption as text above the buses
                    ax.text(i, row_ix+0.3, f'{chance_of_disruption[i]:.3f}', **freq_text_style)
                    # add bus service ID
                    ax.text(i, row_ix, bus3[i].upper(), **bus_text_style)
                # format axes
                format_axes(ax)
        plt.tight_layout()
        if save_fp is not None:
            plt.savefig(save_fp, bbox_inches = 'tight')
        plt.show()
        return

# write a function that fetches the saved csv files
def fetch_data(save_dir,transport_mode,routing=True, stats=True):
    """   
    Args:
        save_dir (str): folder where data is stored
        transport_mode (str): car or bus
        routing (bool): If True, fetches *_routing_*.csv
        stats (bool): If True
    Returns:
        dict: keys are: flooded_edges, routing, stats
    """
    fps = dict()

    def fetch_flooded_edges(flooded_dir):
        # fetch flooded edges fp
        return [os.path.join(flooded_dir,fp) for fp in os.listdir(flooded_dir) if bool(re.search("^flooded_edges.*pkl$",fp))]
    
    if bool(re.search("compoundFloodRouting" ,save_dir)):
        # compound flooding scenario
        coastal_flood_dir = save_dir.replace("compound","coastal")
        pluvial_flood_dir = save_dir.replace("compoundF","f")
        coastal_flood_fp = fetch_flooded_edges(coastal_flood_dir)
        pluvial_flood_fp = fetch_flooded_edges(pluvial_flood_dir)
        flooded_edges_fp = coastal_flood_fp + pluvial_flood_fp
    else:
        flooded_edges_fp = fetch_flooded_edges(save_dir)

    fps['flooded_edges'] = flooded_edges_fp

    if bool(re.search("compoundFloodRouting" ,save_dir)):
        title = f"Mode of transport: {transport_mode}; Scenario: Compound flooding; "
    elif bool(re.search("coastalFloodRouting" ,save_dir)):
        title = f"Mode of transport: {transport_mode}; Scenario: Coastal flooding; "
    else:
        title = f"Mode of transport: {transport_mode}; Scenario: Pluvial flooding; "

    def get_title(fp):
        title = os.path.splitext(os.path.basename(fp))[0].split("_")[-1]
        if "maxspeed" in title:
            title = f"Maximum speed on flooded roads = {title.replace("maxspeed","")} km/h"
        elif "percReducMaxspeed" in title:
            title = f"Percentage speed reduction on flooded roads = {title.replace("percReducMaxspeed","")} %"
        return title
    
    if routing:
        routing_fp = [os.path.join(save_dir,fp) for fp in os.listdir(save_dir) if bool(re.search(f"^{transport_mode}.*_routing_.*csv$",fp))]
        fps['routing'] = routing_fp
        fps['title'] = [title + get_title(fp) for fp in routing_fp]
        
    if stats:
        stats_fp = [os.path.join(save_dir,fp) for fp in os.listdir(save_dir) if bool(re.search(f"^{transport_mode}.*_stats_.*csv$",fp))]
        fps['stats'] = stats_fp
    
    return fps

class PlotTravelTimeDelay:
    def __init__(self, flood_routing_fps,dry_routing_fp,
                 flood_stats_fps, planningArea, units="seconds",
                 start_groupby='start_REGION_N',end_groupby='end_PLN_AREA_N',
                 selected_planningArea=['TAMPINES','JURONG EAST','WOODLANDS','DOWNTOWN CORE','SELETAR']):
        """   
        Args:
            flood_routing_fps (list of str): list of filepaths of flooded dfs
            dry_routing_fp (str): filepath of non-flooded df
            dry_routing_fp (str): filepath of stats df
            planningArea (gpd.GeoDataframe): planning area shape file
            units (str): "seconds" or "hours" for travel time to be displayed
            selected_planningArea (list): if None, plot all planning area. Else, list of planning areas to be plotted
        """
        self.flood_routing_fps = flood_routing_fps
        self.dry_routing_fp = dry_routing_fp
        self.flood_stats_fps = flood_stats_fps
        self.planningArea = planningArea
        self.units = units
        self.start_groupby = start_groupby
        self.end_groupby = end_groupby
        self.selected_planningArea = selected_planningArea

    def get_potential_car_time_delay(self,maxTrafficVol):
        """   Concat all potential car time delay dfs together in one df and make key the simulation assumption
        Args:
            maxTrafficVol (pd.DataFrame): traffic volume from region to planning area
        Returns:
            time delay in seconds
        """
        potential_time_delay_dict = dict()
        dry_df = pd.read_csv(self.dry_routing_fp)
        for i in range(len(self.flood_stats_fps)):
            fn = os.path.splitext(os.path.basename(self.flood_stats_fps[i]))[0].split('_')[-1]
            travelTimeDelay_stats_df = pd.read_csv(self.flood_stats_fps[i])
            # print(travelTimeDelay_stats_df.columns)

            flooded_df = pd.read_csv(self.flood_routing_fps[i])

            TTD = TravelTimeDelay(flooded_df,dry_df,self.planningArea,
                                                        column_value="simulated_total_duration",
                                                        end_groupby=self.end_groupby,
                                                        start_groupby=self.start_groupby)
            potential_time_delay = TTD.get_potential_car_time_delay(maxTrafficVol,travelTimeDelay_stats_df)
            potential_time_delay_dict[fn] = potential_time_delay# drop index column

        # concat all dataframes and use dict keys as keys
        potential_time_delay_dict = pd.concat(potential_time_delay_dict.values(),keys=list(potential_time_delay_dict)).reset_index(names=["simulation_assumption","index"])
        # print("df columns: ",potential_time_delay_dict.columns)
        
        return potential_time_delay_dict
    
    def get_potential_publicTransit_time_delay(self,OD_tripVol, spatialTravelPatterns):
        """  
        Args:
            OD_tripVol (pd.DataFrame): hourly average trip volume from origin to destination pt codes
            spatialTravelPatterns (pd.DataFrame): df that describes the number of commuters travelling from region to planning area via transport modes
        Returns:
            time delay in seconds
        """
        potential_time_delay_dict = dict()
        dry_df = pd.read_csv(self.dry_routing_fp)
        for fp in self.flood_routing_fps:
            flooded_df = pd.read_csv(fp)
            fn = os.path.splitext(os.path.basename(fp))[0].split('_')[-1]
            print(fn)
            TTD = TravelTimeDelay(flooded_df,dry_df,self.planningArea,
                                            column_value="simulated_total_duration",
                                            end_groupby=self.end_groupby,
                                            start_groupby=self.start_groupby)

            # compute travel time delay by comparing the difference in dry_df and flooded_df
            travel_time_delay_df = TTD.compute_travel_time_delay()
            # print("Length of travel time delay df: ", len(travel_time_delay_df))
            potential_time_delay = TTD.get_potential_publicTransit_time_delay(OD_tripVol,spatialTravelPatterns,travel_time_delay_df)
            potential_time_delay_dict[fn] = potential_time_delay# drop index column
        
        # concat all dataframes and use dict keys as keys
        potential_time_delay_dict = pd.concat(potential_time_delay_dict.values(),keys=list(potential_time_delay_dict)).reset_index(names=["simulation_assumption","index1"])
        
        return potential_time_delay_dict
    
    def get_time_delay_sum(self, potential_time_delay, var = "mean"):
        """   sum up travel time delay across the regions to find the whole-island wide travel time delay to the respective planning areas
        Args:
            potential_time_delay (pd.DataFrame): output from calling method: get_potential_*_time_delay
            var (str): determine which stat variable in the column to pick e.g. min, mean, max, 25%, 50%, 75%
        """
        def ensemble_mean(row):
            d = dict()
            d['mean_total_time_delay'] = row[f'potential_total_{var}_delay'].mean()
            d['min_total_time_delay'] = row[f'potential_total_{var}_delay'].min()
            d['max_total_time_delay'] = row[f'potential_total_{var}_delay'].max()
            return pd.Series(d,index=list(d))
        
        # get the ensemble mean across different model assumptions
        time_delay_sum = potential_time_delay.groupby([self.end_groupby,self.start_groupby]).apply(lambda x: ensemble_mean(x)).reset_index()
        # sum up total travel time delay across all the regions
        time_delay_sum = time_delay_sum.groupby(self.end_groupby).sum().reset_index()
        return time_delay_sum
    
    def get_plot_params(self):
        """   get plotting helpers
        Returns:
            tuple: tuple of color mapping for simulation assumption, marker mapping for stats, legend label adjustment
        """
        assum_dict = {'maxspeed5': 'maroon',
                    'maxspeed10': 'tomato', 
                    'maxspeed20': 'darkorange', 
                    'percReducMaxspeed10': 'forestgreen',
                    'percReducMaxspeed20': 'lightseagreen', 
                    'percReducMaxspeed50': 'dodgerblue',
                    'percReducMaxspeed80': 'royalblue'}
        
        marker_dict = {'potential_total_mean_delay':'*',
                            'potential_total_min_delay':'v',
                            'potential_total_25%_delay': 'o',
                            'potential_total_50%_delay':'s',
                            'potential_total_75%_delay':'P',
                            'potential_total_max_delay':'^'}
        # multiple string replacement for legend labels
        str_replace = {'maxspeed': 'Max speed (km/h) = ',
                    'percReducMaxspeed': 'Max speed reduction (%) = ',
                    'potential_total_':'',
                    '%': 'th percentile',
                    '_':' '
                    }
        return assum_dict, marker_dict, str_replace
    
    def plot_potential_time_delay(self,potential_time_delay,time_delay_sum,
                                  title="",loc = (0.1, 0.05),
                                  title_fontsize = 16,axis_fontsize=14,
                                  save_fp = None):
        """   Plots the travel time delay for all simulation assumptions for each region and planning area, and calculates the total mean travel time delay
        Args:
            potential_time_delay (pd.DataFrame): output from calling method: get_potential_*_time_delay()
            time_delay_sum (pd.DataFrame): output from calling method: get_time_delay_sum()
            plotting_params (tuple): tuple of color mapping for simulation assumption, marker mapping for stats, legend label adjustment
            title (str): title for plot
            save_fp (str): file path to save figure to
        """
        # convert units if necessary
        if self.units == "hours":
            # conversion to seconds to hours
            # find columns with delays
            delay_columns = [c for c in potential_time_delay.columns if bool(re.search("^potential.*_delay$",c))]
            potential_time_delay[delay_columns] = potential_time_delay[delay_columns].apply(lambda x: x/3600)
            delay_columns = [c for c in time_delay_sum.columns if bool(re.search(".*total_time_delay$",c))]
            time_delay_sum[delay_columns] = time_delay_sum[delay_columns].apply(lambda x: x/3600)
        # filter planning areas if necessary
        if self.selected_planningArea is not None:
            potential_time_delay = potential_time_delay[potential_time_delay[self.end_groupby].isin(self.selected_planningArea)]
            time_delay_sum = time_delay_sum[time_delay_sum[self.end_groupby].isin(self.selected_planningArea)]
        # pln_areas = potential_time_delay['PLN_AREA_N'].unique()
        regions = potential_time_delay[self.start_groupby].unique()
        simulation_assumptions = potential_time_delay['simulation_assumption'].unique()
        
        # plotting_dict = {region: {pln_area: dict() for pln_area in pln_areas} for region in regions}
        time_delay_columns = [c for c in potential_time_delay.columns if re.search(f"{self.end_groupby}|{self.start_groupby}|^potential.*|simulation_assumption",c)]
        
        time_delay_region = {k:df for k,df in potential_time_delay[time_delay_columns].groupby([self.start_groupby,'simulation_assumption'])}
        # potential_time_delay_columns = [c for c in potential_time_delay.columns if re.search("^potential.*",c)]
        
        # plotting helpers
        assum_dict, marker_dict, str_replace = self.get_plot_params()
        # modify axis title
        ax_text_style = dict(horizontalalignment='center', verticalalignment='center',
                        fontsize=axis_fontsize,weight='bold')
        n_assum = len(simulation_assumptions)
        y_step = n_assum + 2 # plot line every y_step interval

        ncols = len(regions) + 1 # use alst column to show total
        fig, axes = plt.subplots(1,ncols,figsize=(ncols*4,10),sharey=True,sharex=True)
        # get max x limit
        max_xlimit = 0
        # iterate plotting of regions by axes
        for ax, region in zip(axes.flatten()[:-1],regions):
            # plotting of model assumptions in each axis
            for i, (sim_assum,assum_color) in enumerate(assum_dict.items()):
                df = time_delay_region[(region,sim_assum)]
                df_dict = df.to_dict(orient='list')
                # assume that PLN_AREA are all arranged in sequence and exists in all dfs
                
                marker_style = dict(linestyle='None', color='0.8', markersize=6,
                                markerfacecolor=assum_color, markeredgecolor="black",alpha=0.5)
                line_style = dict(ecolor=assum_color,elinewidth=2,alpha=0.5)
                # plot error bar (horizontal straight lines joining up the points)
                min_error_bar = df['potential_total_min_delay'].values
                max_error_bar = df['potential_total_max_delay'].values
                if max_error_bar.max() > max_xlimit:
                    max_xlimit = max_error_bar.max()
                mean_val = np.column_stack([min_error_bar,max_error_bar]).mean(axis=1)
                xerr = max_error_bar - mean_val

                # plot horizontal lines connecting the markers
                y_label = df_dict[self.end_groupby]
                # print(y_label)
                y_list = np.arange(start=i,stop=(y_step)*len(y_label),step=y_step)
                # y_list = [l*(n_assum+2)+i for l in list(range(len(y_label)))]
                x_list = df['potential_total_mean_delay'].values
                ax.errorbar(mean_val, y_list, xerr=xerr, fmt='none',**line_style)

                # plot individual markers
                for column, marker in marker_dict.items():
                    x_list = df_dict[column]
                    ax.plot(x_list,y_list,marker=marker,**marker_style)
                
                y_tick_locations = np.arange(start=0,stop=(y_step)*len(y_label),step=y_step) + n_assum//2
                ax.set_yticks(y_tick_locations, y_label, fontsize=axis_fontsize)
                
                # ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right',fontsize=10)
                ax.invert_yaxis() # labels read top-to-bottom
                if self.units == "seconds":
                    ax.set_xlabel("Travel time delay (s)", fontsize=axis_fontsize)
                elif self.units == "hours":
                    ax.set_xlabel("Travel time delay (h)", fontsize=axis_fontsize)
                ax.set_title(region, **ax_text_style)
        
        # sum of stats for last axes
        # add error bars
        bar_style = dict(color="lightgrey",edgecolor="black",linewidth=2.5,ecolor="black",capsize=5)
        low_err = time_delay_sum['mean_total_time_delay']-time_delay_sum['min_total_time_delay']
        high_err = time_delay_sum['max_total_time_delay']-time_delay_sum['mean_total_time_delay']
        if high_err.max() > max_xlimit:
            max_xlimit = high_err.max()
        xerr = [low_err,high_err]
        # print(y_tick_locations,len(time_delay_sum['mean_total_time_delay']),len(low_err))
        axes[-1].barh(y_tick_locations,time_delay_sum['mean_total_time_delay'],xerr=xerr,
                    height=3,left=0,error_kw={"marker":"*"},**bar_style)
        axes[-1].set_title("Aggregated regions", **ax_text_style)
        if self.units == "seconds":
            axes[-1].set_xlabel("Travel time delay (s)", fontsize=axis_fontsize)
        elif self.units == "hours":
            axes[-1].set_xlabel("Travel time delay (h)", fontsize=axis_fontsize)
        
        # format axis
        for ax_ix, ax in enumerate(axes.flatten()):
            # set x limit
            ax.set_xlim(0,max_xlimit)
            # rotate xtick labels for all axes
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right',fontsize=axis_fontsize-2)
        # axes[0].set_ylabel("Planning Area Work Destinations", fontsize=axis_fontsize)
        # create cax to add colorbar to the figure
        fig.subplots_adjust(bottom=0.2)
        # add legend
        # add legends for sim assum
        # refine legend labels

        def multiple_replace(replacements, text):
            # Create a regular expression from the dictionary keys
            regex = re.compile("(%s)" % "|".join(map(re.escape, replacements.keys())))
            # For each match, look-up corresponding value in dictionary
            return regex.sub(lambda mo: replacements[mo.group()], text) 
        
        handles = [Line2D([0], [0], color=assum_color, label=multiple_replace(str_replace,assum_name),lw=2.5) for assum_name,assum_color in assum_dict.items()]
        # add legends for markers
        handles_markers = [Line2D([0], [0], color="k", marker=m, label=multiple_replace(str_replace,n),lw=0.5) for n,m in marker_dict.items()]
        handles = handles+handles_markers
        # reorder legend by row instead of column        
        fig.legend(handles=plot_utils.reorder_legend(handles,n_assum),loc=loc, ncol=n_assum, fontsize=axis_fontsize-3)
        title_text_style = dict(horizontalalignment='center', verticalalignment='center',
                        fontsize=title_fontsize,weight='bold')
        fig.suptitle(title,y=0.95,**title_text_style)
        if save_fp is not None:
            plt.savefig(save_fp, bbox_inches = 'tight')
        plt.show()

        return 