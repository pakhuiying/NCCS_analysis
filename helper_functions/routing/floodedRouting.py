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
            pd.DataFrame: if save_fp is not None, export it as csv.
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
        fig.suptitle(title)
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

