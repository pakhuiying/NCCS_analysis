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
import helper_functions.routing.floodedRouting as floodedRouting
import helper_functions.plot_utils as plot_utils
from math import ceil

class PlotIsochrone:
    def __init__(self, G, master_itinerary_df, planningArea):
        """ 
        Args:
            G (G): driving route
            master_itinerary_df (pd.DataFrame): an itinerary that shows the simulated travel time from start_nodesID to end_nodesID
            planningArea (gpd.GeoDataFrame): geopandas df of planning areas of SG
        """
        self.G = G
        self.master_itinerary_df = master_itinerary_df
        self.planningArea = planningArea

    def get_grouped_itinerary_df(self,groupby_name="PLN_AREA_N"):
        """ 
        split df based on groupby_name e.g. end_nodesID
        Args:
            groupby_name (str): column name of the itinerary df to group_by
        Returns:
            dict: keys are groupby_name, values are pd.DataFrame
        """
        # reset index
        master_itinerary_df = self.master_itinerary_df.reset_index()
        # split df based on end_nodesID
        return {k:df for k,df in master_itinerary_df.groupby(groupby_name)}

    def plot_shortest_path_route(self,itinerary_df,column_value,
                                    flooded_edges=None,ax=None,
                                    flooded_edge_color="red",
                                    cmap="plasma",cbar=None,
                                    node_size=5,node_alpha=0.8,
                                    edge_linewidth=0.2,edge_color="#999999"):
        """ 
        plot an isochrone using the simulated total duration
        Args:
            itinerary_df (pd.DataFrame): df with columns that describes the travel time delay
            column_value (str): column in itinerary_df which will determine the plotting of node colors on G e.g. travel_time_delay or simulated_total_travel_time
            flooded_edges (list): list of edges in G representing flooded roads. Default is None, it wont plot flooded edges
            ax (mpl.Axes): if None, plot on a new figure, else plot on supplied Axes
            cmap (str): cmap for colouring the isochrones
            cbar (ScalarMappable or None): if None, use cmap to automatically generate unique colours based on number of nodes. Else, use cbar to map values to colours
                Define cbar: cbar = plot_utils.get_colorbar(vmin=0,vmax=3600,cmap="plasma",plot=False)
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
        nc = [node_colors[node] if node in node_colors else "none" for node in self.G.nodes() ]
        ns = [node_size if node in node_colors else 0 for node in self.G.nodes()]
        # plot flooded edges, overlay flooded roads
        if flooded_edges is not None:
            edge_color = [flooded_edge_color if e in flooded_edges else "white" for e in self.G.edges(keys=True) ]
            edge_linewidth = [int(edge_linewidth*10) if e in flooded_edges else edge_linewidth for e in self.G.edges(keys=True) ]
        fig, ax = ox.plot_graph(
            self.G,
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

    def plot_isochrone(self, itinerary_df_list, column_value, flooded_edges=None,
                            cmap="plasma",flooded_edge_color="red",
                            title="",colorbar_label="",ncols=5,
                            workplace_node_color="red",
                            cbar=None,cbar_orientation="horizontal",
                            save_fp=None):
        """ 
        plot gridded isochrones using the simulated total duration
        Args:
            itinerary_df_list (dict): keys are groupby_name, values are pd.DataFrame
            column_value (str): column in itinerary_df which will determine the plotting of node colors on G e.g. travel_time_delay or simulated_total_travel_time
            flooded_edges (list): list of edges in G representing flooded roads. Default is None, it wont plot flooded edges
            cmap (str): cmap for colouring the isochrones
            flooded_edge_color (str): color for showing flooded roads
            title (str): title for figure
            colorbar_label (str): label for colorbar e.g. Total travel time delay (s)
            ncols (int): number of columns in figure
            workplace_node_color (str): color for showing destination aka workplace node
            cbar (ScalarMappable or None): if None, use cmap to automatically generate unique colours based on number of nodes. Else, use cbar to map values to colours
                Define cbar: cbar = plot_utils.get_colorbar(vmin=0,vmax=3600,cmap="plasma",plot=False)
            cbar_orientation (str): orientation of colorbar, either "horizontal" or "vertical"
            save_fp (str): file path to save figure to
        Returns:
            dict: sorted route times, where key are start_nodesID, and values are route times
        """
        # split df based on end_nodesID, where end_nodesID are the work place node id
        # itinerary_df_list = self.get_grouped_itinerary_df()
        # plot grid
        n_clusters = len(list(itinerary_df_list))
        nrows = ceil(n_clusters/ncols)
        if cbar is None:
            # define cbar 
            cbar = plot_utils.get_colorbar(vmin=0,vmax=self.master_itinerary_df[column_value].max(),cmap=cmap,plot=False)
        # plot
        fig, axes = plt.subplots(nrows, ncols, figsize = (ncols*4,nrows*3))
        for i, ((PLN_AREA_N,itinerary_df),ax) in enumerate(zip(itinerary_df_list.items(),axes.flatten())):
            # plot planning area boundary
            self.planningArea.plot(fc='white',ec='k',ax=ax)
            # get coordinates of the end_nodesID, all end coordinates are the same after splitting by end_nodesID
            lat = itinerary_df["end_lat"].values[0]
            lon = itinerary_df["end_lon"].values[0]
            # PLN_AREA_N = itinerary_df["PLN_AREA_N"].values[0]
            REGION_N = itinerary_df["end_REGION_N"].values[0]
            # remove itineraries where there are no bus routes 
            try:
                # (only applicable for bus)
                itinerary_df = itinerary_df[itinerary_df['number_of_busroutes']>0]
            except:
                pass
            # plot isochrone
            self.plot_shortest_path_route(itinerary_df,column_value=column_value,
                                            flooded_edges=flooded_edges,ax=ax,
                                            cbar=cbar,cmap=cmap,flooded_edge_color=flooded_edge_color
                                            )
            # plot orig node
            ax.scatter(lon,lat,marker="X",c=workplace_node_color,s=25,label="Workplace node")
            # add figure label and title 
            # add subplot label ({chr(97+i)}) 
            ax.set_title(f'{PLN_AREA_N} ({REGION_N})')
        
        # remove axis borders for blank axes
        emptyAxis = (ncols*nrows) - n_clusters
        for i in range(emptyAxis):
            axes[-1,-i-1].axis('off')
        # plot colorbar
        # create cax to add colorbar to the figure
        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.15, 0.07, 0.75, 0.05]) # left, bottom, width, height
        fig.colorbar(cbar, cax=cbar_ax, orientation=cbar_orientation, label=colorbar_label)
        fig.suptitle(title, fontsize='large', fontweight='bold',y=1.0)
        # add legend to the last subplot at the loc = lower right corner, box to anchor (x,y)
        # ax.legend(loc='lower right',  bbox_to_anchor=(1.0, 0.0), fontsize='medium')
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc = (0.5, 0.22), ncol=1, fontsize='medium')
        if save_fp is not None:
            plt.savefig(save_fp, bbox_inches = 'tight')
        # plt.tight_layout()
        plt.show()
        return

