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

def plot_car_dry_isochrone(G, planningArea, workplace_cluster,
                           cbar=None,cmap="plasma",vmin=0,vmax=3600,save_fp=None):
    """ 
    Args:
        G (MultiDiGraph): graph of car network
        planningArea (gpd.GeoDataFrame): geopandas df of planning areas of SG
        workplace_cluster (pd.DataFrame): df of workplace coords and nodes ID
        cmap (str): cmap for colouring the isochrones
        cbar (ScalarMappable or None): if None, use cmap to automatically generate unique colours based on number of nodes. Else, use cbar to map values to colours
        save_fp (str): file path to save figure to
    """
    n_clusters = len(workplace_cluster.index)
    ncols = 3
    nrows = n_clusters//ncols
    # define cbar 
    if cbar is None:
        cbar = plot_utils.get_colorbar(vmin=vmin,vmax=vmax,cmap=cmap,plot=False)
    fig, axes = plt.subplots(nrows, ncols, figsize = (ncols*4,nrows*3))
    for i, ax in enumerate(axes.flatten()):
        # plot planning area boundary
        planningArea.plot(fc='white',ec='k',ax=ax)
        # get attributes
        lat = workplace_cluster.loc[i,"latitude"]
        lon = workplace_cluster.loc[i,"longitude"]
        node_id = workplace_cluster.loc[i,"node_ID"]
        # plot isochrone
        drive_route_times = driving.get_shortest_path_driving(G,orig = node_id,dest=None,cbar=cbar,ax=ax)
        # save drive route times 
        # utils.pickle_data(drive_route_times,os.path.join(iso_dir,f'dry_isochrone_car_workClusters{i}.pkl'))
        # plot orig node
        ax.scatter(lon,lat,marker="X",c="r",s=25)
    # plt.tight_layout()
    # plot colorbar
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.15, 0.75, 0.01]) # left, bottom, width, height
    fig.colorbar(cbar, cax=cbar_ax, orientation='horizontal', label='Travel time (seconds)')
    # fp_save = os.path.join(iso_dir,"dry_isochrone_car_workClusters.png")
    if save_fp is not None:
        plt.savefig(save_fp, bbox_inches = 'tight')
    plt.show()
    return

def plot_publicTransit_dry_isochrone(G, itinerary_df_list,planningArea, workplace_cluster,
                           cbar=None,cmap="plasma",vmin=0,vmax=3600,save_fp=None):
    """ 
    Args:
        G (MultiDiGraph): graph of car network
        itinerary_df_list (list of pd.DataFrame): itinerary is split by destination nodes aka workplace nodes
        planningArea (gpd.GeoDataFrame): geopandas df of planning areas of SG
        workplace_cluster (pd.DataFrame): df of workplace coords and nodes ID
        cmap (str): cmap for colouring the isochrones
        cbar (ScalarMappable or None): if None, use cmap to automatically generate unique colours based on number of nodes. Else, use cbar to map values to colours
        save_fp (str): file path to save figure to
    """
    n_clusters = len(workplace_cluster.index)
    ncols = 3
    nrows = n_clusters//ncols
    # define cbar 
    if cbar is None:
        cbar = plot_utils.get_colorbar(vmin=vmin,vmax=vmax,cmap=cmap,plot=False)
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
        publicTransit.plot_shortest_path_publicTransit(G, itinerary_df,cbar=cbar,ax=ax)
        # plot orig node
        ax.scatter(lon,lat,marker="X",c="r",s=25)
    # plt.tight_layout()
    # plot colorbar
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.15, 0.75, 0.01]) # left, bottom, width, height
    fig.colorbar(cbar, cax=cbar_ax, orientation='horizontal', label='Travel time (seconds)')
    # fp_save = os.path.join(iso_dir,"dry_isochrone_publicTransit_workClusters.png")
    if save_fp is not None:
        plt.savefig(save_fp, bbox_inches = 'tight')
    plt.show()