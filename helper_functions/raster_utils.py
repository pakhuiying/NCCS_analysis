import copy
import osmnx as ox
import numpy as np
import matplotlib as mpl
import helper_functions.plot_utils as plot_utils


class RasterToNetwork:
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
    
    def get_flood_coordinates(self,flood_arr,geotransform,flood_depth_thresh=0.15):
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

    def identify_flooded_roads(self,flooded_coordinates, plot=True, ax = None,
                               flooded_edge_color="red",edge_color="white", flooded_edge_linewidth=2):
        """ 
        Add attribute traffic_flow to G. 
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
    
    def get_flood_edges_depth(self,flood_arr,flooded_edges,flood_depth_thresh=0.15,
                              plot=True, cbar=None, ax = None, cmap='plasma', edge_color="grey",flooded_edge_linewidth=2,colorbar_label="",**kwargs):
        """ returns a dictionary that maps flood edge to their average depth 
        Args:
            flood_arr (np.ndarray): flood depth array in metres
            flooded_edgges (list of tuples): list of edges representing flooded roads
            flood_depth_thresh (np.ndarray): threshold (in metres) on the flood_arr, above this threshold, fetch the corresponding coordinates
            edge_color (str): color of non-floode road
            plot (bool): if True, plot traffic volume
            ax (mpl.Axes): if None, plot on a new figure, else, plot on supplied ax
            cbar (ScalarMappable or None): if None, use cmap to automatically generate unique colours based on number of nodes. Else, use cbar to map values to colours
                Define cbar: cbar = plot_utils.get_colorbar(vmin=0,vmax=3600,cmap="plasma",plot=False)
            cmap (str): cmap for colouring the isochrones. Default is plasma

        Returns:
            dict: keys are flood edge, values are averge flood depth
        """
        # get flooded pixels
        flood_depth = flood_arr[flood_arr>flood_depth_thresh]
        assert len(flooded_edges) == len(flood_depth), "flood depth and edges do not have the same length"
        flood_edges_depth = {e: [] for e in flooded_edges}
        for e,d in zip(flooded_edges, flood_depth):
            flood_edges_depth[e].append(d)
        # average the flood depth that falls on the same flood edge
        flood_edges_depth_dict = {k: np.mean(v) for k,v in flood_edges_depth.items()}

        if plot:
            flood_depth_list = list(flood_edges_depth_dict.values())
            flood_edges_list = list(flood_edges_depth_dict)

            # define cbar 
            if cbar is None:
                flood_depth_min = 0#np.min(flood_depth_list)
                flood_depth_max = np.max(flood_depth_list)
                print(f"min avg depth: {flood_depth_min} m; max avg depth: {flood_depth_max} m")
                cbar = plot_utils.get_colorbar(vmin=flood_depth_min,vmax=flood_depth_max,cmap=cmap,plot=False)
            color_map = {e: mpl.colors.rgb2hex(cbar.to_rgba(d),keep_alpha=True) for e,d in flood_edges_depth_dict.items()}
            ec = [color_map[e] if e in flood_edges_list else edge_color for e in self.G.edges(keys=True)]
            ew = [flooded_edge_linewidth if e in flood_edges_list else flooded_edge_linewidth/10 for e in self.G.edges(keys=True) ]

            fig, ax = ox.plot_graph(
                self.G,
                node_size=0,
                edge_color = ec,
                edge_linewidth=ew,
                ax=ax,
                bgcolor='white',
                show = False,
                close = False,
                **kwargs
            )

            # # plot colorbar
            # # create cax to add colorbar to the figure
            # fig.subplots_adjust(bottom=0.1)
            # cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.02]) # left, bottom, width, height
            # fig.colorbar(cbar, cax=cbar_ax, orientation="horizontal", label=colorbar_label)
        
        return flood_edges_depth