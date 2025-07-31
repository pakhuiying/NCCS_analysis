import copy
import osmnx as ox
import numpy as np


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