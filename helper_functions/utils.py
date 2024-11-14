
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

class NcObject:
    def __init__(self,fp):
        """ 
        Args:
            fp (str): file path of netcdf file
        """
        assert fp.endswith('.nc'), "fp should be a netcdf file"
        self.fp = fp
        self.nc_object = netCDF4.Dataset(fp)
        # variables
        # variables = list(self.nc_object.variables)
        
        metadata_fp_dict = self.getMetadataFp()
        for k,info in metadata_fp_dict.items():
            setattr(self,k,info)
        
        metadata_var_dict = vars(self.nc_object.variables[self.param])
        
        for k,info in metadata_var_dict.items():
            setattr(self,k,info)

    def get_compressed_data(self):
        return self.nc_object[self.param][:].compressed()

    def getMetadataFp(self):
        """ 
        Args:
            fp (str): filepath
        """
        param, _, gcm, ssp, model_id, downscaled_model,_,temporal_res, date_range = os.path.splitext(os.path.basename(self.fp))[0].split('_')
        # date_range = date_range.split('-')
        # date_range = '-'.join([d[:4] for d in date_range])
        return {'param':param,'gcm':gcm,'ssp':ssp,'temporal_res':temporal_res,'date_range':date_range}
    
def getPDF(flattened_data, bins=None):
    """ 
    Args:
        flattened_data (np.ndarray): flattened ncdf data from .get_compressed_data()
        bins (int or sequence of scalars): if bins is a sequence, it defines a monotonically increasing array of bin edges
    Returns:
        hist (array): The values of the histogram
        bin_edges (array of dtype float): Return the bin edges (length(hist)+1)
    """
    # flattened_data = self.nc_object[self.param][:].compressed()
    if bins is None:
        n, bin_edges = np.histogram(flattened_data, density=True)
    else:
        n, bin_edges = np.histogram(flattened_data, bins = bins, density=True)

    paired_range = [f'{bin_edges[i]}-{bin_edges[i+1]}' for i in range(len(bin_edges)-1)]
    prob_density = n * np.diff(bin_edges)
    cum_sum = np.cumsum(prob_density)
    return n, bin_edges, paired_range, prob_density, cum_sum
    
def plotPDF(flattened_data, bins= 'auto' , ax= None, **kwargs):
    """ 
    Args:
        bins (int or sequence of scalars): if bins is a sequence, it defines a monotonically increasing array of bin edges. 'auto': automatic binning
        nc_object (netCDF4 read file). all the layers in the nc_object is compressed to extract all the values across space and time
        title (str): title of the plot
        ax (mpl Ax): supply an axis. If none, it will just plot a single plot
        **kwargs: other arguments for ax.hist 
    """
    # rx1 = self.get_compressed_data()
    if ax is None:
        fig, ax = plt.subplots(1,1)
    n, bins, rectangles = ax.hist(flattened_data, bins = bins, density=True, **kwargs)
    ax.set_ylabel('PDF')
    
    if ax is None:
        plt.show()

    # check if integral of pdf is unity
    # np.sum(n * np.diff(bins))
    return n, bins, rectangles

def plotStairs(hist,bin_edges, ax = None,**kwargs):
    """plotting prebinned hist results as a histogram.
    Args:
        hist (int or arrays): count of histogram (step heights)
        edges (sequence): bin edges. The step positions, with len(edges) == len(vals) + 1, between which the curve takes on vals values.
    """
    if ax is None:
        fig, ax = plt.subplots(1,1)
    ax.stairs(hist,bin_edges, **kwargs)
    ax.set_ylabel('PDF')
    
    if ax is None:
        plt.show()

    # check if integral of pdf is unity
    # np.sum(n * np.diff(bins))
    return
    

class GetDistributions:
    def __init__(self,fp_list):
        """ 
        Args:
            fp_list (list of str): filepaths of the netcdf files for a meteorological variable
        """
        assert len(fp_list) > 0, "fp_list must not be length 0!"
        self.fp_list = fp_list
        self.time_period_categories=['1995-2014','2040-2059','2080-2099']
        self.ssp_categories = ['ssp126', 'ssp245', 'ssp585']
        

    def write_metadata(self,fp_save):
        """ 
        Args:
            nc_vars (dict): dictionary of metadata, obtained from vars(nc.variables[nc.param])
        """
        # initialise a single nc object
        nc = NcObject(self.fp_list[0])
        nc_param = nc.param
        nc_vars = vars(nc.nc_object[nc_param])
        # write metadata
        with open(fp_save, "w") as f:
            metadata_var = [f'- {k}: {info}' for k,info in nc_vars.items()]
            metadata_var = f'{nc_param}\n' + '\n'.join(metadata_var)
            f.write(metadata_var)
        return

    def get_nc_metadata(self):
        """ 
        Args:
        get the metadata from a single file path
        """
        # initialise a single nc object
        nc = NcObject(self.fp_list[0])
        nc_units = nc.units
        nc_param = nc.param
        nc_temporal_res = nc.temporal_res
        return nc_units, nc_param, nc_temporal_res
    
    def getMetadata(self,fp):
        """ 
        Args:
        Get metadata from filepath
            fp (str): filepath
        """
        param, _, gcm, ssp, model_id, downscaled_model,_,temporal_res, date_range = os.path.splitext(os.path.basename(fp))[0].split('_')
        # date_range = date_range.split('-')
        # date_range = '-'.join([d[:4] for d in date_range]) # first 4 characters are the year in YYYYMM
        return {'param':param,'gcm':gcm,'ssp':ssp,'temporal_res':temporal_res,'date_range':date_range}
    
    def scenarios_dict(self):
        """ 
        Args:
            dt (datetime obj): datetime object to check if it falls between two datetime objects
        Returns:
            dict: with keys, 
                'historical': {'1995-2014':[]}, 
                'ssp126': {'2040-2059':[], '2080-2099':[]},
                'ssp245': {'2040-2059':[], '2080-2099':[]},
                'ssp585': {'2040-2059':[], '2080-2099':[]},
        """
        time_period = self.time_period_categories
        dt_dict = {'historical': {time_period[0]: []}}
        for s in self.ssp_categories:
            if s not in list(dt_dict):
                dt_dict[s] = {t:[] for t in time_period[1:]}
                
        return dt_dict
        
    def get_monthly_temporal_res(self, date_range):
        """
        Args:
            date_range (str): date range (YYYYMMDD-YYYYMMDD)
        Returns:
            str: date range in YYYYMM-YYYYMM format
        """
        date_start, date_end = date_range.split('-')
        return f'{date_start[:6]}-{date_end[:6]}'
    
    def get_year_from_date_range(self,date_range):
        """ 
        Args:
            date_range (str): date_range obtained from self.getMetadata(m)
        Returns:
            tuple: of the date_start and date_end (int) in YYYY
        """
        date_start, date_end = date_range.split('-')
        return int(date_start[:4]), int(date_end[:4])
    
    def hash_date_range(self, date_range):
        """ 
        Args:
            date_range (str): date range in YYYYMM-YYYYMM format
        this checks if the date_start and the date_end from get_year_from_date_range falls within one of the ssp's time period and returns the time period
            date_range_key (str) e.g. '1995-2014' or '2040-2059' or '2080-2099'
            date_range (tuple of int) from get_year_from_date_range()
        """
        time_period = self.time_period_categories
        for dt in time_period:
            date_start, date_end = dt.split('-')
            if (date_range[0] >= int(date_start)) and (date_range[1] <= int(date_end)):
                return dt

    def get_plotting_dict(self):
        """ 
        Args:
        create a nested dctionary that determines the organisation of the plots
        first level key: global climate model (gcm)
        2nd level key: ssp scenarios/historical
        3rd level key: date range (YYYYMM) of the ncdf files 
        """

        model_dict = dict()
        for m in self.fp_list:
            metadata_dict = self.getMetadata(m)
            if metadata_dict['gcm'] not in model_dict:
                model_dict[metadata_dict['gcm']] = dict()
            if metadata_dict['ssp'] not in model_dict[metadata_dict['gcm']]:
                model_dict[metadata_dict['gcm']][metadata_dict['ssp']] = dict()
            yearly_daterange = self.get_year_from_date_range(metadata_dict['date_range']) # returns a tuple of int based on the date embded in the filepath
            timePeriod = self.hash_date_range(yearly_daterange) # returns the key to be hashed in the dict
            if timePeriod not in model_dict[metadata_dict['gcm']][metadata_dict['ssp']]:
                model_dict[metadata_dict['gcm']][metadata_dict['ssp']][timePeriod] = [m]
            else:
                model_dict[metadata_dict['gcm']][metadata_dict['ssp']][timePeriod].append(m)
            #     model_dict[metadata_dict['gcm']][metadata_dict['ssp']][timePeriod] = dict()
            # monthly_daterange = self.get_monthly_temporal_res(metadata_dict['date_range'])
            # if monthly_daterange not in model_dict[metadata_dict['gcm']][metadata_dict['ssp']][timePeriod]:
            #     model_dict[metadata_dict['gcm']][metadata_dict['ssp']][timePeriod][monthly_daterange] = [m] # initialise a list to store the list of fp
            # else:
            #     model_dict[metadata_dict['gcm']][metadata_dict['ssp']][timePeriod][monthly_daterange].append(m)

        return model_dict
    
    def concatenate_ncdfs(self, fp_list):
        """
        Args:
            fp_list (list of str): list of filepaths
        returns concatenated compressed np.ndarrays in the fp_list
        """
        ncdfs = [NcObject(fp) for fp in fp_list]
        concat_arrays = [nc.get_compressed_data() for nc in ncdfs]
        return np.concatenate(concat_arrays)
    
    def generate_concatenate_ncdfs(self, fp_list):
        """ 
        Args:
        defines a generator function that produces the compressed data from a generator output
        """
        for fp in fp_list:
            nc = NcObject(fp)
            yield nc.get_compressed_data()
    
    def fpList_to_generator(self):
        model_dict = self.get_plotting_dict()
        model_dict_arr = model_dict.copy() # make a copy of the structure of the dict, replaces fp_list with an np.array
        for gcm, scenarios in model_dict.items():
            for scenario, scenario_date_ranges in scenarios.items():
                for date_range, fp_list in scenario_date_ranges.items():
                    concat_data = self.generate_concatenate_ncdfs(fp_list) # instead of using self.concatenate_ncdfs(fp_list) for large files, use generator object
                    model_dict_arr[gcm][scenario][date_range] = concat_data
        return model_dict_arr

    def get_percentiles(self, data, perc_vals = np.arange(0,101,1,dtype=int), save_fp = None):
        """ 
        Args:
        get 0 to 100th percentile values
        """
        perc = np.percentile(data, perc_vals)
        if save_fp is not None:
            df = pd.DataFrame({'percentile':perc_vals,'percentile_values':perc})
            df.to_csv(save_fp,index=False)
        return perc
    
    def rawdata_to_csv(self, save_dir = None):
        """ 
        Args:
            save_dir (str): directory of where to save the csv file
        returns 20 year data for each scenario
        """
        # initialise a single nc object
        nc_units, nc_param, nc_temporal_res = self.get_nc_metadata()
        # set up dict structure
        model_dict = self.fpList_to_generator()
        # save file
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            sub_folder = os.path.join(save_dir,f'{nc_param}_{nc_temporal_res}')
            if not os.path.exists(sub_folder):
                os.mkdir(sub_folder)

        for gcm, scenarios in model_dict.items():
            for scenario, date_ranges in scenarios.items():
                for date_range, gen_object in date_ranges.items():
                    concat_data = np.concatenate(list(gen_object))
                    df = pd.DataFrame(data={f'{gcm}_{scenario}_{date_range}':concat_data})
                    fp_save = os.path.join(sub_folder,f'rawData_{nc_param}_{nc_temporal_res}_{gcm}_{scenario}_{date_range}.csv')
                    df.to_csv(fp_save,index=False)
        return



    
    def distributions_to_csv(self,bins = 100, save_dir = None):
        """ 
        Args:
            bins (int or sequence of scalars): if bins is a sequence, it defines a monotonically increasing array of bin edges
            save_dir (str): directory of where to save the csv file
        returns histogram (pd.DataFrame) and percentile results (pd.DataFrame) in a tuple
        """
        # initialise a single nc object
        nc_units, nc_param, nc_temporal_res = self.get_nc_metadata()
        # set up dict structure
        model_dict = self.fpList_to_generator() #self.get_plotting_dict()
        df_list = [] # store histogram results
        # get percentiles
        perc_v = np.arange(0,101,1,dtype=int)
        df_perc = {'percentile':perc_v} # to store percentile values for various scenarios

        for gcm, scenarios in model_dict.items():
            for scenario, date_ranges in scenarios.items():
                for date_range, gen_object in date_ranges.items():
                    concat_data = np.concatenate(list(gen_object))
                    # get distribution for observations over a 20 year period in historical/each ssp scenario
                    n, bin_edges, paired_range, prob_density, cum_sum = getPDF(concat_data,bins)

                    data = {'hist': n, 
                            f'values ({nc_units})': bin_edges[1:], 
                            f'range ({nc_units})':paired_range, 
                            'prob density':prob_density, 
                            'cumsum':cum_sum,
                            'date range': [date_range]*len(n),
                            'scenario':[scenario]*len(n),
                            'gcm':[gcm]*len(n)}
                    
                    # store histogram results
                    df = pd.DataFrame.from_dict(data)
                    df_list.append(df)
                    
                    # store percentile results
                    perc_values = self.get_percentiles(concat_data,perc_v)
                    df_perc[f'{gcm}_{scenario}_{date_range}'] = perc_values
        
        # export data
        df_concat = pd.concat(df_list, axis=0)
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            # export histogram data to csv file
            fp_save = os.path.join(save_dir,f'hist_{nc_param}_{nc_temporal_res}.csv')
            df_concat.to_csv(fp_save,index=False)
            # export percentile data to csv file
            fp_save = os.path.join(save_dir,f'percentile_{nc_param}_{nc_temporal_res}.csv')
            df_perc = pd.DataFrame(df_perc)
            df_perc.to_csv(fp_save,index=False)
            # export metadata
            fp_save = os.path.join(save_dir,f'metadata_{nc_param}_{nc_temporal_res}.txt')
            self.write_metadata(fp_save)

            return df_concat, df_perc
        else:
            return df_concat, pd.DataFrame(df_perc)

    def label_daterange(self,date_range):
        if date_range.startswith('1995'):
            return 'Historical'
        elif date_range.startswith('2040'):
            return 'Mid-century'
        else:
            return 'End-century'
        
    def load_hist_from_csv(self, save_dir):
        """ 
        Args:
            save_dir (str): directory where csv file containing the hist results is stored
            save_dir (str): directory of where the csv file is stored
        returns a tuple of dict, where the first item of the tuple is hist, and second item of tuple is bin sequence
        returns a dictionary where keys are tuples of (gcm, ssp, date_period)
        """
        # initialise a single nc object
        nc_units, nc_param, nc_temporal_res = self.get_nc_metadata()
        # parse the file path of the csv file
        fp_hist = os.path.join(save_dir, f'hist_{nc_param}_{nc_temporal_res}.csv')
        df = pd.read_csv(fp_hist)
        # get columns of values and attributes
        val_cols = df.columns[:-3].to_list()
        col_cols = df.columns[-3:].to_list()
        # include index for pivoting. 35 scenarios were generated. 
        # 2 time periods for 3 ssp scenarios, 1 time period for historical (2*3+1 = 7). Multiply this for 5 gcms: 5*7=35
        # index = int(len(df.index)/35)
        # df['index'] = list(range(index)) * 35
        index = int(len(df.index)/100)
        df['index'] = list(range(1,101)) * index
        # pivot data
        df_pivot = df.pivot(index='index', values=val_cols,columns=reversed(col_cols))
        # column name of hist count
        hist_col = val_cols[0]
        hist = df_pivot[hist_col].to_dict('list')
        # column name of bins range
        bins_range_col = val_cols[2]
        bins_range = {k: [float(v[0].split('-')[0])] + [float(j.split('-')[1]) for j in v] for k,v in df_pivot[bins_range_col].to_dict('list').items()}
        return hist, bins_range

    def load_perc_from_csv(self, save_dir):
        """ 
        Args:
            save_dir (str): directory where csv file containing the hist results is stored
            save_dir (str): directory of where the csv file is stored
        returns a dictionary where keys are tuples of (gcm, ssp, date_period), and values are a list of float representing 0th - 100th percentile
        subset the 95th percentile by values[95], as the index corresponds to the percentile
        """
        # initialise a single nc object
        nc_units, nc_param, nc_temporal_res = self.get_nc_metadata()
        # parse the file path of the csv file
        fp_save = os.path.join(save_dir,f'percentile_{nc_param}_{nc_temporal_res}.csv')
        # load csv
        df = pd.read_csv(fp_save)
        # convert df to dict
        df_dict = df.to_dict('list')
        df_dict = {tuple(k.split('_')): v for k,v in df_dict.items()}
        return df_dict
    
    def build_dict_from_tuple(self, loaded_dict):
        """ 
        Args:
        returns a dict of unique gcms, ssps, and time_period from a loaded dict from load_hist_from_csv
        """
        tuple_keys = list(loaded_dict.keys())
        nKeys = len(tuple_keys[0])
        plot_dict = dict()
        for k in range(nKeys):
            plot_dict[k] = list(sorted(set([i[k] for i in tuple_keys])))
        return plot_dict

    def plot_distributions_from_csv(self, save_dir, save = False, perc1 = 95, perc2 = 99):
        """ 
        Args:
            save_dir (str): directory where csv file containing the hist results is stored
            plot (bool): whether to save plot or not
            perc1 (float): percentile e.g. 95th percentile to overlay on the plot
            perc2 (float): percentile e.g. 99th percentile to overlay on the plot
        """
        # initialise a single nc object
        nc_units, nc_param, nc_temporal_res = self.get_nc_metadata()

        # load in pre-saved data
        # load in prebinned data
        hist_dict, bins_range_dict = self.load_hist_from_csv(save_dir)
        # laod in percentile data
        percentile_dict = self.load_perc_from_csv(save_dir)

        # get a dict of unique gcms, ssps, and time_period
        keys_dict = self.build_dict_from_tuple(hist_dict)

        ncols = len(keys_dict[0]) # number of columns = number of gcm models
        nrows_names = keys_dict[1] # 3 ssp scenarios
        nrows = len([i for i in nrows_names if i.startswith('ssp')])

        # initialise colours for time period
        color_dict = {'1995-2014':'gray','2040-2059':'orange','2080-2099': 'maroon'}
        
        # initialise
        hist_historical = bin_edges_historical = historical_date_range = perc95 = perc99 =None

        # plot grid
        fig, axes = plt.subplots(nrows, ncols,sharey = True, sharex = True, figsize = (15,10))
        # plot gcms on columns
        for col_index, gcm in enumerate(keys_dict[0]):
            # plot scenarios on rows
            for row_index, scenario in enumerate(keys_dict[1]):
                
                if scenario.startswith('ssp'):
                    row_index -= 1 # to ensure that ssp scenarios are plotted starting from the first row 
                    ax = axes[row_index,col_index]
                    
                    # plot all the date ranges on the same axes e.g. 1995-2014, 2040 - 2059, 2080 - 2099
                    for ndate, date_range in enumerate(keys_dict[2][1:]):
                        label = f'{date_range} ({self.label_daterange(date_range)})'
                        # tuple key to subset pre-saved data
                        tuple_key = (gcm, scenario, date_range)
                        # load hist, bins
                        if tuple_key in list(hist_dict):
                            hist, bin_edges = hist_dict[tuple_key], bins_range_dict[tuple_key]
                            # plot ssp pdf
                            plotStairs(hist,bin_edges, ax = ax, label = label,
                                    lw=2, color = color_dict[date_range],alpha=0.5) # it plots step as a default
                        
                        # tuple key of historical data
                        historical_tuple_key = (gcm, keys_dict[1][0], keys_dict[2][0])
                        if historical_tuple_key in list(hist_dict):
                            hist_historical, bin_edges_historical = hist_dict[historical_tuple_key], bins_range_dict[historical_tuple_key]
                            historical_date_range = f'{keys_dict[2][0]} ({self.label_daterange(keys_dict[2][0])})'
                            # only need to plot historical once for each plot
                            if ndate == 0:
                                # plot historical
                                plotStairs(hist_historical,bin_edges_historical, ax = ax, label = historical_date_range, 
                                        color = color_dict[keys_dict[2][0]],lw=2,alpha=0.5) # it plots step as a default
                                # plot percentile
                                perc95, perc99 = percentile_dict[historical_tuple_key][perc1], percentile_dict[historical_tuple_key][perc2]
                                ax.axvline(x=perc95,c='r',ls = '--',label = f'{perc1}th percentile (Historical)', lw = 1,alpha=0.5)
                                ax.axvline(x=perc99,c='purple',ls = '--',label = f'{perc2}th percentile (Historical)', lw = 1,alpha=0.5)
                        
                if row_index == 0:
                    ax = axes[row_index,col_index]
                    ax.set_title(gcm, fontsize = 15, fontweight = 'bold')
                if col_index == 0:
                    ax.set_ylabel(scenario, fontsize = 15, fontweight = 'bold')
                    
        # add legend
        handles, labels = axes[0,0].get_legend_handles_labels()
        # # sort legend
        legend_sorted = [(x,y) for y, x in sorted(zip(labels, handles))]
        # handles, labels
        fig.legend([i[0]  for i in legend_sorted], [i[1]  for i in legend_sorted], bbox_to_anchor = (0.5,-0.05), loc = 'lower center',ncol = len(handles))

        # update axis labels
        for i,ax in enumerate(axes.flatten()):
            if i%ncols == 0:
                pass
            else:
                ax.set_ylabel('')
            ax.set_xlim(0.95*percentile_dict[historical_tuple_key][0],1.05*percentile_dict[tuple_key][perc2])

        fig.text(0.5, -0.01, nc_units, ha='center',fontsize=15)
        fig.text(-0.04, 0.5, f'PDF of {nc_param} ({nc_temporal_res})', va='center', rotation='vertical',fontsize=15)
        plt.tight_layout()

        if save is True:
            # export csv file
            fp_save = os.path.join(save_dir,f'{nc_param}_{nc_temporal_res}.png')
            plt.savefig(fp_save, bbox_inches = 'tight')

        plt.show()


    def plot_distributions(self, save_dir = None):
        """ 
        Args:
            hist_dir (str): file directory where the csv of the prebinned data that is previously generated from distributions_to_csv() is stored. If not None and exists, then it will load from the csv
            save_dir (str): directory of where to save the plot figure
        TODO: incorporate plotting of distributions using pre-binned data by importing from hist_dir
        TODO: or have a separate plotting function for pre-binned data? Likely that future data will all be computed from pre-binned data
        """
        # initialise a single nc object
        nc_units, nc_param, nc_temporal_res = self.get_nc_metadata()

        model_dict = self.fpList_to_generator()
        ncols = len(list(model_dict)) # number of columns = number of gcm models
        nrows_names = list(model_dict[list(model_dict)[0]]) # 3 ssp scenarios
        nrows = len([i for i in nrows_names if i.startswith('ssp')])
        
        # create label names for legend
        # label_dict = {'1995-2014': 'Historical', '2040-2059':'Mid-century','2080-2099':'End-century'}
        # initialise
        nc_historical = historical_date_range = perc95 = perc99 =None
        # plot grid
        fig, axes = plt.subplots(nrows, ncols,sharey = True, sharex = True, figsize = (15,10))
        # plot gcms on columns
        for col_index, (gcm, scenarios) in enumerate(model_dict.items()):
            # plot scenarios on rows
            for row_index, (scenario, date_ranges) in enumerate(scenarios.items()):
                
                if scenario.startswith('ssp'):
                    row_index -= 1 # to ensure that ssp scenarios are plotted starting from the first row 
                    ax = axes[row_index,col_index]
                    
                    # plot all the date ranges on the same axes e.g. 1995-2014, 2040 - 2059, 2080 - 2099
                    for ndate, (date_range, gen_object) in enumerate(date_ranges.items()):
                        label = f'{date_range} ({self.label_daterange(date_range)})'
                        # concat all observations within YYYMM period
                        # concat_data = self.concatenate_ncdfs(fp_list)
                        concat_data = np.concatenate(list(gen_object))
                        # plot pdf
                        plotPDF(concat_data, bins = 'auto',ax=ax, histtype='step', label = label,lw=2)
                        if nc_historical is None:
                            # skip plotting of historical if historical data is missing
                            pass
                        else:
                            if ndate > 0:
                                # to avoid plotting duplicate of ssp scenarios
                                pass
                            else:
                                plotPDF(nc_historical, bins = 'auto',ax=ax, histtype='step', label = historical_date_range, color = 'gray',lw=2)
                                ax.axvline(x=perc95,c='r',ls = '--',label = '95th percentile (Historical)', lw = 1)
                                ax.axvline(x=perc99,c='purple',ls = '--',label = '99th percentile (Historical)', lw = 1)
                else: # if scenario starts with historical
                    for date_range, gen_object in date_ranges.items():
                        # update initial values
                        nc_historical = np.concatenate(list(gen_object))#self.concatenate_ncdfs(fp_list)
                        historical_date_range = f'{date_range} ({self.label_daterange(date_range)})'
                        # get percentiles of historical
                        perc95, perc99 = np.percentile(nc_historical,[95,99])
                        
                
                if row_index == 0:
                    ax = axes[row_index,col_index]
                    ax.set_title(gcm, fontsize = 15, fontweight = 'bold')
                if col_index == 0:
                    ax.set_ylabel(scenario, fontsize = 15, fontweight = 'bold')
                    
        # add legend
        handles, labels = axes[-1,-1].get_legend_handles_labels()
        # sort legend
        handles_sorted = [(x,y) for y, x in sorted(zip(labels, handles))]
        labels_rename = [i[1]  for i in handles_sorted]
        # handles, labels
        fig.legend([i[0] for i in handles_sorted], labels_rename, bbox_to_anchor = (0.5,-0.05), loc = 'lower center',ncol = len(handles_sorted))

        # update axis labels
        for i,ax in enumerate(axes.flatten()):
            if i%ncols == 0:
                pass
            else:
                ax.set_ylabel('')
            ax.set_xlim(0,1.05*perc99)

        fig.text(0.5, -0.01, nc_units, ha='center',fontsize=15)
        fig.text(-0.04, 0.5, 'PDF', va='center', rotation='vertical',fontsize=15)
        plt.tight_layout()

        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            # export csv file
            fp_save = os.path.join(save_dir,f'{nc_param}_{nc_temporal_res}.png')
            plt.savefig(fp_save, bbox_inches = 'tight')

        plt.show()

        return