import matplotlib.pyplot as plt
from utility.krige_model import KrigeModel
from matplotlib import ticker
from utility.parse_csv import CSVParser
from gstools import Linear
import matplotlib as mpl
import numpy as np 
from matplotlib import colors

class KrigingPlotter():

    r"""Handles plotting the heatmaps and the variance maps.

        Parameters
        ----------

        match_steps: :class:`bool`
            A boolean passed on from organize_kriging_area.
        bin_num: :class:`int`
            Number of bins for the estimated emperical variogram.
        length_scale: Optional, :class:`float`
            Length scale of gstools variogram.
    """

    def __init__(self, mode, bin_num: int=30, length_scale: float=1.0):
        self.mode = mode
        self.bin_num = bin_num
        self.length_scale = length_scale
        self.fig = None
        self.axs = None

        self.initialize_subplots()

    def initialize_subplots(self):
        r"""Sets up rows of subplots based on which legs are being plotted.
        """

        if self.mode == '0' or self.mode == '1' or self.mode == '2' or self.mode == '3':
            self.nrows = 2
            self.ncols = 1
        elif self.mode == 'all':
            self.nrows = 2
            self.ncols = 5
        
        self.fig, self.axs = plt.subplots(self.nrows,self.ncols,figsize=(17,7))
        self.subplot_index = 1
    
    def plot_heatmap(self, file: str, match_steps: bool, match_scale: bool = False, x_interpolation_input_range: list = None, y_interpolation_input_range: list = None):
        r"""Plots heatmap. Calls helper function based on which mode user
            decides upon initializing object – single leg or all legs.

            Parameters
            ----------
            file: :class:`str` 
                Name of the csv file where the data is stored. This file should
                be stored in /kriging/data.  

            Rest of parameters to be reorganized.
        """
        
        csvparser = CSVParser(file)
        
        if self.mode in ['0', '1', '2', '3']:
            x, y, stiff, title = csvparser.access_data(self.mode)
            self.plot_single_mode(x, y, stiff, title, match_steps, x_interpolation_input_range=x_interpolation_input_range, y_interpolation_input_range=y_interpolation_input_range)
        elif self.mode == 'all':
            self.plot_all_legs(csvparser, match_steps, match_scale, x_interpolation_input_range=x_interpolation_input_range, y_interpolation_input_range=y_interpolation_input_range)

    def plot_single_mode(self, x, y, stiff,title, match_steps: bool,  x_interpolation_input_range: list = None, y_interpolation_input_range: list = None):

        plt.figure(self.fig)
        krige_model = KrigeModel(x,y,stiff,self.bin_num, self.length_scale)
        model_type, models_dict, bin_centers, gamma = krige_model.rank_models()
        # self.plot_ranked_variogram(bin_centers, gamma, models_dict,self.axs[0],self.subplot_index, 20)
        model, r2 = krige_model.create_model(model_type.name)
        krige_model.organize_kriging_area(match_steps, x_interpolation_input_range, y_interpolation_input_range)
        z_pred, var, x_interpolation_range, y_interpolation_range = krige_model.execute_kriging(model)

        font = {'size': 7}

        # using rc function
        plt.rc('font', **font)

        ax1 = self.axs[0]
        im1 = ax1.imshow(z_pred, origin='lower', cmap='viridis', extent=(x_interpolation_range[0], x_interpolation_range[1], y_interpolation_range[0], y_interpolation_range[1]))
        ax1.scatter(x, y, c=stiff, edgecolors='k', cmap='viridis') 
        ax1.set_title(f'Kriging Interpolation – {model.name} ' + title)
        ax1.ticklabel_format(useOffset=False)
        # ax1.set_xlabel('X pos', fontsize = 8)
        # ax1.set_ylabel('Y pos', fontsize = 8)
        # ax1.tick_params(axis='both', which='major', labelsize=7)
        ax1.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        plt.colorbar(im1, ax=ax1, shrink=0.5)

        ax2 = self.axs[1]
        im2 = ax2.imshow(var, origin='lower', cmap='viridis', extent=(x_interpolation_range[0], x_interpolation_range[1], y_interpolation_range[0], y_interpolation_range[1]))
        ax2.set_title(f'Kriging Variance – {model.name} ' + title)
        ax2.ticklabel_format(useOffset=False)
        # ax2.set_xlabel('X pos', fontsize = 8)
        # ax2.set_ylabel('Y pos', fontsize = 8)
        # ax2.tick_params(axis='both', which='major', labelsize=7)
        ax2.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        plt.colorbar(im2, ax=ax2, shrink=0.5)
        plt.tight_layout()
        plt.show()

    
    def plot_all_legs(self, csvparser: CSVParser, match_steps: bool, match_scale: bool = False, x_interpolation_input_range: list = None, y_interpolation_input_range: list = None): 

        request_dict = {'0':0, '1':1, '2':2, '3':3, 'all':4}

        z_pred_list = []
        var_list = []
        kriging_results = {}
        
        for request in request_dict.keys():
            x, y, stiff, title = csvparser.access_data(request)
            krige_model = KrigeModel(x, y, stiff, self.bin_num, self.length_scale)
            model_type, models_dict, bin_centers, gamma = krige_model.rank_models()
            model, r2 = krige_model.create_model(model_type.name)
            krige_model.organize_kriging_area(match_steps, x_interpolation_input_range, y_interpolation_input_range)
            z_pred, var, x_interpolation_range, y_interpolation_range = krige_model.execute_kriging(model)

            z_pred_list.append(z_pred)
            var_list.append(var)
            kriging_results[request] = (z_pred, var, x, y, stiff, title, model, x_interpolation_range, y_interpolation_range)

        zmin, zmax, var_min, var_max = self.get_global_color_limits(z_pred_list, var_list)

        for request, (z_pred, var, x, y, stiff, title, model, x_interpolation_range, y_interpolation_range) in kriging_results.items():

            font = {'size': 7}
            plt.rc('font', **font)

            ax1 = self.axs[0, request_dict[request]]
            im1 = ax1.imshow(z_pred, origin='lower', cmap='viridis', extent=(x_interpolation_range[0], x_interpolation_range[1], y_interpolation_range[0], y_interpolation_range[1]))
            
            if match_scale:
                im1.norm.autoscale([zmin,zmax])
            ax1.scatter(x, y, c=stiff, edgecolors='k', cmap='viridis') 
            ax1.set_title(f'Kriging Interpolation – {model.name} ' + title)
            ax1.ticklabel_format(useOffset=False)
            # ax1.set_xlabel('X pos', fontsize = 8)
            # ax1.set_ylabel('Y pos', fontsize = 8)
            ax1.tick_params(axis='both', which='major', labelsize=7)
            ax1.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            self.fig.colorbar(im1, ax=ax1, shrink=0.7) # format=ticker.StrMethodFormatter("{x:.7f}"), shrink=0.7)

            ax2 = self.axs[1, request_dict[request]]
            im2 = ax2.imshow(var, origin='lower', cmap='viridis', extent=(x_interpolation_range[0], x_interpolation_range[1], y_interpolation_range[0], y_interpolation_range[1]))
            if match_scale:
                im2.norm.autoscale([var_min,var_max])
            ax2.set_title(f'Kriging Variance – {model.name} ' + title)
            ax2.ticklabel_format(useOffset=False)
            # ax2.set_xlabel('X pos', fontsize = 8)
            # ax2.set_ylabel('Y pos', fontsize = 8)
            ax2.tick_params(axis='both', which='major', labelsize=7)
            ax2.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            self.fig.colorbar(im2, ax = ax2, shrink=0.7)# format=ticker.StrMethodFormatter("{x:.7f}"))

        plt.tight_layout()
        plt.show()
            
    def plot_ranked_variogram(self, bin_center, gamma, models_dict, ax, subplot_index, vario_x_max: int=30):
        # plt.figure(self.fig)
        plt.subplot(self.nrows, self.ncols, subplot_index)
        self.subplot_index += 1
        plt.scatter(bin_center, gamma, color="k", label="data",)
        plt.title("Variogam model comparison – 2 traversals")
        plt.xlabel("Lag distance")
        plt.ylabel("Semivariogram")

        for fit_model in models_dict.values():
            fit_model.plot(fig=self.fig,x_max=vario_x_max, ax=ax)

    def plot_variogram(self, model, ax):
        plt.figure(self.fig)
        model.plot(ax=ax)
        ax.set_title(f"Fitted {model.name} Variogram")
        ax.set_xlabel("Lag distance")
        ax.set_ylabel("Semivariogram")
        plt.tight_layout()

    def get_global_color_limits(self, z_pred_list: list, var_list: list):

        global_z_min = float('inf')
        global_z_max = float('-inf')
        global_v_min = float('inf')
        global_v_max = float('-inf')

        for z_pred in z_pred_list:
            global_z_min = min(global_z_min, z_pred.min())
            global_z_max = max(global_z_max, z_pred.max())

        for var in var_list:
            global_v_min = min(global_v_min, var.min())
            global_v_max = max(global_v_max, var.max())

        return global_z_min, global_z_max, global_v_min, global_v_max