import matplotlib.pyplot as plt
from utility.krige_model import KrigeModel
from matplotlib import ticker
from utility.parse_csv import CSVParser
import numpy as np 
from numpy import number

class KrigingPlotter():

    r"""Handles plotting the heatmaps and the variance maps.

        Parameters
        ----------

        match_steps: :class:`bool`
            A boolean passed on from organize_kriging_area.
        bin_num: :class:`int`
            Number of bins for the estimated emperical variogram.
        length_scale: Optional, :class:`float`
            Length scale of GSTools variogram.
    """

    def __init__(self, mode: list[int], bin_num: int=30, length_scale: dict = {'0': 1.0, '1': 1.0, '2': 1.0, '3': 1.0, 'all': 1.0}):
        self.mode = mode
        self.bin_num = bin_num
        self.length_scale = length_scale
        self.fig = None
        self.axs = None
        self.ncols = None

        self.initialize_subplots()

    def initialize_subplots(self):
        r"""Sets up rows of subplots based on which legs are being plotted.
        """

        nrows = 2

        if len(self.mode) > 1:
            if 'all' in self.mode:
                self.ncols = 5
            else:
                self.ncols = len(self.mode) + 1 
        elif len(self.mode) == 1:
            self.ncols = 1
        
        self.fig, self.axs = plt.subplots(nrows,self.ncols,figsize=(15,7))
    
    def plot_heatmap(self, file: str, match_steps: bool, match_scale: bool = False,
        x_interpolation_input_range: list = None,
        y_interpolation_input_range: list = None,
        transparent: bool = True):
        r"""Plots heatmap. Calls helper function based on which mode user
            decides upon initializing object – single leg or all legs.

            Parameters
            ----------
            file: :class:`str` 
                Name of the csv file where the data is stored. This file should
                be stored in /kriging/data.  
            match_steps: :class:`bool`
                Boolean that determines whether the kriging area will be fitted
                to match the input data. If False, user must input 
                x and y_interpolation_input_range.
            match_scale: :class:`bool`, optional
                Boolean that determines whether the colorbar scale for the
                interpolation plots will be consistent across images, and same
                for the varance plots. Only matters if mode is 'all'. 
            x_interpolation_input_range: :class:`list[float]`, optional
                Only necessary to provide if match_steps is False. This is a
                array of length 2, where the 0th index is the lower bound
                of the range, and the 1st index is the upper bound of the range.
            y_interpolation_input_range: :class:`list[float]`, optional
                Only necessary to provide if match_steps is False. This is a
                array of length 2, where the 0th index is the lower bound
                of the range, and the 1st index is the upper bound of the range.
        """
        
        csvparser = CSVParser(file)

        self.plot_legs(csvparser, match_steps, 
                            self.length_scale, match_scale, transparent,
                                x_interpolation_input_range=x_interpolation_input_range, 
                                y_interpolation_input_range=y_interpolation_input_range)

    def plot_legs(self, csvparser: CSVParser, match_steps: bool, length_scale: dict,
                    match_scale: bool = False, transparent: bool = True, x_interpolation_input_range: list = None, 
                    y_interpolation_input_range: list = None):
        
        x_arr_list= []
        y_arr_list = []
        stiff_arr_list = []
        
        z_pred_list = []
        var_list = []
        kriging_results = {}
        axis_index = 0

        for request in self.mode:
            x, y, stiff, title = csvparser.access_data([request])

            x_arr_list.append(x)
            y_arr_list.append(y)
            stiff_arr_list.append(stiff)

            krige_model = KrigeModel(x, y, stiff, self.bin_num, length_scale[request])
            model_type, models_dict, bin_centers, gamma = krige_model.rank_models()
            # self.plot_ranked_variogram(bin_centers, gamma, models_dict,self.axs[0], 20)
            fitted_model, r2 = krige_model.fit_model(model_type.name)
            x_interpolation_range, y_interpolation_range = krige_model.organize_kriging_area(match_steps, 
                                                x_interpolation_input_range, 
                                                y_interpolation_input_range)
            z_pred, var, x_interpolation_range, y_interpolation_range = krige_model.execute_kriging(fitted_model)                

            z_pred_list.append(z_pred)
            var_list.append(var)
            kriging_results[request] = (z_pred, var, x, y, stiff, title, fitted_model,
                                        x_interpolation_range,
                                        y_interpolation_range,
                                        axis_index)
            axis_index += 1

        if len(self.mode) > 1:

            x_arr = np.concatenate(x_arr_list)
            y_arr = np.concatenate(y_arr_list)
            stiff_arr = np.concatenate(stiff_arr_list)

            request = ",".join(self.mode)

            krige_model = KrigeModel(x_arr, y_arr, stiff_arr, self.bin_num, length_scale[request])
            model_type, models_dict, bin_centers, gamma = krige_model.rank_models()
            # self.plot_ranked_variogram(bin_centers, gamma, models_dict,self.axs[0], 20)
            fitted_model, r2 = krige_model.fit_model(model_type.name)
            x_interpolation_range, y_interpolation_range = krige_model.organize_kriging_area(match_steps, 
                                                x_interpolation_input_range, 
                                                y_interpolation_input_range)
            all_z_pred, all_var, x_interpolation_range, y_interpolation_range = krige_model.execute_kriging(fitted_model)                

            z_pred_list.append(z_pred)
            var_list.append(var)

        zmin, zmax, var_min, var_max = self.get_global_color_limits(z_pred_list, var_list)

        for request, (z_pred, var, x, y, stiff, title, fitted_model, x_interpolation_range, y_interpolation_range, axis_index) in kriging_results.items():
            
            font = {'size': 7}
            plt.rc('font', **font)

            # Interpolation plotting 
            try:
                ax1 = self.axs[0, axis_index]
            except IndexError:
                ax1 = self.axs[0]

            z_alpha = np.ones_like(z_pred)
            if transparent:
                z_alpha[var > 10] = 0.3

            im1 = ax1.imshow(z_pred, origin='lower', cmap='viridis', 
                                extent=(x_interpolation_range[0], x_interpolation_range[1],
                                    y_interpolation_range[0], y_interpolation_range[1]),
                                alpha = z_alpha)
            ax1.set_xlim([x_interpolation_range[0], x_interpolation_range[1]])
            ax1.set_ylim([y_interpolation_range[0], y_interpolation_range[1]])     
            if match_scale:
                im1.norm.autoscale([zmin,zmax])
            ax1.scatter(x, y, c=stiff, edgecolors='k', cmap='viridis', s=15) 
            ax1.set_title(f'Kriging Interpolation – {title}')
            ax1.ticklabel_format(useOffset=False)
            ax1.tick_params(axis='both', which='major', labelsize=7)
            ax1.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            self.fig.colorbar(im1, ax=ax1, shrink=0.7)

            # Variance plotting
            try:
                ax2 = self.axs[1, axis_index]
            except IndexError:
                ax2 = self.axs[1]
            im2 = ax2.imshow(var, origin='lower', cmap='viridis', 
                                extent=(x_interpolation_range[0], x_interpolation_range[1],
                                        y_interpolation_range[0], y_interpolation_range[1]))
            ax2.set_xlim([x_interpolation_range[0], x_interpolation_range[1]])
            ax2.set_ylim([y_interpolation_range[0], y_interpolation_range[1]])
            if match_scale:
                im2.norm.autoscale([var_min,var_max])
            ax2.set_title(f'Kriging Variance – {title}')
            ax2.ticklabel_format(useOffset=False)

            ax2.tick_params(axis='both', which='major', labelsize=7)
            ax2.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            self.fig.colorbar(im2, ax = ax2, shrink=0.7)

        if len(self.mode) > 1:

            axis_index = len(self.mode)

            font = {'size': 7}
            plt.rc('font', **font)

            z_alpha = np.ones_like(z_pred)
            if transparent:
                z_alpha[all_var > 10] = 0.3

            try:
                ax1 = self.axs[0, axis_index]
            except IndexError:
                ax1 = self.axs[0]

            im1 = ax1.imshow(all_z_pred, origin='lower', cmap='viridis', 
                                extent=(x_interpolation_range[0], x_interpolation_range[1],
                                    y_interpolation_range[0], y_interpolation_range[1]), alpha = z_alpha)
            ax1.set_xlim([x_interpolation_range[0], x_interpolation_range[1]])
            ax1.set_ylim([y_interpolation_range[0], y_interpolation_range[1]])     
            if match_scale:
                im1.norm.autoscale([zmin,zmax])
            ax1.scatter(x_arr, y_arr, c=stiff_arr, edgecolors='k', cmap='viridis', s = 15) 
            ax1.set_title(f'Interpolation – Combined')
            ax1.ticklabel_format(useOffset=False)
            ax1.tick_params(axis='both', which='major', labelsize=7)
            ax1.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            self.fig.colorbar(im1, ax=ax1, shrink=0.7) 

            try:
                ax2 = self.axs[1, axis_index]
            except IndexError:
                ax2 = self.axs[1]

            im2 = ax2.imshow(all_var, origin='lower', cmap='viridis', 
                                extent=(x_interpolation_range[0], x_interpolation_range[1],
                                    y_interpolation_range[0], y_interpolation_range[1]))
            ax2.set_xlim([x_interpolation_range[0], x_interpolation_range[1]])
            ax2.set_ylim([y_interpolation_range[0], y_interpolation_range[1]])     
            if match_scale:
                im2.norm.autoscale([var_min,var_max])
            ax2.set_title(f'Variance – Combined')
            ax2.ticklabel_format(useOffset=False)
            ax2.tick_params(axis='both', which='major', labelsize=7)
            ax2.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            self.fig.colorbar(im2, ax = ax2, shrink=0.7)

        plt.show()

    def plot_ranked_variogram(self, bin_center, gamma, models_dict, ax, vario_x_max: float=30.0):
        r"""Plots emperical variogram, and different variogram models fitted
            to the empirical variogram. NOTE: Probably buggy. 

            Parameters
            ----------

            bin_center: :class:`np.ndarray`
                Bin centers as returned by gs.vario_estimate. 
            gamma: :class:`np.ndarray`
                Empirical semivariogram value as found by gs.vario_estimate.
                Plotted against bin_center to make up empirical variogram
                to be fitted against. 
            models_dict: :class:`dict`
                Dictionary of models to iterate through to find best fitting
                model.
            ax: :class:`mpl.Axes`   
                Instance of mpl.Axes to be plotted on.
            vario_x_max: :class:`float`, optional
                X maximum for variogram plot, default is x = 30. 

        """

        plt.figure(self.fig)
        ax.scatter(bin_center, gamma, color="k", label="data",s=15)
        ax.set_title("Variogam model comparison – 2 traversals")
        ax.set_xlabel("Lag distance")
        ax.set_ylabel("Semivariogram")

        for fit_model in models_dict.values():
            fit_model.plot(fig=self.fig,x_max=vario_x_max, ax=ax)
        
        plt.tight_layout()

    def plot_variogram(self, model, ax):
        r"""Plot variogram based on passed in model.

            Parameters
            ----------

            model: :class:`gs.CovModel` or CovModel wrapper
                Variogram model to be plotted.
        """
        plt.figure(self.fig)
        model.plot(ax=ax)
        ax.set_title(f"Fitted {model.name} Variogram")
        ax.set_xlabel("Lag distance")
        ax.set_ylabel("Semivariogram")
        plt.tight_layout()

    def get_global_color_limits(self, z_pred_list: list[np.ndarray], var_list: list[np.ndarray]):

        r"""Calculates the global color minimum and maximum for both z_pred
        and var based on outputs of every plot.

            Parameters
            ----------

            z_pred_list: :class:`list[np.ndarray]`
                List of each plot's interpolation array.
            var_list: :class:`list[np.ndarray]`
                List of each plot's variance array.
            
            Returns
            -------

            global_z_min: :class:`float`
                Global interpolation minimum for colorbar.
            global_z_max: :class:`float`
                Global interpolation maximum for colorbar.
            global_var_min: :class:`float`
                Global variance minimum for colorbar.
            global_var_max: :class:`float`
                Global variance maximum for colorbar.
        """

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