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
        x_input_range: list = None,
        y_input_range: list = None,
        transparent: dict = None):
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
                                x_input_range, 
                                y_input_range)
    
    def plot_legs(self, csvparser: CSVParser, match_steps: bool, length_scale: dict,
                    match_scale: bool = False, transparent: dict = None, x_input_range: list = None, 
                    y_input_range: list = None):
        
        x_arr_list= []
        y_arr_list = []
        stiff_arr_list = []
        
        z_pred_list = []
        var_list = []
        kriging_results = {}
        axis_index = 0

        for request in self.mode:
            x, y, stiff, title = csvparser.access_data([request])

            len_scale = length_scale[request]

            x_arr_list.append(x)
            y_arr_list.append(y)
            stiff_arr_list.append(stiff)  

            x_range, y_range = self.organize_kriging_area(x, y, match_steps, 
                                                x_input_range, y_input_range)

            z_pred, var, fitted_model = self.perform_kriging(x,y,stiff,len_scale,
                                                x_range,y_range)           

            z_pred_list.append(z_pred)
            var_list.append(var)
            kriging_results[request] = (z_pred, var, x, y, stiff, title, fitted_model,
                                        x_range,
                                        y_range,
                                        axis_index)
            axis_index += 1

        if len(self.mode) > 1:

            x_arr = np.concatenate(x_arr_list)
            y_arr = np.concatenate(y_arr_list)
            stiff_arr = np.concatenate(stiff_arr_list)

            request = ",".join(self.mode)
            len_scale = length_scale[request]

            x_range, y_range = self.organize_kriging_area(x_arr, y_arr, match_steps,
                                                x_input_range, y_input_range)
            z_pred_arr, var_arr, fitted_model_arr = self.perform_kriging(x_arr, y_arr,
                                    stiff_arr, len_scale,x_range,y_range)

            z_pred_list.append(z_pred_arr)
            var_list.append(var_arr)

            axis_index = len(self.mode)
            kriging_results[request] = (z_pred_arr, var_arr, x_arr, y_arr, stiff_arr, title, fitted_model_arr,
                                        x_range,
                                        y_range,
                                        axis_index)

        zmin, zmax, var_min, var_max = self.get_global_color_limits(z_pred_list, var_list)

        for request, (z_pred, var, x, y, stiff, title, fitted_model, x_range, y_range, axis_index) in kriging_results.items():
            self.plot_leg(axis_index,z_pred,var,x,y,stiff,x_range,y_range,title,match_scale,zmin,zmax,var_min,var_max,transparent)

        plt.tight_layout()
        plt.show()

    def perform_kriging(self, x, y, stiff, len_scale, x_range, y_range):
        krige_model = KrigeModel(x, y, stiff, self.bin_num, len_scale)
        model_type, models_dict, bin_centers, gamma = krige_model.rank_models()
        fitted_model, r2 = krige_model.fit_model(model_type.name)
        z_pred, var = krige_model.execute_kriging(fitted_model, x_range, y_range) 

        return z_pred, var, fitted_model

    def plot_field(self, ax, field, x_range, y_range, alpha, match_scale, colormin, colormax, title, x, y, stiff, field_name):
        im = ax.imshow(field, origin='lower', cmap='viridis', extent=(x_range[0], x_range[1], y_range[0], y_range[1]), alpha=alpha)
        ax.set_xlim([x_range[0], x_range[1]])
        ax.set_ylim([y_range[0], y_range[1]])
        if match_scale:
            im.norm.autoscale([colormin, colormax])
        ax.scatter(x, y, c=stiff, edgecolors='k', cmap='viridis', s=15)
        ax.set_title(f'{field_name} – {title}')
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        self.fig.colorbar(im, ax=ax, shrink=0.7)

    def plot_leg(self, axis_index, z_pred, var, x, y, stiff, x_range, y_range, title, match_scale, zmin, zmax, var_min, var_max, transparent: dict = None):
        font = {'size': 7}
        plt.rc('font', **font)

        z_alpha = np.ones_like(z_pred)
        if transparent is not None:
            bound = transparent['var bound']
            transparency = transparent['transparency']
            z_alpha[var > bound] = 1 - transparency

        fields = [('Interpolation', z_pred, zmin, zmax), ('Variance', var, var_min, var_max)]
        for i, (field_name, field, fmin, fmax) in enumerate(fields):
            ax_field = self.axs[i, axis_index] if len(self.mode) > 1 else self.axs[i]
            self.plot_field(ax_field, field, x_range, y_range, z_alpha, match_scale, fmin, fmax, title, x, y, stiff, field_name)

    
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
        r"""Plot variogram based on passed in model. NOTE: Probably buggy. 

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

    def organize_kriging_area(self, x, y, match_steps: bool, x_input_range,
                                                        y_input_range):

        r"""Returns ranges x_range and y_range based on whether the area
            will match the input data or not.

            Parameters
            ----------
            x: :class:`np.ndarray`
                X position array that determines x-range
            y: :class:`np.ndarray`
                Y position array that determines x-range 
            match_steps: :class:`bool`
                Boolean that determines whether the kriging area will be fitted
                to match the input data. If False, user must input 
                x and y_interpolation_input_range. Passed through from 
                KrigePlotter.
            x_input_range: :class:`list[float]`, optional
                Only necessary to provide if match_steps is False. This is a
                array of length 2, where the 0th index is the lower bound
                of the range, and the 1st index is the upper bound of the range.
                Passed through from KrigePlotter.
            y_input_range: :class:`list[float]`, optional
                Only necessary to provide if match_steps is False. This is a
                array of length 2, where the 0th index is the lower bound
                of the range, and the 1st index is the upper bound of the range.
                Passed through from KrigePlotter.
            
            Returns
            -------
            x_range: :class:`list[float]` 
                Range of x values to interpolate over.
            y_range: :class:`list[float]` 
                Range of y values to interpolate over.

        """

        x_range = [0,0]
        y_range = [0,0]

        if match_steps:
            # If values are given, set those equal
            if x_input_range is not None and y_input_range is not None:
                x_range[0] = x_input_range[0]
                x_range[1] = np.max(x)

                y_range[1] = np.max(y)
                y_range[0]= y_input_range[0]
            else:
                x_range[0] = np.min(x) + 0.000001
                x_range[1] = np.max(x) + 0.000001

                y_range[1] = np.max(y) + 0.000001
                y_range[0] = np.min(y) + 0.000001

        else: # If not match steps, then must pass in values

            if x_input_range is None or y_input_range is None:
                raise BaseException('Missing arguments. If match_steps is false, the four other arguments in organize_kriging_area are required.')

            x_range[0] = x_input_range[0]
            x_range[1] = x_input_range[1]

            y_range[1] = y_input_range[1]
            y_range[0] = y_input_range[0]

        return x_range, y_range