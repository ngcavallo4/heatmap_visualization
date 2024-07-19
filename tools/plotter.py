import matplotlib.pyplot as plt
import numpy as np
from utility.parse_csv import CSVParser
from tools.gpregressor import GPRegressor
from matplotlib import ticker

class Plotter():
     
    def __init__(self, mode: list[str]):
        self.mode = mode
        self.fig = None
        self.axs = None
        self.ncols = None

        self.initialize_subplots()

    def plot_heatmap(self, file: str, match_steps:bool, gpregressor: GPRegressor, match_scale: bool = False, transparent: dict = None, optimizer: bool = False, normalize: bool = False):
        csvparser = CSVParser(file)

        self.plot_legs(csvparser, match_steps, gpregressor, match_scale, transparent, optimizer, normalize)

    def plot_legs(self, csvparser: CSVParser, match_steps: bool, gpregressor: GPRegressor, match_scale: bool, transparent: dict, optimizer: bool, normalize: bool):
        x_arr_list = []
        y_arr_list = []
        stiff_arr_list = []

        z_pred_list = []
        var_list = []
        results = {}
        axis_index = 0

        for request in self.mode:
            x, y, stiff, title = csvparser.access_data([request])

            x_arr_list.append(x)
            y_arr_list.append(y)
            stiff_arr_list.append(stiff)

            if normalize:
                x, y = self.normalize(x, y)

            x_range, y_range = self.organize_area(x, y, match_steps)
            z_pred, var = self.perform_kriging(gpregressor, x, y, stiff, x_range, y_range, optimizer)

            z_pred_list.append(z_pred)
            var_list.append(var)

            results[request] = (z_pred, var, x, y, stiff, title, x_range, y_range, axis_index)
            axis_index += 1

        if len(self.mode) > 1:
            x_arr = np.concatenate(x_arr_list)
            y_arr = np.concatenate(y_arr_list)
            stiff_arr = np.concatenate(stiff_arr_list)

            request = ",".join(self.mode)

            if normalize:
                x_arr, y_arr = self.normalize_data(x_arr, y_arr)
            
            x_range, y_range = self.organize_area(x_arr, y_arr, True)
            z_pred, var = self.perform_kriging(gpregressor, x_arr, y_arr, stiff_arr, x_range, y_range, optimizer)

            z_pred_list.append(z_pred)
            var_list.append(var)

            axis_index = len(self.mode)
            results[request] = (z_pred, var, x, y, stiff, title, x_range, y_range, axis_index)

        zmin, zmax, var_min, var_max = self.get_global_color_limits(z_pred_list, var_list)

        for request, (z_pred, var, x, y, stiff, title, x_range, y_range, axis_index) in results.items():
            self.plot_leg(axis_index, z_pred, var, x, y, stiff, x_range, y_range, title, match_scale, zmin, zmax, var_min, var_max, transparent)

        plt.tight_layout()
        plt.show()

    def perform_kriging(self, gpregressor, x, y, stiff, x_range, y_range, optimizer):
        estimated_num = 100
        xx1, xx2 = np.linspace(x_range[0], x_range[1], num=estimated_num), np.linspace(y_range[0], y_range[1], num=estimated_num)
        vals = np.array([[x1_, x2_] for x1_ in xx1 for x2_ in xx2])

        robot_measured_points = np.vstack((x, y)).T

        kernel = gpregressor.kernel
        z_pred, z_std, information_shear = gpregressor.Gaussian_Estimation(robot_measured_points, stiff, vals, optimizer, kernel=kernel)
        z_pred = z_pred.reshape(estimated_num, estimated_num).T
        z_std = z_std.reshape(estimated_num, estimated_num).T

        var = np.square(z_std)

        return z_pred, var
    
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

    def plot_leg(self, axis_index, z_pred, var, x, y, stiff, x_range, y_range, title, match_scale, zmin, zmax, var_min, var_max, transparent=None):
        font = {'size': 7}
        plt.rc('font', **font)

        z_alpha = np.ones_like(z_pred)
        if transparent:
            bound = transparent['var bound']
            transparency = transparent['transparency']
            z_alpha[var > bound] = 1 - transparency

        fields = [('Interpolation', z_pred, zmin, zmax), ('Variance', var, var_min, var_max)]
        for i, (field_name, field, fmin, fmax) in enumerate(fields):
            ax_field = self.axs[i, axis_index] if len(self.mode) > 1 else self.axs[i]
            self.plot_field(ax_field, field, x_range, y_range, z_alpha, match_scale, fmin, fmax, title, x, y, stiff, field_name)

        
    def initialize_subplots(self):
            r"""Sets up rows of subplots based on which legs are being plotted.
            """

            nrows = 2

            if len(self.mode) > 1:
                if 'all' not in self.mode:
                    self.ncols = len(self.mode) + 1
                else:
                    self.ncols = len(self.mode)
            elif len(self.mode) == 1:
                self.ncols = 1

            self.fig, self.axs = plt.subplots(nrows,self.ncols,figsize=(17,7))
    
    def organize_area(self, x, y, match_steps: bool, x_input_range = None,
                                                        y_input_range = None):

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
    
    def normalize(self,x,y):
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        x_norm = (x - x_min) / (x_max - x_min)
        y_norm = (y - y_min) / (y_max - y_min)

        return x_norm, y_norm