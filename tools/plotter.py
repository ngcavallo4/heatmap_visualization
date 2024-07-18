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
    
    def organize_area(self, x, y, match_steps: bool, x_interpolation_input_range: list = None,
                                                        y_interpolation_input_range: list = None):
        x_interpolation_range = [0,0]
        y_interpolation_range = [0,0]

        r"""Initializes fields self.x_interpolation_range and 
            self.y_interpolation.range based on whether the area will
            match the input data or not.

            Parameters
            ----------

            match_steps: :class:`bool`
                Boolean that determines whether the kriging area will be fitted
                to match the input data. If False, user must input 
                x and y_interpolation_input_range. Passed through from 
                KrigePlotter.
            x_interpolation_input_range: :class:`list[float]`, optional
                Only necessary to provide if match_steps is False. This is a
                array of length 2, where the 0th index is the lower bound
                of the range, and the 1st index is the upper bound of the range.
                Passed through from KrigePlotter.
            y_interpolation_input_range: :class:`list[float]`, optional
                Only necessary to provide if match_steps is False. This is a
                array of length 2, where the 0th index is the lower bound
                of the range, and the 1st index is the upper bound of the range.
                Passed through from KrigePlotter.
        """
        if match_steps:
            # If values are given, set those equal
            if x_interpolation_input_range is not None and y_interpolation_input_range is not None:
                x_interpolation_range[0] = x_interpolation_input_range[0]
                y_interpolation_range[0]= y_interpolation_input_range[0]
                x_interpolation_range[1] = np.max(x)
                y_interpolation_range[1] = np.max(y)
            else:
                x_interpolation_range[0] = np.min(x) + 0.000001
                y_interpolation_range[0] = np.min(y) + 0.000001
                x_interpolation_range[1] = np.max(x) + 0.000001
                y_interpolation_range[1] = np.max(y) + 0.000001

        else: # If not match steps, then must pass in values

            if x_interpolation_input_range is None or y_interpolation_input_range is None:
                raise BaseException('Missing arguments. If match_steps is false, the four other arguments in organize_kriging_area are required.')

            x_interpolation_range[0] = x_interpolation_input_range[0]
            x_interpolation_range[1] = x_interpolation_input_range[1]
            y_interpolation_range[0]= y_interpolation_input_range[0]
            y_interpolation_range[1]= y_interpolation_input_range[1]

        return x_interpolation_range, y_interpolation_range
    
    def plot_heatmap(self, file: str, match_steps:bool, gpregressor: GPRegressor, match_scale: bool = False, optimizer: bool = False, normalize: bool = False):
        csvparser = CSVParser(file)

        self.plot_legs(csvparser, match_steps, gpregressor, match_scale, optimizer, normalize)

    def plot_legs(self, csvparser: CSVParser, match_steps: bool, gpregressor: GPRegressor, match_scale: bool, optimizer: bool, normalize: bool):
        x_arr_list= []
        y_arr_list = []
        stiff_arr_list = []

        z_pred_list = []
        z_std_list = []
        var_list = []
        results = {}
        axis_index = 0

        for request in self.mode:
            x, y, stiff, title = csvparser.access_data([request])

            x_arr_list.append(x)
            y_arr_list.append(y)
            stiff_arr_list.append(stiff)

            all_z_std = None
            all_information_shear = None
            all_z_std = None 

            # # Normalize area
            if normalize:
                x_min, x_max = np.min(x), np.max(x)
                y_min, y_max = np.min(y), np.max(y)
                x_norm = (x - x_min) / (x_max - x_min)
                y_norm = (y - y_min) / (y_max - y_min)
                x_range, y_range = self.organize_area(x_norm, y_norm, True)
                robot_measured_points = np.vstack((x_norm, y_norm)).T
            else:
                x_range, y_range = self.organize_area(x, y, True)
                robot_measured_points = np.vstack((x, y)).T

            estimatedNum = 100
            xx1, xx2 = np.linspace(x_range[0], x_range[1], num=estimatedNum), np.linspace(y_range[0], y_range[1], num=estimatedNum)
            vals = np.array([[x1_, x2_] for x1_ in xx1 for x2_ in xx2])


            kernel = gpregressor.kernel
            z_pred, z_std, information_shear = gpregressor.Gaussian_Estimation(robot_measured_points,  stiff,   vals, optimizer, kernel=kernel)
            z_pred = z_pred.reshape(estimatedNum, estimatedNum).T
            information_shear = information_shear.reshape(estimatedNum, estimatedNum).T
            z_std = z_std.reshape(estimatedNum, estimatedNum).T

            var = np.square(z_std)

            z_pred_list.append(z_pred)
            z_std_list.append(z_std)
            var_list.append(var)

            if normalize:
                results[request] = (z_pred, z_std, var, x_norm, y_norm, stiff, title, x_range, y_range, axis_index)
            else:
                results[request] = (z_pred, z_std, var, x, y, stiff, title, x_range, y_range, axis_index)
            
            axis_index += 1

        if len(self.mode) > 1: # Computing combined legs interpolation 
            
            x_arr = np.concatenate(x_arr_list)
            y_arr = np.concatenate(y_arr_list)
            stiff_arr = np.concatenate(stiff_arr_list)

            if normalize:
                x_min, x_max = np.min(x_arr), np.max(x_arr)
                y_min, y_max = np.min(y_arr), np.max(y_arr)
                x_arr_norm = (x_arr - x_min) / (x_max - x_min)
                y_arr_norm = (y_arr - y_min) / (y_max - y_min)
                x_range, y_range = self.organize_area(x_arr_norm, y_arr_norm, True)
                robot_measured_points = np.vstack((x_arr_norm, y_arr_norm)).T
            else:
                x_range, y_range = self.organize_area(x_arr, y_arr, True)
                robot_measured_points = np.vstack((x_arr, y_arr)).T
    
            estimatedNum = 100
            xx1, xx2 = np.linspace(x_range[0], x_range[1], num=estimatedNum), np.linspace(y_range[0], y_range[1], num=estimatedNum)
            vals = np.array([[x1_, x2_] for x1_ in xx1 for x2_ in xx2])

            kernel = gpregressor.kernel
            z_pred, z_std, information_shear = gpregressor.Gaussian_Estimation(robot_measured_points,  stiff_arr,   vals, optimizer=optimizer , kernel=kernel)
            all_z_pred = z_pred.reshape(estimatedNum, estimatedNum).T
            all_information_shear = information_shear.reshape(estimatedNum, estimatedNum).T
            all_z_std = z_std.reshape(estimatedNum, estimatedNum).T

            all_var = np.square(all_z_std)

            z_pred_list.append(all_z_pred)
            z_std_list.append(all_z_std)
            var_list.append(all_var)

        zmin, zmax, var_min, var_max = self.get_global_color_limits(z_pred_list, var_list)

        for request, (z_pred, z_std, var, x, y, stiff, title, x_range, y_range, axis_index) in results.items():
            font = {'size': 7}
            plt.rc('font', **font)

            try:
                ax1 = self.axs[0, axis_index]
            except IndexError:
                ax1 = self.axs[0]

            im1 = ax1.imshow(z_pred, origin='lower', cmap='viridis', 
                                extent=(x_range[0], x_range[1],
                                    y_range[0], y_range[1]))
            ax1.set_xlim([x_range[0], x_range[1]])
            ax1.set_ylim([y_range[0], y_range[1]])     
            if match_scale:
                im1.norm.autoscale([zmin,zmax])
            ax1.scatter(x, y, c=stiff, edgecolors='k', cmap='viridis') 
            ax1.set_title(f'Interpolation – {title}')
            ax1.ticklabel_format(useOffset=False)
            ax1.tick_params(axis='both', which='major', labelsize=7)
            ax1.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            self.fig.colorbar(im1, ax=ax1, shrink=0.7) 

            try:
                ax2 = self.axs[1, axis_index]
            except IndexError:
                ax2 = self.axs[1]

            im2 = ax2.imshow(var, origin='lower', cmap='viridis', 
                                extent=(x_range[0], x_range[1],
                                    y_range[0], y_range[1]))
            ax2.set_xlim([x_range[0], x_range[1]])
            ax2.set_ylim([y_range[0], y_range[1]])
            if match_scale:
                im2.norm.autoscale([var_min,var_max])
            ax2.set_title(f'Variance – {title}')
            ax2.ticklabel_format(useOffset=False)
            ax2.tick_params(axis='both', which='major', labelsize=7)
            ax2.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            self.fig.colorbar(im2, ax = ax2, shrink=0.7)

        if len(self.mode) > 1: 

            axis_index = len(self.mode)

            font = {'size': 7}
            plt.rc('font', **font)

            try:
                ax1 = self.axs[0, axis_index]
            except IndexError:
                ax1 = self.axs[0]

            im1 = ax1.imshow(all_z_pred, origin='lower', cmap='viridis', 
                                extent=(x_range[0], x_range[1],
                                    y_range[0], y_range[1]))
            ax1.set_xlim([x_range[0], x_range[1]])
            ax1.set_ylim([y_range[0], y_range[1]])     
            if match_scale:
                im1.norm.autoscale([zmin,zmax])
            
            if normalize:
                ax1.scatter(x_arr_norm, y_arr_norm, c=stiff_arr, edgecolors='k', cmap='viridis') 
            else:
                ax1.scatter(x_arr, y_arr, c=stiff_arr, edgecolors='k', cmap='viridis') 

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
                                extent=(x_range[0], x_range[1],
                                    y_range[0], y_range[1]))
            ax2.set_xlim([x_range[0], x_range[1]])
            ax2.set_ylim([y_range[0], y_range[1]])
            if match_scale:
                im2.norm.autoscale([var_min,var_max])
            ax2.set_title(f'Variance – Combined')
            ax2.ticklabel_format(useOffset=False)
            ax2.tick_params(axis='both', which='major', labelsize=7)
            ax2.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            self.fig.colorbar(im2, ax = ax2, shrink=0.7)

        plt.tight_layout()
        plt.show()

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