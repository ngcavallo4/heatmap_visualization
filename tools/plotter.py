import matplotlib.pyplot as plt
import numpy as np
from utility.parse_csv import CSVParser
from tools.gpregressor import GPRegressor
from matplotlib import ticker
from utility.convert_gps import gps_coords_to_meters, convert_gps_to_meters

class Plotter():
    """
        A class to plot heatmaps and interpolation results using Gaussian Process Regression.

        Attributes
        ----------
        mode: :class:`list[str]`
            List of legs to plot.
        fig: :class:`matplotlib.figure.Figure`
            The figure object for the plots.
        axs: :class:`np.ndarray`
            Array of axes objects for subplots.
        ncols: :class:`int`
            Number of columns for the subplots.
        latlon: :class:`bool`
            Boolean indicating whether coordinates are in latitude and longitude.
        """
     
    def __init__(self, leg_list: list[str], ):
        """Initialize the Plotter class with the given leg list.
        
        Parameters
        ----------
        mode: :class:`list[str]`
            List of legs to plot.
        """
        self.leg_list = leg_list
        self.fig = None
        self.axs = None
        self.ncols = None
        self.latlon = None 

        self.initialize_subplots()

    def plot_heatmap(self, file: str, match_steps:bool, gpregressor: GPRegressor, match_scale: bool = False, transparent: dict = None, optimizer: bool = False, latlon: bool = False):
        """Plot heatmap for the given CSV file using Gaussian Process Regression.
        
        Parameters
        ----------
        file: :class:`str`
            Path to the CSV file.
        match_steps: :class:`bool`
            Boolean indicating whether to match steps.
        gpregressor: :class:`GPRegressor`
            Gaussian Process Regressor object.
        match_scale: :class:`bool`, optional
            Boolean indicating whether to match the scale of plots, by default False.
        transparent: :class:`dict`, optional
            Dictionary containing transparency settings, by default None.
        optimizer: :class:`bool`, optional
            Boolean indicating whether to use an optimizer, by default False.
        latlon: :class:`bool`, optional
            Boolean indicating whether coordinates are in latitude and longitude, by default False.
        """

        csvparser = CSVParser(file)
    
        self.latlon = latlon
        self.plot_legs(csvparser, match_steps, gpregressor, match_scale, transparent, optimizer)

    def plot_legs(self, csvparser: CSVParser, match_steps: bool, gpregressor: GPRegressor, match_scale: bool, transparent: dict, optimizer: bool):

        """Plot legs based on the parsed CSV data and Gaussian Process Regression.
        
        Parameters
        ----------
        csvparser: :class:`CSVParser`
            CSV parser object to access data.
        match_steps: :class:`bool`
            Boolean indicating whether to match steps.
        gpregressor: :class:`GPRegressor`
            Gaussian Process Regressor object.
        match_scale: :class:`bool`
            Boolean indicating whether to match the scale of plots.
        transparent: :class:`dict`
            Dictionary containing transparency settings.
        optimizer: :class:`bool`
            Boolean indicating whether to use an optimizer.
        """

        x_arr_list = []
        y_arr_list = []
        stiff_arr_list = []

        z_pred_list = []
        var_list = []
        results = {}
        axis_index = 0

        for request in self.leg_list:
            x, y, stiff, title = csvparser.access_data([request])

            if not self.latlon:
                x, y = gps_coords_to_meters(x,y)
                # x, y = convert_gps_to_meters(x,y) # Alternate method 

            x_arr_list.append(x)
            y_arr_list.append(y)
            stiff_arr_list.append(stiff)

            x_range, y_range = self.organize_area(x, y, match_steps)
            z_pred, var = self.perform_kriging(gpregressor, x, y, stiff, x_range, y_range, optimizer, request)

            z_pred_list.append(z_pred)
            var_list.append(var)

            results[request] = (z_pred, var, x, y, stiff, title, x_range, y_range, axis_index)
            axis_index += 1

        if len(self.leg_list) > 1:
            x_combined = np.concatenate(x_arr_list)
            y_combined = np.concatenate(y_arr_list)
            stiff_combined = np.concatenate(stiff_arr_list)

            combined_request = ",".join(self.leg_list)

            x_range_combined, y_range_combined = self.organize_area(x_combined, y_combined, match_steps)
            z_pred_combined, var_combined = self.perform_kriging(gpregressor, x_combined, y_combined, stiff_combined, x_range_combined, y_range_combined, optimizer, combined_request)

            z_pred_list.append(z_pred_combined)
            var_list.append(var_combined)

            results[combined_request] = (z_pred_combined, var_combined, x_combined, y_combined, stiff_combined, 'Combined', x_range_combined, y_range_combined, axis_index)

        zmin, zmax, var_min, var_max = self.get_global_color_limits(z_pred_list, var_list)

        for request, (z_pred, var, x, y, stiff, title, x_range, y_range, axis_index) in results.items():
            self.plot_leg(axis_index, z_pred, var, x, y, stiff, x_range, y_range, title, match_scale, zmin, zmax, var_min, var_max, transparent)

        plt.tight_layout()
        plt.show()
        


    def perform_kriging(self, gpregressor, x, y, stiff, x_range, y_range, optimizer, request) -> tuple[np.ndarray, np.ndarray]:

        """Perform kriging to interpolate data using Gaussian Process Regression.
        
        Parameters
        ----------
        gpregressor: :class:`GPRegressor`
            Gaussian Process Regressor object.
        x: :class:`np.ndarray`
            X coordinate array.
        y: :class:`np.ndarray`
            Y coordinate array.
        stiff: :class:`np.ndarray`
            Stiffness values array.
        x_range: :class:`list[float]`
            Range of X values to interpolate over.
        y_range: :class:`list[float]`
            Range of Y values to interpolate over.
        optimizer: :class:`bool`
            Boolean indicating whether to use an optimizer.
        request: :class:`str`
            Request identifier for the current leg.
        
        Returns
        -------
        z_pred: :class:`np.ndarray`
            Predicted values array.
        var: :class:`np.ndarray`
            Variance of the predicted values.
        """

        estimated_num = 100
        xx1, xx2 = np.linspace(x_range[0], x_range[1], num=estimated_num), np.linspace(y_range[0], y_range[1], num=estimated_num)
        vals = np.array([[x1_, x2_] for x1_ in xx1 for x2_ in xx2]).T

        robot_measured_points = np.vstack((x, y)).T

        kernel = gpregressor.create_kernel(request)
        z_pred, z_std, params = gpregressor.Gaussian_Estimation(robot_measured_points, stiff, vals, optimizer, kernel=kernel)
        z_pred = z_pred.reshape(estimated_num, estimated_num).T
        z_std = z_std.reshape(estimated_num, estimated_num).T

        print(f"\nLeg {request} params: {params}\n")

        var = np.square(z_std)

        # Stiffness should never be negative, so any negative values we round to zero. 
        z_pred[z_pred < 0] = 0

        return z_pred, var

    def plot_leg(self, axis_index, z_pred, var, x, y, stiff, x_range, y_range, title, match_scale, zmin, zmax, var_min, var_max, transparent: dict=None):

        """Plot individual leg with interpolation and variance fields.
        
        Parameters
        ----------
        axis_index: :class:`int`
            Index of the subplot axis.
        z_pred: :class:`np.ndarray`
            Predicted values array.
        var: :class:`np.ndarray`
            Variance of the predicted values.
        x: :class:`np.ndarray`
            X coordinate array.
        y: :class:`np.ndarray`
            Y coordinate array.
        stiff: :class:`np.ndarray`
            Stiffness values array.
        x_range: :class:`list[float]`
            Range of X values to interpolate over.
        y_range: :class:`list[float]`
            Range of Y values to interpolate over.
        title: :class:`str`
            Title for the plot.
        match_scale: :class:`bool`
            Boolean indicating whether to match the scale of plots.
        zmin: :class:`float`
            Global minimum value for color scaling of interpolation.
        zmax: :class:`float`
            Global maximum value for color scaling of interpolation.
        var_min: :class:`float`
            Global minimum value for color scaling of variance.
        var_max: :class:`float`
            Global maximum value for color scaling of variance.
        transparent: :class:`dict`, optional
            Dictionary containing transparency settings, by default None.
        """

        z_alpha = np.ones_like(z_pred)
        if transparent is not None:
            var_percent = transparent['var %']
            transparency = transparent['transparency']
            transparency_var = var_percent*var_max 
            z_alpha[var > transparency_var] = 1 - transparency

        fields = [('Interpolation', z_pred, zmin, zmax, "Stiffness (N/m)"), ('Variance', var, var_min, var_max, "(N/m)^2")]
        for i, (field_name, field, fmin, fmax, field_unit) in enumerate(fields):
            ax_field = self.axs[i, axis_index] if len(self.leg_list) > 1 else self.axs[i]
            self.plot_field(ax_field, field, x_range, y_range, z_alpha, match_scale, fmin, fmax, title, x, y, stiff, field_name, field_unit)
    
    def plot_field(self, ax, field, x_range, y_range, alpha, match_scale, colormin, colormax, title, x, y, stiff, field_name, field_unit):

        """Plot a field (interpolation or variance) on a given axis.
        
        Parameters
        ----------
        ax: :class:`plt.Axes`
            The axis to plot on.
        field: :class:`np.ndarray`
            The field data to plot.
        x_range: :class:`list[float]`
            Range of X values to plot.
        y_range: :class:`list[float]`
            Range of Y values to plot.
        alpha: :class:`np.ndarray`
            Alpha values for transparency.
        match_scale: :class:`bool`
            Boolean indicating whether to match the scale of plots.
        colormin: :class:`float`
            Minimum color value for the plot.
        colormax: :class:`float`
            Maximum color value for the plot.
        title: :class:`str`
            Title for the plot.
        x: :class:`np.ndarray`
            X coordinate array.
        y: :class:`np.ndarray`
            Y coordinate array.
        stiff: :class:`np.ndarray`
            Stiffness values array.
        field_name: :class:`str`
            Name of the field being plotted.
        field_unit: :class:`str`
            Unit of the field being plotted.
        """

        if field_name == "Interpolation":
            im = ax.imshow(field, origin='lower', cmap='viridis', extent=(x_range[0], x_range[1], y_range[0], y_range[1]), alpha=alpha)
        else:
            im = ax.imshow(field, origin='lower', cmap='viridis', extent=(x_range[0], x_range[1], y_range[0], y_range[1]))

        ax.set_xlim([x_range[0], x_range[1]])
        ax.set_ylim([y_range[0], y_range[1]])

        if match_scale:
            im.norm.autoscale([colormin, colormax])
        if field_name == "Interpolation":
            ax.scatter(x, y, c=stiff, edgecolors='k', cmap='viridis', s=15)
        ax.set_title(f'{field_name} – {title}')
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        cbar = self.fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

        if field_name == "Variance":
            min_var = np.round(np.min(field), decimals = 5)
            max_var = np.round(np.max(field), decimals = 5)
            minor_ticks = [min_var, max_var]
            cbar.ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            cbar.ax.tick_params(which='minor', color='red')
            cbar.ax.yaxis.set_tick_params(which='minor', length=4)
            cbar.ax.yaxis.set_ticklabels([f"Min: {min_var}", f"Max: {max_var}"], minor=True, color = 'red')
            print(f"{title} var min: {min_var}\n")
            print(f"{title} var max: {max_var}\n")
        if field_name == "Variance":
            cbar.set_label(f'{field_unit}', rotation=270, labelpad = -15)
        else:
            cbar.set_label(f'{field_unit}', rotation=270, labelpad = 15)

        offset_text = ax.xaxis.get_offset_text()
        offset_text.set_size(7)
        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_size(7)
        if not self.latlon:
            ax.set_xlabel("X Position (m)",fontsize=10)
            ax.set_ylabel("Y Position (m)",loc='center',fontsize=10)
        else: 
            ax.set_xlabel("Longitude(º)",fontsize=10)
            ax.xaxis.set_label_coords(0.5, -0.19)
            ax.set_ylabel("Latitude(º)",loc='center',fontsize=10)
            
    def initialize_subplots(self):
            r"""Sets up rows of subplots based on which legs are being plotted.
            """

            nrows = 2

            if len(self.leg_list) > 1:
                if 'all' not in self.leg_list:
                    self.ncols = len(self.leg_list) + 1
                else:
                    self.ncols = len(self.leg_list)
            elif len(self.leg_list) == 1:
                self.ncols = 1

            self.fig, self.axs = plt.subplots(nrows,self.ncols,figsize=(17,7), layout='tight')
    
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
        if self.latlon:
            offset = 0.000001
        else: 
            offset = 0.25

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
                x_range[0] = np.min(x) - offset
                x_range[1] = np.max(x) + offset

                y_range[1] = np.max(y) + offset
                y_range[0] = np.min(y) - offset 

        else: # If not match steps, then must pass in values

            if x_input_range is None or y_input_range is None:
                raise BaseException('Missing arguments. If match_steps is false, the four other arguments in organize_area are required.')

            x_range[0] = x_input_range[0]
            x_range[1] = x_input_range[1]

            y_range[1] = y_input_range[1]
            y_range[0] = y_input_range[0]

        return x_range, y_range
    
    def get_global_color_limits(self, z_pred_list: list[np.ndarray], var_list: list[np.ndarray]) -> tuple[float]:

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

        global_v_max = np.max(var_list) + 0.1
        global_v_min = np.min(var_list) - 0.1

        global_z_max = np.max(z_pred_list) + 0.1
        global_z_min = np.min(z_pred_list) - 0.1
        
        return global_z_min, global_z_max, global_v_min, global_v_max