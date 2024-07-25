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
        leg_list: :class:`list[str]`
            List of legs to be plotted & interpolated. 
        match_steps: :class:`bool`
            A boolean passed on from organize_kriging_area.
        bin_num: :class:`int`
            Number of bins for the estimated emperical variogram.
        length_scale: Optional, :class:`dict`
            Length scale of GSTools variogram. Each length scale corresponds
            to a specific leg or combination of legs. 
    """

    def __init__(self, leg_list: list[int] = None, bin_num: int=30, length_scale: dict = {'0': 1.0, '1': 1.0, '2': 1.0, '3': 1.0}):
        
        # Initializes fields
        self.leg_list = leg_list
        self.bin_num = bin_num
        self.length_scale = length_scale
        self.fig = None
        self.axs = None

    def initialize_subplots(self, num_rows: int=None, num_cols: int=None):
        r"""Sets up rows of subplots based on which legs are being plotted. 
            Must be called externally before plot_heatmap. No need to store
            the figure and axs that are returned unless plotting additional 
            plots. If just using this class for the plot_fields function,
            pass in 'null' to the leg_list argument when initializing an instance
            of the class. Then, pass in the optional arguments num_rows and
            num_cols. 

            Parameters
            ----------
            num_rows: :class:`int`, optional
                Only used if user desires to use the plot_fields function
                separately from interpolation features (plotting multiple legs
                using data from a CSV). Number of rows in the subplot.
            num_cols: :class:`int`, optional
                Only used if user desires to use the plot_fields function
                separately from interpolation features (plotting multiple legs
                using data from a CSV). Number of rows in the subplot.

        """

        nrows = 2

        # If length of leg list is greater than one, need to add one to plot
        # combined leg plot
        if len(self.leg_list) > 1:
            ncols = len(self.leg_list) + 1 
        # 'null' is keyword for passing in predetermined number of rows and columns
        elif 'null' in self.leg_list:
            nrows = num_rows
            ncols = num_cols
        # If only plotting one leg, num of cols = 1
        elif len(self.leg_list) == 1:
                ncols = 1
        
        self.fig, self.axs = plt.subplots(nrows,ncols,figsize=(15,7))

        # Returns fig & axs for external plotting, but not necessary to store the
        # returned values 
        return self.fig, self.axs
    
    def plot_heatmap(self, file: str, match_steps: bool, match_scale: bool = False,
        x_input_range: list = None,
        y_input_range: list = None,
        transparent: dict = None):
        r"""Plots heatmap. Calls helper function plot_legs.

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
            x_input_range: :class:`list[float]`, optional
                Only necessary to provide if match_steps is False. This is a
                array of length 2, where the 0th index is the lower bound
                of the range, and the 1st index is the upper bound of the range.
            y_input_range: :class:`list[float]`, optional
                Only necessary to provide if match_steps is False. This is a
                array of length 2, where the 0th index is the lower bound
                of the range, and the 1st index is the upper bound of the range.
            transparent: :class:`dict`, optional
                Dictionary of values containing bounds of variance and transparency
                values. If the values of an interpolation point are above a certain
                variance, that point will be plotted at the given transparency. 
        """
        
        csvparser = CSVParser(file)

        self.plot_legs(csvparser, match_steps, 
                            self.length_scale, match_scale, transparent,
                                x_input_range, 
                                y_input_range)
    
    def plot_legs(self, csvparser: CSVParser, match_steps: bool, length_scale: dict,
                    match_scale: bool = False, transparent: dict = None, x_input_range: list = None, 
                    y_input_range: list = None):
        r"""Plots each leg's interpolation, and then each leg combined if there
        are multiple legs. 

        Parameters
        ----------
        csvparser: :class:`CSVParser`
            CSV parser object to access data.
        match_steps: :class:`bool`
            Boolean indicating whether to match steps.
        length_scale: :class:`dict`
            Dictionary mapping leg numbers to corresponding length scale values.
        match_scale: :class:`bool`, optional
            Boolean indicating whether to match the scale of plots, by default False.
        transparent: :class:`dict`, optional
            Dictionary containing transparency settings, by default None.
        x_input_range: :class:`list[float]`, optional
            Array of length 2 with the lower and upper bounds of the x range, required if match_steps is False, by default None.
        y_input_range: :class:`list[float]`, optional
            Array of length 2 with the lower and upper bounds of the y range, required if match_steps is False, by default None.
        """
        x_arr_list= []
        y_arr_list = []
        stiff_arr_list = []
        
        z_pred_list = []
        var_list = []
        kriging_results = {}
        axis_index = 0

        # Iterates through leg requests in leg_list 
        for leg in self.leg_list:
            x, y, stiff, title = csvparser.access_data([leg])

            # length_scale is a dictionart passed into KrigingPlotter when it is initialized.
            len_scale = length_scale[leg]

            # These lists gather all the individual legs together for the combined
            # leg interpolation and plotting.
            x_arr_list.append(x)
            y_arr_list.append(y)
            stiff_arr_list.append(stiff)  

            # Organizes the interpolation area.
            x_range, y_range = self.organize_kriging_area(x, y, match_steps, 
                                                x_input_range, y_input_range)

            # Executes the kriging based on the interpolation area and length scale.
            z_pred, var, fitted_model = self.perform_kriging(x,y,stiff,len_scale,
                                                x_range,y_range)           

            # Adds interpolation and variance to lists to determine color bar ranges
            # so color bar ranges can match if match_scale is true.
            z_pred_list.append(z_pred)
            var_list.append(var)

            # Adds results to a dictionaty for iterating through in plotting function
            kriging_results[leg] = (z_pred, var, x, y, stiff, title, fitted_model,
                                        x_range,
                                        y_range,
                                        axis_index)
            axis_index += 1

        # If there are multiple legs, must repeat the process for the combined legs
        if len(self.leg_list) > 1:

            # Arrays representing the x,y positions and stiffness of the combined
            # datasets
            x_arr = np.concatenate(x_arr_list)
            y_arr = np.concatenate(y_arr_list)
            stiff_arr = np.concatenate(stiff_arr_list)

            # In the length_scale dictionary, there should be an entry for the 
            # combination of legs in the leg list. This retrieves that entry.
            leg = ",".join(self.leg_list)
            len_scale = length_scale[leg]

            # Repeats kriging process for combined legs.
            x_range, y_range = self.organize_kriging_area(x_arr, y_arr, match_steps,
                                                x_input_range, y_input_range)
            z_pred_arr, var_arr, fitted_model_arr = self.perform_kriging(x_arr, y_arr,
                                    stiff_arr, len_scale,x_range,y_range)

            # Adds interpolation and variance to lists to determine color bar ranges
            # so color bar ranges can match if match_scale is true.
            z_pred_list.append(z_pred_arr)
            var_list.append(var_arr)

            axis_index = len(self.leg_list)
            # Adds results to a dictionarty for iterating through in plotting function
            kriging_results[leg] = (z_pred_arr, var_arr, x_arr, y_arr, stiff_arr, title, fitted_model_arr,
                                        x_range,
                                        y_range,
                                        axis_index)

        # Gets global color limits for use in plotting
        zmin, zmax, var_min, var_max = self.get_global_color_limits(z_pred_list, var_list)

        # Iterates through kriging_results dict and plots each leg
        for leg, (z_pred, var, x, y, stiff, title, fitted_model, x_range, y_range, axis_index) in kriging_results.items():
            self.plot_leg(z_pred,var,x,y,stiff,x_range,y_range,
                        title,axis_index,match_scale,zmin,zmax,
                        var_min,var_max,transparent)

        plt.tight_layout()
        plt.show()

    def plot_leg(self, z_pred: np.ndarray, var: np.ndarray,
                x: np.ndarray, y: np.ndarray, stiff: np.ndarray, x_range: list[float],
                y_range: list[float], title: str, axis_index: int, match_scale: bool=True, zmin: float=None,
                zmax: float=None, var_min: float=None, var_max: float=None, transparent: dict=None):
        """Plot an individual leg with interpolation and variance fields.

        Parameters
        ----------
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
        axis_index: :class:`int`
            Index of the subplot axis.
        match_scale: :class:`bool`, optional
            Boolean indicating whether to match the scale of plots, by default True.
        zmin: :class:`float`, optional
            Global minimum value for color scaling of interpolation, by default None.
        zmax: :class:`float`, optional
            Global maximum value for color scaling of interpolation, by default None.
        var_min: :class:`float`, optional
            Global minimum value for color scaling of variance, by default None.
        var_max: :class:`float`, optional
            Global maximum value for color scaling of variance, by default None.
        transparent: :class:`dict`, optional
            Dictionary containing transparency settings, by default None.
        """
        
        font = {'size': 7}
        plt.rc('font', **font)

        # Creates array of transparency values based on variance dictionary
        # where 'var %' corresponds to the percentage (0-1) of max variance that
        # interpolation values can have before they become transparent, and 'transparency'
        # corresponds to the transparency value from 0-1 where 1 is 100% transparent
        # and 0 is 0% transparent. 
        z_alpha = np.ones_like(z_pred)
        if transparent is not None:
            var_percent = transparent['var %']
            transparency = transparent['transparency']
            z_alpha[var > var_percent*var_max] = 1 - transparency

        # Plots interpolation and variance using plot_field function
        fields = [('Interpolation', z_pred, zmin, zmax), ('Variance', var, var_min, var_max)]
        for i, (field_name, field, colormin, colormax) in enumerate(fields):
            ax_field = self.axs[i, axis_index] if len(self.leg_list) > 1 else self.axs[i]
            self.plot_field(ax_field, field, x_range, y_range, z_alpha, title,
                        x,y,stiff, field_name, match_scale, colormin, colormax) 

    def plot_field(self, ax, field: np.ndarray, x_range: list[float], y_range: list[float],
                alpha: np.ndarray, title: str, x: np.ndarray, y: np.ndarray,
                stiff: np.ndarray, field_name: str, match_scale: bool=True,
                colormin: float=None, colormax: float=None):
        
        """Plot a field (interpolation or variance) on a given axis.

        Parameters
        ----------
        ax: :class:`matplotlib.axes.Axes`
            The axis to plot on.
        field: :class:`np.ndarray`
            The field data to plot.
        x_range: :class:`list[float]`
            Range of X values to plot.
        y_range: :class:`list[float]`
            Range of Y values to plot.
        alpha: :class:`np.ndarray`
            Alpha values for transparency.
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
        match_scale: :class:`bool`, optional
            Boolean indicating whether to match the scale of plots, by default True.
        colormin: :class:`float`, optional
            Minimum color value for the plot, by default None.
        colormax: :class:`float`, optional
            Maximum color value for the plot, by default None.
        """
        
        # Plotting field using imshow
        if field_name == "Interpolation":
            im = ax.imshow(field, origin='lower', cmap='viridis', extent=(x_range[0], x_range[1], y_range[0], y_range[1]), alpha=alpha)
        else:
            im = ax.imshow(field, origin='lower', cmap='viridis', extent=(x_range[0], x_range[1], y_range[0], y_range[1]))
        ax.set_xlim([x_range[0], x_range[1]])
        ax.set_ylim([y_range[0], y_range[1]])

        # If match_scale, normalize the image colorscale to the passed in max and min values.
        # These are determined by get_global_color_limits, which includes the combined leg
        # values. 
        if match_scale:
            im.norm.autoscale([colormin, colormax])
        # Scatters footsteps for interpolation plot
        if field_name == 'Interpolation':
            ax.scatter(x, y, c=stiff, edgecolors='k', cmap='viridis', s=15)
        ax.set_title(f'{field_name} – {title}')
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        cbar = self.fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

    def perform_kriging(self, x: np.ndarray, y: np.ndarray, stiff: np.ndarray, len_scale: float, x_range: list[float], y_range: list[float]):
        """Perform kriging on passed in values.

        Parameters
        ----------
        x: :class:`np.ndarray`
            X coordinate array.
        y: :class:`np.ndarray`
            Y coordinate array.
        stiff: :class:`np.ndarray`
            Stiffness values array.
        len_scale: :class:`float`
            Length scale for the kriging model.
        x_range: :class:`list[float]`
            Range of X values to interpolate over.
        y_range: :class=`list[float]`
            Range of Y values to interpolate over.

        Returns
        -------
        z_pred: :class:`np.ndarray`
            Predicted values array.
        var: :class:`np.ndarray`
            Variance of the predicted values.
        fitted_model: :class:`object`
            The fitted kriging model.
        """


        # Performs kriging on passed in values 
        krige_model = KrigeModel(x, y, stiff, self.bin_num, len_scale)
        model_type, models_dict, bin_centers, gamma = krige_model.rank_models()
        fitted_model, r2 = krige_model.fit_model(model_type.name)
        z_pred, var = krige_model.execute_kriging(fitted_model, x_range, y_range) 

        z_pred[z_pred < 0] = 0

        return z_pred, var, fitted_model
    
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

        global_v_max = np.max(var_list)
        global_v_min = np.min(var_list)

        global_z_max = np.max(z_pred_list)
        global_z_min = np.min(z_pred_list)

        return global_z_min, global_z_max, global_v_min, global_v_max

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

        x_range = [0.0,0.0]
        y_range = [0.0,0.0]

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