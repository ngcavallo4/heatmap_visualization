import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

class GroundTruthPlot():

    r"""Class that plots the ground truth function as defined in ground_truth_func.py. 
    
        Attributes
        ----------
        ground_truth_func: :class:`callable`
            Function to be plotted
        x_range: :class:`list`
            Range of values to display the ground truth function over. Oth 
            index is lower bound of x values, 1st index is upper bound. 
        y_range: :class:`list`
            Range of values to display the ground truth function over. Oth 
            index is lower bound of y values, 1st index is upper bound. 
    """

    def __init__(self, ground_truth_func, x_range: list, y_range: list): 
        """Initialize the GroundTruthPlot class with the given parameters.

        Parameters
        ----------
        ground_truth_func: :class:`callable`
            Function to be plotted
        x_range: :class:`list`
            Range of values to display the ground truth function over. Oth 
            index is lower bound of x values, 1st index is upper bound. 
        y_range: :class:`list`
            Range of values to display the ground truth function over. Oth 
            index is lower bound of y values, 1st index is upper bound. 
        """
        self.ground_truth_func = ground_truth_func
        self.x_range = x_range
        self.y_range = y_range

    def plot_ground_truth(self, fig, ax, num_contours):

        r"""Plots the ground truth function as defined in ground_truth_func.py. 

        Parameters
        ----------
        fig: :class:`matplotlib.figure`
            Figure for the plot to be displayed on.
        ax: :class:`matplotlib.Axes`
            Axis instance for the plot to be displayed on.
        num_contours: :class:`int` 
            Number of contours to be displayed, input parameter to 
            matplotlib.contourf.
        """
        grid_x = np.linspace(self.x_range[0], self.x_range[1],100) # N
        grid_y = np.linspace(self.y_range[0], self.y_range[1],100) # M
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        grid_stiff = self.ground_truth_func(grid_x,grid_y)
        contour = ax.contourf(grid_x,grid_y, grid_stiff, levels = num_contours,cmap='viridis')
        ax.set_title('Ground Truth')
        ax.set_xlabel('X Pos')
        ax.set_ylabel('YPos')
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        fig.colorbar(contour, ax= ax, shrink=0.7)