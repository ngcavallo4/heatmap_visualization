import math 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

class VelocityPlotter():

    def __init__(self, d_dict: dict, h: float, r: float, w: float):

        self.d_dict = d_dict
        self.h = h
        self.r = r
        self.w = w

        self.fig = None
        self.axs = None

        self.initialize_subplots()

    def plot(self):
        """This is the plotting function that you call to plot the velocity mesh for
        all stiffness grids predicted by the heatmap. 
        """

        index = 0

        for index, d_mesh in enumerate(self.d_dict.values()):

            row_index = index // 2
            col_index = index % 2
            title = self.find_name(index)
            self.plot_mesh(d_mesh,title, row_index, col_index)
            index += 1

            
        plt.tight_layout()
        plt.show()

    def velocity_function(self,d):
        return ((2*self.r*self.w)/(math.pi))*math.sqrt((1-((d/self.r) + (self.h/self.r) - 1))**2)
    
    def apply_velocity_to_mesh(self, d_mesh: np.ndarray):
        
        vectorized_velocity = np.vectorize(self.velocity_function)
        velocity_mesh = vectorized_velocity(d_mesh)

        return velocity_mesh
    
    def plot_mesh(self, d_mesh, title: str, row_index: int, col_index: int):

        velocity_grid = self.apply_velocity_to_mesh(d_mesh)

        ax = self.axs[row_index, col_index] if self.ncols > 1 else self.axs[col_index]

        nrows, ncols = d_mesh.shape
        extent = (0, ncols, 0, nrows)

        im = ax.imshow(velocity_grid, cmap = 'viridis', origin = 'lower', extent=extent)
        ax.set_title(f'{title}')
        ax.set_xlabel("D (depth of insertion, m)")
        ax.set_ylabel("H (height to belly, m)")
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.set_yticks(ax.get_yticks(), ax.get_yticklabels(), rotation=90)
        cbar = self.fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        cbar.set_label("Velocity (m/s)", rotation = 270, labelpad = 20)

    def initialize_subplots(self):
            r"""Sets up rows of subplots based on which legs are being plotted.
            """

            self.nrows = 2

            length = len(self.d_dict)

            if length <= 2:
                self.ncols = length
            else:
                self.ncols = 2

            self.fig, self.axs = plt.subplots(self.nrows,self.ncols,figsize=(17,7), layout='tight')

    def find_name(self, int):

        match int:
            case 0:
                return "Front Left"
            case 1:
                return "Back Left"
            case 2:
                return "Front Right"
            case 3:
                return "Back Right"

