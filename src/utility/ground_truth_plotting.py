import numpy as np
import matplotlib.pyplot as plt

class GroundTruthPlot():

    def __init__(self, ground_truth_func, x_bounds, y_bounds, num_contours): 
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.num_contours = num_contours
        self.ground_truth_func = ground_truth_func

        self.plot_ground_truth()

    def plot_ground_truth(self):
        grid_x = np.linspace(0.0001, self.x_bounds,100) # N
        grid_y = np.linspace(0.001, self.y_bounds,100) # M
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        grid_stiff = self.ground_truth_func(grid_x,grid_y)
        
        plt.figure(1)
        plt.subplot(2,2,2)
        plt.contourf(grid_x,grid_y, grid_stiff, levels = self.num_contours,cmap='viridis')
        plt.gca().set_aspect('equal',adjustable='box')
        plt.title('Ground Truth')
        plt.xlabel('X Pos')
        plt.ylabel('YPos')
        plt.colorbar()