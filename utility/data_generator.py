import numpy as np
import sys
import os
from utility.krige_array import KrigeArr 

class DataGenerator():

    """The DataGenerator class handles generating fake footstep data. 
        Currently it generates two traversals. It returns an array
        that contains the x, y, and stiffness data for both 
        traversals. 

        
        Parameters
        ----------

        steps_x_max: :class:`float`
            The maximum x-coordinate value for the generated steps.
        steps_y_max: :class:`float`
            The maximum y-coordinate value for the generated steps.
        x: :class:`np.ndarray`
            The x-coordinates of the generated steps.
        y: :class:`np.ndarray`
            The y-coordinates of the generated steps.
        stiff: :class:`np.ndarray`
            The stiffness values of the generated steps.

    """

    def __init__(self):
        self.steps_x_max = None
        self.steps_y_max = None
        self.x = None
        self.y = None
        self.stiff = None

    def generate_steps(self,ground_truth_func, step1_x: float, step1_y: float,
                        slope1: float, nsteps1: int, step2_x: float, 
                        step2_y: float, slope2: float, nsteps2: int, 
                        x_dist: float, y_dist: float):

        r"""Generates fake Spirit traversals and returns them in a single
            NDArray that is vertically stacked. Row 0 is x position, row 1
            is y position, and row 2 is stiffness. 

            Parameters
            ----------

            ground_truth_func: :class:`callable` 
                Stiffness function that generates the stiffness values.
            step1_x: :class:`float`
                Starting x position of p0.
            step1_y: :class:`float`
                Starting y position of p0 and p1.
            slope1: :class:`float`
                Slope of first traversal.
            nsteps1: :class:`int`
                Number of steps in first traversal.
            step2_x: :class:`float`
                Starting x position of p2.
            step2_y: :class:`float`
                Starting y position of p1 and p3.
            slope2: :class:`float`
                Slope of first traversal.
            nsteps2: :class:`int`
                Number of steps in first traversal.
            x_dist: :class:`float`
                Horizontal distance between footsteps.
            y_dist: :class:`float`
                Vertical distance between footsteps. 

            Returns
            -------

            combined_array: :class:`np.ndarray` 
                Vertically stacked array containing two traversals of Spirit.
                0th row = x-position, 1st row = y-position, 2nd row = stiffness. 
            
        """

        # Creates initial point for KrigeArr object, then initializes KrigeArr
        # object.
        p0 = np.array([step1_x,step1_y])  
        arr0 = KrigeArr(p0,slope1,y_dist,nsteps1, ground_truth_func)

        p1 = np.array([step1_x + x_dist,step1_y])
        arr1 = KrigeArr(p1,slope1,y_dist,nsteps1, ground_truth_func)

        p2 = np.array([step2_x,step2_y])
        arr2 = KrigeArr(p2,slope2,y_dist,nsteps2, ground_truth_func)

        p3 = np.array([step2_x + x_dist,step2_y])
        arr3 = KrigeArr(p3,slope2,y_dist,nsteps2, ground_truth_func)

        # Concatenates all arrays.
        self.x = np.concatenate((arr0.x_arr,arr1.x_arr,arr2.x_arr,arr3.x_arr))
        self.y = np.concatenate((arr0.y_arr,arr1.y_arr,arr2.y_arr,arr3.y_arr))
        self.stiff = np.concatenate((arr0.stiff_arr,arr1.stiff_arr,
                                     arr2.stiff_arr,arr3.stiff_arr))
        
        # Vertically stacks all arrays such that row 0 = x,
        # row 1 = y, row 2 = z. 
        combined_array = np.vstack((self.x,self.y,self.stiff))

        # Takes maximum of x and y. 
        self.steps_x_max = np.max(self.x)
        self.steps_y_max = np.max(self.y)

        return combined_array
        