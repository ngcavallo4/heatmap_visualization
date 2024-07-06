import numpy as np
import sys
import os
sys.path.append(os.path.abspath("./src/utility"))
from krige_array import KrigeArr 

class DataGenerator():

    """The DataGenerator class handles generating fake footstep data. 
        Currently it generates two traversals. It returns an array
        that contains the x, y, and stiffness data for both 
        traversals. 
    """

    def __init__(self):
        self.steps_x_max = None
        self.steps_y_max = None
        self.x = None
        self.y = None
        self.stiff = None

    def generate_steps(self,ground_truth_func, step1_x, step1_y, slope1, nsteps1, step2_x, step2_y, slope2, nsteps2, x_dist, y_dist) -> list[float]:

        p0 = np.array([step1_x,step1_y])  
        arr0 = KrigeArr(p0,slope1,y_dist,nsteps1, ground_truth_func)

        p1 = np.array([step1_x + x_dist,step1_y])
        arr1 = KrigeArr(p1,slope1,y_dist,nsteps1, ground_truth_func)

        p2 = np.array([step2_x,step2_y])
        arr2 = KrigeArr(p2,slope2,y_dist,nsteps2, ground_truth_func)

        p3 = np.array([step2_x + x_dist,step2_y])
        arr3 = KrigeArr(p3,slope2,y_dist,nsteps2, ground_truth_func)

        self.x = np.concatenate((arr0.x_arr,arr1.x_arr,arr2.x_arr,arr3.x_arr))
        self.y = np.concatenate((arr0.y_arr,arr1.y_arr,arr2.y_arr,arr3.y_arr))
        self.stiff = np.concatenate((arr0.stiff_arr,arr1.stiff_arr,arr2.stiff_arr,arr3.stiff_arr))

        combined_array = np.vstack((self.x,self.y,self.stiff))

        self.steps_x_max = np.max(self.x)
        self.steps_y_max = np.max(self.y)

        return combined_array
        