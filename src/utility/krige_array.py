import numpy as np

class KrigeArr():

    r"""Class that generates toy Spirit data.

        Parameter
        ---------

        p0: :class:`np.ndarray`
            Initial point, array of length 2. Index 0 is x-position, 
            index 1 is y-position. 
        slope: :class:`float`
            Slope of the footstep traverse. 
        y_dist: :class:`float`
            Vertical distance between points.
        num_points: :class:`int`
            Number of points along the traversal. 
        ground_truth_func: :class:`callable`
            Function that generates the stiffness values based on x and y values.
    """
    
    def __init__(self, p0: np.ndarray, slope: float, y_dist: float, num_points: int, ground_truth_func):
        self.p0 = p0
        self.x_arr = [p0[0]]
        self.y_arr = [p0[1]]
        self.stiff_arr = [ground_truth_func(p0[0],p0[1])]
        self.slope = slope
        self.y_dist = y_dist
        self.num_points = num_points
        self.ground_truth_func = ground_truth_func
        
        self.array_setup()
    
    def array_setup(self):

        r"""Generates traversals based on initial point p0. Creates arrays for x
            position, y position, and stiffness. Stiffness values are generated 
            based on the function in ground_truth_function.ground_truth_function().
        """
        for i in range(1, self.num_points):
            # "Steps" along the line set by input parameters.
            y_val = self.p0[1] + i*self.y_dist
            x_val = self.p0[0] + i*(self.y_dist/self.slope)
            
            # Appends new values to x and y arrays.
            self.x_arr.append(x_val)    
            self.y_arr.append(y_val)
    
        # Makes arrays into np.arrays, applies ground_truth_function to create
        # stiffness array. 
        self.x_arr = np.array(self.x_arr)
        self.y_arr = np.array(self.y_arr)
        self.stiff_arr = self.ground_truth_func(self.x_arr,self.y_arr)
        self.stiff_arr = np.array(self.stiff_arr)