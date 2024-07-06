import numpy as np

class KrigeArr():
    
    def __init__(self,p0,slope,distance,num_points, ground_truth_func):
        self.p0 = p0
        self.x_arr = [p0[0]]
        self.y_arr = [p0[1]]
        self.stiff_arr = [ground_truth_func(p0[0],p0[1])]
        self.slope = slope
        self.distance = distance
        self.num_points = num_points
        self.ground_truth_func = ground_truth_func
        
        self.array_setup()
    
    def array_setup(self):
        for i in range(1, self.num_points):
            y_val = self.p0[1] + i*self.distance
            x_val = self.p0[0] + i*(self.distance/self.slope)
            
            self.x_arr.append(x_val)
            self.y_arr.append(y_val)
    
        self.x_arr = np.array(self.x_arr)
        self.y_arr = np.array(self.y_arr)
        self.stiff_arr = self.ground_truth_func(self.x_arr,self.y_arr)
        self.stiff_arr = np.array(self.stiff_arr)

    def print(self):
        print([self.x_arr,self.y_arr,self.stiff_arr])