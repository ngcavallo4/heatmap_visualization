from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel as C
import numpy as np

class GPRegressor():
    
    def __init__(self, length_scale, noise_level, sigma_f):
        self.kernel = self.create_kernel(length_scale,noise_level,sigma_f)

    def create_kernel(self, length_scale, noise_level, sigma_f):
        # Length scale: Controls the smoothness of the function.
        # length_scale – Small length scale to capture rapid changes
        # noise_level – Noise level
        # sigma_f – Signal variance

        sigma_f = sigma_f * sigma_f

        # Define the kernel components

        # kernel = C(sigma_f, (1e-3, 1e3)) * Matern(length_scale, nu=1.5) + WhiteKernel(noise_level, (1e-5, 1))
        kernel = C(sigma_f, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2)) + WhiteKernel(noise_level, (0, 0.2))
        
        return kernel

    def Gaussian_Estimation(self, data, grid, prediction_range,  optimizer, kernel = None):

    # Instantiate the Gaussian Process Regressor
        if (not optimizer):
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, random_state=0, optimizer=None)
        else:
            kernel = C(10)*RBF(1.0)
            gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=15)
        
        # Fit the model to the data
        gp.fit(data, grid)

        # Make predictions on new data points
        data_new = prediction_range
        z_pred, z_std = gp.predict(data_new, return_std=True)
        
        #calculate discrepancy
        information = np.exp(-np.square(z_std))
        
        # noise_level_optimized = gp.kernel_.get_params()["k2__noise_level"]
    
        return z_pred, z_std, information #, noise_level_optimized