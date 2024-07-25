from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic, ConstantKernel as C
import numpy as np
import time

class GPRegressor():
    
    def __init__(self, length_scale: dict, noise_level: float, sigma_f: float, alpha: float, len_scale_bounds: dict):
        self.length_scale = length_scale 
        self.noise_level = noise_level
        self.sigma_f = sigma_f
        self.alpha = alpha
        self.length_scale_bounds = len_scale_bounds

    def create_kernel(self, request):
        # length scale: controls the smoothness of the function
        # noise_level – Noise level
        # sigma_f – Signal variance

        request_long = str(request + "_long")

        len_scale_bounds = self.length_scale_bounds[request]

        len_scale = self.length_scale[request]
        len_scale_long = self.length_scale[request_long]

        # Define the kernel components
        kernel = (C(self.sigma_f**2, (1e-3, 1e3)) 
                * Matern(length_scale=len_scale, length_scale_bounds=len_scale_bounds, nu=1.0) 
                + WhiteKernel(self.noise_level, noise_level_bounds=(1e-3,1e2)))

        # kernel = (C(self.sigma_f**2, (1e-3, 1e3))
        #         * 0.5 * Matern(length_scale=len_scale, length_scale_bounds=(1e-5,1e5), nu=1.5)
        #         + RBF(length_scale=len_scale_long,length_scale_bounds=(1e-2,1e2))
        #         + WhiteKernel(self.noise_level, noise_level_bounds=(1e-5,1)))
        
        # kernel = (C(self.sigma_f**2, (1e-3, 1e3)) 
        #     * RationalQuadratic(length_scale=len_scale_long,length_scale_bounds=(1e-5,1e5), alpha = self.alpha, alpha_bounds = (1e-5,1e5))
        #     * Matern(length_scale=len_scale, length_scale_bounds=(1e-5,1e5), nu=1.5))
        
        return kernel

    def Gaussian_Estimation(self, data, values, prediction_range,  optimizer: bool, kernel = None):

    # Instantiate the Gaussian Process Regressor
        if (not optimizer):
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, random_state=0)
        else:
            kernel = C(10)*RBF(1.0)
            gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=15)
        
        # Fit the model to the data
        gp.fit(data, values)

        # Make predictions on new data points
        data_new = prediction_range.T
        z_pred, z_std = gp.predict(data_new, return_std=True)
        
        #calculate discrepancy
        information = np.exp(-np.square(z_std))
        kernel = gp.kernel_
        params = kernel.get_params()
        
        # noise_level_optimized = gp.kernel_.get_params()["k2__noise_level"]
    
        return z_pred, z_std, params #, noise_level_optimized