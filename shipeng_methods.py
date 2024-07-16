from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel as C
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Configure warnings to always be triggered
warnings.simplefilter("always")

def get_matrix_value(matrix, x, y):
    # Convert normalized coordinates to matrix indices
    ix = np.int32(x * (matrix.shape[0] - 1))
    iy =  np.int32(y * (matrix.shape[1] - 1))
    
    # Access and return the value at the specified index
    return matrix[ix, iy]

def create_kernel(length_scale, noise_level, sigma_f):
    # Length scale: Controls the smoothness of the function.
    # length_scale – Small length scale to capture rapid changes
    # noise_level – Noise level
    # sigma_f – Signal variance

    # Define the kernel components
    kernel = C(sigma_f**2, (1e-3, 1e3)) * Matern(length_scale, nu=1.5) + WhiteKernel(noise_level, (1e-5, 1))
    
    return kernel

def Gaussian_Estimation(data, grid, prediction_range,  optimizer, kernel):

    # Instantiate the Gaussian Process Regressor
    if (not optimizer):
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0, optimizer=None)
    else:

        kernel = C(10)*RBF(1.0)
        gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=10)
    # x = np.array([x])
    # x = x.T
    # Fit the model to the data
    gp.fit(data, grid)

    # Make predictions on new data points
    X_new = prediction_range
    
    y_pred, y_std = gp.predict(X_new, return_std=True)
    
    #calculate discrepancy
    
    information = np.exp(-np.square(y_std))
    # noise_level_optimized = gp.kernel_.get_params()["k2__noise_level"]
   
    return y_pred, y_std, information#, noise_level_optimized