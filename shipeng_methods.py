from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
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

def Gaussian_Estimation(x, y, prediction_range,  optimizer, noise_level = None, length_scale = None, sigma_f = None):

    # Define the kernel
    noise_level = noise_level
    length_scale = length_scale
    sigma_f = sigma_f * sigma_f
    # kernel = C(sigma_f) * RBF(length_scale,(0.1, 0.3)) 
    
    kernel = C(sigma_f, (1e-3, 1e7)) * RBF(length_scale, (1e-2, 1e7)) + WhiteKernel(noise_level, (0.0001, 10))

    # Instantiate the Gaussian Process Regressor
    if(not optimizer):
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, random_state=0, optimizer=None)
    else:
        gp = GaussianProcessRegressor(kernel=kernel)
    # x = np.array([x])
    # x = x.T
    # Fit the model to the data
    gp.fit(x, y)

    # Make predictions on new data points
    X_new = prediction_range
    
    y_pred, y_std = gp.predict(X_new, return_std=True)
    
    #calculate discrepancy
    
    information = np.exp(-np.square(y_std))
    noise_level_optimized = gp.kernel_.get_params()["k2__noise_level"]
   
    return y_pred, y_std, information#, noise_level_optimized