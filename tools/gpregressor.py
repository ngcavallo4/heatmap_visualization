from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic, ConstantKernel as C
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker 
from scipy.interpolate import interp1d
import scipy.stats 

class GPRegressor():
    """
    A class to perform Gaussian Process Regression with different kernel configurations.

    Attributes
    ----------
    length_scale: :class:`dict`
        Dictionary mapping leg numbers to corresponding length scale values.
    noise_level: :class:`float`
        Noise level parameter for the Gaussian Process.
    sigma_f: :class:`float`
        Signal variance parameter for the Gaussian Process.
    alpha: :class:`float`
        Alpha parameter for the RationalQuadratic kernel.
    length_scale_bounds: :class:`dict`
        Dictionary mapping leg numbers to corresponding length scale bounds.
    """
    
    def __init__(self, length_scale: dict, noise_level: float, sigma_f: dict, nu: float, alpha: float):
        """
        Initialize the GPRegressor class with the given parameters.

        Parameters
        ----------
        length_scale: :class:`dict`
            Dictionary mapping leg numbers to corresponding length scale values.
        noise_level: :class:`float`
            Noise level parameter for the Gaussian Process.
        sigma_f: :class:`float`
            Constant value parameter for the Gaussian Process.
        alpha: :class:`float`
            Alpha parameter for the RationalQuadratic kernel.
        nu: :class:`float`
            Nu paramater for the Matérn kernel. 
        len_scale_bounds: :class:`dict`
            Dictionary mapping leg numbers to corresponding length scale bounds.
        """

        self.length_scale = length_scale 
        self.noise_level = noise_level
        self.sigma_f = sigma_f
        self.nu = nu
        self.alpha = alpha

    def create_kernel(self):
        """
        Create a Gaussian Process kernel based on the request identifier.

        Parameters
        ----------
        request: :class:`str`
            Request identifier for the current leg.

        Returns
        -------
        kernel: :class:`sklearn.gaussian_process.kernels.Kernel`
            The constructed kernel for the Gaussian Process.
        """

        # length scale: controls the smoothness of the function
        # noise_level – Noise level
        # sigma_f – Signal variance


        len_scale_bounds = self.length_scale["bounds"] # tuple
        len_scale = self.length_scale["val"]
        noise_bounds = self.noise_level["bounds"] # tuple
        noise = self.noise_level["val"]
        sigma_f_bounds = self.sigma_f["bounds"] # tuple
        sigma_f_val = self.sigma_f["val"] # tuple

        # # Define the kernel components
        # kernel = (C(sigma_f_val**2, constant_value_bounds=sigma_f_bounds) 
        #         * Matern(length_scale=len_scale, length_scale_bounds=len_scale_bounds, nu=self.nu)
        #         + WhiteKernel(noise, noise_bounds))

        kernel = (C(sigma_f_val**2, constant_value_bounds=sigma_f_bounds) 
                * RBF(length_scale=len_scale, length_scale_bounds=len_scale_bounds))

        return kernel

    def Gaussian_Estimation(self, data, values, prediction_range,  optimizer: bool, kernel = None, normalize_by: str = "median"):
        """
        Perform Gaussian Process Regression to fit the model and make predictions.

        Parameters
        ----------
        data: :class:`np.ndarray`
            Array of input data points.
        values: :class:`np.ndarray`
            Array of values corresponding to the input data points.
        prediction_range: :class:`np.ndarray`
            Array of points where predictions are to be made.
        optimizer: :class:`bool`
            Boolean indicating whether to use an optimizer.
        kernel: :class:`sklearn.gaussian_process.kernels.Kernel`, optional
            Kernel to be used for the Gaussian Process, by default None.

        Returns
        -------
        z_pred: :class:`np.ndarray`
            Predicted values array.
        z_std: :class:`np.ndarray`
            Standard deviation of the predicted values.
        params: :class:`dict`
            Parameters of the fitted kernel.
        """
        match normalize_by:
            case "median":
                normalize = np.median(values)
            case "mean":
                normalize = np.mean(values)
            case "mode":
                scipy.stats.mode(values).mode

        values = np.subtract(values, normalize)

        # values[values < 0] = 0

    # Instantiate the Gaussian Process Regressor
        # if optimizer:
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, random_state=0)
        # else:
        #     kernel = C(10)*RBF(1.0)
        #     gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=10)
        
        # Fit the model to the data
        gp.fit(data, values)

        # Make predictions on new data points
        prediction_range = prediction_range.T
        z_pred, z_std = gp.predict(prediction_range, return_std=True)
        
        #calculate discrepancy
        # information = np.exp(-np.square(z_std))
        kernel = gp.kernel_
        params = kernel.get_params()

        # Stiffness should never be negative, so any negative values we round to zero. 

        z_pred = z_pred + normalize
        # params = (params['k1'], f"noise level: {params['k2__noise_level']}")
    
        return z_pred, z_std, params
    
    def Gaussian_Estimation_1D(self, data, values, prediction_range,  optimizer: bool, kernel = None):
        """
        Perform Gaussian Process Regression to fit the model and make predictions.

        Parameters
        ----------
        data: :class:`np.ndarray`
            Array of input data points.
        values: :class:`np.ndarray`
            Array of values corresponding to the input data points.
        prediction_range: :class:`np.ndarray`
            Array of points where predictions are to be made.
        optimizer: :class:`bool`
            Boolean indicating whether to use an optimizer.
        kernel: :class:`sklearn.gaussian_process.kernels.Kernel`, optional
            Kernel to be used for the Gaussian Process, by default None.

        Returns
        -------
        z_pred: :class:`np.ndarray`
            Predicted values array.
        z_std: :class:`np.ndarray`
            Standard deviation of the predicted values.
        params: :class:`dict`
            Parameters of the fitted kernel.
        """

    # Instantiate the Gaussian Process Regressor
        if optimizer:
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, random_state=0)
        else:
            kernel = C(10)*RBF(1.0)
            gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=15)
        
        # Fit the model to the data
        gp.fit(data, values)

        # Make predictions on new data points
        # prediction_range = prediction_range.T
        z_pred, z_std = gp.predict(prediction_range, return_std=True)
        
        #calculate discrepancy
        # information = np.exp(-np.square(z_std))
        kernel = gp.kernel_
        params = kernel.get_params()
        # params = (params['k1'], f"noise level: {params['k2__noise_level']}")
    
        return z_pred, z_std, params
    
    def plot_interpolation(self, data, values, prediction_range, z_pred, z_std):
        """
        Plot the Gaussian Process interpolation.

        Parameters
        ----------
        data: :class:`np.ndarray`
            Array of input data points.
        values: :class:`np.ndarray`
            Array of values corresponding to the input data points.
        prediction_range: :class:`np.ndarray`
            Array of points where predictions are made.
        z_pred: :class:`np.ndarray`
            Predicted values array.
        z_std: :class:`np.ndarray`
            Standard deviation of the predicted values.
        """
        
        plt.figure(figsize=(10, 5))
        plt.plot(data, values, 'r.', markersize=10, label='Observed data')
        # plt.plot(prediction_range, z_pred.ravel(), 'b-', label='Prediction')

        X_Y_Spline = interp1d(prediction_range.ravel(), z_pred, kind = "cubic")
        X_ = np.linspace(data.min(), data.max(), 500)
        Y_ = X_Y_Spline(X_)
        plt.plot(X_, Y_, 'b-', "Prediction")
        plt.fill_between(prediction_range.ravel(), 
                         z_pred - 1.96*z_std, 
                         z_pred + 1.96*z_std, 
                         alpha=0.5, label='95% confidence interval')
        plt.xlabel('Y Position (m)')
        plt.ylabel('Ground Stiffness (N/m)', labelpad = 10)

        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

        plt.legend()
        plt.show()

