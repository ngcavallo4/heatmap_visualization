from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic, ConstantKernel as C
import numpy as np
import sklearn 

class Heatmap():

    """A class to perform Gaussian Process Regression on passed in data.
    
    Attributes
    ----------
    x: :class:`np.ndarray`
        X position array of Spirit toes.
    y: :class:`np.ndarray`
        Y position array of Spirit toes.
    stiff: :class:`np.ndarray`
        Stiffness values array. 
    length_scale: :class:`dict`
            Dictionary mapping leg numbers to corresponding length scale values.
    sigma_f: :class:`float`
        Constant value parameter for the Gaussian Process.
    noise_level: :class:`float`
        Noise level parameter for the Gaussian Process.
    nu: :class:`float`
        Nu paramater for the Matérn kernel. 

    """

    def __init__(self, length_scale: dict, sigma_f: dict, noise_level: dict, nu: float):
        """Initialize the GPRegressor class with the given parameters.

        Parameters
        ----------
        length_scale: :class:`dict`
            Dictionary mapping leg numbers to corresponding length scale values.
        sigma_f: :class:`float`
            Constant value parameter for the Gaussian Process.
        noise_level: :class:`float`
            Noise level parameter for the Gaussian Process.
        nu: :class:`float`
            Nu paramater for the Matérn kernel. 
        
        """
        self.x = np.array([])
        self.y = np.array([])
        self.stiff = np.array([])
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.noise_level = noise_level
        self.nu = nu 

    def update_kriging(self, x: float, y: float, stiff: float): 
        """Update the kriging model with new data points and perform kriging.

        Parameters
        ----------
        x: :class:`float`
            X position of the new data point.
        y: :class:`float`
            Y position of the new data point.
        stiff: :class:`float`
            Stiffness value of the new data point.
        """

        if stiff < 0:
            raise ValueError("Stiffness can't be negative")

        self.add_points(x, y, stiff)
        x_range, y_range = self.organize_area(x,y)
        z_pred, vars = self.perform_kriging(self.x, self.y, self.stiff, x_range, y_range)

        return z_pred

    def add_points(self, x:float, y:float, stiff: float):

        """Add new data points to the existing arrays.

        Parameters
        ----------
        x: :class:`float`
            X position of the new data point.
        y: :class:`float`
            Y position of the new data point.
        stiff: :class:`float`
            Stiffness value of the new data point.
        """

        self.x = np.concatenate(self.x, x)
        self.y = np.concatenate(self.y, y)
        self.stiff = np.concatenate(self.stiff, stiff)

    def perform_kriging(self, x: np.ndarray, y: np.ndarray, stiff: np.ndarray, x_range: list[int], y_range: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Perform kriging to interpolate data using Gaussian Process Regression.

        Parameters
        ----------
        x: :class:`np.ndarray`
            X position array.
        y: :class:`np.ndarray`
            Y position array.
        stiff: :class:`np.ndarray`
            Stiffness values array.
        x_range: :class:`list[int]`
            Range of X values to interpolate over.
        y_range: :class:`list[int]`
            Range of Y values to interpolate over.

        Returns
        -------
        z_pred: :class:`np.ndarray`
            Predicted values array.
        var: :class:`np.ndarray`
            Variance of the predicted values.
        """
        estimated_num = 100
        xx1, xx2 = np.linspace(x_range[0], x_range[1], num=estimated_num), np.linspace(y_range[0], y_range[1], num=estimated_num)
        vals = np.array([[x1_, x2_] for x1_ in xx1 for x2_ in xx2]).T

        robot_measured_points = np.vstack((x, y)).T

        kernel = self.create_kernel(self.length_scale, self.sigma_f, self.noise_level, self.nu)
        z_pred, z_std= self.Gaussian_Estimation(robot_measured_points, stiff, vals, kernel=kernel)
        z_pred = z_pred.reshape(estimated_num, estimated_num).T
        z_std = z_std.reshape(estimated_num, estimated_num).T

        var = np.square(z_std)

        # Stiffness should never be negative, so any negative values we round to zero. 
        z_pred[z_pred < 0] = 0

        return z_pred, var

    def create_kernel(self, length_scale: dict, sigma_f: dict, noise_level: dict, nu: float):
        """Create a Gaussian Process kernel based on the given parameters.

        Parameters
        ----------
        length_scale: :class:`dict`
            Dictionary mapping leg numbers to corresponding length scale values.
        sigma_f: :class:`dict`
            Constant value parameter for the Gaussian Process.
        noise_level: :class:`dict`
            Noise level parameter for the Gaussian Process.
        nu: :class:`float`
            Nu parameter for the Matérn kernel.

        Returns
        -------
        kernel: :class:`sklearn.gaussian_process.kernels.Kernel`
            The constructed kernel for the Gaussian Process.
        """

        # length scale: controls the smoothness of the function
        # noise_level – Noise level
        # sigma_f – constant value parameter
        # nu – Matern parameter, when equal to 1.5, Matern is equal to RBF

        len_scale_bounds = length_scale["bounds"] # tuple
        len_scale = length_scale["val"]
        noise_bounds = noise_level["bounds"] # tuple
        noise = noise_level["val"]
        sigma_f_bounds = sigma_f["bounds"] # tuple
        sigma_f_val = sigma_f["val"] # tuple

        # Define the kernel components
        kernel = (C(constant_value=sigma_f_val**2, constant_value_bounds=sigma_f_bounds) 
                * Matern(length_scale=len_scale, length_scale_bounds=len_scale_bounds, nu=nu) 
                + WhiteKernel(noise_level=noise, noise_level_bounds=noise_bounds))
        
        return kernel
    
    def Gaussian_Estimation(self, data: np.ndarray, values: np.ndarray, prediction_range: np.ndarray, kernel: sklearn.gaussian_process.kernels.Kernel) -> tuple[np.ndarray, np.ndarray]:
        """Perform Gaussian Process Regression to fit the model and make predictions.

        Parameters
        ----------
        data: :class:`np.ndarray`
            Array of input data points.
        values: :class:`np.ndarray`
            Array of values corresponding to the input data points.
        prediction_range: :class:`np.ndarray`
            Array of points where predictions are to be made.
        kernel: :class:`sklearn.gaussian_process.kernels.Kernel`, optional
            Kernel to be used for the Gaussian Process.

        Returns
        -------
        z_pred: :class:`np.ndarray`
            Predicted values array.
        z_std: :class:`np.ndarray`
            Standard deviation of the predicted values.
        """

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, random_state=0)
        
        # Fit the model to the data
        gp.fit(data, values)

        # Make predictions on new data points
        data_new = prediction_range.T
        z_pred, z_std = gp.predict(data_new, return_std=True)
    
        return z_pred, z_std
    
    def organize_area(self, x: np.ndarray, y: np.ndarray, latlon: bool):

        r"""Returns ranges x_range and y_range based on whether the area
            will match the input data or not.

            Parameters
            ----------
            x: :class:`np.ndarray`
                X position array that determines x-range
            y: :class:`np.ndarray`
                Y position array that determines x-range 
            
            Returns
            -------
            x_range: :class:`list[float]` 
                Range of x values to interpolate over.
            y_range: :class:`list[float]` 
                Range of y values to interpolate over.
        """

        if self.latlon:
            offset = 0.000001
        else: 
            offset = 0.25

        x_range = [0,0]
        y_range = [0,0]

        x_range[0] = np.min(x) - offset
        x_range[1] = np.max(x) + offset

        y_range[1] = np.max(y) + offset
        y_range[0] = np.min(y) - offset 

        return x_range, y_range