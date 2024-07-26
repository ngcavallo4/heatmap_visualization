from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic, ConstantKernel as C
import numpy as np

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
    
    def __init__(self, length_scale: dict, noise_level: float, sigma_f: dict, nu: float, len_scale_bounds: dict, alpha: float):
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
        self.length_scale_bounds = len_scale_bounds

    def create_kernel(self, request):
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

        request_long = str(request + "_long")

        len_scale_bounds = self.length_scale_bounds[request]
        len_scale = self.length_scale[request]

        # len_scale_long = self.length_scale[request_long] # For rational quadratic, along with alpha

        noise_bounds = self.noise_level["bounds"] # tuple
        noise = self.noise_level["val"]
        sigma_f_bounds = self.sigma_f["bounds"] # tuple
        sigma_f_val = self.sigma_f["val"] # tuple

        # Define the kernel components
        kernel = (C(sigma_f_val**2, sigma_f_bounds) 
                * Matern(length_scale=len_scale, length_scale_bounds=len_scale_bounds, nu=self.nu) 
                + WhiteKernel(noise, noise_bounds))

        # kernel = (C(self.sigma_f**2, (1e-3, 1e3))
        #         * 0.5 * Matern(length_scale=len_scale, length_scale_bounds=(1e-5,1e5), nu=1.5)
        #         + RBF(length_scale=len_scale_long,length_scale_bounds=(1e-2,1e2))
        #         + WhiteKernel(self.noise_level, noise_level_bounds=(1e-5,1)))
        
        # kernel = (C(self.sigma_f**2, (1e-3, 1e3)) 
        #     * RationalQuadratic(length_scale=len_scale_long,length_scale_bounds=(1e-5,1e5), alpha = self.alpha, alpha_bounds = (1e-5,1e5))
        #     * Matern(length_scale=len_scale, length_scale_bounds=(1e-5,1e5), nu=1.5))
        
        return kernel

    def Gaussian_Estimation(self, data, values, prediction_range,  optimizer: bool, kernel = None):
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
        # information = np.exp(-np.square(z_std))
        kernel = gp.kernel_
        params = kernel.get_params()
        params = (params['k1'], f"noise level: {params['k2__noise_level']}")
    
        return z_pred, z_std, params