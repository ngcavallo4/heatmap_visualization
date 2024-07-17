import numpy as np
import gstools as gs
from gstools.krige import Ordinary 
import vario_bounds as vario_bounds

class KrigeModel():

    r"""Calculates empirical variogram model and performs kriging on datasets.

        Parameters:

        x: :class:`np.ndarray`
            X position data.
        y: :class:`np.ndarray`
            Y position data.
        stiff: :class:`np.ndarray`
            Stiffness data, tied to x-y position.
        num_bins: :class:`float`
            Number of bins for the emperical variogram.
        length_scale: :class:`float`
            Length scale for the variogram.
        bins: :class:`np.ndarray`
            Vector of bin divisions based on function call to vario_bounds.min_max_dist
    """

    def __init__(self,x,y,stiff,num_bins, length_scale = 1.0):
        self.x = x
        self.y = y
        self.stiff = stiff
        self.x_interpolation_range = [0,0]
        self.y_interpolation_range = [0,0]
        self.x_interpolation_vector = 0
        self.y_interpolation_vector = 0
        self.length_scale = length_scale

        # self.bins = gs.standard_bins((self.x, self.y), latlon=True, bin_no= num_bins, geo_scale=gs.KM_SCALE)

        low_bound, up_bound, step_size = vario_bounds.min_max_dist(self.x,self.y, num_bins)
        # self.bins = np.arange(low_bound, up_bound, step_size, dtype=np.longdouble)

        step_size = (up_bound) / num_bins
        self.bins = np.arange(0, up_bound, step_size, dtype=np.longdouble)
        
    def rank_models(self):
        r"""Ranks variogram models based on how well the model fits the estimated
            emperical variogram model based on r^2 score. 

            Returns
            -------

            top_model: :class:`gs.CovModel`
                Best fitting model based on r^2 score. 
            models: :class:`dict[gs.CovModel]`
                Dictionary containing varios GSTools CovModels. Key is the model
                name as a string, and the value is the gs.CovModel. 
            bin_center: :class:`np.ndarray`
                Bin centers as returned by gs.vario_estimate. 
            gamma: :class:`np.ndarray`
                Empirical semivariogram value as found by gs.vario_estimate.
        """

        bin_center, gamma, return_counts = gs.vario_estimate((self.x,self.y),
                            self.stiff,self.bins,return_counts=True, latlon = True) 

        models = {
            "TPLGaussian": gs.TPLGaussian,
            "Exponential":gs.Exponential,
            "Spherical":gs.Spherical,
            "Linear":gs.Linear,
            "Cubic":gs.Cubic,
            "TPLStable":gs.TPLStable,
            # "TPLSimple":gs.TPLSimple,
            "TPLExponential":gs.TPLExponential,
            "Gaussian":gs.Gaussian
        }

        scores = {}
        
        for model_name, model in models.items():

            # Iterates through models dict, fitting models to the estimated
            # variogram, returns an r^2 value and a fitted_model.
            fitted_model, r2 = self.fit_model(model_name=model_name)
            models[model_name] = fitted_model
    
            scores[model_name] = r2

        # Ranks all the models by r^2 score with 1 being the best 
        ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        print("Ranking by Pseudo-r^2 score")
        for i, (model, score) in enumerate(ranking, 1):
            print(f"{i:>6}. {model:>15}: {score:.5}")
            if i == 1: 
                top_model = models.get(model)

        return top_model, models, bin_center, gamma       

    def fit_model(self,model_name: str): 

        r"""Fits a gs.CovModel to the estimated variogram. 

            Parameters
            ----------

            model_type: :class:`str`
                Name of the model to be fitted.
            
            Returns
            -------
            fitted_model: :class:`gs.CovModel`
                Variogram model fitted to the empirical variogram model.
            r^2: :class:`float`
                r^2 value for how well the fitted model fits the 
                empirical variogram model. 
        """ 
        model_class = getattr(gs,model_name,gs.Gaussian)
        fitted_model = model_class(dim=2)#, latlon = True, geoscale = gs.KM_SCALE)

        # bins = gs.standard_bins((self.x, self.y), dim=2, latlon = True, geoscale = gs.KM_SCALE)

        bin_center, gamma = gs.vario_estimate((self.x,self.y),self.stiff,
                                                            self.bins)

        para, pcov, r2 = fitted_model.fit_variogram(bin_center,gamma, 
                            init_guess = {"len_scale": self.length_scale, "default": "current"},
                            return_r2=True)
        
        # print(f"Fitted variogram parameters: sill={fit_model.sill}, range={fit_model.len_scale}, nugget={fit_model.nugget}")
        return fitted_model, r2

    def organize_kriging_area(self, match_steps: bool, x_interpolation_input_range,
                                                        y_interpolation_input_range):

        r"""Initializes fields self.x_interpolation_range and 
            self.y_interpolation.range based on whether the area will
            match the input data or not. Must be called before 
            execute_kriging.

            Parameters
            ----------

            match_steps: :class:`bool`
                Boolean that determines whether the kriging area will be fitted
                to match the input data. If False, user must input 
                x and y_interpolation_input_range. Passed through from 
                KrigePlotter.
            x_interpolation_input_range: :class:`list[float]`, optional
                Only necessary to provide if match_steps is False. This is a
                array of length 2, where the 0th index is the lower bound
                of the range, and the 1st index is the upper bound of the range.
                Passed through from KrigePlotter.
            y_interpolation_input_range: :class:`list[float]`, optional
                Only necessary to provide if match_steps is False. This is a
                array of length 2, where the 0th index is the lower bound
                of the range, and the 1st index is the upper bound of the range.
                Passed through from KrigePlotter.
        """
        if match_steps:
            # If values are given, set those equal
            if x_interpolation_input_range is not None and y_interpolation_input_range is not None:
                self.x_interpolation_range[0] = x_interpolation_input_range[0]
                self.y_interpolation_range[0]= y_interpolation_input_range[0]
                self.x_interpolation_range[1] = np.max(self.x)
                self.y_interpolation_range[1] = np.max(self.y)
            else:
                print("Matching steps")
                self.x_interpolation_range[0] = np.min(self.x) + 0.000001
                self.y_interpolation_range[0] = np.min(self.y) + 0.000001
                self.x_interpolation_range[1] = np.max(self.x) + 0.000001
                self.y_interpolation_range[1] = np.max(self.y) + 0.000001

        else: # If not match steps, then must pass in values

            if x_interpolation_input_range is None or y_interpolation_input_range is None:
                raise BaseException('Missing arguments. If match_steps is false, the four other arguments in organize_kriging_area are required.')

            self.x_interpolation_range[0] = x_interpolation_input_range[0]
            self.x_interpolation_range[1] = x_interpolation_input_range[1]
            self.y_interpolation_range[0]= y_interpolation_input_range[0]
            self.y_interpolation_range[1]= y_interpolation_input_range[1]

        return self.x_interpolation_range, self.y_interpolation_range
    
    def execute_kriging(self, model):

        r"""Interpolates values based on range calculated from organize_kriging_area.
            
            Parameters
            ----------

            model: :class:`gs.CovModel`
                Covariance model calculated from fit_model. 
            
            Returns
            -------

            z_pred: :class:`np.ndarray`
                2D array of interpolated stiffness values over the range as calculated 
                from organize_kriging_area. 
            var: :class:`np.ndarray`
                2D array of the variance of the interpolated values over the same range
                as the interpolated values. 
            x_interpolation_range: :class:`np.ndarray`
                X interpolation range as calculated from organize_kriging_area, used for
                plotting
            y_interpolation_range: :class:`np.ndarray`
                Y interpolation range as calculated from organize_kriging_area, used for
                plotting
        """
        
        self.x_interpolation_vector = np.linspace(self.x_interpolation_range[0],
                                                   self.x_interpolation_range[1], 100) # N
        self.y_interpolation_vector = np.linspace(self.y_interpolation_range[0],
                                                   self.y_interpolation_range[1], 100) # M
        
        # GSTools
        OK = Ordinary(model=model, cond_pos=[self.x, self.y], cond_val=self.stiff,exact=True)
        print(model._len_scale)
        z_pred, var = OK.structured([self.x_interpolation_vector,self.y_interpolation_vector])

        z_pred = np.array(z_pred)
        z_pred = np.transpose(z_pred)

        var = np.array(var)
        var = np.transpose(var)

        return z_pred, var, self.x_interpolation_range, self.y_interpolation_range 