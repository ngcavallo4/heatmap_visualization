import numpy as np
import gstools as gs
from gstools.krige import Ordinary 
import utility.vario_bounds as vario_bounds

class KrigeModel():

    def __init__(self,x,y,stiff,num_bins, length_scale):
        self.x = x
        self.y = y
        self.stiff = stiff
        self.x_interpolation_range = [0,0]
        self.y_interpolation_range = [0,0]
        self.x_interpolation_vector = 0
        self.y_interpolation_vector = 0
        self.length_scale = length_scale

        low_bound, up_bound, step_size = vario_bounds.min_max_dist(self.x,self.y, num_bins)
        self.bins = np.arange(low_bound, up_bound, step_size)

    def rank_models(self):

        bin_center, gamma, return_counts = gs.vario_estimate((self.x,self.y),self.stiff,self.bins,return_counts=True) 

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
            fit_model, r2 = self.create_model(model_type=model_name)
            models[model_name] = fit_model
    
            scores[model_name] = r2

        ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        print("Ranking by Pseudo-r^2 score")
        for i, (model, score) in enumerate(ranking, 1):
            print(f"{i:>6}. {model:>15}: {score:.5}")
            if i == 1: 
                top_model = models.get(model)

        return top_model, models, bin_center, gamma       

    def create_model(self,model_type) -> gs.CovModel:  
        model_class = getattr(gs,model_type,gs.Gaussian)
        fit_model = model_class(dim=2,len_scale = self.length_scale)

        bin_center, gamma = gs.vario_estimate((self.x,self.y),self.stiff,self.bins)

        para, pcov, r2 = fit_model.fit_variogram(bin_center,gamma,return_r2=True)
        
        # print(f"Fitted variogram parameters: sill={fit_model.sill}, range={fit_model.len_scale}, nugget={fit_model.nugget}")
        return fit_model, r2

    def organize_kriging_area(self, match_steps: bool, x_interpolation_start = None, x_interpolation_stop = None, y_interpolation_start = None, y_interpolation_stop = None):

        if match_steps:
            # If values are given, set those equal
            if x_interpolation_start != None:
                self.x_interpolation_range[0] = x_interpolation_start
                self.y_interpolation_range[0]= y_interpolation_start
            else: # Else set values equal to minimum of steps 
                self.x_interpolation_range[0] = np.min(self.x) - 0.1
                self.y_interpolation_range[0] = np.min(self.y) - 0.1

            # Set upper range equal to maximum of steps
            self.x_interpolation_range[1] = np.max(self.x) + 0.1
            self.y_interpolation_range[1] = np.max(self.y) + 0.1
        else: # If not match steps, then must pass in values

            if x_interpolation_start == None or x_interpolation_stop is None or y_interpolation_start is None or y_interpolation_stop is None:
                raise BaseException('Missing arguments. If match_steps is false, the four other arguments in organize_kriging_area are required.')

            self.x_interpolation_range[0] = x_interpolation_start
            self.x_interpolation_range[1] = x_interpolation_stop
            self.y_interpolation_range[0]= y_interpolation_start
            self.y_interpolation_range[1]= y_interpolation_stop
    
    def execute_kriging(self, model):
        
        self.x_interpolation_vector = np.linspace(self.x_interpolation_range[0], self.x_interpolation_range[1], 100) # N
        self.y_interpolation_vector = np.linspace(self.y_interpolation_range[0], self.y_interpolation_range[1], 100) # M
        
        # GSTools
        OK = Ordinary(model=model, cond_pos=[self.x, self.y], cond_val=self.stiff,exact=True)
        z_pred, var = OK.structured([self.x_interpolation_vector,self.y_interpolation_vector])

        z_pred = np.array(z_pred)
        z_pred = np.transpose(z_pred)

        var = np.array(var)
        var = np.transpose(var)

        return z_pred, var, self.x_interpolation_range, self.y_interpolation_range 