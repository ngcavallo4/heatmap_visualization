import sys
import os
sys.path.append(os.path.abspath("./src/utility"))
import matplotlib.pyplot as plt
from utility.data_generator import DataGenerator
from utility.krige_model import KrigeModel
from ground_truth_plotting import GroundTruthPlot
from ground_truth_func import ground_truth_func

def main():

    # generating dummy data in a class
    data_gen = DataGenerator()
    combined_array = data_gen.generate_steps(ground_truth_func,0.2,0.01,2,2,4,1.01,-1.75,2,0.35,0.15)

    # pulling out position data for use in kriging
    x = combined_array[0,:]
    y = combined_array[1,:]
    stiff = combined_array [2,:]

    # print(x)
    # print(y)
    # print(stiff)

    f1, axs = plt.subplots(1, 3, figsize=((10, 8)))
    plt.figure(f1)

    # Number of bins drastically changes the accuracy of the interpolation

    # KrigeModel is the class that performs all the interpolation
    krige_model = KrigeModel(x,y,stiff,6, 1.0)

    # rank_models iterates through several types of kriging and selects the type that best fits the shape of the data
    # that type of model is returned and passed into create model. 

    model_type, models_dict, bin_centers, gamma = krige_model.rank_models()
    fitted_model, r2 = krige_model.fit_model(model_type.name)
    krige_model.organize_kriging_area(True)
    z_pred, var, x_interpolation_range, y_interpolation_range = krige_model.execute_kriging(fitted_model)
    
    ax1 = axs[0]
    plotter = GroundTruthPlot(ground_truth_func, krige_model.x_interpolation_range, krige_model.y_interpolation_range)
    plotter.plot_ground_truth(f1, ax1, 150)

    ax2 = axs[1]
    im2 = ax2.imshow(z_pred, origin='lower', cmap='viridis', 
                             extent=(x_interpolation_range[0], x_interpolation_range[1],
                                    y_interpolation_range[0], y_interpolation_range[1]))
    ax2.scatter(x, y, c=stiff, edgecolors='k', cmap='viridis') 
    ax2.set_title(f'Kriging Interpolation ({fitted_model.name})')
    f1.colorbar(im2, ax=ax2, shrink=0.7)

    ax3 = axs[2]
    im3 = ax3.imshow(var, origin='lower', cmap='viridis', 
                             extent=(x_interpolation_range[0], x_interpolation_range[1],
                                      y_interpolation_range[0], y_interpolation_range[1]))
    ax3.set_title(f'Kriging Variance ({fitted_model.name})')
    f1.colorbar(im3, ax = ax3, shrink=0.7)

    f1.tight_layout()
    plt.show()

main()