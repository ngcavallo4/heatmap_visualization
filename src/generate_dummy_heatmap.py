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

    f1, axs = plt.subplots(2, 2, figsize=((10, 8)))
    plt.figure(1)

    # Number of bins drastically changes the accuracy of the interpolation

    # KrigeModel is the class that performs all the interpolation
    krige_model = KrigeModel(x,y,stiff,6)

    # rank_models iterates through several types of kriging and selects the type that best fits the shape of the data
    # that type of model is returned and passed into create model. 
    model_type = krige_model.rank_models(30,f1, ax = axs[0,0])
    model, r2 = krige_model.create_model(model_type.name)

    krige_model.organize_kriging_area(True,0,0)
    krige_model.plot_kriging(model,f1,axs)
    GroundTruthPlot(ground_truth_func, krige_model.x_interpolation_len, krige_model.y_interpolation_len,150)
    f1.tight_layout()
    plt.show()

main()