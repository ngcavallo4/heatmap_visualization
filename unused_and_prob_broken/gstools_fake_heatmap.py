import matplotlib.pyplot as plt
from utility.data_generator import DataGenerator
from utility.krige_model import KrigeModel
from utility.ground_truth_plotting import GroundTruthPlot
from utility.ground_truth_func import ground_truth_func
from utility.krige_plotter import KrigingPlotter

#
##
### NOTE: NON FUNCTIONAL 
##
#  

def main():

    # generating dummy data in a class
    data_gen = DataGenerator()
    combined_array = data_gen.generate_steps(ground_truth_func,0.2,0.01,2,
                    10,2,1.01,1.75,10,0.25,0.1)

    # pulling out position data for use in kriging
    x = combined_array[0,:]
    y = combined_array[1,:]
    stiff = combined_array [2,:]

    plotter = KrigingPlotter(['null'])
    f1, axs = plotter.initialize_subplots(1,3)
    x_range, y_range = plotter.organize_kriging_area(x,y,True,None,None)
    z_pred, var, fitted_model = plotter.perform_kriging(x,y,stiff,0.1,x_range, y_range)
    plotter.plot_leg(z_pred,var,x,y,stiff,x_range,y_range,fitted_model.name,0,False)

    ax3 = axs[2]
    ax3.set_xlim([x_range[0], x_range[1]])
    ax3.set_ylim([y_range[0], y_range[1]])
    gtplot = GroundTruthPlot(ground_truth_func, x_range, y_range)
    gtplot.plot_ground_truth(f1, ax3, 150)
    ax3.set_aspect('equal')

    f1.tight_layout()
    plt.show()

main()