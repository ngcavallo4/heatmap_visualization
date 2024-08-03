from tools.gpregressor import GPRegressor
from tools.plotter import Plotter
import time

# Notice! You should be in /heatmap_visualization when running your code, and your csv
# files should live in /heatmap_visualization/data. Otherwise, the code will not run. 

def main():
    # # These are for plotting in lat/lon. 
    # small_scale = 0.03/1111110 # deg/m * m = deg 
    # med_scale = 0.1/1111110 # deg/m * m = deg
    # long_scale = 0.5/1111110 # deg/m * m = deg 

    # Produces weird results
    len_scale = {'val': 0.5,'bounds': (0.3, 1)}
    sigma_f = {"val": 4, "bounds": (1e-7,1e-5)}
    noise_level = {"val": 0.2, "bounds": (1e-3, 10)}

    # # Shipeng initial guesses, produces okay results 
    # len_scale = {'val': 0.5,'bounds': (0.3, 1)}
    # sigma_f = {"val": 2, "bounds": (1e-2, 200)}
    # noise_level = {"val": 0.1, "bounds": (0, 0.2)}

    tool = GPRegressor(len_scale,noise_level,sigma_f, 10.0, alpha=2.5)

    # Path to folder containing CSV files
    PATH = '/Users/natalie/Desktop/heatmap_csvs'

    # {'var %': 0.55, 'transparency': 0.4}
    plotter = Plotter(['0','2'], rotate=None)
    plotter.plot_heatmap(PATH, 'combined-2024-06-19_Mh24_Loc2.csv',
                        True, gpregressor=tool, transparent = {'var %': 0.95, 'transparency': 0.5}, match_scale=True,
                        latlon = False, optimizer=True)
    # {'var %': 0.95, 'transparency': 0.5}

main()

# if __name__ == '__main__':
#     plotter = Plotter(leg_list=['leg1', 'leg2', 'leg3'])
#     plotter.plot_heatmap(PATH, 'combined-2024-06-19_Mh24_Loc2.csv',
#                     True, gpregressor=tool, transparent = {'var %': 0.95, 'transparency': 0.5}, match_scale=True,
#                     latlon = False, optimizer=True)
