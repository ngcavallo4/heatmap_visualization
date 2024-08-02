from tools.gpregressor import GPRegressor
from tools.plotter import Plotter
import time

# Notice! You should be in /heatmap_visualization when running your code, and your csv
# files should live in /heatmap_visualization/data. Otherwise, the code will not run. 

def main():

    # small_scale = 0.03/1111110 # deg/m * m = deg 
    # med_scale = 0.1/1111110 # deg/m * m = deg
    # long_scale = 0.5/1111110 # deg/m * m = deg 

    small_scale = 0.05
    med_scale = 0.1 
    long_scale = 0.5
    longest_scale = 10

    scale = longest_scale
    
    # To plot multiple legs, pass in all the legs you want to plot into the list leg_list. 
    # len_scale = {'val': scale,'bounds': (0.3, 1)}
    # sigma_f = {"val": 4, "bounds": (1e-7,1e-5)}
    # noise_level = {"val": 0.2, "bounds": (1e-3, 10)}

    len_scale = {'val': scale,'bounds': (scale/2, scale*100)}
    sigma_f = {"val": 4, "bounds": (0.2,20)}
    noise_level = {"val": 0.2, "bounds": (1e-3, 1e2)}

    tool = GPRegressor(len_scale,noise_level,sigma_f, 1.0, alpha=2.5)

    PATH = '/Users/natalie/Desktop/heatmap_csvs'

    # {'var %': 0.55, 'transparency': 0.4}
    plotter = Plotter(['0','2'], rotate=None)
    plotter.plot_heatmap(PATH, 'combined-2024-06-19_Mh24_Loc2.csv',
                        True, gpregressor=tool, transparent = {'var %': 0.95, 'transparency': 0.5}, match_scale=True,
                        latlon = False, optimizer=True)
    # {'var %': 0.95, 'transparency': 0.5}
start_time = time.time()  
main()
end_time = time.time()
print(f"Elapsed time {end_time - start_time} seconds to calculate & plot")
