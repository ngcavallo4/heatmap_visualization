from tools.gpregressor import GPRegressor
from tools.plotter import Plotter
import time

# Notice! You should be in /heatmap_visualization when running your code, and your csv
# files should live in /heatmap_visualization/data. Otherwise, the code will not run. 

def main():

    # small_scale = 0.03/1111110 # deg/m * m = deg 
    # med_scale = 0.1/1111110 # deg/m * m = deg
    # long_scale = 0.5/1111110 # deg/m * m = deg 

    small_scale = 0.03
    med_scale = 0.1
    long_scale = 10
    
    # To plot multiple legs, pass in all the legs you want to plot into the list leg_list. 
    len_scale = {'0': long_scale,'2': long_scale,'0,2': long_scale,'0_long': long_scale,'2_long': long_scale,'0,2_long': long_scale}
    len_scale_bounds = {'0': (long_scale/10, long_scale*10), '2': (long_scale/10, long_scale*10),'0,2': (long_scale/10, long_scale*10)}
    sigma_f = {"val": 2, "bounds": (0.2,20)}
    noise_level = {"val": 0.2, "bounds": (1e-3, 1e2)}

    tool = GPRegressor(len_scale,noise_level,sigma_f, 1.0, len_scale_bounds, nu = 2.5)

    # {'var %': 0.55, 'transparency': 0.4}
    plotter = Plotter(['0','2'])
    plotter.plot_heatmap('/Users/natalie/Desktop/heatmap_csvs/combined-2024-06-19x005_Mh24_Loc2.csv',True, gpregressor=tool,
                    transparent = None, match_scale=True, latlon = False)

start_time = time.time()  
main()
end_time = time.time()
print(f"Elapsed time {end_time - start_time} seconds to calculate & plot")
