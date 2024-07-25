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
    long_scale = 0.5
    
    # To plot multiple legs, pass in all the legs you want to plot into the list leg_list. 
    len_scale = {'0': long_scale,'2': long_scale,'0,2': long_scale,'0_long': long_scale,'2_long': long_scale,'0,2_long': long_scale}
    len_scale_bounds = {'0': (long_scale/10, long_scale*10), '2': (long_scale/10, long_scale*10),'0,2': (long_scale/10, long_scale*10)}
    plotter = Plotter(['0','2'])
    tool = GPRegressor(len_scale,0.2,2, 2.5, len_scale_bounds)
    plotter.plot_heatmap('2024-6-18_Mh24_Loc1_Path1_10_12_am_Trial3.csv',True, gpregressor=tool,
                    transparent = None, match_scale=True, normalize=False, optimizer = False)
    {'var %': 0.55, 'transparency': 0.4}

start_time = time.time()  
main()
end_time = time.time()
print(f"Elapsed time {end_time - start_time} seconds to calculate & plot")

# OMFG