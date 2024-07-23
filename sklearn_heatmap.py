from tools.gpregressor import GPRegressor
from tools.plotter import Plotter
import time

# Notice! You should be in /heatmap_visualization when running your code, and your csv
# files should live in /heatmap_visualization/data. Otherwise, the code will not run. 

def main():

    # To plot multiple legs, pass in all the legs you want to plot into the list leg_list. 
    len_scale = {'0': 0.0002, '1':  0.000001, '2':  0.0002, '3':  0.00001, '0,2':  0.0002}
    plotter = Plotter(['0','2'])
    tool = GPRegressor(len_scale,0.2,4)
    plotter.plot_heatmap('combined-2024-06-19x005_Mh24_Loc2.csv',True, gpregressor=tool,
                    transparent=None, match_scale=False, normalize=False)
    # {'var %': 0.15, 'transparency': 0.6}
start_time = time.time()  
main()
end_time = time.time()
print(f"Elapsed time {end_time - start_time} seconds to calculate & plot")


#0.00001
 