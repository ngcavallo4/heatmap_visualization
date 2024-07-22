from tools.gpregressor import GPRegressor
from tools.plotter import Plotter
import time

# Notice! You should be in /heatmap_visualization when running your code, and your csv
# files should live in /heatmap_visualization/data. Otherwise, the code will not run. 

def main():

    # To plot multiple legs, pass in all the legs you want to plot into the list 'mode'. 
    len_scale = {'0': 0.00001, '1':  0.00001, '2':  0.00001, '3':  0.00001, '0,1,2,3':  0.00001}
    plotter = Plotter(['0','1','2','3'])
    tool = GPRegressor(len_scale,0.2,4)
    plotter.plot_heatmap('2024-6-18_Mh24_Loc1_Path1_10_12_am_Trial3.csv',True, gpregressor=tool,
                    transparent={'var %': 0.15, 'transparency': 0.6}, match_scale=False, normalize=False)
start_time = time.time()  
main()
end_time = time.time()
print(f"Elapsed time {end_time - start_time} seconds to calculate & plot")



 