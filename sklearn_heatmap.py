from tools.gpregressor import GPRegressor
from tools.plotter import Plotter

# Notice! You should be in /heatmap_visualization when running your code, and your csv
# files should live in /heatmap_visualization/data. Otherwise, the code will not run. 

def main():

    # To plot multiple legs, pass in all the legs you want to plot into the list 'mode'. 
    plotter = Plotter(['0','1','2','3'])
    tool = GPRegressor(0.00001,0.5,5)
    plotter.plot_heatmap('combined-2024-06-19x005_Mh24_Loc2.csv',True, gpregressor=tool, transparent={'var bound': 18, 'transparency': 0.9}, match_scale=True, normalize=False)

main()

