from tools.gpregressor import GPRegressor
from tools.plotter import Plotter

def main():

    plotter = Plotter(['0','1','2','3'])
    tool = GPRegressor(0.1,0.2, 4)
    plotter.plot_heatmap('2024-6-18_Mh24_Loc1_Path1_10_12_am_Trial3.csv',True, gpregressor=tool, match_scale= True, normalize=True)

main()

