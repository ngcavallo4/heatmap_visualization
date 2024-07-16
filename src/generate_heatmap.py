from utility.krige_plotter import KrigingPlotter
from utility.parse_csv import CSVParser
import matplotlib.pyplot as plt

# Notice! You should be in /heatmap_visualization when running your code, and your csv
# files should live in /heatmap_visualization/data. Otherwise, the code will not run. 

def main():

    # Change mode to 0, 1, 2, 3 to request a single leg, or 'all' to request all legs.
    len_scale = {'0': 1.1, '1': 2.2, '2': 5.2, '3': 0.5, 'all': 10}
    plotter = KrigingPlotter(['0','1','2','3'], bin_num = 5, length_scale=len_scale)
    plotter.plot_heatmap('log00-19_trans.csv', True, match_scale = False)
    
    # parser = CSVParser("2024-6-18_Mh24_Loc1_Path1_10_12_am_Trial3.csv")
    # all_legs_x, all_legs_y, all_legs_stiff, title = parser.access_data("all")

    # plt.scatter(all_legs_x, all_legs_y, c=all_legs_stiff, cmap = 'viridis', edgecolors='k')
    # plt.ticklabel_format(useOffset=False)
    # plt.show()

main()