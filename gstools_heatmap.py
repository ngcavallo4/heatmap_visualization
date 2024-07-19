from utility.krige_plotter import KrigingPlotter
from utility.parse_csv import CSVParser
import matplotlib.pyplot as plt

# Notice! You should be in /heatmap_visualization when running your code, and your csv
# files should live in /heatmap_visualization/data. Otherwise, the code will not run. 

def main():

    # The key for the length scale of the combined legs heatmap is a string of 
    # the numbers in the list mode that you pass into KrigingPlotter. For example,
    # if you pass in the list ['0','1','2'], the key for the combined heatmap 
    # would be '0,1,2'. If you pass in the list ['3','0',2'], the string would be
    # '3,0,2'. 
    len_scale = {'0': 1, '1': 1, '2': 1, '3': 1, '3,2,0': 1}

    # To plot multiple legs, pass in all the legs you want to plot into the list 'mode'. 
    plotter = KrigingPlotter(['3','2','0'], bin_num = 30, length_scale=len_scale)
    plotter.plot_heatmap('combined-2024-06-19x005_Mh24_Loc2.csv', True, match_scale = False, transparent = True)

main()