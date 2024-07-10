from utility.krige_plotter import KrigingPlotter

# Notice! You should be in /heatmap_visualization when running your code, and your csv
# files should live in /heatmap_visualization/data. Otherwise, the code will not run. 

def main():

    # Change mode to 0, 1, 2, 3 to request a single leg, or 'all' to request all legs.
    len_scale = {'0': 1.2, '1': 2.2, '2': 5.2, '3': 0.5, 'all': 300}
    plotter = KrigingPlotter('all', bin_num = 30, length_scale=len_scale)
    plotter.plot_heatmap('log00-19_trans.csv', True, match_scale = False)

main()