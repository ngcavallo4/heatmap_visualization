from utility.krige_plotter import KrigingPlotter

# Notice! You should be in /heatmap_visualization when running your code, and your csv
# files should live in /heatmap_visualization/data. Otherwise, the code will not run. 

def main():

    plotter = KrigingPlotter('all', bin_num = 30)
    plotter.plot_heatmap('log00-19_trans.csv', True)

main()