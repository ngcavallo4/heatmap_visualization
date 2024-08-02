from utility.krige_plotter import KrigingPlotter
import time 

# Notice! You should be in /heatmap_visualization when running your code, and your csv
# files should live in /heatmap_visualization/data. Otherwise, the code will not run. 

def main():

    # The key for the length scale of the combined legs heatmap is a string of 
    # the numbers in the list mode that you pass into KrigingPlotter. For example,
    # if you pass in the list ['0','1','2'], the key for the combined heatmap 
    # would be '0,1,2'. If you pass in the list ['3','0',2'], the string would be
    # '3,0,2'. 

    # For now, the length scale functionality is not working (plots look as though they have a 
    # long lengthscale regardless of the inputs) Unknown as to why. 
    len_scale = {'0': 0.01, '1':  0.01, '2':  0.01, '3':  0.01, '3,2,0':  0.01}

    # To plot multiple legs, pass in all the legs you want to plot as leg_list. 
    plotter = KrigingPlotter(['3','2','0'], bin_num = 30, length_scale=len_scale)
    plotter.initialize_subplots()
    plotter.plot_heatmap('log00-19_trans.csv', True,
                        match_scale = False, transparent={'var %': 0.95, 'transparency': 0.7})

start_time = time.time()  
main()
end_time = time.time()
print(f"Elapsed time {end_time - start_time} seconds to calculate & plot")