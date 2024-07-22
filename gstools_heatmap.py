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
    len_scale = {'0': 0.00001, '1':  0.00001, '2':  0.00001, '3':  0.00001, '3,2,0':  0.00001}

    # To plot multiple legs, pass in all the legs you want to plot into the list 'mode'. 
    plotter = KrigingPlotter(['3','2','0'], bin_num = 30, length_scale=len_scale)
    plotter.initialize_subplots()
    plotter.plot_heatmap('2024-6-18_Mh24_Loc1_Path1_10_12_am_Trial3.csv', True, match_scale = False, transparent={'var bound': 12, 'transparency': 0.9})

start_time = time.time()  
main()
end_time = time.time()
print(f"Elapsed time {end_time - start_time} seconds to calculate & plot")