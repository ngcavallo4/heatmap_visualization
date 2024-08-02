from tools.gpregressor import GPRegressor
from tools.convert_gps import gps_coords_to_meters
from tools.parse_csv import CSVParser
import numpy as np

def main():

    parser = CSVParser("/Users/natalie/Desktop/heatmap_csvs/2024-06-19x001_Mh24_Loc2_Pathsnow_10_49_Trial2a.csv")
    x, y, stiff, title = parser.access_data(['2'])

    x, y = gps_coords_to_meters(x,y) # x and y are vectors of shape (n,)

    long_scale = 20 # 10m
    med_scale = 0.1 # 0.05m
    small_scale = 0.03 # 0.015m 

    len_scale = {'2': long_scale,}
    len_scale_bounds = {'2': (long_scale/2, long_scale*100)}
    sigma_f = {"val": 2, "bounds": (1e-5,1e5)}
    noise_level = {"val": 10, "bounds": (1e-3, 1e2)}

    tool = GPRegressor(len_scale,noise_level,sigma_f, 1.0, len_scale_bounds, alpha=2.5)
    kernel = tool.create_kernel('2')

    y_range = [0,0]

    y_range[0] = np.min(y) - 0.5
    y_range[1] = np.max(y) + 0.5

    y = y.reshape(-1, 1)
    stiff = stiff.flatten()
    
    prediction_range = np.linspace(y_range[0], y_range[1], 100).reshape(-1,1) 
    z_pred, z_std, params = tool.Gaussian_Estimation_1D(y, stiff, prediction_range, True, kernel=kernel)
    print(f"{params}")
    print(z_pred.shape)
    tool.plot_interpolation(y, stiff, prediction_range, z_pred, z_std)

main()