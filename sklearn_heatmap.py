import math
from tools.velocity_plot import VelocityPlotter as vp
from tools.gpregressor import GPRegressor
from tools.plotter import Plotter

# Make sure you put your heatmap CSVs in PATH... or it won't work.

PATH = "/home/sandy/loco_sensing_analysis"
PATH = "/home/sandy/Downloads/heatmap_data"
len_scale = {"val": 1, "bounds": (0.2, 1.0)}
sigma_f = {"val": 4, "bounds": (1, 100)}
noise_level = {"val": 0.2, "bounds": (1e-3, 10)}
tool = GPRegressor(len_scale, noise_level, sigma_f, 10.0, alpha=2.5)

FILE = "test_data.csv"
LAT_LON=False
FILE = "combined-2024-06-19_Mh24_Loc2.csv"
LAT_LON=True
leg_list = ["0", "2"]
# leg_list = ["0", "1", "2", "3"]


# transparent = {"var %": 0.95, "transparency": 0.5}

def main(
    path,
    file,
    leg_list,
    match_steps: bool = True,
    match_scale: bool = True,
    latlon: bool = True,
    optimizer: bool = True,
    transparent: dict = None,
    rotate: int = None,
):
    
    # print(sklearn.__version__)
    plotter = Plotter(leg_list, rotate)
    stiff_dict = plotter.plot_heatmap(
        path, file, match_steps, tool, match_scale, convert_latlon=latlon, optimizer=optimizer
    )
    print(stiff_dict)
    d_data = {}
    m = 35
    g = 9.81
    r = 0.15
    omega = 2*math.pi
    for config in stiff_dict:
        d_data[config] = (m * g + r *omega/0.2 ) / ( 3 * stiff_dict[config] )
    velocity_plotter = vp(d_data,
                          0.09,
                          r,
                          omega)
    
    velocity_plotter.plot()

main(PATH, FILE, leg_list, True, False,latlon=LAT_LON,optimizer=True)