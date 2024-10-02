import math
from tools.velocity_plot import VelocityPlotter as vp
from tools.gpregressor import GPRegressor
from tools.plotter import Plotter

# Make sure you put your heatmap CSVs in PATH... or it won't work.

PATH = "/Users/natalie/Desktop/LASSIE_Spirit/heatmap_csvs"
len_scale = {"val": 1, "bounds": (0.001, 10)}
sigma_f = {"val": 4, "bounds": (1e-4, 10)}
noise_level = {"val": 0.2, "bounds": (1e-3, 1)}
tool = GPRegressor(len_scale, noise_level, sigma_f, 10.0, alpha=2.5)

FILE = "combined-2024-06-19_Mh24_Loc2.csv"
leg_list = ["0", "1", "2"]

# transparent = {"var %": 0.95, "transparency": 0.5}

def main(
    path,
    file,
    leg_list,
    match_steps: bool = True,
    match_scale: bool = True,
    latlon: bool = False,
    optimizer: bool = True,
    transparent: dict = None,
    rotate: int = None,
):
    
    # print(sklearn.__version__)
    plotter = Plotter(leg_list, rotate)
    stiff_dict = plotter.plot_heatmap(
        path, file, match_steps, tool, match_scale, latlon=latlon, optimizer=optimizer
    )

    velocity_plotter = vp(stiff_dict, 4.0, 2*math.pi, 40)
    velocity_plotter.plot()




main(PATH, FILE, leg_list, True, False, optimizer=True)