from tools.gpregressor import GPRegressor
from tools.plotter import Plotter
import time

# Make sure you put your heatmap CSVs in PATH... or it won't work.

PATH = "/Users/natalie/Desktop/LASSIE_Spirit/heatmap_csvs"
len_scale = {"val": 0.5, "bounds": (0.3, 1)}
sigma_f = {"val": 4, "bounds": (1e-4, 5)}
noise_level = {"val": 0.2, "bounds": (1e-3, 10)}
tool = GPRegressor(len_scale, noise_level, sigma_f, 10.0, alpha=2.5)

FILE = "combined-2024-06-19_Mh24_Loc2.csv"
leg_list = ["1", "2", "3"]

# transparent = {"var %": 0.95, "transparency": 0.5}

def main(
    path,
    file,
    leg_list,
    match_scale: bool = True,
    latlon: bool = False,
    optimizer: bool = True,
    transparent: dict = None,
    rotate: int = None,
):
    plotter = Plotter(leg_list, rotate)
    plotter.plot_heatmap(
        path, file, True, tool, match_scale, latlon=latlon, optimizer=optimizer
    )

main(PATH, FILE, leg_list, True, False, True)