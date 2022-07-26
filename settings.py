import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import sys


### LOCAL PATHS ###
sys.path.append(os.sep.join(sys.path[0].split(os.sep)[:-1]))

BASE_DIR = pathlib.Path(__file__).parents[0]
DATA_DIR = BASE_DIR / "data"
AGGREGATION_DIR = DATA_DIR / "aggregation"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "figures"

for path in [DATA_DIR, OUTPUT_DIR, FIGURES_DIR]:
    if not os.path.isdir(path):
        os.mkdir(path)


### GHG PARAMETERS ###
GLOBAL_WARMING_POTENTIAL = {
    "CO2": 1,
    "CH4": 28,
    "N2O": 265,
    "SF6": 23500,
    "HFC": 1,
    "PFC": 1,
}
GHG_LIST = list(GLOBAL_WARMING_POTENTIAL.keys())


### COLORS ###
COLORS = list(plt.cm.tab10(np.arange(10))) + ["gold"]
COLORS_NO_FR = COLORS[1:]


### AGGREGATIONS FOR THE FIGURES ###
# These aggregations are only dedicated to make the figures more legible. They have no influence on the models nor on their results.
# The aggregation used in the models is set through the matrices from data/aggregation as an argument of Model objects in model.py.
# The default REGIONS_AGG and SECTORS_AGG fit the model aggregation "opti_S".
REGIONS_AGG = {
    "FR": ["FR"],
    "UK, Norway, Switzerland": ["UK, Norway, Switzerland"],
    "China+": ["China, RoW Asia and Pacific"],
    "EU": ["EU"],
    "RoW": [
        "United States",
        "Asia, Row Europe",
        "RoW America,Turkey, Taïwan",
        "RoW Middle East, Australia",
        "Brazil, Mexico",
        "South Africa",
        "Japan, Indonesia, RoW Africa",
    ],
}
SECTORS_AGG = {
    "Agriculture": ["Agriculture"],
    "Energy": [
        "Crude coal",
        "Crude oil",
        "Natural gas",
        "Fossil fuels",
        "Electricity and heat",
    ],
    "Industry": [
        "Extractive industry",
        "Biomass_industry",
        "Clothing",
        "Heavy_industry",
        "Automobile",
        "Oth transport equipment",
        "Machinery",
        "Electronics",
        "Construction",
        "Transport services",
    ],
    "Composite": ["Composite"],
}
