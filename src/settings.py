import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import sys


### LOCAL PATHS ###


sys.path.append(os.sep.join(sys.path[0].split(os.sep)[:-1]))

BASE_DIR = pathlib.Path(__file__).parents[1]
DATA_DIR = BASE_DIR / "data"
AGGREGATION_DIR = DATA_DIR / "aggregation"
CAPITAL_CONS_DIR = DATA_DIR / "capital_consumption"
EXIOBASE_DIR = DATA_DIR / "exiobase"
MODELS_DIR = DATA_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_MULTIMODEL_DIR = FIGURES_DIR / "multimodel"

for path in [
    DATA_DIR,
    CAPITAL_CONS_DIR,
    EXIOBASE_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    FIGURES_MULTIMODEL_DIR,
]:
    if not os.path.isdir(path):
        os.mkdir(path)


### GHG PARAMETERS ###


GHG_PARAMS = {
    "name_FR": "GES",
    "name_EN": "GHG",
    "unit": "kgCO2eq",
    "proxy": {
        "CO2": {
            "exiobase_keys": [
                "CO2 - combustion - air",
                "CO2 - non combustion - Cement production - air",
                "CO2 - non combustion - Lime production - air",
                "CO2 - agriculture - peat decay - air",
                "CO2 - waste - biogenic - air",
                "CO2 - waste - fossil - air",
            ],
            "weight": 1,
        },
        "CH4": {
            "exiobase_keys": [
                "CH4 - combustion - air",
                "CH4 - non combustion - Extraction/production of (natural) gas - air",
                "CH4 - non combustion - Extraction/production of crude oil - air",
                "CH4 - non combustion - Mining of antracite - air",
                "CH4 - non combustion - Mining of bituminous coal - air",
                "CH4 - non combustion - Mining of coking coal - air",
                "CH4 - non combustion - Mining of lignite (brown coal) - air",
                "CH4 - non combustion - Mining of sub-bituminous coal - air",
                "CH4 - non combustion - Oil refinery - air",
                "CH4 - agriculture - air",
                "CH4 - waste - air",
            ],
            "weight": 28,
        },
        "N2O": {
            "exiobase_keys": ["N2O - combustion - air", "N2O - agriculture - air"],
            "weight": 265,
        },
        "SF6": {
            "exiobase_keys": ["SF6 - air"],
            "weight": 23500,
        },
        "HFC": {
            "exiobase_keys": ["HFC - air"],
            "weight": 1,
        },
        "PFC": {
            "exiobase_keys": ["PFC - air"],
            "weight": 1,
        },
    },
}

MATERIALS_PARAMS = {
    "name_FR": "matières",
    "name_EN": "materials",
    "unit": "kt",
    "proxy": {
        "Bauxite et aluminium": {
            "exiobase_keys": [
                "Domestic Extraction Used - Metal Ores - Bauxite and aluminium ores",
                "Unused Domestic Extraction - Metal Ores - Bauxite and aluminium ores",
            ],
            "weight": 1,
        },
        "Cuivre": {
            "exiobase_keys": [
                "Domestic Extraction Used - Metal Ores - Copper ores",
                "Unused Domestic Extraction - Metal Ores - Copper ores",
            ],
            "weight": 1,
        },
        "Or": {
            "exiobase_keys": [
                "Domestic Extraction Used - Metal Ores - Gold ores",
                "Unused Domestic Extraction - Metal Ores - Gold ores",
            ],
            "weight": 1,
        },
        "Fer": {
            "exiobase_keys": [
                "Domestic Extraction Used - Metal Ores - Iron ores",
                "Unused Domestic Extraction - Metal Ores - Iron ores",
            ],
            "weight": 1,
        },
        "Plomb": {
            "exiobase_keys": [
                "Domestic Extraction Used - Metal Ores - Lead ores",
                "Unused Domestic Extraction - Metal Ores - Lead ores",
            ],
            "weight": 1,
        },
        "Nickel": {
            "exiobase_keys": [
                "Domestic Extraction Used - Metal Ores - Nickel ores",
                "Unused Domestic Extraction - Metal Ores - Nickel ores",
            ],
            "weight": 1,
        },
        "Autres métaux non-ferreux": {
            "exiobase_keys": [
                "Domestic Extraction Used - Metal Ores - Other non-ferrous metal ores",
                "Unused Domestic Extraction - Metal Ores - Other non-ferrous metal ores",
            ],
            "weight": 1,
        },
        "Platine": {
            "exiobase_keys": [
                "Domestic Extraction Used - Metal Ores - PGM ores",
                "Unused Domestic Extraction - Metal Ores - PGM ores",
            ],
            "weight": 1,
        },
        "Argent": {
            "exiobase_keys": [
                "Domestic Extraction Used - Metal Ores - Silver ores",
                "Unused Domestic Extraction - Metal Ores - Silver ores",
            ],
            "weight": 1,
        },
        "Cassitérite": {
            "exiobase_keys": [
                "Domestic Extraction Used - Metal Ores - Tin ores",
                "Unused Domestic Extraction - Metal Ores - Tin ores",
            ],
            "weight": 1,
        },
        "Uranium et thorium": {
            "exiobase_keys": [
                "Domestic Extraction Used - Metal Ores - Uranium and thorium ores",
                "Unused Domestic Extraction - Metal Ores - Uranium and thorium ores",
            ],
            "weight": 1,
        },
        "Zinc": {
            "exiobase_keys": [
                "Domestic Extraction Used - Metal Ores - Zinc ores",
                "Unused Domestic Extraction - Metal Ores - Zinc ores",
            ],
            "weight": 1,
        },
        "Pierre de construction": {
            "exiobase_keys": [
                "Domestic Extraction Used - Non-Metallic Minerals - Building stones",
                "Unused Domestic Extraction - Non-Metallic Minerals - Building stones",
            ],
            "weight": 1,
        },
        "Minéraux chimiques et d'engrais": {
            "exiobase_keys": [
                "Domestic Extraction Used - Non-Metallic Minerals - Chemical and fertilizer minerals",
                "Unused Domestic Extraction - Non-Metallic Minerals - Chemical and fertilizer minerals",
            ],
            "weight": 1,
        },
        "Argiles et kaolin": {
            "exiobase_keys": [
                "Domestic Extraction Used - Non-Metallic Minerals - Clays and kaolin",
                "Unused Domestic Extraction - Non-Metallic Minerals - Clays and kaolin",
            ],
            "weight": 1,
        },
        "Gravier et sable": {
            "exiobase_keys": [
                "Domestic Extraction Used - Non-Metallic Minerals - Gravel and sand",
                "Unused Domestic Extraction - Non-Metallic Minerals - Gravel and sand",
            ],
            "weight": 1,
        },
        "Calcaire, gypse, craie, dolomite": {
            "exiobase_keys": [
                "Domestic Extraction Used - Non-Metallic Minerals - Limestone, gypsum, chalk, dolomite",
                "Unused Domestic Extraction - Non-Metallic Minerals - Limestone, gypsum, chalk, dolomite",
            ],
            "weight": 1,
        },
        "Autres minéraux": {
            "exiobase_keys": [
                "Domestic Extraction Used - Non-Metallic Minerals - Other minerals",
                "Unused Domestic Extraction - Non-Metallic Minerals - Other minerals",
            ],
            "weight": 1,
        },
        "Sel": {
            "exiobase_keys": [
                "Domestic Extraction Used - Non-Metallic Minerals - Salt",
                "Unused Domestic Extraction - Non-Metallic Minerals - Salt",
            ],
            "weight": 1,
        },
        "Ardoise": {
            "exiobase_keys": [
                "Domestic Extraction Used - Non-Metallic Minerals - Slate",
                "Unused Domestic Extraction - Non-Metallic Minerals - Slate",
            ],
            "weight": 1,
        },
    },
}


### COLORS ###
COLORS = list(plt.cm.tab10(np.arange(10))) + ["gold"]
COLORS_NO_FR = COLORS[1:]


### AGGREGATIONS FOR THE FIGURES ###
# These aggregations are only dedicated to make the figures more legible. They have no influence on the models nor on their results.
# The aggregation used in the models is set through the matrices from data/aggregation as an argument of Model objects in model.py, but it's not the purpose of this section.
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
