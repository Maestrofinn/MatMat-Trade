import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import sys

import plotly.express as px

###################
### LOCAL PATHS ###
###################

sys.path.append(os.sep.join(sys.path[0].split(os.sep)[:-1]))

BASE_DIR = pathlib.Path(__file__).parents[1]
DATA_DIR = BASE_DIR / "data"
AGGREGATION_DIR = DATA_DIR / "aggregation"
CAPITAL_CONS_DIR = DATA_DIR / "capital_consumption"
EXIOBASE_DIR = DATA_DIR / "exiobase"
MODELS_DIR = DATA_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_MULTIMODEL_DIR = FIGURES_DIR / "multimodel"
OUTPUTS_DIR = BASE_DIR/"outputs"

for path in [
    DATA_DIR,
    CAPITAL_CONS_DIR,
    EXIOBASE_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    FIGURES_MULTIMODEL_DIR,
    OUTPUTS_DIR
]:
    if not os.path.isdir(path):
        os.mkdir(path)


##############
### COLORS ###
##############

COLORS=px.colors.qualitative.D3+px.colors.qualitative.Alphabet+px.colors.qualitative.Dark24
# COLORS = list(plt.cm.tab10(np.arange(10))) + ["gold"]
COLORS_NO_FR = COLORS[1:]


######################
### OTHER SETTINGS ###
######################

## max increase of french imports param
CAP_IMPORTS_INCREASE_PARAM= 20/100

## pref_region and tradewar_region default regions
ALLIES=["EU"]
OPPONENTS=["China"]

## scenar_stressors by default
DEFAULT_SCENAR_STRESSORS = 'GES'


####################################
### AGGREGATIONS FOR THE FIGURES ###
####################################
# These aggregations are only dedicated to make the figures more legible. They have no influence on the models nor on their results.
# The aggregation used in the models is set through the matrices from data/aggregation as an argument of Model objects in model.py, but it's not the purpose of this section.
# The default REGIONS_AGG and SECTORS_AGG fit the model aggregation "opti_S".

## ofce, Transitions/sectors/MTOC
REGIONS_AGG = {
    "FR": ["FR"],
    "China+": ["China", "Asia"],
    "Europe+": ["EU", "Row Europe"],
    "North_Amer": ["United States", "North America"],
    "RoW": [
        "Middle East",
        "Africa",
        "Oceania",
        "Russia",
        "South America"
    ],
}

## opti_S
# REGIONS_AGG = {
#     "FR": ["FR"],
#     "UK, Norway, Switzerland": ["UK, Norway, Switzerland"],
#     "China+": ["China, RoW Asia and Pacific"],
#     "EU": ["EU"],
#     "RoW": [
#         "United States",
#         "Asia, Row Europe",
#         "RoW America,Turkey, Taïwan",
#         "RoW Middle East, Australia",
#         "Brazil, Mexico",
#         "South Africa",
#         "Japan, Indonesia, RoW Africa",
#     ],
# }

##Transitions/sectors/MTOC
SECTORS_AGG = {
    "Agriculture": [
        'Paddy rice',
        'Wheat',
        'Cereal grains nec',
        'Vegetables, fruit, nuts',
        'Oil seeds',
        'Sugar cane, sugar beet',
        'Plant-based fibers',
        'Crops nec',
        'Farm_animals',
        'Forestry',
        'Fishing'
    ],
    "Energy": [
        'Crude_coal',
        'Crude_oil',
        'Natural_gas',
        'Motor_fuels',
        'Other_fuels',
        'Coal',
        'Gas',
        'Biofuels',
        'Biogas',
        'Nuclear_fuel',
        'Electricity_by_nuclear',
        'Electricity_by_petroleum',
        'Electricity_by_gas',
        'Coal-fired_electricity',
        'Electricity_by_wind',
        'Electricity_by_solar',
        'Electricity_by_hydro',
        'Electricity_by_Geothermal',
        'Other_electricity',
        'Heat',
        'Gas_distribution',
        'Electricity_distribution'
     ],
    "Industry": [
        'Iron_ores',
        'Aluminium_ores',
        'Copper_ores',
        'Nickel ores and concentrates',
        'Lead_Zinc_Tin_ores',
        'Precious metal ores and concentrates',
        'Other_metal_ores',
        'Stone',
        'Sand and clay',
        'Other_NonMetallic_minerals',
        'Primary_Wood',
        'Secondary_Wood',
        'Primary_Pulp_and_paper',
        'Secondary_Pulp_and_paper',
        'Primary_Plastics',
        'Secondary_Plastics',
        'Primary_Glass',
        'Secondary_Glass',
        'Primary_Steel_iron',
        'Secondary_Steel_iron',
        'Primary_Aluminium',
        'Secondary_Aluminium',
        'Primary_Copper',
        'Secondary_Copper',
        'Primary_Other_metals',
        'Secondary_Other_metals',
        'Primary_Cement_lime_plaster',
        'Secondary_Cement_lime_plaster',
        'Other_Non-Metallic_Mineral_Materials',
        'Animal_Products',
        'Dairy Products',
        'Vegetable_Products',
        'Chemicals',
        'Clothing_Industry',
        'Metal_Product',
        'Computers_communication_equipments',
        'Machinery',
        'Other_manufacturing_industry'
    ],
    "Construction":[
        "Construction"
    ],
    "Transports":[
        'Motor_vehicles',
        'Oth_transport_equipment',
        'Rail_transport',
        'Other land transport',
        'Water_transport',
        'Air_transport'
    ],
    "Composite":[
        'Wood_Waste_incineration',
        'Other_Waste_incineration',
        'Landfill',
        'Water_Other_waste_Treatment',
        'Real estate',
        'Hotel_Restaurant',
        'Commerce',
        'Other_Business_Services',
        'Public_Services'
    ]
}

## ofce
# SECTORS_AGG = {
#     "Agriculture": ["Agriculture"],
#     "Energy": [
#         "Crude coal",
#         "Crude oil",
#         "Natural gas",
#         "Fossil fuels",
#         "Electricity and heat",
#     ],
#     "Industry": [
#         "Extractive industry",
#         "Biomass_industry",
#         "Clothing",
#         "Heavy_industry",
#         "Machinery",
#         "Electronics",
#     ],
#     "Construction":[
#         "Construction"
#     ],
#     "Transports": [
#         "Transport services",
#         "Automobile",
#         "Oth transport equipment",
#     ],
#     "Composite": ["Composite"],
# }

## opti_S
# SECTORS_AGG = {
#     "Agriculture": ["Agriculture"],
#     "Energy": [
#         "Crude coal",
#         "Crude oil",
#         "Natural gas",
#         "Fossil fuels",
#         "Electricity and heat",
#     ],
#     "Industry": [
#         "Extractive industry",
#         "Biomass_industry",
#         "Clothing",
#         "Heavy_industry",
#         "Automobile",
#         "Oth transport equipment",
#         "Machinery",
#         "Electronics",
#         "Construction",
#         "Transport services",
#     ],
#     "Composite": ["Composite"],
# }
