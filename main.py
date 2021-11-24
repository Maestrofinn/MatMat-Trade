""" Python main script of MatMat trade module

    Notes
    ------
    Fill notes if necessary

    """

# general
import sys
import os
import copy

# scientific
import numpy as np
import pandas as pd
import pymrio
import matplotlib.pyplot as plt

# local folder
from local_paths import data_dir
from local_paths import output_dir

# local library
from utils import Tools


###########################
# SETTINGS
###########################

# year to study in [*range(1995, 2022 + 1)]
base_year = 2015

# system type: pxp or ixi
system = 'pxp'

# agg name: to implement in agg_matrix.xlsx
agg_name = {
	'sector': 'ref',
	'region': 'ref'
}

# define filename concatenating settings
concat_settings = str(base_year) + '_' + \
	agg_name['sector']  + '_' +  \
	agg_name['region']

# set if rebuilding calibration from exiobase
calib = False


###########################
# READ/ORGANIZE/CLEAN DATA
###########################

# define file name
file_name = 'IOT_' + str(base_year) + '_' + system + '.zip'


# download data online
if not os.path.isfile(data_dir / file_name):

	pymrio.download_exiobase3(
	    storage_folder = data_dir,
	    system = system, 
	    years = base_year
	)


# import or build calibration data
if calib:

	# import exiobase data
	reference = pymrio.parse_exiobase3(
		data_dir / file_name
	)

	# isolate ghg emissions
	reference.ghg_emissions = Tools.extract_ghg_emissions(reference)

	# del useless extensions
	reference.remove_extension(['satellite', 'impacts'])

	# import agregation matrices
	agg_matrix = {
		key: pd.read_excel(
			data_dir / 'agg_matrix.xlsx',
			sheet_name = key + '_' + value
		) for (key, value) in agg_name.items()
	}
	agg_matrix['sector'].set_index(['category', 'sub_category', 'sector'], inplace = True)
	agg_matrix['region'].set_index(['Country name', 'Country code'], inplace = True)

	# apply regional and sectorial agregations
	reference.aggregate(
		region_agg = agg_matrix['region'].T.values,
		sector_agg = agg_matrix['sector'].T.values,
		region_names = agg_matrix['region'].columns.tolist(),
		sector_names = agg_matrix['sector'].columns.tolist()
	)

	# reset all to flows before saving
	reference = reference.reset_to_flows()
	reference.ghg_emissions.reset_to_flows()

	# save calibration data
	reference.save_all(
		data_dir / ('reference' + '_' + concat_settings)
	)

else:

	# import calibration data built with calib = True
	reference = pymrio.parse_exiobase3(
		data_dir / ('reference' + '_' + concat_settings)
	)


###########################
# CALCULATIONS
###########################

# calculate reference system
reference.calc_all()


# update extension calculations
reference.ghg_emissions_desag = Tools.recal_extensions_per_region(
	reference,
	'ghg_emissions'
)

# init counterfactual(s)
counterfactual = reference.copy()
counterfactual.remove_extension('ghg_emissions_desag')


# read param sets to shock reference system
## ToDo


# build conterfactual(s) using param sets
## ToDo


# calculate counterfactual(s) system
counterfactual.calc_all()
counterfactual.ghg_emissions_desag = Tools.recal_extensions_per_region(
	counterfactual,
	'ghg_emissions'
)


###########################
# FORMAT RESULTS
###########################

# save reference data base
reference.save_all(
	output_dir / ('reference' + '_' + concat_settings)   
)


# save conterfactural(s)
counterfactual.save_all(
	output_dir / ('counterfactual' + '_' + concat_settings)   
)


# concat results for visualisation
## ToDo
print(list(reference.get_extensions()))
#print(reference.ghg_emissions_desag.D_cba.sum(axis=0).FR) #empreinte carbone totale (peut faire si gaz en kgCO2eq)
print(type(reference.ghg_emissions_desag.D_cba))
ghg_list = list(reference.ghg_emissions_desag.D_cba.index.get_level_values(1)[:7])
for ghg in ghg_list:
    with plt.style.context('ggplot'):
        reference.ghg_emissions_desag.plot_account(ghg, figsize=(8,5))
        plt.savefig('figures/ref_'+ghg+'.png', dpi=300)
        plt.show()
###########################
# VISUALIZE
###########################

# reference analysis
## ToDo


# whole static comparative analysis
## ToDo