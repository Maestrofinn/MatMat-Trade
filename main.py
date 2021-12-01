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
ghg_list = ['CO2', 'CH4', 'N2O', 'SF6', 'HFC', 'PFC']
sectors_list=list(reference.get_sectors())
reg_list = list(reference.get_regions())

ref_dcba = pd.DataFrame(reference.ghg_emissions_desag.D_cba)
ref_dpba=pd.DataFrame(reference.ghg_emissions_desag.D_pba)

#empreinte carbone française
empreinte_df = ref_dcba['FR']
print(empreinte_df)

sumonsectors = empreinte_df.sum(axis=1)
total_ges_by_origin = sumonsectors.sum(level=0)
liste_agg_ghg=[]
for ghg in ghg_list:
	liste_agg_ghg.append(sumonsectors.iloc[sumonsectors.index.get_level_values(1)==ghg].sum(level=0))
xs = ['total']+ghg_list
dict_pour_plot = {'Total':total_ges_by_origin,'CO2':liste_agg_ghg[0],
'CH4':liste_agg_ghg[1],'N2O':liste_agg_ghg[2],'SF6':liste_agg_ghg[3],
'HFC':liste_agg_ghg[4],'PFC':liste_agg_ghg[5]}
pour_plot=pd.DataFrame(data=dict_pour_plot,index=reg_list)
pour_plot.transpose().plot.bar(stacked=True)
plt.title("Empreinte carbone de la France")
plt.ylabel("kgCO2eq")
plt.savefig("figures/empreinte_carbone_fr_importation.png")
plt.show()


for ghg in ghg_list:
    df = pd.DataFrame(None, index = reference.get_sectors(), columns = reference.get_regions())
    for reg in reference.get_regions():
        df.loc[:,reg]=empreinte_df.loc[(reg,ghg)]
    ax=df.plot.barh(stacked=True)
    plt.grid()
    plt.xlabel("kgCO2eq")
    plt.title("Provenance des émissions de "+ghg+" françaises par secteurs")
    plt.savefig('figures/french_'+ghg+'emissions_provenance_sectors')
plt.show()


###########################
# VISUALIZE
###########################

# reference analysis
## ToDo


# whole static comparative analysis
## ToDo