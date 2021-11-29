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
#print(reference.ghg_emissions_desag.D_cba.sum(axis=0).FR) #empreinte carbone totale (peut faire si gaz en kgCO2eq)
ghg_list = list(reference.ghg_emissions_desag.D_cba.index.get_level_values(1)[:6])
sectors_list=list(reference.get_sectors())
reg_list = list(reference.get_regions())

#for ghg in ghg_list:
#    with plt.style.context('ggplot'):
#        reference.ghg_emissions_desag.plot_account(ghg, figsize=(8,5))
#        plt.savefig('figures/ref_'+ghg+'.png', dpi=300)
#        plt.show()
ref_dcba = pd.DataFrame(reference.ghg_emissions_desag.D_cba)
filtre_co2 = ref_dcba.index.get_level_values(1)=='CO2'
CO2_total_by_sector = ref_dcba.iloc[filtre_co2]['FR']

# with plt.style.context('ggplot'):
# 	CO2_total_by_sector.plot.barh(stacked=True)
# 	plt.title("Répartition des émissions de chaque région par secteur")
# 	plt.savefig('figures/French_CO2_by_sector_each_region.png')
	#plt.show()
# with plt.style.context('ggplot'):
# 	CO2_total_by_sector.transpose().plot.barh(stacked=True)
# 	plt.title("Origine des émissions de CO2 pour chaque secteur")
# 	plt.savefig('figures/French_CO2_by_origin_each_sector.png')
# 	#plt.show()

ch4toco2=72
n20toco2=289
sf6toco2=15100

Total_ghg_desag_fr = ref_dcba['FR']
sumonsectors = Total_ghg_desag_fr.sum(axis=1)
#print(sumonsectors)
sumonsectors.iloc[sumonsectors.index.get_level_values(1)=='CH4'] *= ch4toco2
sumonsectors.iloc[sumonsectors.index.get_level_values(1)=='N2O'] *= n20toco2
sumonsectors.iloc[sumonsectors.index.get_level_values(1)=='SF6'] *= sf6toco2
#print(sumonsectors)

total_ges_by_origin = sumonsectors.sum(level=0)
print(total_ges_by_origin)
liste_agg_ghg=[]
for ghg in ghg_list:
	liste_agg_ghg.append(sumonsectors.iloc[sumonsectors.index.get_level_values(1)==ghg].sum(level=0))
xs = ['total']+ghg_list
dict_pour_plot = {'Total':total_ges_by_origin,'CO2':liste_agg_ghg[0],
'CH4':liste_agg_ghg[1],'N2O':liste_agg_ghg[2],'SF6':liste_agg_ghg[3],
'HFC':liste_agg_ghg[4],'PFC':liste_agg_ghg[5]}
pour_plot=pd.DataFrame(data=dict_pour_plot,index=reg_list)
print(pour_plot)
pour_plot.transpose().plot.bar(stacked=True)
plt.savefig("figures/empreinte_carbone_fr_importation.png")
plt.show()

#Other version :
ghg_list = list(reference.ghg_emissions_desag.D_cba.index.get_level_values(1)[:6])
print(ghg_list)
for ghg in ghg_list:
    df = pd.DataFrame(None, index = reference.get_sectors(), columns = reference.get_regions())
    for reg in reference.get_regions():
        df.loc[:,reg]=ref_dcba.transpose().loc['FR',(reg,ghg)]
    ax=df.plot.barh(stacked=True)
    plt.grid()
    plt.xlabel("kg")
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