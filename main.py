""" Python main script of MatMat trade module

    Notes
    ------
    Fill notes if necessary

    """

# general
import sys
import os
import copy

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

nbsect = len(list(reference.get_sectors()))
sectors = list(reference.get_sectors())+list(reference.get_sectors())+list(reference.get_sectors())
quantities = [1 for i in range(len(sectors)) ]
moves =[]
for i in range(len(quantities)):
	if i<nbsect:
		moves.append(['BRICS','Europe'])
	if i>= nbsect and i<2*nbsect:
		moves.append(['RoW','Europe'])
	if i>= 2*nbsect:
		moves.append(['Other_OECD','Europe'])
# build conterfactual(s) using param sets
## ToDo
for i in range(len(quantities)):
	counterfactual.Z,counterfactual.Y = Tools.shock(list(reference.get_sectors()),counterfactual.Z,counterfactual.Y,moves[i][0],
	moves[i][1],sectors[i],quantities[i])
counterfactual.A = None
counterfactual.x = None
counterfactual.L = None

# calculate counterfactual(s) system
counterfactual.calc_all()
#print(counterfactual.Z)
counterfactual.ghg_emissions_desag = Tools.recal_extensions_per_region(
	counterfactual,
	'ghg_emissions'
)

#print(counterfactual.x)
#print(reference.x)
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
def visualisation(scenario,scenario_name,type_emissions='D_cba',saveghg=False):
	ghg_list = ['CO2', 'CH4', 'N2O', 'SF6', 'HFC', 'PFC']
	dict_fig_name = {'D_cba' : '_empreinte_carbone_fr_importation','D_pba' : '_emissions_territoriales_fr','D_imp' : '_emissions_importees_fr','D_exp' : '_emissions_exportees_fr'}
	dict_plot_title = {'D_cba' : 'Empreinte carbone de la France', 'D_pba' : 'Emissions territoriales françaises','D_imp' : 'Emissions importées en France','D_exp' : 'Emissions exportées par la France'}
	d_ = pd.DataFrame(getattr(scenario.ghg_emissions_desag,type_emissions))
	emissions_df = d_['FR']
	sumonsectors = emissions_df.sum(axis=1)
	total_ges_by_origin = sumonsectors.sum(level=0)
	liste_agg_ghg=[]
	for ghg in ghg_list:
		liste_agg_ghg.append(sumonsectors.iloc[sumonsectors.index.get_level_values(1)==ghg].sum(level=0))
	xs = ['total']+ghg_list
	dict_pour_plot = {'Total':total_ges_by_origin,'CO2':liste_agg_ghg[0],
	'CH4':liste_agg_ghg[1],'N2O':liste_agg_ghg[2],'SF6':liste_agg_ghg[3],
	'HFC':liste_agg_ghg[4],'PFC':liste_agg_ghg[5]}
	pour_plot=pd.DataFrame(data=dict_pour_plot,index=scenario.get_regions())
	pour_plot.transpose().plot.bar(stacked=True)
	plt.title(dict_plot_title[type_emissions]+" (scenario "+scenario_name+")")
	plt.ylabel("MtCO2eq")
	if saveghg :
		plt.savefig("figures/"+scenario_name+dict_fig_name[type_emissions]+".png")
	plt.close()
	for ghg in ghg_list:
		df = pd.DataFrame(None, index = scenario.get_sectors(), columns = scenario.get_regions())
		for reg in scenario.get_regions():
			df.loc[:,reg]=emissions_df.loc[(reg,ghg)]
		ax=df.plot.barh(stacked=True, figsize=(18,12))
		plt.grid()
		plt.xlabel("MtCO2eq")
		plt.title(dict_plot_title[type_emissions]+" de "+ghg+" par secteurs (scenario "+scenario_name+")")
		if saveghg :
			plt.savefig('figures/'+scenario_name+'_french_'+ghg+dict_fig_name[type_emissions]+'_provenance_sectors')
		plt.close()
	
		ax=emissions_df.sum(level=0).T.plot.barh(stacked=True, figsize=(18,12))
		plt.grid()
		plt.xlabel("MtCO2eq")
		plt.title(dict_plot_title[type_emissions]+" de tous GES par secteurs (scenario "+scenario_name+")")
		if saveghg :
			plt.savefig('figures/'+scenario_name+dict_fig_name[type_emissions]+'_provenance_sectors')
		#plt.show()
		plt.close()

###########################
# VISUALIZE
###########################

# reference analysis
## ToDo
for type in ['D_cba', 'D_pba', 'D_imp', 'D_exp'] :
	visualisation(reference,"Ref",type,True)
	visualisation(counterfactual,"Cont",type,True)
# whole static comparative analysis
## ToDo

def delta_CF(ref,contr):
	""" Compare les EC des deux scenarios, éventuellement par secteur
	"""
	ref_dcba = pd.DataFrame(ref.ghg_emissions_desag.D_cba)
	con_dcba = pd.DataFrame(contr.ghg_emissions_desag.D_cba)
	cf_ref = ref_dcba['FR'].sum(axis=1).sum(level=0)
	cf_con = con_dcba['FR'].sum(axis=1).sum(level=0)
	return 100*(cf_con/cf_ref - 1), 100*(cf_con.sum()/cf_ref.sum() -1), cf_ref, cf_con
res = delta_CF(reference,counterfactual)
print("Variation EC française par provenance")
print(res[0])
print(res[1])
print('Empreinte carbone référence :', res[2].sum(), 'MtCO2eq')
print('Empreinte carbone contrefactuel :', res[3].sum(), 'MtCO2eq')

def compa_monetaire(ref,contr):
	#unité = M€
	return counterfactual.x - reference.x
print("Variation de richesse de la transformation")
print(compa_monetaire(reference,counterfactual).sum(level=0).sum())

