""" Python main script of MatMat trade module

	Notes
	------
	Fill notes if necessary

	"""

# general
import re
import sys
import os
import copy

import warnings
from weakref import ref
warnings.simplefilter(action='ignore', category=FutureWarning)

# scientific
import numpy as np
import pandas as pd
import pymrio
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

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
	print("Début calib")
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
            data_dir / 'agg_matrix_opti.xlsx',
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
	print("Fin calib")

else:

	# import calibration data built with calib = True
    reference = pymrio.parse_exiobase3(
		data_dir / ('reference' + '_' + concat_settings)
	)


###########################
# CALCULATIONS
###########################




reference.calc_all()
reference.ghg_emissions_desag = Tools.recal_extensions_per_region(
	reference,
	'ghg_emissions'
)

replace_reagg = True

if replace_reagg :
	Tools.replace_reagg_scenar_attributes(reference,reaggregation_matrix = pd.read_excel(data_dir / 'agg_matrix_opti.xlsx', sheet_name = 'region_ref'))
	



# save reference data base
reference.save_all(
    output_dir / ('reference' + '_' + concat_settings)  
)

# init counterfactual(s)
counterfactual = reference.copy()
counterfactual.remove_extension('ghg_emissions_desag')

# read param sets to shock reference system
## ToDo
nbsect = len(list(reference.get_sectors()))

def get_least(sector,reloc):
	#par défaut on ne se laisse pas la possibilité de relocaliser en FR
	M = reference.ghg_emissions_desag.M.sum()
	#compute the part of the french economy that depends on broad activities, for each sector :
	final_demfr= reference.Y['FR'].drop(['FR']).sum(axis=1)
	interdemfr=reference.Z['FR'].drop(['FR']).sum(axis=1)
	import_demand_FR = (final_demfr+interdemfr).sum(level=1)

	regs = list(reference.get_regions())[1:]

	if reloc:
		regs = list(reference.get_regions())
	ind=0
	for i in range(1,len(regs)):
		if M[regs[i],sector] < M[regs[ind],sector] and reference.Z.loc[regs[i]].drop(columns=regs[i]).sum(axis=1).loc[sector] > import_demand_FR.loc[sector] : # pour choisir une région comme région de report, elle doit au moins déjà exporter l'équivalent de la partie importée de la demande française
			ind=i
	return regs[ind]

def sort_by_content(sector,regs):
	#sort all regions by carbon content of a sector
	#carbon contents
	M = reference.ghg_emissions_desag.M.sum()
	carbon_content_sector = [M[regs[i],sector] for i in range(len(regs))]
	index_sorted = np.argsort(carbon_content_sector)
	return index_sorted

def worst_moves(sector,reloc):
	if reloc:
		regs = list(reference.get_regions())
	else:
		regs = list(reference.get_regions())[1:] #remove FR
	index_sorted = list(reversed(sort_by_content(sector,regs)))
	sectors_list = list(reference.get_sectors())
	demcats = list(reference.get_Y_categories())

	#compute the part of the french economy that depends on broad activities, for this sector :
	final_demfr= reference.Y['FR'].drop(['FR']).sum(axis=1).sum(level=1).loc[sector]
	interdemfr=reference.Z['FR'].drop(['FR']).sum(axis=1).sum(level=1).loc[sector]
	import_demand_FR = final_demfr+interdemfr
	#part de chaque secteur français dans les importations intermédiaires françaises depuis un secteur étranger
	part_prod_secteurs =[] 
	part_dem_secteurs = []
	for sec in sectors_list:
		part_prod_secteurs.append(reference.Z[('FR',sec)].drop(['FR']).sum(level=1).loc[sector]/import_demand_FR)
	for dem in demcats:
		part_dem_secteurs.append(reference.Y[('FR',dem)].drop(['FR']).sum(level=1).loc[sector]/import_demand_FR)
	
	#parts des importations françaises *totales pour un secteur* à importer depuis le 1er best, 2nd best...
	nbreg=len(regs)
	nbsect=len(sectors_list)
	nbdemcats = len(demcats)
	parts_sects = np.zeros((nbreg,nbsect))
	parts_demcats = np.zeros((nbreg,nbdemcats))
	#construction of french needs of imports
	totalfromsector = np.zeros(nbsect)
	totalfinalfromsector = np.zeros(nbdemcats)
	for j in range(nbsect):
		#sum on regions of imports of imports of sector for french sector j
		totalfromsector[j] = np.sum([reference.Z['FR'].drop('FR')[sectors_list[j]].loc[(regs[k],sector)] for k in range(nbreg)]) 
	for j in range(nbdemcats):
		totalfinalfromsector[j] = np.sum([reference.Y['FR'].drop('FR')[demcats[j]].loc[(regs[k],sector)] for k in range(nbreg)])

	remaining_reg_export = np.zeros(nbreg)
	for i in range(nbreg):
		my_best = regs[index_sorted[i]] #region with ith lowest carbon content for this sector
		reg_export = reference.Z.drop(columns=my_best).sum(axis=1).loc[(my_best,sector)] #exports from this reg/sec
		remaining_reg_export[index_sorted[i]] = reg_export
		for j in range(nbsect):
			if np.sum(parts_sects[:,j]) < totalfromsector[j] and remaining_reg_export[index_sorted[i]] >0:
				#if imp demand from sector j is not satisfied and if my_best can still export some sector
				alloc=0
				if remaining_reg_export[index_sorted[i]]>totalfromsector[j]:
					alloc=totalfromsector[j]
				else:
					alloc=remaining_reg_export[index_sorted[i]]
				parts_sects[index_sorted[i],j] = alloc
				remaining_reg_export[index_sorted[i]] -= alloc
				
		for j in range(nbdemcats):
			#idem for final demand categories
			if remaining_reg_export[index_sorted[i]] >0 and np.sum(parts_demcats[:,j]) < totalfinalfromsector[j]:
				alloc=0
				if remaining_reg_export[index_sorted[i]] > totalfinalfromsector[j]:
					alloc = totalfinalfromsector[j]
				else:
					alloc = remaining_reg_export[index_sorted[i]]
				parts_demcats[index_sorted[i],j] = alloc
				remaining_reg_export[index_sorted[i]] -= alloc
	return parts_sects,parts_demcats,index_sorted

def best_moves(sector,reloc):
	if reloc:
		regs = list(reference.get_regions())
	else:
		regs = list(reference.get_regions())[1:] #remove FR
	index_sorted = sort_by_content(sector,regs)
	sectors_list = list(reference.get_sectors())
	demcats = list(reference.get_Y_categories())

	#compute the part of the french economy that depends on broad activities, for this sector :
	final_demfr= reference.Y['FR'].drop(['FR']).sum(axis=1).sum(level=1).loc[sector]
	interdemfr=reference.Z['FR'].drop(['FR']).sum(axis=1).sum(level=1).loc[sector]
	import_demand_FR = final_demfr+interdemfr
	#part de chaque secteur français dans les importations intermédiaires françaises depuis un secteur étranger
	part_prod_secteurs =[] 
	part_dem_secteurs = []
	for sec in sectors_list:
		part_prod_secteurs.append(reference.Z[('FR',sec)].drop(['FR']).sum(level=1).loc[sector]/import_demand_FR)
	for dem in demcats:
		part_dem_secteurs.append(reference.Y[('FR',dem)].drop(['FR']).sum(level=1).loc[sector]/import_demand_FR)
	
	#parts des importations françaises *totales pour un secteur* à importer depuis le 1er best, 2nd best...
	nbreg=len(regs)
	nbsect=len(sectors_list)
	nbdemcats = len(demcats)
	parts_sects = np.zeros((nbreg,nbsect))
	parts_demcats = np.zeros((nbreg,nbdemcats))
	#construction of french needs of imports
	totalfromsector = np.zeros(nbsect)
	totalfinalfromsector = np.zeros(nbdemcats)
	for j in range(nbsect):
		#sum on regions of imports of imports of sector for french sector j
		totalfromsector[j] = np.sum([reference.Z['FR'].drop('FR')[sectors_list[j]].loc[(regs[k],sector)] for k in range(nbreg)]) 
	for j in range(nbdemcats):
		totalfinalfromsector[j] = np.sum([reference.Y['FR'].drop('FR')[demcats[j]].loc[(regs[k],sector)] for k in range(nbreg)])

	#export capacities of each regions
	remaining_reg_export = np.zeros(nbreg)
	for i in range(nbreg):
		my_best = regs[index_sorted[i]] #region with ith lowest carbon content for this sector
		reg_export = reference.Z.drop(columns=my_best).sum(axis=1).loc[(my_best,sector)] #exports from this reg/sec
		remaining_reg_export[index_sorted[i]] = reg_export

	for j in range(nbsect):
		covered = 0
		for i in range(nbreg):
			if covered < totalfromsector[j] and remaining_reg_export[index_sorted[i]] >0:
				#if imp demand from sector j is not satisfied and if my_best can still export some sector
				if remaining_reg_export[index_sorted[i]]>totalfromsector[j]-covered:
					alloc=totalfromsector[j]-covered
				else:
					alloc=remaining_reg_export[index_sorted[i]]
				parts_sects[index_sorted[i],j] = alloc
				remaining_reg_export[index_sorted[i]] -= alloc
				covered+=alloc
				
	for j in range(nbdemcats):
		#idem for final demand categories
		covered = 0
		for i in range(nbreg):
			if  covered < totalfinalfromsector[j] and remaining_reg_export[index_sorted[i]] >0 :
				if remaining_reg_export[index_sorted[i]] > totalfinalfromsector[j]-covered:
					alloc = totalfinalfromsector[j]-covered
				else:
					alloc = remaining_reg_export[index_sorted[i]]
				parts_demcats[index_sorted[i],j] = alloc
				remaining_reg_export[index_sorted[i]] -= alloc
				covered+=alloc
	return parts_sects,parts_demcats,index_sorted

def scenar_bestv2(reloc=False):
	sectors_list = list(reference.get_sectors())
	moves = {}
	for sector in sectors_list:
		part_sec, part_dem,index_sorted = best_moves(sector,reloc)
		moves[sector] = {'parts_sec' : part_sec, 'parts_dem':part_dem, 'sort':index_sorted, 'reloc':reloc}
	return sectors_list, moves

def scenar_worstv2(reloc=False):
	sectors_list = list(reference.get_sectors())
	moves = {}
	for sector in sectors_list:
		part_sec, part_dem,index_sorted = worst_moves(sector,reloc)
		moves[sector] = {'parts_sec' : part_sec, 'parts_dem':part_dem, 'sort':index_sorted, 'reloc':reloc}
	return sectors_list, moves


def get_worst(sector,reloc):
	#par défaut on ne se laisse pas la possibilité de relocaliser en FR
	M = reference.ghg_emissions_desag.M.sum()
	regs = list(reference.get_regions())[1:]
	if reloc:
		regs = list(reference.get_regions())
	ind=0
	for i in range(1,len(regs)):
		if M[regs[i],sector] > M[regs[ind],sector]:
			ind=i
	return regs[ind]

#construction du scénario least intense
def scenar_best(reloc=False,deloc=False):
	sectors_list = list(reference.get_sectors())
	sectors_gl = []
	moves_gl = []
	for sector in sectors_list:
		best = get_least(sector,reloc)
		if deloc:
			for i in range(len(list(reference.get_regions()))-1):
				sectors_gl.append(sector)
		else:
			for i in range(len(list(reference.get_regions()))-2):
				sectors_gl.append(sector)
		for reg in list(reference.get_regions()):
			if deloc:
				if reg!=best:
					moves_gl.append([reg,best])
			else:
				if reg!=best :
					if reg!='FR':
						moves_gl.append([reg,best])
	quantities = [1 for i in range(len(sectors_gl))]
	return sectors_gl, moves_gl, quantities

def scenar_worst(reloc=False,deloc=False):
	sectors_list = list(reference.get_sectors())
	sectors_gl = []
	moves_gl = []
	for sector in sectors_list:
		worst = get_worst(sector,reloc)
		if deloc:
			for i in range(len(list(reference.get_regions()))-1):
				sectors_gl.append(sector)
		else:
			for i in range(len(list(reference.get_regions()))-2):
				sectors_gl.append(sector)
		for reg in list(reference.get_regions()):
			if deloc:
				if reg!=worst:
					moves_gl.append([reg,worst])
			else:
				if reg!=worst :
					if reg!='FR':
						moves_gl.append([reg,worst])
	quantities = [1 for i in range(len(sectors_gl))]
	return sectors_gl, moves_gl, quantities


def scenar_pref_europe():
	nbreg = len(list(reference.get_regions()))
	sectors = (nbreg-2)*list(reference.get_sectors())
	quantities = [1 for i in range(len(sectors)) ]
	moves =[]
	for i in range(nbreg):
		reg = reference.get_regions()[i]
		if reg != 'Europe' and reg != 'FR':
			for j in range(len(list(reference.get_sectors()))):
				moves.append([reg,'Europe'])
	return sectors,moves,quantities

def scenar_pref_europev3(reloc=False):
	if reloc:
		regs = list(reference.get_regions())
	else:
		regs = list(reference.get_regions())[1:] #remove FR
	sectors_list = list(reference.get_sectors())
	demcats = list(reference.get_Y_categories())
	nbdemcats=len(demcats)
	nbreg=len(regs)
	moves = {}
	for i in range(nbsect):
		#initialization of outputs
		parts_sects = {}
		parts_dem = {}
		for r in regs:
			parts_sects[r] = np.zeros(nbsect)
			parts_dem[r] = np.zeros(nbdemcats)

		#construction of french needs of imports
		totalfromsector = np.zeros(nbsect)
		totalfinalfromsector = np.zeros(nbdemcats)
		for j in range(nbsect):
			#sum on regions of imports of imports of sector for french sector j
			totalfromsector[j] = np.sum([reference.Z['FR'].drop('FR')[sectors_list[j]].loc[(regs[k],sectors_list[i])] for k in range(nbreg)]) 
		for j in range(nbdemcats):
			totalfinalfromsector[j] = np.sum([reference.Y['FR'].drop('FR')[demcats[j]].loc[(regs[k],sectors_list[i])] for k in range(nbreg)])
		
		# exports capacity of all regions for sector i
		reg_export = {}
		for r in range(nbreg):
			reg_export[regs[r]] = reference.Z.drop(columns=regs[r]).sum(axis=1).loc[(regs[r],sectors_list[i])] #exports from this reg/sec
		
		remaining_reg_export_UE = reg_export['Europe']
		for j in range(nbsect):
			if totalfromsector[j] !=0:
				if remaining_reg_export_UE > 0:
					#if europe can still export some sector[i]
					if remaining_reg_export_UE>totalfromsector[j]:
						alloc=totalfromsector[j]
					else:
						alloc= reference.Z.loc[('Europe',sectors_list[i]),('FR',sectors_list[j])] #tout ou rien ici
					parts_sects['Europe'][j] = alloc
					remaining_reg_export_UE -= alloc
					#remove from other regions a part of what has been assigned to the EU
					# this part corresponds to the part of the country in original french imports for sector j 
					for r in regs:
						if r != 'Europe':
							parts_sects[r][j] =  reference.Z.loc[(r,sectors_list[i]),('FR',sectors_list[j])]* (1- alloc /totalfromsector[j])

		for j in range(nbdemcats):
			if totalfinalfromsector[j] != 0:
				if remaining_reg_export_UE > 0:
					#if europe can still export some sector[i]
					if remaining_reg_export_UE>totalfinalfromsector[j]:
						alloc=totalfinalfromsector[j]
					else:
						alloc=reference.Y.loc[('Europe',sectors_list[i]),('FR',demcats[j])] #tout ou rien ici
					parts_dem['Europe'][j] = alloc
					remaining_reg_export_UE -= alloc
					#remove from other regions a part of what has been assigned to the EU
					# this part corresponds to the part of the country in original french imports for sector j 
					for r in regs:
						if r != 'Europe':
							parts_sects[r][j] =  reference.Y.loc[(r,sectors_list[i]),('FR',demcats[j])]* (1- alloc /totalfinalfromsector[j])

		moves[sectors_list[i]] = {'parts_sec' : parts_sects, 'parts_dem':parts_dem, 'sort':[i for i in range(len(regs))], 'reloc':reloc}
	return sectors_list,moves

def scenar_guerre_chine(reloc=False):
	if reloc:
		regs = list(reference.get_regions())
	else:
		regs = list(reference.get_regions())[1:] #remove FR
	sectors_list = list(reference.get_sectors())
	demcats = list(reference.get_Y_categories())
	nbdemcats=len(demcats)
	nbreg=len(regs)
	moves = {}
	for i in range(nbsect):
		#initialization of outputs
		parts_sects = {}
		parts_dem = {}
		for r in regs:
			parts_sects[r] = np.zeros(nbsect)
			parts_dem[r] = np.zeros(nbdemcats)

		#construction of french needs of imports
		totalfromsector = np.zeros(nbsect)
		fromchinasector = np.zeros(nbsect)
		totalfinalfromsector = np.zeros(nbdemcats)
		finalfromchinasector = np.zeros(nbdemcats)
		for j in range(nbsect):
			#sum on regions of imports of imports of sector for french sector j
			totalfromsector[j] = np.sum([reference.Z['FR'].drop('FR')[sectors_list[j]].loc[(regs[k],sectors_list[i])] for k in range(nbreg)]) 
			fromchinasector[j] = reference.Z['FR'].drop('FR')[sectors_list[j]].loc[('Chinafrica',sectors_list[i])]
		
		for j in range(nbdemcats):
			totalfinalfromsector[j] = np.sum([reference.Y['FR'].drop('FR')[demcats[j]].loc[(regs[k],sectors_list[i])] for k in range(nbreg)])
			finalfromchinasector[j] = reference.Y['FR'].drop('FR')[demcats[j]].loc[('Chinafrica',sectors_list[i])]
		# exports capacity of all regions for sector i
		reg_export = {}
		for r in range(nbreg):
			reg_export[regs[r]] = reference.Z.drop(columns=regs[r]).sum(axis=1).loc[(regs[r],sectors_list[i])] #exports from this reg/sec
		
		for j in range(nbsect):
			if totalfromsector[j] !=0:
				for r in regs:
					if r != 'Chinafrica':
						old = reference.Z.loc[(r,sectors_list[i]),('FR',sectors_list[j])]
						parts_sects[r][j] = old
				if fromchinasector[j] > 0:
					for r in regs:
						if r!='Chinafrica':
							old = reference.Z.loc[(r,sectors_list[i]),('FR',sectors_list[j])]
							if fromchinasector[j] +old < reg_export[r]:
								alloc = fromchinasector[j]
								parts_sects[r][j] += alloc
								fromchinasector[j]=0
								reg_export[r]-=alloc
								break
							else:
								alloc = reg_export[r]
								reg_export[r]-=alloc
								fromchinasector[j] -= alloc
					parts_sects['Chinafrica'][j] = fromchinasector[j]

		for j in range(nbdemcats):
			if totalfinalfromsector[j] != 0:
				for r in regs:
					if r != 'Chinafrica':
						old = reference.Y.loc[(r,sectors_list[i]),('FR',demcats[j])]
						parts_dem[r][j] = old
				if finalfromchinasector[j] > 0:
					for r in regs:
						if r != 'Chinafrica':
							old= reference.Y.loc[(r,sectors_list[i]),('FR',demcats[j])]
							if finalfromchinasector[j]+old < reg_export[r]:
								alloc = finalfromchinasector[j]
								parts_dem[r][j] += alloc
								finalfromchinasector[j]=0
								break
							else:
								alloc = reg_export[r]
								finalfromchinasector[j] -= alloc
					parts_dem['Chinafrica'][j] = finalfromchinasector[j]

		moves[sectors_list[i]] = {'parts_sec' : parts_sects, 'parts_dem':parts_dem, 'sort':[i for i in range(len(regs))], 'reloc':reloc}
	return sectors_list,moves
	
# build conterfactual(s) using param sets
## ToDo
sectors_list=list(reference.get_sectors())
reg_list=list(reference.get_regions())
demcat_list = list(reference.get_Y_categories())

#sectors,moves = scenar_bestv2()
#sectors,moves = scenar_pref_europev3()
sectors,moves = scenar_guerre_chine()
for sector in sectors:
	counterfactual.Z,counterfactual.Y = Tools.shockv3(sectors,demcat_list,reg_list,counterfactual.Z,counterfactual.Y,moves[sector],sector)

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
# ###########################
# # FORMAT RESULTS
# ###########################



#save conterfactural(s)
counterfactual.save_all(
   output_dir / ('counterfactual' + '_' + concat_settings)  
)
# concat results for visualisation
## ToDo
ghg_list = ['CO2', 'CH4', 'N2O', 'SF6', 'HFC', 'PFC']
sectors_list=list(reference.get_sectors())
reg_list = list(reference.get_regions())

def vision_commerce():
	df_eco_ref = reference.Y['FR'].sum(axis=1)+reference.Z['FR'].sum(axis=1)
	df_eco_cont = counterfactual.Y['FR'].sum(axis=1)+counterfactual.Z['FR'].sum(axis=1)

	comm_cumul_non_fr = pd.DataFrame({'ref':[df_eco_ref.sum(level=0)[r] for r in reg_list[1:]],
	'cont': [df_eco_cont.sum(level=0)[r] for r in reg_list[1:]]}, index =reg_list[1:])
	comm_cumul_non_fr.T.plot.barh(stacked=True)
	plt.title("Importations totales françaises")
	plt.tight_layout()
	plt.savefig('figures/commerce_imports_totales')
	plt.show()

	dict_sect_plot = {}
	for sec in sectors_list:
		dict_sect_plot[(sec,'ref')] = [df_eco_ref.loc[(r,sec)]/df_eco_ref.drop(['FR']).sum(level=1).loc[sec] for r in reg_list[1:]]
		dict_sect_plot[(sec,'cont')] = []
		for r in reg_list[1:]:
			if df_eco_cont.drop(['FR']).sum(level=1).loc[sec] !=0:
				dict_sect_plot[(sec,'cont')].append(df_eco_cont.loc[(r,sec)]/df_eco_cont.drop(['FR']).sum(level=1).loc[sec])
			else :
				dict_sect_plot[(sec,'cont')].append(df_eco_cont.loc[(r,sec)])

	df_plot = pd.DataFrame(data=dict_sect_plot,index=reg_list[1:])
	#print(df_plot)
	ax=df_plot.T.plot.barh(stacked=True, figsize=(20,16))
	plt.title("Part de chaque région dans les importations françaises")
	plt.tight_layout()
	plt.savefig('figures/commerce_parts_imports_secteur.png')
	plt.show()

def visualisation_carbone(scenario,scenario_name,type_emissions='D_cba',saveghg=False):
	ghg_list = ['CO2', 'CH4', 'N2O', 'SF6', 'HFC', 'PFC']
	dict_fig_name = {'D_cba' : '_empreinte_carbone_fr_importation','D_pba' : '_emissions_territoriales_fr','D_imp' : '_emissions_importees_fr','D_exp' : '_emissions_exportees_fr'}
	dict_plot_title = {'D_cba' : 'Empreinte carbone de la France', 'D_pba' : 'Emissions territoriales françaises','D_imp' : 'Emissions importées en France','D_exp' : 'Emissions exportées vers la France'}
	d_ = pd.DataFrame(getattr(scenario.ghg_emissions_desag,type_emissions))
	if scenario_name =="Cont":
		#pour contrefactuel on affiche la barre de la reference aussi
		emissions_df = d_['FR']
		em_df_ref = pd.DataFrame(getattr(reference.ghg_emissions_desag,type_emissions))['FR']
		sumonsectors = emissions_df.sum(axis=1)
		sumonsectors_ref = em_df_ref.sum(axis=1)
		total_ges_by_origin = sumonsectors.sum(level=0)
		total_ges_by_origin_ref = sumonsectors_ref.sum(level=0)
		liste_agg_ghg=[]
		liste_agg_ghg_ref=[]
		for ghg in ghg_list:
			liste_agg_ghg.append(sumonsectors.iloc[sumonsectors.index.get_level_values(1)==ghg].sum(level=0))
			liste_agg_ghg_ref.append(sumonsectors_ref.iloc[sumonsectors_ref.index.get_level_values(1)==ghg].sum(level=0))
		dict_pour_plot = {('Total','cont'):total_ges_by_origin,('Total','ref'):total_ges_by_origin_ref,
		('CO2','cont'):liste_agg_ghg[0],('CO2','ref'):liste_agg_ghg_ref[0],
		('CH4','cont'):liste_agg_ghg[1],('CH4','ref'):liste_agg_ghg_ref[1],
		('N2O','cont'):liste_agg_ghg[2],('N2O','ref'):liste_agg_ghg_ref[2],
		('SF6','cont'):liste_agg_ghg[3],('SF6','ref'):liste_agg_ghg_ref[3],
		('HFC','cont'):liste_agg_ghg[4],('HFC','ref'):liste_agg_ghg_ref[4],
		('PFC','cont'):liste_agg_ghg[5],('PFC','ref'):liste_agg_ghg_ref[5]}
	else:
		emissions_df = d_['FR']
		sumonsectors = emissions_df.sum(axis=1)
		total_ges_by_origin = sumonsectors.sum(level=0)
		liste_agg_ghg=[]
		for ghg in ghg_list:
			liste_agg_ghg.append(sumonsectors.iloc[sumonsectors.index.get_level_values(1)==ghg].sum(level=0))
		dict_pour_plot = {'Total':total_ges_by_origin,'CO2':liste_agg_ghg[0],
		'CH4':liste_agg_ghg[1],'N2O':liste_agg_ghg[2],'SF6':liste_agg_ghg[3],
		'HFC':liste_agg_ghg[4],'PFC':liste_agg_ghg[5]}

	pour_plot=pd.DataFrame(data=dict_pour_plot,index=scenario.get_regions())
	pour_plot.transpose().plot.bar(stacked=True,rot=45,figsize=(18,12))
	plt.title(dict_plot_title[type_emissions]+" (scenario "+scenario_name+")")
	plt.ylabel("MtCO2eq")
	plt.savefig("figures/"+scenario_name+dict_fig_name[type_emissions]+".png")
	plt.close()
	if saveghg :
		for ghg in ghg_list:
			df = pd.DataFrame(None, index = scenario.get_sectors(), columns = scenario.get_regions())
			for reg in scenario.get_regions():
				df.loc[:,reg]=emissions_df.loc[(reg,ghg)]
			ax=df.plot.barh(stacked=True, figsize=(18,12))
			plt.grid()
			plt.xlabel("MtCO2eq")
			plt.title(dict_plot_title[type_emissions]+" de "+ghg+" par secteurs (scenario "+scenario_name+")")
			plt.savefig('figures/'+scenario_name+'_french_'+ghg+dict_fig_name[type_emissions]+'_provenance_sectors')
			plt.close()
	dict_sect_plot = {}
	for i in range(len(list(scenario.get_sectors()))):
		sector = list(scenario.get_sectors())[i]
		dict_sect_plot[sector] = {'cont':emissions_df.sum(level=0)[sector],'ref':em_df_ref.sum(level=0)[sector]}
	reform = {(outerKey, innerKey): values for outerKey, innerDict in dict_sect_plot.items() for innerKey, values in innerDict.items()}
	df_plot = pd.DataFrame(data=reform)
	ax=df_plot.T.plot.barh(stacked=True, figsize=(20,16))
	plt.grid()
	plt.xlabel("MtCO2eq")
	plt.title(dict_plot_title[type_emissions]+" de tous GES par secteurs (scenario "+scenario_name+")")
	plt.savefig('figures/'+scenario_name+dict_fig_name[type_emissions]+'_provenance_sectors')
	#plt.show()
	plt.close()

###########################
# VISUALIZE
###########################
def heat_S(type):
    S = reference.ghg_emissions_desag.S.sum()
    M = reference.ghg_emissions_desag.M.sum()
    sec_reg = []
    for reg in reg_list:
        in_reg=[]
        for sector in sectors_list:
            if type=='consommation':
                in_reg.append(M[reg,sector])
            if type=='production':
                in_reg.append(S[reg,sector])
        sec_reg.append(in_reg)
    df = pd.DataFrame(data=sec_reg,columns=sectors_list,index=reg_list).T
    df_n = df.div(df.max(axis=1), axis=0)*100
    if type=='consommation':
        title="Contenu carbone du bien importé"
    if type=='production':
        title="Intensité carbone de la production"
    fig, ax = plt.subplots()
    sns.heatmap(df_n,cmap='coolwarm', ax=ax,linewidths=1, linecolor='black').set_title(title)
    fig.tight_layout()
    plt.savefig('figures/heatmap_intensite_'+type)
    plt.show()
#heat_S('consommation')
#heat_S('production')

# reference analysis
## ToDo
for type in ['D_cba', 'D_imp'] :
	#visualisation(reference,"Ref",type,saveghg=False)
	visualisation_carbone(counterfactual,"Cont",type,saveghg=False)
vision_commerce()
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
ref_dcba = pd.DataFrame(reference.ghg_emissions_desag.D_cba)
con_dcba = pd.DataFrame(counterfactual.ghg_emissions_desag.D_cba)
print('EC fossiles référence :', ref_dcba['FR','Crude oil'].sum(), 'MtCO2eq', ref_dcba['FR','Crude coal'].sum(),
 'MtCO2eq', ref_dcba['FR','Natural gas'].sum(), 'MtCO2eq' )
print('EC fossiles contrefactuel :', con_dcba['FR','Crude oil'].sum(), 'MtCO2eq', con_dcba['FR','Crude coal'].sum(), 
'MtCO2eq', con_dcba['FR','Natural gas'].sum(), 'MtCO2eq')

def compa_monetaire(ref,contr):
	#unité = M€
	return counterfactual.x - reference.x
print("Variation de richesse de la transformation")
print(compa_monetaire(reference,counterfactual).sum(level=0).sum())