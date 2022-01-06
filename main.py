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
            data_dir / 'agg_matrix2.xlsx',
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

# calculate reference system
reference.calc_all()


# update extension calculations
reference.ghg_emissions_desag = Tools.recal_extensions_per_region(
    reference,
    'ghg_emissions'
)

# save reference data base
reference.save_all(
    output_dir / ('reference' + '_' + concat_settings)  
)

plot_scatter_desag_spatiale = False
if plot_scatter_desag_spatiale:
    imp_FR_tot = reference.Z['FR'].drop(labels='FR',axis=0,level=0).sum(axis=1).sum(level=0)
    total_imp = imp_FR_tot.sum()
    part_imp = imp_FR_tot/total_imp *100

    list_eur = ['Austria','Belgium','Bulgaria','Cyprus','Czech Republic','Germany','Denmark','Estonia','Spain',
    'Finland','Greece','Croatia','Hungary','Ireland','Italy','Lithuania','Luxembourg','Latvia','Malta','Netherlands',
    'Poland','Romania','Sweden','Slovakia']
    list_brics = ['Brazil','Russia','India','China','South Africa']
    list_row = ['Taiwan','Indonesia','RoW Asia and Pacific','RoW America','RoW Europe',
    'RoW Africa','RoW Middle East']
    list_otheroecd = ['United Kingdom','United States','Japan','Canada',
    'South Korea', 'Mexico', 'Australia','Switzerland','Turkey','Norway']


    M=reference.ghg_emissions_desag.M.sum().drop(labels='FR',axis=0,level=0)
    M_tot = M.sum(level=0)


    color=[]
    contenu=[]
    part=[]
    for reg in list_eur+list_brics+list_row+list_otheroecd:
        contenu.append(M_tot[reg])
        part.append(part_imp[reg])
        if reg in list_eur:
            color.append('blue')
            continue
        if reg in list_otheroecd:
            color.append('red')
            continue
        if reg in list_brics:
            color.append('green')
            continue
        if reg in list_row:
            color.append('gray')
    ['UE','Other_OECD','BRICS','RoW']
    contenu.append(np.mean(contenu[:len(list_eur)-1]))
    contenu.append(np.mean(contenu[len(list_eur):len(list_eur)+len(list_brics)-1]))
    contenu.append(np.mean(contenu[len(list_eur)+len(list_brics):len(list_eur)+len(list_brics)+len(list_row)-1]))
    contenu.append(np.mean(contenu[len(list_eur)+len(list_brics)+len(list_row):len(list_eur)+len(list_brics)+len(list_row)+len(list_otheroecd)-1]))
    part.append(np.sum(part[:len(list_eur)-1]))
    part.append(np.sum(part[len(list_eur):len(list_eur)+len(list_brics)-1]))
    part.append(np.sum(part[len(list_eur)+len(list_brics):len(list_eur)+len(list_brics)+len(list_row)-1]))
    part.append(np.sum(part[len(list_eur)+len(list_brics)+len(list_row):len(list_eur)+len(list_brics)+len(list_row)+len(list_otheroecd)-1]))
    color.append('black')
    color.append('black')
    color.append('black')
    color.append('black')
    df_scatter = pd.DataFrame(data={'Contenu_CO2':contenu,'Part_Imp':part,'pays':list_eur+list_brics+list_row+list_otheroecd+['UE','Other_OECD','BRICS','RoW']})
    print(df_scatter)

    fig, ax = plt.subplots()
    ax.scatter(x=df_scatter['Contenu_CO2'],y=df_scatter['Part_Imp'],color=color)
    ax.set_xlabel("Contenu Carbone d'un bien importé")
    ax.set_ylabel('Part dans les importations françaises')
    for idx, row in df_scatter.iterrows():
        ax.annotate(row['pays'], (row['Contenu_CO2'], row['Part_Imp']),fontsize=5 )
    plt.savefig("figures/scatter_desag_spatiale.png")
    plt.show()

# init counterfactual(s)
counterfactual = reference.copy()
counterfactual.remove_extension('ghg_emissions_desag')

# read param sets to shock reference system
## ToDo
nbsect = len(list(reference.get_sectors()))

def get_least(sector,reloc):
	#par défaut on ne se laisse pas la possibilité de relocaliser en FR
	M = reference.ghg_emissions_desag.M.sum()
	regs = list(reference.get_regions())[1:]
	if reloc:
		regs = list(reference.get_regions())
	ind=0
	for i in range(1,len(regs)):
		if M[regs[i],sector] < M[regs[ind],sector]:
			ind=i
	return regs[ind]

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
    sectors = (nbreg-1)*list(reference.get_sectors())
    quantities = [1 for i in range(len(sectors)) ]
    moves =[]
    for i in range(nbreg):
        reg = reference.get_regions()[i]
        if reg != 'Europe':
            for j in range(len(list(reference.get_sectors()))):
                moves.append([reg,'Europe'])
    return sectors,moves,quantities

# build conterfactual(s) using param sets
## ToDo
sectors,moves,quantities = scenar_best(reloc=True,deloc=True)
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
def visualisation(scenario,scenario_name,type_emissions='D_cba',saveghg=False):
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
    print(emissions_df.sum(level=0)['Agriculture'])
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
heat_S('consommation')
heat_S('production')

# reference analysis
## ToDo
for type in ['D_cba', 'D_pba', 'D_imp', 'D_exp'] :
    #visualisation(reference,"Ref",type,saveghg=False)
    visualisation(counterfactual,"Cont",type,saveghg=False)
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