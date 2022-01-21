""" Python toolbox for MatMat trade module

    Notes
    ------
    Function calc_accounts copy/paste from subpackage pymrio.tools.iomath
    and then updated to get consumption based indicator per origin of supply.

    See: https://github.com/konstantinstadler/pymrio

    """

# general
import sys
import os

# scientific
import numpy as np
from numpy.core.fromnumeric import var
import pandas as pd
import pymrio
from pymrio.tools import ioutil
import matplotlib.pyplot as plt

class Tools:

    def extract_ghg_emissions(IOT):

        mult_to_CO2eq = {'CO2':1, 'CH4':28 , 'N2O':265 , 'SF6':23500 , 'HFC':1 , 'PFC':1}
        ghg_emissions = ['CO2', 'CH4', 'N2O', 'SF6', 'HFC', 'PFC']
        extension_list = list()

        for ghg_emission in ghg_emissions:

            ghg_name = ghg_emission.lower() + '_emissions'

            extension = pymrio.Extension(ghg_name)

            ghg_index = IOT.satellite.F.reset_index().stressor.apply(
                lambda x: x.split(' ')[0] in [ghg_emission]
            )

            for elt in ['F', 'F_Y', 'unit']:

                component = getattr(IOT.satellite, elt)

                if elt == 'unit':
                    index_name = 'index'
                else:
                    index_name = str(component.index.names[0])
                    
                component = component.reset_index().loc[ghg_index].set_index(index_name)

                if elt == 'unit':
                    component = pd.DataFrame(
                        component.values[0],
                        index = pd.Index([ghg_emission]), 
                        columns = component.columns
                    )
                else:
                    component = component.sum(axis = 0).to_frame(ghg_emission).T
                    component.loc[ghg_emission] *= mult_to_CO2eq[ghg_emission]*1e-9
                    component.index.name = index_name

                setattr(extension, elt, component)

            extension_list.append(extension)

        ghg_emissions = pymrio.concate_extension(extension_list, name = 'ghg_emissions')

        return ghg_emissions


    def calc_accounts(S, L, Y, nr_sectors):

        Y_diag = ioutil.diagonalize_blocks(Y.values, blocksize=nr_sectors)
        Y_diag = pd.DataFrame(
            Y_diag,
            index = Y.index,
            columns = Y.index
        )
        x_diag = L.dot(Y_diag)

        region_list = x_diag.index.get_level_values('region').unique()
        
        # calc carbon footprint
        D_cba = pd.concat(
            [
                S[region].dot(x_diag.loc[region])\
                for region in region_list
            ],
            axis = 0,
            keys = region_list,
            names = ['region']
        )

        # calc production based account
        x_tot = x_diag.sum(axis = 1, level = 0)
        D_pba = pd.concat(
            [
                S.mul(x_tot[region])
                for region in region_list
            ],
            axis = 0,
            keys = region_list,
            names = ['region']
        )

        # for the traded accounts set the domestic industry output to zero
        dom_block = np.zeros((nr_sectors, nr_sectors))
        x_trade = pd.DataFrame(
            ioutil.set_block(x_diag.values, dom_block),
            index = x_diag.index,
            columns = x_diag.columns
        )
        D_imp = pd.concat(
            [
                S[region].dot(x_trade.loc[region])\
                for region in region_list
            ],
            axis = 0,
            keys = region_list,
            names = ['region']
        )

        x_exp = x_trade.sum(axis = 1, level = 0)
        D_exp = pd.concat(
            [
                S.mul(x_exp[region])
                for region in region_list
            ],
            axis = 0,
            keys = region_list,
            names = ['region']
        )
    
        return (D_cba, D_pba, D_imp, D_exp)


    def recal_extensions_per_region(IOT, extension_name):

        extension = getattr(IOT, extension_name).copy()

        (
            extension.D_cba,
            extension.D_pba,
            extension.D_imp,
            extension.D_exp,
        ) = Tools.calc_accounts(
            getattr(IOT, extension_name).S, 
            IOT.L,
            IOT.Y.sum(level="region", axis=1), 
            IOT.get_sectors().size
        )

        return extension


    def shock(sector_list,Z,Y,region1,region2,sector,quantity):
        #sector : secteur concerné par la politique de baisse d'émissions importées
        #region1 : region dont on veut diminuer les émissions importées en France
        #region2 : region de report pour alimenter la demande
        #quantity : proportion dont on veut faire baisser les émissions importées pour le secteur de la région concernée.
        Z_modif=Z.copy()
        Y_modif = Y.copy()
        for sec in sector_list:
            Z_modif.loc[(region1,sector),('FR',sec)]=(1-quantity)*Z.loc[(region1,sector),('FR',sec)]
            Z_modif.loc[(region2,sector),('FR',sec)]+=quantity*Z.loc[(region1,sector),('FR',sec)]
        for demcat in list(Y.columns.get_level_values(1).unique()):
            Y_modif.loc[(region1,sector),('FR',demcat)]=(1-quantity)*Y.loc[(region1,sector),('FR',demcat)]
            Y_modif.loc[(region2,sector),('FR',demcat)]+= quantity*Y.loc[(region1,sector),('FR',demcat)]
        return Z_modif,Y_modif

    def shockv2(sector_list,demcatlist,reg_list,Z,Y,move,sector):
        Z_modif=Z.copy()
        Y_modif = Y.copy()
        if move['reloc']:
            regs = reg_list 
        else:
            regs = reg_list[1:]

        for i in range(len(sector_list)):
            for j in range(len(regs)):
                Z_modif.loc[(regs[move['sort'][j]],sector),('FR',sector_list[i])] = move['parts_sec'][move['sort'][j],i]
        for i in range(len(demcatlist)):
            for j in range(len(regs)):
                Y_modif.loc[(regs[move['sort'][j]],sector),('FR',demcatlist[i])]=move['parts_dem'][move['sort'][j],i]
        return Z_modif, Y_modif

    def shockv3(sector_list,demcatlist,reg_list,Z,Y,move,sector):
        Z_modif=Z.copy()
        Y_modif = Y.copy()
        if move['reloc']:
            regs = reg_list 
        else:
            regs = reg_list[1:]

        for j in range(len(sector_list)):
            for r in regs:
                Z_modif.loc[(r,sector),('FR',sector_list[j])] = move['parts_sec'][r][j]
        for i in range(len(demcatlist)):
            for r in regs:
                Y_modif.loc[(r,sector),('FR',demcatlist[i])]=move['parts_dem'][r][i]
        return Z_modif, Y_modif


    def get_attribute(obj, path_string):
        """allows to easily get nested attributes"""
        parts = path_string.split('.')
        final_attribute_index = len(parts)-1
        current_attribute = obj
        i = 0
        for part in parts:
            new_attr = getattr(current_attribute, part, None)
            if current_attribute is None:
                print('Error %s not found in %s' % (part, current_attribute))
                return None
            if i == final_attribute_index:
                return getattr(current_attribute, part)
            current_attribute = new_attr
            i += 1
        
    def set_attribute(obj, path_string, new_value):
        """allows to easily set nested attributes"""
        parts = path_string.split('.')
        final_attribute_index = len(parts)-1
        current_attribute = obj
        i = 0
        for part in parts:
            new_attr = getattr(current_attribute, part, None)
            if current_attribute is None:
                print('Error %s not found in %s' % (part, current_attribute))
                break
            if i == final_attribute_index:
                setattr(current_attribute, part, new_value)
            current_attribute = new_attr
            i+=1

    def reag_D_sectors(scenario,inplace = False,dict_reag_sectors=None,type='D_cba',list_reg=None):#can easily be extended to Z or Y
        """Reaggregate any account matrix with a new set of sectors"""
        #create dict for sector reaggregation for visualisation:
        if dict_reag_sectors is None :
            dict_reag_sectors={'Agriculture':['Agriculture'],
                        'Energy':['Crude coal','Crude oil','Natural gas','Fossil fuels','Electricity and heat'],
                        'Industry':['Extractive industry','Biomass_industry','Clothing','Heavy_industry',
                                    'Automobile','Oth transport equipment','Machinery','Electronics',
                                    'Construction','Transport services'],
                        'Composite':['Composite']}
        ghg_list = ['CO2', 'CH4', 'N2O', 'SF6', 'HFC', 'PFC']
        if list_reg is None :
            list_reg = scenario.get_regions()

        list_sec_new =[]
        for sec in dict_reag_sectors:
            list_sec_new.append(sec)
        

        D_mat = getattr(scenario.ghg_emissions_desag,type)

        #creating new_col and new_index for the new matrix :
        multi_reg = []
        multi_sec = []
        for reg in list_reg :
            for sec in list_sec_new :
                multi_reg.append(reg)
                multi_sec.append(sec)
        arrays = [multi_reg, multi_sec]
        new_col = pd.MultiIndex.from_arrays(arrays, names=('region', 'sector'))

        multi_reg2 = []
        multi_ghg = []
        for reg in list_reg :
            for ghg in ghg_list :
                multi_reg2.append(reg)
                multi_ghg.append(ghg)
        arrays2 = [multi_reg2, multi_ghg]
        new_index = pd.MultiIndex.from_arrays(arrays2, names=('region', 'stressor'))


        D_reag_sec = pd.DataFrame(None, index =new_index,columns = new_col)
        D_reag_sec.fillna(value=0,inplace=True)

        for reg_import in list_reg :
            for sec_agg in dict_reag_sectors:
                list_sec_agg_2 = dict_reag_sectors[sec_agg]
                for sec2 in list_sec_agg_2 :
                    D_reag_sec.loc[:,(reg_import,sec_agg)] += D_mat.loc[:,(reg_import,sec2)]
        if inplace :
            Tools.set_attribute(scenario,'ghg_emissions_desag.'+type,D_reag_sec)
            return
        else :
            return D_reag_sec



    def reag_D_regions(scenario,dict_reag_regions,inplace = False,type='D_cba',list_sec=None): #can easily be extended to Z or Y
        """Reaggregate any account matrix with a new set of region"""

        ghg_list = ['CO2', 'CH4', 'N2O', 'SF6', 'HFC', 'PFC']
        if list_sec is None :
            list_sec = scenario.get_sectors()

        list_reg_new =[]
        for reg in dict_reag_regions:
            list_reg_new.append(reg)

        D_mat = getattr(scenario.ghg_emissions_desag,type)
        #creating new_col and new_index for the new matrix :
        multi_reg = []
        multi_sec = []
        for reg in list_reg_new :
            for sec in list_sec :
                multi_reg.append(reg)
                multi_sec.append(sec)
        arrays = [multi_reg, multi_sec]
        new_col = pd.MultiIndex.from_arrays(arrays, names=('region', 'sector'))

        multi_reg2 = []
        multi_ghg = []
        for reg in list_reg_new :
            for ghg in ghg_list :
                multi_reg2.append(reg)
                multi_ghg.append(ghg)
        arrays2 = [multi_reg2, multi_ghg]
        new_index = pd.MultiIndex.from_arrays(arrays2, names=('region', 'stressor'))


        D_reag_reg = pd.DataFrame(None, index =new_index,columns = new_col)
        D_reag_reg.fillna(value=0.,inplace=True)

        for reg_export in dict_reag_regions  :
                list_reg_agg_1 = dict_reag_regions[reg_export]
                for reg_import in dict_reag_regions :
                    list_reg_agg_2 = dict_reag_regions[reg_import]
                    s1=pd.DataFrame(np.zeros_like(D_reag_reg.loc[reg_export,reg_import]),
                    index=D_reag_reg.loc[reg_export,reg_import].index, 
                    columns = D_reag_reg.loc[reg_export,reg_import].columns)
                    for reg1 in list_reg_agg_1 :
                        for reg2 in list_reg_agg_2 :
                            s1 += D_mat.loc[reg1,reg2]
                    for line in s1.index :
                        for col in s1.columns :
                            D_reag_reg.at[(reg_export,line),(reg_import,col)]=s1.loc[line,col]
        if inplace :
            Tools.set_attribute(scenario,'ghg_emissions_desag.'+type,D_reag_reg)
            return
        else :
            return D_reag_reg