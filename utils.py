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

    def compute_new_multi_index(ind_names,sectors_list,ghg_list,conso_sect_list,list_reg_reag_new):

        if ind_names == ('region', 'conso') :
            multi_reg = []
            multi_sec = []
            for reg in list_reg_reag_new :
                for sec in conso_sect_list :
                    multi_reg.append(reg)
                    multi_sec.append(sec)
            arrays = [multi_reg, multi_sec]
            new_index = pd.MultiIndex.from_arrays(arrays, names=('region', 'sector'))

        elif ind_names == ('region', 'sector') :
            multi_reg = []
            multi_sec = []
            for reg in list_reg_reag_new :
                for sec in sectors_list :
                    multi_reg.append(reg)
                    multi_sec.append(sec)
            arrays = [multi_reg, multi_sec]
            new_index = pd.MultiIndex.from_arrays(arrays, names=('region', 'sector'))

        elif ind_names == ('region', 'stressor') :
            multi_reg = []
            multi_ghg = []
            for reg in list_reg_reag_new :
                for ghg in ghg_list :
                    multi_reg.append(reg)
                    multi_ghg.append(ghg)
            arrays2 = [multi_reg, multi_ghg]
            new_index = pd.MultiIndex.from_arrays(arrays2, names=('region', 'stressor'))

        elif ind_names == ('stressor',) :
            new_index = pd.Index(ghg_list,name='stressor')

        elif ind_names == ('indout',) :
            new_index = pd.Index(['indout'],name='indout')

        return new_index


    def replace_reagg_scenar_attributes(scenario,reaggregation_matrix):

        ghg_list = ['CO2', 'CH4', 'N2O', 'SF6', 'HFC', 'PFC']

        conso_sect_list = ['Final consumption expenditure by households',
                        'Final consumption expenditure by non-profit organisations serving households (NPISH)',
                        'Final consumption expenditure by government',
                        'Gross fixed capital formation',
                        'Changes in inventories',
                        'Changes in valuables',
                        'Exports: Total (fob)']

        list_reg_reag_new=list(reaggregation_matrix.columns[2:])

        #create dic for region reaggregation :
        dict_reag={}
        dict_reag['FR']=['FR']
        for reg_agg in list(reaggregation_matrix.columns[3:]):
            list_reg_agg = []
            for i in reaggregation_matrix.index:
                reg = reaggregation_matrix.iloc[i].loc['Country name']
                if reaggregation_matrix[reg_agg].iloc[i] == 1:
                    list_reg_agg.append(reg)
            dict_reag[reg_agg]=list_reg_agg
        dict_reag

        to_replace_list = ['A','L','x','Y','Z','ghg_emissions.D_cba','ghg_emissions.D_pba','ghg_emissions.D_exp',
                        'ghg_emissions.D_imp','ghg_emissions.F','ghg_emissions.F_Y','ghg_emissions.M','ghg_emissions.S',
                        'ghg_emissions.S_Y','ghg_emissions_desag.D_cba','ghg_emissions_desag.D_pba','ghg_emissions_desag.D_exp',
                        'ghg_emissions_desag.D_imp','ghg_emissions_desag.F','ghg_emissions_desag.F_Y','ghg_emissions_desag.M',
                        'ghg_emissions_desag.S','ghg_emissions_desag.S_Y']

        dict_index_reag = {'A':[('region','sector'),('region','sector')],
                    'L':[('region','sector'),('region','sector')],
                    'x':[('region','sector'),('indout',)],
                    'Y':[('region','sector'),('region','conso')],
                    'Z':[('region','sector'),('region','sector')],
                    'ghg_emissions.D_cba':[('stressor',),('region','sector')],
                    'ghg_emissions.D_pba':[('stressor',),('region','sector')],
                    'ghg_emissions.D_exp':[('stressor',),('region','sector')],
                    'ghg_emissions.D_imp':[('stressor',),('region','sector')],
                    'ghg_emissions.F':[('stressor',),('region','sector')],
                    'ghg_emissions.F_Y':[('stressor',),('region','conso')],
                    'ghg_emissions.M':[('stressor',),('region','sector')],
                    'ghg_emissions.S':[('stressor',),('region','sector')],
                    'ghg_emissions.S_Y':[('stressor',),('region','conso')],
                    'ghg_emissions_desag.D_cba':[('region','stressor'),('region','sector')],
                    'ghg_emissions_desag.D_pba':[('region','stressor'),('region','sector')],
                    'ghg_emissions_desag.D_exp':[('region','stressor'),('region','sector')],
                    'ghg_emissions_desag.D_imp':[('region','stressor'),('region','sector')],
                    'ghg_emissions_desag.F':[('stressor',),('region','sector')],
                    'ghg_emissions_desag.F_Y':[('stressor',),('region','conso')],
                    'ghg_emissions_desag.M':[('stressor',),('region','sector')],
                    'ghg_emissions_desag.S':[('stressor',),('region','sector')],
                    'ghg_emissions_desag.S_Y':[('stressor',),('region','conso')]}

        dict_func_reag = {'A':'sum','L':'sum',
                    'x':'sum','Y':'sum','Z':'sum',
                    'ghg_emissions.D_cba':'sum',
                    'ghg_emissions.D_pba':'sum',
                    'ghg_emissions.D_exp':'sum',
                    'ghg_emissions.D_imp':'sum',
                    'ghg_emissions.F':'sum',
                    'ghg_emissions.F_Y':'sum',
                    'ghg_emissions.M':'mean',
                    'ghg_emissions.S':'mean',
                    'ghg_emissions.S_Y':'mean',
                    'ghg_emissions_desag.D_cba':'sum',
                    'ghg_emissions_desag.D_pba':'sum',
                    'ghg_emissions_desag.D_exp':'sum',
                    'ghg_emissions_desag.D_imp':'sum',
                    'ghg_emissions_desag.F':'sum',
                    'ghg_emissions_desag.F_Y':'sum',
                    'ghg_emissions_desag.M':'mean',
                    'ghg_emissions_desag.S':'mean',
                    'ghg_emissions_desag.S_Y':'mean'}

        for attr in to_replace_list:
            #print(attr)
            if True :
                mat = Tools.get_attribute(scenario,attr)
                
                new_ind = Tools.compute_new_multi_index(dict_index_reag[attr][0],list(scenario.get_sectors()),ghg_list,conso_sect_list,list_reg_reag_new)
                new_col = Tools.compute_new_multi_index(dict_index_reag[attr][1],list(scenario.get_sectors()),ghg_list,conso_sect_list,list_reg_reag_new)

                new_mat = pd.DataFrame(None,index = new_ind, columns = new_col)
                new_mat.fillna(value=0.,inplace=True)
                
                dict_reshape={('region','sector'):(11,17),
                    ('indout',):(1,),
                    ('region','conso'):(11,7),
                    ('region','stressor'):(11,6),
                    ('stressor',):(6,)}

                for line in np.reshape(new_ind,dict_reshape[dict_index_reag[attr][0]]) :
                    if np.shape(line)==() or np.shape(line)==(1,) :
                        elt_line = line
                    else :
                        elt_line = line[0][0]
                    if 'region' in new_ind.names :
                        list_reg_agg_1 = dict_reag[elt_line]

                        for col in np.reshape(new_col,dict_reshape[dict_index_reag[attr][1]]) :
                            if np.shape(col)==() or np.shape(col)==(1,) :
                                elt_col = col
                            else :
                                elt_col = col[0][0]

                            if 'region' in new_col.names :
                                list_reg_agg_2 = dict_reag[elt_col]
                                s1=pd.DataFrame(np.zeros_like(new_mat.loc[elt_line,elt_col]),
                                                index=new_mat.loc[elt_line,elt_col].index, 
                                                columns = new_mat.loc[elt_line,elt_col].columns, 
                                                dtype=np.float64)
                                count=0
                                for reg1 in list_reg_agg_1 :
                                    for reg2 in list_reg_agg_2 :
                                        s1 += mat.loc[reg1,reg2]
                                        count+=1
                                if dict_func_reag[attr] == 'mean':
                                    s1 = s1/count

                                for line_s1 in s1.index :
                                    for col_s1 in s1.columns :
                                        new_mat.at[(elt_line,line_s1),(elt_col,col_s1)]=s1.loc[line_s1,col_s1]

                            else :
                                s1=pd.DataFrame(np.zeros_like(new_mat.loc[elt_line]),
                                                index=new_mat.loc[elt_line].index, 
                                                columns = new_mat.loc[elt_line].columns,
                                                dtype=np.float64)
                                count=0
                                for reg1 in list_reg_agg_1 :
                                        s1 += mat.loc[reg1]
                                        count+=1
                                
                                if dict_func_reag[attr] == 'mean':
                                    s1 = s1/count

                                for line_s1 in s1.index :
                                    for col_s1 in s1.columns :
                                        new_mat.at[(elt_line,line_s1),(elt_col,col_s1)]=s1.loc[line_s1,col_s1]
                                
                                

                    elif 'region' in new_col.names:
                        for col in np.reshape(new_col,dict_reshape[dict_index_reag[attr][1]]) :

                            if np.shape(col)==() or np.shape(col)==(1,) :
                                elt_col = col
                            else :
                                elt_col = col[0][0]
                            list_reg_agg_2 = dict_reag[elt_col]
                            s1=pd.DataFrame(np.zeros_like(new_mat.loc[:,elt_col]),
                                            index=new_mat.loc[:,elt_col].index,
                                            columns = new_mat.loc[:,elt_col].columns,
                                            dtype=np.float64)
                            count=0
                            for reg2 in list_reg_agg_2 :
                                s1 += mat.loc[:,reg2]
                                count+=1

                            if dict_func_reag[attr] == 'mean':
                                s1 = s1/count

                            for line_s1 in s1.index :
                                for col_s1 in s1.columns :
                                    new_mat.at[(elt_line,line_s1),(elt_col,col_s1)]=s1.loc[line_s1,col_s1]

                    else :
                        for col in np.reshape(new_col,dict_reshape[dict_index_reag[attr][1]]) :
                            
                            elt_col = col[0][0]
                            s1=pd.DataFrame(mat.loc[elt_line,elt_col],index=new_mat.loc[elt_line,elt_col].index,
                                            columns = new_mat.loc[elt_line,elt_col].columns, dtype=np.float64)
                        for line_s1 in s1.index :
                            for col_s1 in s1.columns :
                                new_mat.at[(elt_line,line_s1),(elt_col,col_s1)]=s1.loc[line_s1,col_s1]

                                    

                Tools.set_attribute(scenario,attr,new_mat)
            else :
                Tools.set_attribute(scenario,attr,None)
            print(attr)#,np.shape(new_mat))
        #scenario.remove_extension('ghg_emissions_desag')
        #scenario.calc_all()
        #scenario.ghg_emissions_desag = Tools.recal_extensions_per_region(
        #    scenario,
        #    'ghg_emissions'
        #)
        return