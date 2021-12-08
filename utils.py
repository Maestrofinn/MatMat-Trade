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