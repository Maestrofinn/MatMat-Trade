import os
import numpy as np
import pandas as pd
import pathlib
import pickle as pkl
import pycountry
import pymrio
from pymrio.tools import ioutil
from pymrio.tools.iomath import calc_F_Y
from scipy.io import loadmat
from typing import Dict,List
import warnings
import wget
from unidecode import unidecode
import logging as log


from src.settings import AGGREGATION_DIR, OUTPUTS_DIR, DATA_DIR
# from src.stressors import GHG_STRESSOR_NAMES,MATERIAL_STRESSOR_NAMES,STRESSOR_DICT_GHG_MAT,GHG_AND_MATERIALS_PARAM,STRESSORS_DICT_DEF, ALL_STRESSORS
from src.stressors import GHG_STRESSOR_NAMES,MATERIAL_STRESSOR_NAMES,STRESSORS_DICT_DEF, ALL_STRESSORS


# remove pandas warning related to pymrio future deprecations
warnings.simplefilter(action="ignore", category=FutureWarning)


### AUXILIARY FUNCTION FOR DATA BUILDERS ###


def recal_stressor_per_region(
    iot: pymrio.IOSystem,
    recalc_F_Y: bool = True,
) -> pymrio.core.mriosystem.Extension:
    """Computes the account matrices D_cba, D_pba, D_imp, D_exp and optionally F_Y
       Based on pymrio.tools.iomath's function 'calc_accounts', see https://github.com/konstantinstadler/pymrio

    Args:
        iot (pymrio.IOSystem): pymrio MRIO object
        recalc_F_Y (Bool, optional) : allows to make the function recalculate F_Y as well, usefull if a change of final demand happend

    Returns:
        pymrio.core.mriosystem.Extension: extension with account matrices completed
    """
    extension = iot.stressor_extension.copy()

    S = iot.stressor_extension.S
    L = iot.L
    Y_vect = iot.Y.sum(level=0, axis=1)
    nbsectors = len(iot.get_sectors())
    extension.M=S@L

    Y_diag = ioutil.diagonalize_blocks(Y_vect.values, blocksize=nbsectors)
    Y_diag = pd.DataFrame(Y_diag, index=Y_vect.index, columns=Y_vect.index)
    x_diag = L.dot(Y_diag)

    regions = x_diag.index.get_level_values("region").unique()

    # calc footprint
    extension.D_cba = pd.concat(
        [S[reg].dot(x_diag.loc[reg]) for reg in regions],
        axis=0,
        keys=regions,
        names=["region"],
    )

    # calc production based account
    x_tot = x_diag.sum(axis=1, level=0)
    extension.D_pba = pd.concat(
        [S.mul(x_tot[reg]) for reg in regions],
        axis=0,
        keys=regions,
        names=["region"],
    )

    # for the traded accounts set the domestic industry output to zero
    dom_block = np.zeros((nbsectors, nbsectors))
    x_trade = pd.DataFrame(
        ioutil.set_block(x_diag.values, dom_block),
        index=x_diag.index,
        columns=x_diag.columns,
    )
    extension.D_imp = pd.concat(
        [S[reg].dot(x_trade.loc[reg]) for reg in regions],
        axis=0,
        keys=regions,
        names=["region"],
    )

    x_exp = x_trade.sum(axis=1, level=0)
    extension.D_exp = pd.concat(
        [S.mul(x_exp[reg]) for reg in regions],
        axis=0,
        keys=regions,
        names=["region"],
    )

    if recalc_F_Y:
        S_Y=iot.stressor_extension.S_Y
        y=iot.Y.sum()
        extension.F_Y=calc_F_Y(S_Y,y)
    
    return extension

def get_very_detailed_emissions(iot: pymrio.IOSystem,stressors_groups: Dict =None,production_diag=None) -> pd.DataFrame:
    """Computes a precise accounting of the impacts on stressors of the production, with easy traceability to both producing industry
    and the final demand for which it is being produced

    Args:
        iot (pymrio.IOSystem): pymrio MRIO object
        stressor_groups (Dict, optional) : allows to aggregate stressors to limit computation time, arg must follow the scheme :  stressors_groups={"GHG":GHG_STRESSOR_NAMES,"MATERIAL":MATERIAL_STRESSOR_NAMES},
            with GHG_STRESSOR_NAMES the list of names of the stressor in this group
        production_diag (pd.DataFrame, optional) : allows for specifying amounts of production and their destination of consumption, if the standard demand is not the variable of interest.

    Returns:
        pd.DataFrame : A detailed accounting matrix with rows for each region/sectors/stressor of production and columns for each region/goods of final demand.
    """
    if stressors_groups is None:
        S = iot.stressor_extension.S
    else:
        S=pd.concat([iot.stressor_extension.S.loc[stressors].sum() for stressors in stressors_groups.values()],axis=1,keys=stressors_groups.keys()).T
    
    if production_diag is None:
        L = iot.L
        Y_vect = iot.Y.sum(level=0, axis=1)
        nbsectors = len(iot.get_sectors())

        Y_diag = ioutil.diagonalize_blocks(Y_vect.values, blocksize=nbsectors)
        Y_diag = pd.DataFrame(Y_diag, index=Y_vect.index, columns=Y_vect.index)
        
        #for x_diag the row is the sector/region producing and the column the region/sector consumming
        x_diag = L.dot(Y_diag)
    
    else : x_diag=production_diag
    
    # DPDS is short for Detailed Production Demand Stressor matrix. It needs one value of stressor for
    # each region/sector of production responding to each region/sector of demand
    # here for each production type and location we compute for each stressor the amount of impact.  
    DPDS=pd.concat([x_diag.multiply(S.loc[stressor],axis=0) for stressor in S.index],
            keys=S.index,
            names=("stressor","region","sector")
            )
    
    
    return DPDS.reorder_levels(["region","sector","stressor"])


def get_total_imports_region(iot: pymrio.IOSystem,region:str,otherY=None, scope:int=3)-> pd.Series:
    """Computes the amount of imported goods by types of demands (intermediate and final). 
    
    Args:
        iot (pymrio.IOSystem): pymrio MRIO object
        region (str): the region for which we want to compute the average stressor coefficents

    Returns:
        pd.Series : A Series of the stressor coefficient MultiIndexed by sector of final demand and stressor names
    """ 
    
    
    if otherY is None:
        Y_vect = iot.Y.sum(axis=1,level=0)[region]
        # Y_vect = iot.Y
    else : 
        Y_vect = otherY[region]

    if scope==1:
        total_imports = iot.Z[region].sum(axis = 1) + Y_vect   # includes also all intermediate inputs used to produce exports of region 'region'
        total_imports[region]=0

    if scope==3:
        ## we isolate the imported final demand 
        imported_final=Y_vect.copy() # gets the final demand that goes to the region of interest
        imported_final[region]=0    # ignore the amount of goods that is locally produced
        
        ## we calculate the imported intermediairy inputs used to produce domestic final demand (without exports)
        A_made_in_region=iot.A.copy()
        for other_region in A_made_in_region.columns.get_level_values("region").unique().drop(region):
            A_made_in_region[other_region]=0
        A_made_in_region.loc[region, region]=0 # ignore the amount of goods that is locally produced
        imported_intermediate=(A_made_in_region @ iot.L @ Y_vect)
        
        total_imports=imported_final + imported_intermediate # for some reason the dataframe imported_intermediate has one column named 0, but in practice it is simply one vector
    
    
    return total_imports


def get_import_mean_stressor(iot: pymrio.IOSystem,region:str,otherY=None,scope:int =3)-> pd.Series:
    """Computes the average stressor impact of imported goods by types of demands (intermediate and final). 
    This corresponds to CoefRoW in MatMat.
    
    Args:
        iot (pymrio.IOSystem): pymrio MRIO object
        region (str): the region for which we want to compute the average stressor coefficents
        scope (int, optional) : the scope used to compute the mean stressor of import ie either taking the whole supply chain or only the last stop. values accepted : 1 or 3

    Returns:
        pd.Series : A Series of the stressor coefficient MultiIndexed by sector of final demand and stressor names
    """
    
    
    S = iot.stressor_extension.S
    L = iot.L
    
    total_imports=get_total_imports_region(iot,region,otherY=otherY, scope=1)
    
    # Know those totals imports are used to get a weighted average of stressor impact per industry over the different import sources
    
    S_L=S.dot(L)
    if scope==1:
        S_L=S
    import_mean_stressor=pd.concat([total_imports.mul(S_L.loc[stressor]).sum(level=1)/total_imports.sum(level=1)  for stressor in S_L.index.get_level_values(0).unique()],
                                     keys=S_L.index.get_level_values(0).unique())
    
    return import_mean_stressor.fillna(0)
    


def convert_region_from_capital_matrix(reg: str) -> str:
    """Converts a capital matrix-formatted region code into an Exiobase-formatted region code

    Args:
        reg (str): capital matrix-formatted region code

    Returns:
        str: Exiobase v3-formatted region code
    """

    try:
        return pycountry.countries.get(alpha_3=reg).alpha_2
    except AttributeError:
        if reg == "ROM":
            return "RO"  # for Roumania, UNDP country code differs from alpha_3
        elif reg in ["WWA", "WWL", "WWE", "WWF", "WWM"]:
            return reg[1:]
        else:
            raise (ValueError, f"the country code {reg} is unkown.")


def load_Kbar(year: int, system: str, path: pathlib.PosixPath) -> pd.DataFrame:
    """Loads capital consumption matrix as a Pandas dataframe (including downloading if necessary)

    Args:
        year (int): year in 4 digits
        system (str): system ('pxp', 'pxi')
        path (pathlib.PosixPath): where to save the .mat file

    Returns:
        pd.DataFrame: same formatting than pymrio's Z matrix
    """
    if not os.path.isfile(path):
        wget.download(
            f"https://zenodo.org/record/3874309/files/Kbar_exio_v3_6_{year}{system}.mat",
            str(path),
        )
    data_dict = loadmat(path)
    data_array = data_dict["KbarCfc"]
    capital_regions = [
        convert_region_from_capital_matrix(reg[0][0]) for reg in data_dict["countries"]
    ]
    capital_sectors = [sec[0][0] for sec in data_dict["prodLabels"]]
    capital_multiindex = pd.MultiIndex.from_tuples(
        [(reg, sec) for reg in capital_regions for sec in capital_sectors],
        names=["region", "sector"],
    )
    return pd.DataFrame.sparse.from_spmatrix(
        data_array, index=capital_multiindex, columns=capital_multiindex
    )


### DATA BUILDERS ###


def build_reference_data(model) -> pymrio.IOSystem:
    """Builds the pymrio object given reference's settings

    Args:
        model (Model): object Model defined in model.py

    Returns:
        pymrio.IOSystem: pymrio object
    """

    # checks if calibration is necessary
    force_calib = not os.path.isfile(model.model_dir / "file_parameters.json")

    # create directories if necessary
    for path in [model.exiobase_dir, model.model_dir, model.figures_dir]:
        if not os.path.isdir(path):
            os.mkdir(path)

    # downloading data if necessary
    if not os.path.isfile(model.exiobase_dir / model.raw_file_name):
        print("Downloading data... (may take a few minutes)")
        pymrio.download_exiobase3(
            storage_folder=model.exiobase_dir,
            system=model.system,
            years=model.base_year,
        )
        print("Data downloaded successfully !")

    if model.calib or force_calib:

        print("Loading data... (may take a few minutes)")

        # import exiobase data
        if os.path.isfile(model.exiobase_dir / model.exiobase_pickle_file_name):
            with open(model.exiobase_dir / model.exiobase_pickle_file_name, "rb") as f:
                iot = pkl.load(f)
        else:
            iot = pymrio.parse_exiobase3(  # may need RAM + SWAP ~ 15 Gb
                model.exiobase_dir / model.raw_file_name
            )
            with open(model.exiobase_dir / model.exiobase_pickle_file_name, "wb") as f:
                pkl.dump(iot, f)

        ## Pre-treatment of capital data
        if model.capital or model.recalib_capital:
            Kbar = load_Kbar(
                year=model.base_year,
                system=model.system,
                path=model.capital_consumption_path,
                )
            CCF = iot.satellite.F.loc['Operating surplus: Consumption of fixed capital']
            Kbar = CCF*(Kbar.divide(Kbar.sum(axis = 0), axis = 1))
            Kbar = Kbar.fillna(0)

        ## replace GFCF vector by pseudo-GFCF calculated from Kbar
        if model.recalib_capital:
            Kvector = Kbar.groupby(axis=1, level=0).sum()
            Kvector.columns = pd.MultiIndex.from_arrays(
                                    [
                                        Kvector.columns.tolist(),
                                        ["Gross fixed capital formation"]*len(Kvector.columns.tolist())
                                    ],
                                    names = iot.Y.columns.names
                                    )
            iot.Y.update(Kvector)
            
            # calculation of new system
            # A,S remain unchanged, Z, x, F, F_Y  are recalculated to take into account the changes on GFCF.
            iot.x = None
            iot.Z = None
            iot.L = None
            iot.satellite.F = None
            iot.satellite.F_Y = None
            iot.calc_all()
        
        
        ## endogenizes capital except for residential buildings (most of the investments made by real_estate_services sector)
        if model.capital:
            Real_estate_services = Kbar.xs('Real estate services (70)', axis = 1, level = 1, drop_level = True)
            Real_estate_services = pd.concat([Real_estate_services], axis = 1, keys = ['Gross fixed capital formation'])
            Real_estate_services = Real_estate_services.reorder_levels([1, 0], axis = 1)
            Kbar.loc[slice(None), (slice(None), "Real estate services (70)")] = 0
            
            iot.Z += Kbar
            iot.A = None
            iot.L = None
            # removes endogenized capital from final demand:
            iot.Y.loc[slice(None), (slice(None), "Gross fixed capital formation")] = 0
            iot.Y.update(Real_estate_services)
            iot.calc_all() # A is therefore recalculated with previously modified x value.
            
            # capital endogenization check
            # supply = iot.Y.sum(axis=1)+ iot.Z.sum(axis=1)
            # use = (
            #     iot.Z.sum(axis=0)
            #     + iot.satellite.F.iloc[:9].sum(axis=0)
            #     - iot.satellite.F.loc["Operating surplus: Consumption of fixed capital"]
            # )
            # print(
            #     "--- Vérification de l'équilibre emplois/ressources après endogénéisation du capital ---"
            # )
            # print(f"Le R² des vecteurs emplois/ressources est de {supply.corr(use)}.")
            # print(f"Emplois - Ressources = {use.sum() - supply.sum()}")
            # print(
            #     f"abs(Emplois - Ressources) / Emplois = {abs(use.sum() - supply.sum()) / use.sum()}"
            # )
            # print(f"max(Emplois - Ressources) = {max(use - supply)}")

        # extract emissions
        extension_list = list()

        for stressor in model.stressor_dict.keys():

            extension = pymrio.Extension(stressor)

            for elt in ["F", "F_Y", "unit"]:

                if elt == "unit":
                    
                    component = pd.DataFrame(
                        model.stressor_dict[stressor]["unit"],
                        index=pd.Index([stressor]),
                        columns=["unit"],
                    )
                else:
                    
                    component = getattr(iot.satellite, elt).loc[
                        model.stressor_dict[stressor]["exiobase_keys"]
                    ]
    
                    component = (
                        component.sum(axis=0).to_frame(stressor).T
                        * model.stressor_dict[stressor]["weight"]
                    )

                setattr(extension, elt, component)

            extension_list.append(extension)

        iot.stressor_extension = pymrio.concate_extension(
            extension_list, name="stressors"
        )

        # del useless extensions
        iot.remove_extension(["satellite", "impacts"])

        # import aggregation matrices
        agg_matrix = {
            axis: pd.read_excel(
                AGGREGATION_DIR / f"{model.aggregation_name}.xlsx", sheet_name=axis
            )
            for axis in ["sectors", "regions"]
        }
        agg_matrix["sectors"].set_index(
            ["category", "sub_category", "sector"], inplace=True
        )
        agg_matrix["regions"].set_index(["Country name", "Country code"], inplace=True)

        # apply regional and sectorial agregations
        iot.aggregate(
            region_agg=agg_matrix["regions"].T.values,
            sector_agg=agg_matrix["sectors"].T.values,
            region_names=agg_matrix["regions"].columns.tolist(),
            sector_names=agg_matrix["sectors"].columns.tolist(),
        )

        # reset A, L, S, S_Y, M and all of the account matrices
        iot = iot.reset_to_flows()
        iot.stressor_extension.reset_to_flows()

        # compute missing matrices
        iot.calc_all()

        # compute emission accounts by region
        iot.stressor_extension = recal_stressor_per_region(iot=iot)

        # save model
        iot.save_all(model.model_dir)

        print("Data loaded successfully !")

    else:

        # import calibration data previously built with calib = True
        iot = pymrio.parse_exiobase3(model.model_dir)

    return iot


def build_counterfactual_data(
    model,
    scenar_function,
    **kwargs
) -> pymrio.IOSystem:
    """Builds the pymrio object given reference's settings and the scenario parameters

    Args:
        model (Model): object Model defined in model.py
        scenar_function (Callable[[Model, bool], Tuple[pd.DataFrame]]): builds the new Z and Y matrices
        **kwargs used :
            reloc (bool, optional): True if relocation is allowed. Defaults to False.
            year (int,optional) : Year for which the scenario is created. 
    Returns:
        pymrio.IOSystem: modified pymrio model
    """

    iot=scenar_function(model=model,**kwargs)
    
    iot.calc_all()

    iot.stressor_extension = recal_stressor_per_region(
        iot=iot,
    )

    return iot


### AGGREGATORS ###


def reverse_mapper(mapper: Dict) -> Dict:
    """Reverse a mapping dictionary

    Args:
        mapper (Dict): dictionnary with new categories as keys and old ones as values, no aggregation if is None. Defaults to None.

    Returns:
        Dict: dictionnary with old categories as keys and new ones as values, no aggregation if is None.
    """

    if mapper is None:
        return None

    new_mapper = {}
    for key, value in mapper.items():
        for elt in value:
            new_mapper[elt] = key

    return new_mapper


def aggregate_avg_simple_index(
    df: pd.DataFrame,
    axis: int,
    new_index: pd.Index,
    reverse_mapper: Dict = None,
) -> pd.DataFrame:
    """Aggregates data along given axis according to mapper.
    WARNING: the aggregation is based on means, so it works only with intensive data.

    Args:
        df (pd.DataFrame): multiindexed DataFrame
        axis (int): axis of aggregation (0 or 1)
        new_index (pd.Index): aggregated index (useful to set the right order)
        reverse_mapper (Dict, optional): dictionnary with old categories as keys and new ones as values, no aggregation if is None. Defaults to None.
    Returns:
        pd.DataFrame: aggregated DataFrame
    """
    if reverse_mapper is None:
        return df
    if axis == 1:
        df = df.T
    df = df.groupby(df.index.map(lambda x: reverse_mapper[x])).mean().reindex(new_index)
    if axis == 1:
        df = df.T
    return df


def aggregate_sum(
    df: pd.DataFrame,
    level: int,
    axis: int,
    new_index: pd.Index,
    reverse_mapper: Dict = None,
) -> pd.DataFrame:
    """Aggregates data at given level along given axis according to mapper.
    WARNING: the aggregation is based on sums, so it works only with additive data.

    Args:
        df (pd.DataFrame): multiindexed DataFrame
        level (int): level of aggregation on multiindex (0 or 1)
        axis (int): axis of aggregation (0 or 1)
        new_index (pd.Index): aggregated index (useful to set the right order)
        reverse_mapper (Dict, optional): dictionnary with old categories as keys and new ones as values, no aggregation if is None. Defaults to None.
    Returns:
        pd.DataFrame: aggregated DataFrame
    """
    if reverse_mapper is None:
        return df
    if axis == 1:
        df = df.T
    if level == 0:
        index_level_1 = df.index.get_level_values(1).drop_duplicates()
        df = (
            df.groupby(
                [
                    df.index.get_level_values(0).map(lambda x: reverse_mapper[x]),
                    df.index.get_level_values(1),
                ]
            )
            .sum()
            .reindex(new_index, level=0)
            .reindex(index_level_1, level=1)
        )
    else:  # level = 1
        index_level_0 = df.index.get_level_values(0).drop_duplicates()
        df = (
            df.groupby(
                [
                    df.index.get_level_values(0),
                    df.index.get_level_values(1).map(lambda x: reverse_mapper[x]),
                ]
            )
            .sum()
            .reindex(new_index, level=1)
            .reindex(index_level_0, level=0)
        )
    if axis == 1:
        df = df.T
    return df


def aggregate_sum_axis(
    df: pd.DataFrame,
    axis: int,
    new_index_0: pd.Index,
    new_index_1: pd.Index,
    reverse_mapper_0: Dict = None,
    reverse_mapper_1: Dict = None,
) -> pd.DataFrame:
    """Aggregates data at all levels (0 and 1) along given axis according to both mappers.
    WARNING: the aggregation is based on sums, so it works only with additive data.

    Args:
        df (pd.DataFrame): multiindexed DataFrame
        axis (int): axis of aggregation (0 or 1)
        new_index_0 (pd.Index): aggregated index at level 0 (useful to set the right order)
        new_index_1 (pd.Index): aggregated index at level 1 (useful to set the right order)
        reverse_mapper_0 (Dict, optional): dictionnary with old categories as keys and new ones as values at level 0, no aggregation if is None. Defaults to None.
        reverse_mapper_1 (Dict, optional): dictionnary with old categories as keys and new ones as values at level 1, no aggregation if is None. Defaults to None.

    Returns:
        pd.DataFrame: aggregated DataFrame
    """

    return aggregate_sum(
        df=aggregate_sum(
            df=df,
            level=0,
            axis=axis,
            new_index=new_index_0,
            reverse_mapper=reverse_mapper_0,
        ),
        level=1,
        axis=axis,
        new_index=new_index_1,
        reverse_mapper=reverse_mapper_1,
    )


def aggregate_sum_2levels_2axes(
    df: pd.DataFrame,
    new_index_0: pd.Index,
    new_index_1: pd.Index,
    reverse_mapper_0: Dict = None,
    reverse_mapper_1: Dict = None,
) -> pd.DataFrame:
    """Aggregates data at all levels (0 and 1) along all axes (0 and 1) according to both mappers, with a DataFrame whose MultiIndex is identical along both axes.
    WARNING: the aggregation is based on sums, so it works only with additive data.

    Args:
        df (pd.DataFrame): multiindexed DataFrame
        new_index_0 (pd.Index): aggregated index at level 0 (useful to set the right order)
        new_index_1 (pd.Index): aggregated index at level 1 (useful to set the right order)
        reverse_mapper_0 (Dict, optional): dictionnary with old categories as keys and new ones as values at level 0, no aggregation if is None. Defaults to None.
        reverse_mapper_1 (Dict, optional): dictionnary with old categories as keys and new ones as values at level 1, no aggregation if is None. Defaults to None.

    Returns:
        pd.DataFrame: aggregated DataFrame
    """

    return aggregate_sum_axis(
        df=aggregate_sum_axis(
            df=df,
            axis=0,
            new_index_0=new_index_0,
            new_index_1=new_index_1,
            reverse_mapper_0=reverse_mapper_0,
            reverse_mapper_1=reverse_mapper_1,
        ),
        axis=1,
        new_index_0=new_index_0,
        new_index_1=new_index_1,
        reverse_mapper_0=reverse_mapper_0,
        reverse_mapper_1=reverse_mapper_1,
    )


def aggregate_sum_2levels_on_axis1_level0_on_axis0(
    df: pd.DataFrame,
    new_index_0: pd.Index,
    new_index_1: pd.Index,
    reverse_mapper_0: Dict = None,
    reverse_mapper_1: Dict = None,
) -> pd.DataFrame:
    """Aggregates data at all levels (0 and 1) along axis 1 and at level 0 only along axis 0 according to both mappers.
    WARNING: the aggregation is based on sums, so it works only with additive data.

    Args:
        df (pd.DataFrame): multiindexed DataFrame
        new_index_0 (pd.Index): aggregated index at level 0 (useful to set the right order)
        new_index_1 (pd.Index): aggregated index at level 1 (useful to set the right order)
        reverse_mapper_0 (Dict, optional): dictionnary with old categories as keys and new ones as values at level 0, no aggregation if is None. Defaults to None.
        reverse_mapper_1 (Dict, optional): dictionnary with old categories as keys and new ones as values at level 1, no aggregation if is None. Defaults to None.

    Returns:
        pd.DataFrame: aggregated DataFrame
    """

    return aggregate_sum(
        df=aggregate_sum_2levels_2axes(
            df=df,
            new_index_0=new_index_0,
            new_index_1=None,
            reverse_mapper_0=reverse_mapper_0,
            reverse_mapper_1=None,
        ),
        level=1,
        axis=1,
        new_index=new_index_1,
        reverse_mapper=reverse_mapper_1,
    )


def aggregate_sum_level0_on_axis1_2levels_on_axis0(
    df: pd.DataFrame,
    new_index_0: pd.Index,
    new_index_1: pd.Index,
    reverse_mapper_0: Dict = None,
    reverse_mapper_1: Dict = None,
) -> pd.DataFrame:
    """Aggregates data at level 0 along axis 1 and at all levels (0 and 1) along axis 0 according to both mappers.
    WARNING: the aggregation is based on sums, so it works only with additive data.

    Args:
        df (pd.DataFrame): multiindexed DataFrame
        new_index_0 (pd.Index): aggregated index at level 0 (useful to set the right order)
        new_index_1 (pd.Index): aggregated index at level 1 (useful to set the right order)
        reverse_mapper_0 (Dict, optional): dictionnary with old categories as keys and new ones as values at level 0, no aggregation if is None. Defaults to None.
        reverse_mapper_1 (Dict, optional): dictionnary with old categories as keys and new ones as values at level 1, no aggregation if is None. Defaults to None.

    Returns:
        pd.DataFrame: aggregated DataFrame
    """

    return aggregate_sum(
        df=aggregate_sum_2levels_2axes(
            df=df,
            new_index_0=new_index_0,
            new_index_1=None,
            reverse_mapper_0=reverse_mapper_0,
            reverse_mapper_1=None,
        ),
        level=1,
        axis=0,
        new_index=new_index_1,
        reverse_mapper=reverse_mapper_1,
    )


### FEATURE EXTRACTORS ###


def footprint_extractor(model, region: str = "FR",stressor_list=GHG_STRESSOR_NAMES) -> Dict:
    """Computes region's footprint (D_pba-D_exp+D_imp+F_Y)

    Args:
        model (Union[Model, Counterfactual]): object Model or Counterfactual defined in model.py
        region (str, optional): region name. Defaults to "FR".
        stressor_list (list(str),optional) : names of the stressor to sum, default to the GHG stressor list.

    Returns:
        Dict: values of -D_exp, D_pba, D_imp and F_Y
    """
    stressor_extension = model.iot.stressor_extension
    return {
        "Exportations": -stressor_extension.D_exp[region].loc[(slice(None),stressor_list),:].sum().sum(),
        "Production": stressor_extension.D_pba[region].loc[(slice(None),stressor_list),:].sum().sum(),
        "Importations": stressor_extension.D_imp[region].loc[(slice(None),stressor_list),:].sum().sum(),
        "Consommation": stressor_extension.F_Y[region].loc[stressor_list].sum().sum(),
    }

def multi_footprints_extractor(iot: pymrio.IOSystem,region : str = "FR", stressor_lists : Dict ={"GHG":GHG_STRESSOR_NAMES,"MATERIAL":MATERIAL_STRESSOR_NAMES}):
    """Computes region's footprint indicators : D_imp, D_cba, D_pba, D_exp, Footprint for several groups of stressors. 

    Args:
        model (Union[Model, Counterfactual]): object Model or Counterfactual defined in model.py
        region (str, optional): region name. Defaults to "FR".
        stressor_lists (Dict[str:List[str]],optional) : list of the groups of the stressor to sum, default to the {"GHG":GHG_STRESSOR_NAMES,"MATERIAL":MATERIAL_STRESSOR_NAMES}.

    Returns:
        pandas Dataframe with axis ["D_imp_MatMat","D_imp_pymrio","Footprint","D_cba","D_pba","D_exp"]) and columns the stressor_lists groups
    """
    
    Footprints=pd.DataFrame(index=["D_imp_MatMat","D_imp_pymrio","Footprint","D_cba","D_pba","D_exp"])# initialte empty datarframe
    total_imports=get_total_imports_region(iot,region)
    mean_stressors=get_import_mean_stressor(iot,region)
                                   
    for name,stressor_list in stressor_lists.items() :
        D_imp_MatMat=(mean_stressors.loc[stressor_list].sum(level=1)*total_imports.groupby("sector").sum()).sum()
        D_imp_pymrio=iot.stressor_extension.D_imp[region].loc[(slice(None),stressor_list),:].sum().sum()
        Footprint=iot.stressor_extension.D_cba[region].loc[(slice(None),stressor_list),:].sum().sum()+iot.stressor_extension.F_Y[region].sum().sum()
        D_cba=iot.stressor_extension.D_cba[region].loc[(slice(None),stressor_list),:].sum().sum()
        D_pba=iot.stressor_extension.D_pba[region].loc[(slice(None),stressor_list),:].sum().sum()
        D_exp=iot.stressor_extension.D_exp[region].loc[(slice(None),stressor_list),:].sum().sum()
        Footprints[name]=[D_imp_MatMat,D_imp_pymrio,Footprint,D_cba,D_pba,D_exp]
        
    return Footprints

def save_footprints(model,path:str,region : str, stressor_lists : Dict):
    Footprints=multi_footprints_extractor(model.iot,region=region, stressor_lists=stressor_lists)
    
    with pd.ExcelWriter(path) as writer:
        Footprints.to_excel(writer,sheet_name="base_year")
        for counterfactual in model.get_counterfactuals_list():
            Footprints=multi_footprints_extractor(model.counterfactuals[counterfactual].iot,region=region, stressor_lists=stressor_lists)
            Footprints.to_excel(writer,sheet_name=counterfactual)
        pd.DataFrame.from_dict(stressor_lists,orient='index').T.to_excel(writer,sheet_name="Stressor_groups",index=False)
        
        
def save_CoefRoW(model,
                region : str, #region to compute the CoefRoW's
                stressors_list : list, # list of the calibrated stressors to save
                SRIO_filename :str, # name of the file where MatMat data are stored
                # stressors_list : Dict ={"GHG":GHG_STRESSOR_NAMES,"MATERIAL":MATERIAL_STRESSOR_NAMES}
                ):
    
    # import MatMat SRIO data
    SRIO_hyb=pd.read_excel(DATA_DIR/("MatMat/{}.xlsx".format(SRIO_filename)),sheet_name="hybride",header=[0,1,2],index_col=[0,1,2,3])
    SRIO_monetary=pd.read_excel(DATA_DIR/("MatMat/{}.xlsx".format(SRIO_filename)),sheet_name="monetaire",header=[0,1,2],index_col=[0,1,2,3])
    
    # isolating the gross outputs
    X_hyb=SRIO_hyb.loc[:,(slice(None),slice(None),"x")]
    X_monetary=SRIO_monetary.loc[:,(slice(None),slice(None),"x")]

    # compute conversation factors monetary <-> hybrid
    conversion_hybrid_monetary_RoW=(X_monetary/X_hyb).fillna(0).loc["RoW"]
    conversion_hybrid_monetary_RoW=conversion_hybrid_monetary_RoW.reset_index(["category","sub_category"])[("Supply","Supply","x")].fillna(0)


    # compute CoefRoW for model and its all couterfactuals, in absolute and relative variations
    CoefRoW = dict()
    for stressors_category in stressors_list:
        
        stressors = STRESSORS_DICT_DEF[stressors_category]['dict']
        stressors_short_name = STRESSORS_DICT_DEF[stressors_category]['name_EN']
        
        CoefRoW[stressors_short_name] = pd.DataFrame()
        
        CoefRoW[stressors_short_name]['base_year'] = get_import_mean_stressor(model.iot,region)\
                .loc[stressors].groupby("sector").sum().reindex(model.agg_sectors)
        for counterfactual in model.get_counterfactuals_list():
            CoefRoW[stressors_short_name][counterfactual] = get_import_mean_stressor(model.counterfactuals[counterfactual].iot,region)\
                    .loc[stressors].groupby("sector").sum().reindex(model.agg_sectors)
        
        CoefRoW[stressors_short_name + '_relat'] = (CoefRoW[stressors_short_name].divide(CoefRoW[stressors_short_name]['base_year'], axis = 0)-1).fillna(0)
        
        # CoefRoW[stressors_short_name]['unit'] = find_stressor_unit(stressors_category)+'/M€'
        
        CoefRoW[stressors_short_name] = CoefRoW[stressors_short_name].mul(conversion_hybrid_monetary_RoW, axis = 0)\
                                        /(1E3 if find_stressor_unit(stressors_category) == 'kgCO2eq' else 1)
   
    # CoefRoW['unit'] = pd.DataFrame(['(tCO2eq or kt) / MatMat\'s sector unit (kt or ktoe or M€)'])

    CoefRoW['unit'] = pd.DataFrame(
        [find_stressor_unit(stressors_category) for stressors_category in stressors_list],
        index = [stressors_list],
        columns = ['unit']
        ).where(lambda x: x!='kgCO2eq', 'tCO2eq')
    
    
    #save CoefRoW
    path = OUTPUTS_DIR/'CoefRoW'
    path.mkdir(exist_ok = True)
    path = path/('CoefRoW_'+model.summary_long+'.xlsx')
    
    # with pd.ExcelWriter(path/("CoefRow_{}.xlsx".format(model.summary_long))) as writer:
    with pd.ExcelWriter(path) as writer:
        for key in CoefRoW.keys():
            CoefRoW[key].to_excel(writer,sheet_name=key)

    return None #Coef_Row



### AUXILIARY FUNCTIONS FOR FIGURES EDITING AND DATA SAVING ###


def build_description(model, counterfactual_name: str = None) -> str:
    """Builds a descriptions of the parameters used to edit a figure

    Args:
        model (Model): object Model defined in model.py
        counterfactual_name (str, optional): name of the counterfactual in model.counterfactuals. None for the reference, False for multiscenario figures. Defaults to None.

    Returns:
        str: description
    """
    if counterfactual_name is None:
        output = "Scénario de référence\n"
    elif not counterfactual_name:
        output = ""
    else:
        output = f"Scénario : {counterfactual_name}\n"
        output += f"Scénarisation sur: {model.counterfactuals[counterfactual_name].scenar_stressors}\n"
    output += f"Année : {model.base_year}\n"
    output += f"Système : {model.system}\n"
    output += f"Base de données : Exiobase {model.iot.meta.version}\n"
    if model.capital:
        output += "Modèle à capital endogène"
    else:
        output += "Modèle sans capital endogène"
    if counterfactual_name:
        if model.counterfactuals[counterfactual_name].reloc:
            output += "\nScénario avec relocalisation"
        else:
            output += "\nScénario sans relocalisation"
    return output



def shortnaming(stressor_params:str) -> str:
    
    stressor_shortname = unidecode(''.join(elt[:4] +'_' for elt in stressor_params.split(' '))[:-1])
    
    return stressor_shortname


def subsetting(stressor_subset:str) -> set:
    
    PARAMS_SUBSET = {elt: ALL_STRESSORS[elt] for elt in STRESSORS_DICT_DEF[stressor_subset]['dict']}
    
    return PARAMS_SUBSET


def check_coherent_units(stressors_to_display, stressor_unit):
    
    for elt in STRESSORS_DICT_DEF[stressors_to_display]['dict']:
        if ALL_STRESSORS[elt]['unit'] != stressor_unit:
            log.warning(f'Unit of {elt} stressor is not equal to the aggregated unit given for {stressors_to_display} in STRESSORS_DICT_DEF variable.')

    return None


def find_stressor_unit(stressors_to_display:str) -> str:
    
    # stressor_unit = GHG_AND_MATERIALS_PARAM["proxy"][STRESSOR_DICT_GHG_MAT[stressors_to_display][0]]["unit"]
    stressor_unit = STRESSORS_DICT_DEF[stressors_to_display]["unit"]
    check_coherent_units(stressors_to_display, stressor_unit)

    return stressor_unit





from typing import Union

def diagonalize_columns_to_sectors(
    df: pd.DataFrame, sector_index_level: Union[str, int] = "sector"
) -> pd.DataFrame:
    """Adds the resolution of the rows to columns by diagonalizing
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to diagonalize
    sector_index_name : string, optional
        Name or number of the index level containing sectors.
    Returns
    -------
    pd.DataFrame, diagonalized
    Example
    --------
        input       output
         (all letters are index or header)
            A B     A A A B B B
                    x y z x y z
        A x 3 1     3 0 0 1 0 0
        A y 4 2     0 4 0 0 2 0
        A z 5 3     0 0 5 0 0 3
        B x 6 9     6 0 0 9 0 0
        B y 7 6     0 7 0 0 6 0
        B z 8 4     0 0 8 0 0 4
    """

    sectors = df.index.get_level_values(sector_index_level).unique()
    sector_name = sector_index_level if type(sector_index_level) is str else "sector"

    new_col_index = [
        tuple(list(orig) + [new]) for orig in df.columns for new in sectors
    ]

    diag_df = pd.DataFrame(
        data=ioutil.diagonalize_blocks(df.values, blocksize=len(sectors)),
        index=df.index,
        columns=pd.MultiIndex.from_product(
            [df.columns, sectors], names=[*df.columns.names, sector_name]
        ),
    )
    return diag_df