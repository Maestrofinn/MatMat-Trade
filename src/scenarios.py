import numpy as np
import pandas as pd
import pymrio
import copy
from typing import Callable, Dict, List, Tuple

from src.utils import recal_stressor_per_region
from src.advance import extract_data
from src.model import Model


### AUXILIARY FUNCTIONS FOR SCENARIOS ###


def moves_from_sorted_index_by_sector(
    model, sector: str, regions_index: List[int], reloc: bool = False
) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """Allocates french importations for a sector in the order given by region_index

    Args:
        model (Model): object Model defined in model.py
        sector (str): name of a product (or industry)
        regions_index (List[int]): list of ordered region indices
        reloc (bool): True if relocation is allowed. Defaults to None.

    Returns:
        Tuple[pd.DataFrame]: tuple with 2 elements :
            - DataFrame with the imports of 'sector' from regions (rows) for french intermediary sectors (columns)
            - DataFrame with the imports of 'sector' from regions (rows) for french final demands (columns)
    """

    if reloc:
        regions = model.regions
    else:
        regions = model.regions[1:] # remove FR
    Z = model.iot.Z   
    Y = model.iot.Y   
     
    # french total importations demand for each sector / final demand
    inter_imports = Z["FR"].drop("FR", level=0).groupby(level=1).sum().loc[sector]
    final_imports = Y["FR"].drop("FR", level=0).groupby(level=1).sum().loc[sector]
    total_imports = inter_imports.sum() + final_imports.sum()
    inter_autoconso_FR = (
        Z.loc[("FR", sector), ("FR", slice(None))].groupby(level=1).sum()
    )
    final_autoconso_FR = (
        Y.loc[("FR", sector), ("FR", slice(None))].groupby(level=1).sum()
    )

    # export capacities of each regions
    export_capacities = {
        reg: Z.drop(columns=reg, level=0).sum(axis=1).loc[(reg, sector)]*0/100
        + Y.drop(columns=reg, level=0).sum(axis=1).loc[(reg, sector)]*0/100
        + Z["FR"].sum(axis=1).loc[(reg, sector)]*120/100
        + Y["FR"].sum(axis=1).loc[(reg, sector)]*120/100
        for reg in regions
    }

    # choice of the trade partners
    imports_from_regions = pd.Series(0, index=model.regions)
    remaining_imports = total_imports
    for i in regions_index:
        reg = regions[i]
        if export_capacities[reg] <= remaining_imports:
            imports_from_regions.loc[reg] = export_capacities[reg]
            remaining_imports -= export_capacities[reg]
        else:
            imports_from_regions.loc[reg] = remaining_imports
            break 

    # allocations for intermediary imports
    new_inter_imports = imports_from_regions.to_frame("").dot(
        (inter_imports / total_imports).to_frame("").T
    )
    new_inter_imports.loc["FR"] += inter_autoconso_FR

    # allocations for final imports
    new_final_imports = imports_from_regions.to_frame("").dot(
        (final_imports / total_imports).to_frame("").T
    )
    new_final_imports.loc["FR"] += final_autoconso_FR
 
    return new_inter_imports, new_final_imports


def moves_from_sort_rule(
    model,
    sorting_rule_by_sector: Callable[[Model,str, bool], List[int]],
    reloc: bool = False,
) -> pymrio.IOSystem:
    """Allocates french importations for all sectors, sorting the regions with a given rule for each sector

    Args:
        model (Model): object Model defined in model.py
        sorting_rule_by_sector (Callable[[str, bool], List[int]]): given a sector name and the reloc value, returns a sorted list of regions' indices
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        pymrio.IOSystem: modified pymrio model
    """

    sectors_list = model.sectors
    new_Z = model.iot.Z.copy()
    new_Y = model.iot.Y.copy()
    new_Z["FR"] = new_Z["FR"] * 0
    new_Y["FR"] = new_Y["FR"] * 0
    

    for sector in sectors_list:
        regions_index = sorting_rule_by_sector(model, sector, reloc)
        new_inter_imports, new_final_imports = moves_from_sorted_index_by_sector(
            model=model, sector=sector, regions_index=regions_index, reloc=reloc
        )
        new_Z.loc[(slice(None), sector), ("FR", slice(None))] = new_inter_imports.values
        new_Y.loc[(slice(None), sector), ("FR", slice(None))] = new_final_imports.values
    
    # integrate into mrio model 
    iot=model.iot.copy()
    iot.reset_to_flows()
    iot.L=None
    iot.A=None
    iot.x=None
    iot.Z=new_Z
    iot.Y=new_Y
    iot.calc_all()
    
    return iot


def sort_by_content(model, sector: str, reloc: bool = False) -> List[int]:
    """Ascendantly sorts all regions by stressor content of a sector

    Args:
        model (Model): object Model defined in model.py
        sector (str): name of a product (or industry)
        reloc (bool, optional): True if relocation is allowed. Defaults to False.


    Returns:
        np.array: array of indices of regions sorted by stressor content
    """

    M = model.iot.stressor_extension.M.sum(axis=0)
    regions_index = np.argsort(M[:, sector].values[1 - reloc :])
    return regions_index # type: ignore


### BEST AND WORST SCENARIOS ###


def scenar_best(model: Model, reloc: bool = False) -> pymrio.IOSystem:
    """Finds the least stressor-intense imports reallocation for all sectors


    Args:
        model (Model): object Model defined in model.py
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Tuple[pd.DataFrame]: tuple with 2 elements :
            - reallocated Z matrix
            - reallocated Y matrix
    """

    return moves_from_sort_rule(
        model=model, sorting_rule_by_sector=sort_by_content, reloc=reloc
    )


def scenar_worst(model: Model, reloc: bool = False) -> pymrio.IOSystem:
    """Finds the most stressor-intense imports reallocation for all sectors


    Args:
        model (Model): object Model defined in model.py
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Tuple[pd.DataFrame]: tuple with 2 elements :
            - reallocated Z matrix
            - reallocated Y matrix
    """

    return moves_from_sort_rule(
        model=model,
        sorting_rule_by_sector=lambda *args: sort_by_content(*args)[::-1],
        reloc=reloc
    )


### PREFERENCE SCENARIOS ###


def scenar_pref(model, allies: List[str], reloc: bool = False) -> pymrio.IOSystem:
    """Finds imports reallocation in order to trade as much as possible with the allies

    Args:
        model (Model): object Model defined in model.py
        allies (List[str]): list of regions' names
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Tuple[pd.DataFrame]: tuple with 2 elements :
            - reallocated Z matrix
            - reallocated Y matrix
    """

    if reloc:
        regions = model.regions
        if not "FR" in allies:
            allies += ["FR"]
    else:
        regions = model.regions[1:]  # remove FR

    new_Z = model.iot.Z.copy()
    new_Y = model.iot.Y.copy()
    new_Z["FR"] = new_Z["FR"] * 0
    new_Y["FR"] = new_Y["FR"] * 0
  
    



    sectors = model.sectors

    for sector in sectors:

        ## overall trade related with sector
        sector_exports_Z = model.iot.Z.loc[(regions, sector), :].sum(axis=0, level=0)
        sector_exports_Y = model.iot.Y.loc[(regions, sector), :].sum(axis=0, level=0)
        sector_exports_by_region = sector_exports_Z.sum(
            axis=1, level=0
        ) + sector_exports_Y.sum(axis=1, level=0)

        ## french importations
        sector_imports_FR_Z = sector_exports_Z["FR"]
        sector_imports_FR_Y = sector_exports_Y["FR"]
        sector_imports_FR_nonallies_Z = sector_imports_FR_Z.drop(allies).sum()
        sector_imports_FR_nonallies_Y = sector_imports_FR_Y.drop(allies).sum()
        total_imports_FR_nonallies = (
            sector_imports_FR_nonallies_Z.sum() + sector_imports_FR_nonallies_Y.sum()
        )

        ## allies' exportation capacities
        allies_export_except_FR = (
            sector_exports_by_region.loc[allies].drop(columns=["FR"]).sum(axis=1)
        )
        allies_autoexport = pd.Series(
            np.diagonal(sector_exports_by_region.loc[allies, allies]),
            index=allies,
        )
        allies_export_capacity = allies_export_except_FR.add(
            -allies_autoexport.drop("FR", errors="ignore"), fill_value=0
        )  # .add() handles the possible values of reloc
        allies_total_export_capacity = allies_export_capacity.sum()

        ## specific case leading to a division by 0
        if total_imports_FR_nonallies == 0 and allies_total_export_capacity == 0:
            for reg in regions:
                new_Z.loc[(reg, sector), ("FR", slice(None))] = sector_imports_FR_Z.loc[
                    reg
                ].values
                new_Y.loc[(reg, sector), ("FR", slice(None))] = sector_imports_FR_Y.loc[
                    reg
                ].values
            continue

        ## reallocations
        if total_imports_FR_nonallies < allies_total_export_capacity:

            coef_Z = (
                sector_imports_FR_nonallies_Z / allies_total_export_capacity
            )  # alpha_s for s in Z
            coef_Y = (
                sector_imports_FR_nonallies_Y / allies_total_export_capacity
            )  # alpha_s for s in Y

            for reg in regions:
                if reg in allies:
                    new_Z.loc[(reg, sector), ("FR", slice(None))] = (
                        sector_imports_FR_Z.loc[reg]
                        + coef_Z * allies_export_capacity[reg]
                    ).values
                    new_Y.loc[(reg, sector), ("FR", slice(None))] = (
                        sector_imports_FR_Y.loc[reg]
                        + coef_Y * allies_export_capacity[reg]
                    ).values
        else:
            coef_allies_Z = 1 + (
                allies_export_capacity.to_frame().dot(
                    sector_imports_FR_nonallies_Z.to_frame().T
                )
                / (sector_imports_FR_Z.loc[allies] * total_imports_FR_nonallies)
            ).replace([np.inf, -np.inf], np.nan).fillna(
                0
            )  # beta_r,s for Z
            coef_allies_Y = 1 + (
                allies_export_capacity.to_frame().dot(
                    sector_imports_FR_nonallies_Y.to_frame()
                    .replace([np.inf, -np.inf], np.nan)
                    .T
                )
                / (sector_imports_FR_Y.loc[allies] * total_imports_FR_nonallies)
            ).replace([np.inf, -np.inf], np.nan).fillna(
                0
            )  # beta_r,s for Y
            coef_nonallies = (
                1 - allies_total_export_capacity / total_imports_FR_nonallies
            )  # gamma

            for reg in regions:
                if reg not in allies:
                    new_Z.loc[(reg, sector), ("FR", slice(None))] = (
                        coef_nonallies * sector_imports_FR_Z.loc[reg]
                    ).values
                    new_Y.loc[(reg, sector), ("FR", slice(None))] = (
                        coef_nonallies * sector_imports_FR_Y.loc[reg]
                    ).values
                else:
                    new_Z.loc[(reg, sector), ("FR", slice(None))] = (
                        coef_allies_Z.loc[reg] * sector_imports_FR_Z.loc[reg]
                    ).values
                    new_Y.loc[(reg, sector), ("FR", slice(None))] = (
                        coef_allies_Y.loc[reg] * sector_imports_FR_Y.loc[reg]
                    ).values

    ## process autoproduction
    new_Z.loc[("FR", slice(None)), ("FR", slice(None))] += model.iot.Z.loc[
        ("FR", slice(None)), ("FR", slice(None))
    ].values
    new_Y.loc[("FR", slice(None)), ("FR", slice(None))] += model.iot.Y.loc[
        ("FR", slice(None)), ("FR", slice(None))
    ].values
    
    # integrate into mrio model 
    iot=model.iot.copy()
    iot.reset_to_flows()
    iot.L=None
    iot.A=None
    iot.x=None
    iot.Z=new_Z
    iot.Y=new_Y
    iot.calc_all()
    
    return iot


def scenar_pref_eu(model: Model, reloc: bool = False) -> pymrio.IOSystem:
    """Finds imports reallocation that prioritize trade with European Union

    Args:
        model (Model): object Model defined in model.py
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Tuple[pd.DataFrame]: tuple with 2 elements :
            - reallocated Z matrix
            - reallocated Y matrix
    """

    return scenar_pref(model=model, allies=["EU"], reloc=reloc)


### TRADE WAR SCENARIOS ###


def scenar_tradewar(model: Model, opponents: List[str], reloc: bool = False) -> pymrio.IOSystem:
    """Finds imports reallocation in order to exclude a list of opponents as much as possible

    Args:
        model (Model): object Model defined in model.py
        opponents (List[str]): list of regions' names
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Tuple[pd.DataFrame,pd.DataFrame]: tuple with 2 elements :
            - reallocated Z matrix
            - reallocated Y matrix
    """

    allies = list(set(model.regions) - set(opponents))
    if not reloc:
        allies.remove("FR")
    return scenar_pref(model=model, allies=allies, reloc=reloc)


def scenar_tradewar_china(model: Model, reloc: bool = False) -> pymrio.IOSystem:
    """Finds imports reallocation that prevents trade with China as much as possible


    Args:
        model (Model): object Model defined in model.py
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Tuple[pd.DataFrame]: tuple with 2 elements :
            - reallocated Z matrix
            - reallocated Y matrix
    """

    return scenar_tradewar(
        model=model, opponents=["China, RoW Asia and Pacific"], reloc=reloc
    )

### DUMMY SCENARIO

def scenar_dummy(model, reloc: bool = False) -> pymrio.IOSystem:
    """Dummy scenario functions that return an unchanged scenario


    Args:
        model (Model): object Model defined in model.py
        reloc (bool, optional): True if relocation is allowed. Defaults to False. only available for compatibility

    Returns:
        
    """
    
    return model.iot.copy()

### IMACLIM SCENARIOS



def emissivity_imaclim(model,year:int = 2050,scenario="INDC",**kwargs) -> pymrio.IOSystem:
    
    data=extract_data(aggregation=model.aggregation_name)
    final_data_ratio,Link_country=data[0],data[2]
    
    final_data_ratio=final_data_ratio.swaplevel().sort_index()
    
    indexes=pd.Series(zip(final_data_ratio.index.get_level_values("scenario"),final_data_ratio.index.get_level_values("sector"))).unique()
    final_data_ratio=pd.concat([Link_country.dot(final_data_ratio.loc[scenario,sector]) for scenario,sector in indexes],
                                names=("scenario","sector","regions"),
								keys=indexes,
								axis=0)
    
    final_data_ratio=final_data_ratio.swaplevel().sort_index()

    iot=model.iot.copy()
    iot.stressor_extension.S.loc["CO2"]= \
            pd.concat([iot.stressor_extension.S.loc["CO2",region]*pd.Series(1+final_data_ratio.loc[(scenario,region),year],name="CO2")  if region!="FR"
                    else iot.stressor_extension.S.loc["CO2",region] for region in model.regions ],
                    axis=0,
                    names=("region","sector"),
                    keys=model.regions)

    # recompute the sorrect stressors based on the modifiations
    iot.stressor_extension=recal_stressor_per_region(
    iot=iot)
    
    return iot


def tech_change_imaclim(model,year:int = 2050,scenario="INDC",**kwargs) -> pymrio.IOSystem:
    
    
    final_technical_coef=extract_data(aggregation=model.aggregation_name)[1]
    
    final_technical_coef_FR=final_technical_coef.copy()
    if "FR" in final_technical_coef.columns.get_level_values("region"):
        final_technical_coef_FR.loc[:,(slice(None),slice(None),"FR")]=0 #not modifying French technologies (useful because that is done in MATMAT)
        
    
    
    iot=model.iot.copy()
    A=iot.A.copy()
    Y=iot.Y.copy()
    iot.reset_to_coefficients()
    iot.A=pd.concat([ pd.concat([A.loc[region_export,region_import]*(1+final_technical_coef_FR[scenario,year,region_import]) for region_import in model.regions],
                                names=("region","sector"),
                                keys=model.regions,
                                axis=1)
                     for region_export in model.regions],
                    names=("region","sector"),
                    keys=model.regions,
                    axis=0)
    iot.Y=Y
    iot.x = None
    iot.L=None
    
    #some checks and safeguards might be needed here in order to prevent coefficient sums in each columns of A to be greater than 1 (which can lead to negative results of consumption/production etc)
    columns_problem=iot.A.sum(axis=0)>1
    if (iot.A.sum(axis=0)).any():
        print("Problems on sums of columns ",iot.A.loc[:,columns_problem].sum(axis=0))
        iot.A.loc[:,columns_problem]=iot.A.loc[:,columns_problem]/(iot.A.loc[:,columns_problem].sum(axis=0)+1) #easy but dirty fix
    
    # completing the iot by calculating the missing parts
    iot.calc_all()
    
    # recompute the correct stressors based on the modifiations
    iot.stressor_extension=recal_stressor_per_region(
        iot=iot,)
    
    return iot
    
def production_change_imaclim(model,year:int =2050,scenario:str ="INDC",x_ref=None,**kwargs):
    
    if x_ref is None:
        x_ref=model.iot.x.sort_index().copy()
    else :
        x_ref=copy.deepcopy(x_ref.sort_index())
    
    production_data=extract_data(aggregation=model.aggregation_name)[4]
    
    #get the relative change in production over all sectors/region.
    production_change=production_data.loc[(scenario),year].sort_index()/production_data.loc[(scenario),2015].sort_index()
    
    #create the new scenario iot tables that will include production changes
    iot=model.iot.copy()
    
    #include the production changes in the gross output x of new scenario
    iot.x["indout"]=x_ref["indout"]*production_change
    
    ### The following intend to enable the use of integaretd footprint calculator (they rely on Y and not x, hence modifying x makes no difference,
    ### we therfore create an artificial final demand Y that corresponds to the x we modififed)
    
    # Build a fictional Y matrix which would correspond to the current x : corresponds to the equation Y=(I-A)x
    Y_we_need=iot.x-iot.A.dot(iot.x) 

    # creates a fake detailed Y such that the row sums corresponds to the one we aim for ( we simply scale all values of a row)
    coeffs=Y_we_need["indout"]/iot.Y.sum(axis=1)  #dropped a values here for yweneed
    for row_index in iot.Y.index:
        iot.Y.loc[row_index]=float(coeffs.loc[row_index])*iot.Y.loc[row_index]
        
    # we should now be able to compute some emissions, be carefull, the consumption based acounts don't make much sense here since we started with gross productions and made fictional Y
    iot.stressor_extension=recal_stressor_per_region(
            iot=iot,)
    
    return iot

def consumption_change_imaclim(model,year:int = 2050, scenario:str = "INDC", Y_ref=None,ref_year:int=2015,**kwargs):
    
    if Y_ref is None:
        Y_ref=model.iot.Y.sort_index().copy()
    else :
        Y_ref=copy.deepcopy(Y_ref.sort_index())
        
    consumption_data=extract_data(aggregation=model.aggregation_name)[5]
    
    consumption_change=consumption_data.loc[(scenario),year].sort_index()/consumption_data.loc[(scenario),ref_year].sort_index()
    
    iot=model.iot.copy()
    Y=iot.Y.copy()
    iot.reset_to_flows()
    for region in model.regions:
        for column in Y[region].columns:
            Y[(region,column)]=Y[(region,column)]*consumption_change.loc[region]
    iot.Y=Y
    
    iot.calc_all()
    
    iot.stressor_extension=recal_stressor_per_region(
            iot=iot,)
    return iot

### AVAILABLE SCENARIOS ###

DICT_SCENARIOS = {
    "best": scenar_best,
    "worst": scenar_worst,
    "pref_eu": scenar_pref_eu,
    "tradewar_china": scenar_tradewar_china,
    "dummy":scenar_dummy,
    "emissivity_IMACLIM":emissivity_imaclim,
    "technical_change_IMACLIM":tech_change_imaclim,
    "production_change_IMACLIM":production_change_imaclim,
    "consumption_change_imaclim":consumption_change_imaclim,
}
