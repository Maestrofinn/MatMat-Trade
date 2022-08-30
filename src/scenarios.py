import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple

from src.model import Model


### AUXILIARY FUNCTIONS FOR SCENARIOS ###


def moves_from_sorted_index_by_sector(
    model: Model, sector: str, regions_index: List[int], reloc: bool = None
) -> Tuple[pd.DataFrame]:
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
        regions = model.regions[1:]  # remove FR
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
        reg: Z.drop(columns=reg, level=0).sum(axis=1).loc[(reg, sector)]
        + Y.drop(columns=reg, level=0).sum(axis=1).loc[(reg, sector)]
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
    model: Model,
    sorting_rule_by_sector: Callable[[str, bool], List[int]],
    reloc: bool = False,
) -> Tuple[pd.DataFrame]:
    """Allocates french importations for all sectors, sorting the regions with a given rule for each sector

    Args:
        model (Model): object Model defined in model.py
        sorting_rule_by_sector (Callable[[str, bool], List[int]]): given a sector name and the reloc value, returns a sorted list of regions' indices
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Tuple[pd.DataFrame]: tuple with 2 elements :
            - reallocated Z matrix
            - reallocated Y matrix
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
    return new_Z, new_Y


def sort_by_content(model: Model, sector: str, reloc: bool = False) -> np.array:
    """Ascendantly sorts all regions by carbon content of a sector

    Args:
        model (Model): object Model defined in model.py
        sector (str): name of a product (or industry)
        reloc (bool, optional): True if relocation is allowed. Defaults to False.


    Returns:
        np.array: array of indices of regions sorted by carbon content
    """

    M = model.iot.ghg_emissions_desag.M.sum(axis=0)
    regions_index = np.argsort(M[:, sector].values[1 - reloc :])
    return regions_index


### BEST AND WORST SCENARIOS ###


def scenar_best(model: Model, reloc: bool = False) -> Dict:
    """Finds the least carbon-intense imports reallocation for all sectors


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


def scenar_worst(model: Model, reloc: bool = False) -> Dict:
    """Finds the most carbon-intense imports reallocation for all sectors


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
        reloc=reloc,
    )


### PREFERENCE SCENARIOS ###


def scenar_pref(model, allies: List[str], reloc: bool = False) -> Dict:
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

    return new_Z, new_Y


def scenar_pref_eu(model: Model, reloc: bool = False) -> Dict:
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


def scenar_tradewar(model: Model, opponents: List[str], reloc: bool = False) -> Dict:
    """Finds imports reallocation in order to exclude a list of opponents as much as possible

    Args:
        model (Model): object Model defined in model.py
        opponents (List[str]): list of regions' names
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Tuple[pd.DataFrame]: tuple with 2 elements :
            - reallocated Z matrix
            - reallocated Y matrix
    """

    allies = list(set(model.regions) - set(opponents))
    if not reloc:
        allies.remove("FR")
    return scenar_pref(model=model, allies=allies, reloc=reloc)


def scenar_tradewar_china(model: Model, reloc: bool = False) -> Dict:
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


### AVAILABLE SCENARIOS ###

DICT_SCENARIOS = {
    "best": scenar_best,
    "worst": scenar_worst,
    "pref_eu": scenar_pref_eu,
    "tradewar_china": scenar_tradewar_china,
}
