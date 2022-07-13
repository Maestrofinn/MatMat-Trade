from model import Model
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple

### AUXILIARY FUNCTIONS FOR SCENARIOS ###


def moves_from_sorted_index_by_sector(
    model: Model, sector: str, regions_index: List[int], reloc: bool = None
) -> Tuple[np.array]:
    """Allocates french importations for a sector in the order given by region_index

    Args:
        model (Model): object Model defined in model.py
        sector (str): name of a product (or industry)
        regions_index (List[int]): list of ordered region indices
        reloc (bool): True if relocation is allowed. Defaults to None.

    Returns:
        Tuple[np.array]: tuple with 2 elements :
            - 2D-array of imports of 'sector' from regions (rows) for french intermediary demands (columns)
            - 2D-array of imports of 'sector' from regions (rows) for french final demands (columns)
    """

    if reloc:
        regions = model.regions
    else:
        regions = model.regions[1:]  # remove FR
    sectors = model.sectors
    demcats = model.y_categories
    Z = model.database.Z
    Y = model.database.Y

    import_demfr = (
        Z["FR"].drop(["FR"], level=0).sum(axis=1).sum(level=1).loc[sector]
        + Y["FR"].drop(["FR"], level=0).sum(axis=1).sum(level=1).loc[sector]
    )

    # share of each french intermediary or final demand in french importations from a given sector
    part_prod_secteurs = []
    part_dem_secteurs = []
    for sec in sectors:
        part_prod_secteurs.append(
            Z[("FR", sec)].drop(["FR"], level=0).sum(level=1).loc[sector] / import_demfr
        )
    for dem in demcats:
        part_dem_secteurs.append(
            Y[("FR", dem)].drop(["FR"], level=0).sum(level=1).loc[sector] / import_demfr
        )

    # french total importations demand for each sector / final demand
    totalinterfromsector = [
        Z["FR"].drop("FR", level=0)[sec].sum(level=1).loc[sector] for sec in sectors
    ]
    totalfinalfromsector = [
        Y["FR"].drop("FR", level=0)[dem].sum(level=1).loc[sector] for dem in demcats
    ]
    totalinterfromsectorFR = [Z["FR"].loc["FR"][sec].loc[sector] for sec in sectors]
    totalfinalfromsectorFR = [Y["FR"].loc["FR"][dem].loc[sector] for dem in demcats]

    # intialization of parts_sects and parts_demcats (result of the function)
    nbreg = len(regions)
    nbsect = len(sectors)
    nbdemcats = len(demcats)
    parts_sects = np.zeros((nbreg, nbsect))
    parts_demcats = np.zeros((nbreg, nbdemcats))

    # export capacities of each regions
    remaining_reg_export = {}
    for reg in regions:
        remaining_reg_export[reg] = (
            Z.drop(columns=reg, level=0).sum(axis=1).loc[(reg, sector)]
        ) + (Y.drop(columns=reg, level=0).sum(axis=1).loc[(reg, sector)])

    # allocations for intermediary demand
    for j in range(nbsect):
        covered = 0
        for i in regions_index:
            reg = regions[i]
            if covered < totalinterfromsector[j] and remaining_reg_export[reg] > 0:
                # if french importations demand from sector j is not satisfied
                # and the ith region is able to export
                if remaining_reg_export[reg] > totalinterfromsector[j] - covered:
                    alloc = totalinterfromsector[j] - covered
                else:
                    alloc = remaining_reg_export[reg]
                parts_sects[i, j] = alloc
                remaining_reg_export[reg] -= alloc
                covered += alloc
            if reg == "FR":
                parts_sects[i, j] += totalinterfromsectorFR[j]

    # allocations for final demands
    for j in range(nbdemcats):
        covered = 0
        for i in regions_index:
            reg = regions[i]
            if covered < totalfinalfromsector[j] and remaining_reg_export[reg] > 0:
                if remaining_reg_export[reg] > totalfinalfromsector[j] - covered:
                    alloc = totalfinalfromsector[j] - covered
                else:
                    alloc = remaining_reg_export[reg]
                parts_demcats[i, j] = alloc
                remaining_reg_export[reg] -= alloc
                covered += alloc
            if reg == "FR":
                parts_demcats[i, j] += totalfinalfromsectorFR[j]

    return parts_sects, parts_demcats


def moves_from_sort_rule(
    model: Model,
    sorting_rule_by_sector: Callable[[str, bool], List[int]],
    reloc: bool = False,
) -> Dict:
    """Allocates french importations for all sectors, sorting the regions with a given rule for each sector

    Args:
        model (Model): object Model defined in model.py
        sorting_rule_by_sector (Callable[str, bool], List[int]]): given a sector name and the reloc value, returns a sorted list of regions' indices
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Dict: dictionnary associating to each sector a dictionnary with :
            - parts_sec : 2D-array of imports of 'sector' from regions (rows) for french intermediary demands (columns)
            - parts_dem : 2D-array of imports of 'sector' from regions (rows) for french final demands (columns)
            - sort : array of indices of regions ascendantly sorted by carbon content
            - reloc : True if relocation is allowed
    """

    sectors_list = model.sectors
    moves = {}
    for sector in sectors_list:
        regions_index = sorting_rule_by_sector(model, sector, reloc)
        parts_sec, parts_dem = moves_from_sorted_index_by_sector(
            model, sector, regions_index, reloc
        )
        moves[sector] = {
            "parts_sec": parts_sec,
            "parts_dem": parts_dem,
            "sort": regions_index,
            "reloc": reloc,
        }
    return moves


def sort_by_content(model: Model, sector: str, reloc: bool = False) -> np.array:
    """Ascendantly sorts all regions by carbon content of a sector

    Args:
        model (Model): object Model defined in model.py
        sector (str): name of a product (or industry)
        reloc (bool, optional): True if relocation is allowed. Defaults to False.


    Returns:
        np.array: array of indices of regions sorted by carbon content
    """

    M = model.database.ghg_emissions_desag.M.sum(axis=0)
    regions_index = np.argsort(M[:, sector].values[1 - reloc :])
    return regions_index


### BEST AND WORST SCENARIOS ###


def scenar_best(model: Model, reloc: bool = False) -> Dict:
    """Finds the least carbon-intense imports reallocation for all sectors


    Args:
        model (Model): object Model defined in model.py
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Dict: dictionnary associating to each sector a dictionnary with :
            - parts_sec : 2D-array of imports of 'sector' from regions (rows) for french intermediary demands (columns)
            - parts_dem : 2D-array of imports of 'sector' from regions (rows) for french final demands (columns)
            - sort : array of indices of regions ascendantly sorted by carbon content
            - reloc : True if relocation is allowed
    """

    return moves_from_sort_rule(model, sort_by_content, reloc)


def scenar_worst(model: Model, reloc: bool = False) -> Dict:
    """Finds the most carbon-intense imports reallocation for all sectors


    Args:
        model (Model): object Model defined in model.py
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Dict: dictionnary associating to each sector a dictionnary with :
            - parts_sec : 2D-array of imports of 'sector' from regions (rows) for french intermediary demands (columns)
            - parts_dem : 2D-array of imports of 'sector' from regions (rows) for french final demands (columns)
            - sort : array of indices of regions ascendantly sorted by carbon content
            - reloc : True if relocation is allowed
    """

    return moves_from_sort_rule(
        model, lambda *args: sort_by_content(*args)[::-1], reloc
    )


### PREFERENCE SCENARIOS ###


def scenar_pref(model, allies: List[str], reloc: bool = False) -> Dict:
    """Finds imports reallocation in order to trade as much as possible with the allies

    Args:
        model (Model): object Model defined in model.py
        allies (List[str]): list of regions' names
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Dict: dictionnary associating to each sector a dictionnary with :
            - parts_sec : 2D-array of imports of 'sector' from regions (rows) for french intermediary demands (columns)
            - parts_dem : 2D-array of imports of 'sector' from regions (rows) for french final demands (columns)
            - sort : array of indices of regions
            - reloc : True if relocation is allowed
    """

    if reloc:
        regions = model.regions
        if not "FR" in allies:
            allies += ["FR"]
    else:
        regions = model.regions[1:]  # remove FR

    moves = {}

    sectors = model.sectors
    demcats = model.y_categories

    for sector in sectors:

        parts_sectors = {reg: [] for reg in regions}
        parts_demcats = {reg: [] for reg in regions}

        ## overall trade related with sector
        sector_exports_Z = model.database.Z.loc[(regions, sector), :].sum(
            axis=0, level=0
        )
        sector_exports_Y = model.database.Y.loc[(regions, sector), :].sum(
            axis=0, level=0
        )
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
        )  # add() handles the possible values of reloc
        allies_total_export_capacity = allies_export_capacity.sum()

        ## specific case leading to a division by 0
        if total_imports_FR_nonallies == 0 and allies_total_export_capacity == 0:
            for reg in regions:
                parts_sectors[reg] = sector_imports_FR_Z.loc[reg]
                parts_demcats[reg] = sector_imports_FR_Y.loc[reg]
            moves[sector] = {
                "parts_sec": parts_sectors,
                "parts_dem": parts_demcats,
                "sort": list(range(len(regions))),
                "reloc": reloc,
            }
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
                if reg not in allies:
                    parts_sectors[reg] = pd.Series(0, index=sectors)
                    parts_demcats[reg] = pd.Series(0, index=demcats)
                else:
                    parts_sectors[reg] = (
                        sector_imports_FR_Z.loc[reg]
                        + coef_Z * allies_export_capacity[reg]
                    )
                    parts_demcats[reg] = (
                        sector_imports_FR_Y.loc[reg]
                        + coef_Y * allies_export_capacity[reg]
                    )
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
                    parts_sectors[reg] = coef_nonallies * sector_imports_FR_Z.loc[reg]
                    parts_demcats[reg] = coef_nonallies * sector_imports_FR_Y.loc[reg]
                else:
                    parts_sectors[reg] = (
                        coef_allies_Z.loc[reg] * sector_imports_FR_Z.loc[reg]
                    )
                    parts_demcats[reg] = (
                        coef_allies_Y.loc[reg] * sector_imports_FR_Y.loc[reg]
                    )

        moves[sector] = {
            "parts_sec": parts_sectors,
            "parts_dem": parts_demcats,
            "sort": list(range(len(regions))),
            "reloc": reloc,
        }

    return moves


def scenar_pref_eu(model: Model, reloc: bool = False) -> Dict:
    """Finds imports reallocation that prioritize trade with European Union


    Args:
        model (Model): object Model defined in model.py
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Dict: dictionnary associating to each sector a dictionnary with :
            - parts_sec : 2D-array of imports of 'sector' from regions (rows) for french intermediary demands (columns)
            - parts_dem : 2D-array of imports of 'sector' from regions (rows) for french final demands (columns)
            - sort : array of indices of regions
            - reloc : True if relocation is allowed
    """

    return scenar_pref(model, ["EU"], reloc)


### TRADE WAR SCENARIOS ###


def scenar_tradewar(model: Model, opponents: List[str], reloc: bool = False) -> Dict:
    """Finds imports reallocation in order to exclude a list of opponents as much as possible

    Args:
        model (Model): object Model defined in model.py
        opponents (List[str]): list of regions' names
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Dict: dictionnary associating to each sector a dictionnary with :
            - parts_sec : 2D-array of imports of 'sector' from regions (rows) for french intermediary demands (columns)
            - parts_dem : 2D-array of imports of 'sector' from regions (rows) for french final demands (columns)
            - sort : array of indices of regions
            - reloc : True if relocation is allowed
    """

    allies = list(set(model.regions) - set(opponents))
    if not reloc:
        allies.remove("FR")
    return scenar_pref(model, allies, reloc)


def scenar_tradewar_china(model: Model, reloc: bool = False) -> Dict:
    """Finds imports reallocation that prevents trade with China as much as possible


    Args:
        model (Model): object Model defined in model.py
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Dict: dictionnary associating to each sector a dictionnary with :
            - parts_sec : 2D-array of imports of 'sector' from regions (rows) for french intermediary demands (columns)
            - parts_dem : 2D-array of imports of 'sector' from regions (rows) for french final demands (columns)
            - sort : array of indices of regions
            - reloc : True if relocation is allowed
    """

    return scenar_tradewar(model, ["China, RoW Asia and Pacific"], reloc)


### SHOCK FUNCTIONS ###


def shockv2(sector_list, demcatlist, reg_list, Z, Y, move, sector):
    Z_modif = Z.copy()
    Y_modif = Y.copy()
    if move["reloc"]:
        regs = reg_list
    else:
        regs = reg_list[1:]

    for i in range(len(sector_list)):
        for j in range(len(regs)):
            Z_modif.loc[(regs[move["sort"][j]], sector), ("FR", sector_list[i])] = move[
                "parts_sec"
            ][move["sort"][j], i]
    for i in range(len(demcatlist)):
        for j in range(len(regs)):
            Y_modif.loc[(regs[move["sort"][j]], sector), ("FR", demcatlist[i])] = move[
                "parts_dem"
            ][move["sort"][j], i]
    return Z_modif, Y_modif


def shockv3(sector_list, demcatlist, reg_list, Z, Y, move, sector):
    Z_modif = Z.copy()
    Y_modif = Y.copy()
    if move["reloc"]:
        regs = reg_list
    else:
        regs = reg_list[1:]

    for j in range(len(sector_list)):
        for r in regs:
            Z_modif.loc[(r, sector), ("FR", sector_list[j])] = move["parts_sec"][r][j]
    for i in range(len(demcatlist)):
        for r in regs:
            Y_modif.loc[(r, sector), ("FR", demcatlist[i])] = move["parts_dem"][r][i]
    return Z_modif, Y_modif


### FROM SCENARIO TO COUNTERFACTUAL ###

DICT_SCENARIOS = {
    "best": {
        "scenario_function": scenar_best,
        "shock_function": shockv2,
    },
    "worst": {
        "scenario_function": scenar_worst,
        "shock_function": shockv2,
    },
    "pref_eu": {
        "scenario_function": scenar_pref_eu,
        "shock_function": shockv3,
    },
    "tradewar_china": {
        "scenario_function": scenar_tradewar_china,
        "shock_function": shockv3,
    },
}
