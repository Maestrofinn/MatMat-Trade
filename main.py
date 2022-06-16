""" Python main script of MatMat trade module
	"""

###########################
#%% IMPORT MODULES
###########################
# general

from typing import Callable, Dict, List, Tuple
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# scientific
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# local folder
from local_paths import data_dir
from local_paths import output_dir

# local library
from utils import Tools

###########################
#%% SETTINGS
###########################
# creating colormap for 11 regions and for 10 regions for plotting
import matplotlib.colors as mpl_col

colors = [plt.cm.tab10(i) for i in range(10)]
colors.append(mpl_col.to_rgba("gold", alpha=1.0))
colors_no_FR = colors[1:]

my_cmap = mpl_col.LinearSegmentedColormap.from_list("my_cmap", colors)
my_cmap_noFR = mpl_col.LinearSegmentedColormap.from_list("my_cmap_noFR", colors[1:])

# year to study in [*range(1995, 2022 + 1)]
base_year = 2015

# system type: pxp or ixi
system = "pxp"

# agg name: to implement in agg_matrix.xlsx
agg_name = {"sector": "ref", "region": "ref"}

# define filename concatenating settings
concat_settings = str(base_year) + "_" + agg_name["sector"] + "_" + agg_name["region"]

# set if rebuilding calibration from exiobase
calib = False


###########################
#%% READ/ORGANIZE/CLEAN DATA
###########################

# build calibrated data
reference = Tools.build_reference(calib, data_dir, base_year, system, agg_name)


###########################
#%%CALCULATIONS
###########################
# Calculate reference system
reference.calc_all()
reference.ghg_emissions_desag = Tools.recal_extensions_per_region(
    reference, "ghg_emissions"
)


# save reference data base
reference.save_all(output_dir / ("reference" + "_" + concat_settings))

# init counterfactual(s)
counterfactual = reference.copy()
counterfactual.remove_extension("ghg_emissions_desag")

nbsect = len(reference.get_sectors())  # number of economic sectors

###########################
#%% DEFINE FUNCTIONS FOR SCENARIOS
###########################


def get_least(sector: str, reloc: bool = False) -> str:
    """Find the least carbon-intense region associated with a given sector,
    among these who export more than French demand for this sector

    Args:
        sector (str): name of a product (or industry)
        reloc (bool): True if relocation is allowed. Defaults to False.

    Returns:
        str: name of a region
    """

    M = reference.ghg_emissions_desag.M.sum(axis=0)

    import_demfr = Tools.compute_french_demands(reference, sector)

    if reloc:
        regs = reference.get_regions()
    else:
        regs = reference.get_regions()[1:]  # remove FR

    ind = 0
    for i in range(len(regs)):
        if (
            M[regs[i], sector] < M[regs[ind], sector]
            and reference.Z.loc[regs[i]].drop(columns=regs[i]).sum(axis=1).loc[sector]
            > import_demfr
        ):  # choose a region iff it emits less than the previous best AND its export are higher than french demand
            ind = i
    return regs[ind]


def sort_by_content(sector: str, regs: List[str]) -> np.array:
    """Sort all regions by carbon content of a sector

    Args:
        sector (str): name of a product (or industry)
        regs (List[str]): list of the names of regions

    Returns:
        np.array: array of indices of regions sorted by carbon content
    """
    M = reference.ghg_emissions_desag.M.sum(axis=0)
    carbon_content_sector = [M[reg, sector] for reg in regs]
    index_sorted = np.argsort(carbon_content_sector)
    return index_sorted


def moves_from_sorted_index_by_sector(
    sector: str, regions_index: List[int], reloc: bool
) -> Tuple[np.array]:
    """Allocate french importations for a sector in the order given by region_index

    Args:
        sector (str): name of a product (or industry)
        regions_index (List[int]): list of ordered region indices
        reloc (bool): True if relocation is allowed

    Returns:
        Tuple[np.array]: tuple with 3 elements :
                        - 2D-array of imports of 'sector' from regions (rows) for french intermediary demands (columns)
                        - 2D-array of imports of 'sector' from regions (rows) for french final demands (columns)
                        - array of indices of regions sorted by carbon content
    """

    if reloc:
        regs = reference.get_regions()
    else:
        regs = reference.get_regions()[1:]  # remove FR

    sectors = reference.get_sectors()
    demcats = reference.get_Y_categories()

    import_demfr = Tools.compute_french_demands(reference, sector)

    # share of each french intermediary or final demand in french importations from a given sector
    part_prod_secteurs = []
    part_dem_secteurs = []
    for sec in sectors:
        part_prod_secteurs.append(
            reference.Z[("FR", sec)].drop(["FR"]).sum(level=1).loc[sector]
            / import_demfr
        )
    for dem in demcats:
        part_dem_secteurs.append(
            reference.Y[("FR", dem)].drop(["FR"]).sum(level=1).loc[sector]
            / import_demfr
        )

    # french total importations demand for each sector / final demand
    totalinterfromsector = [
        reference.Z["FR"].drop("FR")[sec].sum(level=1).loc[sector] for sec in sectors
    ]
    totalfinalfromsector = [
        reference.Y["FR"].drop("FR")[dem].sum(level=1).loc[sector] for dem in demcats
    ]

    # intialization of parts_sects and parts_demcats (result of the function)
    nbreg = len(regs)
    nbsect = len(sectors)
    nbdemcats = len(demcats)
    parts_sects = np.zeros((nbreg, nbsect))
    parts_demcats = np.zeros((nbreg, nbdemcats))

    # export capacities of each regions
    remaining_reg_export = []
    for i in regions_index:
        reg = regs[i]
        remaining_reg_export.append(
            reference.Z.drop(columns=reg).sum(axis=1).loc[(reg, sector)]
        )

    # allocations for intermediary demand
    for j in range(nbsect):
        covered = 0
        for i in regions_index:
            if covered < totalinterfromsector[j] and remaining_reg_export[i] > 0:
                # if french importations demand from sector j is not satisfied
                # and the ith region is able to export
                if remaining_reg_export[i] > totalinterfromsector[j] - covered:
                    alloc = totalinterfromsector[j] - covered
                else:
                    alloc = remaining_reg_export[i]
                parts_sects[i, j] = alloc
                remaining_reg_export[i] -= alloc
                covered += alloc

    # allocations for final demands
    for j in range(nbdemcats):
        covered = 0
        for i in regions_index:
            if covered < totalfinalfromsector[j] and remaining_reg_export[i] > 0:
                if remaining_reg_export[i] > totalfinalfromsector[j] - covered:
                    alloc = totalfinalfromsector[j] - covered
                else:
                    alloc = remaining_reg_export[i]
                parts_demcats[i, j] = alloc
                remaining_reg_export[i] -= alloc
                covered += alloc

    return parts_sects, parts_demcats, regions_index


def worst_moves(sector: str, reloc: bool) -> Tuple[np.array]:
    """Find the most carbon-intense imports reallocation for a sector

    Args:
        sector (str): name of a product (or industry)
        reloc (bool): True if relocation is allowed
    Returns:
        Tuple[np.array]: tuple with 3 elements :
                        - 2D-array of imports of 'sector' from regions (rows) for french intermediary demands (columns)
                        - 2D-array of imports of 'sector' from regions (rows) for french final demands (columns)
                        - array of indices of regions descendantly sorted by carbon content
    """

    if reloc:
        regs = reference.get_regions()
    else:
        regs = reference.get_regions()[1:]  # remove FR

    return moves_from_sorted_index_by_sector(
        sector, sort_by_content(sector, regs)[::-1], reloc
    )


def best_moves(sector: str, reloc: bool) -> Tuple[np.array]:
    """Find the least carbon-intense imports reallocation for a sector

    Args:
        sector (str): name of a product (or industry)
        reloc (bool): True if relocation is allowed
    Returns:
        Tuple[np.array]: tuple with 3 elements :
                        - 2D-array of imports of 'sector' from regions (rows) for french intermediary demands (columns)
                        - 2D-array of imports of 'sector' from regions (rows) for french final demands (columns)
                        - array of indices of regions ascendantly sorted by carbon content
    """

    if reloc:
        regs = reference.get_regions()
    else:
        regs = reference.get_regions()[1:]  # remove FR

    return moves_from_sorted_index_by_sector(
        sector, sort_by_content(sector, regs), reloc
    )


def moves_from_sort_rule(
    sorting_rule_by_sector: Callable[[List[str], str], List[int]], reloc: bool = False
) -> Dict:
    """Allocate french importations for all sectors, sorting the regions with a given rule for each sector

    Args:
            sorting_rule_by_sector (Callable[[List[str], str], List[int]]): given a list of regions' names and a sector's name, returns a sorted list of regions' indices
            reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
            Dict: dictionnary associating to each sector a dictionnary with :
                            - parts_sec : 2D-array of imports of 'sector' from regions (rows) for french intermediary demands (columns)
                            - parts_dem : 2D-array of imports of 'sector' from regions (rows) for french final demands (columns)
                            - sort : array of indices of regions ascendantly sorted by carbon content
                            - reloc : True if relocation is allowed
    """

    sectors_list = reference.get_sectors()
    if reloc:
        regs = reference.get_regions()
    else:
        regs = reference.get_regions()[1:]  # remove FR
    moves = {}
    for sector in sectors_list:
        regions_index = sorting_rule_by_sector(sector, regs)
        parts_sec, parts_dem, _ = moves_from_sorted_index_by_sector(
            sector, regions_index, reloc
        )
        moves[sector] = {
            "parts_sec": parts_sec,
            "parts_dem": parts_dem,
            "sort": regions_index,
            "reloc": reloc,
        }
    return sectors_list, moves


def scenar_bestv2(reloc: bool = False) -> Dict:
    """Find the least carbon-intense imports reallocation for all sectors


    Args:
            reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
            Dict: dictionnary associating to each sector a dictionnary with :
                            - parts_sec : 2D-array of imports of 'sector' from regions (rows) for french intermediary demands (columns)
                            - parts_dem : 2D-array of imports of 'sector' from regions (rows) for french final demands (columns)
                            - sort : array of indices of regions ascendantly sorted by carbon content
                            - reloc : True if relocation is allowed
    """

    return moves_from_sort_rule(sort_by_content, reloc)


def scenar_worstv2(reloc: bool = False) -> Dict:
    """Find the most carbon-intense imports reallocation for all sectors


    Args:
            reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
            Dict: dictionnary associating to each sector a dictionnary with :
                            - parts_sec : 2D-array of imports of 'sector' from regions (rows) for french intermediary demands (columns)
                            - parts_dem : 2D-array of imports of 'sector' from regions (rows) for french final demands (columns)
                            - sort : array of indices of regions ascendantly sorted by carbon content
                            - reloc : True if relocation is allowed
    """

    return moves_from_sort_rule(lambda *args: sort_by_content(*args)[::-1], reloc)


def get_worst(sector, reloc):
    """Find the worst region for a given sector, i.e. the most carbon-intense region associated with this sector"""
    # par défaut on ne se laisse pas la possibilité de relocaliser en FR
    M = reference.ghg_emissions_desag.M.sum()
    regs = list(reference.get_regions())[1:]
    if reloc:
        regs = list(reference.get_regions())
    ind = 0
    for i in range(1, len(regs)):
        if M[regs[i], sector] > M[regs[ind], sector]:
            ind = i
    return regs[ind]


def scenar_best(reloc=False, deloc=False):
    """Implement a scenario illustrating the best possible trade policy"""
    sectors_list = list(reference.get_sectors())
    sectors_gl = []
    moves_gl = []
    for sector in sectors_list:
        best = get_least(sector, reloc)
        if deloc:
            for i in range(len(list(reference.get_regions())) - 1):
                sectors_gl.append(sector)
        else:
            for i in range(len(list(reference.get_regions())) - 2):
                sectors_gl.append(sector)
        for reg in list(reference.get_regions()):
            if deloc:
                if reg != best:
                    moves_gl.append([reg, best])
            else:
                if reg != best:
                    if reg != "FR":
                        moves_gl.append([reg, best])
    quantities = [1 for i in range(len(sectors_gl))]
    return sectors_gl, moves_gl, quantities


def scenar_worst(reloc=False, deloc=False):
    """Implement a scenario illustrating the worst possible trade policy"""
    sectors_list = list(reference.get_sectors())
    sectors_gl = []
    moves_gl = []
    for sector in sectors_list:
        worst = get_worst(sector, reloc)
        if deloc:
            for i in range(len(list(reference.get_regions())) - 1):
                sectors_gl.append(sector)
        else:
            for i in range(len(list(reference.get_regions())) - 2):
                sectors_gl.append(sector)
        for reg in list(reference.get_regions()):
            if deloc:
                if reg != worst:
                    moves_gl.append([reg, worst])
            else:
                if reg != worst:
                    if reg != "FR":
                        moves_gl.append([reg, worst])
    quantities = [1 for i in range(len(sectors_gl))]
    return sectors_gl, moves_gl, quantities


def scenar_pref_europe():
    """Implement a scenario illustrating the preference for trade with European Union"""
    nbreg = len(list(reference.get_regions()))
    sectors = (nbreg - 2) * list(reference.get_sectors())
    quantities = [1 for i in range(len(sectors))]
    moves = []
    for i in range(nbreg):
        reg = reference.get_regions()[i]
        if reg != "Europe" and reg != "FR":
            for j in range(len(list(reference.get_sectors()))):
                moves.append([reg, "Europe"])
    return sectors, moves, quantities


def scenar_pref_europev3(reloc=False):
    """Implement a scenario illustrating the preference for trade with European Union"""
    if reloc:
        regs = list(reference.get_regions())
    else:
        regs = list(reference.get_regions())[1:]  # remove FR
    sectors_list = list(reference.get_sectors())
    demcats = list(reference.get_Y_categories())
    nbdemcats = len(demcats)
    nbreg = len(regs)
    moves = {}
    for i in range(nbsect):
        # initialization of outputs
        parts_sects = {}
        parts_dem = {}
        for r in regs:
            parts_sects[r] = np.zeros(nbsect)
            parts_dem[r] = np.zeros(nbdemcats)

        # construction of french needs of imports
        totalfromsector = np.zeros(nbsect)
        totalfinalfromsector = np.zeros(nbdemcats)
        for j in range(nbsect):
            # sum on regions of imports of imports of sector for french sector j
            totalfromsector[j] = np.sum(
                [
                    reference.Z["FR"]
                    .drop("FR")[sectors_list[j]]
                    .loc[(regs[k], sectors_list[i])]
                    for k in range(nbreg)
                ]
            )
        for j in range(nbdemcats):
            totalfinalfromsector[j] = np.sum(
                [
                    reference.Y["FR"]
                    .drop("FR")[demcats[j]]
                    .loc[(regs[k], sectors_list[i])]
                    for k in range(nbreg)
                ]
            )

        # exports capacity of all regions for sector i
        reg_export = {}
        for r in range(nbreg):
            reg_export[regs[r]] = (
                reference.Z.drop(columns=regs[r])
                .sum(axis=1)
                .loc[(regs[r], sectors_list[i])]
            )  # exports from this reg/sec

        remaining_reg_export_UE = reg_export["EU"]
        for j in range(nbsect):
            if totalfromsector[j] != 0:
                if remaining_reg_export_UE > 0:
                    # if europe can still export some sector[i]
                    if remaining_reg_export_UE > totalfromsector[j]:
                        alloc = totalfromsector[j]
                    else:
                        alloc = reference.Z.loc[
                            ("EU", sectors_list[i]), ("FR", sectors_list[j])
                        ]  # tout ou rien ici
                    parts_sects["EU"][j] = alloc
                    remaining_reg_export_UE -= alloc
                    # remove from other regions a part of what has been assigned to the EU
                    # this part corresponds to the part of the country in original french imports for sector j
                    for r in regs:
                        if r != "EU":
                            parts_sects[r][j] = reference.Z.loc[
                                (r, sectors_list[i]), ("FR", sectors_list[j])
                            ] * (1 - alloc / totalfromsector[j])

        for j in range(nbdemcats):
            if totalfinalfromsector[j] != 0:
                if remaining_reg_export_UE > 0:
                    # if europe can still export some sector[i]
                    if remaining_reg_export_UE > totalfinalfromsector[j]:
                        alloc = totalfinalfromsector[j]
                    else:
                        alloc = reference.Y.loc[
                            ("EU", sectors_list[i]), ("FR", demcats[j])
                        ]  # tout ou rien ici
                    parts_dem["EU"][j] = alloc
                    remaining_reg_export_UE -= alloc
                    # remove from other regions a part of what has been assigned to the EU
                    # this part corresponds to the part of the country in original french imports for sector j
                    for r in regs:
                        if r != "EU":
                            parts_sects[r][j] = reference.Y.loc[
                                (r, sectors_list[i]), ("FR", demcats[j])
                            ] * (1 - alloc / totalfinalfromsector[j])

        moves[sectors_list[i]] = {
            "parts_sec": parts_sects,
            "parts_dem": parts_dem,
            "sort": [i for i in range(len(regs))],
            "reloc": reloc,
        }
    return sectors_list, moves


def scenar_guerre_chine(reloc=False):
    """Implement a scenario illustrating an economic war with the region containing China"""
    china_region = "China, RoW Asia and Pacific"
    if reloc:
        regs = list(reference.get_regions())
    else:
        regs = list(reference.get_regions())[1:]  # remove FR
    sectors_list = list(reference.get_sectors())
    demcats = list(reference.get_Y_categories())
    nbdemcats = len(demcats)
    nbreg = len(regs)
    moves = {}
    for i in range(nbsect):
        # initialization of outputs
        parts_sects = {}
        parts_dem = {}
        for r in regs:
            parts_sects[r] = np.zeros(nbsect)
            parts_dem[r] = np.zeros(nbdemcats)

        # construction of french needs of imports
        totalfromsector = np.zeros(nbsect)
        fromchinasector = np.zeros(nbsect)
        totalfinalfromsector = np.zeros(nbdemcats)
        finalfromchinasector = np.zeros(nbdemcats)
        for j in range(nbsect):
            # sum on regions of imports of imports of sector for french sector j
            totalfromsector[j] = np.sum(
                [
                    reference.Z["FR"]
                    .drop("FR")[sectors_list[j]]
                    .loc[(regs[k], sectors_list[i])]
                    for k in range(nbreg)
                ]
            )
            fromchinasector[j] = (
                reference.Z["FR"]
                .drop("FR")[sectors_list[j]]
                .loc[(china_region, sectors_list[i])]
            )

        for j in range(nbdemcats):
            totalfinalfromsector[j] = np.sum(
                [
                    reference.Y["FR"]
                    .drop("FR")[demcats[j]]
                    .loc[(regs[k], sectors_list[i])]
                    for k in range(nbreg)
                ]
            )
            finalfromchinasector[j] = (
                reference.Y["FR"]
                .drop("FR")[demcats[j]]
                .loc[(china_region, sectors_list[i])]
            )
        # exports capacity of all regions for sector i
        reg_export = {}
        for r in range(nbreg):
            reg_export[regs[r]] = (
                reference.Z.drop(columns=regs[r])
                .sum(axis=1)
                .loc[(regs[r], sectors_list[i])]
            )  # exports from this reg/sec

        for j in range(nbsect):
            if totalfromsector[j] != 0:
                for r in regs:
                    if r != china_region:
                        old = reference.Z.loc[
                            (r, sectors_list[i]), ("FR", sectors_list[j])
                        ]
                        parts_sects[r][j] = old
                if fromchinasector[j] > 0:
                    for r in regs:
                        if r != china_region:
                            old = reference.Z.loc[
                                (r, sectors_list[i]), ("FR", sectors_list[j])
                            ]
                            if fromchinasector[j] + old < reg_export[r]:
                                alloc = fromchinasector[j]
                                parts_sects[r][j] += alloc
                                fromchinasector[j] = 0
                                reg_export[r] -= alloc
                                break
                            else:
                                alloc = reg_export[r]
                                reg_export[r] -= alloc
                                fromchinasector[j] -= alloc
                    parts_sects[china_region][j] = fromchinasector[j]

        for j in range(nbdemcats):
            if totalfinalfromsector[j] != 0:
                for r in regs:
                    if r != china_region:
                        old = reference.Y.loc[(r, sectors_list[i]), ("FR", demcats[j])]
                        parts_dem[r][j] = old
                if finalfromchinasector[j] > 0:
                    for r in regs:
                        if r != china_region:
                            old = reference.Y.loc[
                                (r, sectors_list[i]), ("FR", demcats[j])
                            ]
                            if finalfromchinasector[j] + old < reg_export[r]:
                                alloc = finalfromchinasector[j]
                                parts_dem[r][j] += alloc
                                finalfromchinasector[j] = 0
                                break
                            else:
                                alloc = reg_export[r]
                                finalfromchinasector[j] -= alloc
                    parts_dem[china_region][j] = finalfromchinasector[j]

        moves[sectors_list[i]] = {
            "parts_sec": parts_sects,
            "parts_dem": parts_dem,
            "sort": [i for i in range(len(regs))],
            "reloc": reloc,
        }
    return sectors_list, moves


###########################
#%% PLOT FIGURES
###########################

sectors_list = list(reference.get_sectors())  # List of economic sectors
reg_list = list(reference.get_regions())  # List of regions
demcat_list = list(reference.get_Y_categories())  # List of final demand categories

plot_EC_France = False  # True for plotting French carbon footprint

if plot_EC_France:  # Plots the french carbon footprint (D_pba-D_exp+D_imp+F_Y)
    reference.ghg_emissions_desag.D_imp.sum(level=0)["FR"].sum(axis=1).sum()  # ['EU']
    new_df = pd.DataFrame(
        None,
        columns=["Exportees", "Production", "Importees", "Conso finale"],
        index=[""],
    )
    new_df.fillna(value=0.0, inplace=True)
    new_df["Exportees"] = -reference.ghg_emissions_desag.D_exp["FR"].sum().sum()
    new_df["Production"] = reference.ghg_emissions_desag.D_pba["FR"].sum().sum()
    new_df["Importees"] = reference.ghg_emissions_desag.D_imp["FR"].sum().sum()
    new_df["Conso finale"] = reference.ghg_emissions_desag.F_Y["FR"].sum().sum()
    new_df.plot.barh(stacked=True, fontsize=17, figsize=(10, 5), rot=0)
    plt.title("Empreinte carbone de la France", size=17)
    plt.xlabel("MtCO2eq", size=15)
    plt.tight_layout()
    plt.grid(visible=True)
    plt.legend(prop={"size": 17})
    plt.savefig("figures/EC_France.png")
    # plt.show()


# Dictionary to reaggreagate account matrices with less regions for the sake of better visibility
dict_regions = {
    "FR": ["FR"],
    "UK, Norway, Switzerland": ["UK, Norway, Switzerland"],
    "China+": ["China, RoW Asia and Pacific"],
    "EU": ["EU"],
    "RoW": [
        "United States",
        "Asia, Row Europe",
        "RoW America,Turkey, Taïwan",
        "RoW Middle East, Australia",
        "Brazil, Mexico",
        "South Africa",
        "Japan, Indonesia, RoW Africa",
    ],
}

scenarios_dict = {
    "Best": {"sector_moves": scenar_bestv2(), "shock_function": Tools.shockv2},
    "Worst": {"sector_moves": scenar_worstv2(), "shock_function": Tools.shockv2},
    "Pref_EU": {
        "sector_moves": scenar_pref_europev3(),
        "shock_function": Tools.shockv3,
    },
    "War_China": {
        "sector_moves": scenar_guerre_chine(),
        "shock_function": Tools.shockv3,
    },
}


def compare_scenarios():
    """Draw figures to compare the carbont footprints associated with the different scenarios"""
    print("compare_scenarios")

    D_cba_all_scen = pd.DataFrame(
        None,
        index=["FR", "UK, Norway, Switzerland", "China+", "EU", "RoW"],
        columns=["Best", "Pref_EU", "War_China", "Reference", "Worst"],
    )
    D_cba_all_scen.fillna(value=0.0, inplace=True)
    D_cba_all_scen["Reference"] = (
        Tools.reag_D_regions(reference, dict_reag_regions=dict_regions)["FR"]
        .sum(level=0)
        .sum(axis=1)
    )
    D_cba_all_scen.loc["FR", "Reference"] += (
        reference.ghg_emissions_desag.F_Y["FR"].sum().sum()
    )
    Commerce_all_scen = pd.DataFrame(
        None,
        index=["FR", "UK, Norway, Switzerland", "China+", "EU", "RoW"],
        columns=["Best", "Pref_EU", "War_China", "Reference", "Worst"],
    )
    Commerce_all_scen.fillna(value=0.0, inplace=True)
    for reg in dict_regions:
        for reg_2 in dict_regions[reg]:
            Commerce_all_scen.loc[reg, "Reference"] += (
                reference.Y["FR"].sum(axis=1) + reference.Z["FR"].sum(axis=1)
            ).sum(level=0)[reg_2]
    # calculate couterfactual systems
    for scenar in ["Best", "Pref_EU", "Worst", "War_China"]:
        print(scenar)
        counterfactual = Tools.compute_counterfactual(
            counterfactual, scenarios_dict[scenar], demcat_list, reg_list
        )
        counterfactual.calc_all()
        counterfactual.ghg_emissions_desag = Tools.recal_extensions_per_region(
            counterfactual, "ghg_emissions"
        )
        D_cba_all_scen[scenar] = (
            Tools.reag_D_regions(counterfactual, dict_reag_regions=dict_regions)["FR"]
            .sum(level=0)
            .sum(axis=1)
        )
        for reg in dict_regions:
            for reg_2 in dict_regions[reg]:
                Commerce_all_scen.loc[reg, scenar] += (
                    counterfactual.Y["FR"].sum(axis=1)
                    + counterfactual.Z["FR"].sum(axis=1)
                ).sum(level=0)[reg_2]
        D_cba_all_scen.loc["FR", scenar] += (
            counterfactual.ghg_emissions_desag.F_Y["FR"].sum().sum()
        )
    # print(D_cba_all_scen)
    # print(Commerce_all_scen)

    D_cba_all_scen.T.plot.bar(
        stacked=True, fontsize=17, figsize=(12, 8), rot=0, color=colors[:5]
    )

    plt.title("Empreinte carbone de la France", size=17)
    plt.ylabel("MtCO2eq", size=15)
    plt.tight_layout()
    plt.grid(visible=True)
    plt.legend(prop={"size": 15})
    plt.savefig("figures/comparaison_5_scenarios.png")
    # plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2)
    D_cba_all_scen.drop("FR").T.drop(["War_China", "Pref_EU"]).plot.bar(
        ax=axes[0],
        stacked=True,
        fontsize=17,
        figsize=(12, 8),
        rot=0,
        color=colors_no_FR[:4],
    )
    axes[0].set_title("Emissions de GES importées par la France", size=17)
    axes[0].legend(prop={"size": 15})
    axes[0].set_ylabel("MtCO2eq", size=15)
    Commerce_all_scen.drop("FR").T.drop(["War_China", "Pref_EU"]).plot.bar(
        ax=axes[1],
        stacked=True,
        fontsize=17,
        figsize=(12, 8),
        rot=0,
        legend=False,
        color=colors_no_FR[:4],
    )
    axes[1].set_title("Importations françaises", size=17)
    axes[1].set_ylabel("M€", size=15)
    # axes[1].legend(prop={'size': 15})
    plt.tight_layout()
    plt.savefig("figures/comparaison_3_scenarios_bornes.png")
    plt.show()

    return


plot_compare_scenarios = False  # True for plotting scenarios comparison

if plot_compare_scenarios:
    compare_scenarios()
    exit()

###########################
#%% CHOICE OF THE SCENARIO
###########################

scenarios = ["Best", "Worst", "Pref_EU", "War_china"]
chosen_scenario = scenarios[2]


###########################
#%% COMPUTE COUNTERFACTUAL SYSTEM
###########################

counterfactual = Tools.compute_counterfactual(
    counterfactual, scenarios_dict[chosen_scenario], demcat_list, reg_list
)


# calculate counterfactual(s) system
counterfactual.calc_all()

counterfactual.ghg_emissions_desag = Tools.recal_extensions_per_region(
    counterfactual, "ghg_emissions"
)

# save conterfactural(s)
counterfactual.save_all(output_dir / ("counterfactual" + "_" + concat_settings))


# ###########################
#%% DEFINE FUNCTIONS FOR VISUALISATION
# ###########################


ghg_list = ["CO2", "CH4", "N2O", "SF6", "HFC", "PFC"]
sectors_list = list(reference.get_sectors())  # List of economic sectors
reg_list = list(reference.get_regions())  # List of regions


def vision_commerce(notallsectors=False):
    """Plot the importations variations for every sector"""
    if notallsectors:
        sectors_list = ["Agriculture", "Energy", "Industry", "Composite"]
    else:
        sectors_list = list(reference.get_sectors())
    df_eco_ref = reference.Y["FR"].sum(axis=1) + reference.Z["FR"].sum(axis=1)
    df_eco_cont = counterfactual.Y["FR"].sum(axis=1) + counterfactual.Z["FR"].sum(
        axis=1
    )

    comm_ref = pd.DataFrame(
        [df_eco_ref.sum(level=0)[r] for r in reg_list[1:]], index=reg_list[1:]
    )

    comm_ref.T.plot.barh(stacked=True, fontsize=17, color=colors_no_FR, figsize=(12, 5))
    plt.title("Importations totales françaises, scenario Ref", size=17)
    plt.tight_layout()
    plt.grid(visible=True)
    plt.legend(prop={"size": 12})
    plt.savefig("figures/commerce_ref.png")
    plt.close()

    comm_ref.T.plot.barh(stacked=True, fontsize=17, color=colors)
    plt.title("Importations totales françaises", size=17)
    plt.tight_layout()
    plt.grid(visible=True)
    plt.legend(prop={"size": 12})
    # plt.show()
    comm_cumul_non_fr = pd.DataFrame(
        {
            "ref": [df_eco_ref.sum(level=0)[r] for r in reg_list[1:]],
            "cont": [df_eco_cont.sum(level=0)[r] for r in reg_list[1:]],
        },
        index=reg_list[1:],
    )
    comm_cumul_non_fr.T.plot.barh(
        stacked=True, fontsize=17, figsize=(12, 8), color=colors_no_FR
    )
    plt.title("Importations totales françaises", size=17)
    plt.tight_layout()
    plt.grid(visible=True)
    plt.legend(prop={"size": 12})
    plt.savefig("figures/commerce_imports_totales")
    # plt.show()

    dict_sect_plot = {}
    for sec in sectors_list:
        dict_sect_plot[(sec, "ref")] = [
            df_eco_ref.loc[(r, sec)] / df_eco_ref.drop(["FR"]).sum(level=1).loc[sec]
            for r in reg_list[1:]
        ]
        dict_sect_plot[(sec, "cont")] = []
        for r in reg_list[1:]:
            if df_eco_cont.drop(["FR"]).sum(level=1).loc[sec] != 0:
                dict_sect_plot[(sec, "cont")].append(
                    df_eco_cont.loc[(r, sec)]
                    / df_eco_cont.drop(["FR"]).sum(level=1).loc[sec]
                )
            else:
                dict_sect_plot[(sec, "cont")].append(df_eco_cont.loc[(r, sec)])

    df_plot = pd.DataFrame(data=dict_sect_plot, index=reg_list[1:])

    ax = df_plot.T.plot.barh(
        stacked=True, figsize=(18, 12), fontsize=17, color=colors_no_FR
    )

    plt.title("Part de chaque région dans les importations françaises", size=17)
    plt.tight_layout()
    plt.grid(visible=True)
    plt.legend(prop={"size": 15})
    plt.savefig("figures/commerce_parts_imports_secteur.png")
    # plt.show()


def visualisation_carbone(
    scenario, scenario_name, type_emissions="D_cba", saveghg=False, notallsectors=False
):
    """Plot the emissions associated with each sector for all ghg, and associated with the different ghg,
    for the chosen scenario. Can be used with every type of emissions account"""
    ghg_list = ["CO2", "CH4", "N2O", "SF6", "HFC", "PFC"]
    dict_fig_name = {
        "D_cba": "_empreinte_carbone_fr_importation",
        "D_pba": "_emissions_territoriales_fr",
        "D_imp": "_emissions_importees_fr",
        "D_exp": "_emissions_exportees_fr",
    }
    dict_plot_title = {
        "D_cba": "Empreinte carbone de la France",
        "D_pba": "Emissions territoriales françaises",
        "D_imp": "Emissions importées en France",
        "D_exp": "Emissions exportées vers la France",
    }
    d_ = pd.DataFrame(getattr(scenario.ghg_emissions_desag, type_emissions))
    if scenario_name == "Cont":
        # pour contrefactuel on affiche la barre de la reference aussi
        emissions_df = d_["FR"]
        em_df_ref = pd.DataFrame(
            getattr(reference.ghg_emissions_desag, type_emissions)
        )["FR"]
        sumonsectors = emissions_df.sum(axis=1)
        sumonsectors_ref = em_df_ref.sum(axis=1)
        total_ges_by_origin = sumonsectors.sum(level=0)
        total_ges_by_origin_ref = sumonsectors_ref.sum(level=0)
        liste_agg_ghg = []
        liste_agg_ghg_ref = []
        for ghg in ghg_list:
            liste_agg_ghg.append(
                sumonsectors.iloc[sumonsectors.index.get_level_values(1) == ghg].sum(
                    level=0
                )
            )
            liste_agg_ghg_ref.append(
                sumonsectors_ref.iloc[
                    sumonsectors_ref.index.get_level_values(1) == ghg
                ].sum(level=0)
            )
        dict_pour_plot = {
            ("Total", "cont"): total_ges_by_origin,
            ("Total", "ref"): total_ges_by_origin_ref,
            ("CO2", "cont"): liste_agg_ghg[0],
            ("CO2", "ref"): liste_agg_ghg_ref[0],
            ("CH4", "cont"): liste_agg_ghg[1],
            ("CH4", "ref"): liste_agg_ghg_ref[1],
            ("N2O", "cont"): liste_agg_ghg[2],
            ("N2O", "ref"): liste_agg_ghg_ref[2],
            ("SF6", "cont"): liste_agg_ghg[3],
            ("SF6", "ref"): liste_agg_ghg_ref[3],
            ("HFC", "cont"): liste_agg_ghg[4],
            ("HFC", "ref"): liste_agg_ghg_ref[4],
            ("PFC", "cont"): liste_agg_ghg[5],
            ("PFC", "ref"): liste_agg_ghg_ref[5],
        }
    else:
        emissions_df = d_["FR"]
        sumonsectors = emissions_df.sum(axis=1)
        total_ges_by_origin = sumonsectors.sum(level=0)
        liste_agg_ghg = []
        for ghg in ghg_list:
            liste_agg_ghg.append(
                sumonsectors.iloc[sumonsectors.index.get_level_values(1) == ghg].sum(
                    level=0
                )
            )
        dict_pour_plot = {
            "Total": total_ges_by_origin,
            "CO2": liste_agg_ghg[0],
            "CH4": liste_agg_ghg[1],
            "N2O": liste_agg_ghg[2],
            "SF6": liste_agg_ghg[3],
            "HFC": liste_agg_ghg[4],
            "PFC": liste_agg_ghg[5],
        }

    if notallsectors:
        sectors_list = ["Agriculture", "Energy", "Industry", "Composite"]
    else:
        sectors_list = list(reference.get_sectors())

    pour_plot = pd.DataFrame(data=dict_pour_plot, index=scenario.get_regions())
    if type_emissions == "D_cba":
        pour_plot.transpose().plot.bar(
            stacked=True, rot=45, figsize=(18, 12), fontsize=17, color=colors
        )
    elif type_emissions == "D_imp":
        pour_plot.drop("FR").transpose().plot.bar(
            stacked=True, rot=45, figsize=(18, 12), fontsize=17, color=colors_no_FR
        )
    plt.title(
        dict_plot_title[type_emissions] + " (scenario " + scenario_name + ")", size=17
    )
    plt.ylabel("MtCO2eq", size=17)
    plt.grid(visible=True)
    plt.legend(prop={"size": 25})
    plt.tight_layout()
    plt.savefig("figures/" + scenario_name + dict_fig_name[type_emissions] + ".png")
    plt.close()
    if saveghg:
        for ghg in ghg_list:
            df = pd.DataFrame(None, index=sectors_list, columns=scenario.get_regions())
            for reg in scenario.get_regions():
                df.loc[:, reg] = emissions_df.loc[(reg, ghg)]
            if type_emissions == "D_cba":
                ax = df.plot.barh(
                    stacked=True, figsize=(18, 12), fontsize=17, color=colors
                )
            elif type_emissions == "D_imp":
                ax = df.drop("FR").plot.barh(
                    stacked=True, figsize=(18, 12), fontsize=17, color=colors_no_FR
                )
            plt.grid(visible=True)
            plt.xlabel("MtCO2eq", size=17)
            plt.legend(prop={"size": 25})
            plt.title(
                dict_plot_title[type_emissions]
                + " de "
                + ghg
                + " par secteurs (scenario "
                + scenario_name
                + ")",
                size=17,
            )
            plt.savefig(
                "figures/"
                + scenario_name
                + "_french_"
                + ghg
                + dict_fig_name[type_emissions]
                + "_provenance_sectors"
            )
            plt.close()
    dict_sect_plot = {}
    for i in range(len(sectors_list)):
        sector = sectors_list[i]
        dict_sect_plot[sector] = {
            "ref": em_df_ref.sum(level=0)[sector],
            "cont": emissions_df.sum(level=0)[sector],
        }
    reform = {
        (outerKey, innerKey): values
        for outerKey, innerDict in dict_sect_plot.items()
        for innerKey, values in innerDict.items()
    }
    df_plot = pd.DataFrame(data=reform)
    if type_emissions == "D_cba":
        ax = df_plot.T.plot.barh(
            stacked=True, figsize=(18, 16), fontsize=17, color=colors
        )
    elif type_emissions == "D_imp":
        ax = df_plot.drop("FR").T.plot.barh(
            stacked=True, figsize=(18, 16), fontsize=17, color=colors_no_FR
        )
    plt.grid(visible=True)
    plt.xlabel("MtCO2eq", size=20)
    plt.legend(prop={"size": 25})
    plt.tight_layout()
    plt.title(
        dict_plot_title[type_emissions]
        + " de tous GES par secteurs (scenario "
        + scenario_name
        + ")",
        size=17,
    )
    plt.savefig(
        "figures/"
        + scenario_name
        + dict_fig_name[type_emissions]
        + "_provenance_sectors"
    )
    # plt.show()
    plt.close()


def visualisation_carbone_ref(
    scenario, scenario_name, type_emissions="D_cba", saveghg=False, notallsectors=False
):
    ghg_list = ["CO2", "CH4", "N2O", "SF6", "HFC", "PFC"]
    dict_fig_name = {
        "D_cba": "_empreinte_carbone_fr_importation",
        "D_pba": "_emissions_territoriales_fr",
        "D_imp": "_emissions_importees_fr",
        "D_exp": "_emissions_exportees_fr",
    }
    dict_plot_title = {
        "D_cba": "Empreinte carbone de la France",
        "D_pba": "Emissions territoriales françaises",
        "D_imp": "Emissions importées en France",
        "D_exp": "Emissions exportées vers la France",
    }
    d_ = pd.DataFrame(getattr(scenario.ghg_emissions_desag, type_emissions))
    emissions_df = d_["FR"]
    if type_emissions == "D_cba":
        reg_list = list(reference.get_regions())
    if type_emissions == "D_imp":
        reg_list = list(reference.get_regions())[1:]
        emissions_df = emissions_df.drop(["FR"])
    sumonsectors = emissions_df.sum(axis=1)
    total_ges_by_origin = sumonsectors.sum(level=0)
    liste_agg_ghg = []
    for ghg in ghg_list:
        liste_agg_ghg.append(
            sumonsectors.iloc[sumonsectors.index.get_level_values(1) == ghg].sum(
                level=0
            )
        )
    dict_pour_plot = {
        "Total": total_ges_by_origin,
        "CO2": liste_agg_ghg[0],
        "CH4": liste_agg_ghg[1],
        "N2O": liste_agg_ghg[2],
        "SF6": liste_agg_ghg[3],
        "HFC": liste_agg_ghg[4],
        "PFC": liste_agg_ghg[5],
    }
    if notallsectors:
        sectors_list = ["Agriculture", "Energy", "Industry", "Composite"]
    else:
        sectors_list = list(reference.get_sectors())

    pour_plot = pd.DataFrame(data=dict_pour_plot, index=reg_list)
    if type == "D_cba" or type == "D_pba":
        pour_plot.transpose().plot.bar(
            stacked=True, rot=45, figsize=(18, 12), fontsize=17, color=colors
        )
    else:
        pour_plot.transpose().plot.bar(
            stacked=True, rot=45, figsize=(18, 12), fontsize=17, color=colors_no_FR
        )
    plt.title(
        dict_plot_title[type_emissions] + " (scenario " + scenario_name + ")", size=17
    )
    plt.ylabel("MtCO2eq", size=17)
    plt.legend(prop={"size": 25})
    plt.grid(visible=True)
    plt.savefig("figures/" + scenario_name + dict_fig_name[type_emissions] + ".png")
    plt.close()

    dict_sect_plot = {}
    for i in range(len(sectors_list)):
        sector = sectors_list[i]
        dict_sect_plot[sector] = [
            emissions_df.sum(level=0)[sector].loc[r] for r in reg_list
        ]

    df_plot = pd.DataFrame(data=dict_sect_plot, index=reg_list)
    if type == "D_cba" or type == "D_pba":
        ax = df_plot.T.plot.barh(
            stacked=True, figsize=(17, 16), fontsize=17, rot=45, color=colors
        )
    else:
        ax = df_plot.T.plot.barh(
            stacked=True, figsize=(17, 16), fontsize=17, rot=45, color=colors_no_FR
        )
    plt.grid(visible=True)
    plt.xlabel("MtCO2eq", size=17)
    plt.legend(prop={"size": 25})
    plt.title(
        dict_plot_title[type_emissions]
        + " de tous GES par secteurs (scenario "
        + scenario_name
        + ")",
        size=17,
    )
    plt.savefig(
        "figures/"
        + scenario_name
        + dict_fig_name[type_emissions]
        + "_provenance_sectors"
    )
    # plt.show()
    plt.close()


def heat_S(type, notallsectors=False):
    if notallsectors:
        sectors_list = ["Agriculture", "Energy", "Industry", "Composite"]
    else:
        sectors_list = list(reference.get_sectors())
    S = reference.ghg_emissions_desag.S.sum()
    M = reference.ghg_emissions_desag.M.sum()
    sec_reg = []
    for reg in reg_list:
        in_reg = []
        for sector in sectors_list:
            if type == "consommation":
                in_reg.append(M[reg, sector])
            if type == "production":
                in_reg.append(S[reg, sector])
        sec_reg.append(in_reg)
    df = pd.DataFrame(data=sec_reg, columns=sectors_list, index=reg_list).T
    df_n = df.div(df.max(axis=1), axis=0) * 100
    if type == "consommation":
        title = "Contenu carbone du bien importé"
    if type == "production":
        title = "Intensité carbone de la production"
    fig, ax = plt.subplots()
    sns.heatmap(
        df_n, cmap="coolwarm", ax=ax, linewidths=1, linecolor="black"
    ).set_title(title, size=13)
    plt.yticks(size=11)
    plt.xticks(size=11)
    fig.tight_layout()
    plt.savefig("figures/heatmap_intensite_" + type)
    # plt.show()
    return


###########################
#%% VISUALISE
###########################

# reference analysis
for type in ["D_cba", "D_imp"]:
    print(type)
    visualisation_carbone_ref(reference, "Ref", type, saveghg=False)


##reagreate from 17 to 4 sectors :
Tools.reag_D_sectors(reference, inplace=True, type="D_cba")
Tools.reag_D_sectors(counterfactual, inplace=True, type="D_cba")
Tools.reag_D_sectors(reference, inplace=True, type="D_imp")
Tools.reag_D_sectors(counterfactual, inplace=True, type="D_imp")

##reagreate from 11 to 5 regions :
Tools.reag_D_regions(
    reference,
    inplace=True,
    type="D_imp",
    dict_reag_regions=dict_regions,
    list_sec=["Agriculture", "Energy", "Industry", "Composite"],
)
Tools.reag_D_regions(
    counterfactual,
    inplace=True,
    type="D_imp",
    dict_reag_regions=dict_regions,
    list_sec=["Agriculture", "Energy", "Industry", "Composite"],
)

# whole static comparative analysis

# compare reference and counterfactual
for type in ["D_cba", "D_imp"]:
    print(type)
    visualisation_carbone(
        counterfactual, "Cont", type, saveghg=False, notallsectors=True
    )
vision_commerce()


def delta_CF(ref, contr):
    """Compare the carbon footprints of the two scenarios, without the F_Y component."""
    ref_dcba = pd.DataFrame(ref.ghg_emissions_desag.D_cba)
    con_dcba = pd.DataFrame(contr.ghg_emissions_desag.D_cba)
    cf_ref = ref_dcba["FR"].sum(axis=1).sum(level=0)
    cf_con = con_dcba["FR"].sum(axis=1).sum(level=0)
    return (
        100 * (cf_con / cf_ref - 1),
        100 * (cf_con.sum() / cf_ref.sum() - 1),
        cf_ref,
        cf_con,
    )


res = delta_CF(reference, counterfactual)
print("Variation EC française par provenance")
print(res[0])
print(res[1])
print("Empreinte carbone référence :", res[2].sum(), "MtCO2eq")
print("Empreinte carbone contrefactuel :", res[3].sum(), "MtCO2eq")
print(
    "Variation relative EC incomplète :",
    np.round(100 * (res[3].sum() - res[2].sum()) / res[2].sum(), 2),
    "%",
)


def compa_monetaire(ref, contr):
    # unité = M€
    return contr.x - ref.x


print("Variation de richesse de la transformation")
print(compa_monetaire(reference, counterfactual).sum(level=0).sum())

# Compare the global carbon footprints of the two scenarios, including the F_Y component
EC_ref = reference.ghg_emissions_desag.D_cba_reg.copy()
EC_cont = pd.DataFrame(
    counterfactual.ghg_emissions_desag.D_cba.copy()
    .sum(level="region", axis=1)
    .sum(level="stressor")
    + counterfactual.ghg_emissions_desag.F_Y.groupby(
        axis=1, level="region", sort=False
    ).sum()
)
print("Véritable EC reference:", EC_ref["FR"].sum(), "MtCO2eq")
print("Véritable EC contre-factuelle:", EC_cont["FR"].sum(), "MtCO2eq")
print(
    "Variation relative EC réelle :",
    np.round(100 * (EC_cont["FR"].sum() - EC_ref["FR"].sum()) / EC_ref["FR"].sum(), 2),
    "%",
)
