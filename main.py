""" Python main script of MatMat trade module
	"""

###########################
#%% IMPORT MODULES
###########################
# general

from typing import Callable, Dict, List, Optional, Tuple
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# scientific
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymrio
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

###########################
#%% GENERIC FUNCTIONS FOR SCENARIOS
###########################


def moves_from_sorted_index_by_sector(
    sector: str, regions_index: List[int], reloc: bool
) -> Tuple[np.array]:
    """Allocate french importations for a sector in the order given by region_index

    Args:
        sector (str): name of a product (or industry)
        regions_index (List[int]): list of ordered region indices
        reloc (bool): True if relocation is allowed

    Returns:
        Tuple[np.array]: tuple with 2 elements :
            - 2D-array of imports of 'sector' from regions (rows) for french intermediary demands (columns)
            - 2D-array of imports of 'sector' from regions (rows) for french final demands (columns)
    """

    if reloc:
        regions = reference.get_regions()
    else:
        regions = reference.get_regions()[1:]  # remove FR

    sectors = reference.get_sectors()
    demcats = reference.get_Y_categories()

    import_demfr = (
        reference.Z["FR"].drop(["FR"]).sum(axis=1).sum(level=1).loc[sector]
        + reference.Y["FR"].drop(["FR"]).sum(axis=1).sum(level=1).loc[sector]
    )

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
    totalinterfromsectorFR = [
        reference.Z["FR"].loc["FR"][sec].loc[sector] for sec in sectors
    ]
    totalfinalfromsectorFR = [
        reference.Y["FR"].loc["FR"][dem].loc[sector] for dem in demcats
    ]

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
            reference.Z.drop(columns=reg).sum(axis=1).loc[(reg, sector)]
        ) + (reference.Y.drop(columns=reg).sum(axis=1).loc[(reg, sector)])

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
    sorting_rule_by_sector: Callable[[str, bool], List[int]], reloc: bool = False
) -> Dict:
    """Allocate french importations for all sectors, sorting the regions with a given rule for each sector

    Args:
        sorting_rule_by_sector (Callable[str, bool], List[int]]): given a sector name and the reloc value, returns a sorted list of regions' indices
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Dict: dictionnary associating to each sector a dictionnary with :
            - parts_sec : 2D-array of imports of 'sector' from regions (rows) for french intermediary demands (columns)
            - parts_dem : 2D-array of imports of 'sector' from regions (rows) for french final demands (columns)
            - sort : array of indices of regions ascendantly sorted by carbon content
            - reloc : True if relocation is allowed
    """

    sectors_list = reference.get_sectors()
    moves = {}
    for sector in sectors_list:
        regions_index = sorting_rule_by_sector(sector, reloc)
        parts_sec, parts_dem = moves_from_sorted_index_by_sector(
            sector, regions_index, reloc
        )
        moves[sector] = {
            "parts_sec": parts_sec,
            "parts_dem": parts_dem,
            "sort": regions_index,
            "reloc": reloc,
        }
    return moves


###########################
#%% BEST AND WORST SCENARIOS
###########################


def sort_by_content(sector: str, reloc: bool = False) -> np.array:
    """Ascendantly sort all regions by carbon content of a sector

    Args:
        sector (str): name of a product (or industry)
        reloc (bool, optional): True if relocation is allowed. Defaults to False.


    Returns:
        np.array: array of indices of regions sorted by carbon content
    """

    M = reference.ghg_emissions_desag.M.sum(axis=0)
    regions_index = np.argsort(M[:, sector].values[1 - reloc :])
    return regions_index


def scenar_best(reloc: bool = False) -> Dict:
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


def scenar_worst(reloc: bool = False) -> Dict:
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


###########################
#%% Preference for Europe
###########################


def scenar_pref(allies: List[str], reloc: bool = False) -> Dict:
    """Find imports reallocation in order to trade as much as possible with the allies

    Args:
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
        regions = reference.get_regions()
        if not "FR" in allies:
            allies += ["FR"]
    else:
        regions = reference.get_regions()[1:]  # remove FR

    moves = {}

    sectors = reference.get_sectors()
    demcats = reference.get_Y_categories()

    for sector in sectors:

        parts_sectors = {reg: [] for reg in regions}
        parts_demcats = {reg: [] for reg in regions}

        ## overall trade related with sector
        sector_exports_Z = reference.Z.loc[(regions, sector), :].sum(axis=0, level=0)
        sector_exports_Y = reference.Y.loc[(regions, sector), :].sum(axis=0, level=0)
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


def scenar_pref_eu(reloc: bool = False) -> Dict:
    """Find imports reallocation that prioritize trade with European Union


    Args:
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Dict: dictionnary associating to each sector a dictionnary with :
            - parts_sec : 2D-array of imports of 'sector' from regions (rows) for french intermediary demands (columns)
            - parts_dem : 2D-array of imports of 'sector' from regions (rows) for french final demands (columns)
            - sort : array of indices of regions
            - reloc : True if relocation is allowed
    """

    return scenar_pref(["EU"], reloc)


###########################
#%% Trade war with China
###########################


def scenar_tradewar(opponents: List[str], reloc: bool = False) -> Dict:
    """Find imports reallocation in order to exclude a list of opponents as much as possible

    Args:
        opponents (List[str]): list of regions' names
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Dict: dictionnary associating to each sector a dictionnary with :
            - parts_sec : 2D-array of imports of 'sector' from regions (rows) for french intermediary demands (columns)
            - parts_dem : 2D-array of imports of 'sector' from regions (rows) for french final demands (columns)
            - sort : array of indices of regions
            - reloc : True if relocation is allowed
    """

    allies = list(set(reference.get_regions()) - set(opponents))
    if not reloc:
        allies.remove("FR")
    return scenar_pref(allies, reloc)


def scenar_tradewar_china(reloc: bool = False) -> Dict:
    """Find imports reallocation that prevents trade with China as much as possible


    Args:
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Dict: dictionnary associating to each sector a dictionnary with :
            - parts_sec : 2D-array of imports of 'sector' from regions (rows) for french intermediary demands (columns)
            - parts_dem : 2D-array of imports of 'sector' from regions (rows) for french final demands (columns)
            - sort : array of indices of regions
            - reloc : True if relocation is allowed
    """

    return scenar_tradewar(["China, RoW Asia and Pacific"], reloc)


###########################
#%% PLOT FIGURES
###########################


# Carbon footprints


def plot_carbon_footprint(
    region: str = "FR", display: bool = True, title: Optional[str] = None
) -> None:
    """Plots region's carbon footprint (D_pba-D_exp+D_imp+F_Y)

    Args:
        region (str, optional): region name. Defaults to "FR".
        display (bool, optional): True to display the figure. Defaults ro True.
        title (Optional[str], optional): title of the figure. Defaults to None.
    """
    carbon_footprint = pd.DataFrame(
        {
            "Exportées": [-reference.ghg_emissions_desag.D_exp[region].sum().sum()],
            "Production": [reference.ghg_emissions_desag.D_pba[region].sum().sum()],
            "Importées": [reference.ghg_emissions_desag.D_imp[region].sum().sum()],
            "Conso finale": [reference.ghg_emissions_desag.F_Y[region].sum().sum()],
        }
    )
    carbon_footprint.plot.barh(stacked=True, fontsize=17, figsize=(10, 5), rot=0)

    if title is None:
        title = f"Empreinte carbone de la région {region}"
    plt.title(title, size=17)
    plt.xlabel("MtCO2eq", size=15)
    plt.tight_layout()
    plt.grid(visible=True)
    plt.legend(prop={"size": 17})

    plt.savefig(f"figures/empreinte_carbone_{region}.png")
    if display:
        plt.show()


def plot_carbon_footprint_FR(display: bool = True) -> None:
    """Plots french carbon footprint (D_pba-D_exp+D_imp+F_Y)

    Args:
        display (bool, optional): True to display the figure. Defaults ro True.
    """
    plot_carbon_footprint("FR", display, "Empreinte carbone de la France")


# Dictionary to reaggreagate account matrices with less regions for the sake of better visibility
DICT_REGIONS = {
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

DICT_SCENARIOS = {
    "best": {"sector_moves": scenar_best(True), "shock_function": Tools.shockv2},
    "worst": {"sector_moves": scenar_worst(True), "shock_function": Tools.shockv2},
    "pref_eu": {
        "sector_moves": scenar_pref_eu(True),
        "shock_function": Tools.shockv3,
    },
    "tradewar_china": {
        "sector_moves": scenar_tradewar_china(True),
        "shock_function": Tools.shockv3,
    },
}

# Scenarios comparison


def create_counterfactual_base() -> pymrio.IOSystem:
    """Compute a counterfactual pymrio model from the reference

    Returns:
        pymrio.IOSystem: pymrio model
    """
    counterfactual_base = reference.copy()
    counterfactual_base.remove_extension("ghg_emissions_desag")
    return counterfactual_base


def compare_scenarios(reloc=False):
    """Draw figures to compare the carbon footprints associated with the different scenarios"""
    print("compare_scenarios")

    regions = list(DICT_REGIONS.keys())
    scenarios = list(DICT_SCENARIOS.keys()) + ["reference"]

    ghg_all_scen = pd.DataFrame(
        0.0,
        index=regions,
        columns=scenarios,
    )
    trade_all_scen = pd.DataFrame(
        0.0,
        index=regions,
        columns=scenarios,
    )

    for scenar in scenarios:

        print(scenar)

        if scenar == "reference":
            counterfactual = reference.copy()
        else:
            counterfactual = Tools.compute_counterfactual(
                create_counterfactual_base(),
                DICT_SCENARIOS[scenar],
            )
            counterfactual.ghg_emissions_desag = Tools.recal_extensions_per_region(
                counterfactual,
                "ghg_emissions",
            )

        for reg in regions:
            ghg_all_scen.loc[reg, scenar] = (
                (counterfactual.ghg_emissions_desag.D_cba["FR"].sum(axis=1))
                .sum(level=0)[DICT_REGIONS[reg]]
                .sum()
            )
        ghg_all_scen.loc["FR", scenar] += (
            counterfactual.ghg_emissions_desag.F_Y["FR"].sum().sum()
        )
        for reg in regions:
            trade_all_scen.loc[reg, scenar] = (
                (
                    counterfactual.Y["FR"].sum(axis=1)
                    + counterfactual.Z["FR"].sum(axis=1)
                )
                .sum(level=0)[DICT_REGIONS[reg]]
                .sum()
            )

    print(trade_all_scen)
    print(ghg_all_scen)

    ghg_all_scen.T.plot.bar(
        stacked=True, fontsize=17, figsize=(12, 8), rot=0, color=colors[:5]
    )

    plt.title("Empreinte carbone de la France", size=17)
    plt.ylabel("MtCO2eq", size=15)
    plt.tight_layout()
    plt.grid(visible=True)
    plt.legend(prop={"size": 15})
    plt.savefig("figures/comparaison_5_scenarios.png")

    trade_all_scen.T.plot.bar(
        stacked=True, fontsize=17, figsize=(12, 8), rot=0, color=colors[:5]
    )

    plt.title("Provenance de la consommation de la France", size=17)
    plt.ylabel("x 1000 milliards d'€", size=15)
    plt.tight_layout()
    plt.grid(visible=True)
    plt.legend(prop={"size": 15})
    plt.savefig("figures/comparaison_6_scenarios.png")

    _, axes = plt.subplots(nrows=1, ncols=2)
    ghg_all_scen.drop("FR").T.plot.bar(
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
    trade_all_scen.drop("FR").T.plot.bar(
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
    axes[1].legend(prop={"size": 15})
    plt.tight_layout()
    plt.savefig("figures/comparaison_3_scenarios_bornes.png")
    plt.show()


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
    create_counterfactual_base(),
    DICT_SCENARIOS[chosen_scenario],
    reference.get_Y_categories(),
    reference.get_regions(),
)


# calculate counterfactual(s) system

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
    if type_emissions == "D_cba" or type_emissions == "D_pba":
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
    if type_emissions == "D_cba" or type_emissions == "D_pba":
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


def heat_S(activity, notallsectors=False):
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
            if activity == "consommation":
                in_reg.append(M[reg, sector])
            if activity == "production":
                in_reg.append(S[reg, sector])
        sec_reg.append(in_reg)
    df = pd.DataFrame(data=sec_reg, columns=sectors_list, index=reg_list).T
    df_n = df.div(df.max(axis=1), axis=0) * 100
    if activity == "consommation":
        title = "Contenu carbone du bien importé"
    if activity == "production":
        title = "Intensité carbone de la production"
    fig, ax = plt.subplots()
    sns.heatmap(
        df_n, cmap="coolwarm", ax=ax, linewidths=1, linecolor="black"
    ).set_title(title, size=13)
    plt.yticks(size=11)
    plt.xticks(size=11)
    fig.tight_layout()
    plt.savefig("figures/heatmap_intensite_" + activity)
    # plt.show()
    return


###########################
#%% VISUALISE
###########################

# reference analysis
for type_emissions in ["D_cba", "D_imp"]:
    print(type_emissions)
    visualisation_carbone_ref(reference, "Ref", type_emissions, saveghg=False)


##reagreate from 17 to 4 sectors :
Tools.reag_D_sectors(reference, inplace=True, type_emissions="D_cba")
Tools.reag_D_sectors(counterfactual, inplace=True, type_emissions="D_cba")
Tools.reag_D_sectors(reference, inplace=True, type_emissions="D_imp")
Tools.reag_D_sectors(counterfactual, inplace=True, type_emissions="D_imp")

##reagreate from 11 to 5 regions :
Tools.reag_D_regions(
    reference,
    inplace=True,
    type_emissions="D_imp",
    dict_reag_regions=DICT_REGIONS,
    list_sec=["Agriculture", "Energy", "Industry", "Composite"],
)
Tools.reag_D_regions(
    counterfactual,
    inplace=True,
    type_emissions="D_imp",
    dict_reag_regions=DICT_REGIONS,
    list_sec=["Agriculture", "Energy", "Industry", "Composite"],
)

# whole static comparative analysis

# compare reference and counterfactual
for type_emissions in ["D_cba", "D_imp"]:
    print(type_emissions)
    visualisation_carbone(
        counterfactual, "Cont", type_emissions, saveghg=False, notallsectors=True
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
