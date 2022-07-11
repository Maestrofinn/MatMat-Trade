""" Python main script of MatMat trade module
	"""

###########################
#%% IMPORT MODULES
###########################
# general

from typing import Callable, Dict, List, Optional, Tuple
from unidecode import unidecode
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# scientific
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_theme()

# local folder
from local_paths import data_dir, figures_dir, output_dir

# local library
from utils import *

###########################
#%% SETTINGS
###########################
# creating colormap for 11 regions and for 10 regions for plotting
import matplotlib.colors as mpl_col

colors = list(plt.cm.tab10(np.arange(10))) + ["gold"]
colors_no_FR = colors[1:]

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
reference = build_reference(calib, data_dir, base_year, system, agg_name)


###########################
#%%CALCULATIONS
###########################
# Calculate reference system
reference.calc_all()
reference.ghg_emissions_desag = recal_extensions_per_region(reference, "ghg_emissions")

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

    plt.savefig(figures_dir / f"empreinte_carbone_{region}.png")
    if display:
        plt.show()


def plot_carbon_footprint_FR(display: bool = True) -> None:
    """Plots french carbon footprint (D_pba-D_exp+D_imp+F_Y)

    Args:
        display (bool, optional): True to display the figure. Defaults to True.
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

GHG_LIST = list(reference.ghg_emissions_desag.get_index())


def get_dict_scenarios(reloc: bool = False) -> Dict:
    """Get the different scenarios

    Args:
        reloc (bool, optional): True if relocation is allowed. Defaults to False.

    Returns:
        Dict: dictionnary associating the scenarios' names to their parameters
    """

    return {
        "best": {
            "sector_moves": scenar_best(reloc=reloc),
            "shock_function": shockv2,
        },
        "worst": {
            "sector_moves": scenar_worst(reloc=reloc),
            "shock_function": shockv2,
        },
        "pref_eu": {
            "sector_moves": scenar_pref_eu(reloc=reloc),
            "shock_function": shockv3,
        },
        "tradewar_china": {
            "sector_moves": scenar_tradewar_china(reloc=reloc),
            "shock_function": shockv3,
        },
    }


# Scenarios comparison


def compare_scenarios(
    reloc: bool = False, verbose: bool = False, display: bool = True
) -> None:
    """Plot the carbon footprints and the imports associated with the different scenarios

    Args:
        reloc (bool, optional): True if relocation is allowed. Defaults to False.
        verbose (bool, optional): True to print infos. Defaults to False.
        display (bool, optional): True to display the figures. Defaults to True.
    """

    DICT_SCENARIOS = get_dict_scenarios(reloc=reloc)

    if verbose:
        print("Comparing scenarios...")

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

        if verbose:
            print(f"Processing scenario : {scenar}")

        if scenar == "reference":
            counterfactual = reference.copy()
        else:
            counterfactual = compute_counterfactual(
                reference,
                DICT_SCENARIOS[scenar],
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

    ghg_all_scen.T.plot.bar(
        stacked=True, fontsize=17, figsize=(12, 8), rot=0, color=colors[:5]
    )
    plt.title("Empreinte carbone de la France", size=17)
    plt.ylabel("MtCO2eq", size=15)
    plt.tight_layout()
    plt.grid(visible=True)
    plt.legend(prop={"size": 15})
    plt.savefig(figures_dir / "compare_scenarios_ghg.png")

    trade_all_scen.T.plot.bar(
        stacked=True, fontsize=17, figsize=(12, 8), rot=0, color=colors[:5]
    )
    plt.title("Provenance de la consommation de la France", size=17)
    plt.ylabel("x 1000 milliards d'€", size=15)
    plt.tight_layout()
    plt.grid(visible=True)
    plt.legend(prop={"size": 15})
    plt.savefig(figures_dir / "compare_scenarios_trade.png")

    _, axes = plt.subplots(nrows=1, ncols=2)
    ghg_all_scen.drop("FR").T.plot.bar(
        ax=axes[0],
        stacked=True,
        fontsize=17,
        figsize=(12, 8),
        rot=0,
        color=colors_no_FR[: len(regions)],
    )
    axes[0].set_title("Emissions de GES importées par la France", size=17)
    axes[0].legend(prop={"size": 15})
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("MtCO2eq", size=15)
    trade_all_scen.drop("FR").T.plot.bar(
        ax=axes[1],
        stacked=True,
        fontsize=17,
        figsize=(12, 8),
        rot=0,
        legend=False,
        color=colors_no_FR[: len(regions)],
    )
    axes[1].set_title("Importations françaises", size=17)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_ylabel("M€", size=15)
    axes[1].legend(prop={"size": 15})
    plt.tight_layout()
    plt.savefig(figures_dir / "compare_scenarios_imports.png")

    if display:
        plt.show()


# ###########################
#%% DEFINE FUNCTIONS FOR VISUALISATION
# ###########################


def plot_df_synthesis(
    reference_df: pd.Series,
    counterfactual_df: pd.Series,
    account_name: str,
    account_unit: str,
    scenario_name: str,
    sectors: List[str] = None,
    display: bool = True,
) -> None:
    """Plot some figures for a given scenario

    Args:
        reference_matrix (pd.DataFrame): series with rows multiindexed by (region, sector) associated to the reference
        couterfactual_matrix (pd.DataFrame): series with rows multiindexed by (region, sector) associated to the counterfactual
        account_name (str): name of the account considered in french, for display purpose (eg: "importations françaises", "empreinte carbone française")
        account_unit (str): account unit for display purpose (must be the same in both dataframes)
        scenario_name(str): name of the scenario (used to save the figures)
        sectors (List[str], optional): sublist of sectors. Defaults to None.
        display (bool, optional): True to display the figures. Defaults to True.
    """

    regions = list(reference_df.index.get_level_values(level=0).drop_duplicates())

    account_name = (
        account_name[0].upper() + account_name[1:]
    )  # didn't use .capitalize() in order to preserve capital letters in the middle
    account_name_file = unidecode(account_name.lower().replace(" ", "_"))
    current_dir = figures_dir / (scenario_name + "__" + account_name_file)

    if not os.path.isdir(current_dir):
        os.mkdir(current_dir)  # can overwrite existing files

    # plot reference importations
    ref_conso_by_sector_FR = reference_df
    ref_imports_by_region_FR = ref_conso_by_sector_FR.drop("FR").sum(level=0)

    ref_imports_by_region_FR.T.plot.barh(
        stacked=True, fontsize=17, color=colors_no_FR, figsize=(12, 5)
    )
    plt.title(f"{account_name} par région (référence)", size=17)
    plt.xlabel(account_unit)
    plt.tight_layout()
    plt.grid(visible=True)
    plt.savefig(current_dir / "reference.png")
    if display:
        plt.show()

    # plot counterfactual importations
    scen_conso_by_sector_FR = counterfactual_df
    scen_imports_by_region_FR = scen_conso_by_sector_FR.drop("FR").sum(level=0)

    scen_imports_by_region_FR.T.plot.barh(
        stacked=True, fontsize=17, color=colors_no_FR, figsize=(12, 5)
    )
    plt.title(f"{account_name} par région (scénario {scenario_name})", size=17)
    plt.xlabel(account_unit)
    plt.tight_layout()
    plt.grid(visible=True)
    plt.savefig(current_dir / f"{scenario_name}.png")
    if display:
        plt.show()

    # compare counterfactual and reference importations
    compare_imports_by_region_FR = pd.DataFrame(
        {
            "Référence": ref_imports_by_region_FR,
            f"Scénario {scenario_name}": scen_imports_by_region_FR,
        }
    )
    compare_imports_by_region_FR.T.plot.barh(
        stacked=True, fontsize=17, figsize=(12, 8), color=colors_no_FR
    )
    plt.title(f"{account_name} (comparaison)", size=17)
    plt.xlabel(account_unit)
    plt.tight_layout()
    plt.grid(visible=True)
    plt.legend(prop={"size": 12})
    plt.savefig(current_dir / f"comparison_by_region.png")
    if display:
        plt.show()

    # compare each region for each importation sector for the reference and the counterfactual

    def grouped_and_stacked_plot(
        df_ref: pd.DataFrame,
        df_scen: pd.DataFrame,
        percent_x_scale: bool,
        plot_title: str,
        plot_filename: str,
    ) -> None:
        """Nested function. Plot a grouped stacked horizontal bar plot.

        Args:
            df_ref (pd.DataFrame): series with rows multiindexed by (region, sector) associated to the reference
            df_scen (pd.DataFrame): series with rows multiindexed by (region, sector) associated to the counterfactual
            percent_scale (bool): True if the x_axis should be labelled with percents (otherwise labelled with values)
            plot_title (str): title of the figure, in french for display purpose
            plot_filename (str): to save the figure
        """
        df_to_display = pd.DataFrame(
            columns=regions[1:],
            index=pd.MultiIndex.from_arrays(
                [
                    sum([2 * [sec] for sec in sectors], []),
                    len(sectors) * ["Référence", f"Scénario {scenario_name}"],
                ],
                names=("sector", "scenario"),
            ),
        )
        for sec in sectors:
            df_to_display.loc[(sec, "Référence"), :] = df_ref.loc[(slice(None), sec)]
            df_to_display.loc[(sec, f"Scénario {scenario_name}"), :] = df_scen.loc[
                (slice(None), sec)
            ]
        fig, axes = plt.subplots(
            nrows=len(sectors), ncols=1, sharex=True, figsize=(10, 10)
        )
        graph = dict(zip(df_to_display.index.levels[0], axes))
        for ax in axes:
            ax.yaxis.tick_right()
            ax.tick_params(axis="y", which="both", rotation=0)
            if percent_x_scale:
                ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        list(
            map(
                lambda x: df_to_display.xs(x)
                .plot(
                    kind="barh",
                    stacked="True",
                    ax=graph[x],
                    legend=False,
                    color=colors_no_FR,
                )
                .set_ylabel(
                    x,
                    rotation=0,
                    size=15,
                    horizontalalignment="right",
                    verticalalignment="center",
                ),
                graph,
            )
        )
        fig.subplots_adjust(wspace=0)
        fig.suptitle(
            plot_title,
            size=17,
        )
        plt.tight_layout()
        if not percent_x_scale:
            plt.xlabel(account_unit)
        plt.legend(ncol=3, loc="lower left", bbox_to_anchor=(-0.35, -4.5))
        plt.savefig(current_dir / plot_filename)
        if display:
            plt.show()

    df_ref_parts = (
        (
            ref_conso_by_sector_FR.drop("FR")
            / ref_conso_by_sector_FR.drop("FR").sum(level=1)
        )
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    df_scen_parts = (
        (
            scen_conso_by_sector_FR.drop("FR")
            / scen_conso_by_sector_FR.drop("FR").sum(level=1)
        )
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    grouped_and_stacked_plot(
        df_ref_parts,
        df_scen_parts,
        True,
        f"{account_name} : comparaison par secteur de la part de chaque région",
        f"comparison_parts_region_sector.png",
    )

    df_ref_values = (
        ref_conso_by_sector_FR.drop("FR").replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    df_scen_values = (
        scen_conso_by_sector_FR.drop("FR").replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    grouped_and_stacked_plot(
        df_ref_values,
        df_scen_values,
        False,
        f"{account_name} : comparaison par secteur de chaque région",
        f"comparison_values_region_sector.png",
    )


def plot_trade_synthesis(
    scenario_parameters: Dict,
    scenario_name: str,
    sectors: List[str] = None,
    notallsectors: bool = False,
    display: bool = True,
) -> None:
    """Plot the french importations for a given scenario

    Args:
        scenario_parameters (Dict): contains the changes for each sector ('sector_moves') and the shock function to apply ('shock_function')
        scenario_name(str): name of the scenario (used to save the figures)
        sectors (List[str], optional): sublist of sectors. Defaults to None.
        notallsectors (bool): True to set the sectors as ['Agriculture','Energy','Industry','Composite']
        display (bool, optional): True to display the figures. Defaults to True.
    """
    counterfactual = compute_counterfactual(reference, scenario_parameters)
    if notallsectors:
        sectors = ["Agriculture", "Energy", "Industry", "Composite"]
    elif sectors is None:
        sectors = list(reference.get_sectors())

    reference_trade = reference.Y["FR"].sum(axis=1) + reference.Z["FR"].sum(axis=1)
    counterfactual_trade = counterfactual.Y["FR"].sum(axis=1) + counterfactual.Z[
        "FR"
    ].sum(axis=1)

    plot_df_synthesis(
        reference_trade,
        counterfactual_trade,
        "importations françaises",
        "M€",
        scenario_name,
        sectors,
        display,
    )


def plot_co2eq_synthesis(
    scenario_parameters: Dict,
    scenario_name: str,
    sectors: List[str] = None,
    notallsectors: bool = False,
    display: bool = True,
) -> None:
    """Plot the french emissions per sector for a given scenario

    Args:
        scenario_parameters (Dict): contains the changes for each sector ('sector_moves') and the shock function to apply ('shock_function')
        scenario_name(str): name of the scenario (used to save the figures)
        sectors (List[str], optional): sublist of sectors. Defaults to None.
        display (bool, optional): True to display the figures. Defaults to True.
    """
    counterfactual = compute_counterfactual(reference, scenario_parameters)
    if notallsectors:
        sectors = ["Agriculture", "Energy", "Industry", "Composite"]
    elif sectors is None:
        sectors = list(reference.get_sectors())

    emissions_types = {
        "D_cba": "empreinte carbone de la France",
        "D_pba": "émissions territoriales de la France",
        "D_imp": "émissions importées par la France",
        "D_exp": "émissions exportées par la France",
    }

    for name, description in emissions_types.items():

        reference_trade = (
            getattr(reference.ghg_emissions_desag, name)["FR"].sum(level=0).stack()
        )
        counterfactual_trade = (
            getattr(counterfactual.ghg_emissions_desag, name)["FR"].sum(level=0).stack()
        )

        plot_df_synthesis(
            reference_trade,
            counterfactual_trade,
            description,
            "MtCO2eq",
            scenario_name,
            sectors,
            display,
        )


def plot_ghg_synthesis(
    scenario_parameters: Dict,
    scenario_name: str,
    sectors: List[str] = None,
    display: bool = True,
) -> None:
    """Plot the french emissions per GHG for a given scenario

    Args:
        scenario_parameters (Dict): contains the changes for each sector ('sector_moves') and the shock function to apply ('shock_function')
        scenario_name(str): name of the scenario (used to save the figures)
        sectors (List[str], optional): sublist of sectors. Defaults to None.
        display (bool, optional): True to display the figures. Defaults to True.
    """
    counterfactual = compute_counterfactual(reference, scenario_parameters)
    if sectors is None:
        sectors = list(reference.ghg_emissions_desag.get_index())

    emissions_types = {
        "D_cba": "empreinte en GES de la France",
        "D_pba": "émissions territoriales de GES de la France",
        "D_imp": "émissions de GES importées par la France",
        "D_exp": "émissions de GES exportées par la France",
    }

    for name, description in emissions_types.items():

        reference_trade = getattr(reference.ghg_emissions_desag, name)["FR"].sum(axis=1)
        counterfactual_trade = getattr(counterfactual.ghg_emissions_desag, name)[
            "FR"
        ].sum(axis=1)

        plot_df_synthesis(
            reference_trade,
            counterfactual_trade,
            description,
            "MtCO2eq",
            scenario_name,
            sectors,
            display,
        )


def ghg_content_heatmap(
    reference: pymrio.IOSystem,
    prod: bool = False,
    sectors: List[str] = None,
    display: bool = True,
) -> None:
    """Plot the GHG contents each sector for each region in a heatmap

    Args:
        reference (pymrio.IOSystem): MRIO model
        prod (bool, optional): True to focus on production values, otherwise focus on consumption values. Defaults to False.
        sectors (List[str], optional): sublist of sectors. Defaults to None.
        display (bool, optional): True to display the figures. Defaults to True.
    """
    if sectors is None:
        sectors = list(reference.get_sectors())
    regions = reference.get_regions()
    if prod:
        title = "Intensité carbone de la production"
        activity = "production"
        to_display = reference.ghg_emissions_desag.S.sum().unstack().T
    else:
        title = "Contenu carbone du bien importé"
        activity = "consumption"
        to_display = reference.ghg_emissions_desag.M.sum().unstack().T
    to_display = to_display.reindex(sectors)[regions]  # sort rows and columns
    to_display = 100 * to_display.div(
        to_display.max(axis=1), axis=0
    )  # compute relative values
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        to_display,
        cmap="coolwarm",
        ax=ax,
        linewidths=1,
        linecolor="black",
        cbar_kws={"format": "%.0f%%"},
    ).set_title(title, size=13)
    plt.yticks(size=11)
    plt.xticks(size=11)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    fig.tight_layout()
    plt.savefig(figures_dir / ("ghg_content_heatmap_" + activity))
    if display:
        plt.plot()


'''
###########################
#%% VISUALISE
###########################

# reference analysis
for type_emissions in ["D_cba", "D_imp"]:
    print(type_emissions)
    visualisation_carbone_ref(reference, "Ref", type_emissions, saveghg=False)


##reagreate from 17 to 4 sectors :
reag_D_sectors(reference, inplace=True, type_emissions="D_cba")
reag_D_sectors(counterfactual, inplace=True, type_emissions="D_cba")
reag_D_sectors(reference, inplace=True, type_emissions="D_imp")
reag_D_sectors(counterfactual, inplace=True, type_emissions="D_imp")

##reagreate from 11 to 5 regions :
reag_D_regions(
    reference,
    inplace=True,
    type_emissions="D_imp",
    dict_reag_regions=DICT_REGIONS,
    list_sec=["Agriculture", "Energy", "Industry", "Composite"],
)
reag_D_regions(
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
plot_trade_figures()


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
)'''
