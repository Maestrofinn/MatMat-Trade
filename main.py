from matplotlib import pyplot as plt
import os
import pandas as pd
import pickle as pkl
from typing import Callable, Dict, Tuple, Union

from src.model import Counterfactual, Model
from src.scenarios import DICT_SCENARIOS
from src.settings import (
    COLORS,
    FIGURES_MULTIMODEL_DIR,
    MODELS_DIR,
    REGIONS_AGG,
    SECTORS_AGG,
)
from src.stressors import STRESSORS_DICT_DEF, GHG_PARAMS#, MATERIAL_PARAMS, COPPER_PARAMS, LANDUSE_PARAMS
from src.utils import footprint_extractor


def load_model(
    base_year: int = 2015,
    system: str = "pxp",
    aggregation_name: str = "opti_S",
    capital: bool = False,
    stressor_subset: str = "full",
    verbose: bool = True,
) -> Model:
    """Loads an existing model

    Args:
        base_year (int): year in 4 digits. Defaults to 2015.
        system (str): product ('pxp') or industry ('ixi'). Defaults to 'pxp'.
        aggregation_name (str): name of the aggregation matrix used. Defaults to "opti_S".
        capital (bool, optional): True to endogenize investments and capital. Defaults to False.
        stressor_subset (str, optional): stressors group used to calibrate the model (see STRESSORS_DICT_DEF in stressors.py).
        verbose (bool, optional): True to print infos. Defaults to True.

    Returns:
        Model: object Model defined in model.py
    """
    backup_path = MODELS_DIR / (
        str(base_year)
        + "__"
        + system
        + "__"
        + aggregation_name
        + "__"
        + STRESSORS_DICT_DEF[stressor_subset]['name_EN']
        + capital * "__with_capital"
        + "/backup.pickle"
    )
    if os.path.isfile(backup_path):
        with open(backup_path, "rb") as f:
            return pkl.load(f)
    if verbose:
        print(
            f"Couldn't find an existing model at {backup_path}.\n You should create a new one using the class Model."
        )


### COMPARISONS OF EVOLUTIONS ###


def endogenous_capital_comparison(
    start_year: int,
    end_year: int,
    system: str = "pxp",
    aggregation_name: str = "opti_S",
    stressor_params: Dict = GHG_PARAMS,
    scenario_function: Callable[[Model, bool], Tuple[pd.DataFrame]] = None,
    scenario_name: str = None,
    reloc: bool = False,
    feature_extractor: Callable[[Model], Dict] = footprint_extractor,
    feature_name: str = "Composantes de l'empreinte carbone de la France",
    feature_name_short: str = "composantes_empreinte_carbone_FR",
) -> Dict:
    """Plots the evolution of the features extracted with feature_extractor from start_year to end_year in the given scenario, both with and without capital endogenization, and returns the different models.

    Args:
        start_year (int): when to start from (4 digits).
        end_year (int): when to stop (4 digits).
        system (str): product ('pxp') or industry ('ixi'). Defaults to 'pxp'.
        aggregation_name (str): name of the aggregation matrix used. Defaults to "opti_S".
        stressor_params (Dict, optional): dictionnary with the stressors' french name, english name, unit and a proxy as a dictionnary of comparable stressors (name as key, dictionnary as value with the list of corresponding Exiobase stressors and their weight). Defaults to a dictionnary with the GHGs.
        scenario_function (Callable[[Model, bool], Tuple[pd.DataFrame]], optional): builds the new Z and Y matrices from the model in a given scenario, is None to work with reference data. Defaults to None.
        scenario_name (str, optional): scenario name, in order to build the Model object, is None to work with reference data. Defaults to None.
        feature_extractor (Callable[[Model], Dict], optional): extracts one or several value(s) from a Model in a Dict. Defaults to carbon_footprint_extractor.
        reloc (bool, optional): True if relocation is allowed. Defaults to False.
        feature_name (str, optional): name of the extracted feature(s). Defaults to "empreinte carbone de la France".
        feature_name_short (str, optional): name of the extracted feature(s) formatted to be part of a file path. Defaults to "composantes_empreinte_carbone_FR".

    Returns:
        Dict: contains all the models
    """
    stressor_name = stressor_params["name_EN"]
    stressor_unit = stressor_params["unit"]

    models = {}
    years_range = range(start_year, end_year + 1)
    to_display = None

    def new_model(year: int, capital: bool) -> Union[Model, Counterfactual]:
        """Loads or creates a model, adds it to the models dictionnary and returns either the model or the selected counterfactual

        Args:
            year (int): year in 4 digits.
            capital (bool): True to endogenize investments and capital.

        Returns:
            Union[Model, Counterfactual]: object Model or Counterfactual defined in model.py
        """
        mod = load_model(
            base_year=year,
            system=system,
            aggregation_name=aggregation_name,
            capital=capital,
            stressor_name=stressor_name,
            verbose=False,
        )
        if mod is None:
            mod = Model(
                base_year=year,
                system=system,
                aggregation_name=aggregation_name,
                capital=capital,
                stressor_params=stressor_params,
            )
        models[mod.summary_long] = mod
        if scenario_name is not None:
            if scenario_name not in mod.get_counterfactuals_list():
                mod.new_counterfactual(
                    name=scenario_name, scenar_function=scenario_function, reloc=reloc
                )
            return mod.counterfactuals[scenario_name]
        return mod

    for year in years_range:
        print(f"\n--- PROCESSING YEAR {year} ---")
        for capital in [True, False]:
            print("--> With" + (1 - capital) * "out" + " capital")
            mod = new_model(year=year, capital=capital)
            features = feature_extractor(mod)
            if to_display is None:  # creates the DataFrame at first iteration
                feature_names = features.keys()
                nb_features = len(feature_names)
                capital_labels = ["Capital endogène", "Capital exogène"]
                multiindex = pd.MultiIndex.from_tuples(
                    [(k, f) for k in capital_labels for f in feature_names]
                )
                to_display = pd.DataFrame(columns=multiindex, index=years_range)
            for f in feature_names:
                to_display.loc[year, (capital_labels[1 - capital], f)] = features[f]

    to_display.plot(
        color=2 * COLORS[:nb_features],
        style=nb_features * ["-"] + nb_features * ["--"],
        fontsize=17,
        figsize=(10, 7),
    )

    plt.suptitle(
        feature_name[0].upper() + feature_name[1:],
        size=17,
    )
    plt.title("Évolutions avec et sans endogénéisation du capital", size=17)
    plt.xlabel("Années", size=15)
    plt.ylabel(stressor_unit)
    plt.tight_layout()
    plt.grid(visible=True)
    plt.legend(prop={"size": 12}, loc="center left", bbox_to_anchor=(1.0, 0.5))

    if scenario_name is None:
        scenario_name = "reference"
    reloc_suffix = reloc * "__with_reloc"
    plt.savefig(
        FIGURES_MULTIMODEL_DIR
        / f"endogenous_capital_comparison__{start_year}_{end_year}__{feature_name_short}__{system}__{aggregation_name}__{stressor_name}__{scenario_name}{reloc_suffix}.png",
        bbox_inches="tight",
    )

    return models
    
def apply_scenarios_from_excel(model, file_path, counterfactual_name):
    df = pd.read_excel(file_path)  # Lecture du fichier Excel
     # Crée le contrefactuel initial si nécessaire
    if counterfactual_name not in model.get_counterfactuals_list():
        model.new_counterfactual(counterfactual_name, lambda model: model.iot)  # Fonction neutre pour initialiser

    for index, row in df.iterrows():
        scenario_name = row['Scénario']
        sectors_list = row.get('SectorsList', None)
        if sectors_list is not None:
            sectors_list = sectors_list.split(',')  
        percentage_change = row.get('PourcentageChange', None)
    reloc = row.get('Reloc', False)

        if scenario_name in DICT_SCENARIOS:
            scenario_func = DICT_SCENARIOS[scenario_name]

            # Modifie le contrefactuel existant
            model.modify_counterfactual(counterfactual_name, scenario_func, reloc=reloc, sectors_list=sectors_list, percentage_change=percentage_change)
        else:
            print(f"Scénario non reconnu: {scenario_name}")

    model.save()
