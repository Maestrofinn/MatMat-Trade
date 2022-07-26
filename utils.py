import os
import numpy as np
import pandas as pd
import pickle as pkl
import pymrio
from pymrio.tools import ioutil
from settings import AGGREGATION_DIR, DATA_DIR, GLOBAL_WARMING_POTENTIAL, OUTPUT_DIR
from typing import Dict
import warnings

# remove pandas warning related to pymrio future deprecations
warnings.simplefilter(action="ignore", category=FutureWarning)


### AUXILIARY FUNCTION FOR DATA BUILDERS ###


def recal_extensions_per_region(
    iot: pymrio.IOSystem, extension_name: str
) -> pymrio.core.mriosystem.Extension:
    """Computes the account matrices D_cba, D_pba, D_imp and D_exp
       Based on pymrio.tools.iomath's function 'calc_accounts', see https://github.com/konstantinstadler/pymrio

    Args:
        iot (pymrio.IOSystem): pymrio MRIO object
        extension_name (str): extension name

    Returns:
        pymrio.core.mriosystem.Extension: extension with account matrices completed
    """
    extension = getattr(iot, extension_name).copy()

    S = getattr(iot, extension_name).S
    L = iot.L
    Y_vect = iot.Y.sum(level=0, axis=1)
    nbsectors = len(iot.get_sectors())

    Y_diag = ioutil.diagonalize_blocks(Y_vect.values, blocksize=nbsectors)
    Y_diag = pd.DataFrame(Y_diag, index=Y_vect.index, columns=Y_vect.index)
    x_diag = L.dot(Y_diag)

    regions = x_diag.index.get_level_values("region").unique()

    # calc carbon footprint
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

    return extension


### DATA BUILDERS ###


def build_reference_data(model) -> pymrio.IOSystem:
    """Builds the pymrio object given reference's settings

    Args:
        model (Model): object Model defined in model.py

    Returns:
        pymrio.IOSystem: pymrio object
    """

    # create model directory if necessary
    if not os.path.isdir(model.model_dir):
        os.mkdir(model.model_dir)

    # downloading data if necessary
    if not os.path.isfile(DATA_DIR / model.raw_file_name):
        print("Downloading data... (may take a few minutes)")
        pymrio.download_exiobase3(
            storage_folder=DATA_DIR, system=model.system, years=model.base_year
        )
        print("Data downloaded successfully !")

    force_calib = not os.path.isdir(
        model.model_dir / ("reference" + "_" + model.concat_settings)
    )

    if model.calib or force_calib:

        print("Loading data... (may take a few minutes)")

        # import exiobase data
        if os.path.isfile(model.model_dir / model.pickle_file_name):
            with open(model.model_dir / model.pickle_file_name, "rb") as f:
                iot = pkl.load(f)
        else:
            iot = pymrio.parse_exiobase3(  # may need RAM + SWAP ~ 15 Gb
                DATA_DIR / model.raw_file_name
            )
            with open(model.model_dir / model.pickle_file_name, "wb") as f:
                pkl.dump(iot, f)

        # extract GHG emissions
        extension_list = list()

        for ghg_emission in GLOBAL_WARMING_POTENTIAL.keys():

            ghg_name = ghg_emission.lower() + "_emissions"
            extension = pymrio.Extension(ghg_name)
            ghg_index = iot.satellite.F.reset_index().stressor.apply(
                lambda x: x.split(" ")[0] in [ghg_emission]
            )

            for elt in ["F", "F_Y", "unit"]:

                component = getattr(iot.satellite, elt)

                if elt == "unit":
                    index_name = "index"
                else:
                    index_name = str(component.index.names[0])

                component = component.reset_index().loc[ghg_index].set_index(index_name)

                if elt == "unit":
                    component = pd.DataFrame(
                        component.values[0],
                        index=pd.Index([ghg_emission]),
                        columns=component.columns,
                    )
                else:
                    component = component.sum(axis=0).to_frame(ghg_emission).T
                    component.loc[ghg_emission] *= (
                        GLOBAL_WARMING_POTENTIAL[ghg_emission] * 1e-9
                    )
                    component.index.name = index_name

                setattr(extension, elt, component)

            extension_list.append(extension)

        iot.ghg_emissions = pymrio.concate_extension(
            extension_list, name="ghg_emissions"
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

        # reset all to flows before saving
        iot = iot.reset_to_flows()
        iot.ghg_emissions.reset_to_flows()

        # compute missing matrices
        iot.calc_all()

        # compute emission accounts by region
        iot.ghg_emissions_desag = recal_extensions_per_region(iot, "ghg_emissions")

        # save model
        iot.save_all(model.model_dir / ("reference" + "_" + model.concat_settings))

        print("Data loaded successfully !")

    else:

        # import calibration data previously built with calib = True
        iot = pymrio.parse_exiobase3(
            model.model_dir / ("reference" + "_" + model.concat_settings)
        )

    return iot


def build_counterfactual_data(
    model,
    scenar_function,
    reloc: bool = False,
) -> pymrio.IOSystem:
    """Builds the pymrio object given reference's settings and the scenario parameters

    Args:
        model (Model): object Model defined in model.py
        scenar_function (Callable[[Model, bool], Tuple[pd.DataFrame]]): builds the new Z and Y matrices
        reloc (bool, optional): True if relocation is allowed. Defaults to False.
    Returns:
        pymrio.IOSystem: modified pymrio model, with A, x and L set as None
    """

    counterfactual = model.iot.copy()
    counterfactual.remove_extension("ghg_emissions_desag")

    counterfactual.Z, counterfactual.Y = scenar_function(model, reloc=reloc)

    counterfactual.A = None
    counterfactual.x = None
    counterfactual.L = None

    counterfactual.calc_all()

    counterfactual.ghg_emissions_desag = recal_extensions_per_region(
        counterfactual,
        "ghg_emissions",
    )

    counterfactual.save_all(
        OUTPUT_DIR / ("counterfactual" + "_" + model.concat_settings)
    )

    return counterfactual


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
        aggregate_sum(df, 0, axis, new_index_0, reverse_mapper_0),
        1,
        axis,
        new_index_1,
        reverse_mapper_1,
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
        aggregate_sum_axis(
            df, 0, new_index_0, new_index_1, reverse_mapper_0, reverse_mapper_1
        ),
        1,
        new_index_0,
        new_index_1,
        reverse_mapper_0,
        reverse_mapper_1,
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
        aggregate_sum_2levels_2axes(df, new_index_0, None, reverse_mapper_0, None),
        1,
        1,
        new_index_1,
        reverse_mapper_1,
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
        aggregate_sum_2levels_2axes(df, new_index_0, None, reverse_mapper_0, None),
        1,
        0,
        new_index_1,
        reverse_mapper_1,
    )
