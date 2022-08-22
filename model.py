import figures
import os
import pandas as pd
from settings import CAPITAL_CONS_DIR, EXIOBASE_DIR, FIGURES_DIR, MODELS_DIR
from typing import Callable, Dict, List, Tuple
from utils import build_reference_data, build_counterfactual_data, reverse_mapper


class Model:
    def __init__(
        self,
        base_year: int,
        system: str,
        aggregation_name: str,
        calib: bool = False,
        regions_mapper: Dict = None,
        sectors_mapper: Dict = None,
        capital: bool = False,
    ):
        """Inits Model class

        Args:
            base_year (int): year in 4 digits
            system (str): product ('pxp') or industry ('ixi')
            aggregation_name (str): name of the aggregation matrix used
            calib (bool, optional): True to recalibrate the model from downloaded data. Defaults to False.
            regions_mapper (Dict, optional): regions aggregation for figures editing, no aggregation if is None. Defaults to None.
            sectors_mapper (Dict, optional): sectors aggregation for figures editing, no aggregation if is None. Defaults to None.
            capital (bool, optional): True to endogenize investments and capital. Defaults to False.
        """

        self.base_year = base_year
        self.system = system
        self.aggregation_name = aggregation_name
        self.calib = calib
        self.capital = capital

        self.summary_short = str(base_year) + "__" + system + "__" + aggregation_name
        self.summary_long = self.summary_short + capital * "__with_capital"
        self.exiobase_dir = EXIOBASE_DIR / self.summary_short
        self.model_dir = MODELS_DIR / self.summary_long
        self.raw_file_name = f"IOT_{base_year}_{system}.zip"
        self.pickle_file_name = self.summary_short + ".pickle"
        self.figures_dir = FIGURES_DIR / self.summary_long
        if self.capital:
            self.capital_consumption_path = (
                CAPITAL_CONS_DIR / f"Kbar_exio_v3_6_{self.base_year}{self.system}.mat"
            )

        self.iot = build_reference_data(model=self)
        self.regions = list(self.iot.get_regions())
        self.sectors = list(self.iot.get_sectors())
        self.y_categories = list(self.iot.get_Y_categories())

        self._regions_mapper = regions_mapper
        self._sectors_mapper = sectors_mapper
        self.rev_regions_mapper = reverse_mapper(mapper=regions_mapper)
        self.rev_sectors_mapper = reverse_mapper(mapper=sectors_mapper)
        if regions_mapper is None:
            self.new_regions_index = None
            self.agg_regions = self.regions
        else:
            self.agg_regions = list(regions_mapper.keys())
            self.new_regions_index = pd.Index(self.agg_regions)
        if sectors_mapper is None:
            self.new_sectors_index = None
            self.agg_sectors = self.sectors
        else:
            self.agg_sectors = list(sectors_mapper.keys())
            self.new_sectors_index = pd.Index(self.agg_sectors)

        self.counterfactuals = {}

    ## counterfactuals

    def new_counterfactual(
        self,
        name: str,
        scenar_function: Callable[["Model", bool], Tuple[pd.DataFrame]],
        reloc: bool = False,
    ) -> None:
        """Creates a new counterfactual from scenario_parameters in self.counterfactuals

        Args:
            name (str): counterfactual name
            scenar_function (Callable[[Model, bool], Tuple[pd.DataFrame]]): functions that builds the new Z and Y matrices
            reloc (bool, optional): True if relocation is allowed. Defaults to False.
        """
        self.counterfactuals[name] = Counterfactual(name, self, scenar_function, reloc)

    def create_counterfactuals_from_dict(
        self,
        parameters_dict: Dict = None,
        reloc: bool = False,
        verbose: bool = True,
    ) -> None:
        """Creates all new counterfactuals from scenario_parameters in self.counterfactuals

        Args:
            parameters_dict (Dict, optional): dictionnary with counterfactuals' names as keys and scenario parameters as values, set as DICT_SCENARIOS from scenarios.py if None. Defaults to None.
            reloc (bool, optional): True if relocation is allowed. Defaults to False.
            verbose (bool, optional): True to print infos. Defaults to True.
        """

        if parameters_dict is None:
            from scenarios import DICT_SCENARIOS

            parameters_dict = DICT_SCENARIOS
        for name, scenar_function in parameters_dict.items():
            self.new_counterfactual(
                name=name, scenar_function=scenar_function, reloc=reloc
            )
            if verbose:
                print(f"New counterfactual created : {name}")

        print(f"Available counterfactuals : {self.get_counterfactuals_list()}")

    def get_counterfactuals_list(self) -> List[str]:
        """Returns the list of the names of the available counterfactuals

        Returns:
            List[str]: names of the available counterfactuals
        """
        return list(self.counterfactuals.keys())

    ## aggregation mapping for figure editing

    @property
    def regions_mapper(self):
        return self._regions_mapper

    @regions_mapper.setter
    def regions_mapper(self, mapper):
        self._regions_mapper = mapper
        self.rev_regions_mapper = reverse_mapper(mapper=mapper)
        if mapper is None:
            self.new_regions_index = None
            self.agg_regions = self.regions
        else:
            self.agg_regions = list(mapper.keys())
            self.new_regions_index = pd.Index(self.agg_regions)

    @property
    def sectors_mapper(self):
        return self._sectors_mapper

    @sectors_mapper.setter
    def sectors_mapper(self, mapper):
        self._sectors_mapper = mapper
        self.rev_sectors_mapper = reverse_mapper(mapper=mapper)
        if mapper is None:
            self.agg_sectors = self.sectors
            self.new_sectors_index = None
        else:
            self.agg_sectors = list(mapper.keys())
            self.new_sectors_index = pd.Index(self.agg_sectors)

    ## description plots

    def plot_carbon_footprint(
        self,
        counterfactual_name: str = None,
        region: str = "FR",
        title: str = None,
    ) -> None:
        """Plots region's carbon footprint (D_pba-D_exp+D_imp+F_Y)

        Args:
            counterfactual_name (str, optional): name of the counterfactual to plot, or None to plot the reference (self). Defaults to None.
            region (str, optional): region name. Defaults to "FR".
            title (str, optional): title of the figure. Defaults to None.
        """
        if counterfactual_name is None:
            figures.plot_carbon_footprint(model=self, region=region, title=title)
        else:
            counterfactual = self.counterfactuals[counterfactual_name]
            figures.plot_carbon_footprint(
                model=counterfactual, region=region, title=title
            )

    def plot_carbon_footprint_FR(
        self, counterfactual_name: str = None, title: str = None
    ) -> None:
        """Plots french carbon footprint (D_pba-D_exp+D_imp+F_Y)

        Args:
            counterfactual_name (str, optional): name of the counterfactual to plot, or None to plot the reference (self). Defaults to None.
            title (str, optional): title of the figure. Defaults to None.
        """
        if counterfactual_name is None:
            figures.plot_carbon_footprint_FR(model=self)
        else:
            counterfactual = self.counterfactuals[counterfactual_name]
            figures.plot_carbon_footprint_FR(model=counterfactual)

    def ghg_content_heatmap(
        self,
        counterfactual_name: str = None,
        prod: bool = False,
    ) -> None:
        """Plots the GHG contents each sector for each region in a heatmap

        Args:
            counterfactual_name (str, optional): name of the counterfactual to plot, or None to plot the reference (self). Defaults to None.
            prod (bool, optional): True to focus on production values, otherwise focus on consumption values. Defaults to False.
        """
        figures.ghg_content_heatmap(
            model=self, counterfactual_name=counterfactual_name, prod=prod
        )

    ## comparison plots

    def plot_trade_synthesis(self, counterfactual_name: str) -> None:
        """Plots the french importations for a given counterfactual

        Args:
            counterfactual_name (str): name of the counterfactual in model.counterfactuals
        """
        figures.plot_trade_synthesis(
            model=self, counterfactual_name=counterfactual_name
        )

    def plot_co2eq_synthesis(self, counterfactual_name: str) -> None:
        """Plots the french emissions per sector for a given counterfactual

        Args:
            counterfactual_name (str): name of the counterfactual in model.counterfactuals
        """
        figures.plot_co2eq_synthesis(
            model=self, counterfactual_name=counterfactual_name
        )

    def plot_ghg_synthesis(self, counterfactual_name: str) -> None:
        """Plots the french emissions per GHG for a given counterfactual

        Args:
            counterfactual_name (str): name of the counterfactual in model.counterfactuals
        """
        figures.plot_ghg_synthesis(model=self, counterfactual_name=counterfactual_name)

    ## plots for all scenarios

    def compare_scenarios(self, verbose: bool = False) -> None:
        """Plots the carbon footprints and the imports associated with the different counterfactuals

        Args:
            verbose (bool, optional): True to print infos. Defaults to True.
        """
        figures.compare_scenarios(model=self, verbose=verbose)

    def ghg_content_heatmap_all(self, prod: bool = False) -> None:
        """Plots the GHG contents each sector for each region in a heatmap for each counterfactual

        Args:
            prod (bool, optional): True to focus on production values, otherwise focus on consumption values. Defaults to False.
        """
        for counterfactual_name in self.get_counterfactuals_list():
            self.ghg_content_heatmap(counterfactual_name=counterfactual_name, prod=prod)

    def plot_trade_synthesis_all(self) -> None:
        """Plots the french importations for each counterfactual"""
        for counterfactual_name in self.get_counterfactuals_list():
            self.plot_trade_synthesis(counterfactual_name=counterfactual_name)

    def plot_co2eq_synthesis_all(self) -> None:
        """Plots the french emissions by sector for each counterfactual"""
        for counterfactual_name in self.get_counterfactuals_list():
            self.plot_co2eq_synthesis(counterfactual_name=counterfactual_name)

    def plot_ghg_synthesis_all(self) -> None:
        """Plot the french emissions per GHG for each counterfactual"""
        for counterfactual_name in self.get_counterfactuals_list():
            self.plot_ghg_synthesis(counterfactual_name=counterfactual_name)


class Counterfactual:
    def __init__(
        self,
        name: str,
        model: Model,
        scenar_function: Callable[[Model, bool], Tuple[pd.DataFrame]],
        reloc: bool = False,
    ):
        """Inits Counterfactual class

        Args:
            name (str): name of the counterfactual
            model (Model): object Model defined in model.py
            scenar_function (Callable[[Model, bool], Tuple[pd.DataFrame]]): builds the new Z and Y matrices
            reloc (bool, optional): True if relocation is allowed. Defaults to False.
        """

        self.name = name
        self.iot = build_counterfactual_data(
            model=model, scenar_function=scenar_function, reloc=reloc
        )
        self.figures_dir = model.figures_dir / name
        if not os.path.isdir(self.figures_dir):
            os.mkdir(self.figures_dir)

        self.regions = model.regions
        self.sectors = model.sectors
        self.y_categories = model.y_categories
