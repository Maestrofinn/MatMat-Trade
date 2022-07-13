import figures
import os
import pandas as pd
from settings import DATA_DIR, FIGURES_DIR, REGIONS_AGG, SECTORS_AGG
from typing import Dict, List
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
    ):
        """Inits Model class

        Args:
            base_year (int): year in 4 digits
            system (str): product ('pxp') or industry ('ixi')
            aggregation_name (str): name of the aggregation matrix used
            calib (bool, optional): True to recalibrate the model from downloaded data. Defaults to False.
            regions_mapper (Dict, optional): regions aggregation for figures editing, no aggregation if is None. Defaults to None.
            sectors_mapper (Dict, optional): sectors aggregation for figures editing, no aggregation if is None. Defaults to None.
        """

        self.base_year = base_year
        self.system = system
        self.aggregation_name = aggregation_name
        self.calib = calib

        self.concat_settings = str(base_year) + "_" + system + "_" + aggregation_name
        self.model_dir = DATA_DIR / self.concat_settings
        self.raw_file_name = f"IOT_{base_year}_{system}.zip"
        self.pickle_file_name = self.concat_settings + ".pickle"
        self.local_figures_dir = FIGURES_DIR / self.concat_settings
        if not os.path.isdir(self.local_figures_dir):
            os.mkdir(self.local_figures_dir)

        self.database = build_reference_data(self)
        self.regions = list(self.database.get_regions())
        self.sectors = list(self.database.get_sectors())
        self.y_categories = list(self.database.get_Y_categories())

        self._regions_mapper = regions_mapper
        self._sectors_mapper = sectors_mapper
        self.rev_regions_mapper = reverse_mapper(regions_mapper)
        self.rev_sectors_mapper = reverse_mapper(sectors_mapper)
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
        self, name: str, scenario_parameters: Dict, reloc: bool = False
    ) -> None:
        """Creates a new counterfactual from scenario_parameters in self.counterfactuals

        Args:
            name (str): counterfactual name
            scenario_parameters (Dict): contains the scenario function ('scenario_fucntion') and the shock function ('shock_function')
            reloc (bool, optional): True if relocation is allowed. Defaults to False.
        """
        self.counterfactuals[name] = Counterfactual(
            name, self, scenario_parameters, reloc
        )

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
        for name, scenario_parameters in parameters_dict.items():
            self.new_counterfactual(name, scenario_parameters, reloc)
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
        self.rev_regions_mapper = reverse_mapper(mapper)
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
        self.rev_sectors_mapper = reverse_mapper(mapper)
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
            figures.plot_carbon_footprint(self, region, title)
        else:
            counterfactual = self.counterfactuals[counterfactual_name]
            figures.plot_carbon_footprint(counterfactual, region, title)

    def plot_carbon_footprint_FR(
        self, counterfactual_name: str = None, title: str = None
    ) -> None:
        """Plots french carbon footprint (D_pba-D_exp+D_imp+F_Y)

        Args:
            counterfactual_name (str, optional): name of the counterfactual to plot, or None to plot the reference (self). Defaults to None.
            title (str, optional): title of the figure. Defaults to None.
        """
        if counterfactual_name is None:
            figures.plot_carbon_footprint_FR(self, title)
        else:
            counterfactual = self.counterfactuals[counterfactual_name]
            figures.plot_carbon_footprint_FR(counterfactual, title)

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
        figures.ghg_content_heatmap(self, counterfactual_name, prod)

    ## comparison plots

    def plot_trade_synthesis(self, counterfactual_name: str) -> None:
        """Plots the french importations for a given counterfactual

        Args:
            counterfactual_name (str): name of the counterfactual in model.counterfactuals
        """
        figures.plot_trade_synthesis(self, counterfactual_name)

    def plot_co2eq_synthesis(self, counterfactual_name: str) -> None:
        """Plots the french emissions per sector for a given counterfactual

        Args:
            counterfactual_name (str): name of the counterfactual in model.counterfactuals
        """
        figures.plot_co2eq_synthesis(self, counterfactual_name)

    def plot_ghg_synthesis(self, counterfactual_name: str) -> None:
        """Plots the french emissions per GHG for a given counterfactual

        Args:
            counterfactual_name (str): name of the counterfactual in model.counterfactuals
        """
        figures.plot_ghg_synthesis(self, counterfactual_name)

    ## plots for all scenarios

    def compare_scenarios(self, verbose: bool = False) -> None:
        """Plots the carbon footprints and the imports associated with the different counterfactuals

        Args:
            verbose (bool, optional): True to print infos. Defaults to True.
        """
        figures.compare_scenarios(self, verbose)

    def ghg_content_heatmap_all(self, prod: bool = False) -> None:
        """Plots the GHG contents each sector for each region in a heatmap for each counterfactual

        Args:
            prod (bool, optional): True to focus on production values, otherwise focus on consumption values. Defaults to False.
        """
        for counterfactual_name in self.get_counterfactuals_list():
            self.ghg_content_heatmap(counterfactual_name, prod)

    def plot_trade_synthesis_all(self) -> None:
        """Plots the french importations for each counterfactual"""
        for counterfactual_name in self.get_counterfactuals_list():
            self.plot_trade_synthesis(counterfactual_name)

    def plot_co2eq_synthesis_all(self) -> None:
        """Plots the french emissions by sector for each counterfactual"""
        for counterfactual_name in self.get_counterfactuals_list():
            self.plot_co2eq_synthesis(counterfactual_name)

    def plot_ghg_synthesis_all(self) -> None:
        """Plot the french emissions per GHG for each counterfactual"""
        for counterfactual_name in self.get_counterfactuals_list():
            self.plot_ghg_synthesis(counterfactual_name)


class Counterfactual:
    def __init__(
        self, name: str, model: Model, scenario_parameters: Dict, reloc: bool = False
    ):
        """Inits Counterfactual class

        Args:
            name (str): name of the counterfactual
            model (Model): object Model defined in model.py
            scenario_parameters (Dict): contains the scenario function ('scenario_fucntion') and the shock function ('shock_function')
            reloc (bool, optional): True if relocation is allowed. Defaults to False.
        """

        self.name = name
        self.database = build_counterfactual_data(model, scenario_parameters, reloc)
        self.local_figures_dir = model.local_figures_dir / name
        if not os.path.isdir(self.local_figures_dir):
            os.mkdir(self.local_figures_dir)

        self.regions = model.regions
        self.sectors = model.sectors
        self.y_categories = model.y_categories
