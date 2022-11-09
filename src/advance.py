""" Template for python scripts in MatMat model

    Notes
    ------
    This template follow pep8 standard (checking with Pylint)
    and Google style docstrings ()

    """

# fix module path dependencies
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

del sys, Path

# general
import os

# scientific
import pandas as pd
import numpy as np

# local modules
from core.constants import advance_dir 


# settings
components = ['Production value', 'Production prices', 'World prices', 'Direct CO2 emissions', 'Investment', \
	'Secondary Energy', 'Final Energy']
regions = ['World']
base_year = 2015
final_year = 2050

# folder name
folder = (advance_dir / 'imaclimR_world')

# read data template
template = pd.read_excel(folder / 'template_data_output.xlsx')
template.drop(['Model', 'Scenario'], axis = 1, inplace = True)
template.set_index(['Region', 'Variable', 'Unit'], inplace = True)
template.insert(
	len(template.columns),
	template.columns[-1] + 1,
	np.nan
)

# read scenario name
scenario_name = pd.read_excel(folder / 'scenario_names.xlsx')
scenario_name = scenario_name.loc[~scenario_name.equivalent.isin(['-'])]

# read data
data = pd.concat(
	[
		pd.DataFrame(
			np.genfromtxt(
				(folder / ('outputs_advance_wp6' + str(scenario_name.nb[x]) + '.tsv')),
				delimiter = '\t'
			),
			index = template.index,
			columns = template.columns
		) for x in scenario_name.index 
	],
	axis = 0,
	keys = scenario_name.equivalent,#.tolist(),
	names = ['Scenario']
)
data.drop(template.columns[-1], axis = 1, inplace = True)

# reset index
data.reset_index(inplace = True)

# drop total quantities to avoid error
data = data.loc[~data.Variable.isin(components)]

# extract energy balance
energy_balance = data.loc[data.Variable.apply(lambda x: x.split('|')[0] in components[-2:])]

# isolate needed components: variables (except energy balance)
data = data.loc[data.Variable.apply(lambda x: x.split('|')[0] in components[0:-2])]
ind_var = data.Variable.apply(lambda x: x.split('|')[0])
ind_sect = data.Variable.apply(lambda x: x.split('|')[1] if x.split('|')[0] != 'Investment' else x.split('|')[-1])
data.drop(['Variable'], axis = 1, inplace = True)
data.insert(1, 'Sector', ind_sect)
data.insert(1, 'Variable', ind_var)

# drop invest not electricity
data = data.loc[
	(
		(data.Variable == 'Investment') & (data.Sector == 'Electricity')

	) | (data.Variable != 'Investment')
]

# update value of prices of world region
data.loc[
	(data.Region == 'World') & (data.Variable == 'Production prices'),
	range(2002, 2100 + 1)
] = data.loc[
	(data.Region == 'USA') & (data.Variable == 'World prices'),
	range(2002, 2100 + 1)
].values

# drop world prices
data.drop(data.loc[data.Variable == 'World prices'].index, inplace = True)

# convert value in US$ of BY
data.loc[data.Variable == 'Production value', range(2002, 2100+1)] = \
	data.loc[data.Variable == 'Production value', range(2002, 2100+1)].mul(
		data.loc[data.Variable == 'Production prices', base_year].divide(
			data.loc[data.Variable == 'Production prices', 2005]
		).values,
		axis = 0
	)
data.loc[data.Variable == 'Production value', 'Unit'] = 'billion US$' + str(base_year)

# convert price in US$ of BY
data.loc[data.Variable == 'Production prices', range(2002, 2100+1)] = \
	data.loc[data.Variable == 'Production prices', range(2002, 2100+1)].divide(
		data.loc[data.Variable == 'Production prices', base_year],
		axis = 0
	)
data.loc[data.Variable == 'Production prices', 'Unit'] = 'index' + str(base_year)

# isolate needed components: regions
data = data.loc[data.Region.apply(lambda x: x in regions)]
data.drop('Region', axis = 1, inplace = True)
data.set_index(['Variable', 'Scenario', 'Sector', 'Unit',], inplace = True)

# isolate needed components: years
data = data[range(base_year, final_year + 1, 1)]

# drop unit
data.index = data.index.droplevel('Unit')

# build production volume varible
data = pd.concat(
	[
		data,
		pd.concat(
			{'Production volume': data.loc['Production value'].divide(data.loc['Production prices'])},
			axis = 0, 
			names = [data.index.names[0]]
		)
	],
	axis = 0
)

# build emission factors of production
data = pd.concat(
	[
		data,
		pd.concat(
			{'CO2 emission factors': data.loc['Direct CO2 emissions'].divide(data.loc['Production volume'])},
			axis = 0, 
			names = [data.index.names[0]]
		)
	],
	axis = 0
)

# data.loc['Direct CO2 emissions'].loc['2°C'][range(2015, 2055, 5)].to_csv('advance_2°C_CO2emissions.csv')
# data.loc['CO2 emission factors'].loc['2°C'][range(2015, 2055, 5)].to_csv('advance_2°C_CO2content.csv')

# breakpoint()

# build emission factors evolution
final_data = data.loc['CO2 emission factors'].divide(data.loc['CO2 emission factors'][2015], axis = 0)
final_data = final_data - 1
final_data.drop(2015, axis = 1, inplace = True)

# set carbon content evolution of primary fossil fuels at zero
for scenario_name in final_data.index.get_level_values(0):
	for primary_energy in ['Coal', 'Oil', 'Gas']:
		final_data.loc[(scenario_name, primary_energy), :] = 0.0

#### NOTE ####
# liquid fuels = refined fossil fuels + biofuels
# gas / oil / coal = extractive sector ==> do not take into account EF evolution

# test calc K invest elec
kappa_elec = data.loc['Investment'].droplevel('Sector', axis = 0).divide(
	data.loc['Production volume'].xs('Elec', level = 'Sector')
)
kappa_elec = kappa_elec.divide(kappa_elec[2015], axis = 0)
kappa_elec = kappa_elec - 1
kappa_elec.drop(2015, axis = 1, inplace = True)

## kappa_elec result strange... don't remember !!! 
## il faut prendre celui de l'lectricité ! je pense qu'il y a un souci

########################
# build energy balance #
########################

# keep only world info
energy_balance = energy_balance.loc[energy_balance.Region == 'World']
energy_balance.drop(['Region', 'Unit'], axis = 1, inplace = True)

# drop totals
energy_balance = energy_balance.loc[energy_balance.Variable.apply(lambda x: len(x.split('|'))>2)]

# drop final and secondary name from variable
energy_balance.loc[:, 'Variable'] = energy_balance.Variable.apply(lambda x: '|'.join(x.split('|')[1:]))

# build sector columns
energy_balance.insert(1, column = 'Sector',
	value = energy_balance.Variable.apply(lambda x: x.split('|')[0])
)
energy_balance.insert(2, column = 'Product',
	value = energy_balance.Variable.apply(lambda x: x.split('|')[-1])
)

# keep usefull information
sector_list = {
	'Electricity': ['Coal', 'Oil', 'Gas'],#, 'Biomass'],
	'Liquids': ['Coal', 'Oil', 'Gas'],#, 'Biomass'],
	'Residential and Commercial': ['Coal', 'Liquids', 'Gases', 'Electricity'],
	'Transportation': ['Liquids', 'Electricity'],
	'Other Sector':['Coal', 'Liquids', 'Gases', 'Electricity']
}

energy_balance = pd.concat(
	[energy_balance.loc[
		energy_balance.Sector.isin([elt]) & energy_balance.Product.isin(sector_list[elt])
	] for elt in sector_list.keys()],
	axis = 0
).drop('Variable', axis = 1)

# harmonize sector and energy names
matching_list = {
	'Electricity': 'Elec',
	'Liquids': 'Liquid fuels',
	'Gases': 'Gas',
	'Transportation': 'Terrestrial transport',
	'Residential and Commercial': 'Services'
}

for column in ['Sector', 'Product']:
	for (key, value) in matching_list.items():
		energy_balance.loc[energy_balance.loc[:, column].isin([key]), column] = value

energy_balance.loc[energy_balance.Sector == 'Services', range(2002, 2100+1)] += \
	energy_balance.loc[energy_balance.Sector == 'Other Sector', range(2002, 2100+1)].values
energy_balance = energy_balance.loc[energy_balance.Sector != 'Other Sector']

# drop lines with nan
energy_balance = energy_balance.loc[~energy_balance[2015].isna()]

# set and sort index
energy_balance.set_index(['Scenario', 'Sector', 'Product'], inplace = True)
energy_balance = energy_balance.sort_index()

# keep useful year
energy_balance = energy_balance.loc[:, range(2015, 2050+1)]

# build technical coefficients evolution / 2015
technical_coef = energy_balance.copy()
for elt in technical_coef.index.droplevel('Product').unique():
	technical_coef.loc[elt, :] = energy_balance.loc[elt].divide(data.loc['Production volume'].loc[elt], axis = 1).values
	
technical_coef = technical_coef.divide(technical_coef[2015], axis = 0) - 1
technical_coef.drop(2015, axis = 1, inplace = True)
technical_coef.replace(np.nan, 0.0, inplace = True)
technical_coef.replace(np.inf, 0.0, inplace = True)

# # quick check: il y a des choses étranges sur les services !
# for elt in [2030, 2050]:
# 	print(
# 		technical_coef[elt].to_frame('data').reset_index().pivot(
# 			values = 'data', index = ['Scenario', 'Product'], columns = ['Sector'])
# 	)

# final technical coef with all sectors
final_technical_coef = pd.concat(
	{
		elt: pd.DataFrame(
			np.nan, index = final_data.index.get_level_values('Sector').unique(),
			columns = technical_coef.columns
		) for elt in final_data.index
	},
)
final_technical_coef.index.names = technical_coef.index.names
final_technical_coef.update(technical_coef)
final_technical_coef.replace(np.nan, 0.0, inplace = True)


## ToDo
# faire un graphique avec les hypothèse de décarbonation retenue par secteur !

