# fix module path dependencies
import sys
from pathlib import Path


# general
import os

# scientific
import pandas as pd
import numpy as np

# local modules



# settings
components = ['Production value', 'Production prices', 'World prices', 'Direct CO2 emissions', 'Investment', \
	'Secondary Energy', 'Final Energy']
regions = ['USA', 'CAN', 'EUR', 'JAN', 'CIS', 'CHN', 'IND', 'BRA', 'MDE', 'AFR','RAS', 'RAL', 'World']
base_year = 2015
final_year = 2050

# folder name
folder = "../Data/IMACLIM"

# read data template
template = pd.read_excel(folder + '/template_data_output.xlsx')
template.drop(['Model', 'Scenario'], axis = 1, inplace = True)
template.set_index(['Region', 'Variable', 'Unit'], inplace = True)
template.insert(
	len(template.columns),
	template.columns[-1]  + 1,    # type: ignore , template.columns[-1] should be the last year of the scenario, hence an int
	np.nan  # type: ignore 
)

# read scenario name
scenario_name = pd.read_excel(folder + '/scenario_names.xlsx')
scenario_name = scenario_name.loc[~scenario_name.equivalent.isin(['-'])]

# read data
data = pd.concat(
	[
		pd.DataFrame(
			np.genfromtxt(
				(folder + ('/outputs_advance_wp6' + str(scenario_name.nb[x]) + '.tsv')),
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
data.set_index(['Variable', 'Scenario','Region','Sector', 'Unit',], inplace = True)

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
# for scenario_name in final_data.index.get_level_values(0):
# 	for primary_energy in ['Coal', 'Oil', 'Gas']:
# 		for region in regions:
# 			final_data.loc[(scenario_name,region, primary_energy), :] = 0.0
   
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

# keep only the regions we are interested in
energy_balance = energy_balance.loc[energy_balance.Region.isin(regions)]
energy_balance.drop(['Unit'], axis = 1, inplace = True)

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


reverse_matching_list={matching_list[key]:key for key in matching_list.keys()}

for column in ['Sector', 'Product']:
	for (key, value) in matching_list.items():
		energy_balance.loc[energy_balance.loc[:, column].isin([key]), column] = value

# add "Other Sectors" to the "Service" industry
energy_balance.loc[energy_balance.Sector == 'Services', range(2002, 2100+1)] += \
	energy_balance.loc[energy_balance.Sector == 'Other Sector', range(2002, 2100+1)].values
energy_balance = energy_balance.loc[energy_balance.Sector != 'Other Sector']

# drop lines with nan
energy_balance = energy_balance.loc[~energy_balance[2015].isna()]

# set and sort index
energy_balance.set_index(['Scenario', 'Sector', 'Product',"Region"], inplace = True)
energy_balance = energy_balance.sort_index()

# keep useful year
energy_balance = energy_balance.loc[:, range(2015, 2050+1)]


# build technical coefficients evolution / 2015
technical_coef = energy_balance.copy()
for scenario,sector in zip(technical_coef.index.get_level_values("Scenario"),technical_coef.index.get_level_values("Sector")):
	extractor=data.loc['Production volume']
	extractor=extractor.reorder_levels(["Scenario","Sector","Region"]).sort_index()
	totals=extractor.loc[(scenario,sector)]
	technical_coef.loc[scenario,sector, :] = energy_balance.loc[scenario,sector].divide(totals, axis = 1).values

 
 
 
	
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

technical_coef=technical_coef.reorder_levels(["Scenario","Region","Sector","Product"]).sort_index()

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
final_technical_coef.sort_index(inplace=True)


### Building the links between sectors and countris of IMACLIM and MATMAT Trade ### 


# Link between sectors

Region_list=['FR', 'UK, Norway, Switzerland', 'United States', 'Asia, Row Europe', 'China, RoW Asia and Pacific', 'RoW America,Turkey, Taïwan', 'RoW Middle East, Australia', 'Brazil, Mexico', 'South Africa', 'Japan, Indonesia, RoW Africa', 'EU']
Sector_list=['Agriculture', 'Crude coal', 'Crude oil', 'Natural gas',
       'Extractive industry', 'Biomass_industry', 'Clothing', 'Heavy_industry',
       'Construction', 'Automobile', 'Oth transport equipment', 'Machinery',
       'Electronics', 'Fossil fuels', 'Electricity and heat',
       'Transport services', 'Composite']

Link=pd.DataFrame(data=0,index=np.sort(Sector_list),columns=np.sort(final_data.loc["INDC"].index.get_level_values(1).unique()))
for i in Link.index:
    for j in Link.columns:
        if i==j:
            Link.loc[i,j]=1
pairing=[("Crude oil","Oil"),("Crude coal","Coal"),("Automobile","Industry"),
         ("Transport services","Maritime"),("Biomass_industry","Industry"),("Clothing","Industry"),
         ("Composite","Industry"),("Electricity and heat",'Elec'),("Electronics","Industry"),
         ("Extractive industry","Industry"),("Fossil fuels","Liquid fuels"),("Heavy_industry","Industry"),
         ("Machinery","Industry"),("Natural gas","Gas"),("Oth transport equipment","Air")]
for pair in pairing: 
    Link.loc[pair[0],pair[1]]=1
    
# Link between countries 

Link_country=pd.DataFrame(data=0,index=np.sort(Region_list),columns=np.sort(final_data.loc["INDC"].index.get_level_values(0).unique()))
pairing=[("EU","EUR"),("Brazil, Mexico","BRA"),("FR","EUR"),
         ("China, RoW Asia and Pacific","CHN"),("United States","USA"),("Asia, Row Europe","IND"),
         ("UK, Norway, Switzerland","EUR"),("RoW Middle East, Australia","MDE"),("Japan, Indonesia, RoW Africa","AFR"),
         ("RoW America,Turkey, Taïwan","World"),("South Africa","AFR")]
for pair in pairing: 
    Link_country.loc[pair[0],pair[1]]=1
    

### Adapating IMACLIM exctracted data to the country and sector aggregation of MATMAT Trade ###


scenarios=final_data.index.get_level_values("Scenario").unique()

final_data_ratio=pd.concat([pd.concat([ Link.dot(final_data.loc[(scenario,region),:]) for region in regions],
                                            axis=0,
                                            keys=regions,
                                            names=["regions","sector"]) for scenario in scenarios],
                                axis=0,
                                keys=scenarios,
                                names=["scenario","regions","sector"])



indexes=pd.Series(zip(final_technical_coef.index.get_level_values("Scenario"),final_technical_coef.index.get_level_values("Region"),final_technical_coef.index.get_level_values("Sector"))).unique()
final_technical_coef=pd.concat([Link.dot(final_technical_coef.loc[(scenario,region,sector)]) for scenario,region,sector in indexes],
                               names=("Scenario","Region","Sector","Product"),
                               keys=indexes,
                               axis=0)

final_technical_coef=final_technical_coef.swaplevel("Sector","Product").sort_index()

# Making sectors match the recieving end pattern
indexes=pd.Series(zip(final_technical_coef.index.get_level_values("Scenario"),final_technical_coef.index.get_level_values("Region"),final_technical_coef.index.get_level_values("Product"))).unique()
final_technical_coef=pd.concat([Link.dot(final_technical_coef.loc[(scenario,region,product)]) for scenario,region,product in indexes],
                               names=("Scenario","Region","Product","Sector"),
                               keys=indexes,
                               axis=0)

final_technical_coef=final_technical_coef.swaplevel("Sector","Product")

# Linking the region match the recieving end pattern
final_technical_coef=final_technical_coef.swaplevel("Region","Product").sort_index()

indexes=pd.Series(zip(final_technical_coef.index.get_level_values("Scenario"),final_technical_coef.index.get_level_values("Product"),final_technical_coef.index.get_level_values("Sector"))).unique()
final_technical_coef=pd.concat([Link_country.dot(final_technical_coef.loc[(scenario,product,sector)]) for scenario,product,sector in indexes],
                               names=("Scenario","Product","Sector","Region"),
                               keys=indexes,
                               axis=0)


final_technical_coef=final_technical_coef.swaplevel("Region","Product").sort_index()

# Formating the dataframe more similarly to A

indexes=pd.Series(zip(final_technical_coef.index.get_level_values("Scenario"),final_technical_coef.index.get_level_values("Region"),final_technical_coef.index.get_level_values("Sector"))).unique()
final_technical_coef=pd.concat([final_technical_coef.loc[(scenario,product,sector)] for scenario,product,sector in indexes],
                               names=("Scenario","Region","Sector","Year"),
                               keys=indexes,
                               axis=1)


final_technical_coef=final_technical_coef.reorder_levels(("Scenario","Year","Region","Sector"),axis=1)
final_technical_coef.columns.rename(["Scenario","Year","region","sector"],inplace=True)
final_technical_coef.index.rename("sector",inplace=True)