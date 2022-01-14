""" Compute the optimal spatial desagregation using a clustering method in order to robustify this desagregation """


#%% import modules

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt


# general
import os


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# scientific
import numpy as np
import pandas as pd
import pymrio
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# local folder
from local_paths import data_dir
from local_paths import output_dir

# local library
from utils import Tools


#%% Calibration
###########################
# SETTINGS
###########################

# year to study in [*range(1995, 2022 + 1)]
base_year = 2015

# system type: pxp or ixi
system = 'pxp'

# agg name: to implement in agg_matrix2.xlsx
agg_name = {
	'sector': 'ref',
	'region': 'ref'
}

# define filename concatenating settings
concat_settings = str(base_year) + '_' + \
	agg_name['sector']  + '_' +  \
	agg_name['region']

# set if rebuilding calibration from exiobase
calib = True


###########################
# READ/ORGANIZE/CLEAN DATA
###########################

# define file name
file_name = 'IOT_' + str(base_year) + '_' + system + '.zip'


# download data online
if not os.path.isfile(data_dir / file_name):

	pymrio.download_exiobase3(
		storage_folder = data_dir,
		system = system,
		years = base_year
	)


# import or build calibration data
if calib:
	print("Début calib")
	# import exiobase data
	reference = pymrio.parse_exiobase3(
		data_dir / file_name
	)

	# isolate ghg emissions
	reference.ghg_emissions = Tools.extract_ghg_emissions(reference)

	# del useless extensions
	reference.remove_extension(['satellite', 'impacts'])

	# import agregation matrices
	agg_matrix = {
		key: pd.read_excel(
			data_dir / 'agg_matrix2.xlsx',
			sheet_name = key + '_' + value
		) for (key, value) in agg_name.items()
	}
	agg_matrix['sector'].set_index(['category', 'sub_category', 'sector'], inplace = True)
	agg_matrix['region'].set_index(['Country name', 'Country code'], inplace = True)

	# apply regional and sectorial agregations
	reference.aggregate(
		region_agg = agg_matrix['region'].T.values,
		sector_agg = agg_matrix['sector'].T.values,
		region_names = agg_matrix['region'].columns.tolist(),
		sector_names = agg_matrix['sector'].columns.tolist()
	)

	# reset all to flows before saving
	reference = reference.reset_to_flows()
	reference.ghg_emissions.reset_to_flows()

	# save calibration data
	print("Fin calib")

else:

	# import calibration data built with calib = True
	reference = pymrio.parse_exiobase3(
		data_dir / ('reference' + '_' + concat_settings)
	)


#%%Initialize data
###########################
# CALCULATIONS
###########################

# calculate reference system
reference.calc_all()


# update extension calculations
reference.ghg_emissions_desag = Tools.recal_extensions_per_region(
	reference,
	'ghg_emissions'
)

sectors_list=list(reference.get_sectors())
reg_list = list(reference.get_regions())

print(reg_list)
print(sectors_list)


A=pd.DataFrame(reference.ghg_emissions_desag.M.sum())
Carbon_content = pd.DataFrame(np.zeros((len(reg_list),len(sectors_list))),index=reg_list,columns=sectors_list)
#print(Carbon_content)

for reg in reg_list:
    for sec in sectors_list:
        Carbon_content.loc[reg,sec] = A.loc[reg,sec].values


A=pd.DataFrame(reference.ghg_emissions_desag.M.T).sum(axis=1)
Carbon_content_sec = pd.DataFrame(np.zeros((len(sectors_list),len(reg_list))),index=sectors_list,columns=reg_list)
for reg in reg_list:
    Carbon_content_sec[reg] = A.loc[reg]

imports = (reference.Z['FR'].drop(['FR']).sum(level=0)).sum(axis=1)
sum_imports = imports.sum()

data = pd.DataFrame(np.zeros((len(reg_list[1:]),len(sectors_list)+1)),index = reg_list[1:],columns = np.append(sectors_list,'Import_FR_share'))
for sec in sectors_list :
    data[sec] = Carbon_content.copy()[sec]
#data[:,'Carbon_content'] = Carbon_content.copy()
data['Import_FR_share'] = imports.copy()/sum_imports

data_sec = Carbon_content_sec

#centrage réduction des données
data_cr = preprocessing.scale(data)
data_sec_cr = preprocessing.scale(data_sec)

#%% Trouver le nombre optimal de clusters

x1 = np.array(data_cr[:,0])
x2 = np.array(data_cr[:,1])
x3 = np.array(data_cr[:,2])
x4 = np.array(data_cr[:,3])
x5 = np.array(data_cr[:,4])
x6 = np.array(data_cr[:,5])
x7 = np.array(data_cr[:,6])
x8 = np.array(data_cr[:,7])
x9 = np.array(data_cr[:,8])
x10 = np.array(data_cr[:,9])
x11 = np.array(data_cr[:,10])
x12 = np.array(data_cr[:,11])
x13 = np.array(data_cr[:,12])
x14 = np.array(data_cr[:,13])
x15 = np.array(data_cr[:,14])
x16 = np.array(data_cr[:,15])
x17 = np.array(data_cr[:,16])
x18 = np.array(data_cr[:,17])

plt.plot()
plt.grid()
#plt.xlim([-3, 3])
#plt.ylim([-3, 3])
plt.title('Dataset')
plt.xlabel('Carbon content')
plt.ylabel('French import share')
plt.scatter(x1, x2)
plt.show()
# create new plot and data
plt.plot()
X = np.array(list(zip(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18))).reshape(len(x1), 18)

# k means determine k
distortions = []
K = range(1,len(reg_list))
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.grid()
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#%%Tracer le dendrogramme

nb_clusters_opti = 10
height_cut_opti = 7.5 #A mettre à jour

#générer la matrice des liens
Z = linkage(data_cr,method='ward',metric='euclidean')

plt.figure(2)
#affichage du dendrogramme
plt.title("CAH")
dendrogram(Z,labels=data.index,orientation='left',color_threshold=0)
plt.show()


#matérialisation des 4 classes (hauteur t = 7)
plt.title('CAH avec matérialisation des '+str(nb_clusters_opti)+' classes')

dendrogram(Z,labels=data.index,orientation='left',color_threshold=height_cut_opti)
plt.show()

#découpage à la hauteur t = 7 ==> 4 identifiants de groupes obtenus
groupes_cah = fcluster(Z,t=height_cut_opti,criterion='distance')
print(groupes_cah)

#index triés des groupes
import numpy as np
idg = np.argsort(groupes_cah)

#affichage des observations et leurs groupes
print(pd.DataFrame(data.index[idg],groupes_cah[idg]))


nb_clusters_opti_sec = 7
height_cut_opti_sec = 5.5 #A mettre à jour

nb_clusters_opti_sec = 3
height_cut_opti_sec = 17 #A mettre à jour

#générer la matrice des liens
Z_sec = linkage(data_sec_cr,method='ward',metric='euclidean')

plt.figure(21)
#affichage du dendrogramme
plt.title("CAH")
dendrogram(Z_sec,labels=data_sec.index,orientation='left',color_threshold=0)
plt.show()


#matérialisation des 4 classes (hauteur t = 7)
plt.title('CAH avec matérialisation des '+str(nb_clusters_opti_sec)+' classes')

dendrogram(Z_sec,labels=data_sec.index,orientation='left',color_threshold=height_cut_opti_sec)
plt.show()

#découpage à la hauteur t = 7 ==> 4 identifiants de groupes obtenus
groupes_cah_sec = fcluster(Z_sec,t=height_cut_opti,criterion='distance')
print(groupes_cah_sec)

#index triés des groupes
import numpy as np
idg_sec = np.argsort(groupes_cah_sec)

#affichage des observations et leurs groupes
print(pd.DataFrame(data_sec.index[idg_sec],groupes_cah_sec[idg_sec]))


#%% k-means sur les données centrées et réduites
kmeans = cluster.KMeans(n_clusters=nb_clusters_opti)

kmeans.fit(data_cr)

print(kmeans.inertia_)

#index triés des groupes
idk = np.argsort(kmeans.labels_)

#affichage des observations et leurs groupes
print(pd.DataFrame(data.index[idk],kmeans.labels_[idk]))

#distances aux centres de classes des observations
print(kmeans.transform(data_cr))

#correspondance avec les groupes de la CAH
pd.crosstab(groupes_cah,kmeans.labels_)


#%%
#Illustrer les clusters dans l'espace centré réduit basé sur Carbon_content x Import_share

colors_list = list(colors.cnames)
colors_list = colors_list[10:10+nb_clusters_opti]
#avec un code couleur selon le groupe
plt.figure(figsize=(18,12))
for couleur,k in zip(colors_list,np.arange(nb_clusters_opti)):
    plt.scatter(data_cr[kmeans.labels_==k,0],data_cr[kmeans.labels_==k,1],c=couleur)

plt.xlabel('Normalized carbon content')
plt.ylabel('Normalized French Imports share')

#mettre les labels des points
for i,label in enumerate(data.index):
    print(i,label)
    plt.annotate(label,(data_cr[i,0],data_cr[i,1]))
plt.savefig('figures/optim_clustering.png')
plt.show() 

# %%
#ACP
from sklearn.decomposition import PCA
acp_data_cr = PCA()
acp_data_cr.fit_transform(data_cr[:-1])
print(acp_data_cr.explained_variance_ratio_)

#projeter dans le plan factoriel
#avec un code couleur selon le groupe
#plt.figure(figsize=(18,7.715))
#for couleur,k in zip(['blue','lawngreen','aqua'],[0,1,2]):
#    plt.scatter(acp_data_cr[kmeans.labels_==k,0],acp_data_cr[kmeans.labels_==k,1],c=couleur)

#mettre les labels des points
#for i,label in enumerate(data.index):
#    plt.annotate(label,(acp_data_cr[i,0],acp_data_cr[i,1]))

#plt.show() 
# %%
