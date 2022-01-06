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


#%% Initialise data
###########################
# SETTINGS
###########################

# year to study in [*range(1995, 2022 + 1)]
base_year = 2015

# system type: pxp or ixi
system = 'pxp'

# agg name: to implement in agg_matrix.xlsx
agg_name = {
	'sector': 'ref',
	'region': 'ref'
}

# define filename concatenating settings
concat_settings = str(base_year) + '_' + \
	agg_name['sector']  + '_' +  \
	agg_name['region']

# set if rebuilding calibration from exiobase
calib = False


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


reference = pymrio.parse_exiobase3(
	data_dir / ('reference' + '_' + concat_settings)
)


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

Carbon_content = pd.DataFrame(reference.ghg_emissions_desag.M.sum()).sum(level=0)

imports = (reference.Z['FR'].drop(['FR']).sum(level=0)).sum(axis=1)
sum_imports = imports.sum()

data = pd.DataFrame(np.zeros((len(reg_list[1:]),2)),index = reg_list[1:],columns = ['Carbon_content','Import_FR_share'])

data.loc[:,'Carbon_content'] = Carbon_content.copy()
data.loc[:,'Import_FR_share'] = imports.copy()/sum_imports
#centrage réduction des données
data_cr = preprocessing.scale(data)

#%% Trouver le nombre optimal de clusters

x1 = np.array(data_cr[:,0]) #carbon content
x2 = np.array(data_cr[:,1]) #import share

plt.plot()
plt.grid()
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.title('Dataset')
plt.xlabel('Carbon content')
plt.ylabel('French import share')
plt.scatter(x1, x2)
plt.show()
# create new plot and data
plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

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

nb_clusters_opti = 4
height_cut_opti = 1.5 #A mettre à jour

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
for couleur,k in zip(colors_list,np.arange(nb_clusters_opti)):
    plt.scatter(data_cr[kmeans.labels_==k,0],data_cr[kmeans.labels_==k,1],c=couleur)

plt.xlabel('Normalized carbon content')
plt.ylabel('Normalized French Imports share')

#mettre les labels des points
for i,label in enumerate(data.index):
    print(i,label)
    plt.annotate(label,(data_cr[i,0],data_cr[i,1]))
plt.show() 
