"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from kneed import KneeLocator


##################################################################
# Exemple :  k-Means Clustering

path = './artificial/'
name="spiral.arff"

#donut curves 
#spiral

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()



tableauinertie =[]
diff = 0
i = 0
calinski_data = []
silhouette_data = []
davins_data = []
run_time_data = []
historique = []

for k in range(1,50):
    # Run clustering method for a given number of clusters
    print("----------Appel KMeans pour une valeur de k fixée : ", k," ----------------")
    tps1 = time.time()
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    # informations sur le clustering obtenu
    iteration = model.n_iter_
    inertie = model.inertia_
    centroids = model.cluster_centers_

    tableauinertie.append(inertie)
    run_time_data.append(tps2-tps1)
    
    historique.append([centroids,labels,tableauinertie])
    
    if k>1 and diff<tableauinertie[k-2]-tableauinertie[k-1]:
        
        kneedle = KneeLocator(range(1, len(tableauinertie) + 1), tableauinertie, curve='convex', direction='decreasing')
        i = kneedle.elbow
        
        

    if k!=1:
        calinski_mesure = metrics.calinski_harabasz_score(datanp,labels)
        silhouette_mesure = metrics.silhouette_score(datanp, labels)
        davins_mesure = metrics.davies_bouldin_score(datanp, labels)
        
        
        calinski_data.append(calinski_mesure)
        silhouette_data.append(silhouette_mesure)
        davins_data.append(davins_mesure)
    else:
        calinski_data.append(0)
        silhouette_data.append(0)
        davins_data.append(0)

    print("Inertie : ", inertie)
    print("Calinski metric : ",calinski_data[:-1])
    print("Silhouette mesure : ",silhouette_data[:-1])
    print("Davins metric : ",davins_data[:-1])

from sklearn.metrics.pairwise import euclidean_distances
dists = euclidean_distances(centroids)
print(dists)


print("Tableau inertie (Liste)", tableauinertie)    
print("Supposé nombre de clusters : ", i)


plt.axvline(i,c='r')
plt.plot([i for i in range(1,50)],tableauinertie,c='b')
plt.xlabel("Nombre de Cluster" )
plt.ylabel("Inertie")
plt.show()


plt.plot([i for i in range(1,50)],calinski_data,c='g')
plt.axvline(i,c='r')
plt.xlabel("Nombre de Cluster" )
plt.ylabel("Calinski mesure")
plt.show()

plt.plot([i for i in range(1,50)],silhouette_data,c='r')
plt.axvline(i,c='r')
plt.xlabel("Nombre de Cluster" )
plt.ylabel("Silhouette mesure")
plt.show()

plt.plot([i for i in range(1,50)],davins_data,c='y')
plt.axvline(i,c='r')
plt.xlabel("Nombre de Cluster" )
plt.ylabel("Davins mesure")
plt.show()


plt.plot([i for i in range(1,50)],run_time_data,c='y')
plt.axvline(i,c='r')
plt.xlabel("Nombre de Cluster" )
plt.ylabel("Runtime")
plt.show()

print(historique[i-1][2])

plt.scatter(f0, f1, c=historique[i-1][1], s=8)
plt.scatter(historique[i-1][0][:,0],historique[i-1][0][:,1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après supposé bon nombre de cluster : "+ str(name) + " - Nb clusters ="+ str(i))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()