import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from kneed import KneeLocator

##################################################################
# Exemple : DBSCAN Clustering


path = './artificial/'
name="donutcurves.arff"

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

# #plt.figure(figsize=(6, 6))
# plt.scatter(f0, f1, s=8)
# plt.title("Donnees initiales : "+ str(name))
# #plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
# plt.show()


# # Run DBSCAN clustering method 
# # for a given number of parameters eps and min_samples
# # 
# print("------------------------------------------------------")
# print("Appel DBSCAN (1) ... ")
# tps1 = time.time()
# epsilon=2 #2  # 4
# min_pts= 5 #10   # 10
# model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
# model.fit(datanp)
# tps2 = time.time()
# labels = model.labels_

# # Number of clusters in labels, ignoring noise if present.
# n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise = list(labels).count(-1)
# print('Number of clusters: %d' % n_clusters)
# print('Number of noise points: %d' % n_noise)

# plt.scatter(f0, f1, c=labels, s=8)
# plt.title("Données après clustering DBSCAN (1) - Epislon= "+str(epsilon)+" MinPts= "+str(min_pts))
# plt.show()


####################################################
# Standardisation des donnees

#Strategie : Parcourir plusieurs valeurs de min_samples, pour decider d'un eps optimal 
# et mesurer les indices pour trouver le meileur clustering 


calinski_data = []
silhouette_data = []
davies_data = []
noise_ratios = []


epsilon=0.15 #0.05
min_pts= [i for i in range(1,10)]




scaler = preprocessing.StandardScaler().fit(datanp)
data_scaled = scaler.transform(datanp)
print("Affichage données standardisées            ")
f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne



#plt.figure(figsize=(10, 10))
plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Donnees standardisées")
plt.show()

nb_clusters = []
eps_record = []


print("------------------------------------------------------")
print("Appel DBSCAN (2) sur données standardisees ... ")
for j in min_pts: 

    neighbor = NearestNeighbors(n_neighbors=j)
    neighbor.fit(data_scaled)
    dist, i = neighbor.kneighbors(data_scaled)
    

    # diff = np.diff(dist)
    # epsilon= dist[np.argmax(diff) +1]
    #epsilon= np.mean(dist)

    #Plusieurs méthodes pour reperer le coude mais la librairie Kneedle reste la meilleure. 



    newDistances = np.asarray([np.average(dist[i][1:]) for i in range (0 ,
    dist.shape[0])])
    # trier par ordre croissant
    distancetrie = np.sort(newDistances)


    if j!=1:
        kneedle = KneeLocator(range(len(distancetrie)), distancetrie, curve="convex", direction="increasing")
        i = kneedle.elbow
        epsilon = distancetrie[kneedle.elbow]
        print(epsilon)

        if epsilon==0: 
            epsilon=0.05
            
        print("Eps : ", epsilon)
        plt.axvline(i,c='r')

        
    eps_record.append(epsilon)
    plt.plot(distancetrie)
    plt.xlabel("Points")
    plt.ylabel(f"Distance au {j}-ème plus proche voisin")
    plt.title(f"Analyse des plus proches voisins - Distance au {j}-ème voisin")
    plt.show()

    
    tps1 = time.time()
    model = cluster.DBSCAN(eps=epsilon, min_samples=j)
    model.fit(data_scaled)

    tps2 = time.time()
    labels = model.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print('Number of clusters: %d' % n_clusters)
    nb_clusters.append(n_clusters)
    print('Ratio of noise points: ' + str(int(n_noise)/len(list(labels))))

    noise_ratios.append(int(n_noise)/len(list(labels)))

    if n_clusters !=1: 
        calinski_data.append(metrics.calinski_harabasz_score(datanp,labels))
        silhouette_data.append(metrics.silhouette_score(datanp, labels))
        davies_data.append(metrics.davies_bouldin_score(datanp, labels))
    else: 
        calinski_data.append(0)
        silhouette_data.append(0)
        davies_data.append(0)



plt.figure("Calinski")        
plt.plot(min_pts,calinski_data ,label="Calinski", color='r')
plt.title("Indice de calinksi en fonction de min_samples ")
plt.xlabel("min_samples")
plt.ylabel("Resultat Metric")
plt.legend()


plt.figure("Indice de Silhouette")        
plt.plot(min_pts,silhouette_data ,label="Silhouette", color='b')
plt.title("Indice de silhouette en fonction de min_samples ")
plt.xlabel("min_samples")
plt.ylabel("Resultat Metric")
plt.legend()


plt.figure("Davies")        
plt.plot(min_pts,davies_data ,label="Davies", color='black')
plt.title("Indice de Davies en fonction de min_samples ")
plt.xlabel("min_samples")
plt.ylabel("Resultat Metric")
plt.legend()


plt.figure("Noise Ration")        
plt.plot(min_pts,noise_ratios ,label="Noise ratio", color='y')
plt.title("Noise Ration en fonction de min_samples ")
plt.xlabel("min_samples")
plt.ylabel("Resultat Metric")
plt.legend()



# plt.scatter(f0_scaled, f1_scaled, c=labels, s=8)
# plt.title("Données après clustering DBSCAN (2) - Epislon= "+str(epsilon)+" MinPts= "+str(j))
plt.show()

print("Nombre de clusters : " + str(nb_clusters))
print("Historique des Eps : " + str(eps_record))


