import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics


###################################################################
# Exemple : Agglomerative Clustering


path = './artificial/'
name="spiral.arff"

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


#### Dendogramme fonction##############""
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix) #, **kwargs)



### FIXER la distance
# 
link=["average","complete","ward","single"]

nb_cluster = []

average_data = []

complete_data = []

ward_data = []

single_data = []

data = [] #tableau ou on range nos labels 

for i in link:
    

    
    calinski_met = []
    silhouette_met = []
    davins_met = []

    seuil_dist_tab  = []
    Tab= [i for i in np.arange(0.5,25,0.5)]

    for j in Tab:
        seuil_dist_tab.append(j)
        print("--------------"+i+"--------------" +"Seuil_dist:" + str(j) + "-----------------")
        tps1 = time.time()
        seuil_dist=j
        model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage=i, n_clusters=None)
        model = model.fit(datanp)
        tps2 = time.time()
        labels = model.labels_

        
        # Nb iteration of this method
        #iteration = model.n_iter_

        k = model.n_clusters_
        leaves=model.n_leaves_

        # plt.scatter(f0, f1, c=labels, s=8)
        # plt.title("Clustering agglomératif (average, distance_treshold= "+str(seuil_dist)+") "+str(name))
        # plt.show()

        print("nb clusters =",k,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")
        if(k !=1 ):
            
            resu= metrics.calinski_harabasz_score(datanp,labels)
            print("Kalinski_Harabasz 1:")
            print(resu)
            calinski_met.append(resu)


            resu = metrics.silhouette_score(datanp, labels)
            print("Indice de silhouette 1:")
            print(resu)
            silhouette_met.append(resu)

            resu = metrics.davies_bouldin_score(datanp, labels)
            print("Indice de davies bouldin 1:")
            print(resu)
            davins_met.append(resu)
        else:
            calinski_met.append(0)
            silhouette_met.append(0)
            davins_met.append(0)
        data.append([i,j,model,tps2-tps1,calinski_met,silhouette_met,davins_met])

        

    

    if i=='average':
        average_data.append(calinski_met)
        average_data.append(silhouette_met)
        average_data.append(davins_met)
        average_data.append(seuil_dist_tab)

    if i=='complete':
        complete_data.append(calinski_met)
        complete_data.append(silhouette_met)
        complete_data.append(davins_met)
        complete_data.append(seuil_dist_tab)

    if i=='ward':
        ward_data.append(calinski_met)
        ward_data.append(silhouette_met)
        ward_data.append(davins_met)
        ward_data.append(seuil_dist_tab)
    if i=='single':
        single_data.append(calinski_met)
        single_data.append(silhouette_met)
        single_data.append(davins_met)
        single_data.append(seuil_dist_tab)

plt.figure("Calinski")        
plt.plot(average_data[3], average_data[0],label="average", color='r')
plt.plot(complete_data[3], complete_data[0],label="complete", color='b')
plt.plot(ward_data[3], ward_data[0],label="ward", color='g')
plt.plot(single_data[3], single_data[0],label="single", color='m')
plt.title("Clustering agglomératif (Calinski, distance_treshold= "+str(seuil_dist)+") "+str(name))
plt.xlabel("Distance treshold")
plt.ylabel("Resultat Metric")
plt.legend()


plt.figure("Silhouette")
plt.plot(average_data[3], average_data[1],label="average", color='r')
plt.plot(complete_data[3], complete_data[1],label="complete", color='b')
plt.plot(ward_data[3], ward_data[1],label="ward", color='g')
plt.plot(single_data[3], single_data[1],label="single", color='m')
plt.title("Clustering agglomératif (Silhouette, distance_treshold= "+str(seuil_dist)+") "+str(name))
plt.xlabel("Distance treshold")
plt.ylabel("Resultat Metric")
plt.legend()

plt.figure("Davins")
plt.plot(average_data[3], average_data[2],label="average", color='r')
plt.plot(complete_data[3], complete_data[2],label="complete", color='b')
plt.plot(ward_data[3], ward_data[2],label="ward", color='g')
plt.plot(single_data[3], single_data[2],label="single", color='m')
plt.title("Clustering agglomératif (Davins, distance_treshold= "+str(seuil_dist)+") "+str(name))
plt.xlabel("Distance treshold")
plt.ylabel("Resultat Metric")
plt.legend()
plt.show()

a = input("Choisissez un seuil de distance ")
for i in link:
    print("--------------"+i+"--------------" +"Seuil_dist:" + str(a) + "-----------------")
    tps1 = time.time()
    model = cluster.AgglomerativeClustering(distance_threshold=float(a), linkage=i, n_clusters=None)
    model = model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    
   # iteration = model.n_iter_

    k = model.n_clusters_
    leaves=model.n_leaves_
    plot_dendrogram(model)    
    plt.show()

met = input("Selectionner la méthode :  ")
clu=input("Nombre de cluster : ")
tps1 = time.time()
model = cluster.AgglomerativeClustering(distance_threshold=None, linkage=met, n_clusters=int(clu))
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_

# iteration = model.n_iter_

k = model.n_clusters_
leaves=model.n_leaves_

plt.scatter(f0, f1, c=labels, s=8)
plt.show()




