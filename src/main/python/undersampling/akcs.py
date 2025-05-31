"""
AKCS 
"""
from src.main.python.undersampling.base import InstanceSelectionMixin
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.validation import check_X_y
from sklearn.metrics import pairwise_distances
from sklearn.utils import safe_indexing
from sklearn.cluster import KMeans
import numpy as np
import random
import math

class AKCS(InstanceSelectionMixin):
    """ Method (AKCS)
    Descrição:
    ==========
    Parametros:
    ===========
    Atributos:
    ==========
    Ref.
    ====
    
    """
    def __init__(self, sampling_strategy='auto'):
        self.sample_indices_ = []
        self.sampling_strategy = sampling_strategy

    def __get_classes_numbers(self, y):
        #Conta quantos elementos tem em cada casa
        contagem_classes = {}
        for item in y:
            if item in contagem_classes:
                contagem_classes[item] += 1
            else:
                contagem_classes[item] = 1

        min_classe = min(contagem_classes, key=contagem_classes.get) #classe minoritaria

        max_classe = max(contagem_classes, key=contagem_classes.get)  # classe majoritária
        
        return min_classe, max_classe, contagem_classes

    def __equation2(self, C_line, N_max, N_min):
        res = math.ceil((N_min * C_line) / N_max)
        return res >=2

    def __N_sampling(self, C_line, N_max, N_min):
        # res = math.ceil((N_min * C_line) / N_max)
        res = round((N_min * C_line) / N_max)

        if res == 1:
            res+=1
        
        return res

    def __AKmeans(self, X, N_min):

        # N_att = len(X)
        N_att = X.shape[0]

        k = 2
        while True:
            
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)

            clusters = kmeans.labels_

            len_Clusters = {}
            for item in clusters:
                if item in len_Clusters:
                    len_Clusters[item] += 1
                else:
                    len_Clusters[item] = 1

            min_cluster = min(len_Clusters, key=len_Clusters.get) #cluster com menos elementos

            eq2 = self.__equation2(len_Clusters[min_cluster], N_att, N_min)

            if (not eq2) or (k+1 > N_min): 
                break

            k+=1

            # break
        
        k-=1
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)

        clusters = kmeans.labels_
        centroids = kmeans.cluster_centers_

        len_Clusters = {}
        for cluster in range(k):
            len_Clusters[cluster] = 0

        for item in clusters:
            len_Clusters[item] += 1

        n_select = {}
        for cluster in range(k):
            n_select[cluster] = self.__N_sampling(len_Clusters[cluster], N_att, N_min)
        

        return k, clusters, centroids, n_select

    def __cosine_based_selection(self, X, k, clusters, centroids, n_select):
        idx_s = []

        for cluster in range(k):
            translate_array_cluster = np.where(clusters == cluster)[0]
            X_cluster = safe_indexing(X, translate_array_cluster)


            centroid_cluster = centroids[cluster].reshape(1, -1)

            similarities = cosine_similarity(X_cluster, centroid_cluster).flatten()

            sorted_indices = np.argsort(similarities)[::-1]

            for elemento in range(n_select[cluster]):
                idx_s.append(translate_array_cluster[sorted_indices[elemento]])

        idx_s.sort()
    
        return np.array(idx_s)

    def select_data(self, X, y):
        
        X, y = check_X_y(X, y, accept_sparse="csr")
                
        print("\tShape:",X.shape)

        idx_s = []

        min_classe, max_classe, contagem_classes = self.__get_classes_numbers(y)

        translate_array = np.where(y == max_classe)[0]
        X_maj = safe_indexing(X, translate_array)

        print("\tAdaptative Kmeans", end=" .", flush=True)
        k, clusters, centroids, n_select = self.__AKmeans(X_maj, contagem_classes[min_classe])
        print("..ok")

        print("\tCosine-based selection", end=" .", flush=True)
        idx_maj_nao_traduzido = self.__cosine_based_selection(X_maj, k, clusters, centroids, n_select)
        print("..ok")

        idx_maj = translate_array[idx_maj_nao_traduzido]

        idx_min = np.where(y == min_classe)[0]

        idx_s = np.concatenate((idx_min, idx_maj))

        self.X_ = np.asarray(X[idx_s])
        self.y_ = np.asarray(y[idx_s])
        self.sample_indices_ = list(sorted(idx_s))
       
        self.reduction_ = 1.0 - float(len(self.y_))/len(y)
        # exit()
        return self.X_, self.y_