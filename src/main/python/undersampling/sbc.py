
"""
SBC 
"""

from src.main.python.undersampling.base import InstanceSelectionMixin
import numpy as np
import random
from sklearn.utils.validation import check_X_y
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.utils import safe_indexing
from sklearn.cluster import KMeans
import random

class SBC(InstanceSelectionMixin):
    """ SBC (sbc)
    Descrição:
    ==========


    Parametros:
    ===========


    Atributos:
    ==========


    Ref.
    ====
    
    """

    def __init__(self, n_neighbors=1, n_ratio=1, n_clusters=3, sampling_strategy='auto'):
        self.n_neighbors = n_neighbors
        self.classifier = None
        self.sample_indices_ = []

        self.n_ratio = n_ratio
        self.n_clusters = n_clusters
        self.sampling_strategy = sampling_strategy

    def _get_clusters(self, X):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(X)

        labels = kmeans.labels_

        cluster_indices = [[] for _ in range(max(labels) + 1)]

        for i, label in enumerate(labels):
            cluster_indices[label].append(i)
            
        return cluster_indices

    def select_data(self, X, y):
        
        X, y = check_X_y(X, y, accept_sparse="csr")

        # X = X.toarray() #AQUI pode dar errado
        # if self.classifier == None:
        #     self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        idx_s = np.array([], dtype=int)
        """
        Implementação aqui
        """

        #vetor com as classes
        dataset_classes = np.unique(y) #O(n)

        #Conta quantos elementos tem em cada casa
        contagem_classes = {}
        for item in y:
            if item in contagem_classes:
                contagem_classes[item] += 1
            else:
                contagem_classes[item] = 1

        min_classe = min(contagem_classes, key=contagem_classes.get) #classe minoritaria
        max_classe = max(contagem_classes, key=contagem_classes.get)  # classe majoritária
        n_min_classe = contagem_classes[min_classe]

        # acha o indice certo
        for class_att in dataset_classes:
            
            if(class_att == min_classe): 
                idx_classe_n = []
                for idx, class_analysi in enumerate(y):
                    if class_analysi == class_att:
                        idx_classe_n.append(idx)
                idx_s = np.append(idx_s, idx_classe_n) 
                continue #não tira da classe minoritaia

            #O if aqui
            if(class_att != max_classe) and (self.sampling_strategy == 'majority'):
                print("SO A MAIOR DESSA VEZ")
                idx_classe_n = []
                for idx, class_analysi in enumerate(y):
                    if class_analysi == class_att:
                        idx_classe_n.append(idx)
                idx_s = np.append(idx_s, idx_classe_n) 
                continue  


            #array de tradução que cada index indica o index no vetor global
            translate_array = []
            X_classe_idx = []

            #pega todos os elementos da classe e cria o vetor de tradução
            for idx, class_analysi in enumerate(y):
                if class_analysi == class_att or class_analysi == min_classe:
                    translate_array.append(idx)
                    # X_classe_idx.append(X[idx])
                    X_classe_idx.append(idx)

            clusters = self._get_clusters(safe_indexing(X, X_classe_idx))

            # Calcula o total ratio
            total_ratio = 0
            for cluster_elements in clusters:
                n_min = 0
                n_max = 0
                for element in cluster_elements:
                    if y[translate_array[element]] == min_classe:
                        n_min+=1
                    else:
                        n_max+=1
                
                if n_min!=0:
                    total_ratio += (n_max/n_min)
                else:
                    total_ratio += (n_max)
            
            # number_selecte_samples = self.n_ratio*n_min_classe
            number_selecte_samples = self.n_ratio*n_min_classe

            for cluster_elements in clusters:
                n_min = 0
                n_max = 0
                majority_elements = []
                for element in cluster_elements:
                    if y[translate_array[element]] == min_classe:
                        n_min+=1
                    else:
                        majority_elements.append(translate_array[element])
                        n_max+=1

                if n_min!=0:
                    n_sampling = int(number_selecte_samples*(n_max/n_min)/total_ratio)
                else:
                    n_sampling = int(number_selecte_samples*(n_max)/total_ratio)

                # sampled_elements = random.sample(population=majority_elements, k=n_sampling)
                sampled_elements = random.sample(population=majority_elements, k=min(n_max, n_sampling))
                idx_s = np.append(idx_s, sampled_elements) 
                   

        """
        Fim Implementação
        """
        idx_s = idx_s.astype(int)
        self.X_ = np.asarray(X[idx_s])
        self.y_ = np.asarray(y[idx_s])
        self.sample_indices_ = list(sorted(idx_s))
       
        #print(sorted(idx_prots_s))
        #print(float(len(self.y_))/len(y))

        self.reduction_ = 1.0 - float(len(self.y_))/len(y)
        return self.X_, self.y_
    