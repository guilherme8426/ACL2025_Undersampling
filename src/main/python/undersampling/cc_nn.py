
"""
CLUSTER CENTROIDS
"""

from src.main.python.undersampling.base import InstanceSelectionMixin
import numpy as np
import random
from sklearn.utils.validation import check_X_y
from sklearn.neighbors.classification import KNeighborsClassifier
from imblearn.under_sampling import ClusterCentroids
from sklearn.neighbors import NearestNeighbors
# from imblearn.under_sampling import AllKNN
# from imblearn.under_sampling import InstanceHardnessThreshold
# from sklearn.linear_model import LogisticRegression
from sklearn.utils import safe_indexing

class CC_NN(InstanceSelectionMixin):
    """ Edited Nearest Neighbours (CC_NN)
    Descrição:
    ==========


    Parametros:
    ===========


    Atributos:
    ==========


    Ref.
    ====
    
    """

    def __init__(self, n_neighbors=1, sampling_strategy='auto'):
        self.n_neighbors = n_neighbors
        self.classifier = None
        self.sample_indices_ = []
        self.sampling_strategy = sampling_strategy

    def select_data(self, X, y):
        
        X, y = check_X_y(X, y, accept_sparse="csr")

        # X = X.toarray()

        idx_s = np.array([], dtype=int)

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

        limit = min(contagem_classes[max_classe] - contagem_classes[min_classe], contagem_classes[max_classe] - int(len(y)*0.5))
        aux = {}
        for chave, valor in contagem_classes.items():
            if valor > limit: aux[chave] = int(limit)

        self.sampling_strategy = aux

        cc = ClusterCentroids(voting='hard', random_state=0, sampling_strategy=self.sampling_strategy)
        
        X_sampled, y_sampled = cc.fit_resample(X, y) #elementos

        # acha o indice certo
        for class_att in dataset_classes:

            #array de tradução que cada index indica o index no vetor global
            translate_array = []
            X_Real_classe_idx = []
            X_sampled_classe_idx = []

            #pega todos os elementos da classe e cria o vetor de tradução
            for idx, class_analysi in enumerate(y):
                if class_analysi == class_att:
                    translate_array.append(idx)
                    # X_Real_classe_idx.append(X[idx])
                    X_Real_classe_idx.append(idx)

            for idx, class_analysi in enumerate(y_sampled):
                if class_analysi == class_att:
                    # X_sampled_classe_idx.append(X_sampled[idx])
                    X_sampled_classe_idx.append(idx)

            nearest_neighbors = NearestNeighbors(n_neighbors=1)
            nearest_neighbors.fit(safe_indexing(X, X_Real_classe_idx))
            indices_nao_traduzido = nearest_neighbors.kneighbors(safe_indexing(X_sampled ,X_sampled_classe_idx), return_distance=False)

            result = [translate_array[indice] for indice in np.squeeze(indices_nao_traduzido)]
            
            idx_s = np.append(idx_s, result)   
            

        self.X_ = np.asarray(X[idx_s])
        self.y_ = np.asarray(y[idx_s])
        self.sample_indices_ = list(sorted(idx_s))
       
        #print(sorted(idx_prots_s))
        #print(float(len(self.y_))/len(y))

        self.reduction_ = 1.0 - float(len(self.y_))/len(y)
        return self.X_, self.y_
        