
"""
NEARMISS
"""

from src.main.python.undersampling.base import InstanceSelectionMixin
import numpy as np
import random
from sklearn.utils.validation import check_X_y
from sklearn.neighbors.classification import KNeighborsClassifier
from imblearn.under_sampling import NearMiss

class NEARMISS(InstanceSelectionMixin):
    """ Edited Nearest Neighbours (NEARMISS)
    Descrição:
    ==========


    Parametros:
    ===========


    Atributos:
    ==========


    Ref.
    ====
    
    """

    def __init__(self, version=1, n_neighbors=1, sampling_strategy='auto'):
        self.n_neighbors = n_neighbors
        self.classifier = None
        self.sample_indices_ = []
        self.version = version
        self.sampling_strategy = sampling_strategy

    def get_satrategy_array(self, y):

        #Conta quantos elementos tem em cada casa
        contagem_classes = {}
        for item in y:
            if item in contagem_classes:
                contagem_classes[item] += 1
            else:
                contagem_classes[item] = 1

        min_classe = min(contagem_classes, key=contagem_classes.get) # classe minoritaria
        max_classe = max(contagem_classes, key=contagem_classes.get)  # classe majoritária
        limit = min(contagem_classes[max_classe] - contagem_classes[min_classe], contagem_classes[max_classe] - int(len(y)*0.5))

        aux = {}
        for chave, valor in contagem_classes.items():
            if valor > limit: aux[chave] = int(limit)

        self.sampling_strategy = aux

    def select_data(self, X, y):
        
        X, y = check_X_y(X, y, accept_sparse="csr")

        idx_s = []

        self.get_satrategy_array(y)

        nearmiss = NearMiss(version=self.version, sampling_strategy=self.sampling_strategy)
        nearmiss.fit_resample(X, y) #Pode retornar X, y que ficaram

        idx_s = nearmiss.sample_indices_
        
        self.X_ = np.asarray(X[idx_s])
        self.y_ = np.asarray(y[idx_s])
        self.sample_indices_ = list(sorted(idx_s))
       
        #print(sorted(idx_prots_s))
        #print(float(len(self.y_))/len(y))

        self.reduction_ = 1.0 - float(len(self.y_))/len(y)
        return self.X_, self.y_
    