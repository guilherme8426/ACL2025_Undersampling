
"""
CNN 
"""

from src.main.python.undersampling.base import InstanceSelectionMixin
import numpy as np
import random
from sklearn.utils.validation import check_X_y
from sklearn.neighbors.classification import KNeighborsClassifier
from imblearn.under_sampling import CondensedNearestNeighbour 

class CNN(InstanceSelectionMixin):
    """ Condensed Nearest Neighbour (CNN)
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

    def get_satrategy_array(self, y):

        if self.sampling_strategy not in ["mediana", "average"]: return

    def select_data(self, X, y):
        
        X, y = check_X_y(X, y, accept_sparse="csr")

        idx_s = []

        self.get_satrategy_array(y)

        cnn = CondensedNearestNeighbour(random_state=42, sampling_strategy=self.sampling_strategy)
        cnn.fit_resample(X, y)

        idx_s = cnn.sample_indices_

        self.X_ = np.asarray(X[idx_s])
        self.y_ = np.asarray(y[idx_s])
        self.sample_indices_ = list(sorted(idx_s))
       
        #print(sorted(idx_prots_s))
        #print(float(len(self.y_))/len(y))

        self.reduction_ = 1.0 - float(len(self.y_))/len(y)
        return self.X_, self.y_
    