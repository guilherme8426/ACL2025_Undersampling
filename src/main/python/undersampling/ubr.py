
"""
UBR 
"""

from src.main.python.undersampling.base import InstanceSelectionMixin
import numpy as np
import random
from sklearn.utils.validation import check_X_y
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
from sklearn.utils import safe_indexing
from src.main.python.weakClassifiers.nmslibKNN import NMSlibKNNClassifier
import time

class UBR(InstanceSelectionMixin):
    """ Method (UBR)
    Descrição:
    ==========


    Parametros:
    ===========


    Atributos:
    ==========


    Ref.
    ====
    
    """

    def __init__(self, sampling_strategy='auto', alfa=5, beta=10):
        self.sample_indices_ = []
        
        self.sampling_strategy = sampling_strategy

        self.alfa = alfa
        self.beta = beta

    def get_classes_numbers(self, y):
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

    def difficulty_(self, y_1, y_2, cosine_similarity, alfa, beta):
        
        a = alfa - beta*cosine_similarity

        if y_1 != y_2:
            return 1 / (1 + np.exp(a))
        
        return 1 / (1 + np.exp(-a))

    def get_difficulty_array(self, X, y, alfa, beta, n_neighbors=10):


        classifier = NMSlibKNNClassifier(n_neighbors=min(n_neighbors+1, X.shape[0]), n_jobs=10)
        classifier.fit(X, y)

        queryResults = classifier.predict_and_dist(X)

        difficulty_array = np.zeros(X.shape[0])

        for X_idx in range(X.shape[0]):

            idx_nn  = queryResults[X_idx][0].tolist()
            dist_nn = queryResults[X_idx][1].tolist()

            X_difficulty = 0

            n_nn = n_neighbors
            i = 0
            while(n_nn>0):
                if X_idx == idx_nn[i]: 
                    i+=1
                    continue
                
                X_difficulty += self.difficulty_(y[X_idx], y[idx_nn[i]], 1-dist_nn[i], alfa, beta)

                i+=1
                n_nn-=1
            
            # quanto maior for o X_difficulty mais dificil é
            # 1-X_difficulty quanto mais dificil menor é
            difficulty_array[X_idx] = 1 - X_difficulty 
            # difficulty_array[X_idx] = X_difficulty 

        difficulty_array_norm = 1 - (difficulty_array - difficulty_array.min()) / (difficulty_array.max() - difficulty_array.min())

        return difficulty_array_norm

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def select_data(self, X, y):
        
        X, y = check_X_y(X, y, accept_sparse="csr")
        
        idx_s = np.array([], dtype=int)

        #vetor com as classes
        dataset_classes = np.unique(y) 

        min_classe, max_classe, contagem_classes = self.get_classes_numbers(y)

        X_size_global = X.shape[0]

        difficulty_array = self.get_difficulty_array(X, y, self.alfa, self.beta)


        idx_remove = np.array([], dtype=int)
        for class_att in dataset_classes:

            if(class_att == min_classe): continue #não tira da classe minoritaia

            
            #array de tradução que cada index indica o index no vetor global
            #pega todos os elementos da classe e cria o vetor de tradução
            translate_array = np.where(y == class_att)[0]

            difficulty_array_new = safe_indexing(difficulty_array, translate_array)

            X_size  = difficulty_array_new.shape[0]
            
            remove_size = min(X_size - contagem_classes[min_classe], int(X_size_global*0.5)) #Numero de elementos a serem removidos

            probability_array = difficulty_array_new / np.sum(difficulty_array_new)

            idx_remove_aux =  np.random.choice(a=translate_array,
                                               size=remove_size,
                                               replace=False,
                                               p=probability_array)

            idx_remove = np.concatenate((idx_remove, idx_remove_aux))

        mask = np.ones(y.size, dtype=bool)
        mask[idx_remove] = False
        idx_s = np.asarray(range(len(y)))[mask]           

        self.X_ = np.asarray(X[idx_s])
        self.y_ = np.asarray(y[idx_s])
        self.sample_indices_ = list(sorted(idx_s))
       

        self.reduction_ = 1.0 - float(len(self.y_))/len(y)

        return self.X_, self.y_
