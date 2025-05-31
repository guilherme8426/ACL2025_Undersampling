
"""
OBU 
"""

from src.main.python.undersampling.base import InstanceSelectionMixin
import numpy as np
import random
from sklearn.utils.validation import check_X_y
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.cluster import KMeans
import random
import skfuzzy 

class OBU(InstanceSelectionMixin):
    """ OBU (obu)
    Descrição:
    ==========


    Parametros:
    ===========


    Atributos:
    ==========


    Ref.
    ====
    
    """

    def __init__(self, n_neighbors=1, a_cut=0.5, sampling_strategy='auto'):
        self.n_neighbors = n_neighbors
        self.classifier = None
        self.sample_indices_ = []

        self.a_cut = a_cut

        self.sampling_strategy = sampling_strategy

    def _get_clusters(self, X):

        n_clusters = 2

        # Parâmetro de pertinência (fuzziness)
        m = 2.0

        # Número máximo de iterações
        max_iter = 100

        # Critério de parada
        error_tol = 1e-5

        # Aplicando o FCM
        clusters_fuzzy = skfuzzy.cmeans(X.T, n_clusters, m, error_tol, max_iter)

        # matriz de pertencimento
        clusters_fuzzy[1].T
            
        return clusters_fuzzy[1].T

    def select_data(self, X, y):
        
        X, y = check_X_y(X, y, accept_sparse="csr")

        X = X.toarray()

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
            X_classe = []

            #pega todos os elementos da classe e cria o vetor de tradução
            for idx, class_analysi in enumerate(y):
                if class_analysi == class_att or class_analysi == min_classe:
                    translate_array.append(idx)
                    X_classe.append(X[idx])

            clusters = self._get_clusters(np.array(X_classe))

            # Calcula o total ratio
            
            n_c0_minority = 0
            n_c0_majority = 0
            n_c1_minority = 0
            n_c1_majority = 0
            cluster_majority = 0
            for idx, cluster_elements in enumerate(clusters):
                # cluster 0
                if cluster_elements[0]>cluster_elements[1]:
                    if y[translate_array[idx]] == min_classe:
                        n_c0_minority += 1
                    else:
                        n_c0_majority += 1
                else:
                    if y[translate_array[idx]] == min_classe:
                        n_c1_minority += 1
                    else:
                        n_c1_majority += 1

            if(n_c0_majority > n_c1_majority):
                cluster_majority = 0
            else:
                cluster_majority = 1

            sampled_elements = []
            for idx, cluster_elements in enumerate(clusters):

                if y[translate_array[idx]] == min_classe: continue
                if cluster_elements[cluster_majority]<self.a_cut: continue

                # caso a instanciaseja daclasse não minoritaria e tenha uma probabilidade para
                # o cluster majoritario maior que a_cut ela fica no treinamento
                sampled_elements.append(translate_array[idx])

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