"""
ENU 
"""
from src.main.python.undersampling.base import InstanceSelectionMixin
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.validation import check_X_y
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.utils import safe_indexing
from sklearn.cluster import KMeans
import numpy as np
import random
import math

class ENU(InstanceSelectionMixin):
    """ Method (ENU)
    Descrição:
    ==========
    Parametros:
    ===========
    Atributos:
    ==========
    Ref.
    ====
    
    """
    def __init__(self, sampling_strategy='auto', estrategia="ENUT"):
        self.sample_indices_ = []
        self.sampling_strategy = sampling_strategy

        if estrategia=="ENUB":
            self.ENU_sel = self.ENUB
        if estrategia=="ENUT":
            self.ENU_sel = self.ENUT
        if estrategia=="ENUC":
            self.ENU_sel = self.ENUC
        if estrategia=="ENUR":
            self.ENU_sel = self.ENUR
        self.classMaj = None        

    def local_density_estimation(self, class_C):

        # Listas para armazenar densidades locais e distâncias de corte
        N = class_C.shape[0]  # Número de instâncias na classe C
        LDX = np.zeros(N)  # Densidade local de cada ponto
        DCX = np.zeros(N)  # Distâncias de corte
        # omega = int(0.015 * N)  # Valor de omega (entre 1% e 2% de N)
        omega = int(0.02 * N)  # Valor de omega (entre 1% e 2% de N)

        dist_matrix = pairwise_distances(class_C, metric='euclidean')

        # Calculando densidade local de cada ponto na classe C
        for i in range(N):
            LDi = 0
            dc = np.sort(dist_matrix[i])[omega]

            for j in range(N):
                if i==j: continue
                
                d = dist_matrix[i][j]
                LDi = LDi + np.exp(-(d**2 / dc**2))
                if dc == 0: print("Divisao por 0")
            
            LDX[i] = LDi
            DCX[i] = dc

        # Pode ter erro aqui no pseudo codigo deles?
        wi = 0
        sumWi = 0
        for i in range(N):
            wi = wi+(1/DCX[i])
            sumWi = sumWi+wi
        
        LDc = 0
        for i in range(N):
            LDc = LDc + ((LDX[i]*wi)/(sumWi))
        
        return LDc

    def compute_k_imb(self, N, nMin, nMaj, LDMin, LDMaj):
        imb = nMaj/nMin
        alpha = LDMaj/LDMin

        K = np.sqrt(N) + np.sqrt(imb)
        K_imb = K*alpha

        return K_imb

    def __compute_initial_membership(self, X, y, m=2):
        # if not isinstance(X, csr_matrix):
        #     X = csr_matrix(X)
        
        classes = np.unique(y)
        centroids = {c: X[y == c].mean(axis=0).A1 for c in classes}  # Convertendo para array unidimensional
        
        membership = np.zeros((X.shape[0], len(classes)))
        
        for i in range(X.shape[0]):
            xi = X[i].toarray().flatten()
            # denom = sum(1 / (np.linalg.norm(xi - centroids[c]) ** (2 / (m - 1))) for c in classes)
            denom = sum(1 / (np.linalg.norm(xi - centroids[c]) ** (2 / (m - 1)) + 1e-10) for c in classes)  # Evita divisão por zero

            
            for j, c in enumerate(classes):
                # membership[i, j] = (1 / (np.linalg.norm(xi - centroids[c]) ** (2 / (m - 1)))) / denom
                membership[i, j] = (1 / (np.linalg.norm(xi - centroids[c]) ** (2 / (m - 1)) + 1e-10)) / denom

        
        return membership

    def __compute_fknn(self, X_train, y_train, X_Maj, K, m=2):
        
        neigh = NearestNeighbors(n_neighbors=K, metric='euclidean')
        neigh.fit(X_train)
        
        initial_membership = self.__compute_initial_membership(X_train, y_train, m)
        
        fuzzy_memberships = np.zeros((X_Maj.shape[0], len(np.unique(y_train))))
        
        for i in range(X_Maj.shape[0]):
            x = X_Maj[i].toarray().flatten()
            distances, indices = neigh.kneighbors([x])
            distances = distances.flatten()
            indices = indices.flatten()
            
            distances = np.maximum(distances, 1e-10)
            denom = np.sum(1 / (distances ** (2 / (m - 1))))
            
            for j in range(len(fuzzy_memberships[i])):
                fuzzy_memberships[i, j] = np.sum((1 / (distances ** (2 / (m - 1)))) * initial_membership[indices, j]) / denom
        
        return fuzzy_memberships

    def __compute_information_entropy(self, fuzzy_memberships):
        entropy = -np.sum(fuzzy_memberships * np.log2(fuzzy_memberships + 1e-10), axis=1)  # Pequeno valor para evitar log(0)
        return entropy

    def computeIE(self, X_train, y_train, X_Maj, K, m=2):
        fknn_memberships = self.__compute_fknn(X_train, y_train, X_Maj, K, m)
        entropy_scores = self.__compute_information_entropy(fknn_memberships)

        return entropy_scores

    def computeIETH(self, X, IE):
        # Calcular o centroide (média dos elementos ao longo de cada coluna)
        centroide = [X.mean(axis=0).A1]

        # # Usar NearestNeighbors para calcular a distância euclidiana até o centroide
        neigh = NearestNeighbors(n_neighbors=1, metric='euclidean')
        neigh.fit(centroide)  # Converte csr_matrix para denso e ajusta

        # # Calcular as distâncias
        distancias, _ = neigh.kneighbors(X)
        distancias = distancias.flatten()
        w = 1/distancias
        wl = w/np.sum(w)
        
        IETH = np.sum(wl*IE)

        return IETH

    def ENUBT(self, X, y, X_filtro_IE, X_min, K_imb, nMaj, nMin):
        return [0, 9]
    
    def ENUB(self, X, y, translate_array_X_Maj_filtro_IE, translate_array_X_Min, K_imb, nMaj, nMin):
        print("ENUB")
        X_maj_IE = safe_indexing(X, translate_array_X_Maj_filtro_IE)
        # X_min = safe_indexing(X, translate_array_X_Min)

        knn = NearestNeighbors(n_neighbors=K_imb+1)  # Pegamos um a mais para pq a propria instancia esta no treino
        knn.fit(X)

        distances, indices = knn.kneighbors(X_maj_IE)
        distances, indices = distances[:, 1:], indices[:, 1:] # Removendo o primeiro vizinho q é o proprio ponto
        
        idx_s = []
        nIdx_s = 0
        for i in range(X_maj_IE.shape[0]):
            y_NN = y[indices[i]]
            
            if self.classMaj in y_NN:
                idx_s.append(translate_array_X_Maj_filtro_IE[i])
                nIdx_s+=1

            imb = (nMaj-nIdx_s)/nMin

            if imb <= 1:
                break
        
        return np.array(idx_s)

    def ENUT(self, X, y, translate_array_X_Maj_filtro_IE, translate_array_X_Min, K_imb, nMaj, nMin):
        print("ENUT")
        X_maj_IE = safe_indexing(X, translate_array_X_Maj_filtro_IE)
        X_min = safe_indexing(X, translate_array_X_Min)

        knn = NearestNeighbors(n_neighbors=K_imb+1)  # Pegamos um a mais para pq a propria instancia esta no treino
        knn.fit(X)

        distances_maj_IE, indices_maj_IE = knn.kneighbors(X_maj_IE)
        distances_maj_IE, indices_maj_IE = distances_maj_IE[:, 1:], indices_maj_IE[:, 1:] # Removendo o primeiro vizinho q é o proprio ponto

        distances_min, indices_min = knn.kneighbors(X_min)
        distances_min, indices_min = distances_min[:, 1:], indices_min[:, 1:] # Removendo o primeiro vizinho q é o proprio ponto

        idx_s = []
        nIdx_s = 0
        for i in range(X_maj_IE.shape[0]):
            # y_NN = y[indices_maj_IE[i]]
            
            for j in indices_maj_IE[i]:
                if y[j] == self.classMaj: continue

                idx_in_Min_classe_set = np.where(translate_array_X_Min == j)[0]

                if translate_array_X_Maj_filtro_IE[i] in indices_min[idx_in_Min_classe_set]:
                    idx_s.append(translate_array_X_Maj_filtro_IE[i])
                    nIdx_s+=1
                    break

            imb = (nMaj-nIdx_s)/nMin

            if imb <= 1:
                break
        
        return np.array(idx_s)

    def ENUC(self, X, y, translate_array_X_Maj_filtro_IE, translate_array_X_Min, K_imb, nMaj, nMin):
        print("ENUC")
        X_maj_IE = safe_indexing(X, translate_array_X_Maj_filtro_IE)
        X_min = safe_indexing(X, translate_array_X_Min)

        F = np.zeros(X_maj_IE.shape[0])

        knn = NearestNeighbors(n_neighbors=K_imb+1)  # Pegamos um a mais para pq a propria instancia esta no treino
        knn.fit(X)

        # distances_maj_IE, indices_maj_IE = knn.kneighbors(X_maj_IE)
        # distances_maj_IE, indices_maj_IE = distances_maj_IE[:, 1:], indices_maj_IE[:, 1:] # Removendo o primeiro vizinho q é o proprio ponto

        distances_min, indices_min = knn.kneighbors(X_min)
        distances_min, indices_min = distances_min[:, 1:], indices_min[:, 1:] # Removendo o primeiro vizinho q é o proprio ponto

        for j in range(X_min.shape[0]):
            for i in indices_min[j]:
                if y[i] != self.classMaj: continue
                idx_i_IE_arr = np.where(translate_array_X_Maj_filtro_IE == i)[0]
                if idx_i_IE_arr.shape[0] < 1: continue

                F[idx_i_IE_arr] += 1

        idx_s = []
        nIdx_s = 0
        for i in range(X_maj_IE.shape[0]):
            
            if F[i] > 0:
                idx_s.append(translate_array_X_Maj_filtro_IE[i])
                nIdx_s+=1

            imb = (nMaj-nIdx_s)/nMin

            if imb <= 1:
                break
        
        return np.array(idx_s)

    def __ENUC_aux_ENUR(self, X, y, translate_array_X_Maj_filtro_IE, translate_array_X_Min, K_imb, nMaj, nMin):

        X_maj_IE = safe_indexing(X, translate_array_X_Maj_filtro_IE)
        X_min = safe_indexing(X, translate_array_X_Min)

        F = np.zeros(X_maj_IE.shape[0])

        knn = NearestNeighbors(n_neighbors=K_imb+1)  # Pegamos um a mais para pq a propria instancia esta no treino
        knn.fit(X)

        # distances_maj_IE, indices_maj_IE = knn.kneighbors(X_maj_IE)
        # distances_maj_IE, indices_maj_IE = distances_maj_IE[:, 1:], indices_maj_IE[:, 1:] # Removendo o primeiro vizinho q é o proprio ponto

        distances_min, indices_min = knn.kneighbors(X_min)
        distances_min, indices_min = distances_min[:, 1:], indices_min[:, 1:] # Removendo o primeiro vizinho q é o proprio ponto

        for j in range(X_min.shape[0]):
            for i in indices_min[j]:
                if y[i] != self.classMaj: continue
                idx_i_IE_arr = np.where(translate_array_X_Maj_filtro_IE == i)[0]
                if idx_i_IE_arr.shape[0] < 1: continue

                F[idx_i_IE_arr] += 1

        idx_s = []
        nIdx_s = 0
        for i in range(X_maj_IE.shape[0]):
            
            if F[i] > 0:
                idx_s.append(i)
                nIdx_s+=1

            imb = (nMaj-nIdx_s)/nMin

            if imb <= 1:
                break
        
        return np.array(idx_s) # idx em relação ao conjunto X maj filtrado IE

    def ENUR(self, X, y, translate_array_X_Maj_filtro_IE, translate_array_X_Min, K_imb, nMaj, nMin):
        print("ENUR")
        # idx em relação ao vetor so com as instancias maj IE
        idx_ENUC = self.__ENUC_aux_ENUR(X, y, translate_array_X_Maj_filtro_IE, translate_array_X_Min, K_imb, nMaj, nMin)

        translate_array_X_Maj_ENUC = safe_indexing(translate_array_X_Maj_filtro_IE, idx_ENUC)

        X_maj_IE_ENUC = safe_indexing(X, translate_array_X_Maj_ENUC)
        X_maj_IE = safe_indexing(X, translate_array_X_Maj_filtro_IE)

        F = np.zeros(X_maj_IE.shape[0])

        knn = NearestNeighbors(n_neighbors=K_imb+1)  # Pegamos um a mais para pq a propria instancia esta no treino
        knn.fit(X)

        distances_maj_ENUC, indices_maj_ENUC = knn.kneighbors(X_maj_IE_ENUC)
        distances_maj_ENUC, indices_maj_ENUC = distances_maj_ENUC[:, 1:], indices_maj_ENUC[:, 1:] # Removendo o primeiro vizinho q é o proprio ponto

        for j in range(X_maj_IE_ENUC.shape[0]):
            for i in indices_maj_ENUC[j]:
                if y[i] != self.classMaj: continue
                idx_i_IE_arr = np.where(translate_array_X_Maj_filtro_IE == i)[0]
                if idx_i_IE_arr.shape[0] < 1 or idx_i_IE_arr in idx_ENUC: continue

                F[idx_i_IE_arr] += 1

        idx_s = translate_array_X_Maj_ENUC.tolist()
        nIdx_s = len(idx_s)
        for i in range(X_maj_IE.shape[0]):
            
            if F[i] > 0:
                idx_s.append(translate_array_X_Maj_filtro_IE[i])
                nIdx_s+=1

            imb = (nMaj-nIdx_s)/nMin

            if imb <= 1:
                break
        
        return np.unique(idx_s)

    def eliminationMajorityInstance(self, X, y, translate_array_X_Maj, translate_array_X_Min, K_imb, IE, IETH):
        idx_X_maj_filtro_IE = np.where(IE < IETH)[0]

        translate_array_X_Maj_filtro_IE = safe_indexing(translate_array_X_Maj, idx_X_maj_filtro_IE)

        idx = self.ENU_sel(X, y, translate_array_X_Maj_filtro_IE, translate_array_X_Min, K_imb, translate_array_X_Maj.shape[0], translate_array_X_Min.shape[0])

        return idx

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

    def select_data(self, X, y):
        
        X, y = check_X_y(X, y, accept_sparse="csr")
        
        # X = X.toarray()
        
        idx_s = []

        N = X.shape[0]

        min_classe, max_classe, contagem_classes = self.__get_classes_numbers(y)
        print("Maj:", contagem_classes[max_classe], "Min:", contagem_classes[min_classe])

        self.classMaj = max_classe

        translate_array_maj = np.where(y == max_classe)[0]
        X_maj = safe_indexing(X, translate_array_maj)

        translate_array_min = np.where(y == min_classe)[0]
        X_min = safe_indexing(X, translate_array_min)

        LDMaj = self.local_density_estimation(X_maj)
        LDMin = self.local_density_estimation(X_min)        

        K_imb = self.compute_k_imb(N, X_min.shape[0], X_maj.shape[0], LDMin, LDMaj)
        K_imb = int(K_imb)

        IE = self.computeIE(X, y, X_maj, K_imb, m=2)
        IETH = self.computeIETH(X_maj, IE)

        idx_elimination = self.eliminationMajorityInstance(X, y, translate_array_maj, translate_array_min, K_imb, IE, IETH)

        mask = np.ones(y.size, dtype=bool)
        mask[idx_elimination] = False
        idx_s = np.asarray(range(len(y)))[mask]           

        self.X_ = np.asarray(X[idx_s])
        self.y_ = np.asarray(y[idx_s])
        self.sample_indices_ = list(sorted(idx_s))
       

        self.reduction_ = 1.0 - float(len(self.y_))/len(y)

        return self.X_, self.y_
