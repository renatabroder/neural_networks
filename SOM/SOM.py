import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def normaliza(dados):
    normal = MinMaxScaler()
    normalizados = normal.fit_transform(dados)
    normalizados = pd.DataFrame(normalizados)
    indices = dados.index
    normalizados.index = indices
    return normalizados

def tratamento(dataset, r):
    n_entradas, cols = dataset.shape

    # retiradada coluna de classes
    dataset = dataset.drop(columns=[cols-1])

    # randomizar dataset
    dataset = dataset.sample(frac=1, random_state=r)
    dataset = dataset.reset_index()
    dataset = dataset.drop(columns=['index'])

    # normalização do dataset
    dataset = normaliza(dataset)
    #print(dataset)

    # transformação do pandas dataset em um numpy array
    dataset = dataset.values

    return dataset, n_entradas

def distancia(a, b):
    return np.sqrt(np.sum([(a[i]-b[i])**2 for i in range(len(a))]))   

def criaGrafico(rede, dim):
    aux_grafico = np.zeros((dim, dim))
    for l in range(dim):
        for c in range(dim):
            aux_grafico[l][c] = math.sqrt(sum([a**2 for a in rede[l][c]]))
    plt.imshow(aux_grafico)
    plt.show()

def SOM(dataset, eta0=0.5, max_epocas=500, r=24):
    entradas, n_entradas = tratamento(dataset, r)

    dim = int(np.sqrt(np.sqrt(2) * n_entradas))
    np.random.seed(r)
    rede = np.random.uniform(low=-0.1, high=0.1, size=((dim, dim, len(entradas[0]))))

    eta = eta0
    sigma = sigma0 = None

    # criaGrafico(rede, dim)

    # TREINO
    for epoca in range(max_epocas):
        
        print("Época: ", epoca, "- ALFA: ", eta, " - SIGMA: ", sigma)

        for i in range(n_entradas):
            
            # etapa competitiva:
            distancias = []
            for l in range(dim):
                for c in range(dim):
                    distancias.append(distancia(entradas[i], rede[l][c]))
            minimo = np.argmin(distancias)

            lmin = (minimo // dim )
            cmin = (minimo % dim )

            # etapa cooperativa:
            d = []
            for l in range(dim):
                for c in range(dim):
                    d.append(distancia([lmin, cmin], [l, c]))
            
            if(sigma is None):
                # QUILES
                #sigma = sigma0 = math.sqrt(-(np.argmax(d)**2) / (2*math.log(0.1)))
                sigma = sigma0 = math.sqrt(-(dim**2) / (2*math.log(0.1)))
                tau = max_epocas/np.log(sigma0/0.1)


            # h = [np.exp((-d[j]**2)/(2*sigma**2)) for j in range(len(d))]
            h = [np.exp((-d[j]**2)/(2*sigma**2)) for j in range(len(d))]
            # h = [np.floor(np.exp((-d[j]**2)/(2*sigma**2))*100) for j in range(len(d))]
            h = np.reshape(h, (dim, dim))
            # print(np.floor(h*100))
            #print((h))
            #quit()

            #atualização dos pesos:
            for l in range(dim):
                for c in range(dim):
                    for p in range(len(entradas[i])):
                        rede[l][c][p] += eta * (entradas[i][p]-rede[l][c][p]) * h[l][c]

        eta = eta0 * np.exp(-epoca/tau)
        #sigma = sigma0 * np.exp(-epoca/tau)
        
    #criaGrafico(rede, dim)

    return rede, dim

def tratamentoTeste(dataset):
    n_entradas, cols = dataset.shape
    # retiradada coluna de classes
    rotulos = pd.factorize(dataset[cols-1])[0]
    dataset = normaliza(dataset.drop(columns=[cols-1])).values
    return dataset, rotulos, n_entradas

def rotulaNeuronio(dataset, rede, dim):
    
    dataset, rotulos, n_entradas = tratamentoTeste(dataset)
    rotuloNeuronio = []
    for l in range(dim):
        for c in range(dim):
            distancias = []
            for i in range(n_entradas):
                distancias.append(distancia(rede[l][c], dataset[i]))
            minimo = np.argmin(distancias)
            
            rotuloNeuronio.append(rotulos[minimo])
    rotuloNeuronio = np.reshape(rotuloNeuronio, (dim, dim))

    return rotuloNeuronio

def umatrix(dataset, rede, dim):
    
    matriz = np.zeros((dim,dim))

    for l in range(dim):
        for c in range(dim):
            if(l == 0):
                if(c == 0):
                    aux = distancia(rede[l][c], rede[l][c+1]) + distancia(rede[l][c], rede[l+1][c]) + distancia(rede[l][c], rede[l+1][c+1])
                    matriz[l][c] = aux/3
                elif(c == dim-1):
                    aux = distancia(rede[l][c], rede[l][c-1]) + distancia(rede[l][c], rede[l+1][c-1]) + distancia(rede[l][c], rede[l+1][c])
                    matriz[l][c] = aux/3
                else:
                    aux = distancia(rede[l][c], rede[l][c-1]) + distancia(rede[l][c], rede[l][c+1]) + distancia(rede[l][c], rede[l+1][c-1]) + distancia(rede[l][c], rede[l+1][c]) + distancia(rede[l][c], rede[l+1][c+1])
                    matriz[l][c] = aux/5

            elif(l == dim-1):
                if(c == 0):
                    aux = distancia(rede[l][c], rede[l-1][c]) + distancia(rede[l][c], rede[l-1][c+1]) + distancia(rede[l][c], rede[l][c+1])
                    matriz[l][c] = aux/3
                elif(c == dim-1):
                    aux = distancia(rede[l][c], rede[l-1][c-1]) + distancia(rede[l][c], rede[l-1][c]) + distancia(rede[l][c], rede[l][c-1])
                    matriz[l][c] = aux/3
                else:
                    aux = distancia(rede[l][c], rede[l-1][c-1]) + distancia(rede[l][c], rede[l-1][c]) + distancia(rede[l][c], rede[l-1][c+1]) + distancia(rede[l][c], rede[l][c-1]) + distancia(rede[l][c], rede[l][c+1])
                    matriz[l][c] = aux/5
            
            else:
                if(c == 0):
                    aux = distancia(rede[l][c], rede[l-1][c]) + distancia(rede[l][c], rede[l-1][c+1]) + distancia(rede[l][c], rede[l][c+1]) + distancia(rede[l][c], rede[l+1][c]) + distancia(rede[l][c], rede[l+1][c+1])
                    matriz[l][c] = aux/5
                elif(c == dim-1):
                    aux = distancia(rede[l][c], rede[l-1][c-1]) + distancia(rede[l][c], rede[l-1][c]) + distancia(rede[l][c], rede[l][c-1]) + distancia(rede[l][c], rede[l+1][c-1]) + distancia(rede[l][c], rede[l+1][c])
                    matriz[l][c] = aux/5
                else:
                    aux = distancia(rede[l][c], rede[l-1][c-1]) + distancia(rede[l][c], rede[l-1][c]) + distancia(rede[l][c], rede[l-1][c+1]) + distancia(rede[l][c], rede[l][c-1]) + distancia(rede[l][c], rede[l][c+1]) + distancia(rede[l][c], rede[l+1][c-1]) + distancia(rede[l][c], rede[l+1][c]) + distancia(rede[l][c], rede[l+1][c+1])
                    matriz[l][c] = aux/8
    rotulos = rotulaNeuronio(dataset, rede, dim)
    #print(matriz)
    #print(rotulos)

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    im = ax.imshow(matriz, interpolation='bilinear', cmap='gray')
    #plt.imshow(matriz)
    
    for l in range(dim):
        for c in range(dim):
            ax.text(c, l, rotulos[l][c], color='cornflowerblue', ha='center', va='center')
    
    fig.colorbar(im)
    plt.show()

def main():
    # conjunto de dados tem de estar com as classes na última coluna
    dataset = pd.read_csv("../Datasets/wine.csv", sep=',', header=None)
    rede, dim = SOM(dataset, eta0=0.5, max_epocas=100)
    umatrix(dataset, rede, dim)

main()