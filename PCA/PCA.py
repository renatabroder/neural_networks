import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def normaliza(dados):
    # normaliza com média 0 e variância 1
    return StandardScaler().fit_transform(dados)


def tratamento(dataset):
    n_entradas, cols = dataset.shape

    # retiradada coluna de classes e transformação de classificação nominal em numérica
    rotulos = rot = dataset[cols-1]
    rotulos = pd.factorize(rotulos)[0]

    # associar cada cor a uma classe (para plotar no final) - máximo de 30 classes
    cores = []
    for i in range(len(rotulos)):
        if(rotulos[i] == 0): cores.append('blue')
        elif(rotulos[i] == 1): cores.append('red')
        elif(rotulos[i] == 2): cores.append('green')
        elif(rotulos[i] == 3): cores.append('yellow')
        elif(rotulos[i] == 4): cores.append('darkorange')
        elif(rotulos[i] == 5): cores.append('pink')
        elif(rotulos[i] == 6): cores.append('darkviolet')
        elif(rotulos[i] == 7): cores.append('magenta')
        elif(rotulos[i] == 8): cores.append('darkgray')
        elif(rotulos[i] == 9): cores.append('olive')
        elif(rotulos[i] == 10): cores.append('aquamarine')
        elif(rotulos[i] == 11): cores.append('lightcoral')
        elif(rotulos[i] == 12): cores.append('midnightblue')
        elif(rotulos[i] == 13): cores.append('lime')
        elif(rotulos[i] == 14): cores.append('saddlebrown')
        elif(rotulos[i] == 15): cores.append('gray')
        elif(rotulos[i] == 16): cores.append('lightslategray')
        elif(rotulos[i] == 17): cores.append('burlywood')
        elif(rotulos[i] == 18): cores.append('maroon')
        elif(rotulos[i] == 19): cores.append('rosybrown')
        elif(rotulos[i] == 20): cores.append('chocolate')
        elif(rotulos[i] == 21): cores.append('silver')
        elif(rotulos[i] == 22): cores.append('lightgreen')
        elif(rotulos[i] == 23): cores.append('papayawhip')
        elif(rotulos[i] == 24): cores.append('thistle')
        elif(rotulos[i] == 26): cores.append('cornflowerblue')
        elif(rotulos[i] == 27): cores.append('darkblue')
        elif(rotulos[i] == 28): cores.append('salmon')
        elif(rotulos[i] == 29): cores.append('goldenrod')
        else: cores.append('b')

    dataset = dataset.drop(columns=[cols-1])
    dataset = normaliza(dataset)

    return dataset, rot, cores


def conjunto(lista):
    # retorna um conjunto (opção set() não retornava na ordem desejada)
    saida = []
    for x in lista:
        if x not in saida:
            saida.append(x)
    return saida


def grafico2d(pc, pcdataframe, rot, cores):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Componente principal 1', fontsize = 10)
    ax.set_ylabel('Componente principal 2', fontsize = 10)
    ax.set_title('PCA com 2 componentes principais', fontsize = 20)

    targets = conjunto(rot)
    colors = conjunto(cores)
    a = list(zip(targets, colors))

    for target, color in a:
        indicesToKeep = pcdataframe[2] == target
        ax.scatter(pcdataframe.loc[indicesToKeep, 0]
                , pcdataframe.loc[indicesToKeep, 1]
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()

    #plt.savefig("grafico2D.png")
    plt.show()


def grafico3d(pc, pcdataframe, rot, cores):
    fig = plt.figure(figsize=(8,8))
    ax = Axes3D(fig)
    ax.set_xlabel('Componente principal 1', fontsize=10)
    ax.set_ylabel('Componente principal 2', fontsize=10)
    ax.set_zlabel('Componente principal 3', fontsize=10)
    ax.set_title('PCA com 3 componentes principais', fontsize=20)

    targets = conjunto(rot)
    colors = conjunto(cores)
    a = list(zip(targets, colors))

    for target, color in a:
        indicesToKeep = pcdataframe[3] == target
        ax.scatter(pcdataframe.loc[indicesToKeep, 0]
                , pcdataframe.loc[indicesToKeep, 1]
                , pcdataframe.loc[indicesToKeep, 2]
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()

    #plt.savefig("grafico3D.png")
    plt.show()


def processamento(data, n=2):
    # conjunto de dados tem de estar com as classes na última coluna
    dataset = pd.read_csv(data + ".csv", sep=',', header=None)
    entradas, rot, cores = tratamento(dataset)
    
    pca = PCA(n_components=n)
    pc = pca.fit_transform(entradas)
    
    pcdataframe = pd.DataFrame(data = pc)
    pcdataframe[n] = rot
    
    if(n==2):
        grafico2d(pc, pcdataframe, rot, cores)
    elif(n==3):
        grafico3d(pc, pcdataframe, rot, cores)

    var = pca.explained_variance_ratio_
    print("\nConjunto de dados: ", data.upper())
    print("Quantidade de componentes: ", n)
    print("Variância por componente: ", var)
    print("Porcentagem da variância original: ", round(sum(var)*100, 2), '%\n')


def main():
    processamento("seeds", n=3)

main()