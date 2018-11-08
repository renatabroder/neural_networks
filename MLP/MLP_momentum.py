import random
import math
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def leDados(conjuntoDados):
    return pd.read_csv(conjuntoDados, sep=',', header=None)

def normaliza(dados):
    normal = MinMaxScaler()
    normalizados = normal.fit_transform(dados)
    normalizados = pd.DataFrame(normalizados)
    indices = dados.index
    normalizados.index = indices
    return normalizados

def tratamento(dataset):
    lins, cols = dataset.shape
    saida_temp = pd.factorize(dataset[cols-1])[0]
    n_classes = len(set(saida_temp))
    
    # transformação da saída esperada em um nparray
    saida_esperada = []
    for classe in saida_temp:
        aux = [1 if classe == i else 0 for i in range(n_classes)]
        saida_esperada.append(aux)
    saida_esperada = np.array(saida_esperada)

    # retiradada coluna de classes
    dataset = dataset.drop(columns=[cols-1])

    # normalização do dataset
    dataset = normaliza(dataset)

    # adição do bias na entrada
    dataset[cols-1] = 1

    # transformação do pandas dataset em um numpy array
    dataset = dataset.values

    return dataset, saida_esperada, lins, n_classes

def inicializaPesos(n_camadas, n_entrada, n_perceptron):
    pesos = []
    for c in range(n_camadas):
        if(c==0): aux = [np.array([random.uniform(0, 0.1) for i in range(n_entrada)]) for n in range(n_perceptron[c])]
        else: aux = [np.array([random.uniform(0, 0.1) for i in range(n_perceptron[c-1]+1)]) for n in range(n_perceptron[c])]
        aux = np.array(aux)
        pesos.append(aux)
    pesos = np.array(pesos)
    return pesos

def ativacao(u):
    return 1.0/(1.0+math.exp(-u))

def derivada(u):
    #return (math.exp(-u))/((1+math.exp(-u))**2)
    return u * (1-u)

def acuracia(saida, rotulos):
    saida = [np.argmax(s) for s in saida]
    rotulo = [np.argmax(r) for r in rotulos]

    acuracia = [1 if saida[i] == rotulo[i] else 0 for i in range(len(saida))]
    acuracia = sum(acuracia)/float(len(acuracia))

    return acuracia

def mlp(dataset, n_camadas=3, n_perceptron=[5, 5], taxa_erro=0.01, eta=0.5, m=0.5, max_epocas=500, r=50):
    # tratamento dos dados: separação entre entradas e rotulos
    entradas, saida_esperada, n_linhas, n_saida = tratamento(dataset)

    # separação em conjuntos de treino e teste (80%/20%)
    treino, teste, rotulos_treino, rotulos_teste = train_test_split(entradas, saida_esperada, test_size=0.2, random_state=r)
    n_entrada = len(treino[0])
    n_perceptron.append(n_saida)

    # inicializar pesos
    random.seed(r)
    pesos = inicializaPesos(n_camadas, n_entrada, n_perceptron)
    pesos_anteriores = cp.deepcopy(pesos)

    epoca = 0
    epocas = []

    erros_treino = []
    erros_teste = []
    
    acur = []

    e = 1
    while(e > taxa_erro):
    #while(epoca < max_epocas):

        erro_epoca = []

        ### Início do TREINO ###

        # uma época:
        for i in range(len(treino)):

            # guarda saída de cada processador para um padrão apresentado
            saida_ativacao = [[]for c in range(n_camadas)]
            # guarda deltas de cada processador para um padrão apresentado
            deltas = [[] for c in range(n_camadas)]

            erro_saida_treino = []
            erro_saida_teste = []
        
            # feed forward
            for c in range(n_camadas):
                # primeira camada
                if(c == 0):
                    # para cada processador em uma camada: somat w*x
                    for n in range(n_perceptron[c]):
                        saida_ativacao[c].append(ativacao(np.matmul(treino[i], np.transpose(pesos[c][n]))))
                    # adicão do bias
                    if(c != n_camadas-1): saida_ativacao[c].append(1)

                # outras camadas
                else:
                    # para cada processador em uma camada: somat w*x
                    for n in range(n_perceptron[c]):
                        saida_ativacao[c].append(ativacao(np.matmul(saida_ativacao[c-1], np.transpose(pesos[c][n]))))
                    # adicão do bias - se não for a última
                    if(c != n_camadas-1): saida_ativacao[c].append(1)
            
            # feed back - cálculo do deltas
            for c in range(n_camadas-1, -1, -1):
                # camada de saída
                if(c == n_camadas-1):
                    for n in range(n_perceptron[c]):
                        erro_padrao = rotulos_treino[i][n] - saida_ativacao[c][n]
                        erro_saida_treino.append(erro_padrao)
                        deltas[c].append(erro_padrao * derivada(saida_ativacao[c][n]))
                        
                # camadas ocultas intermediárias
                elif(c > 0):
                    for n in range(n_perceptron[c]):
                        aux = 0
                        for p in range(n_perceptron[c+1]):
                            aux += pesos[c+1][p][n] * deltas[c+1][p]
                        deltas[c].append((derivada(saida_ativacao[c][n]) * aux))

                # primeira camada oculta
                elif(c == 0):
                    for n in range(n_perceptron[c]):
                        aux = 0
                        for p in range(n_perceptron[c+1]):
                            aux += pesos[c+1][p][n] * deltas[c+1][p]
                        deltas[c].append((derivada(saida_ativacao[c][n]) * aux))
            
            # cálculo do erro quadrático médio do padrão
            erro_quadr_padrao = np.mean([(er**2) for er in erro_saida_treino])

            # adiciona erro quadrático médio do padrão à lista de erros da época
            erro_epoca.append(erro_quadr_padrao)
            
            if(erro_quadr_padrao > taxa_erro):
            #if(np.abs(np.amax(erro_padrao)) > taxa_erro):

                # atualiza todos os pesos se o erro da camada de saida não for aceitável
                for c in range(n_camadas-1, -1, -1):
                    for n in range(n_perceptron[c]):
                        pesos_aux = pesos - pesos_anteriores
                        
                        if(c == 0):
                            for p in range(len(pesos[c][n])):
                                momentum = m * pesos_aux[c][n][p]
                                variacao = eta*treino[i][p]*deltas[c][n] + momentum
                                pesos_anteriores[c][n][p] = cp.copy(pesos[c][n][p])
                                pesos[c][n][p] += variacao
                        else:
                            for p in range(len(pesos[c][n])):
                                momentum = m * pesos_aux[c][n][p]
                                variacao = eta*saida_ativacao[c-1][p]*deltas[c][n] + momentum
                                pesos_anteriores[c][n][p] = cp.copy(pesos[c][n][p])
                                pesos[c][n][p] += variacao

        # média de erros da época
        e = np.mean(erro_epoca)
        # adiciona média de erros à lista que será plotada
        erros_treino.append(e)
        
        print("\nÉpoca ", epoca)
        print("Erro de treino: ", e)
        
        epocas.append(epoca)
        epoca+=1
        
        ### Início do TESTE ###
        
        # guarda as saídas da camada de saída para calcular a acurácia
        teste_estimado = []
        # guarda os erros da camada de saída
        erro_quadr_saida = []

        for i in range(len(teste)):
            
            # saídas de um padrão apresentado
            saida_ativacao_teste = [[] for c in range(n_camadas)]

            for c in range(n_camadas):   
                # primeira camada
                if(c == 0):
                    for n in range(n_perceptron[c]):
                        aux = np.matmul(teste[i], np.transpose(pesos[c][n]))
                        saida_ativacao_teste[c].append(ativacao(aux))
                    # adicão do bias
                    if(c != n_camadas-1): saida_ativacao_teste[c].append(1)
                
                # outras camadas
                else:
                    for n in range(n_perceptron[c]):
                        aux = np.matmul(saida_ativacao_teste[c-1], np.transpose(pesos[c][n]))
                        saida_ativacao_teste[c].append(ativacao(aux))
                    # adicão do bias - se não for a última
                    if(c != n_camadas-1): saida_ativacao_teste[c].append(1)
        
            # cálculo do erro quadrático médio da camada de saída
            aux_erros = []
            for s in range(n_perceptron[n_camadas-1]):
                aux_erros.append(rotulos_teste[i][s] - saida_ativacao_teste[n_camadas-1][s])
            erro_quadr_saida.append(np.mean([e**2 for e in aux_erros]))

            # guarda das saídas da camada de saída para cálculo da acurácia
            teste_estimado.append(saida_ativacao_teste[n_camadas-1])
        
        erros_teste.append(np.mean(erro_quadr_saida))
        print('Erro de teste: ', np.mean(erro_quadr_saida))
        acur.append(acuracia(teste_estimado, rotulos_teste))

    print(np.amax(acur), np.argmax(acur))
    
    plt.subplot(211)
    plt.plot(epocas, erros_treino, label='treino')
    plt.plot(epocas, erros_teste, label='teste')
    plt.xlabel("Épocas")
    plt.ylabel("Erro")
    plt.legend()

    plt.subplot(212)
    plt.plot(epocas, acur, label='acuracia')
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.show()

    return


def main():
    # conjunto de dados tem de estar com as classes na última coluna
    dataset = leDados("seeds.csv")
    mlp(dataset, n_camadas=1, n_perceptron=[], taxa_erro=0.03, eta=0.5, m =0.5, r=10)


main()