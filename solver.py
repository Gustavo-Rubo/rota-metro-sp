import csv
from os import path
import numpy as np

from dijkstra import dijkstra

transicoes = {}
estacoes = []
plataformas = []
grafo = []

estacoes_visitadas = []
estacoes_nao_visitadas = []

with open(path.join('problems', 'metro-sp-azul verde.csv')) as arquivo:
    leitor = csv.reader(arquivo, delimiter=',')
    for row in leitor:
        plataformas.append(','.join([row[0], row[1]]))
        plataformas.append(','.join([row[2], row[3]]))
        estacoes.append(row[1])
        estacoes.append(row[3])
    
    plataformas = np.unique(plataformas)
    estacoes = np.unique(estacoes)

    grafo = np.ones([len(plataformas), len(plataformas)]) * np.inf
    for i in range(len(grafo)): grafo[i][i] = 0

    arquivo.seek(0)
    for row in leitor:
        plataforma1 = ','.join([row[0], row[1]])
        plataforma2 = ','.join([row[2], row[3]])

        i = np.where(plataformas == plataforma1)[0][0]
        j = np.where(plataformas == plataforma2)[0][0]

        distancia = float(row[4].replace(',', '.'))
        grafo[i][j] = distancia
        grafo[j][i] = distancia

    dijkstra(grafo)