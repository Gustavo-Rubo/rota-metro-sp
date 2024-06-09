import csv
from os import path
import numpy as np

from dijkstra import dijkstra
import sample_routes

def read_graph(filename):
    graph = []
    platforms = []
    # stations = []

    with open(filename) as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            platforms.append(','.join([row[0], row[1]]))
            platforms.append(','.join([row[2], row[3]]))
            # stations.append(row[1])
            # stations.append(row[3])
        
        platforms = np.unique(platforms)
        # stations = np.unique(stations)

        graph = np.ones([len(platforms), len(platforms)]) * np.inf
        for i in range(len(graph)): graph[i][i] = 0

        file.seek(0)
        for row in reader:
            platform1 = ','.join([row[0], row[1]])
            platform2 = ','.join([row[2], row[3]])

            i = name2index(platform1, platforms)
            j = name2index(platform2, platforms)

            distance = float(row[4].replace(',', '.'))
            graph[i][j] = distance
            graph[j][i] = distance

    return graph, platforms


def name2index(name, platforms):
    return np.where(platforms == name)[0][0]

def index2name(index, platforms):
    return platforms[index]

def get_equivalent_platforms():
    return 0

def measure_distance(graph, platforms, route):
    distance = 0
    current_i = name2index(route[0], platforms)
    for next_platform in route[1:]:
        next_i = name2index(next_platform, platforms)
        distance += graph[current_i][next_i]
        # print(graph[current_i][next_i], current_i, next_i)

        current_i = next_i

    return distance


if __name__ == "__main__":
    # stations_visitadas = []
    # stations_nao_visitadas = []

    filename = path.join('problems', 'metro-sp-azul verde.csv')

    graph_simple, platforms = read_graph(filename)
    graph_complete = dijkstra(graph_simple)

    sample_route = sample_routes.AZUL_VERDE
    sample_route_distance = measure_distance(graph_complete, platforms, sample_route)

    print(sample_route_distance)