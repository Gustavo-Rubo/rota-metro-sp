import csv
from os import path
import numpy as np

from dijkstra import dijkstra
import sample_routes


# WHEN TO USE PLATFORM NAMES VS INDEX:
# names should only be for input and output.
# everywhere else the program should be using indexes
# variables that store names should always be named *_names or *_n


def read_graph(filename):
    global PLATFORM_NAMES
    global GRAPH_SIMPLE

    with open(filename) as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            PLATFORM_NAMES.append(','.join([row[0], row[1]]))
            PLATFORM_NAMES.append(','.join([row[2], row[3]]))

        PLATFORM_NAMES = np.unique(PLATFORM_NAMES)

        GRAPH_SIMPLE = np.ones(
            [len(PLATFORM_NAMES), len(PLATFORM_NAMES)]) * np.inf
        # for i in range(len(GRAPH_SIMPLE)): GRAPH_SIMPLE[i][i] = 0

        file.seek(0)
        for row in reader:
            platform1 = ','.join([row[0], row[1]])
            platform2 = ','.join([row[2], row[3]])

            i = name2index(platform1)
            j = name2index(platform2)

            distance = float(row[4].replace(',', '.'))
            GRAPH_SIMPLE[i][j] = distance
            GRAPH_SIMPLE[j][i] = distance


def memoize_equivalent_platforms():
    global EQUIVALENT_PLATFORMS

    EQUIVALENT_PLATFORMS = [[] for _ in range(len(PLATFORM_NAMES))]
    for i, platform_name_i in enumerate(PLATFORM_NAMES):
        station_i = get_station_from_name(platform_name_i)
        for j in range(i+1, len(PLATFORM_NAMES)):
            platform_name_j = PLATFORM_NAMES[j]
            station_j = get_station_from_name(platform_name_j)
            if station_i == station_j:
                EQUIVALENT_PLATFORMS[i].append(name2index(platform_name_j))
                EQUIVALENT_PLATFORMS[j].append(name2index(platform_name_i))


def name2index(name):
    return np.where(PLATFORM_NAMES == name)[0][0]


def index2name(index):
    return PLATFORM_NAMES[index]


def route_name2index(route_names):
    return [name2index(name) for name in route_names]


def route_index2name(route_names):
    # TODO
    return [index2name(index) for index in route]


def get_station_from_name(platform_name):
    return platform_name.split(',')[1]


def is_node_leaf(index):
    direct_connections = np.count_nonzero(GRAPH_SIMPLE[index] != np.inf)
    return True if direct_connections <= 1 else False


def measure_distance(route):
    distance = 0
    current = route[0]
    for next in route[1:]:
        distance += GRAPH_COMPLETE[current][next]
        # print(graph[current_i][next_i], current_i, next_i)

        current = next

    return distance


def get_next_platforms():
    # get list of direct unvisited options
    # if only one option, return that
    # if leaf nodes + 1 option, return leaf nodes then option
    # if no direct unvisited, greedy on global
    # if more than one unvisited non leaf option, store [distance, route, visited, [options]] in the stack

    return


def add_to_route(next, route, visited):
    route.append(next)
    if next not in visited:
        visited.append(next)
        visited.extend(EQUIVALENT_PLATFORMS[next])


if __name__ == "__main__":
    global PLATFORM_NAMES
    global GRAPH_SIMPLE
    global GRAPH_COMPLETE
    global EQUIVALENT_PLATFORMS

    PLATFORM_NAMES = []
    GRAPH_SIMPLE = []
    GRAPH_COMPLETE = []
    EQUIVALENT_PLATFORMS = []

    filename = path.join('problems', 'complete.csv')
    # filename = path.join('problems', 'blue_green.csv')

    read_graph(filename)
    GRAPH_COMPLETE = dijkstra(GRAPH_SIMPLE)
    memoize_equivalent_platforms()

    # sample_route = sample_routes.BLUE_GREEN
    sample_route1 = route_name2index(sample_routes.COMPLETE_MANUAL1)
    sample_route_distance1 = measure_distance(sample_route1)
    sample_route2 = route_name2index(sample_routes.COMPLETE_MANUAL2)
    sample_route_distance2 = measure_distance(sample_route2)

    distance_sum = 0
    route = []
    visited = []
    start = name2index('7 - rubi,jundiaí')
    # start = '1 - azul,tucuruvi'

    add_to_route(start, route, visited)
    solutions = []
    stack = []

    while (len(visited) < len(PLATFORM_NAMES)):
        # get list of direct unvisited options
        # if only one option, return that
        # if no direct unvisited (no options), greedy on global
        # if leaf nodes + 1 option, return leaf nodes then option
        # if more than one unvisited non leaf option, store [distance, route, visited, [options]] in the stack
        current = route[-1]
        next = -1

        connections_complete = GRAPH_COMPLETE[current].copy()
        connections_simple = GRAPH_SIMPLE[current].copy()
        for v in visited:
            # this may not be necessary if i make distance infinite directly on the whole graph column every time a station is visited
            connections_complete[v] = np.inf
            connections_simple[v] = np.inf

        direct_connections = np.where(connections_simple != np.inf)[0]
        if (len(direct_connections) == 1):
            next = direct_connections[0]
        elif (len(direct_connections) == 0):
            next = connections_complete.argmin()
        else:
            leaf_nodes = [
                d_c for d_c in direct_connections if is_node_leaf(d_c)]
            not_leaf_nodes = [
                d_c for d_c in direct_connections if d_c not in leaf_nodes]

            if (len(leaf_nodes) == len(direct_connections - 1)):
                for l_n in leaf_nodes:
                    add_to_route(l_n, route, visited)
                    distance_sum += connections_complete[l_n]

            # TODO: stack stuff
            next = not_leaf_nodes[0]

        add_to_route(next, route, visited)
        distance_sum += connections_complete[next]

    print(f'manual 1: {sample_route_distance1:.1f}')
    print(f'manual 2: {sample_route_distance2:.1f}')
    print(f'guloso: {distance_sum:.1f}')

    print('rota gulosa: ' + '\n'.join(route_index2name(route)))
