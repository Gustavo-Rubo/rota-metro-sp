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
        
        platforms = np.unique(platforms)

        graph = np.ones([len(platforms), len(platforms)]) * np.inf
        # for i in range(len(graph)): graph[i][i] = 0

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

def get_station(platform):
    return platform.split(',')[1]

def memoize_equivalent_platforms(platforms):
    equivalent_platforms = [[] for _ in range(len(platforms))]
    for i, platform_i in enumerate(platforms):
        station_i = get_station(platform_i)
        for j in range(i+1, len(platforms)):
            platform_j = platforms[j]
            station_j = get_station(platform_j)
            if station_i == station_j:
                equivalent_platforms[i].append(platform_j)
                equivalent_platforms[j].append(platform_i)
    
    return equivalent_platforms

def is_node_leaf(index, graph_simple):
    direct_connections = np.count_nonzero(graph_simple[index] != np.inf)
    return True if direct_connections <= 1 else False

def measure_distance(graph, platforms, route):
    distance = 0
    current_i = name2index(route[0], platforms)
    for next_platform in route[1:]:
        next_i = name2index(next_platform, platforms)
        distance += graph[current_i][next_i]
        # print(graph[current_i][next_i], current_i, next_i)

        current_i = next_i

    return distance


def get_next_platforms():
    # get list of direct unvisited options
    # if only one option, return that
    # if leaf nodes + 1 option, return leaf nodes then option
    # if no direct unvisited, greedy on global
    # if more than one unvisited non leaf option, store [distance, route, visited, [options]] in the stack

    return

def add_to_route(next_platform_i, route, visited, equivalent_platforms):
    route.append(next_platform)
    if next_platform not in visited:
        visited.append(next_platform)
        visited.extend(equivalent_platforms[name2index(next_platform, platforms)])

    route.append(index2name(next_platform_i, platforms))

if __name__ == "__main__":
    filename = path.join('problems', 'complete.csv')
    # filename = path.join('problems', 'blue_green.csv')

    graph_simple, platforms = read_graph(filename)
    graph_complete = dijkstra(graph_simple)
    equivalent_platforms = memoize_equivalent_platforms(platforms)

    # sample_route = sample_routes.BLUE_GREEN
    sample_route = sample_routes.COMPLETE_MANUAL1
    sample_route_distance1 = measure_distance(graph_complete, platforms, sample_route)
    sample_route = sample_routes.COMPLETE_MANUAL2
    sample_route_distance2 = measure_distance(graph_complete, platforms, sample_route)

    distance_sum = 0
    route = []
    visited = []
    start = '7 - rubi,jundiaí'
    # start = '1 - azul,tucuruvi'

    route.append(start)
    if start not in visited:
        visited.append(start)
        visited.extend(equivalent_platforms[name2index(start, platforms)]
)
    solutions = []
    stack = []

    while(len(visited) < len(platforms)):
        # get list of direct unvisited options
        # if only one option, return that
        # if no direct unvisited (no options), greedy on global
        # if leaf nodes + 1 option, return leaf nodes then option
        # if more than one unvisited non leaf option, store [distance, route, visited, [options]] in the stack
        current_platform = route[-1]
        current_i = name2index(current_platform, platforms)
        next_platform_i = -1

        connections_complete = graph_complete[current_i].copy()
        connections_simple = graph_simple[current_i].copy()
        for v in visited:
            # this may not be necessary if i make distance infinite directly on the whole graph column every time a station is visited
            connections_complete[name2index(v, platforms)] = np.inf
            connections_simple[name2index(v, platforms)] = np.inf
        
        direct_connections_i = np.where(connections_simple != np.inf)[0]
        if (len(direct_connections_i == 1)):
            next_platform_i = direct_connections_i[0]
        elif(len(direct_connections_i == 0)):
            next_platform_i = connections_complete.argmin()
        else:
            leaf_nodes_i = [d_c_i for d_c_i in direct_connections_i if is_node_leaf(d_c_i, graph_simple)]
            not_leaf_nodes_i = [d_c_i for d_c_i in direct_connections_i if d_c_i not in leaf_nodes_i]

            if (len(leaf_nodes_i) == len(direct_connections_i - 1)):
                for l_n_i in leaf_nodes_i:
                    add_to_route(l_n_i, route, visited, equivalent_platforms)
                    distance_sum += connections_complete[l_n_i]
            
            # TODO: stack stuff
            next_platform_i = not_leaf_nodes_i[0]

        add_to_route(next_platform_i, route, visited, equivalent_platforms)
        distance_sum += connections_complete[next_platform_i]


    print(f'manual 1: {sample_route_distance1:.1f}')
    print(f'manual 2: {sample_route_distance2:.1f}')
    print(f'guloso: {distance_sum:.1f}')