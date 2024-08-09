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

        GRAPH_SIMPLE = np.ones([len(PLATFORM_NAMES), len(PLATFORM_NAMES)]) * np.inf

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


def route_index2name(route):
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
        current = next

    return distance


def get_next_options(path_options, visited):
    next_options = []
    path_options = [p_o for p_o in path_options if p_o != []]
    for p_o in path_options:
        for node in p_o:
            if node not in visited:
                next_options.append(node)
                break
    next_options = list(set(next_options))

    return next_options


def add_to_route(next, route, visited):
    route.append(next)
    if next not in visited:
        visited.append(next)
        visited.extend(EQUIVALENT_PLATFORMS[next])


if __name__ == "__main__":
    global PLATFORM_NAMES
    global GRAPH_SIMPLE
    global GRAPH_COMPLETE
    global GRAPH_PATHS
    global EQUIVALENT_PLATFORMS

    PLATFORM_NAMES = []
    GRAPH_SIMPLE = []
    GRAPH_COMPLETE = []
    EQUIVALENT_PLATFORMS = []

    filename = path.join('problems', 'complete.csv')
    # filename = path.join('problems', 'blue_green.csv')

    read_graph(filename)
    GRAPH_COMPLETE, GRAPH_PATHS = dijkstra(GRAPH_SIMPLE)
    memoize_equivalent_platforms()

    # sample_route1 = route_name2index(sample_routes.BLUE_GREEN)
    sample_route1 = route_name2index(sample_routes.COMPLETE_MANUAL1)
    sample_route_distance1 = measure_distance(sample_route1)
    sample_route2 = route_name2index(sample_routes.COMPLETE_MANUAL2)
    sample_route_distance2 = measure_distance(sample_route2)

    start = name2index('7 - rubi,jundiaí')
    # start = name2index('1 - azul,tucuruvi')
    distance_sum = 0
    route = []
    visited = []

    solutions = []
    stack_routes = []
    stack_next = []
    stack_distances = []
    stack_visited = []

    stack_next.append(start)
    stack_routes.append(route)
    stack_distances.append(distance_sum)
    stack_visited.append(visited)

    loop_counter = 0
    benchmark_distance = 1100

    while (len(stack_next) >= 1):
        if (loop_counter % 50000 == 0):
            print(f'loops: {loop_counter}\tstack length: {len(stack_next)}')
        loop_counter += 1

        current = stack_next.pop()
        route = stack_routes.pop()
        distance_sum = stack_distances.pop()
        visited = stack_visited.pop()

        if (len(route) >= 1):
            distance_sum += GRAPH_COMPLETE[route[-1]][current]
        add_to_route(current, route, visited)

        while (len(visited) < len(PLATFORM_NAMES) and distance_sum < benchmark_distance):

            connections_complete = GRAPH_COMPLETE[current].copy()
            connections_simple = GRAPH_SIMPLE[current].copy()
            path_options = GRAPH_PATHS[current].copy()

            for v in visited:
                # this may not be necessary if i make distance infinite directly on the whole graph column every time a station is visited
                connections_complete[v] = np.inf
                connections_simple[v] = np.inf
                path_options[v] = []

            direct_connections = np.where(connections_simple != np.inf)[0]
            leaf_nodes = [d_c for d_c in direct_connections if is_node_leaf(d_c)]
            not_leaf_nodes = [d_c for d_c in direct_connections if d_c not in leaf_nodes]

            if (len(leaf_nodes) > 0):
                for l_n in leaf_nodes:
                    add_to_route(l_n, route, visited)
                    distance_sum += connections_complete[l_n] * 2 + 2
                    add_to_route(current, route, visited)

            if (len(not_leaf_nodes) == 1):
                next = not_leaf_nodes[0]
                add_to_route(next, route, visited)
                distance_sum += connections_complete[next]
                current = next
            else:
                next_options = get_next_options(path_options, visited)

                stack_next.extend(next_options.copy())
                for _ in next_options:
                    stack_distances.append(distance_sum)
                    stack_routes.append(route.copy())
                    stack_visited.append(visited.copy())

                break

        # if (len(visited) == len(PLATFORM_NAMES)):
        if (len(solutions) == 0 or len(route) >= len(solutions[-1][1])):
            solutions.append([distance_sum, route])
            print(f'solution distance: {distance_sum}\t solution stations: {len(visited)}')
            print('\n'.join(route_index2name(route)))

    print(f'manual 1: {sample_route_distance1:.1f}')
    print(f'manual 2: {sample_route_distance2:.1f}')

    if (len(solutions) > 0):
        top_solution = sorted(solutions, key=lambda x: x[0])[0]
        print(f'solution distance: {top_solution[0]}\t solution stations: {len(top_solution[1])}')
        print('\n'.join(route_index2name(top_solution[1])))
    else:
        print('no solutions found')
