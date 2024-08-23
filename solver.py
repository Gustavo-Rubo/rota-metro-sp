import csv
from os import path
import numpy as np

from dijkstra import dijkstra
import sample_routes

from animations.Plot import Plot


# WHEN TO USE PLATFORM NAMES VS INDEX:
# names should only be for input and output.
# everywhere else the program should be using indexes
# variables that store names should always be named *_names or *_n


def read_graph(filename):
    global PLATFORM_NAMES
    global GRAPH_SIMPLE


<< << << < HEAD
 # lines_filter = [7, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15][:7]
 lines_filter = [7, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15]
== == == =
 lines_filter = [7, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15][:8]
  # lines_filter = [7, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15]
>>>>>> > origin/main

 with open(filename) as file:
      reader = csv.reader(file, delimiter=',')
       for row in reader:
            if (int(row[0].split(' ')[0]) in lines_filter and int(row[2].split(' ')[0]) in lines_filter):
                PLATFORM_NAMES.append(','.join([row[0], row[1]]))
                PLATFORM_NAMES.append(','.join([row[2], row[3]]))

        PLATFORM_NAMES = np.unique(PLATFORM_NAMES)

        GRAPH_SIMPLE = np.ones([len(PLATFORM_NAMES), len(PLATFORM_NAMES)]) * np.inf

        file.seek(0)
        for row in reader:
            if (int(row[0].split(' ')[0]) in lines_filter and int(row[2].split(' ')[0]) in lines_filter):
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


def get_next_options(path_options, visited, visited_strict):
    next_options = []
    path_options = [p_o for i, p_o in enumerate(path_options) if i not in visited]

    for p_o in path_options:
        for node in p_o:
            if node not in visited_strict:
                next_options.append(node)
                break
    next_options = list(set(next_options))

    return next_options


def add_to_route(next, route, visited, visited_strict):
    if (len(route) > 0):
        current = route[-1]
    else:
        current = next
    route.append(next)

    if next not in visited:
        visited.append(next)
        visited.extend(EQUIVALENT_PLATFORMS[next])

    if next not in visited_strict:
        visited_strict.extend(GRAPH_PATHS[current][next])
        visited_strict = list(set(visited_strict))


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
    # sample_route1 = route_name2index(sample_routes.COMPLETE_MANUAL1)
    # sample_route_distance1 = measure_distance(sample_route1)
    # sample_route2 = route_name2index(sample_routes.COMPLETE_MANUAL2)
    # sample_route_distance2 = measure_distance(sample_route2)

    start = name2index('7 - rubi,jundiaí')
    # start = name2index('1 - azul,tucuruvi')
    distance_sum = 0
    route = []
    # TODO: rename to visited_station and visited_platform
    visited = []
    visited_strict = []

    solutions = []
    stack_routes = []
    stack_next = []
    stack_distances = []
    stack_visited = []
    stack_visited_strict = []

    stack_next.append(start)
    stack_routes.append(route)
    stack_distances.append(distance_sum)
    stack_visited.append(visited)
    stack_visited_strict.append(visited_strict)

    loop_counter = 0
    benchmark_distance = 500

    plot = Plot()

    while (len(stack_next) >= 1):
        if (loop_counter % 200000 == 0):
            print(f'loops: {loop_counter}\tstack length: {len(stack_next)}')
        loop_counter += 1

        current = stack_next.pop()
        route = stack_routes.pop()
        distance_sum = stack_distances.pop()
        visited = stack_visited.pop()
        visited_strict = stack_visited_strict.pop()

        if (len(route) >= 1):
            distance_sum += GRAPH_COMPLETE[route[-1]][current]
        add_to_route(current, route, visited, visited_strict)

        while (len(visited) < len(PLATFORM_NAMES) and distance_sum < benchmark_distance):

            connections_complete = GRAPH_COMPLETE[current].copy()
            connections_simple = GRAPH_SIMPLE[current].copy()
            path_options = GRAPH_PATHS[current].copy()

            for v in visited:
                # this may not be necessary if i make distance infinite directly on the whole graph column every time a station is visited
                connections_complete[v] = np.inf
                connections_simple[v] = np.inf
                # path_options[v] = []

            direct_connections = np.where(connections_simple != np.inf)[0]
            leaf_nodes = [d_c for d_c in direct_connections if is_node_leaf(d_c)]
            not_leaf_nodes = [d_c for d_c in direct_connections if d_c not in leaf_nodes]

            if (len(leaf_nodes) > 0):
                for l_n in leaf_nodes:
                    add_to_route(l_n, route, visited, visited_strict)
                    plot.update(route)
                    distance_sum += connections_complete[l_n]
                    if (len(visited) < len(PLATFORM_NAMES)):
                        add_to_route(current, route, visited, visited_strict)
                        plot.update(route)
                        distance_sum += connections_complete[l_n]

            if (len(not_leaf_nodes) == 1):
                next = not_leaf_nodes[0]
                add_to_route(next, route, visited, visited_strict)
                plot.update(route)
                distance_sum += connections_complete[next]
                current = next
            else:
                # TODO: anti-double backtracking
                # figure how to detect backtracking
                # when backtrack, set a flag. flag is cleared after new visited station
                next_options = get_next_options(path_options, visited, visited_strict)
                # this sorting is complete heuristics
                next_options_prioritized = sorted(next_options, key=lambda x: -len(path_options[x]))

                stack_next.extend(next_options_prioritized.copy()[-3:])
                for _ in next_options[-3:]:
                    stack_distances.append(distance_sum)
                    stack_routes.append(route.copy())
                    stack_visited.append(visited.copy())
                    stack_visited_strict.append(visited_strict.copy())

                break

        if (len(visited) == len(PLATFORM_NAMES) and distance_sum <= benchmark_distance):
            # if (len(solutions) == 0 or len(route) >= len(solutions[-1][1])):
            if (distance_sum <= min([benchmark_distance] + [s[0] for s in solutions])+5):
                print(f'solution distance: {distance_sum}\t solution stations: {len(visited)}')
                print('\n'.join(route_index2name(route)))

            solutions.append([distance_sum, route])

    # print(f'manual 1: {sample_route_distance1:.1f}')
    # print(f'manual 2: {sample_route_distance2:.1f}')

    if (len(solutions) > 0):
        top_solution = sorted(solutions, key=lambda x: [-len(x[1]), x[0]])[0]
        print(f'solution distance: {top_solution[0]}\t solution stations: {len(top_solution[1])}')
        print('\n'.join(route_index2name(top_solution[1])))
    else:
        print('no solutions found')
