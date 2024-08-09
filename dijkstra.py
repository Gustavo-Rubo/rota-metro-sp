import numpy as np

grafo = [
    [0, 1, 9, 9],
    [1, 0, 1, 9],
    [9, 1, 0, 3],
    [9, 9, 3, 0]
]


def dijkstra(graph_simple):
    distances = graph_simple.copy()
    paths = []
    for i in range(len(distances)):
        paths.append([])
        for j in range(len(distances[0])):
            path = [] if distances[i][j] == np.inf else [j]
            paths[i].append(path)

    for start in range(len(distances)-1):
        unvisited = [i for i in range(len(distances)) if i != start]

        while (len(unvisited) > 1):
            next = -1
            min_distance = np.inf
            for u in unvisited:
                distance = distances[start][u]
                if (distance <= min_distance):
                    min_distance = distance
                    next = u

            for u in unvisited:
                distance1 = distances[start][u]
                distance2 = distances[start][next] + distances[next][u]
                if (distance2 <= distance1):
                    distances[start][u] = distance2
                    distances[u][start] = distance2
                    paths[start][u] = paths[start][next]+[next]+paths[next][u]+[u]
                    paths[u][start] = paths[u][next]+[next]+paths[next][start]+[start]

            unvisited.remove(next)

    return distances, paths
    # print('\n'.join([','.join([f'{n:.1f}' for n in row]) for row in distances]))
