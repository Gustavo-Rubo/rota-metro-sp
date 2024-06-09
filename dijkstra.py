import numpy as np

grafo = [
    [0, 1, 9, 9],
    [1, 0, 1, 9],
    [9, 1, 0, 3],
    [9, 9, 3, 0]
]

def dijkstra(graph_simple):
    graph = graph_simple.copy()
    for start in range(len(graph)-1):
        unvisited = [i for i in range(len(graph)) if i != start]

        while (len(unvisited) > 1):
            next = -1
            min_distance = np.inf
            for u in unvisited:
                distance = graph[start][u]
                if (distance <= min_distance):
                    min_distance = distance
                    next = u

            unvisited.remove(next)
            for u in unvisited:
                distance1 = graph[start][u]
                distance2 = graph[start][next] + graph[next][u]
                if (distance2 < distance1):
                    graph[start][u] = distance2
                    graph[u][start] = distance2
    
    return graph

    # print('\n'.join([','.join([f'{n:.1f}' for n in row]) for row in grafo]))