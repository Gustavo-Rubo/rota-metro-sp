import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from animations.station_coords import STATION_COORDS


# com texto
# IM_W = 4399
# IM_H = 3302
# X0 = 5.102677 - 1.5
# Y0 = -27.722373 - 8.5
# XD = X0 - 356.468680 - 19
# YD = Y0 - 240.377410 - 4.2

# limpo
IM_W = 4194
IM_H = 3208
X0 = 5.102677 - 1.5
Y0 = -27.722373 - 1
XD = X0 - 356.468680 - 1.5
YD = Y0 - 240.377410 - 1


class Plot:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        im_metro = plt.imread('/media/rubo/HD/Projetos/rota-metro-sp/animations/limpopb.png')
        self.ax.imshow(im_metro, zorder=0)

        # coords = np.array(STATION_COORDS)
        # self.ax.scatter(
        #     -IM_W*(coords[:, 0].astype('float')-X0)/XD,
        #     -IM_H*(coords[:, 1].astype('float')-Y0)/YD,
        #     color='red', marker='.', s=20, label='panoids with ocr')
        # plt.show()

        plt.axis('off')
        plt.tight_layout()

    def update(self, route):
        route_coords = np.array([STATION_COORDS[station] for station in route])

        points = list(zip(
            -IM_W*(route_coords[:, 0].astype('float')-X0)/XD,
            -IM_H*(route_coords[:, 1].astype('float')-Y0)/YD))
        segments = [[points[i], points[i+1]] for i in range(len(points)-1)]

        coll = LineCollection(segments, array=np.linspace(
            0, 1, num=100), cmap='rainbow', linewidths=4)

        self.ax.add_collection(coll)
        p = self.ax.scatter(
            -IM_W*(route_coords[-1, 0].astype('float')-X0)/XD,
            -IM_H*(route_coords[-1, 1].astype('float')-Y0)/YD,
            marker='o', color='blue', s=20)
        plt.pause(0.00001)
        self.ax.collections.pop()
        self.ax.collections.pop()
