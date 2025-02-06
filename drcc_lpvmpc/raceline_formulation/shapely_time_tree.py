import shapely
import shapely.ops
import numpy as np
from shapely.ops import Point
from shapely import STRtree

class RTree():
    def __init__(self,x,y,tau) -> None:
        n = len(x)

        assert(len(x)==len(tau))
        self.time_maps = []
        for i in range(n):
            indice_i = {"geometry": Point(x[i],y[i]), "value": tau[i]}
            self.time_maps.append(indice_i)

        self.tau_rtree = STRtree([time_map["geometry"] for time_map in self.time_maps])
        self.items = np.array([time_map["value"] for time_map in self.time_maps])

        print("search tree is built")

    def findNearest(self,vec_x,vec_y):

        size = len(vec_x)
        tau_vec = []

        for i in range(size):
            n_p_i = self.items.take(self.tau_rtree.query_nearest(Point(vec_x[i],vec_y[i])))[0]
            tau_vec.append(n_p_i)
        
        return tau_vec

