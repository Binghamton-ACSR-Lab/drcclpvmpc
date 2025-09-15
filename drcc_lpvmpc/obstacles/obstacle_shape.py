from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

# class Circle_obs:
#     def __init__(self,center_ls,side_avoid,radius,safe_range=0.02) -> None:
#         """
#         center_ls : a numpy array [2,n] where n represents the number of obstacles
#         radius : all share the same radius
#         side_avoid : side_avoid list, 1 => left avoidance, -1 => right avoidance
#         """
#         self.centers = []
#         for i in range(center_ls.shape[1]):
#             self.centers.append((Point(center_ls[0,i],center_ls[1,i]),side_avoid[i]))

#         self.radius = radius
#         self.safe_range = safe_range

#     def find_nearest_point(self, current_point):
#         assert(current_point.shape[0] == 2)
#         current_point = Point(current_point)
#         nearest_point_side_avoid = min(self.centers, key=lambda point: current_point.distance(point[0]))
#         return nearest_point_side_avoid
    
#     def check_points(self,points):
#         """
#         Check which points are within which circles.
#         points: [[a,b],...]
#         """
#         results = {Point(point): [] for _, point in enumerate(points)}

#         for center, side_avoid in self.centers:
#             circle = center.buffer(self.radius)
#             for _, point in enumerate(points):
#                 if circle.contains(Point(point)):
#                     results[Point(point)].append([center,side_avoid])
        
#         return results
    
#     def get_linear_function(self, current_point, direction, obs_center):

#         """
#         current_point : np array, denote as the reference point lies inside the circle
#         direction: radian float
#         obs_center: a np array class, denote as x1, y1
#         """
#         circle = obs_center.buffer(self.radius+self.safe_range)
#         line = LineString([(current_point.x,current_point.y),(current_point.x+10000*np.cos(direction),current_point.y+10000*np.sin(direction))])
#         intersection = circle.intersection(line) # denote as x2, y2
#         #  the slop of the tangent line
#         x1 = obs_center.x
#         y1 = obs_center.y

#         x2 = intersection.xy[0][1]
#         y2 = intersection.xy[1][1]
#         # return a, b, c so as ax + by >= c

#         return x2-x1, y2-y1, (x2-x1)*x2 + (y2-y1)*y2
    

class Rectangle_obs:
    def __init__(self, centers, widths, lengths, angles, side_avoid) -> None:
        """
        Create a rotated rectangle using Shapely.

        :param centers: Tuple of (x, y) representing the center of the rectangle.
        :param width: Width of the rectangle.
        :param length: Length of the rectangle.
        :param angle: Rotation angle in degrees.
        :return: A Shapely Polygon representing the rotated rectangle.
        """
        self.centers = centers
        self.widths = widths
        self.lengths = lengths
        self.angles = angles
        self.side_avoid = side_avoid.tolist()

        # initialize the list of polygons' points that used for forming kx

        self.end_ls = [] # a, b represent the two end points of the segments

        self.rectangle_obs = []
        for i,center in enumerate(self.centers):
            rectangle = self.create_rotated_rectangle(center,self.widths,self.lengths,self.angles[i])
            # print("rectangle coods:",list(rectangle.exterior.coords))
            self.rectangle_obs.append(rectangle)

        self.abc3 = []

        for i,rectangle in enumerate(self.rectangle_obs):
            coords = list(rectangle.exterior.coords)
            # print("coords :",coords)
            longer_edges = [(coords[0], coords[1]), (coords[2], coords[3])]
            if side_avoid[i] == -1:
                line = self.get_line_equation(longer_edges[0][0], longer_edges[0][1])
                self.end_ls.append([longer_edges[0][0],longer_edges[0][1]])
                # print("line is :",line)
                line = tuple(-x for x in line)
                # print("line is :",line)
            elif side_avoid[i] == 1:
                line = self.get_line_equation(longer_edges[1][0], longer_edges[1][1])
                self.end_ls.append([longer_edges[1][0],longer_edges[1][1]])

            self.abc3.append(line)
            
        self.patches = []

        # print("rec abc3 :",self.abc3)
        # print("end ls:",self.end_ls)




    def create_rotated_rectangle(self, center, width, length, angle):
        """
        Create a rotated rectangle using Shapely.

        :param center: Tuple of (x, y) representing the center of the rectangle.
        :param width: Width of the rectangle.
        :param length: Length of the rectangle.
        :param angle: Rotation angle in degrees.
        :return: A Shapely Polygon representing the rotated rectangle.
        """
        # Create a rectangle centered at the origin
        rect = Polygon([(-length / 2, -width / 2),
                        (length / 2, -width / 2),
                        (length / 2, width / 2),
                        (-length / 2, width / 2)])

        # Rotate the rectangle around the origin
        rotated_rect = rotate(rect, angle, origin=(0, 0), use_radians=False)

        # Translate the rectangle to the desired center
        translated_rect = translate(rotated_rect, xoff=center[0], yoff=center[1])

        return translated_rect
    
    def check_points(self,points):
        """
        Check which points are within which circles.
        points: [[a,b],...]
        """
        results = {Point(point): [] for _, point in enumerate(points)}

        for i, rectangle in enumerate(self.rectangle_obs):
            for _, point in enumerate(points):
                if rectangle.contains(Point(point)):
                    results[Point(point)].append([self.abc3[i],self.end_ls[i],i])
        
        return results
    
    def plot_rectangle(self,ax):
        for rectangle in self.rectangle_obs:
            x, y = rectangle.exterior.xy

            patch = ax.fill(x,y,color='black',zorder=1, alpha=1)[0]
            self.patches.append(patch)

    def remove_plot(self):
        for patch in self.patches:
            patch.remove()
        self.patches.clear()
    def get_line_equation(self, p1, p2):
        """
        Calculate the line equation in the form ax + by - c = 0 for the line passing through points p1 and p2.

        :param p1: Tuple of (x1, y1) representing the first point.
        :param p2: Tuple of (x2, y2) representing the second point.
        :return: Tuple (a, b, c) representing the coefficients of the line equation ax + by - c = 0.
        """

        x1, y1 = p1
        x2, y2 = p2

        # Calculate coefficients
        a = y2 - y1
        b = x1 - x2
        c = a * x1 + b * y1

        return a, b, c
    
    