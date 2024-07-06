import numpy as np
from scipy.spatial import KDTree

def min_max_dist(x, y, num_bins):

    points = np.stack((x,y),axis = -1)

    tree = KDTree(points)  # O(m log m)
    max_distance = 0
    min_distance = float('inf')

    # Iterate over all points to find the farthest and closest pair
    for i, point in enumerate(points):  # O(m)
        # Query all other points
        distances, indices = tree.query(point, k=points.shape[0])  # O(log m) for each query
        farthest_distance = distances[-1]
        closest_distance = distances[1]  # Skip the first element which is the point itself

        if farthest_distance > max_distance:
            max_distance = farthest_distance

        if closest_distance < min_distance:
            min_distance = closest_distance

    step_size = (max_distance - min_distance + 0.1)/num_bins

    up_bound = max_distance + 0.1
    low_bound = min_distance + 0.1   

    return low_bound, up_bound, step_size 
