import numpy as np
from scipy.spatial import KDTree

def min_max_dist(x, y, num_bins):

    r"""Calculates lower and upper bound of bin sizes for variogram calculation,
        in addition to the step size. 

        Parameters
        ----------

        x: :class:`np.ndarray`
            X position array.
        y: :class:`np.ndarray`
            Y position array.
        num_bins: :class:`int`
            Number of variogram bins.

        Returns
        -------

        low_bound: :class:`float`
            Lower bound of variogram bin size.
        up_bound: :class:`float`
            Upper bound of variogram bin size.
        step_size: :class:`float`
            Step size, calculated as maximum distance between data points - 
            minimum distance between data points divided by the number of bins.
    """

    points = np.stack((x,y),axis = -1)

    tree = KDTree(points)
    max_distance = 0
    min_distance = float('inf')

    # Iterate over all points to find the farthest and closest pair
    for i, point in enumerate(points):  
        # Query all other points
        distances, indices = tree.query(point, k=points.shape[0])

        farthest_distance = distances[-1]
        closest_distance = distances[1]  
        
        if farthest_distance > max_distance:
            max_distance = farthest_distance

        if closest_distance < min_distance:
            min_distance = closest_distance

    step_size = (max_distance - min_distance + 0.1)/num_bins

    up_bound = max_distance + 0.1
    low_bound = min_distance + 0.1   

    return low_bound, up_bound, step_size 
