import numpy as np


#Initialized with Contiguous U.S. Rectangular Boundary
def create_grid(top_left=(49.3547868, -124.7844079), bottom_right=(24.396308, -66.93457), n_lat=25, n_lon=35):
    """
    Create a grid of cells over a geographic bounding box w/ number of cells along lats and longs

    Args:
        top_left (tuple): (lat, lon) of top left bounding box
        bottom_right (tuple): (lat, long) of bottom right bounding box
        n_lat (int): number of nodes along lat (north to south)
        n_long (int): number of nodes along lon (east to west)

    Returns:
        dict containing:
            - 'centers': np array with (lat, lon) of grid centers
            - 'edges': np array with (lat_min, lat_max, lon_min, lon_max):

                    (lat_max, lon_min)       (lat_max, lon_max)                    
                                +-------------+
                                |             |
                                |             |
                                |             |
                                +-------------+
                    (lat_min, lon_min)       (lat_min, lon_max)

            - 'lat_res': lat resolution in degrees (north to south)
            - 'lon_res": lon resolution in degrees (east to west)
    """

    lat_start, lon_start = top_left
    lat_end, lon_end = bottom_right

    lat_res = (lat_start - lat_end) / n_lat
    lon_res = (lon_end - lon_start) / n_lon

    lats = np.linspace(lat_start, lat_end, n_lat + 1)
    lons = np.linspace(lon_start, lon_end, n_lon + 1)

    centers = []
    edges = []
    for i in range(n_lat):
        for j in range(n_lon):
            lat_min, lat_max = lats[i+1], lats[i]
            lon_min, lon_max = lons[j], lons[j+1]
            edges.append([lat_min, lat_max, lon_min, lon_max])

            center_lat = (lat_min + lat_max) / 2
            center_lon = (lon_min + lon_max) / 2
            centers.append([center_lat, center_lon])
    
    centers = np.array(centers)
    edges = np.array(edges)

    return {
        "centers": centers, 
        "edges": edges,
        "lat_res": lat_res,
        "lon_res": lon_res
    }

