import pandas as pd
import math
import json
from tqdm import tqdm


def load_df(grid, path_to_raw, save=False):
    """
    Put together & process all dynamic/time series data and static data w/ assignments to grid nodes

    Args:
        grid (dict): grid from create_grid
        path_to_raw (str): path to raw data json

    Returns:
        Pandas DataFrame df with processed data per node
    """

    with open(path_to_raw, 'r') as f:
        dat = json.load(f)
        data = dat["earthquakes"]
        plate_bounds = dat["plate_boundaries"]
        faults = dat["fault_lines"]

    prep = {
        "node_id":[],
        "node_lat":[],
        "node_lon":[],
        "node_edges":[],
        "node_plate_dist":[],
        "node_fault_dist":[],
        "time":[],
        "mag":[],
        "depth":[]
    }

    for j in tqdm(range(len(grid["centers"])), desc="Aggregating Graph Data"):
        center = grid["centers"][j]
        edges = grid["edges"][j]

        node_id = j + 1
        node_lat = center[0]
        node_lon = center[1]
        node_edges = edges

        plate_dists = []
        for i in range(len(plate_bounds)):
            plate_dists.append(haversine(node_lat, node_lon, plate_bounds[i][0], plate_bounds[i][1]))
        node_plate_dist = min(plate_dists)

        fault_dists = []
        for i in range(len(faults)):
            fault_dists.append(haversine(node_lat, node_lon, faults[i][0], faults[i][1]))
        node_fault_dist = min(fault_dists)


        times = []
        mags = []
        depths = []
        for k in range(len(data["time"])):  # Added tqdm iterator
            time = data["time"][k]
            mag = data["mag"][k]
            lat = data["lat"][k]
            lon = data["lon"][k]
            depth = data["depth"][k]
            
            if lat > edges[0] and lat < edges[1] and lon > edges[2] and lon < edges[3]:

                times.append(time)
                mags.append(mag)
                depths.append(depth)

            elif lat >= edges[0] and lat <= edges[1] and lon >= edges[2] and lon <= edges[3]:

               
                times.append(time)
                mags.append(mag)
                depths.append(depth)
                data["time"].pop(k)
                data["mag"].pop(k)
                data["lat"].pop(k)
                data["lon"].pop(k)
                data["depth"].pop(k)
                k = k-1

        prep["depth"].append(depths)
        prep["mag"].append(mags)
        prep["node_edges"].append(node_edges)
        prep["node_id"].append(node_id)
        prep["node_lat"].append(node_lat)
        prep["node_lon"].append(node_lon)
        prep["time"].append(times)
        prep["node_fault_dist"].append(node_fault_dist)
        prep["node_plate_dist"].append(node_plate_dist)


    df = pd.DataFrame(prep)
    if save:
        df.to_csv("eq_forecast/data/raw/combined.csv")

    return prep




def haversine(lat1, lon1, lat2, lon2): #returns in Kilometers
     
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0
 

    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0
 

    a = (pow(math.sin(dLat / 2), 2) +
         pow(math.sin(dLon / 2), 2) *
             math.cos(lat1) * math.cos(lat2));
    rad = 6371
    c = 2 * math.asin(math.sqrt(a))
    return rad * c
