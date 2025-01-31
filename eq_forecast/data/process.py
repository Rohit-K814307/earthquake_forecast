import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from eq_forecast.data.aggregate import haversine
from tqdm import tqdm
from eq_forecast.visualization.distributions import *
from eq_forecast.visualization.heatmap import *


def fill_missing(data_dict):

    #first impute 0's into nodes with no earthquakes & then impute meds

    for i in range(len(data_dict["time"])):
        times = data_dict["time"][i]
        depths = data_dict["depth"][i]

        if len(times) == 0:

            data_dict["time"][i] = [0]
            data_dict["mag"][i] = [0]
            data_dict["depth"][i] = [0]

        if None in depths:

            med = np.median([x for x in depths if x != None])
            data_dict["depth"][i] = [med if x == None else x for x in depths]

    return data_dict


def preprocess_earthquake_data_global(data_dict):
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()

    #print(f"Original node_plate_dist shape: {len(data_dict['node_plate_dist'])}")
    node_plate_dist_scaled = scaler_standard.fit_transform(np.array(data_dict["node_plate_dist"]).reshape(-1, 1)).flatten()
    #print(f"Processed node_plate_dist shape: {len(node_plate_dist_scaled)}")

    #print(f"Original node_fault_dist shape: {len(data_dict['node_fault_dist'])}")
    node_fault_dist_scaled = scaler_standard.fit_transform(np.array(data_dict["node_fault_dist"]).reshape(-1, 1)).flatten()
    #print(f"Processed node_fault_dist shape: {len(node_fault_dist_scaled)}")

    def flatten_and_scale_global(nested_list, scaler, feature_name):
        flattened = [item for sublist in nested_list for item in sublist]
        #print(f"Original {feature_name} shape: {len(nested_list)} nodes, {sum(len(sublist) for sublist in nested_list)} total items")
        if len(flattened) == 0:
            return nested_list  

        flattened_scaled = scaler.fit_transform(np.array(flattened).reshape(-1, 1)).flatten()

        result = []
        index = 0
        for sublist in nested_list:
            result.append(flattened_scaled[index:index + len(sublist)].tolist())
            index += len(sublist)
        #print(f"Processed {feature_name} shape: {len(result)} nodes, {sum(len(sublist) for sublist in result)} total items")
        return result

    #time_scaled = flatten_and_scale_global(data_dict["time"], scaler_minmax, "time")
    #mag_scaled = flatten_and_scale_global(data_dict["mag"], scaler_standard, "mag")
    depth_scaled = flatten_and_scale_global(data_dict["depth"], scaler_standard, "depth")


    return {
        "node_id": data_dict["node_id"],
        "node_lat": data_dict["node_lat"],
        "node_lon": data_dict["node_lon"],
        "node_edges": data_dict["node_edges"],
        "node_plate_dist": node_plate_dist_scaled.tolist(),
        "node_fault_dist": node_fault_dist_scaled.tolist(),
        "time": data_dict["time"],
        "mag": data_dict["mag"],
        "depth": depth_scaled
    }


def get_deltas(data_dict, time_sensitivity):


    for i in range(1, len(data_dict)):

        time = data_dict[i]["time"]
        prev_time = data_dict[i-1]["time"]


        time = int(time) - int(prev_time)
        time = np.log1p(time)

        delt = np.exp(-time_sensitivity * time)
        data_dict[i]["delta"] = delt
    
    data_dict[0]["delta"] = 0

    return data_dict




def create_time_series(data_dict, n_lat, n_lon, grid, window_duration=3600, overlap_duration=1800, time_sensitivity=0.01):
    
    data = []
    num_nodes = n_lat * n_lon

    for j in tqdm(range(len(data_dict["time"])), desc="Identifying Events"):
        times = data_dict["time"][j]
        mags = data_dict["mag"][j]
        depths = data_dict["depth"][j]
        id = data_dict["node_id"][j]
        lat = data_dict["node_lat"][j]
        lon = data_dict["node_lon"][j]
        edges = data_dict["node_edges"][j]
        plate_dist = data_dict["node_plate_dist"][j]
        fault_dist = data_dict["node_fault_dist"][j]

        for k in range(len(times)):
            if times[k] != 0:
                data.append({
                    "time": times[k],
                    "mag": mags[k],
                    "depth": depths[k],
                    "node_id": id,
                    "node_lat": lat,
                    "node_lon": lon,
                    "node_edges": edges,
                    "node_plate_dist": plate_dist,
                    "node_fault_dist": fault_dist
                })

    sorted_data = get_deltas(sorted(data, key=lambda x: x['time']), time_sensitivity=time_sensitivity)

    def get_time_graphs(current_start, current_end):
        time_window = [event for event in sorted_data if current_start <= event["time"] < current_end]

        window_graphs = [] 
        for event in time_window:
            graph = []
            affected_node = event["node_id"]

            for i in range(num_nodes):
                if i + 1 != affected_node:
                    graph.append({
                        "node_id": i + 1,
                        "node_lat": grid["centers"][i][0],
                        "node_lon": grid["centers"][i][1],
                        "node_plate_dist": data_dict["node_plate_dist"][i],
                        "node_fault_dist": data_dict["node_fault_dist"][i],
                        "mag": 0,
                        "depth": 0,
                        "delta": 0
                    })
                else:
                    graph.append({
                        "node_id": affected_node,
                        "node_lat": grid["centers"][i][0],
                        "node_lon": grid["centers"][i][1],
                        "node_plate_dist": data_dict["node_plate_dist"][i],
                        "node_fault_dist": data_dict["node_fault_dist"][i],
                        "mag": event["mag"],
                        "depth": event["depth"],
                        "delta": event["delta"]
                    })

            window_graphs.append(graph) 


        return window_graphs

    # Initialize time-based windows
    start_time = sorted_data[0]["time"]
    end_time = sorted_data[-1]["time"]
    current_start = start_time
    current_end = current_start + window_duration

    all_graphs = []
    window_labels = [] 

    pbar = tqdm(total=(end_time - current_start) // (window_duration - overlap_duration), 
                desc="Collecting Time Window Graphs", unit="window")

    while current_start < end_time:


        window_graphs = get_time_graphs(current_start, current_end)
        
        next_window_start = current_end
        next_window_end = next_window_start + window_duration
        if next_window_end <= end_time:
            
            label_graphs = get_time_graphs(next_window_start, next_window_end)

            all_graphs.append(window_graphs)  
            window_labels.append(label_graphs)  
        else:
            
            break 

        current_start += (window_duration - overlap_duration)
        current_end = current_start + window_duration
        pbar.update(1)

    pbar.close()


    # MAKE FEATURE MATRICES IN TIME SERIES ORDER
    max_events_x = max(len(window) for window in all_graphs)
    max_events_y = max(len(window) for window in window_labels)
    max_events = max(max_events_x, max_events_y)
    print(f"Max events per sequence: {max_events}")

    def create_feature_matrices(all_graphs):
        feature_matrices = []
        for window_num in tqdm(range(len(all_graphs)), desc="Building Feature Matrices"):
            feature_matrices_window = []
            for graph_num in range(len(all_graphs[window_num])):
                graph = all_graphs[window_num][graph_num]
                feature_matrix = np.zeros((num_nodes, 5)) 

                for i in range(num_nodes):
                    feature_matrix[i][0] = graph[i]["node_plate_dist"]
                    feature_matrix[i][1] = graph[i]["node_fault_dist"]
                    feature_matrix[i][2] = graph[i]["mag"]  
                    feature_matrix[i][3] = graph[i]["depth"] 
                    feature_matrix[i][4] = graph[i]["delta"]   

                feature_matrices_window.append(feature_matrix)

  
            while len(feature_matrices_window) < max_events:
                zero_matrix = np.zeros((num_nodes, 5)) 
                for i in range(num_nodes):
                    zero_matrix[i][0] = data_dict["node_plate_dist"][i]  
                    zero_matrix[i][1] = data_dict["node_fault_dist"][i]  
                    zero_matrix[i][2] = 0
                    zero_matrix[i][3] = 0
                    zero_matrix[i][4] = 0  
                feature_matrices_window.append(zero_matrix)

            feature_matrices.append(feature_matrices_window)
        return feature_matrices
    
    def create_label_matrices(all_graphs):
        padding_matrix =[]
        feature_matrices = []
        for window_num in tqdm(range(len(all_graphs)), desc="Building Label Matrices"):
            feature_matrices_window = []
            pad_window = []
            for graph_num in range(len(all_graphs[window_num])):
                graph = all_graphs[window_num][graph_num]
                feature_matrix = np.zeros((num_nodes, 3))
                pad_window.append(1) 

                for i in range(num_nodes):
                    feature_matrix[i][0] = graph[i]["mag"] 
                    feature_matrix[i][1] = graph[i]["depth"]
                    feature_matrix[i][2] = graph[i]["delta"]

                feature_matrices_window.append(feature_matrix)

            while len(feature_matrices_window) < max_events:
                zero_matrix = np.zeros((num_nodes, 3))  
                for i in range(num_nodes):
                    zero_matrix[i][0] = 0 
                    zero_matrix[i][1] = 0 
                    zero_matrix[i][2] = 0
                feature_matrices_window.append(zero_matrix)
                pad_window.append(0)

            feature_matrices.append(feature_matrices_window)
            padding_matrix.append(pad_window)
        return np.array(feature_matrices), np.array(padding_matrix)
    

    feature_matrices = np.array(create_feature_matrices(all_graphs))
    label_matrices, lab_pad_mat  = create_label_matrices(window_labels)

    #MAKE GRAPH EDGE MATRIX WEIGHTED BY DISTANCE
    edge_matrix = np.zeros((num_nodes, num_nodes))

    #first make binary matrix to determine connections
    for row_i in range(n_lat):
        for col_i in range(n_lon):

            current_node = row_i * n_lon + col_i

            if col_i < n_lon - 1:
                right_neighbor = current_node + 1
                edge_matrix[current_node, right_neighbor] = 1
                edge_matrix[right_neighbor, current_node] = 1

            if row_i < n_lat - 1:
                bottom_neighbor = current_node + n_lon
                edge_matrix[current_node, bottom_neighbor] = 1
                edge_matrix[bottom_neighbor, current_node] = 1

    for j in range(len(edge_matrix)):
        for k in range(len(edge_matrix[j])):

            if edge_matrix[j][k] == 1:

                node1_lat = data_dict["node_lat"][j]
                node1_lon = data_dict["node_lon"][j]
                node2_lat = data_dict["node_lat"][k]
                node2_lon = data_dict["node_lon"][k]
                distance = haversine(node1_lat, node1_lon, node2_lat, node2_lon)

                edge_matrix[j][k] = distance
                edge_matrix[k][j] = distance

    non_zero_distances = edge_matrix[edge_matrix > 0]
    min_distance = non_zero_distances.min()
    max_distance = non_zero_distances.max()

    if max_distance > min_distance:
        edge_matrix[edge_matrix > 0] = 1 + (edge_matrix[edge_matrix > 0] - min_distance) * (1 / (max_distance - min_distance))


    return edge_matrix, feature_matrices, label_matrices, lab_pad_mat, max_events


def make_visualizations(output_dir, data_dict):

    print("making magnitude distribution...")
    plot_magnitude_distribution(data_dict, output_dir)

    print("making depth distribution...")
    plot_depth_distribution(data_dict, output_dir)

    print("making plate distance distribution")
    plot_node_plate_distance_distribution(data_dict, output_dir)

    print("making seismic heatmap...")
    interactive_seismic_heatmap(data_dict, output_dir)



def preprocess(data, grid, n_lat, n_lon, window_size=864000000, overlap=432000000, time_sensitivity=0.01, save=False):
    data = fill_missing(data)

    make_visualizations("eq_forecast/visualization/raw", data)
    data = preprocess_earthquake_data_global(data)
    make_visualizations("eq_forecast/visualization/preprocessed", data)

    edge_matrix, X, y, y_pad_matrix, events = create_time_series(data, n_lat, n_lon, grid, window_size, overlap, time_sensitivity=time_sensitivity)
    
    if save:
        pd.DataFrame(data).to_csv("eq_forecast/data/raw/preprocessed.csv", index=False)

    return X, y, edge_matrix, y_pad_matrix, events