import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import numpy as np

from eq_forecast.data.grid import create_grid
from eq_forecast.data.aggregate import load_df
from eq_forecast.data.process import preprocess
from eq_forecast.data.load import load_agg_data

class EQ_Dataset(Dataset):

    def __init__(self, 
                 batch_size=32,
                 start_time="1950-01-01", 
                 end_time="2025-01-01", 
                 min_lat=24.39630, 
                 max_lat=49.3547868, 
                 min_lon=-124.7844079, 
                 max_lon=-66.93457, 
                 min_magnitude=3.5, 
                 n_lat=15, 
                 n_lon=20, 
                 window_time_length=864000000, 
                 overlap_time_length=432000000,
                 time_sensitivity=0.01, 
                 save_data=False, 
                 agg_new_data=False):
        
        super().__init__()

        self.grid = create_grid(n_lat = n_lat, n_lon = n_lon)

        if agg_new_data:
            load_agg_data(start_time=start_time, end_time=end_time, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon, min_magnitude=min_magnitude, save=save_data)

        self.raw_data = load_df(self.grid, "eq_forecast/data/raw/eq_data.json", save_data)

        self.feature_mats, self.labels, self.adj_mat, y_pad = preprocess(self.raw_data, self.grid, n_lat, n_lon, window_time_length, overlap_time_length, time_sensitivity=time_sensitivity, save=save_data)

        self.feature_mats = self.batch_and_pad_features(np.transpose(self.feature_mats, (0, 2, 3, 1)), batch_size)
        self.labels, self.pad_labels = self.batch_and_pad_labels(np.transpose(self.labels, (0, 2, 3, 1)), y_pad, batch_size)

        self.edge_index, self.edge_attr = self.adjacency_conversion(self.adj_mat)


    def batch_and_pad_features(self, arr, batch_size):
        total_length, num_nodes, num_features, num_timesteps = arr.shape  

        num_batches = int(np.ceil(total_length / batch_size))
        
        pad_size = num_batches * batch_size - total_length


        ex_arr = arr[0].copy() #shape (num_nodes, num_features, num_timesteps)
        ex_arr[:,-3:,:] = 0
        padding = np.array([ex_arr for _ in range(pad_size)])

        return np.concatenate([arr, padding], axis=0).reshape(num_batches, batch_size, num_nodes, num_features, num_timesteps)
    


    def batch_and_pad_labels(self, arr, pad_label, batch_size):
        total_length, num_nodes, num_features, num_timesteps = arr.shape  

        num_batches = int(np.ceil(total_length / batch_size))
        
        pad_size = num_batches * batch_size - total_length
        
        padding = np.zeros((pad_size, num_nodes, num_features, num_timesteps))

        pad_pad_label = np.zeros((pad_size, num_timesteps))

        pad_label = np.concatenate([pad_label, pad_pad_label], axis=0)
        
        padded_arr = np.concatenate([arr, padding], axis=0)
        
        batches = padded_arr.reshape(num_batches, batch_size, num_nodes, num_features, num_timesteps)
        batches_pad = pad_label.reshape(num_batches, batch_size, num_timesteps)

        return batches, batches_pad
    

    def adjacency_conversion(self, matrix):
        adjacency = torch.tensor(matrix, dtype=torch.float)

        conns = []

        for j in range(len(adjacency)):
            for k in range(len(adjacency[j])):
                if adjacency[j][k] > 0:
                    conns.append((j,k))

        conns_remove_extra = []
        
        for i in range(len(conns)):
            conn = conns[i]
            if (conn[1], conn[0]) not in conns_remove_extra and conn not in conns_remove_extra:
                conns_remove_extra.append(conn)

        row1 = [x[0] for x in conns_remove_extra]
        row2 = [x[1] for x in conns_remove_extra]
        edge_index = np.array([row1, row2]).astype(int)

        edge_attr = adjacency[edge_index[0], edge_index[1]]

        return edge_index, edge_attr


    def __len__(self):
        return len(self.feature_mats)
    

    def __getitem__(self, idx):
        feature_matrices = self.feature_mats[idx]
        label_matrices = self.labels[idx]

        return feature_matrices, label_matrices