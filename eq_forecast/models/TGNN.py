import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN2


class TemporalGNN(nn.Module):
    def __init__(self, batch_size, hidden_dim, node_features, periods, out_features):  
        super(TemporalGNN, self).__init__()

        self.tgnn = A3TGCN2(in_channels=node_features, 
                           out_channels=hidden_dim,  
                           periods=periods,
                           batch_size=batch_size)
        
        self.linear = torch.nn.Linear(hidden_dim, out_features * periods) 

        self.periods = periods
        self.out_features = out_features
        self.batch_size = batch_size

    def forward(self, x, edge_index, edge_attr):
        h = self.tgnn(x, edge_index, edge_attr)
        h = F.relu(h)
        h = self.linear(h)  #(b, num_nodes, out_features * periods)

        #(b, num_nodes, out_features, periods)
        h = h.reshape(self.batch_size, -1,  self.out_features, self.periods) 

        return h