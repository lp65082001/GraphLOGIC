#!/usr/bin/env python
# coding: utf-8

import torch
import os
import random

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool
from torch import nn


# model #
class GAT_n_tot(torch.nn.Module):
    def __init__(self,train_dataset,layer_n):
        super(GAT_n_tot, self).__init__()
        torch.manual_seed(12345) 
        
        hidden_channels = 9

        self.conv_block = nn.ModuleList([])
        for i in range(layer_n):
            dim_input = train_dataset.num_node_features if i == 0 else hidden_channels
            conv = GATConv(dim_input, hidden_channels)
            # default arguments to GCNConv (and MessagePassing)
            # aggr='add', improved=False, add_self_loops=True
            self.conv_block.append(conv)
        self.fc1 = Linear(layer_n*hidden_channels,train_dataset.num_classes)
        self.drop = nn.Dropout(p=0.5)
        
    def get_embedding(self,data, edge_index):
        batch = data.batch
        x = data.x
        jk_connection = []
        for layer_idx, conv in enumerate(self.conv_block):
            x = conv(x, edge_index)
            x = F.leaky_relu(x)
            jk_connection.append(x)
        # Readout layer
        jk_connection = torch.cat(jk_connection, dim=1)
        return global_max_pool(jk_connection, batch)

    def forward(self,data, edge_index):  
        batch = data.batch
        x = data.x
        jk_connection = []
        for layer_idx, conv in enumerate(self.conv_block):
            x = conv(x, edge_index)
            x = F.relu(x)
            jk_connection.append(x)
        # Readout layer
        jk_connection = torch.cat(jk_connection, dim=1)
        x = global_max_pool(jk_connection, batch)
        x = self.fc1(x)
        x = self.drop(x)
        return x 
class GAT_n_tot_only(torch.nn.Module):
    def __init__(self,train_dataset,layer_n):
        super(GAT_n_tot_only, self).__init__()
        torch.manual_seed(12345) 
        
        hidden_channels = 8

        self.conv_block = nn.ModuleList([])
        for i in range(layer_n):
            dim_input = train_dataset.num_node_features if i == 0 else hidden_channels
            conv = GATConv(dim_input, hidden_channels)
            # default arguments to GCNConv (and MessagePassing)
            # aggr='add', improved=False, add_self_loops=True
            self.conv_block.append(conv)
        self.fc1 = Linear(layer_n*hidden_channels,train_dataset.num_classes)
        self.drop = nn.Dropout(p=0.5)

    def forward(self,data, edge_index):  
        batch = data.batch
        x = data.x
        jk_connection = []
        for layer_idx, conv in enumerate(self.conv_block):
            x = conv(x, edge_index)
            x = F.relu(x)
            jk_connection.append(x)
        # Readout layer
        jk_connection = torch.cat(jk_connection, dim=1)
        x = global_max_pool(jk_connection, batch)
        x = self.fc1(x)
        x = self.drop(x)
        return x 
