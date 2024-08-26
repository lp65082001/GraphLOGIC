# coding: utf-8
import warnings
#from Bio import BiopythonDeprecationWarning
#warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from os import path
from sys import path as systemPath
#from cam import SingleGCN_CAM_res, SingleGCN_CAM_gp, df_device
from dataset import dataset_sel
import torch
import numpy as np
from torch import nn
from visualization import Plotter
from model import GAT_n_tot
from torch_geometric.loader import DataLoader

def ecoding_amino_analysis(w,x,y,z):
    amino_table = np.array(["Ala","Cys","Asp","Glu","Arg","Ser","Val","Trp","Pro"])
    amino_ec = np.where(amino_table==w)[0]
    if (y=="1.0"):
        b = 1
    else:
        b = 2
    if(z==True):
        c = 0
    else:
        c = 1
    return int(amino_ec), int(x), int(b), c 


save_dir = './bert4_final_d15/'
t = "a1"

if __name__ == '__main__':

    total_data_list = np.load("./dataset/data15-23.npy")

    dataset_name = "bert4_pred"
    data = dataset_sel(dataset_name)
    total_data = data.load()

    #total_data_list = total_data_list[test_label,:]

    full_loader = DataLoader(total_data, len(total_data), shuffle=False)
    #full_loader = DataLoader(test_dataset, len(test_dataset), shuffle=False)

    device = torch.device('cpu')
    model = GAT_n_tot(total_data,1)
    model.load_state_dict(torch.load(f'{save_dir}model{3}_dict.pt',
        map_location='cpu'))

    with torch.no_grad():
        model.eval()
        for data_ in full_loader:
            data = data_.to('cpu')
            out = model(data,data.edge_index)
            out = nn.Softmax(dim=1)(out)
            out_ = out[:,1]
            pred = out.argmax(dim=1)
    print(pred)


