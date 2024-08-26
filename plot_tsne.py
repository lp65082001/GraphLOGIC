# coding: utf-8

import warnings
#from Bio import BiopythonDeprecationWarning
#warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
import matplotlib.pyplot as plt
from os import path
from sys import path as systemPath
#from cam import SingleGCN_CAM_res, SingleGCN_CAM_gp, df_device
from dataset import dataset_sel
import torch
import numpy as np
from torch import nn
from model import  GAT_n_tot
from torch_geometric.loader import DataLoader

plt.rcParams['figure.figsize']=(4,4)
save_dir = './bert4_final_d15/'

if __name__ == '__main__':

    total_data_train_alpha1 = np.load("./dataset/alpha1_train.npy")
    total_data_train_alpha1 = np.hstack((total_data_train_alpha1,np.ones((total_data_train_alpha1.shape[0],1))))
    total_data_test_alpha1 = np.load("./dataset/alpha1_test.npy")
    total_data_test_alpha1 = np.hstack((total_data_test_alpha1,np.ones((total_data_test_alpha1.shape[0],1))))
    total_data_train_alpha2 = np.load("./dataset/alpha2_train.npy")
    total_data_train_alpha2 = np.hstack((total_data_train_alpha2,np.ones((total_data_train_alpha2.shape[0],1))*2))
    total_data_test_alpha2 = np.load("./dataset/alpha2_test.npy")
    total_data_test_alpha2 = np.hstack((total_data_test_alpha2,np.ones((total_data_test_alpha2.shape[0],1))*2))
    t1_ = np.vstack((total_data_train_alpha1,total_data_test_alpha1))
    t2_ = np.vstack((t1_,total_data_train_alpha2))
    total_data_list = np.vstack((t2_,total_data_test_alpha2))

    dataset_name = "bert4_total_real"
    data = dataset_sel(dataset_name)
    train_dataset, test_dataset = data.load()
    total_data = data.get_all_dataset()

    full_loader = DataLoader(total_data, len(total_data), shuffle=False)
 
    device = torch.device('cpu')
    model = GAT_n_tot(total_data,1)
    model.load_state_dict(torch.load(f'{save_dir}model{3}_dict.pt',
        map_location='cpu'))

    with torch.no_grad():
        model.eval()
        for data_ in full_loader:
            data = data_.to('cpu')
            out = model.get_embedding(data,data.edge_index).numpy()

from sklearn import manifold

tsne  = manifold.TSNE(n_components=2, init="pca",random_state=42)
X_tsne = tsne.fit_transform(out)

x_min, x_max = X_tsne.min(0),X_tsne.max(0)
X_norm = (X_tsne-x_min)/(x_max-x_min)

sequence_dict = {"0":"lightcoral","1":"cornflowerblue"}

colorlist = []
for i in range(0,total_data_list.shape[0]):
    colorlist.append(sequence_dict[total_data_list[i,2]])


for i in range(X_norm.shape[0]):
    plt.scatter(X_norm[i,0],X_norm[i,1],color=colorlist[i],edgecolor="black")

plt.title("t-sne (shuffle)",fontsize=12)
plt.axis('off')
plt.xticks([])
plt.yticks([])

plt.savefig(f"./figure/tsne_total.png",dpi=300,bbox_inches="tight")
plt.close()

