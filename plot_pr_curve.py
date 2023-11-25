# coding: utf-8

import warnings
#from Bio import BiopythonDeprecationWarning
#warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
import gc

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
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import pickle



def prediction(dataset_name,save_dir):
    #t = "a1"
    #save_dir = './bert4_total_gat_0.0008_0.45_8/'
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

    data = dataset_sel(dataset_name)
    train_dataset, test_dataset = data.load()
    total_data = data.get_all_dataset()
    test_label = np.array(data.get_info()[2])
    total_data_list = total_data_list[test_label,:]
    full_loader = DataLoader(test_dataset, len(test_dataset), shuffle=False)

    device = torch.device('cpu')
    model = GAT_n_tot(train_dataset,1)
    model.load_state_dict(torch.load(f'{save_dir}model{3}_dict.pt',
        map_location='cpu'))
    
    correct = 0
    with torch.no_grad():
        model.eval()
        for data_ in full_loader:
            data = data_.to('cpu')
            out = model(data,data.edge_index)
            out = nn.Softmax(dim=1)(out)
            out_ = out[:,1]
            pred = out.argmax(dim=1)
            g_true = (data.y).argmax(dim=1)
            correct +=int((pred==g_true).sum())
            g_true_ = g_true.cpu()
    
    c_gtrue_a1 = g_true_[np.where(total_data_list[:,3]=='1.0')[0]]
    c_out_true_a1 = out_[np.where(total_data_list[:,3]=='1.0')[0]]
    
    c_gtrue_a2 = g_true_[np.where(total_data_list[:,3]=='2.0')[0]]
    c_out_true_a2 = out_[np.where(total_data_list[:,3]=='2.0')[0]]
    
    gc.collect()
    with open(f"{dataset_name}.pt", "wb") as fp:   #Pickling
        pickle.dump([c_gtrue_a1, c_out_true_a1, c_gtrue_a2, c_out_true_a2], fp)

def max_f1(precision,recall):
    with np.errstate(divide='ignore', invalid='ignore'):
        aupr = np.trapz(np.flip(precision), x=np.flip(recall))
        f1 = 2*recall*precision / (recall+precision)
        f1_max_idx = np.nanargmax(f1)
        f1_max = f1[f1_max_idx]
    return f1_max,f1_max_idx,aupr,

if __name__ == '__main__':

    #prediction("bert4_total_real_8", './bert4_test_total8/')
    prediction("bert4_total_real", './bert4_shuffle/')
    
    with open("./bert4_total_real.pt", "rb") as fp:   # Unpickling
        contact_12 = pickle.load(fp)

    #fpr_roc, tpr_roc, thresholds_roc = roc_curve(g_true_, out_)
    precision_pr_12_a1, recall_pr_12_a1, thresholds_pr_12_a1 = precision_recall_curve(contact_12[0], contact_12[1])
    precision_pr_12_a2, recall_pr_12_a2, thresholds_pr_12_a2 = precision_recall_curve(contact_12[2], contact_12[3])


    fig = plt.figure('pr', figsize=(3,3), dpi=300, constrained_layout=True)
    ax = fig.gca()

    # f1 contour
    levels = 10
    spacing = np.linspace(0, 1, 1000)
    x, y = np.meshgrid(spacing, spacing)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = 2 / (1/x + 1/y)
    locx = np.linspace(0, 1, levels, endpoint=False)[1:]
    cs = ax.contour(x, y, f1, levels=levels, linewidths=1, colors='k',
                    alpha=0.3)
    ax.clabel(cs, inline=True, fmt='F1=%.1f',
                    manual=np.tile(locx,(2,1)).T)

    # compute f1_max and aupr
    a1_12_f1_max,a1_12_f1_max_index, a1_12_aupr = max_f1(precision_pr_12_a1, recall_pr_12_a1)
    a2_12_f1_max,a2_12_f1_max_index, a2_12_aupr = max_f1(precision_pr_12_a2, recall_pr_12_a2)

    ax.plot(recall_pr_12_a1, precision_pr_12_a1, lw=1, color='slateblue', linestyle='solid', label=f"ɑ1 (shuffle),AUPR:{a1_12_aupr:.2}")
    ax.plot(recall_pr_12_a2, precision_pr_12_a2, lw=1, color='slateblue', linestyle='dashed', label=f"ɑ2 (shuffle),AUPR:{a2_12_aupr:.2}")
    plt.legend(loc='lower left',edgecolor="black")

    plt.xlabel('recall',fontsize=12)
    plt.ylabel('precision',fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("./figure/pr_curve_total.png")

    plt.close() 



    
        
