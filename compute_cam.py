# coding: utf-8

import warnings
#from Bio import BiopythonDeprecationWarning
#warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
import torch
from os import path
from sys import path as systemPath
from cam import SingleGCN_CAM_res, df_device
from dataset import dataset_sel
import numpy as np
from visualization import Plotter
from model import GAT_n_tot

save_dir = './bert4_shuffle/'

if __name__ == '__main__':

    dataset_name = "bert4_total_real"
    data = dataset_sel(dataset_name)
    train_dataset, test_dataset = data.load()
    total_data = data.get_all_dataset()

    # compute Grad-CAM by all cam #
    device = torch.device('cpu')
    model = GAT_n_tot(total_data,1)
    model.load_state_dict(torch.load(f'{save_dir}model{3}_dict.pt',
        map_location='cpu'))
    CAM_res = SingleGCN_CAM_res(model, save_dir)
    CAM_res.set_dataset(total_data)
    CAM_res.cam_this(0, name_tag='tbert4',
                      thres=0.5, save=True)
    '''
    # plot CAM by one case #
    res_cam = np.load(f'{save_dir}/cam_data/cam.55.tbert4-resid.npy')
    plot_figure_model = Plotter("./")
    plot_figure_model.plot_cam(res_cam,name="arg")
    '''

