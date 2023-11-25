# coding: utf-8

#from .control_suite import Experiment, get_mfgo_dict

import sys
from os import path
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_max_pool
from torch_geometric.loader import DataLoader

df_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Feature_Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, model, input, output):
        self.feature = output.detach().clone()

class Gradient_Hook():
    def __init__(self, module):
        self.hook = module.register_hook(self.hook_fn)
    def hook_fn(self, grad):
        self.gradient = grad.detach().clone()

class CAM():
    def __init__(self, nn_model, save_dir, device=df_device):

        if isinstance(nn_model, nn.DataParallel):
            nn_model = nn_model.module

        self.model = nn_model.to(device)
        self.device = device
        self.save_dir = save_dir

    def set_dataset(self, dataset):

        #self.n_GO_terms = dataset.n_GO_terms
        self.dataset = dataset

        #self.mfgo_dict = get_mfgo_dict(dataset)
        #self.n_GO_terms = dataset.n_GO_terms

        #print(f'Dataset: {len(dataset)} entries')

    def cam_this(self, ID, thres):
        pass

    def save_output(self, ID, output, name_tag):
        np.savetxt(
            path.join(self.save_dir, f'pred.{ID}.{name_tag}.csv'),
            output.T,
            header='confidence',
            fmt='%e'
        )

class SingleGCN_CAM_res(CAM):

    def __init__(self, nn_model, save_dir, device=df_device):
        return super().__init__(nn_model, save_dir, device)

    def set_dataset(self, dataset):
        return super().set_dataset(dataset)

    def cam_this(self, ID, name_tag, thres=0.5, save=False, save_dir='.'):
        #print(self.dataset)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=0
        )

        for i, data in enumerate(self.dataloader):
            # prepare data
            #print(i)
            data.x = data.x.float()
            data.y = data.y.float()
            data.edge_index = data.edge_index.long()
            data = data.to(self.device)

            # switch off training-specific layers, e.g. dropout
            self.model.eval()

            # register hook to all conv layers to extract feature map
            f_hooks = []
            for conv in self.model.conv_block:
                f_hooks.append(Feature_Hook(conv))
            

            # forward pass and extract the feature values
            self.model(data,data.edge_index)

            feat = torch.cat([f_hook.feature for f_hook in f_hooks],
                             dim=1).requires_grad_(True)
            n_res = feat.shape[0]
            
            # register hooks to extract gradients
            g_hook = Gradient_Hook(feat)

            # forward pass to obtain prediction
            x = global_max_pool(feat, data.batch)
            x = self.model.fc1(x)

            #x = self.model.lin2(x)
            output = torch.sigmoid(x)
            n_classes = output.size(dim=1)

            # back-propagate using output layer values and extract gradient
            pred = torch.where(output >= thres, 1, 0)
            cam = np.empty((n_res, n_classes+1))
            # compute CAM for entirety of prediction
            output.backward(gradient=pred, retain_graph=True)
            grad = g_hook.gradient.detach()
            inner_prod = torch.mul(grad, feat.detach())
            cam[:,0] = F.relu(torch.sum(inner_prod, dim=1)).numpy().T
            # iteratively get gradient corresponding to each GO class
            for idx in range(n_classes):
                self.model.zero_grad()
                ext_grad = torch.zeros(output.shape).to(self.device)
                ext_grad[:,idx] = 1
                output.backward(gradient=ext_grad, retain_graph=True)
                grad = g_hook.gradient.detach()

                # compute CAM
                inner_prod = torch.mul(grad, feat.detach())
                cam[:,idx+1] = F.relu(torch.sum(inner_prod, dim=1)).numpy().T

            if save:
                np.save(self.save_dir+f'./cam_data/cam.{i}.{name_tag}-resid.npy', cam[np.newaxis,:,:])
                #np.save(self.save_dir+f'cam.{i}.{name_tag}-gp.npy', cam2[np.newaxis,:,:])
                output = output.detach().numpy()
                super().save_output(i, output, name_tag)

        return cam, output



