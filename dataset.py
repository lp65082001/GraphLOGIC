#!/usr/bin/env python
# coding: utf-8

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import pandas as pd
import os
import numpy as np
import math
from scipy.spatial import distance_matrix
import pickle

real_structure_a11 = np.load("./reference_structure/alpha1_1.npy")
real_structure_a2 = np.load("./reference_structure/alpha2.npy")
real_structure_a12 = np.load("./reference_structure/alpha1_2.npy")
#import matplotlib.pyplot as plt
np.random.seed(0)
#0

def onehot(x1,x2,x3):
    total_t = np.zeros((x1.shape[0]+x2.shape[0]+x3.shape[0],22))
    num = 0
    for i in x1:
        total_t[num,i] = 1 
        num+=1
    for i in x2:
        total_t[num,i] = 1 
        num+=1
    for i in x3:
        total_t[num,i] = 1 
        num+=1
    return total_t
def one_hot_pos(x):
    #total_list = np.ones((1,338))
    #total_list[0,int(x)-5:int(x)+4] = 0
    #return np.array(total_list).reshape(-1,338)
    # ov1-ov5, gp1-gp4
    #overlap=35, gap=41, ov5=34
    total_list = np.zeros((1,9))
    
    a = int((int(x)-5)/76)
    b = int((int(x)+4)/76)
    a_ = (int(x)-5)%76
    b_ = (int(x)+4)%76
    a_og = int(a_/35)
    b_og = int(b_/35)
    
    if (a_og>=1):
        a_og=1
    else:
        a_og=0
    if (b_og>=1):
        b_og=1
    else:
        b_og=0
    #print(x)
    total_list[0,a*2+a_og]=1
    if (b==4 and b_>=35):
        total_list[0,8]=1
    else:
        total_list[0,b*2+b_og]=1
    
    return np.array(total_list).reshape(-1,9)
def one_hot_pos_mutation(x,y):
    mu_num = mutation_embedding(y)
    #total_list = np.ones((1,338))
    #total_list[0,int(x)-5:int(x)+4] = 0
    #return np.array(total_list).reshape(-1,338)
    # ov1-ov5, gp1-gp4
    #overlap=35, gap=41, ov5=34
    total_list = np.zeros((1,9))
    
    a = int((int(x)-5)/76)
    b = int((int(x)+4)/76)
    a_ = (int(x)-5)%76
    b_ = (int(x)+4)%76
    a_og = int(a_/35)
    b_og = int(b_/35)
    
    if (a_og>=1):
        a_og=1
    else:
        a_og=0
    if (b_og>=1):
        b_og=1
    else:
        b_og=0
    #print(x)
    total_list[0,a*2+a_og]=mu_num
    if (b==4 and b_>=35):
        total_list[0,8]=mu_num
    else:
        total_list[0,b*2+b_og]=mu_num
    
    return np.array(total_list).reshape(-1,9)
def embedding_node(x1,x2,x3):
    total_t = np.zeros((x1.shape[0]+x2.shape[0]+x3.shape[0],1024))
    num = 0
    for i in x1:
        total_t[num,:] = i
        num+=1
    for i in x2:
        total_t[num,:] = i
        num+=1
    for i in x3:
        total_t[num,:] = i
        num+=1
    return total_t
def mutation_embedding(x1):
    sequence_dict = {"Ala":"A" ,"Phe":"F","Cys":"C","Sec":"U","Asp":"D","Asn":"N","Glu":"E","Gln":"Q","Gly":"G","Leu":"L","Ile":"I","His":"H","Pyl":"O","Met":"M","Pro":"P","Arg":"R","Ser":"S","Thr":"T","Val":"V","Trp":"W","Tyr":"Y","Lys":"K","Hyp":"P","Hse":"H"}
    sequence_list = np.array(["A" ,"F","C","U","D","N","E","Q","G","L","I","H","O","M","P","R","S","T","V","W","Y","K"])
    return np.where(sequence_list == sequence_dict[x1])[0]
def mutation_embedding2(x1):
    amino_table = np.array([0,0,0,0,0,0,0,0,0])
    sequence_dict = {"Ala":"A" ,"Cys":"C","Asp":"D","Glu":"E","Pro":"P","Arg":"R","Ser":"S","Val":"V","Trp":"W"}
    sequence_list = np.array(["A" ,"C","D","E","R","S","V","W","P"])
    amino_table[np.where(sequence_list == sequence_dict[x1])[0][0]] = 1
    return amino_table
def mutation_embedding3(x1):
    sequence_dict = {"Ala":"A" ,"Cys":"C","Asp":"D","Glu":"E","Pro":"P","Arg":"R","Ser":"S","Val":"V","Trp":"W"}
    sequence_list = np.array(["A" ,"C","D","E","R","S","V","W","P"])
    #amino_table[np.where(sequence_list == sequence_dict[x1])[0][0]] = 1
    return np.where(sequence_list == sequence_dict[x1])[0][0]
def one_hot_pos_338(x):
    total_list = np.zeros((1,338))
    low_ = int(x)-4
    high_ = int(x)+5
    if low_<=0:
        low_=0
    if high_>=338:
        high_=338
    total_list[0,low_:high_] = 1
    return torch.tensor(total_list,dtype=torch.float)

def get_pos_type(t,p,tt,pt):
    tn = mutation_embedding3(t)
    tn_s = np.where(tt==tn)[0]
    pt_s = np.where(pt==p)[0]
    return np.intersect1d(tn_s,pt_s)[0]

def real_edge(pos,cutoff=12):
    text_pos = int((pos[0]-1)*3+12)
    text_pos2 = int((pos[0]-1)*3+9)
    #print(text_pos)
    a11 = real_structure_a11[text_pos-12:text_pos+15,:]
    a12 = real_structure_a12[text_pos-12:text_pos+15,:]
    if (pos[0]==1):
        a2 = real_structure_a2[text_pos2-9:text_pos2+15,:]
    elif(pos[0]>=335):
        a2 = real_structure_a2[text_pos2-12:1026,:]
    else:
        a2 = real_structure_a2[text_pos2-12:text_pos2+15,:]
    p_ = np.vstack((a11,a2))
    tot_p_ = np.vstack((p_,a12))
    dm = distance_matrix(tot_p_,tot_p_)
    result = np.where((np.triu(dm,1)<=cutoff) & (np.triu(dm,1)!=0))

    return np.array(np.vstack((result[0].T,result[1].T)))

### alpha1 ###

class dataset_a1_topology_protbert (InMemoryDataset):
    def __init__(self, root,path,path2,pos_ecode=None, transform=None, pre_transform=None):
        self.path = path
        self.path2 = path2
        super(dataset_a1_topology_protbert, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        
        total_data = np.load(self.path+".npy")
        aa_index  = (((((total_data[:,0]).astype('int'))-13)/3)+5).reshape(-1,1)
        #ecoding_data_a1 = np.load(self.path+"_tokenzer_a1.npy")
        #ecoding_data_a2 = np.load(self.path+"_tokenzer_a2.npy")
        #ecoding_data_ref = np.load(self.path+"_tokenzer_ref.npy")
        ma1_alpha1 = np.load(self.path2+"ma1_alpha1_bert.npy")
        ma1_alpha2 = np.load(self.path2+"ma1_alpha2_bert.npy")
        
        pos_tabel = np.load(self.path2+"pos.npy").reshape(-1,1)
        type_tabel = np.load(self.path2+"label.npy")
        
        
        i1 = []
        for i in range(0,26):
            i1.append([i,i+1])
        for i in range(27,53):
            i1.append([i,i+1])
        for i in range(54,80):
            i1.append([i,i+1])
        i1 = np.array(i1).reshape(-1,2)
        i2 = []
        for i in range(8):
            i2.append([i*3+1,i*3+30])
            i2.append([i*3+28,i*3+57])
            i2.append([i*3+3,i*3+55])
        i2 = np.array(i2).reshape(-1,2)
        i3 = []
        for i in range(8):
            i3.append([i*3,i*3+29])
            i3.append([i*3+27,i*3+56])
            i3.append([i*3+2,i*3+54])
        i3 = np.array(i3).reshape(-1,2)

        edge_ = np.vstack((i1,i2))
        edge = np.vstack((edge_,i3)).T
        edge_index = torch.tensor(edge, dtype=torch.long)

        for i in range(0,total_data.shape[0]):

            pt_index = get_pos_type(total_data[i,1],aa_index[i],type_tabel,pos_tabel)
            x = torch.tensor(embedding_node(ma1_alpha1[pt_index,:,:],ma1_alpha2[pt_index,:,:],ma1_alpha1[pt_index,:,:]), dtype=torch.float)
        
            if (total_data[i,2]=='0'):
                Yl = torch.tensor([[1,0]],dtype=torch.float)
            elif(total_data[i,2]=='1'):
                Yl = torch.tensor([[0,1]],dtype=torch.float)
            
            data_list.append(Data(x=x,edge_index=edge_index,y=Yl ))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class dataset_a1_topology_protbert_real (InMemoryDataset):
    def __init__(self, root,path,path2,cut=12, transform=None, pre_transform=None):
        self.path = path
        self.path2 = path2
        self.cut = cut
        super(dataset_a1_topology_protbert_real, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        
        total_data = np.load(self.path+".npy")
        aa_index  = (((((total_data[:,0]).astype('int'))-13)/3)+5).reshape(-1,1)
        
        pos_tabel = np.load(self.path2+"pos.npy").reshape(-1,1)
        type_tabel = np.load(self.path2+"label.npy")
        
        with open(self.path2+"ma1_alpha1_bert.pt", "rb") as fp: 
            ma1_alpha1 = pickle.load(fp)
        with open(self.path2+"ma1_alpha2_bert.pt", "rb") as fp: 
            ma1_alpha2 = pickle.load(fp)

        pos_tabel = np.load(self.path2+"pos.npy").reshape(-1,1)
        type_tabel = np.load(self.path2+"label.npy")
        

        for i in range(0,total_data.shape[0]):

            pt_index = get_pos_type(total_data[i,1],aa_index[i],type_tabel,pos_tabel)
            x = torch.tensor(embedding_node(ma1_alpha1[pt_index],ma1_alpha2[pt_index],ma1_alpha1[pt_index]), dtype=torch.float)
        
            if (total_data[i,2]=='0'):
                Yl = torch.tensor([[1,0]],dtype=torch.float)
            elif(total_data[i,2]=='1'):
                Yl = torch.tensor([[0,1]],dtype=torch.float)

            edge = real_edge(aa_index[i],cutoff=self.cut)

            edge_index = torch.tensor(edge, dtype=torch.long)
            
            data_list.append(Data(x=x,edge_index=edge_index,y=Yl ))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

 
class dataset_topology_protbert_total (InMemoryDataset):
    def __init__(self, root,path,path2,path3,pos_ecode=None, transform=None, pre_transform=None):
        self.path = path
        self.path2 = path2
        self.path3 = path3
        self.pos_e = pos_ecode
        super(dataset_topology_protbert_total, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):

        data_list = []
        
        total_data_train_alpha1 = np.load(self.path+"alpha1_train.npy")
        total_data_test_alpha1 = np.load(self.path+"alpha1_test.npy")
        total_data_train_alpha2 = np.load(self.path2+"alpha2_train.npy")
        total_data_test_alpha2 = np.load(self.path2+"alpha2_test.npy")


        aa_index_train_alpha1  = (((((total_data_train_alpha1[:,0]).astype('int'))-13)/3)+5).reshape(-1,1)
        aa_index_test_alpha1  = (((((total_data_test_alpha1[:,0]).astype('int'))-13)/3)+5).reshape(-1,1)
        aa_index_train_alpha2  = (((((total_data_train_alpha2[:,0]).astype('int'))-13)/3)+5).reshape(-1,1)
        aa_index_test_alpha2  = (((((total_data_test_alpha2[:,0]).astype('int'))-13)/3)+5).reshape(-1,1)

        ma1_alpha1 = np.load(self.path3+"ma1_alpha1_bert.npy")
        ma1_alpha2 = np.load(self.path3+"ma1_alpha2_bert.npy")
        ma2_alpha1 = np.load(self.path3+"ma2_alpha1_bert.npy")
        ma2_alpha2 = np.load(self.path3+"ma2_alpha2_bert.npy")

        pos_tabel = np.load(self.path3+"pos.npy").reshape(-1,1)
        type_tabel = np.load(self.path3+"label.npy")

        i1 = []
        for i in range(0,26):
            i1.append([i,i+1])
        for i in range(27,53):
            i1.append([i,i+1])
        for i in range(54,80):
            i1.append([i,i+1])
        i1 = np.array(i1).reshape(-1,2)
        i2 = []
        for i in range(8):
            i2.append([i*3+1,i*3+30])
            i2.append([i*3+28,i*3+57])
            i2.append([i*3+3,i*3+55])
        i2 = np.array(i2).reshape(-1,2)
        i3 = []
        for i in range(8):
            i3.append([i*3,i*3+29])
            i3.append([i*3+27,i*3+56])
            i3.append([i*3+2,i*3+54])
        i3 = np.array(i3).reshape(-1,2)

        edge_ = np.vstack((i1,i2))
        edge = np.vstack((edge_,i3)).T
        edge_index = torch.tensor(edge, dtype=torch.long)

        for i in range(0,total_data_train_alpha1.shape[0]):

            pt_index = get_pos_type(total_data_train_alpha1[i,1],aa_index_train_alpha1[i],type_tabel,pos_tabel)
            x = torch.tensor(embedding_node(ma1_alpha1[pt_index,:,:],ma1_alpha2[pt_index,:,:],ma1_alpha1[pt_index,:,:]), dtype=torch.float)
        
            if (total_data_train_alpha1[i,2]=='0'):
                Yl = torch.tensor([[1,0]],dtype=torch.float)
            elif(total_data_train_alpha1[i,2]=='1'):
                Yl = torch.tensor([[0,1]],dtype=torch.float)
            
            data_list.append(Data(x=x,edge_index=edge_index,y=Yl ))
        
        for i in range(0,total_data_test_alpha1.shape[0]):
            
            pt_index = get_pos_type(total_data_test_alpha1[i,1],aa_index_test_alpha1[i],type_tabel,pos_tabel)
            x = torch.tensor(embedding_node(ma1_alpha1[pt_index,:,:],ma1_alpha2[pt_index,:,:],ma1_alpha1[pt_index,:,:]), dtype=torch.float)
        
            if (total_data_test_alpha1[i,2]=='0'):
                Yl = torch.tensor([[1,0]],dtype=torch.float)
            elif(total_data_test_alpha1[i,2]=='1'):
                Yl = torch.tensor([[0,1]],dtype=torch.float)
            
            data_list.append(Data(x=x,edge_index=edge_index,y=Yl))
        
        for i in range(0,total_data_train_alpha2.shape[0]):
            
            pt_index = get_pos_type(total_data_train_alpha2[i,1],aa_index_train_alpha2[i],type_tabel,pos_tabel)
            x = torch.tensor(embedding_node(ma2_alpha1[pt_index,:,:],ma2_alpha2[pt_index,:,:],ma2_alpha1[pt_index,:,:]), dtype=torch.float)
        
            if (total_data_train_alpha2[i,2]=='0'):
                Yl = torch.tensor([[1,0]],dtype=torch.float)
            elif(total_data_train_alpha2[i,2]=='1'):
                Yl = torch.tensor([[0,1]],dtype=torch.float)
            
            data_list.append(Data(x=x,edge_index=edge_index,y=Yl))
        
        for i in range(0,total_data_test_alpha2.shape[0]):
            
            pt_index = get_pos_type(total_data_test_alpha2[i,1],aa_index_test_alpha2[i],type_tabel,pos_tabel)
            x = torch.tensor(embedding_node(ma2_alpha1[pt_index,:,:],ma2_alpha2[pt_index,:,:],ma2_alpha1[pt_index,:,:]), dtype=torch.float)
        
            if (total_data_test_alpha2[i,2]=='0'):
                Yl = torch.tensor([[1,0]],dtype=torch.float)
            elif(total_data_test_alpha2[i,2]=='1'):
                Yl = torch.tensor([[0,1]],dtype=torch.float)
            
            data_list.append(Data(x=x,edge_index=edge_index,y=Yl))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class dataset_topology_protbert_total_real (InMemoryDataset):
    def __init__(self, root,path,path2,cut=12, transform=None, pre_transform=None):
        self.path = path
        self.path2 = path2
        self.cut = cut
        super(dataset_topology_protbert_total_real, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):

        data_list = []
        edge_topology = []
        
        total_data_train_alpha1 = np.load(self.path+"alpha1_train.npy")
        total_data_test_alpha1 = np.load(self.path+"alpha1_test.npy")
        total_data_train_alpha2 = np.load(self.path+"alpha2_train.npy")
        total_data_test_alpha2 = np.load(self.path+"alpha2_test.npy")

        aa_index_train_alpha1  = (((((total_data_train_alpha1[:,0]).astype('int'))-13)/3)+5).reshape(-1,1)
        aa_index_test_alpha1  = (((((total_data_test_alpha1[:,0]).astype('int'))-13)/3)+5).reshape(-1,1)
        aa_index_train_alpha2  = (((((total_data_train_alpha2[:,0]).astype('int'))-13)/3)+5).reshape(-1,1)
        aa_index_test_alpha2  = (((((total_data_test_alpha2[:,0]).astype('int'))-13)/3)+5).reshape(-1,1)

        with open(self.path2+"ma1_alpha1_bert.pt", "rb") as fp: 
            ma1_alpha1 = pickle.load(fp)
        with open(self.path2+"ma1_alpha2_bert.pt", "rb") as fp: 
            ma1_alpha2 = pickle.load(fp)
        with open(self.path2+"ma2_alpha1_bert.pt", "rb") as fp:
            ma2_alpha1 = pickle.load(fp)
        with open(self.path2+"ma2_alpha2_bert.pt", "rb") as fp: 
            ma2_alpha2 = pickle.load(fp)

        pos_tabel = np.load(self.path2+"pos.npy").reshape(-1,1)
        type_tabel = np.load(self.path2+"label.npy")

        for i in range(0,total_data_train_alpha1.shape[0]):

            pt_index = get_pos_type(total_data_train_alpha1[i,1],aa_index_train_alpha1[i],type_tabel,pos_tabel)
            x = torch.tensor(embedding_node(ma1_alpha1[pt_index],ma1_alpha2[pt_index],ma1_alpha1[pt_index]), dtype=torch.float)
        
            if (total_data_train_alpha1[i,2]=='0'):
                Yl = torch.tensor([[1,0]],dtype=torch.float)
            elif(total_data_train_alpha1[i,2]=='1'):
                Yl = torch.tensor([[0,1]],dtype=torch.float)

            #if (aa_index_train_alpha1[i]==1 or aa_index_train_alpha1[i]>=335):
            #    continue
            edge = real_edge(aa_index_train_alpha1[i],cutoff=self.cut)
            edge_topology.append(edge)
            edge_index = torch.tensor(edge, dtype=torch.long)
            #print(x.size(dim=0))
            #print(np.max(edge))
            #print(np.max(edge)==(x.size(dim=0)-1))
            data_list.append(Data(x=x,edge_index=edge_index,y=Yl ))
        
        for i in range(0,total_data_test_alpha1.shape[0]):
            
            pt_index = get_pos_type(total_data_test_alpha1[i,1],aa_index_test_alpha1[i],type_tabel,pos_tabel)
            x = torch.tensor(embedding_node(ma1_alpha1[pt_index],ma1_alpha2[pt_index],ma1_alpha1[pt_index]), dtype=torch.float)
        
            if (total_data_test_alpha1[i,2]=='0'):
                Yl = torch.tensor([[1,0]],dtype=torch.float)
            elif(total_data_test_alpha1[i,2]=='1'):
                Yl = torch.tensor([[0,1]],dtype=torch.float)

            #if (aa_index_test_alpha1[i]==1 or aa_index_test_alpha1[i]>=335):
            #    continue
            edge = real_edge(aa_index_test_alpha1[i],cutoff=self.cut)
            edge_topology.append(edge)
            edge_index = torch.tensor(edge, dtype=torch.long)
            #print(x.size(dim=0))
            #print(np.max(edge)==(x.size(dim=0)-1))
            
            data_list.append(Data(x=x,edge_index=edge_index,y=Yl))
        
        for i in range(0,total_data_train_alpha2.shape[0]):
            
            pt_index = get_pos_type(total_data_train_alpha2[i,1],aa_index_train_alpha2[i],type_tabel,pos_tabel)
            x = torch.tensor(embedding_node(ma2_alpha1[pt_index],ma2_alpha2[pt_index],ma2_alpha1[pt_index]), dtype=torch.float)
        
            if (total_data_train_alpha2[i,2]=='0'):
                Yl = torch.tensor([[1,0]],dtype=torch.float)
            elif(total_data_train_alpha2[i,2]=='1'):
                Yl = torch.tensor([[0,1]],dtype=torch.float)
            
            #if (aa_index_train_alpha2[i]==1 or aa_index_train_alpha2[i]>=335):
            #    continue
        
            edge = real_edge(aa_index_train_alpha2[i],cutoff=self.cut)
            edge_topology.append(edge)
            edge_index = torch.tensor(edge, dtype=torch.long)
            #print(x.size(dim=0))
            #print(np.max(edge)==(x.size(dim=0)-1))

            data_list.append(Data(x=x,edge_index=edge_index,y=Yl))
        
        for i in range(0,total_data_test_alpha2.shape[0]):
            
            pt_index = get_pos_type(total_data_test_alpha2[i,1],aa_index_test_alpha2[i],type_tabel,pos_tabel)
            x = torch.tensor(embedding_node(ma2_alpha1[pt_index],ma2_alpha2[pt_index],ma2_alpha1[pt_index]), dtype=torch.float)

            if (total_data_test_alpha2[i,2]=='0'):
                Yl = torch.tensor([[1,0]],dtype=torch.float)
            elif(total_data_test_alpha2[i,2]=='1'):
                Yl = torch.tensor([[0,1]],dtype=torch.float)

            #if (aa_index_test_alpha2[i]==1 or aa_index_test_alpha2[i]>=335):
            #    continue

            edge = real_edge(aa_index_test_alpha2[i],cutoff=self.cut)
            edge_topology.append(edge)
            edge_index = torch.tensor(edge, dtype=torch.long)
            #print(x.size(dim=0))
            #print(np.max(edge)==(x.size(dim=0)-1))
            data_list.append(Data(x=x,edge_index=edge_index,y=Yl))
        
        with open(f"contact_map_{self.cut}.pt", "wb") as fp:   #Pickling
            pickle.dump(edge_topology, fp)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class dataset_a1_topology_protbert_pred (InMemoryDataset):
    def __init__(self, root,path,path2,mode="a1", transform=None, pre_transform=None):
        self.path = path
        self.path2 = path2
        self.mode = mode
        super(dataset_a1_topology_protbert_pred, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        
        total_data = np.load(self.path+"data15-23.npy")
        aa_index  = (((((total_data[:,0]).astype('int'))-13)/3)+5).reshape(-1,1)
        #ecoding_data_a1 = np.load(self.path+"_tokenzer_a1.npy")
        #ecoding_data_a2 = np.load(self.path+"_tokenzer_a2.npy")
        #ecoding_data_ref = np.load(self.path+"_tokenzer_ref.npy")
        with open(self.path2+"ma1_alpha1_bert.pt", "rb") as fp: 
            ma1_alpha1 = pickle.load(fp)
        with open(self.path2+"ma1_alpha2_bert.pt", "rb") as fp: 
            ma1_alpha2 = pickle.load(fp)
        with open(self.path2+"ma2_alpha1_bert.pt", "rb") as fp:
            ma2_alpha1 = pickle.load(fp)
        with open(self.path2+"ma2_alpha2_bert.pt", "rb") as fp: 
            ma2_alpha2 = pickle.load(fp)

        pos_tabel = np.load(self.path2+"pos.npy").reshape(-1,1)
        type_tabel = np.load(self.path2+"label.npy")

        for i in range(0,total_data.shape[0]):
            
            if (total_data[i,2]=="1"):
                pt_index = get_pos_type(total_data[i,1],aa_index[i],type_tabel,pos_tabel)
                x = torch.tensor(embedding_node(ma1_alpha1[pt_index],ma1_alpha2[pt_index],ma1_alpha1[pt_index]), dtype=torch.float)
            elif (total_data[i,2]=="2"):
                pt_index = get_pos_type(total_data[i,1],aa_index[i],type_tabel,pos_tabel)
                x = torch.tensor(embedding_node(ma2_alpha1[pt_index],ma2_alpha2[pt_index],ma2_alpha1[pt_index]), dtype=torch.float)
            
            edge = real_edge(aa_index[i],cutoff=12)
            edge_index = torch.tensor(edge, dtype=torch.long)

            data_list.append(Data(x=x,edge_index=edge_index,y=torch.tensor([[0,0]],dtype=torch.float) ))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class dataset_sel:
    def __init__(self,name):
        self.name = name

    def data_split(self,t_data,path):

        total_data_train_alpha1 = np.load(path+"alpha1_train.npy")
        total_data_test_alpha1 = np.load(path+"alpha1_test.npy")
        total_data_train_alpha2 = np.load(path+"alpha2_train.npy")
        total_data_test_alpha2 = np.load(path+"alpha2_test.npy")

        total_lab = []
        for i in t_data:
            if((i.y)[0][0]==1):
                total_lab.append(0)
            else:
                total_lab.append(1)
        total_lab = np.array(total_lab).astype('int')
        #print(np.where(total_lab==0)[0].shape)
        #print(np.where(total_lab==1)[0].shape)
        
        lethal_label = np.where(total_lab==0)[0]
        nonlethal_label = np.where(total_lab==1)[0]
        np.random.shuffle(lethal_label)
        np.random.shuffle(nonlethal_label)
        balance_0 = lethal_label[0:40]
        balance_1 = nonlethal_label[0:80]
        
        balance_test = np.sort(np.hstack((balance_0,balance_1)))
        balance_train = np.setdiff1d(np.array(range(0,len(t_data))),balance_test)
        
        #print(balance_train.shape)
        #print(balance_test.shape)

        train_dataset = t_data[balance_train]
        test_dataset = t_data[balance_test]

        t1_ = np.vstack((total_data_train_alpha1,total_data_test_alpha1))
        t2_ = np.vstack((t1_,total_data_train_alpha2))
        t3_ = np.vstack((t2_,total_data_test_alpha2))
        return train_dataset, test_dataset, [t3_,balance_train,balance_test]
    def data_split_ref(self,t_data,path,path2):

        total_data_train_alpha1 = np.load(path+"alpha1_train.npy")
        total_data_test_alpha1 = np.load(path+"alpha1_test.npy")
        total_data_train_alpha2 = np.load(path2+"alpha2_train.npy")
        total_data_test_alpha2 = np.load(path2+"alpha2_test.npy")

        train_2007 = list(range(0,207)) + list(range(301,487))
        test_2015 =list(range(207,301)) + list(range(487,587))

        train_dataset = t_data[train_2007]
        test_dataset = t_data[test_2015]

        t1_ = np.vstack((total_data_train_alpha1,total_data_test_alpha1))
        t2_ = np.vstack((t1_,total_data_train_alpha2))
        t3_ = np.vstack((t2_,total_data_test_alpha2))
        return train_dataset, test_dataset, [t3_,train_2007,test_2015]

    def dot(self,v1, v2):
        return sum(x*y for x, y in zip(v1, v2))
    def norm(self,vector):
        return math.sqrt(sum(x[0]**2 for x in vector))
    
    def get_info(self):
        return self.info
    
    def get_all_dataset(self):
        return self.data
    
    def load(self):
        if (self.name=='bert4_total'):
            total_dataset = dataset_topology_protbert_total('./total_topology_bert_total','./','./','./')
            train_dataset, test_dataset, data_info = self.data_split(total_dataset,'./','./')
            self.info = data_info
            self.data = total_dataset
            return train_dataset, test_dataset
        elif (self.name=='bert4_total_real'):
            total_dataset = dataset_topology_protbert_total_real('./total_topology_bert_total_real','./dataset/','./node_embedding/')
            train_dataset, test_dataset, data_info = self.data_split(total_dataset,'./dataset/')
            self.info = data_info
            self.data = total_dataset
            return train_dataset, test_dataset
        elif (self.name=='bert4_ref_real'):
            train_dataset_topology_bert = dataset_a1_topology_protbert_real('./training_a1_topology_bert_a1_real','./dataset/alpha1_train','./node_embedding/')
            test_dataset_topology_bert = dataset_a1_topology_protbert_real('./test_a1_topology_bert_a1_real','./dataset/alpha1_test','./node_embedding/')
            return train_dataset_topology_bert, test_dataset_topology_bert
        elif (self.name=='bert4_pred'):
            total_dataset = dataset_a1_topology_protbert_pred('./topology_bert_pred','./dataset/','./node_embedding/')
            return total_dataset
        



        
