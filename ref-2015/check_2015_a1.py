# coding: utf-8

import warnings
#from Bio import BiopythonDeprecationWarning
#warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from os import path
from sys import path as systemPath
#from cam import SingleGCN_CAM_res, SingleGCN_CAM_gp, df_device
#from dataset import dataset_sel
#import torch
import numpy as np
#from torch import nn
#from visualization import Plotter
#from model import GCN, GAT, GATs, GCNs, GCNs_vae
#from torch_geometric.loader import DataLoader
#from sklearn.metrics import precision_recall_curve, roc_curve,confusion_matrix,auc
from pysam import FastaFile
sequences_object = FastaFile('../reference_structure/homo_sapiens.fasta')
alpha1 = list(sequences_object.fetch("NP_000079.2"))
alpha2 = list(sequences_object.fetch("NP_000080.2"))

def real_sequence(pos):
    text_pos = int((pos-1)*3+12)
    text_pos2 = int((pos-1)*3+9)
    #print(text_pos)
    a11 = alpha1[text_pos-12:text_pos+15]
    if (pos==1):
        a2 = alpha2[text_pos2-9:text_pos2+15]
    elif(pos>=335):
        a2 = alpha2[text_pos2-12:1026]
    else:
        a2 = alpha2[text_pos2-12:text_pos2+15]
    return a11,a2

def batch_seq(seq):
    a1 = []
    a2 = []
    for i in seq:
        a1n,a2n = real_sequence(i)
        a1.append(a1n)
        a2.append(a2n)
    return np.array(a1).reshape(-1,27),np.array(a2).reshape(-1,27)

type_dict = {"P":0,"G":1,"A":1,"C":1,"S":1,"T":1,"H":2,"K":2,"R":2,"E":2,"D":2,"V":3,"I":3,"L":3,"M":3,"F":3,"Y":3,"W":3,"N":4,"Q":4}
sequence_dict = {"Ala":"A" ,"Phe":"F","Cys":"C","Sec":"U","Asp":"D","Asn":"N","Glu":"E","Gln":"Q","Gly":"G","Leu":"L","Ile":"I","His":"H","Pyl":"O","Met":"M","Pro":"P","Arg":"R","Ser":"S","Thr":"T","Val":"V","Trp":"W","Tyr":"Y","Lys":"K","Hyp":"P","Hse":"H"}

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

def seq2type(x):
    return type_dict[x]

def ser_tree(seq):
    #print(seq[8])
    if (seq2type(seq[8]) == 0 or seq2type(seq[8])==1 or seq2type(seq[8])==2):
        if (seq2type(seq[11]) == 0 or seq2type(seq[11])==1 or seq2type(seq[11])==2):
            return "1"
        elif (seq2type(seq[11]) == 3 or seq2type(seq[11])==4):
            if (seq2type(seq[10]) == 2):
                return "0"
            elif (seq2type(seq[10]) == 0 or seq2type(seq[10]) == 1 or seq2type(seq[10]) == 3):
                if (seq2type(seq[14]) == 2):
                    return "0"
                elif (seq2type(seq[14]) == 0 or seq2type(seq[14]) == 1 or seq2type(seq[14]) == 3):
                    return "1"
    elif(seq2type(seq[8]) == 3 or seq2type(seq[8])==4):
        if (seq2type(seq[11]) == 2 or seq2type(seq[11])==4):
            return "1"
        elif (seq2type(seq[11]) == 0 or seq2type(seq[11])==1):
           if (seq2type(seq[14])==1):
               return "1"
           elif (seq2type(seq[14])==0 or seq2type(seq[14])==2):
               return "0"
    print("out of cases")
def cys_tree(seq):
    if (seq2type(seq[17]) == 1 or seq2type(seq[17])==3):
        if (seq2type(seq[11]) == 1 or seq2type(seq[11])==4):
            return "1"
        elif(seq2type(seq[11]) == 0 or seq2type(seq[11])==2 or seq2type(seq[11])==3):
            return "0"
    elif(seq2type(seq[17]) == 0 or seq2type(seq[17])==2 or seq2type(seq[17])==4):
        if (seq2type(seq[8]) == 2 or seq2type(seq[8]) == 3 or seq2type(seq[8]) == 4):
            return "1"
        elif (seq2type(seq[8]) == 0 or seq2type(seq[8]) == 1 ):
            if (seq2type(seq[16]) == 3):
                return "1"
            elif(seq2type(seq[16]) == 0 or seq2type(seq[16]) == 1 or seq2type(seq[16]) == 2 or seq2type(seq[16]) == 4):
                if (seq2type(seq[17])==0):
                    return "1"
                elif(seq2type(seq[17])==2 or seq2type(seq[17])==4):
                    return "0"
    print("out of cases")
def arg_tree(seq):
    if (seq2type(seq[10]) == 0 or seq2type(seq[10])==2 or seq2type(seq[10])==4):
        if(seq2type(seq[13])==0 or seq2type(seq[13])==4):
            return "0"
        elif(seq2type(seq[13])==1 or seq2type(seq[13])==2 or seq2type(seq[13])==3):
            if (seq2type(seq[14])==0 or seq2type(seq[14])==1):
                return "1"
            elif(seq2type(seq[14])==2 or seq2type(seq[14])==14):
                return "0"
        print("up")
    elif(seq2type(seq[10]) == 1 or seq2type(seq[10])==3):
        if (seq2type(seq[16]) == 1):
            return "0"
        elif(seq2type(seq[16]) == 0 or seq2type(seq[16]) == 2 or seq2type(seq[16]) == 3):
            if(seq2type(seq[13])==0 or seq2type(seq[13])==1 or seq2type(seq[13])==4):
                return "1"
            elif (seq2type(seq[13])==3):  
                return "0"  
        print("low")
    print("out of cases")



def cal_amino(t_d,t_d_f,type="ser"):
    lcorr = 0
    lncorr = 0
    nlcorr = 0
    nlncorr = 0
    for i in range(0,t_d.shape[0]):
        seq = t_d_f[i]
        if (type=="ser"):
            pred = ser_tree(seq)
        elif (type=="cys"):
            pred = cys_tree(seq)
        elif (type=="arg"):
            pred = arg_tree(seq)
        else:
            print("error")
        
        if (t_d[i,2] == pred and t_d[i,2]=='0'):
            lcorr += 1
        elif (t_d[i,2] == pred and t_d[i,2]=='1'):
            nlcorr += 1
        elif (t_d[i,2] != pred and t_d[i,2]=='0'):
            lncorr += 1
        elif (t_d[i,2] != pred and t_d[i,2]=='1'):
            nlncorr += 1

    return lcorr, lncorr, nlcorr, nlncorr


if __name__ == '__main__':

    total_data_train_alpha1 = np.load("../dataset/alpha1_train.npy")
    total_data_test_alpha1 = np.load("../dataset/alpha1_test.npy")
    total_alpha1 = np.vstack((total_data_train_alpha1,total_data_test_alpha1))
    '''
    total_data_train_frag_alpha1 = np.load("../alpha1_data/alpha1_train_fragment_a1.npy")
    total_data_test_frag_alpha1 = np.load("../alpha1_data/alpha1_test_fragment_a1.npy")
    total_alpha1_frag = np.vstack((total_data_train_frag_alpha1,total_data_test_frag_alpha1))
    '''

    ser_sample = total_alpha1[np.where(total_alpha1[:,1]=='Ser')[0],:]
    cys_sample = total_alpha1[np.where(total_alpha1[:,1]=='Cys')[0],:]
    arg_sample = total_alpha1[np.where(total_alpha1[:,1]=='Arg')[0],:]

    ser_sample_pos = (((((ser_sample[:,0]).astype('int'))-13)/3)+5).reshape(-1,1)
    cys_sample_pos = (((((cys_sample[:,0]).astype('int'))-13)/3)+5).reshape(-1,1)
    arg_sample_pos = (((((arg_sample[:,0]).astype('int'))-13)/3)+5).reshape(-1,1)

    ser_frag = (batch_seq(ser_sample_pos)[0])
    cys_frag = (batch_seq(cys_sample_pos)[0])
    arg_frag = (batch_seq(arg_sample_pos)[0])

    print("ser")
    print(cal_amino(ser_sample,ser_frag))
    print("cys")
    print(cal_amino(cys_sample,cys_frag,type="cys"))
    print("arg")
    print(cal_amino(arg_sample,arg_frag,type="arg"))

    #ser_sample_test = np.where(total_data_test_alpha1[:,1]=='Ser')[0]
    #cys_sample_test = np.where(total_data_test_alpha1[:,1]=='Cys')[0]
    #arg_sample_test = np.where(total_data_test_alpha1[:,1]=='Arg')[0]

    #print(cal_amino(total_data_test_alpha1,total_data_test_frag_alpha1 ,ser_sample_test))
    #print(cal_amino(total_data_test_alpha1,total_data_test_frag_alpha1 ,cys_sample_test,type="cys"))
    #print(cal_amino(total_data_test_alpha1,total_data_test_frag_alpha1 ,arg_sample_test,type="arg"))







