# coding: utf-8

import warnings
#from Bio import BiopythonDeprecationWarning
#warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings('ignore')
#from cam import SingleGCN_CAM_res, SingleGCN_CAM_gp, df_device
from dataset import dataset_sel
import torch
import numpy as np
from torch import nn
#from visualization import Plotter
from model import GAT_n_tot_only,GAT_n_tot
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix

# parameter setting (d15)#
result_type = "shuffle"
result_dataset = "test" #(or total)
dataset_name =  "bert4_total_real"
save_dir = './bert4_shuffle/'
model_arch = "GAT_n_tot"
t = "a2" # (a1 or a2)
'''
# parameter setting (d07)#
result_type = "control"
result_dataset = "test" #(or total)
dataset_name =  "bert4_ref_real"
save_dir = './bert4_control/'
model_arch = "GAT_n_tot_only"
t = "a1" # a1 only
'''
# define function #
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

def ref_model(p,t):
    pred = []
    for i in range(0,p.shape[0]):
        if (int(p[i])<=178):
            pred.append(1)
        else:
            if (t[i]=='Ala' or t[i]=='Ser'):
                pred.append(1)
            elif(t[i]=='Arg' or t[i]=='Val' or t[i]=='Glu' or t[i]=='Asp'):
                pred.append(0)
            elif(t[i]=='Cys'):
                if (int(p[i])<=688):
                    pred.append(1)
                else:
                    pred.append(0)
            else:
                print(t[i]) 
    return np.array(pred)

if __name__ == '__main__':
    if (result_type=="control"):
        total_data_train_alpha1 = np.load("./dataset/alpha1_train.npy")
        total_data_train_alpha1 = np.hstack((total_data_train_alpha1,np.ones((total_data_train_alpha1.shape[0],1))))
        total_data_test_alpha1 = np.load("./dataset/alpha1_test.npy")
        total_data_test_alpha1 = np.hstack((total_data_test_alpha1,np.ones((total_data_test_alpha1.shape[0],1))))
        total_data_list = np.vstack((total_data_train_alpha1,total_data_test_alpha1))
        data = dataset_sel(dataset_name)
        train_dataset, test_dataset = data.load()
        total_data = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        full_loader = DataLoader(total_data, len(total_data), shuffle=False)
        if (result_dataset=="test"):
            total_data_list = total_data_list[207:302,:]
            full_loader = DataLoader(test_dataset, len(test_dataset), shuffle=False)
        else:
            print("error for test")

    elif (result_type=="shuffle"):
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
        full_loader = DataLoader(total_data, len(total_data), shuffle=False)

        if (result_dataset=="test"):
            total_data_list = total_data_list[test_label,:]
            full_loader = DataLoader(test_dataset, len(test_dataset), shuffle=False)
        else:
            print("error for test")
    else:
        print("error for dataset")

    device = torch.device('cpu')
    if (model_arch=="GAT_n_tot_only"):
        model = GAT_n_tot_only(test_dataset,1)
        model.load_state_dict(torch.load(f'{save_dir}model{9}_dict.pt',
            map_location='cpu'))
    elif(model_arch=="GAT_n_tot"):
        model = GAT_n_tot(test_dataset,1)
        model.load_state_dict(torch.load(f'{save_dir}model{3}_dict.pt',
            map_location='cpu'))
    else:
        print("error for model")

    with torch.no_grad():
        model.eval()
        for data_ in full_loader:
            data = data_.to('cpu')
            out = model(data,data.edge_index)
            out = nn.Softmax(dim=1)(out)
            pred = out.argmax(dim=1)
            g_true = (data.y).argmax(dim=1)
            pred_label = pred==g_true
            tn, fp, fn, tp = confusion_matrix(g_true, pred).ravel()
    print(confusion_matrix(g_true, pred))
    print(f"accuracy: {(tp+tn)/(tp+fn+tn+fp)}")
    print(f"recall: {tp/(tp+fn)}")
    print(f"precision: {tp/(tp+fp)}")
    print(f"specificity: {tn/(fp+tn)}")
    print(f"f1-score: {2/(1/(tp/(tp+fn))+1/(tp/(tp+fp)))}")
    print("= = = = = = ")
    alpha1_amino_label_lc = np.array([0,0,0,0,0,0,0,0,0])
    alpha1_amino_label_nlc = np.array([0,0,0,0,0,0,0,0,0])
    alpha1_amino_label_lnc = np.array([0,0,0,0,0,0,0,0,0])
    alpha1_amino_label_nlnc = np.array([0,0,0,0,0,0,0,0,0])

    alpha2_amino_label_lc = np.array([0,0,0,0,0,0,0,0,0])
    alpha2_amino_label_nlc = np.array([0,0,0,0,0,0,0,0,0])
    alpha2_amino_label_lnc = np.array([0,0,0,0,0,0,0,0,0])
    alpha2_amino_label_nlnc = np.array([0,0,0,0,0,0,0,0,0])

    for i in range(total_data_list.shape[0]):
        a1,a2,a3,a4 = ecoding_amino_analysis(total_data_list[i,1],
        total_data_list[i,2],
        total_data_list[i,3],
        pred_label[i])
        
        
        if(a3==1 and a2==0 and a4==0):
            alpha1_amino_label_lc[a1] += 1
        elif(a3==1 and a2==1 and a4==0):
            alpha1_amino_label_nlc[a1] += 1
        elif(a3==1 and a2==0 and a4==1):
            alpha1_amino_label_lnc[a1] += 1
        elif(a3==1 and a2==1 and a4==1):
            alpha1_amino_label_nlnc[a1] += 1
        elif(a3==2 and a2==0 and a4==0):
            alpha2_amino_label_lc[a1] += 1
        elif(a3==2 and a2==1 and a4==0):
            alpha2_amino_label_nlc[a1] += 1
        elif(a3==2 and a2==0 and a4==1):
            alpha2_amino_label_lnc[a1] += 1
        elif(a3==2 and a2==1 and a4==1):
            alpha2_amino_label_nlnc[a1] += 1
        else:
            print("error")

    alpha1_all = alpha1_amino_label_lc+alpha1_amino_label_lnc+alpha1_amino_label_nlc+alpha1_amino_label_nlnc
    alpha2_all = alpha2_amino_label_lc+alpha2_amino_label_lnc+alpha2_amino_label_nlc+alpha2_amino_label_nlnc

    alpha1_lcor =  (alpha1_amino_label_lc*100)/(alpha1_amino_label_lc+alpha1_amino_label_lnc)
    alpha1_nlcor =  (alpha1_amino_label_nlc*100)/(alpha1_amino_label_nlc+alpha1_amino_label_nlnc)
    alpha1_lcor[np.isnan(alpha1_lcor)] = 0
    alpha1_nlcor[np.isnan(alpha1_nlcor)] = 0


    alpha1_ltotal = np.nansum(alpha1_amino_label_lc)
    alpha1_nltotal = np.nansum(alpha1_amino_label_nlc+alpha1_amino_label_nlnc)
    alpha2_lcor =  (alpha2_amino_label_lc*100)/(alpha2_amino_label_lc+alpha2_amino_label_lnc)
    alpha2_nlcor =  (alpha2_amino_label_nlc*100)/(alpha2_amino_label_nlc+alpha2_amino_label_nlnc)
    alpha2_lcor_ = alpha2_lcor[np.isnan(alpha2_lcor)] = 0
    alpha2_nlcor_ = alpha2_nlcor[np.isnan(alpha2_nlcor)] = 0
    alpha2_ltotal = np.nansum(alpha2_amino_label_lc+alpha2_amino_label_lnc)
    alpha2_nltotal = np.nansum(alpha2_amino_label_nlc+alpha2_amino_label_nlnc)
    amino_table = ["Ala","Cys","Asp","Glu","Arg","Ser","Val","Trp","Pro"]

    headers = ["Total","Lethal\n(correct)","Lethal\n(wrong)","Correct (\%)","Non-lethal\n(correct)","Non-lethal\n(wrong)","Correct (\%)"]
    data = dict()
    for i in range(len(amino_table)):
        if (t=='a1'):
            data[amino_table[i]] = [alpha1_all[i],alpha1_amino_label_lc[i],alpha1_amino_label_lnc[i],np.round(alpha1_lcor[i],2),alpha1_amino_label_nlc[i],alpha1_amino_label_nlnc[i],np.round(alpha1_nlcor[i],2)]
        else:
            data[amino_table[i]] = [alpha2_all[i],alpha2_amino_label_lc[i],alpha2_amino_label_lnc[i],np.round(alpha2_lcor[i],2),alpha2_amino_label_nlc[i],alpha2_amino_label_nlnc[i],np.round(alpha2_nlcor[i],2)]
    
    if (t=='a1'):
        data["Total"] = [np.nansum(alpha1_all),np.nansum(alpha1_amino_label_lc),np.nansum(alpha1_amino_label_lnc),np.round((np.nansum(alpha1_amino_label_lc)*100)/(np.nansum(alpha1_amino_label_lc)+np.nansum(alpha1_amino_label_lnc)),2),np.nansum(alpha1_amino_label_nlc),np.nansum(alpha1_amino_label_nlnc),np.round((np.nansum(alpha1_amino_label_nlc)*100)/(np.nansum(alpha1_amino_label_nlc)+np.nansum(alpha1_amino_label_nlnc)),2)]
    else:
        data["Total"] = [np.nansum(alpha2_all),np.nansum(alpha2_amino_label_lc),np.nansum(alpha2_amino_label_lnc),np.round((np.nansum(alpha2_amino_label_lc)*100)/(np.nansum(alpha2_amino_label_lc)+np.nansum(alpha2_amino_label_lnc)),2),np.nansum(alpha2_amino_label_nlc),np.nansum(alpha2_amino_label_nlnc),np.round((np.nansum(alpha2_amino_label_nlc)*100)/(np.nansum(alpha2_amino_label_nlc)+np.nansum(alpha2_amino_label_nlnc)),2)]
    textabular = f"l|{'c'*len(headers)}"
    texheader = " & " + " & ".join(headers) + "\\\\"
    texdata = "\\hline\n"
    for label in data:
        if label == "Total":
            texdata += "\\hline\n"
        texdata += f"{label} & {' & '.join(map(str,data[label]))} \\\\\n"   

    print('\documentclass{article}\n\\usepackage{array}\n\\newcolumntype{P}[1]{>{\centering\arraybackslash}p{#1}}\n\\usepackage[hmargin=1cm]{geometry}\n\\begin{document}\n\\begin{center}')
    if (t=='a1'):
        print("\\begin{tabular}{"+textabular+"}[{alpha1}]")
    else:
        print("\\begin{tabular}{"+textabular+"}[{alpha2}]")
    print(texheader)
    print(texdata,end="")
    print("\\end{tabular}\n\\end{center}\n\\end{document}")

#print("\n\n\n\n")
#test_index = np.where(total_data_list[:,3]=="1.0")[0]

#print(confusion_matrix(total_data_list[test_index,2].astype('int'),ref_model(total_data_list[test_index,0].astype("int"),total_data_list[test_index,1])))