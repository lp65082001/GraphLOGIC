import gc
import os
import torch
import random
import numpy as np
from torch import nn
from dataset import dataset_sel
from prettytable import PrettyTable
from torch_geometric.utils import dropout
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader
from model import GAT_n_tot,GAT_n_tot_only
from sklearn.model_selection import StratifiedKFold

df_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameter setting (shuffle setting)#

dataset_name = "bert4_total_real"
save_path = "./bert4_shuffle/"
dropedge_ = True
graph_layer = "GAT_n_tot"
nlayer = 1
split_seed = 3
edge_drop_ratio = 0.5
epoch_ = 500
lr_= 0.001
tolerance_=5
min_delta_=0.2

'''
# hyperparameter setting (control setting)#

dataset_name = "bert4_ref_real"
save_path = "./bert4_control/"
dropedge_ = True
graph_layer = "GAT_n_tot_only"
nlayer = 1
split_seed = 9
edge_drop_ratio = 0.5
epoch_ = 500
lr_= 0.0005
tolerance_=20
min_delta_=0.1
'''

# define function #

class EarlyStopping: # early stop
    def __init__(self, tolerance, min_delta):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

def count_parameters(model): # model imformation
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def train(loader):
    model.train()
    tot_loss = 0
    num=0
    for data_ in loader:
        data = data_.to(df_device)

        if(dropedge_==True):
            edge_index = dropout.dropout_edge(data.edge_index,p=edge_drop_ratio,training=True)[0]
        elif(dropedge_==False):
            edge_index = data.edge_index
        out_ = model(data,edge_index)
        out = nn.Softmax(dim=1)(out_)
        optimizer.zero_grad()
        loss = criterion(out,data.y)
        loss.backward()
        optimizer.step()
                
        tot_loss += loss.item()
        num+=1

    return tot_loss/num

def test(loader):
    correct = 0
    tot_loss = 0
    num = 0
    with torch.no_grad():
        model.eval()
        for data_ in loader:
            data = data_.to(df_device)
            out_ = model(data,data.edge_index)
            out = nn.Softmax(dim=1)(out_)
            pred = out.argmax(dim=1)
            g_true = (data.y).argmax(dim=1)
            correct +=int((pred==g_true).sum())
            loss = criterion(out,data.y)
            tot_loss += loss.item()
            num += 1
    return correct/len(loader.dataset), tot_loss/num

# data processing # 
data = dataset_sel(dataset_name)
train_dataset, test_dataset = data.load()


if not os.path.isdir(save_path): # build save folder
    os.mkdir(save_path)

kf = StratifiedKFold(n_splits=10,shuffle=True,random_state=0)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
full_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
total_ll = []
for i in train_dataset:
    if((i.y)[0][0]==1):
        total_ll.append(0)
    else:
        total_ll.append(1)
total_ll = np.array(total_ll)

num_kfold = 0
train_av = []
valid_av = []
test_av = []
test_recall = []
test_precision = []
test_spec = []
test_f1 = []

f = open(save_path+'result.txt', 'w')

for train_index, valid_index in kf.split(train_dataset,total_ll): # run with ratio
    if (num_kfold!=split_seed):
        num_kfold +=1
        continue

    torch.manual_seed(12345) #Repeatability

    train_loader = DataLoader(train_dataset[train_index], batch_size=16, shuffle=True) #data loader
    valid_loader = DataLoader(train_dataset[valid_index], batch_size=32, shuffle=False)

    # calculate lethal and non-lethal label
    total_lab = []
    for i in train_dataset[train_index]:
        if((i.y)[0][0]==1):
            total_lab.append(0)
        else:
            total_lab.append(1)
    total_lab = np.array(total_lab)

    # model setting #      
    if(graph_layer=="GAT_n_tot"):
        model = GAT_n_tot(train_dataset,nlayer)
    elif(graph_layer=="GAT_n_tot_only"):
        model = GAT_n_tot_only(train_dataset,nlayer)
    model.to(df_device)

    balance_w = torch.tensor([(len(np.where(total_lab==1)[0])/len(np.where(total_lab==0)[0])),(len(np.where(total_lab==1)[0])/len(np.where(total_lab==1)[0]))])
    balance_w.to(df_device)
    # optimizer and loss-function # 
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr_,weight_decay=0.0005)
    criterion = torch.nn.CrossEntropyLoss(weight=balance_w.to(df_device))
            
    count_parameters(model) # print out model framework

    # training #

    best_model = 5
    best_ep = 0
    early_ep = 0
    early_stopping = EarlyStopping(tolerance=tolerance_, min_delta=min_delta_)
            
    for epoch in range(1,epoch_):
        t_loss = train(train_loader)
        train_acc,train_loss_ = test(train_loader)
        valid_acc, valid_loss_ = test(valid_loader)
        test_acc, test_loss_ = test(test_loader)
        early_stopping(train_loss_, valid_loss_)
        if early_stopping.early_stop:
            #print("We are at epoch:", epoch)
            #break
            pass
        else:
            if valid_loss_ < best_model:
                torch.save(model, f'{save_path}model{num_kfold}.pt')
                torch.save(model.state_dict(), f'{save_path}model{num_kfold}_dict.pt')
                best_model = valid_loss_
                best_ep = epoch
            early_ep = epoch+1

        f.write(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {t_loss:.4f}, Valid Acc: {valid_acc:.4f}, Valid Loss: {valid_loss_:.4f}, Test Acc: {test_acc:.4f}, Test Loss: {test_loss_:.4f}\n')
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {t_loss:.4f}, Valid Acc: {valid_acc:.4f}, Valid Loss: {valid_loss_:.4f}, Test Acc: {test_acc:.4f}, Test Loss: {test_loss_:.4f}')
                
    print(f"{best_ep:.4f}")
    print(f"{best_model:.4f}")
    print(f"{early_ep:.4f}")
    model_pre = torch.load(f'{save_path}model{num_kfold}.pt')
    model_pre.to('cpu')
            
    correct = 0
    with torch.no_grad():
        model.eval()
        for data_ in full_loader:
            data = data_.to('cpu')
            out = model_pre(data,data.edge_index)
            out = nn.Softmax(dim=1)(out)
            pred = out.argmax(dim=1)

            g_true = (data.y).argmax(dim=1)
            correct +=int((pred==g_true).sum())
        train_av.append(correct/len(full_loader.dataset))
    correct = 0
    with torch.no_grad():
        model.eval()   
        for data_ in valid_loader:
            data = data_.to('cpu')
            out = model_pre(data,data.edge_index)
            out = nn.Softmax(dim=1)(out)
            pred = out.argmax(dim=1)
            g_true = (data.y).argmax(dim=1)
            correct +=int((pred==g_true).sum())
        valid_av.append(correct/len(valid_loader.dataset))
    correct = 0
    with torch.no_grad():
        model.eval()
        for data_ in test_loader:
            data = data_.to('cpu')
            out = model_pre(data,data.edge_index)
            out = nn.Softmax(dim=1)(out)
            pred = out.argmax(dim=1)
            g_true = (data.y).argmax(dim=1)
            correct +=int((pred==g_true).sum())
            pred_ = pred.cpu()
            g_true_ = g_true.cpu()
            tn, fp, fn, tp = confusion_matrix(pred_, g_true_).ravel()
        test_av.append(correct/len(test_loader.dataset))
        test_recall.append(tp/(tp+fn))
        test_precision.append(tp/(tp+fp))
        test_spec.append(tn/(fp+tn))
        test_f1.append(2/(1/(tp/(tp+fn))+1/(tp/(tp+fp))))
            
    del model
    del model_pre
    torch.cuda.empty_cache()
    num_kfold +=1

train_av = np.array(train_av)
valid_av = np.array(valid_av)
test_av = np.array(test_av)

test_recall = np.array(test_recall)
test_precision = np.array(test_precision)
test_spec = np.array(test_spec)
test_f1 = np.array(test_f1)

train_av_mean = np.mean(train_av)
train_av_std = np.std(train_av)
valid_av_mean = np.mean(valid_av)
valid_av_std = np.std(valid_av)
test_av_mean = np.mean(test_av)
test_av_std = np.std(test_av)

test_recall_mean = np.mean(test_recall)
test_recall_std = np.std(test_recall)
test_precision_mean = np.mean(test_precision)
test_precision_std = np.std(test_precision)
test_spec_mean = np.mean(test_spec)
test_spec_std = np.std(test_spec)
test_f1_mean = np.mean(test_f1)
test_f1_std = np.std(test_f1)
   
# output results #    
f.write(f'train mean {train_av_mean:.4f}, valid mean {valid_av_mean:.4f}, test mean {test_av_mean:.4f}\n')
f.write(f'train: {train_av[0]:.4f}\n')
f.write(f'valid: {valid_av[0]:.4f}\n')
f.write(f'test: {test_av[0]:.4f}\n')
f.write(f'recall mean {test_recall_mean:.4f}, recall std {test_recall_std:.4f}, precision mean {test_precision_mean:.4f}, precision std {test_precision_std:.4f}, specificity mean {test_spec_mean:.4f}, specificity std {test_spec_std:.4f}, f1 mean {test_f1_mean:.4f}, f1 std {test_f1_std:.4f}\n')
f.write(f'test_recall: {test_recall[0]:.4f}\n')
f.write(f'test_precision: {test_precision[0]:.4f}\n')
f.write(f'test_specificity: {test_spec[0]:.4f}\n')
f.write(f'test_f1: {test_f1[0]:.4f}\n')
        
        
print(f'train mean {train_av_mean:.4f}, valid mean {valid_av_mean:.4f}, test mean {test_av_mean:.4f}')
print(f'train: {train_av[0]:.4f}')
print(f'valid: {valid_av[0]:.4f}')
print(f'test: {test_av[0]:.4f}')
print(f'test_recall: {test_recall[0]:.4f}')
print(f'test_precision: {test_precision[0]:.4f}')
print(f'test_specificity: {test_spec[0]:.4f}')

f.close()