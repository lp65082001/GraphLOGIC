import shutil
import torch 
from torch import nn
import numpy as np 
import pickle
from argparse import ArgumentParser
from model import GAT_n_tot
from dataset import dataset_sel
from scipy.spatial import distance_matrix
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# label #
lethal_label = ["Lethal", "Non-Lethal"]

# load graph #
real_structure_a11 = np.load("./reference_structure/alpha1_1.npy")
real_structure_a2 = np.load("./reference_structure/alpha2.npy")
real_structure_a12 = np.load("./reference_structure/alpha1_2.npy")

# load embedding #
with open("./node_embedding/ma1_alpha1_bert.pt", "rb") as fp: 
    ma1_alpha1 = pickle.load(fp)
with open("./node_embedding/ma1_alpha2_bert.pt", "rb") as fp: 
    ma1_alpha2 = pickle.load(fp)
with open("./node_embedding/ma2_alpha1_bert.pt", "rb") as fp:
    ma2_alpha1 = pickle.load(fp)
with open("./node_embedding/ma2_alpha2_bert.pt", "rb") as fp: 
        ma2_alpha2 = pickle.load(fp)

# load pos and label #
pos_tabel = np.load("./node_embedding/pos.npy").reshape(-1,1)
type_tabel = np.load("./node_embedding/label.npy")

# get edge index #
def real_edge(pos,cutoff=12):
    text_pos = int((pos-1)*3+12)
    text_pos2 = int((pos-1)*3+9)
    #print(text_pos)
    a11 = real_structure_a11[text_pos-12:text_pos+15,:]
    a12 = real_structure_a12[text_pos-12:text_pos+15,:]
    if (pos==1):
        a2 = real_structure_a2[text_pos2-9:text_pos2+15,:]
    elif(pos>=335):
        a2 = real_structure_a2[text_pos2-12:1026,:]
    else:
        a2 = real_structure_a2[text_pos2-12:text_pos2+15,:]
    p_ = np.vstack((a11,a2))
    tot_p_ = np.vstack((p_,a12))
    dm = distance_matrix(tot_p_,tot_p_)
    result = np.where((np.triu(dm,1)<=cutoff) & (np.triu(dm,1)!=0))

    return np.array(np.vstack((result[0].T,result[1].T)))

# get embedding node #
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

# get mutation type #
def get_pos_type(t,p,tt,pt):
    tn = mutation_embedding3(t)
    tn_s = np.where(tt==tn)[0]
    pt_s = np.where(pt==p)[0]
    return np.intersect1d(tn_s,pt_s)[0]

# mutation_embeddin #
def mutation_embedding3(x1):
    sequence_dict = {"Ala":"A" ,"Cys":"C","Asp":"D","Glu":"E","Pro":"P","Arg":"R","Ser":"S","Val":"V","Trp":"W"}
    sequence_list = np.array(["A" ,"C","D","E","R","S","V","W","P"])
    #amino_table[np.where(sequence_list == sequence_dict[x1])[0][0]] = 1
    return np.where(sequence_list == sequence_dict[x1])[0][0]

# data format #
class predict_data(InMemoryDataset):
    def __init__(self, pos, mutation, chain, root="./predict_data", transform = None, pre_transform = None):
        self.pos = pos
        self.mutation = mutation
        self.chain = chain
        super(predict_data, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):

        triplet_num = int(((((int(args.pos))-13)/3)+5))

        edge = real_edge(triplet_num,cutoff=12)
        edge_index = torch.tensor(edge, dtype=torch.long)

        if args.chn=="a1":
            pt_index = get_pos_type(args.mut,triplet_num,type_tabel,pos_tabel)
            x = torch.tensor(embedding_node(ma1_alpha1[pt_index],ma1_alpha2[pt_index],ma1_alpha1[pt_index]), dtype=torch.float)
        elif args.chn=="a2":
            pt_index = get_pos_type(args.mut,triplet_num,type_tabel,pos_tabel)
            x = torch.tensor(embedding_node(ma2_alpha1[pt_index],ma2_alpha2[pt_index],ma2_alpha1[pt_index]), dtype=torch.float)
        else:
            print("Please input correct argument for chain")

        data_list = []
        data_list.append(Data(x=x,edge_index=edge_index))
        self.dataset = data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    def get_dataset(self):
        return self.dataset

# model path #
save_dir = "./bert4_final_d15/"

# load dataset (only or model initialization) #
data = dataset_sel("bert4_total_real")
train_dataset, test_dataset = data.load()

# initial model #
model = GAT_n_tot(test_dataset,1)
model.load_state_dict(torch.load(f'{save_dir}model{3}_dict.pt',
            map_location='cpu'))


# load input argument #
parser = ArgumentParser()
parser.add_argument("-p",dest="pos", help="where position?")
parser.add_argument("-m",dest="mut", help="what substitute type? (Ser, Arg ...)")
parser.add_argument("-c",dest="chn", help="which chains? (a1,a2)")
args = parser.parse_args()

# data process #
data = predict_data(args.pos, args.mut, args.chn)
full_loader = DataLoader(data, len(test_dataset), shuffle=False)

# prediction #
with torch.no_grad():
    model.eval()
    for data_ in full_loader:
        out = model(data_ ,data_ .edge_index)
        out = nn.Softmax(dim=1)(out)
        pred = out.argmax(dim=1)

print(f"Predict result: {lethal_label[pred]}")
shutil.rmtree('predict_data')
