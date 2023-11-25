from pysam import FastaFile
from transformers import BertModel, BertTokenizer,BertConfig
import re
import numpy as np
import os
import warnings
import torch
import pickle
warnings.filterwarnings("ignore")
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# load model #
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False )
model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.eval()

# read FASTA file #
sequences_object = FastaFile('./homo_sapiens.fasta')
alpha1 = sequences_object.fetch("NP_000079.2")
alpha2 = sequences_object.fetch("NP_000080.2")

#print(len(alpha1))
#print(len(alpha2))
#input()

def seq_fragment(x1,x2,seq,seq2,mode=1):
    xx = x1+3
    seq_list = []
    seq2_list = []
    s_label = []
    s_label2 = []
    num=0
    if (mode==1):
        for i in xx:
            seq_list.append([seq[0:i*3]+x2[num]+seq[i*3+1:len(seq)]])
            seq2_list.append([seq2[0:(i-1)*3]+'G'+seq2[(i-1)*3+1:len(seq2)]])
            s_label.append([i*3-12,i*3+15])
            if ((i-1)*3+15<=len(seq2)):
                if (i==4):
                    s_label2.append([(i-1)*3-9,(i-1)*3+15])
                else:
                    s_label2.append([(i-1)*3-12,(i-1)*3+15])
            else:
                s_label2.append([(i-1)*3-12,len(seq2)])
            num+=1
        return np.array(seq_list),np.array(seq2_list),np.array(s_label).reshape(-1,2),np.array(s_label2).reshape(-1,2)
    elif(mode==2):
        for i in xx:
            seq_list.append([seq[0:(i-1)*3]+x2[num]+seq[(i-1)*3+1:len(seq)]])
            seq2_list.append([seq2[0:i*3]+'G'+seq2[i*3+1:len(seq2)]])
            if ((i-1*3+15<=len(seq))):
                if (i==4):
                    s_label.append([(i-1)*3-9,(i-1)*3+15])
                else:
                    s_label.append([(i-1)*3-12,(i-1)*3+15])
            else:
                s_label.append([(i-1)*3-12,len(seq)])

            s_label2.append([i*3-12,i*3+15])

            num+=1
        return np.array(seq_list),np.array(seq2_list),np.array(s_label).reshape(-1,2),np.array(s_label2).reshape(-1,2)
    


def proteinbert(path,data,s_l):
    
    features = []
    num = 0
    for i in data:
        print(num)
        sequences_Example = []
        #print(i)
        ws = list(i[0])
        #print(ws)
        #sequences_Example.append(f"{ws[0]} {ws[1]} {ws[2]} {ws[3]} {ws[4]} {ws[5]} {ws[6]} {ws[7]} {ws[8]} {ws[9]} {ws[10]} {ws[11]} {ws[12]} {ws[13]} {ws[14]} {ws[15]} {ws[16]} {ws[17]} {ws[18]} {ws[19]} {ws[20]} {ws[21]} {ws[22]} {ws[23]} {ws[24]} {ws[25]} {ws[26]}")
        sequences_Example.append(" ".join(str(x) for x in ws))

        #sequences_Example = ["A E T C H A O", "T C A E H A O"]
        sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

        ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding = model(input_ids,attention_mask=attention_mask)[0]
        
        embedding = embedding.cpu().numpy()

        #embedding = np.asarray(embedding)
        #attention_mask = np.asarray(attention_mask)

        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len-1]
            features.append(seq_emd[s_l[num,0]:s_l[num,1],:])
        #num += 1
        print(s_l[num])
        print(features[-1].shape)
        num += 1

    #features_np = np.asarray(features)
    #print(features_np.shape)
    
    with open(path+"_bert.pt","wb") as fp:
        pickle.dump(features,fp)
    #np.save(path+"_bert.npy",features_np)




total_type = np.array(["A","C","D","E","R","S","V","W","P"])

alpha12_pos = []
alpha12_type = []
alpha12_type_n = []
for i in range(338):
    for j in range(9):
        alpha12_pos.append(i+1)
        alpha12_type.append(total_type[j])
        alpha12_type_n.append(j)
alpha12_pos = np.array(alpha12_pos).astype("int")
alpha12_type_n = np.array(alpha12_type_n).reshape(-1,1)


ma1_alpha1_seq, ma1_alpha2_seq, m1_alpha12_seq1,m1_alpha12_seq2 = seq_fragment(alpha12_pos,alpha12_type,alpha1,alpha2,mode=1)
ma2_alpha2_seq, ma2_alpha1_seq, m2_alpha12_seq2,m2_alpha12_seq1 = seq_fragment(alpha12_pos,alpha12_type,alpha2,alpha1,mode=2)


#print(m1_alpha12_seq2[0])
#print(m1_alpha12_seq2[20])
#input()


proteinbert("./ma1_alpha1",ma1_alpha1_seq,m1_alpha12_seq1)
proteinbert("./ma1_alpha2",ma1_alpha2_seq,m1_alpha12_seq2)
proteinbert("./ma2_alpha1",ma2_alpha1_seq,m2_alpha12_seq1)
proteinbert("./ma2_alpha2",ma2_alpha2_seq,m2_alpha12_seq2)
np.save("label.npy",alpha12_type_n)
np.save("pos.npy",alpha12_pos)
