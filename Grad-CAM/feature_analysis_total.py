import numpy as np
import pandas as pd
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

def check_pos(pos,old_pos):
    if (pos==1):
        shift = 3
    elif(pos>=335):
         shift = 0 
    else:
        shift = 0

    return old_pos+shift
type_dict = {"P":0,"G":1,"A":1,"C":1,"S":1,"T":1,"H":2,"K":2,"R":2,"E":2,"D":2,"V":3,"I":3,"L":3,"M":3,"F":3,"Y":3,"W":3,"N":4,"Q":4}
real_posl = np.array([1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26])
save_dir = '../bert4_shuffle'

total_data_train_alpha1 = np.load("../dataset/alpha1_train.npy")
total_data_train_alpha1 = np.hstack((total_data_train_alpha1,np.ones((total_data_train_alpha1.shape[0],1))))
total_data_test_alpha1 = np.load("../dataset/alpha1_test.npy")
total_data_test_alpha1 = np.hstack((total_data_test_alpha1,np.ones((total_data_test_alpha1.shape[0],1))))
total_data_train_alpha2 = np.load("../dataset/alpha2_train.npy")
total_data_train_alpha2 = np.hstack((total_data_train_alpha2,np.ones((total_data_train_alpha2.shape[0],1))*2))
total_data_test_alpha2 = np.load("../dataset/alpha2_test.npy")
total_data_test_alpha2 = np.hstack((total_data_test_alpha2,np.ones((total_data_test_alpha2.shape[0],1))*2))
t1_ = np.vstack((total_data_train_alpha1,total_data_test_alpha1))
t2_ = np.vstack((t1_,total_data_train_alpha2))
total_data_list = np.vstack((t2_,total_data_test_alpha2))

pred_label = np.load(f"{save_dir}/pred_total.npy")

#find_list_a = ["Ala","Cys","Asp","Glu","Arg","Ser","Val","Ala","Cys","Asp","Glu","Arg","Ser","Val","Ala","Cys","Asp","Glu","Arg","Ser","Val","Ala","Cys","Asp","Glu","Arg","Ser","Val"]
#find_list_l = ["0","0","0","0","0","0","0","1","1","1","1","1","1","1","0","0","0","0","0","0","0","1","1","1","1","1","1","1"]
#find_list_c = ["1.0","1.0","1.0","1.0","1.0","1.0","1.0","1.0","1.0","1.0","1.0","1.0","1.0","1.0","2.0","2.0","2.0","2.0","2.0","2.0","2.0","2.0","2.0","2.0","2.0","2.0","2.0","2.0"]
find_list_l = ["0","1","0","1"]
find_list_c = ["1.0","1.0","2.0","2.0"]


total_case = []

for i in range(0,4):
    t_case = np.zeros((27,3))

    index = np.where((total_data_list[:,2]==find_list_l[i]) & (total_data_list[:,3]==find_list_c[i]) & (pred_label==True))[0]
    if (len(index)!=0):
        for j in index:
            data = np.load(save_dir+f'/cam_data/cam.{j}.tbert4-resid.npy')[0,:,0]
            #f1_neighbor = np.array(list(f1_total[j][0]))
            #f2_neighbor = np.array(list(f2_total[j][0]))
            if(data.shape[0]!=81):
                data_a1 = data[0:27]
                data_a12 = data[data.shape[0]-27:data.shape[0]]
                data_a2 = data[27:data.shape[0]-27]
            else:
                data_a1 = data[0:27]
                data_a12 = data[54:81]
                data_a2 = data[27:54]

            a1_index = np.where(data_a1!=0)[0]
            a12_index = np.where(data_a12!=0)[0]
            a2_index = np.where(data_a2!=0)[0]

            a2_index = check_pos(int(((int(total_data_list[j,0])-13)/3)+5),a2_index)

            for a1i in a1_index:
                t_case[a1i,0] += 1
            for a12i in a12_index:
                t_case[a12i,1] += 1
            for a2i in a2_index:
                t_case[a2i,2] += 1

            #input()

        print(len(index))
    else:
        pass

    total_case.append(t_case)

total_case = np.array(total_case)


np.save('./Grad_CAM.feature_total.npy',total_case)

