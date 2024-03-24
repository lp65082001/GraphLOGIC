import pickle
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (4,4)

with open("./contact_map_12.pt", "rb") as fp:   # Unpickling
    b12 = pickle.load(fp)


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
pos_list = total_data_list[:,0].astype(int)
print(pos_list)
num_edge = []


num_edge_label = []

a12 = []
a12_l = []

num = 0
for i in b12:
    if i.shape[1]<=625:
        num_edge.append(pos_list[num])
        num_edge_label.append("edges < 625")
    elif i.shape[1]>625 and i.shape[1]<=675:
        num_edge.append(pos_list[num])
        num_edge_label.append("625 < edges < 675")
    elif i.shape[1]>675:
        num_edge.append(pos_list[num])
        num_edge_label.append("edges > 675")
    else:
        print(i.shape[1])
    a12.append(i.shape[1])
    a12_l.append('contact-12Å')
    num += 1

a12 = np.array(a12).reshape(-1,1)
a12_l = np.array(a12_l).reshape(-1,1)
l12 = np.hstack((a12_l,a12))

num_edge = np.array(num_edge).reshape(-1,1)
num_edge_label = np.array(num_edge_label).reshape(-1,1)
edge_list = np.hstack((num_edge,num_edge_label))

'''
df = pd.DataFrame(l12, columns = ["sample","edges"])

df["edges"] = df["edges"].astype(int)

sns.displot(df, x="edges", hue="sample", kind="kde", fill=True)

plt.savefig("./figure/edge_distribution.png",dpi=300,bbox_inches="tight")
plt.show()
'''

df = pd.DataFrame(edge_list, columns = ["position","edges"])

df["position"] = df["position"].astype(int)

sns.displot(df, x="position", hue="edges", multiple="stack")

plt.savefig("./figure/edge_position.png",dpi=300,bbox_inches="tight")
plt.show()

