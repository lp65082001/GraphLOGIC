import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (8,6)

def mutation_embedding3(x1):
    sequence_dict = {"Ala":"A" ,"Cys":"C","Asp":"D","Glu":"E","Pro":"P","Arg":"R","Ser":"S","Val":"V","Trp":"W"}
    sequence_list = np.array(["A" ,"C","D","E","R","S","V","W","P"])
    #amino_table[np.where(sequence_list == sequence_dict[x1])[0][0]] = 1
    return np.where(sequence_list == sequence_dict[x1])[0][0]

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

pred_label = np.array([1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1,
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1,
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0,
        1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
        1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1])
pred = np.load("./dataset/data15-23.npy")
pred_pos = pred[:,0].astype("int")
pred_realt = []
for i in pred[:,1]:
    pred_realt.append(mutation_embedding3(i))
pred_realt = np.array(pred_realt)
pred_chain = pred[:,2].astype("float")

a1_chain_l = np.where((pred_label==0) & (pred_chain==1.0))
a1_chain_nl = np.where((pred_label==1) & (pred_chain==1.0))
a2_chain_l = np.where((pred_label==0) & (pred_chain==2.0))
a2_chain_nl = np.where((pred_label==1) & (pred_chain==2.0))



plt.rcParams['figure.figsize']=(8,4)
#fig, (ax1, ax2) = plt.subplots(2,1)
#sample1_a1 ,sample1_a2,sample2_a1,sample2_a2 = self.cal_intersect(data1,data2,data3,data4,threshold

fig, ax = plt.subplots(1,2)

ax[0].scatter(pred_pos[a1_chain_l],pred_realt[a1_chain_l]+0.1,c="red",s=10,linewidth=1,edgecolor="black",label = "lethal")
ax[0].scatter(pred_pos[a1_chain_nl],pred_realt[a1_chain_nl]-0.1,c="blue",s=10,linewidth=1,edgecolor="black",label = "non-lethal")
ax[0].set_ylim(-0.5,8.5)

ax[1].scatter(pred_pos[a2_chain_l],pred_realt[a2_chain_l]+0.1,c="red",s=10,linewidth=1,edgecolor="black",label = "lethal")
ax[1].scatter(pred_pos[a2_chain_nl],pred_realt[a2_chain_nl]-0.1,c="blue",s=10,linewidth=1,edgecolor="black",label = "non-lethal")
ax[1].set_ylim(-0.5,8.5)

ax[0].legend(loc="upper right",edgecolor="black")
ax[1].legend(loc="upper right",edgecolor="black")
ax[0].set_xlabel("position")
ax[1].set_xlabel("position")
ax[0].set_ylabel("type")
ax[0].set_title("α1 sample (prediction)")
ax[1].set_title("α2 sample (prediction)")
#ax[1].set_ylabel("type")

plt.sca(ax[0])
plt.yticks(range(9), ["Ala","Cys","Asp","Glu","Arg","Ser","Val","Trp","Pro"])
plt.sca(ax[1])
plt.yticks(range(9), ["Ala","Cys","Asp","Glu","Arg","Ser","Val","Trp","Pro"])


ax[0].grid()
ax[1].grid()
plt.savefig("./dataset/prediction_15-23.png",dpi=300,bbox_inches="tight")
plt.show()
plt.close()






