import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (3.8,2)

def smooth(x):
    X_Y_Spline = make_interp_spline(range(27), x)
    Y_ = X_Y_Spline(np.linspace(0, 26,100))

    return np.linspace(0, 17,100),Y_


save_dir = './bert4_shuffle/'

amino_table = {"Ala":0,"Cys":1,"Asp":2,"Glu":3,"Arg":4,"Ser":5,"Val":6}
real_posl = np.array(["G12","P11","P10","G9","P8","P7","G6","P5","P4","G3","P2","P1","G0(mut)","P1'","P2'","G3'","P4'","P5'","G6'","P7'","P8'","G9'","P10'","P11'","G12'","P13'","P14'"])

a1_feature = np.load('Grad_CAM_total_ser.npy')
#a2_feature = np.load('a2.feature_a_total_ser.npy')/559

find_list_l = ["Lethal","Non-lethal","Lethal","Non-lethal"]
find_list_c = ["a1","a1","a2","a2"]

'''
21
55
5
54
tot = 135
'''
cax = plt.matshow(a1_feature[0,:,:].T/135,cmap="Reds", vmin=0, vmax=0.1)
plt.colorbar(cax)
plt.xticks(range(27),real_posl ,fontsize=12,rotation=90)
plt.yticks([0,1,2],['$ɑ_1$','$ɑ_{12}$','$ɑ_2$'] ,fontsize=12,rotation=0)
plt.tight_layout()
plt.savefig(f"../figure/total_ser_heatmap_l.png",dpi=300,bbox_inches="tight")
plt.close()

cax = plt.matshow(a1_feature[1,:,:].T/135,cmap="Reds", vmin=0, vmax=0.1)
plt.colorbar(cax)
plt.xticks(range(27),real_posl ,fontsize=12,rotation=90)
plt.yticks([0,1,2],['$ɑ_1$','$ɑ_{12}$','$ɑ_2$'] ,fontsize=12,rotation=0)
plt.tight_layout()
plt.savefig(f"../figure/total_ser_heatmap_nl.png",dpi=300,bbox_inches="tight")
plt.close()






