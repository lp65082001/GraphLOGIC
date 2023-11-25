import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (3.8,3.8)

def smooth(x):
    X_Y_Spline = make_interp_spline(range(27), x)
    Y_ = X_Y_Spline(np.linspace(0, 26,100))

    return np.linspace(0, 17,100),Y_


save_dir = '../bert4_shuffle/'

amino_table = {"Ala":0,"Cys":1,"Asp":2,"Glu":3,"Arg":4,"Ser":5,"Val":6}
real_posl = np.array(["G12","P11","P10","G9","P8","P7","G6","P5","P4","G3","P2","P1","G0(mut)","P1'","P2'","G3'","P4'","P5'","G6'","P7'","P8'","G9'","P10'","P11'","G12'","P13'","P14'"])

a1_feature = np.load('Grad_CAM_total.npy')

find_list_l = ["Lethal","Non-lethal","Lethal","Non-lethal"]
find_list_c = ["a1","a1","a2","a2"]

'''
129
147
a1 = 276
61
207
a2 = 268
'''
# plot total Grad-CAM (a1) # 
plt.plot((a1_feature[0,:,0]+a1_feature[0,:,1]+a1_feature[1,:,0]+a1_feature[1,:,1])/268, lw=1, color='lightcoral', linestyle='solid',label="$ɑ_1$")
#plt.plot(a1_feature[0,:,1], lw=1, color='cornflowerblue', linestyle='solid',label="Lethal ($ɑ_1$)")
plt.plot((a1_feature[0,:,2]+a1_feature[1,:,2])/276, lw=1, color='seagreen', linestyle='solid',label="$ɑ_2$")

plt.plot(12,0,"r*",label="mutation position")
plt.legend(edgecolor="black",fontsize=12)
plt.xticks(range(27),real_posl ,fontsize=12,rotation=90)
plt.ylabel("normalized quantity (276)",fontsize=12)
plt.ylim(0,1)
#plt.ylim(0,0.5)
plt.tight_layout()
plt.savefig(f"../figure/Grad_CAM_total_a1.png",dpi=300,bbox_inches="tight")
plt.close()

# plot total Grad-CAM (a2) # 
plt.plot((a1_feature[2,:,0]+a1_feature[2,:,1]+a1_feature[3,:,0]+a1_feature[3,:,1])/268, lw=1, color='lightcoral', linestyle='solid',label="$ɑ_1$")
#plt.plot(a1_feature[0,:,1], lw=1, color='cornflowerblue', linestyle='solid',label="Lethal ($ɑ_1$)")
plt.plot((a1_feature[2,:,2]+a1_feature[3,:,2])/268, lw=1, color='seagreen', linestyle='solid',label="$ɑ_2$")
plt.plot(12,0,"r*",label="mutation position")
plt.legend(edgecolor="black",fontsize=12)
plt.xticks(range(27),real_posl ,fontsize=12,rotation=90)
plt.ylabel("normalized quantity (268)",fontsize=12)
plt.ylim(0,1)
#plt.ylim(0,0.5)
plt.tight_layout()
plt.savefig(f"../figure/Grad_CAM_total_a2.png",dpi=300,bbox_inches="tight")




