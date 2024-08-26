import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (7,3)

def smooth(x):
    X_Y_Spline = make_interp_spline(range(27), x)
    Y_ = X_Y_Spline(np.linspace(0, 26,100))

    return np.linspace(0, 17,100),Y_


save_dir = '../bert4_final_d15'

amino_table = {"Ala":0,"Cys":1,"Asp":2,"Glu":3,"Arg":4,"Ser":5,"Val":6}
real_posl = np.array(["G12","P11","P10","G9","P8","P7","G6","P5","P4","G3","P2","P1","G0(mut)","P1'","P2'","G3'","P4'","P5'","G6'","P7'","P8'","G9'","P10'","P11'","G12'","P13'","P14'"])

'''
[ 55  56  57 225]
[['211' 'Arg' '0' '1.0']
 ['211' 'Cys' '1' '1.0']
 ['211' 'Ser' '1' '1.0']
 ['211' 'Ala' '1' '1.0']]
'''
a1_1 = (np.load(f"{save_dir}/cam_data/cam.{99}.tbert4-resid.npy")).astype('float')
a1_2 = (np.load(f"{save_dir}/cam_data/cam.{100}.tbert4-resid.npy")).astype('float')
a1_3 = (np.load(f"{save_dir}/cam_data/cam.{101}.tbert4-resid.npy")).astype('float')
a1_4 = (np.load(f"{save_dir}/cam_data/cam.{246}.tbert4-resid.npy")).astype('float')


plt.plot(a1_1[0,0:27,0],"o-",markersize=1.5,linewidth=1,label="Cys-Nonlethal")
plt.plot(a1_2[0,0:27,0],"v-",markersize=1.5,linewidth=1,label="Ser-Nonlethal")
plt.plot(a1_3[0,0:27,0],"^-",markersize=1.5,linewidth=1,label="Asp-Lethal")
plt.plot(a1_4[0,0:27,0],"*-",markersize=1.5,linewidth=1,label="Ala-Lethal")

plt.legend(edgecolor="black",fontsize=12)
plt.xticks(range(27),real_posl ,fontsize=12,rotation=90)
plt.ylabel("Grad-CAM",fontsize=12)
plt.tight_layout()
plt.savefig(f"../figure/total_415.png",dpi=300,bbox_inches="tight")
plt.close()
