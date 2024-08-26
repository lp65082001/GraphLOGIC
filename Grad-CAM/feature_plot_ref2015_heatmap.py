import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (3,2)
real_posl = np.array(["G12","P11","P10","G9","P8","P7","G6","P5","P4","G3","P2","P1","G0(mut)","P1'","P2'","G3'","P4'","P5'","G6'","P7'","P8'","G9'","P10'","P11'","G12'","P13'","P14'"])

save_dir = '../bert4_final_d15/'
t='a1'


ser_l = np.array([[0,0,0,0,1,0,0,0,1,0,1,1,0,0,1,0,0,1,0,1,0,0,1,1,0,0,0]])
ser_nl = np.array([[0,0,1,0,1,0,0,0,1,0,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0]])
arg_l = np.array([[0,1,1,0,1,1,0,1,0,0,1,0,0,1,0,0,1,0,0,1,1,0,1,1,0,0,0]])
arg_nl = np.array([[0,1,1,0,1,1,0,1,0,0,1,0,0,1,1,0,1,0,0,1,1,0,1,1,0,0,0]])

# Ser l # 
cax = plt.matshow(ser_l,cmap="Reds", vmin=0, vmax=0.1)
plt.colorbar(cax)
plt.xticks(range(27),real_posl ,fontsize=12,rotation=90)
plt.xticks([])
plt.yticks([0],["$ɑ_{12}$"],fontsize=12)
plt.tight_layout()
plt.savefig(f"../figure/total_serref_heatmap_l.png",dpi=300,bbox_inches="tight")
plt.close()

# Ser nl # 
cax = plt.matshow(ser_nl,cmap="Reds", vmin=0, vmax=0.1)
plt.colorbar(cax)
plt.xticks(range(27),real_posl ,fontsize=12,rotation=90)
plt.xticks([])
plt.yticks([0],["$ɑ_{12}$"],fontsize=12)
plt.tight_layout()
plt.savefig(f"../figure/total_serref_heatmap_nl.png",dpi=300,bbox_inches="tight")
plt.close()


# Arg l # 
cax = plt.matshow(arg_l,cmap="Reds", vmin=0, vmax=0.1)
plt.colorbar(cax)
plt.xticks(range(27),real_posl ,fontsize=12,rotation=90)
plt.xticks([])
plt.yticks([0],["$ɑ_{12}$"],fontsize=12)
plt.tight_layout()
plt.savefig(f"../figure/total_argref_heatmap_l.png",dpi=300,bbox_inches="tight")
plt.close()

# Arg nl # 
cax = plt.matshow(arg_nl,cmap="Reds", vmin=0, vmax=0.1)
plt.colorbar(cax)
plt.xticks(range(27),real_posl ,fontsize=12,rotation=90)
plt.xticks([])
plt.yticks([0],["$ɑ_{12}$"],fontsize=12)
plt.tight_layout()
plt.savefig(f"../figure/total_argref_heatmap_nl.png",dpi=300,bbox_inches="tight")
plt.close()



