import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (4,4)


ref_case = np.array([0.7872, 0.7766, 0.7979, 0.7766, 0.8085,
            0.7553, 0.7979, 0.7766, 0.7872, 0.8617])
total_case = np.array([0.6583, 0.8, 0.7417, 0.825, 0.7083,
            0.7833, 0.6417, 0.825, 0.8083, 0.8])

plt.bar([1],np.mean(ref_case),yerr=[np.std(ref_case)],color="blue",alpha=0.7,width=0.5,capsize=4,edgecolor="black")
plt.plot([1,1,1,1,1,1,1,1,1,1],ref_case,"b.",markeredgecolor="black")
plt.bar([2],np.mean(total_case),yerr=[np.std(total_case)],color="red",alpha=0.7,width=0.5,capsize=4,edgecolor="black")
plt.plot([2,2,2,2,2,2,2,2,2,2],ref_case,"r.",markeredgecolor="black")
plt.xlim(0,3)
plt.ylabel("accuracy",fontsize=14)
plt.xticks([1,2],[f"d07\n({round(np.mean(ref_case),2)})",f"d15\n({round(np.mean(total_case),2)})"],fontsize=14)
plt.title("cross-validation (10-fold)")
plt.show()