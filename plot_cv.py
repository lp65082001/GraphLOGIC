import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (3,3)


ref_case = np.array([0.7872, 0.7766, 0.7979, 0.7766, 0.8085,
            0.7553, 0.7979, 0.7766, 0.7872, 0.8617])
total_case = np.array([0.6583, 0.8, 0.7417, 0.825, 0.7083,
            0.7833, 0.6417, 0.825, 0.8083, 0.8])

ref_c = np.array([0.667,  0.8095, 0.8095, 0.619, 0.9048,
                  0.7143, 0.8571,0.70, 0.75, 0.85])

print(np.mean(ref_case))
print(np.mean(total_case))
print(np.mean(ref_c))

plt.bar([1],np.mean(ref_case),yerr=[np.std(ref_case)],color="blue",alpha=0.7,width=0.5,capsize=4,edgecolor="black")
plt.plot([1,1,1,1,1,1,1,1,1,1],ref_case,"b.",markeredgecolor="black")
plt.bar([2],np.mean(total_case),yerr=[np.std(total_case)],color="red",alpha=0.7,width=0.5,capsize=4,edgecolor="black")
plt.plot([2,2,2,2,2,2,2,2,2,2],ref_case,"r.",markeredgecolor="black")
plt.xlim(0,3)
plt.ylabel("accuracy",fontsize=14)
plt.xticks([1,2],[f"d07",f"d15"],fontsize=12,rotation=30)
plt.title("testing (10-fold)",fontsize=14)
plt.savefig("./figure/cv_test.png",dpi=300,bbox_inches="tight")
#plt.show()
plt.close()

plt.bar([1],np.mean(ref_c),yerr=[np.std(ref_c)],color="blue",alpha=0.7,width=0.5,capsize=4,edgecolor="black")
plt.plot([1,1,1,1,1,1,1,1,1,1],ref_case,"b.",markeredgecolor="black")
plt.bar([2],0.73,color="green",alpha=0.7,width=0.5,capsize=4,edgecolor="black")

plt.xlim(0,3)
plt.ylabel("accuracy",fontsize=14)
plt.xticks([1,2],[f"GraphLOGIC\n(d07)",f"Bodian et al"],fontsize=12,rotation=30)
plt.title("cross-validation (10-fold)",fontsize=14)
plt.savefig("./figure/cv_valid.png",dpi=300,bbox_inches="tight")
#plt.show()
plt.close()

