import MDAnalysis as mda
import numpy as np

collagen_structure = mda.Universe("./reference.pdb")

# alpha1 : 179 (167-1205)
# alpha2 : 1064 (1052-2090)
alpha1_1 = []
alpha2 = []
alpha1_2 = []

for i in range(1038):
    alpha1_1.append(collagen_structure.select_atoms(f"segid 0A and resid {167+i}").center_of_mass())
    #print(collagen_structure.select_atoms(f"segid A and resid {i+1}").center_of_mass())
    alpha1_2.append(collagen_structure.select_atoms(f"segid 0C and resid {167+i}").center_of_mass())
    #print(collagen_structure.select_atoms(f"segid C and resid {i+1}").center_of_mass())
for i in range(1026):
        alpha2.append(collagen_structure.select_atoms(f"segid 0B and resid {i+1055}").center_of_mass())

np.save("alpha1_1.npy",np.array(alpha1_1).reshape(-1,3))
np.save("alpha2.npy",np.array(alpha2).reshape(-1,3))
np.save("alpha1_2.npy",np.array(alpha1_2).reshape(-1,3))

