import numpy as np



def build_xyz(a1,name):
    f = open(f'./{name}.xyz', 'w')
    f.write(f"{a1.shape[0]}\n")
    f.write("graph collagen\n")
    num = 1
    for i in range(0,a1.shape[0]):
        f.write(f"{1} {a1[i][0]} {a1[i][1]} {a1[i][2]}\n")
        num += 1
    f.close()

if __name__ == '__main__':
    a1_1 = np.load("./alpha1_1.npy")
    a2 = np.load("./alpha2.npy")
    a1_2 = np.load("./alpha1_2.npy")
    build_xyz(a1_1,"./a1_1") 
    build_xyz(a2,"./a2") 
    build_xyz(a1_2,"./a1_2") 