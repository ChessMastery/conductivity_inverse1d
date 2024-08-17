# Imprting necessary libraries
import numpy as np
from matplotlib import pyplot as plt 
import matplotlib.cm as cm 

r_nodes_vec = []
with open("dir_output/r_coords.txt") as r_nodes_file:
    for line in r_nodes_file:
        r_nodes_vec.append([float(x) for x in line.split()])


z_nodes_vec = []
with open("dir_output/z_coords.txt") as z_nodes_file:
    for line in z_nodes_file:
        z_nodes_vec.append([float(x) for x in line.split()])

q0 = []
with open("dir_output/q70.txt") as q0_file:
    for line in q0_file:
        q0.append([float(x) for x in line.split()]) 

q0_arr = np.asarray(q0)
q0_arr2d = np.reshape(q0, (-1, len(r_nodes_vec)))

# Converting arrays into meshgrid:
R, Z = np.meshgrid(r_nodes_vec, z_nodes_vec)

# Computing Z: 2D Scalar Field
U = q0_arr2d
#U_corr = np.flipud(U) #если узлы пронумерованы "сверху вниз" (как в тельме)
# Plotting our 2D Scalar Field using pcolormesh()

plt.pcolormesh(R, Z, U, cmap = 'magma')
plt.show()

#plt.axis([1e-3, 500, -500, 500])
#plt.pcolormesh(R, Z, U)
#plt.show()

#plt.axis([1e-3, 100, -100, 100])
#plt.pcolormesh(R, Z, U)
#plt.show()


plt.axis([0, 100, -50, 50])
plt.pcolormesh(R, Z, U, cmap = 'magma')
plt.show()