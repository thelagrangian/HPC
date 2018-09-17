import time
import numpy as np

n_in = 50176
n0  = 4000
n1  = 1000

w0 = [[0 for j in range(n0)] for i in range(n_in)]
w1 = [[0 for j in range(n1)] for i in range(n0)]
x0 = [0 for j in range(n_in)]
z0 = [0 for j in range(n0)]
z1 = [0 for j in range(n1)]

for i in range(n_in):
    for j in range(n0):
        w0[i][j] = 0.5 + ((i+j)%50-30.)/50.

for i in range(n0):
    for j in range(n1):
        w1[i][j] = 0.5 + ((i+j)%50-30.)/50.

for j in range(n_in):
    x0[j] = 0.5 + (j%50-30.)/50.

start_time = time.time()

for j in range(n0):
    for i in range(n_in):
        z0[j] = z0[j] + x0[i]*w0[i][j]
    z0[j] = z0[j] if z0[j]>0. else 0.


for j in range(n1):
    for i in range(n0):
        z1[j] = z1[j] + z0[i]*w1[i][j]
    z1[j] = z1[j] if z1[j]>0. else 0.

end_time = time.time()

checksum = 0.
for j in range(n1):
    checksum += z1[j]

print("C2:")
print("python code used %s seconds" %(end_time - start_time))
print("python code found checksum %f" %checksum)
print("\n")

w0_np = np.asarray(w0,dtype = np.float64)
w1_np = np.asarray(w1,dtype = np.float64)
x0_np = np.asarray(x0,dtype = np.float64)

start_time = time.time()

z0_np = x0_np.dot(w0_np)
np.clip(z0_np,0,np.inf,out=z0_np)
z1_np = z0_np.dot(w1_np)
np.clip(z1_np,0,np.inf,out=z1_np)

end_time = time.time()

checksum_np = np.sum(z1_np)

print("C3:")
print("numpy code used %s seconds" %(end_time-start_time))
print("numpy code found checksum %f" %checksum_np)
print("\n")
