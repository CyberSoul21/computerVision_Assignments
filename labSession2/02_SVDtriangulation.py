from labSession1.plotData_v15 import ensamble_T
import matplotlib.pyplot as plt
import numpy as np
import cv2

#de camara a mundo,
T_w_c1 = np.loadtxt('T_w_c1.txt')
T_w_c2 = np.loadtxt('T_w_c2.txt')
#Queremos lo contrario de mundo a camara
T_c1_w = np.linalg.inv(T_w_c1)
T_c2_w = np.linalg.inv(T_w_c2)
K_c = np.loadtxt('K_c.txt')

I_3x3 = np.identity(3)          # 3x3 identity
col_zeros = np.zeros((3,1))     # column of zeros (3x1)
I_matrix = np.hstack((I_3x3, col_zeros))

#Projection Matrices P1 and P2 // P = K[T_w_c]
#print("#Projection Matrices P1 and P2 // P = K[T_w_c]")
P1 = K_c @ I_matrix @ T_c1_w
#print("P1: ",P1)
P2 = K_c @ I_matrix @ T_c2_w
#print("P2: ",P2)


x1 = np.loadtxt('x1Data.txt')
x2 = np.loadtxt('x2Data.txt')

u_1_i, v_1_i = x1
u_2_i, v_2_i = x2

#print(P1.shape)
#print("***********************************************")
#print(P1[2,: ])
#print("***********************************************")
#print(P1[0,: ])


A = np.array([
    u_1_i[0]*P1[2,:] - P1[0,: ],
    v_1_i[0]*P1[2,:] - P1[1,: ],
    u_2_i[0]*P2[2,:] - P2[0,: ],
    v_2_i[0]*P2[2,:] - P2[1,: ],
])
#print("A: ",A)

# solving AX = 0 
U,sigma,VT = np.linalg.svd(A)
#print("U: ",U)
#print("***********************************************")
#print("sigma: ",sigma)
#print("***********************************************")
#print("VT: ",VT)

X = VT[-1];
#Homogeneous cord
X /= X[3]

print(X)


# Split the data into x and y points (assuming alternating x, y pairs)
#x_1_points = x1Data[::2]  # Even indices for x points
#y_1_points = x2Data[1::2]  # Odd indices for y points
#print(x_1_points)
# Example: Access point i (for example, i=5)
#i = 5
#point_i = (x_1_points[i], x_1_points[i])

#print(f"Point {i}: {point_i}")
