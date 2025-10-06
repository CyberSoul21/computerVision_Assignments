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
print("P2: ",P2)

