from labSession1.plotData_v15 import ensamble_T
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load ground truth
R_w_c1 = np.loadtxt('labSession1/R_w_c1.txt')
R_w_c2 = np.loadtxt('labSession1/R_w_c2.txt')

t_w_c1 = np.loadtxt('labSession1/t_w_c1.txt')
t_w_c2 = np.loadtxt('labSession1/t_w_c2.txt')

#de camara a mundo,
T_w_c1 = ensamble_T(R_w_c1, t_w_c1)
T_w_c2 = ensamble_T(R_w_c2, t_w_c2)
#Queremos lo contrario de mundo a camara
T_c1_w = np.linalg.inv(T_w_c1)
T_c2_w = np.linalg.inv(T_w_c2)

K_c = np.loadtxt('labSession1/K.txt')

I_3x3 = np.identity(3)          # 3x3 identity
col_zeros = np.zeros((3,1))     # column of zeros (3x1)
I_matrix = np.hstack((I_3x3, col_zeros))


X_A =  np.array([3.44, 0.80, 0.82, 1])
X_B =  np.array([4.20, 0.80, 0.82, 1])
X_C =  np.array([4.20, 0.60, 0.82, 1])
X_D =  np.array([3.55, 0.60, 0.82, 1])
X_E =  np.array([-0.01, 2.6, 1.21, 1])


points = np.array([X_A,X_B,X_C,X_D,X_E]).transpose()
"""
print("Points: ")
print(points)
"""
#Projection Matrices P1 and P2 // P = K[T_w_c]
#print("#Projection Matrices P1 and P2 // P = K[T_w_c]")
P1 = K_c @ I_matrix @ T_c1_w
#print("P1: ")
#print(P1)
P2 = K_c @ I_matrix @ T_c2_w
#print("P2: ")
#print(P2)

#######################################################################################
#Projection matrices  / One homogeneous point X_A = np.array([3.44, 0.80, 0.82, 1.0])
#**************************************************************************************
#P1: Image 1
#**************************************************************************************
# Dot products with rows of P
As_u1 = np.dot(P1[0], X_A)   # row 1 · point
As_v1 = np.dot(P1[1], X_A)   # row 2 · point
s1   = np.dot(P1[2], X_A)   # row 3 · point
A_2d_im1 = (np.array([As_u1 , As_v1]).transpose())*(1/s1)
print("A_2d_im1: ",A_2d_im1)


# Dot products with rows of P
Bs_u1 = np.dot(P1[0], X_B)   # row 1 · point
Bs_v1 = np.dot(P1[1], X_B)   # row 2 · point
s1   = np.dot(P1[2], X_B)   # row 3 · point
B_2d_im1 = (np.array([Bs_u1 , Bs_v1]).transpose())*(1/s1)
print("B_2d_im1: ",B_2d_im1)

# Dot products with rows of P
Cs_u1 = np.dot(P1[0], X_C)   # row 1 · point
Cs_v1 = np.dot(P1[1], X_C)   # row 2 · point
s1   = np.dot(P1[2], X_C)   # row 3 · point
C_2d_im1 = (np.array([Cs_u1 , Cs_v1]).transpose())*(1/s1)
print("C_2d_im1: ",C_2d_im1)


# Dot products with rows of P
Ds_u1 = np.dot(P1[0], X_D)   # row 1 · point
Ds_v1 = np.dot(P1[1], X_D)   # row 2 · point
s1   = np.dot(P1[2], X_D)   # row 3 · point
D_2d_im1 = (np.array([Ds_u1 , Ds_v1]).transpose())*(1/s1)
print("D_2d_im1: ",D_2d_im1)


# Dot products with rows of P
Es_u1 = np.dot(P1[0], X_E)   # row 1 · point
Es_v1 = np.dot(P1[1], X_E)   # row 2 · point
s1   = np.dot(P1[2], X_E)   # row 3 · point
E_2d_im1 = (np.array([Es_u1 , Es_v1]))*(1/s1)
print("E_2d_im1: ",E_2d_im1.transpose())
#**************************************************************************************

#**************************************************************************************
#P2 Image 2
#**************************************************************************************
# Dot products with rows of P
As_u2 = np.dot(P2[0], X_A)   # row 1 · point
As_v2 = np.dot(P2[1], X_A)   # row 2 · point
s2   = np.dot(P2[2], X_A)   # row 3 · point
A_2d_im2 = (np.array([As_u2 , As_v2]).transpose())*(1/s2) #no homogeneous
print("A_2d_im2: ",A_2d_im2)


# Dot products with rows of P
Bs_u2 = np.dot(P2[0], X_B)   # row 1 · point
Bs_v2 = np.dot(P2[1], X_B)   # row 2 · point
s2   = np.dot(P2[2], X_B)   # row 3 · point
B_2d_im2 = (np.array([Bs_u2 , Bs_v2]).transpose())*(1/s2)
print("B_2d_im2: ",B_2d_im2)

# Dot products with rows of P
Cs_u2 = np.dot(P2[0], X_C)   # row 1 · point
Cs_v2 = np.dot(P2[1], X_C)   # row 2 · point
s2   = np.dot(P2[2], X_C)   # row 3 · point
C_2d_im2 = (np.array([Cs_u2 , Cs_v2]).transpose())*(1/s2)
print("C_2d_im2: ",C_2d_im2)


# Dot products with rows of P
Ds_u2 = np.dot(P2[0], X_D)   # row 1 · point
Ds_v2 = np.dot(P2[1], X_D)   # row 2 · point
s2   = np.dot(P2[2], X_D)   # row 3 · point
D_2d_im2 = (np.array([Ds_u2 , Ds_v2]).transpose())*(1/s2)
print("D_2d_im2: ",D_2d_im2)


# Dot products with rows of P
Es_u2 = np.dot(P2[0], X_E)   # row 1 · point
Es_v2 = np.dot(P2[1], X_E)   # row 2 · point
s2   = np.dot(P2[2], X_E)   # row 3 · point
E_2d_im2 = (np.array([Es_u2 , Es_v2]))*(1/s2)
print("E_2d_im1: ",E_2d_im2.transpose())
#**************************************************************************************




#########################################################################
#2D Lines and vanishing points
########################################################################
#**********************************************************************
#2.1 Compute and plot in each image the line l_ab defined by a, and b (projections on A and B).
#line ab on image 1
a_im1 = np.array([A_2d_im1[0], A_2d_im1[1], 1.0]).transpose()
b_im1 = np.array([B_2d_im1[0], B_2d_im1[1], 1.0]).transpose()
l_ab_im1 = np.cross(a_im1, b_im1)    
print("Line ab image 1",l_ab_im1)

l_ab_im1_norm = l_ab_im1 / np.linalg.norm(l_ab_im1[:2])  # optional normalization

# check the points lie on the line (≈0)
res_a = a_im1 @ l_ab_im1
res_b = b_im1 @ l_ab_im1

#TODO: Correct all lines!
#line ab on image 2
a_im2 = np.array([A_2d_im2[0], A_2d_im2[1], 1.0]).transpose()
b_im2 = np.array([B_2d_im2[0], B_2d_im2[1], 1.0]).transpose()
l_ab_im2 = np.cross(a_im2, b_im2)    
print("Line ab image 1",l_ab_im2)

l_ab_im2_norm = l_ab_im2 / np.linalg.norm(l_ab_im2[:2])  # optional normalization

# check the points lie on the line (≈0)
res_a = a_im2 @ l_ab_im2
res_b = b_im2 @ l_ab_im2
#**********************************************************************

#**********************************************************************
#2.2
#line cd on image 1
c_im1 = np.array([C_2d_im1[0], C_2d_im1[1], 1.0])
d_im1 = np.array([D_2d_im1[0], D_2d_im1[1], 1.0])
l_cd_im1 = np.cross(d_im1, c_im1)            # (A,B,C)
print("Line cd image 1",l_cd_im1)

#line cd on image 2
c_im2 = np.array([C_2d_im2[0], C_2d_im2[1], 1.0])
d_im2 = np.array([D_2d_im2[0], D_2d_im2[1], 1.0])
l_cd_im2 = np.cross(d_im2, c_im2)            # (A,B,C)
print("Line cd image 1",l_cd_im2)
#**********************************************************************

#**********************************************************************
#2.3
#Compute p_12 the intersection point of l_ab and l_cd

#**********************************************************************
#Image 1

P_12_im1 = np.cross(l_ab_im1,l_cd_im1)
P_12_im1 = P_12_im1/P_12_im1[2]
print("P_12_im1: ", P_12_im1)

#Image 2

P_12_im2 = np.cross(l_ab_im2,l_cd_im2)
P_12_im2 = P_12_im2/P_12_im2[2]
print("P_12_im2: ", P_12_im2)
#**********************************************************************

#**********************************************************************
#2.4
#Compute the 3D infinite point corresponding to the 3D direction defined by points A and B, AB_inf.
#Get the direction of AB

D_AB = X_B - X_A
print("D_AB: ",D_AB)
#for i, val in enumerate(D_AB):
#    if val > 0:
#        D_AB[i] = 1
#    elif val < 0:
#        D_AB[i] = -1
AB_inf = D_AB
print("AB_inf: ",AB_inf)
#**********************************************************************

#**********************************************************************
#2.5
#Project the point AB_inf with matrix P to obtain the corresponding vanishing point ab_inf.
#Image 1
ABs_u1 = np.dot(P1[0], AB_inf)   # row 1 · point
ABs_v1 = np.dot(P1[1], AB_inf)   # row 2 · point
s1   = np.dot(P1[2], AB_inf)   # row 3 · point
AB_2d_im1 = (np.array([ABs_u1 , ABs_v1]).transpose())*(1/s1)
print("AB_2d_im1: ",AB_2d_im1)


#Image 2


#**********************************************************************






#########################################################################



#TODO: Explain better
#########################################################################
#Understanding SVD with 2D lines fitting
########################################################################
#**********************************************************************
#3.1
#**********************************************************************

#**********************************************************************
#3.2
#**********************************************************************

#**********************************************************************
#3.3
#**********************************************************************

#**********************************************************************
#3.4
#**********************************************************************

#########################################################################


#########################################################################
#Planes in homogeneous coordinates
########################################################################
#**********************************************************************
#4.1
#**********************************************************************

pts = np.array([
    [3.44, 0.80, 0.82, 1.0],  # A
    [4.20, 0.80, 0.82, 1.0],  # B
    [4.20, 0.60, 0.82, 1.0],  # C
    [3.55, 0.60, 0.82, 1.0],  # D
    [-0.01, 2.60, 1.21, 1.0], # E
])

U, S, Vt = np.linalg.svd(pts, full_matrices=True)
pi = Vt[-1]                        # [a,b,c,D] up to scale
n = pi[:3]; pi = pi / np.linalg.norm(n)  # unit-normal form


#**********************************************************************
#4.2
#**********************************************************************
# distances (signed); abs() for unsigned
dists = (pts @ pi)                 # since ||n||=1
print("plane:", pi)
print("dists A..E:", dists)
#########################################################################
#TODO: Make some function in order to simplify the code!