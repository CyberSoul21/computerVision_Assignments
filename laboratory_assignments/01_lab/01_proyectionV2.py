from labSession1.plotData_v15 import *
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

X = np.array([X_A,X_B,X_C,X_D,X_E]).transpose()
print(X)


labels = ['a','b','c','d','e']

def project(P, Xw_h):
    """Vectorized 3x4 * 4xN -> 3xN, then dehomogenize -> 2xN."""
    S = P @ Xw_h              # 3xN (s*u, s*v, s)
    uv = S[:2, :] / S[2:, :]  # divide each row by s
    print("uv dehomogenize: ")
    print(uv)
    return uv 

def lineFromPoints(p1,p2):
    p1_h = np.array([p1[0,0], p1[1,0], 1.0])
    p2_h = np.array([p2[0,0], p2[1,0], 1.0])
    l_p1p2 = np.cross(p1_h, p2_h)    
    #print("Line p1p2 image 1",l_p1p2)
    l_ab_im1_norm = l_p1p2 / np.linalg.norm(l_p1p2[:2])  # optional normalization
    # check the points lie on the line (=0)
    #res_p1 = a_im1 @ l_ab_im1
    #res_p2 = b_im1 @ l_ab_im1
    #print("res_p1 = ",res_p1)
    return l_p1p2

def drawLine(l, strFormat, lWidth):
    p_l_y = np.hstack((0, -l[2] / l[1]))
    p_l_x = np.hstack((-l[2] / l[0], 0))
    plt.plot([p_l_y[0], p_l_x[0]], [p_l_y[1], p_l_x[1]], strFormat, linewidth=lWidth)


#######################################################################################
#Projection matrices  / One homogeneous point X_A = np.array([3.44, 0.80, 0.82, 1.0])
#**************************************************************************************

#**************************************************************************************
#1.1 Compute the projection matrices P1 and P2.
#P1: Image 1
#Projection Matrices P1 and P2 // P = K[T_w_c]
#print("#Projection Matrices P1 and P2 // P = K[T_w_c]")
P1 = K_c @ I_matrix @ T_c1_w
#print("P1: ")
#print(P1)
P2 = K_c @ I_matrix @ T_c2_w
#print("P2: ")
#print(P2)
#**************************************************************************************

#**************************************************************************************
# 1.2 Compute and plot all the projections of the 3D points A,B,C,D,E in both images.
print("*************************")
print("x1: ")
x1 = project(P1, X)   # 2x5 #Projection on image 1
print("x2: ")
x2 = project(P2, X)   # 2x5 #Projection on image 2
#**************************************************************************************
#########################################################################


#########################################################################
#2D Lines and vanishing points
########################################################################
#**********************************************************************
#2.1 Compute and plot in each image the line l_ab defined by a, and b (projections on A and B).

#get each point inhomogenized
A_2d_im1 = x1[:,0:1]
B_2d_im1 = x1[:,1:2]
C_2d_im1 = x1[:,2:3]
D_2d_im1 = x1[:,3:4]
E_2d_im1 = x1[:,4:]

A_2d_im2 = x2[:,0:1]
B_2d_im2 = x2[:,1:2]
C_2d_im2 = x2[:,2:3]
D_2d_im2 = x2[:,3:4]
E_2d_im2 = x2[:,4:]


#line ab on image 1
l_ab_im1 = lineFromPoints(A_2d_im1,B_2d_im1)
#line cd on image 1
l_cd_im1 = lineFromPoints(C_2d_im1,D_2d_im1)

#line ab on image 2
l_ab_im2 = lineFromPoints(A_2d_im2,B_2d_im2)
#line cd on image 2
l_cd_im2 = lineFromPoints(C_2d_im2,D_2d_im2)
#**********************************************************************

#**********************************************************************
#2.3
#Compute p_12 the intersection point of l_ab and l_cd

#Image 1
P_12_im1 = np.cross(l_ab_im1,l_cd_im1)
P_12_im1 = P_12_im1/P_12_im1[2]
P_12_im1 = np.array([P_12_im1[:2]]).transpose() 
print("P_12_im1: ", P_12_im1)

#Image 2
P_12_im2 = np.cross(l_ab_im2,l_cd_im2)
P_12_im2 = P_12_im2/P_12_im2[2]
P_12_im2 = np.array([P_12_im2[:2]]).transpose() 
print("P_12_im2: ", P_12_im2)
#**********************************************************************

#**********************************************************************
#2.4
#Compute the 3D infinite point corresponding to the 3D direction defined by points A and B, AB_inf.
#Get the direction of AB
D_AB = X_B - X_A
print("D_AB: ",D_AB)
for i, val in enumerate(D_AB):
    if val > 0:
        D_AB[i] = 1
    elif val < 0:
        D_AB[i] = -1
AB_inf = np.array([D_AB]).transpose()
print("AB_inf: ",AB_inf)
#**********************************************************************

#**********************************************************************
#2.5
#Project the point AB_inf with matrix P to obtain the corresponding vanishing point ab_inf.
#Image 1
AB_2d_im1 = project(P1,AB_inf)
print("AB_2d_im1: ",AB_2d_im1)
#Image 2
AB_2d_im2 = project(P1,AB_inf)
print("AB_2d_im2: ",AB_2d_im2)


# ---- Plot on image 1 ----
plt.figure(figsize=(9,5))
plt.imshow(img1)
#plot points
plt.plot(x1[0,:], x1[1,:], '+r', markersize=12)
plotLabeledImagePoints(x1, labels, 'r', (18, -18))   # offset (dx,dy) in pixels
#plot lines
drawLine(l_ab_im1, 'r-', 2) 
drawLine(l_cd_im1, 'g-', 2)
#plot intersection l_ab amd l_cd
plt.plot(P_12_im1[0,:], P_12_im1[1,:], '+b', markersize=12)
plt.text(P_12_im1[0,:] + 5, P_12_im1[1,:] - 5, 'P12', fontsize=10, color='blue')  # Add label near point
#plot infinite point
plt.plot(AB_2d_im1[0,:], AB_2d_im1[1,:], '*c', markersize=12)
plt.text(P_12_im1[0,:] + 5, P_12_im1[1,:] - 5, 'P12', fontsize=10, color='blue')  # Add label near point
# or: plotNumberedImagePoints(x1, 'r', (18, 24))
plt.title('Image 1: projections of A..E')
#plt.axis('off')
plt.tight_layout()

# ---- Plot on image 2 ----
plt.figure(figsize=(9,5))
plt.imshow(img2)
#plot points
plt.plot(x2[0,:], x2[1,:], '+c', markersize=12)
plotLabeledImagePoints(x2, labels, 'c', (18, -18))
#plot lines
drawLine(l_ab_im2, 'r-', 2) 
drawLine(l_cd_im2, 'g-', 2)
#plot intersection l_ab amd l_cd
plt.plot(P_12_im2[0,:], P_12_im2[1,:], '+b', markersize=12)
plt.text(P_12_im2[0,:] + 5, P_12_im2[1,:] - 5, 'P12', fontsize=10, color='blue')  # Add label near point
#plot infinite point
plt.plot(AB_2d_im2[0,:], AB_2d_im2[1,:], '*c', markersize=12)
plt.text(P_12_im2[0,:] + 5, P_12_im2[1,:] - 5, 'P12', fontsize=10, color='blue')  # Add label near point
plt.title('Image 2: projections of A..E')
#plt.axis('off')
plt.tight_layout()
plt.show()
#**********************************************************************

#########################################################################



#TODO: Explain better
#########################################################################
#Understanding SVD with 2D lines fitting
########################################################################
#**********************************************************************
#3.1
#**********************************************************************
# Go to the folder labSession1 into the file line2DFittingSVD_v2.py
# from lines 91 to 111 is the solotion.
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
A = pts[:4, :]            # use only A..D
U, S, Vt = np.linalg.svd(A, full_matrices=True)
pi = Vt[-1]                        # [a,b,c,D] up to scale
print("Pi = ",pi)
n = pi[:3]; pi = pi / np.linalg.norm(n)  # unit-normal form



#**********************************************************************

#**********************************************************************
#4.2
#**********************************************************************
# --- Distances for all points ---
Nnorm = np.linalg.norm(pi[:3])            # should be 1 now
dists = (pts @ pi) / Nnorm                # signed distances

print("Plane (unit-normal):", pi)
print("Signed distances (A..E):", dists)

#**********************************************************************
#########################################################################
#TODO: Make some function in order to simplify the code!