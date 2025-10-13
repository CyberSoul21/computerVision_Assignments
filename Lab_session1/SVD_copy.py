#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: Line fitting with SVD
#
# Date: 5 September 2024
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.5
#
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.linalg as scAlg

def drawLine(l,strFormat,lWidth):
    """
    Draw a line
    -input:
      l: image line in homogenous coordinates
      strFormat: line format
      lWidth: line width
    -output: None
    """
    # p_l_y is the intersection of the line with the axis Y (x=0)
    p_l_y = np.hstack((0, -l[2] / l[1]))
    # p_l_x is the intersection point of the line with the axis X (y=0)
    p_l_x = np.hstack((-l[2] / l[0], 0))
    # Draw the line segment p_l_x to  p_l_y
    plt.plot([p_l_y[0], p_l_x[0]], [p_l_y[1], p_l_x[1]], strFormat, linewidth=lWidth)


if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)

    # This is the ground truth
    l_GT = np.array([[2], [1], [-1500]])

    plt.figure(1)
    plt.plot([-100, 1800], [0, 0], '--k', linewidth=1)
    plt.plot([0, 0], [-100, 1800], '--k', linewidth=1)
    # Draw the line segment p_l_x to  p_l_y
    drawLine(l_GT, 'g-', 1)
    plt.draw()
    plt.axis('equal')

    print('Click in the image to continue...')
    plt.waitforbuttonpress()

    ## Generating points lying on the line but adding perpendicular Gaussian noise
    l_GTNorm = l_GT/np.sqrt(np.sum(l_GT[0:2]**2, axis=0)) #Normalized the line with respect to the normal norm

    x_l0 = np.vstack((-l_GTNorm[0:2]*l_GTNorm[2],1))  #The closest point of the line to the origin
    plt.plot([0, x_l0[0,0]], [0, x_l0[1,0]], '-r')
    plt.draw()

    # mu = np.arange(-1000, 1000, 250)
    # noiseSigma = 15 #Standard deviation
    # xGT = x_l0 + np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]) @ (l_GTNorm * mu)
    # x = xGT + np.diag([1, 1, 0]) @ np.random.normal(0, noiseSigma, (3, len(mu)))

    xGT = np.loadtxt('D:/Dev_Space/python/Practice1/x2DGTLineFittingSVD.txt')
    x = np.loadtxt('D:/Dev_Space/python/Practice1/x2DLineFittingSVD.txt')
    plt.plot(xGT[0, :], xGT[1, :], 'b.')
    plt.plot(x[0, :], x[1, :], 'rx')
    plt.draw()
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    
    """Aqui hare las modificaciones para el 3.1"""
    x_extreme= x[:, [0,-1]] # Primer y último punto
    A_2points = x_extreme.T
    u, s, vh = np.linalg.svd(A_2points)
    v=vh.T
    A_2pts_l_ls = v[:, -1]
    A_2pts_l_ls = A_2pts_l_ls / np.linalg.norm(A_2pts_l_ls[0:2])
    print (u.shape, s.shape, vh.shape)

    """Aqui hare las modificaciones para el 3.2"""
    A_xGTpts = xGT.T 
    u_gt, s_gt, vh_gt = np.linalg.svd(A_xGTpts)
    v=vh_gt.T
    A_gt_l_ls = v[:, -1]
    A_gt_l_ls = A_gt_l_ls / np.linalg.norm(A_gt_l_ls[0:2])
    print (u_gt.shape, s_gt.shape, vh_gt.shape)
    # ============================================================
    # 3.2 – Sizes of U, S, and V matrices in each case
    # ============================================================

    # Case 3.1 (using only 2 extreme points)
    # A has shape (2×3): two points → 2 equations, 3 unknowns (x, y, 1)
    # → U: (2×2)
    # → S: (2,)
    # → V: (3×3)
    # The system is underdetermined (infinite solutions), 
    # and the line is given by the last column of V (smallest singular value).

    # Case 3.2 (using 5 perfect points)
    # A has shape (5×3): five points → 5 equations, 3 unknowns
    # → U: (5×5)
    # → S: (3,)
    # → V: (3×3)
    # The system is overdetermined, but since the 5 points are perfectly aligned,
    # the smallest singular value ≈ 0, confirming perfect collinearity.

    # --- 3.3: Inspect and interpret singular values ---

    print("\n===== 3.3 Singular Values Comparison =====")
    print("Case 3.1 (2 points):", s)
    print("Case 3.2 (5 perfect points):", s_gt)

    # Observation:
    # - The last singular value (s[-1]) should be very close to 0 in both cases.
    # - If s[-1] = 0 exactly → points are perfectly on a line.
    # - If s[-1] > 0 → there is some perpendicular noise or deviation.
        ## Fit the least squares solution from points with noise using svd
        # we want to solve the equation x.T @ l = 0
    
    
    """3.4 no tengo ni idea S"""  
    
    
    
    
    u, s, vh = np.linalg.svd(x.T) # svd function returns vh which is the tranpose version of V matrix.
    sM = scAlg.diagsvd(s, u.shape[0], vh.shape[0])  # svd function returns the diagonal s values instead of the S matrix. 
    l_ls = vh[-1, :]

    # Notice that the input matrix A of the svd has been decomposed such that A = u @ sM @ vh 

    drawLine(l_ls, 'r--', 1)
    plt.draw()
    plt.waitforbuttonpress()


    ## Project the points on the line using SVD
    s[2] = 0  # If all the points are lying on the line s[2] = 0, therefore we impose it
    xProjectedOnTheLine = (u @ scAlg.diagsvd(s, u.shape[0], vh.shape[0]) @ vh).T
    xProjectedOnTheLine /= xProjectedOnTheLine[2, :]

    plt.plot(xProjectedOnTheLine[0,:], xProjectedOnTheLine[1, :], 'bx')
    plt.show()
    print('End')