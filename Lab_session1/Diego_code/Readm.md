# 📘 Computer Vision – Laboratory 1  
## Homogeneous Coordinates, Camera Projection, and Geometric Fitting

**Author:** Diego Méndez Carter  
**Course:** MRGCV – University of Zaragoza  
**Date:** October 2025  

---

## 🧩 Files Overview

| File | Description | Topics Covered |
|------|--------------|----------------|
| **usr_lab1.py** | Main implementation for parts 1 & 2 of the lab (3D–2D projection, lines, vanishing points). | Homogeneous coordinates, camera projection matrices, line equations, vanishing point computation. |
| **SVD_copy.py** | Modified script for part 3 (SVD line fitting). | Singular Value Decomposition (SVD), least-squares line fitting, noise handling, projection of points onto lines. |
| **4_0.py** | Final script for part 4 (plane computation). | Plane fitting using SVD, distance from 3D points to a plane. |

---

## 🧠 `usr_lab1.py`

Implements the **core geometry** between world and image coordinates using camera parameters.

### 🔧 Functions

| Function | Description |
|-----------|--------------|
| `ensamble_T(R, t)` | Builds a 4×4 homogeneous transformation matrix \(T_{w\_c}\) from rotation and translation. |
| `invertir_T(T)` | Inverts a homogeneous transformation \(T_{w\_c} \to T_{c\_w}\). |
| `multi_M(A,B)` | Performs safe matrix multiplication with dimension check. |
| `lineFromPoints(p1, p2)` | Returns a 2D line in homogeneous coordinates given two image points. |
| `drawLine(l, fmt, width)` | Draws a 2D line (ax+b+c=0) on a Matplotlib image. |

### 🧮 Process Summary

1. Load camera parameters (**R**, **t**, **K**) for two cameras.  
2. Compute projection matrices \( P_1 = K[R|t] \) and \( P_2 = K[R|t] \).  
3. Project 3D points A–E onto both images.  
4. Compute and draw:
   - Lines \( l_{AB} \) and \( l_{CD} \) in both images.  
   - Intersection \( p_{12} \) (vanishing point).  
   - Vanishing point at infinity (projection of direction AB).  
5. Display projections over **Image 1** and **Image 2**.

---

## 🧩 `SVD_copy.py`

Focuses on **line fitting using Singular Value Decomposition (SVD)**.

### 🔧 Key Steps

1. Load perfect (`xGT`) and noisy (`x`) 2D points lying on a line.  
2. Compute SVD of both sets to find the best-fit line coefficients \( l = [a,b,c] \).  
3. Compare:
   - **3.1:** Using only 2 extreme points → underdetermined system.  
   - **3.2:** Using all 5 perfect points → overdetermined, smallest singular value ≈ 0.  
4. Interpret singular values (3.3).  
5. Force smallest singular value to 0 (3.4) to reconstruct **denoised points projected on the fitted line**.

### 🧠 Functions

| Function | Description |
|-----------|--------------|
| `drawLine(l, fmt, width)` | Draws a line given its homogeneous coefficients. |

---

## 🌍 `4_0.py`

Implements **plane estimation in 3D using SVD** and computes distances from 3D points to that plane.

### 🔧 Functions

| Function | Description |
|-----------|--------------|
| `point_to_plane_distance(point, plane)` | Computes perpendicular distance from a 3D point \((x,y,z)\) to a plane \(aX+bY+cZ+d=0\). |

### 🧮 Process Summary

1. Define 3D points A, B, C, D (forming a rectangle at z=0.82).  
2. Construct matrix \( X = [A\ B\ C\ D; 1\ 1\ 1\ 1] \).  
3. Compute SVD of \( X^T \); the last column of \(V\) gives plane coefficients \(\pi = [a,b,c,d]\).  
4. Normalize \(\pi\) so that \(\sqrt{a^2+b^2+c^2}=1\).  
5. Compute distances of A–E to the plane.  
   - A–D → 0 m (they define the plane).  
   - E → ~0.39 m above the plane.

---

## ⚙️ Libraries Used

- **NumPy** – Matrix and vector math.  
- **Matplotlib** – Visualization of projections and fitted geometry.  
- **SciPy.linalg** – SVD and matrix reconstruction.  
- **OpenCV (cv2)** – Image loading and RGB color conversion.  

---

## ✅ Summary of Results

| Task | Result |
|------|--------|
| 3D→2D Projection | Correct projection of world points onto both camera images. |
| 2D Lines & Vanishing Points | Lines \(l_{AB}\), \(l_{CD}\) and intersection \(p_{12}\) computed and visualized. |
| SVD Line Fitting | Accurate line estimation; smallest singular value ≈ 0 for perfect data. |
| Plane Estimation | Plane \(\pi = [0, 0, -1, 0.82]\) corresponding to z = 0.82 m. |
| Point Distances | \(d_A, d_B, d_C, d_D = 0.0\ m\); \(d_E ≈ 0.39\ m\). |

---

> 🧩 *This laboratory reinforced the connection between linear algebra, camera geometry, and visual representation, using SVD as a unifying mathematical tool.*

