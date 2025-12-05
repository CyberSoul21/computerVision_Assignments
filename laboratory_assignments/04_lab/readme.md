  # **Laboratory Session 4 — Bundle Adjustment & Multiview Geometry**  
*(Computer Vision — UNIZAR)*  
:contentReference[oaicite:0]{index=0}

---

## **Table of Contents**
   [General Description](#general-description)  
   [Objectives of the Laboratory](#objectives-of-the-laboratory)  
   [Prerequisites and Provided Material](#prerequisites-and-provided-material)  
   [1. Line Fitting with Least Squares](#1-line-fitting-with-least-squares)  
   - [1.1 Theoretical Concepts](#11-theoretical-concepts)  
   - [1.2 Implementation](#12-implementation)  
   - [1.3 Results and Visualizations](#13-results-and-visualizations)  
   [2. Multiview Geometry and Pose Estimation](#2-multiview-geometry-and-pose-estimation)  
   - [2.1 Initial Reconstruction from Two Views](#21-initial-reconstruction-from-two-views)  
   - [2.2 Residual Function](#22-residual-function)  
   - [2.3 Bundle Adjustment (Two Views)](#23-bundle-adjustment-two-views)  
   - [2.4 Comparison with Ground Truth](#24-comparison-with-ground-truth)  
    [3. Perspective-n-Point (PnP) for View 3](#3-perspective-n-point-pnp-for-view-3)  
   [4. Bundle Adjustment with Three Views](#4-bundle-adjustment-with-three-views)  
   [Appendix A — Rotation Representation in SO(3)](#appendix-a--rotation-representation-in-so3)  
   [References](#references)

---

## **General Description**
In this laboratory session we implement a complete multiview reconstruction pipeline, including:

- Estimation of relative pose between **three cameras**  
- Triangulation of 3D scene points  
- Geometric refinement through **Bundle Adjustment (BA)**  
- Comparison against *ground truth*  
- Use of **Perspective-n-Point (PnP)** to estimate the pose of the third view  

---

## **Objectives of the Laboratory**
- Understand the relationship between multiple views and how to retrieve relative camera poses.  
- Implement residual functions for nonlinear optimization problems.  
- Execute Bundle Adjustment to improve initial geometric estimates.  
- Integrate a third view using OpenCV solvePnP and extend the BA formulation.
  
# **2. Multiview Geometry and Pose Estimation**

## **2.1 Initial Reconstruction from Two Views**

You should document:

## **Overview — Initial Two-View Geometry**

### **Essential Matrix**
- Converts the Fundamental matrix into calibrated geometry:

$$
E = K^{T} F K
$$

- Encodes only the relative rotation and translation.

---

### **Essential Matrix Constraints**
- The essential matrix must have **two identical singular values** and the **third equal to zero**.

(SVD enforcement)

$$
\Sigma = \text{diag}(1,\,1,\,0)
$$

---

### **Recovering \(R\) and \(t\)**
- Using SVD decomposition \(E = U \Sigma V^{T}\), motion is extracted:

Rotation candidates:

$$
R = U\,W\,V^{T}
$$

where  

$$
W = 
\begin{bmatrix}
0 & -1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

Translation direction:

$$
t = U[:, 2]
$$

---

### **Four-Solution Ambiguity**
- The decomposition yields four possible motion pairs:

$$
(R_1,\, t),\quad (R_1,\,-t),\quad (R_2,\, t),\quad (R_2,\,-t)
$$

---

### **Cheirality Check**
- The physically correct pose is the one where triangulated points satisfy:

$$
Z_1 > 0
$$

and

$$
Z_2 > 0
$$

(Points must lie **in front of both cameras**.)

---

### **Initial Linear Triangulation**
- 3D points are estimated by solving the linear camera projection equations:

$$
x = P X
$$

Result: a first approximation of the scene structure before applying BA.


Pipeline steps:

1. Load **F**  
2. Convert to **E**  
3. Extract candidate (**R**, **t**) pairs  
4. Select the valid configuration  
5. Triangulate 3D points in camera 1 reference
## **Before Bundle Adjustment: Initial Pose & 3D Reconstruction**

### **Step 1 — Initial Pose from the Essential Matrix**

1. **Compute the Essential Matrix**

$$
E = K^\top F K
$$

3. **Decompose \(E\)** into the four possible camera poses:
   
$$
(R, t),\; (R, -t),\; (R', t),\; (R', -t)
$$

5. **Prepare 2D points** for triangulation  
   - Convert homogeneous points to (u, v)

6. **Select the physically correct pose** using the cheirality condition:  
   Points must satisfy:

$$
Z_1 > 0 \quad \text{and} \quad Z_2 > 0
$$

8. **Triangulate initial 3D points** using the selected pose.

9. **Construct the initial transformation**
    
$$
T_{21} = \begin{bmatrix} R_{21} & t_{21} \\ 0 & 1 \end{bmatrix}
$$

---

### **Step 2 — Initial Residual Visualization**

1. **Project initial 3D points** onto both cameras:
   
$$
\hat{x}_1 = K[I|0]X,\qquad 
\hat{x}_2 = K[R_{21}|t_{21}]X
$$

3. **Normalize projections** (divide by \(z\)).

4. **Plot observed vs. projected points** to visualize initial reprojection error.

5. **Project ground-truth 3D points** for comparison before optimization.

<img width="1895" height="901" alt="image" src="https://github.com/user-attachments/assets/c37114f2-ad2c-4764-a6c3-1c87c6f80093" />


## **2.2 Residual Function**

Template:

```python
def resBundleProjection(Op, x1Data, x2Data, K_c, nPoints):
    """
    Residuals between predicted projections and measured 2D points.
    """
```
This section should explain:

- Parameter vector construction:  
  **Op = [θ₂₁, t₂₁, X₁_points]**
- Minimal rotation parameterization using **θ ∈ so(3)**
- Exponential map to compute **R** from **θ**
- Projection formula:  
  **x̂ = K (R X + t)**
- Residual per point:  
  **r = x_measured – x̂**

  ## **Step 3 — Bundle Adjustment (Two Views)**

### **1. Compute ground-truth baseline length**
- Invert GT poses to obtain camera-to-world transforms.  
- Compute relative pose:

$$
T_{12}^{GT} = T_{c1}^{w^{-1}} \, T_{c2}^{w}
$$

- Extract translation vector \(t_{12}\).  
- Compute baseline magnitude:

$$
\|t_{12}\| = \text{baseline length (meters)}
$$

- Used to **scale the BA results** (monocular geometry has unknown scale).

---

### **2. Run Bundle Adjustment**
Call the optimizer:

```python
T_21_opt, X1_opt, res_init, res_final = bundleAdjustment(
    x1Data, x2Data, K_c, T_21_init, X1_init, normal_t12
)
```
### **What BA optimizes jointly**
- Rotation: \(R_{21}\)  
- Translation: \(t_{21}\)  
- 3D points: \(X_1\)

---

### **Optimization objective**

$$
\min \sum_i 
\left\| x_{1,i} - \hat{x}_{1,i} \right\|^2
+
\left\| x_{2,i} - \hat{x}_{2,i} \right\|^2
$$

---

### **Projection model**

$$
\hat{x} = K(RX + t)
$$

---

### **Outputs of the BA step**
- `T_21_opt` → optimized camera pose  
- `X1_opt` → optimized 3D points  
- `res_init` → residuals before BA  
- `res_final` → residuals after BA  

---


---

## **2.3 Bundle Adjustment (Two Views)**

### **Steps:**

- Build **Op** with rotation, translation, and 3D points  
- Implement projection for both views  
- Call `least_squares()`  
- Analyze:  
  - Residual magnitude  
  - Convergence  
  - Improvement of 3D structure  
- Visualize final reconstruction  

**Figure:**  
<img width="1899" height="830" alt="image" src="https://github.com/user-attachments/assets/ca2ea862-075f-4843-96ac-447a2e6d4f29" />


---

## **2.4 Comparison with Ground Truth**

### **Include:**

- Plot of estimated vs. true camera poses  
- Plot of reconstructed vs. true 3D points  
- Numerical error analysis  

**Example figure:**  
<img width="846" height="794" alt="image" src="https://github.com/user-attachments/assets/2ee8e4c9-5a54-4c98-b134-f10bbe45aa9c" />


---

# **3. Perspective-n-Point (PnP) for View 3**

### **Steps:**

1. Take the 3D points from initial BA  
2. Extract the 2D correspondences in image 3  
3. Format image points to shape **(N, 1, 2)**  
4. Call `solvePnP` using `cv2.SOLVEPNP_EPNP`  
5. Convert rotation vector to matrix using **Rodrigues**  
6. Build transformation **T₃₁**  

### **Example:**

```python
imagePoints = np.ascontiguousarray(
    x[0:2, :].T
).reshape((x.shape[1], 1, 2))

retval, rvec, tvec = cv2.solvePnP(
    objectPoints,
    imagePoints,
    K_c,
    np.zeros(5),
    flags=cv2.SOLVEPNP_EPNP
)
```

### **Visualization:**

*(Insert corresponding figure)*

---

# **4. Bundle Adjustment with Three Views**

### **You should explain:**

- Extended parameter vector:  
  **Op = [θ₂₁, t₂₁, θ₃₁, t₃₁, X₃D]**
- Degrees of freedom:  
  - **Camera 1 → fixed**  
  - **Cameras 2 & 3 → rotation + translation**
- Combined residuals for three-view projections  
- Increase in parameter count and constraints  
- Final scaling using **T₁₂** from ground truth  

---

### **Steps:**

1. Build the extended **Op**  
2. Implement residuals for all three views  
3. Execute `least_squares()`  
4. Scale the final reconstruction  
5. Visualize the entire system  

**Figure:**  
*(Insert when available)*

---

# **Appendix A — Rotation Representation in SO(3)**

### **Helpful code:**

```python
def crossMatrix(x):
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])
```
### **Explain:**

- Vector **θ** and the skew-symmetric matrix **[θ]×**  
- Exponential map:  
  **R = exp([θ]×)**  
- Logarithmic map:  
  **θ = log(R)** using `scipy.linalg.logm`  
- Numerical issues with **float32**  
- Recommended use of **float64**  

---

# **References**

- Official laboratory material  
- *Multiple View Geometry in Computer Vision* — Hartley & Zisserman  
- SciPy documentation — `least_squares`  
- OpenCV documentation — `solvePnP`

