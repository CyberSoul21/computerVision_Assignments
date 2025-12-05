  # **Laboratory Session 4 — Bundle Adjustment & Multiview Geometry**  
*(Computer Vision — UNIZAR)*  
:contentReference[oaicite:0]{index=0}

---

## **Table of Contents**
- [General Description](#general-description)
- [Objectives of the Laboratory](#objectives-of-the-laboratory)
- [2. Multiview Geometry and Pose Estimation](#2-multiview-geometry-and-pose-estimation)
  - [2.1 Initial Reconstruction from Two Views](#21-initial-reconstruction-from-two-views)
  - [2.2 Residual Function](#22-residual-function)
  - [2.3 Bundle Adjustment (Two Views)](#23-bundle-adjustment-two-views)
  - [2.4 Comparison with Ground Truth](#24-comparison-with-ground-truth)
- [3. Perspective-n-Point (PnP) for View 3](#3-perspective-n-point-pnp-for-view-3)
- [4. Bundle Adjustment with Three Views](#4-bundle-adjustment-with-three-views)
- [Appendix A — Rotation Representation in SO(3)](#appendix-a--rotation-representation-in-so3)
- [References](#references)

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

Pipeline steps:

## **Before Bundle Adjustment: Initial Pose & 3D Reconstruction**

### **1. Load F (Fundamental Matrix)**
- The **Fundamental Matrix** encodes the epipolar geometry **between two uncalibrated images**.  
- It relates corresponding pixels in image 1 and image 2 through the epipolar constraint.  
- It tells us how points in one image restrict the position of their match in the other.

---

### **2. Convert to E (Essential Matrix)**
- The **Essential Matrix** is the calibrated version of the fundamental matrix.  
- It incorporates the camera intrinsic parameters and describes **only the relative pose (R, t)** between two cameras.  
- Once converted, the geometry depends only on 3D motion, not on pixel coordinates.

---

### **3. Extract Candidate (R, t) Pairs**
- The essential matrix can be decomposed into **four possible motion solutions**: two possible rotations and two possible translation directions.  
- These represent all mathematically valid relative camera poses consistent with the epipolar geometry.

---

### **4. Select the Valid Configuration (Cheirality Check)**
- Only **one** of the four candidates places the triangulated 3D points **in front of both cameras**.  
- This physical feasibility test (cheirality) determines the correct rotation and translation.

---

### **5. Triangulate 3D Points (Initial Reconstruction)**
- With a valid camera pose, corresponding pixels are back-projected into 3D.  
- Intersecting their rays from each camera yields **initial 3D scene points** in the reference frame of camera 1.  
- This gives a coarse but consistent 3D structure — the starting point for Bundle Adjustment.


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
Bundle Adjustment solves for the camera pose and 3D structure that best explain the observed 2D points.  
It does this by minimizing the reprojection error of all points in all images:

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

## **Step 3 — Bundle Adjustment (Two Views)**

The residual function measures the mismatch between **predicted** image points (obtained by projecting 3D points using the current pose estimate) and the **observed** 2D image points.  
Bundle Adjustment minimizes these residuals to refine camera pose and 3D structure.
---

### **Parameter Vector Construction**
The optimization vector groups all variables BA must refine:

$$
O_p = [\theta_{21}, t_{21}, X_{1\_points}]
$$



Where:

- **θ₂₁** → 3-parameter rotation representation (so(3))  
- **t₂₁** → translation from camera 1 to camera 2  
- **X₁_points** → all 3D points expressed in camera 1 coordinates  

---

### **Minimal Rotation Parameterization (θ ∈ so(3))**

Rotation is encoded using only **3 parameters** instead of a 3×3 matrix.  
This avoids redundancy and ensures the rotation stays valid during optimization.

---

### **Exponential Map (θ → R)**

The 3-vector θ is converted into a valid rotation matrix using the exponential map:

$$
R = \exp([\theta]_{\times})
$$

---

### **Projection Model**

The predicted image coordinates for a 3D point are:

$$
\hat{x} = K ( R X + t )
$$

---

### **Residual per Point**

The residual measures the difference between observed and predicted image points:

$$
r = x_{\text{measured}} - \hat{x}
$$


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

**Meaning:**  
For each correspondence \(i\), we compare the true image points \(x_{1,i}, x_{2,i}\) with the predicted ones \(\hat{x}_{1,i}, \hat{x}_{2,i}\).  
If the prediction is perfect → error is zero.  
BA adjusts **R**, **t**, and the **3D points X** to make these errors as small as possible.

---

### **Projection model**
This is how we predict where a 3D point should appear in the image:

$$
\hat{x} = K(RX + t)
$$

**Meaning:**  
- First we transform the 3D point from camera 1's frame into camera 2 using \( R X + t \).  
- Then we project it through the camera intrinsics \(K\).  
- The result is the pixel location where that point *should* be if the geometry were perfect.

This prediction is what produces the residuals that BA tries to minimize.

---

### **Outputs of the BA step**

- **\(T_{21,\text{opt}}\)** → optimized relative pose  
  - Final estimate of rotation & translation between camera 1 and camera 2.

- **\(X_{1,\text{opt}}\)** → optimized 3D structure  
  - Improved 3D points after optimization, usually much more consistent with the images.

- **res\_init** → residuals before BA  
  - Tells you how inaccurate the initial triangulation was.

- **res\_final** → residuals after BA  
  - Should be significantly smaller; indicates the improvement made by BA.


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

The goal of this section is to estimate the **pose of camera 3** with respect to **camera 1** using:

- The **3D points** reconstructed in Section 2  
- The **2D matched points** detected in image 3  


### **Steps:**

1. **Initial pose estimation** (cameras 1 & 2) using the Essential Matrix  
2. **Bundle Adjustment** to refine pose and 3D structure  
3. **PnP** to estimate the pose of camera 3  
4. **Validation and visualization**

# **Explanation of the Complete Pipeline Code (BA + PnP)**

This script implements the full 3-camera reconstruction pipeline:

1. **Initial pose estimation** (cameras 1 & 2) using the Essential Matrix  
2. **Bundle Adjustment** to refine pose and 3D structure  
3. **PnP** to estimate the pose of camera 3  
4. **Validation and visualization**

Below are the important conceptual actions of each part of the code.

---

## **1. Loading Data**

The script loads:

- 2D keypoints for **three cameras** (`x1Data`, `x2Data`, `x3Data`)
- Camera intrinsics `K_c`
- Precomputed fundamental matrix `F_21`
- Ground truth camera poses and 3D points

These are the inputs required for:

- Essential matrix estimation  
- Initial triangulation  
- PnP  
- Error evaluation  

---

## **2. Initial Pose from Essential Matrix**

The code:

- Computes the **essential matrix** using  
  `E = essentialMatrix(F_21, K_c)`
- Decomposes \(E\) into **four possible (R,t) solutions**
- Uses **cheirality + triangulation** via  
  `selectCorrectPose()`  
  to identify the physically valid configuration.

This produces:

- **Initial relative pose** \(T_{21}\)  
- **Initial 3D points** triangulated from the first two cameras  

This serves as the **starting point** for Bundle Adjustment.

---

## **3. Bundle Adjustment (Cameras 1 & 2)**

Before running BA, the code:

- Computes the **true scale** from the ground-truth baseline  
- Passes this to the optimizer so the BA result can be scaled to meters

The call:

```python
T_21_opt, X1_opt, _, _ = bundleAdjustment(...)
````

### **Returns of the Bundle Adjustment Step**

- **Optimized camera pose** between cameras 1 and 2  
- **Optimized 3D structure**  
- **Initial and final residuals** (for evaluation)

BA refines:

- Rotation \( R_{21} \)  
- Translation \( t_{21} \)  
- All 3D points \( X_1 \)  

making them consistent with the real image measurements.

---

# **4. PnP for Camera 3**

Once the 3D points are refined by BA, the pose of **camera 3** is estimated.

### **Key operations:**

- `objectPoints = X1_opt.T` → 3D world points  
- `imagePoints = ... reshape((N,1,2))` (OpenCV-required format)  
- Solve PnP using EPnP:
retval, rvec, tvec = cv2.solvePnP(...


- Convert rotation vector → rotation matrix using **Rodrigues**  
- Build full transformation:

\[
T_{31}
\]

This produces the **initial pose of camera 3** before multi-view BA.

---

# **5. Reprojection Error for Camera 3**

To evaluate the PnP pose:

- Transform 3D points using \( T_{31} \)  
- Project them with intrinsics \( K_c \)  
- Compare projected points with real image points  
- Compute RMS reprojection error  

This quantifies the **accuracy of the PnP pose estimate**.

---

# **6. Comparison with Ground Truth**

The script compares:

- Estimated camera 3 pose vs. ground truth  
- Optimized 3D points vs. ground truth 3D structure  

Validates correctness of:

- Essential matrix initialization  
- Bundle Adjustment  
- PnP estimation  

---

# **7. Visualization**

### **1. Reprojection in Image 3**
- Observed points  
- PnP-projected points  
- Residual vectors  

### **2. 3D Reconstruction Plot**

Shows:

- Camera 1 frame  
- Optimized camera 2 frame  
- PnP-estimated camera 3 frame  
- Ground truth camera 3 frame  
- 3D points (optimized and GT)

This visualization represents the **full geometric result of the reconstruction pipeline**.
<img width="1208" height="863" alt="image" src="https://github.com/user-attachments/assets/748551e7-7ca2-4464-b04c-a835bb709d51" />

<img width="755" height="841" alt="image" src="https://github.com/user-attachments/assets/8f60c0ab-9025-4d8a-bb8a-fef3655ef3c3" />


Reconstruction of the 3 cameras: 
**<img width="1021" height="834" alt="image" src="https://github.com/user-attachments/assets/1d56eeff-775e-4ebb-b38c-c90b29909856" />
**

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
<img width="1896" height="702" alt="image" src="https://github.com/user-attachments/assets/0f03f59d-1d57-4b77-8bc0-3db789524e27" />

<img width="892" height="810" alt="image" src="https://github.com/user-attachments/assets/e79a2923-9264-448a-8d11-420fa4d7f360" />

<img width="815" height="536" alt="image" src="https://github.com/user-attachments/assets/7fbbe76f-52dc-4344-b349-ad7f6a644d0a" />




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

