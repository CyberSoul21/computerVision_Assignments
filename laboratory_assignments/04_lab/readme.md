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

The goal is to understand how real 3D reconstruction systems based on epipolar geometry and nonlinear optimization are built.

---

## **Objectives of the Laboratory**
- Understand the relationship between multiple views and how to retrieve relative camera poses.  
- Implement residual functions for nonlinear optimization problems.  
- Execute Bundle Adjustment to improve initial geometric estimates.  
- Integrate a third view using OpenCV solvePnP and extend the BA formulation.  

---

## **Prerequisites and Provided Material**
- 35 perfectly matched keypoints across three images.  
- Example fundamental matrix:

```python
F_21 = [
 [0.00022244, 0.000624  , 0.10418026],
 [-0.00015211, -0.00004897, 0.60030525],
 [-0.30234655, -0.712 24975, 100.]
]
```

Script `plotGroundTruth.py` for 3D visualization.

Code from Laboratory Session 2 for initial pose estimation.

Ground truth camera poses, 3D points, and scale information.

---

# **1. Line Fitting with Least Squares**

## **1.1 Theoretical Concepts**

The purpose of this first exercise is to introduce the mathematical foundations of
optimization-based estimation, as used later in Bundle Adjustment. The following
concepts are essential:

---

### **Residuals in Optimization**

A residual represents the discrepancy between an observed measurement and the
prediction made by a model:


$$
r_i = y_i - \hat{y}_i 
$$


Residuals quantify how well the model explains the data. In optimization-based
methods, residuals are the quantities that the solver attempts to minimize.

---

### **Least-Squares Minimization**

Least-squares aims to find the parameters that minimize the sum of squared residuals:

$$
\min \sum_i r_i^2 = r^\top r
$$

This formulation is widely used because:

- It yields a smooth and differentiable cost function  
- It penalizes large deviations more strongly  
- It corresponds to the Maximum Likelihood estimator under Gaussian noise  
- It has a clear geometric interpretation in Euclidean space  

---

### **“Best Fit” Interpretation**

The best-fitting model is the one that minimizes the overall squared distance
between the predicted values and the observed data points. This ensures:

- Stability against noise  
- Reduction of measurement errors  
- Statistically unbiased estimation under Gaussian assumptions  

---

### **Minimal Parametrization of a 2D Line**

A line in the image plane can be expressed minimally using two parameters:

- **m** – slope  
- **b** – intercept  

Through:

$$
y = m x + b
$$

Other representations such as \( ax + by + c = 0 \) exist, but they introduce an
additional scale ambiguity and are therefore not minimal.

---

### **Geometric Meaning of a Residual**

A residual in this context represents the **vertical distance** between an observed
2D point and the line predicted by the model:

$$
r_i = y_i - (m x_i + b)
$$

Minimizing these distances forces the fitted line to pass as close as possible to
all measurements.

---

### **Motivation for This Exercise**

This simple line-fitting problem introduces:

- The structure of residual-based optimization  
- Parameter vector construction  
- How Levenberg–Marquardt adjusts parameters  
- The mathematical framework later used in full Bundle Adjustment  

Although the geometric model is simple, it is based on the same principles used to
optimize camera poses and 3D structures in multiview geometry.



---

## **1.2 Implementation**

Steps to document:

1. Load the 2D points `xData`.  
2. Define initial model parameters `Op`.  
3. Implement `resLineFitting(Op, xData)` returning a vector of residuals.  
4. Call `least_squares` using Levenberg–Marquardt.  
5. Analyze convergence and optimization behavior.  

Example:

```python
OpOptim = scOptim.least_squares(
    resLineFitting,
    Op,
    args=(xData,),
    method='lm'
)
```
## **1.3 Results and Visualizations**

Example figure:

![LineFit](figs/line_fit.png)

---

# **2. Multiview Geometry and Pose Estimation**

## **2.1 Initial Reconstruction from Two Views**

You should document:

- How to compute the essential matrix:  
  **E = Kᵀ F K**
- Constraints of the essential matrix (two equal singular values)  
- Recovery of **R** and **t** through SVD decomposition  
- Four-solution ambiguity  
- Correct solution selection using **cheirality**  
- Initial linear triangulation of points  

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
*(Insert when available)*

---

## **2.4 Comparison with Ground Truth**

### **Include:**

- Plot of estimated vs. true camera poses  
- Plot of reconstructed vs. true 3D points  
- Numerical error analysis  

**Example figure:**  
*(Insert when available)*

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

