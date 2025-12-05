# **Laboratory Session 5 — Omnidirectional Vision**  
*(Computer Vision — UNIZAR)*  
:contentReference[oaicite:0]{index=0}

---

# **Table of Contents**

1. [General Description](#general-description)  
2. [Goals of the Assignment](#goals-of-the-assignment)  
3. [Provided Data](#provided-data)  
4. [1. Kannala–Brandt Projection & Unprojection Model](#1-kannala–brandt-projection--unprojection-model)  
   - [1.1 Theory Summary](#11-theory-summary)  
   - [1.2 Implementation Requirements](#12-implementation-requirements)  
   - [1.3 Validation Using Virtual Points](#13-validation-using-virtual-points)  
5. [2. Stereo Triangulation Using Fisheye Rays](#2-stereo-triangulation-using-fisheye-rays)  
   - [2.1 Unprojection to Rays](#21-unprojection-to-rays)  
   - [2.2 Plane-Based Triangulation](#22-plane-based-triangulation)  
   - [2.3 Results for Pose A](#23-results-for-pose-a)  
6. [3. Bundle Adjustment with Fisheye Stereo (Optional)](#3-bundle-adjustment-with-fisheye-stereo-optional)  
   - [3.1 Optimization Parameters](#31-optimization-parameters)  
   - [3.2 Residual Definition](#32-residual-definition)  
   - [3.3 Final Reconstruction](#33-final-reconstruction)  
7. [Appendix A — Rotation Representation in SO(3)](#appendix-a--rotation-representation-in-so3)  
8. [References](#references)

---

# **General Description**

In this laboratory session, you will implement a **non-linear omnidirectional projection model** capable of handling any radially symmetric fisheye camera.  
Specifically, you will work with the **Kannala–Brandt model**, used for reconstruction in stereo fisheye systems through:

- Unprojection into rays  
- Linear triangulation using plane intersections  
- Optional bundle adjustment  

Source: Laboratory Session 5 PDF :contentReference[oaicite:1]{index=1}

---

# **Goals of the Assignment**

1. Understand the non-linear **Kannala–Brandt projection/unprojection** model  
2. Triangulate 3D points from fisheye stereo using **ray-plane geometry**  
3. Adapt classical stereo algorithms to work with calibrated non-linear models  

---

# **Provided Data**

The lab includes the following files:

### **Camera intrinsics**
- `K_1.txt`, `K_2.txt` — intrinsic matrices  
- `D1_k_array.txt`, `D2_k_array.txt` — distortion parameters  
  - Polynomial:  
    θ_d = θ + k₁ θ³ + k₂ θ⁵ + k₃ θ⁷ + k₄ θ⁹  

### **Camera extrinsics**
- `T_wc1.txt` — pose of left camera wrt world  
- `T_wc2.txt` — pose of right camera wrt world  

### **Pose transformation between A and B**
- Ground truth: `T_wAwB_gt.txt`  
- Initial seed for BA: `T_wAwB_seed.txt`  

### **Point correspondences**
Pose A: `x1.txt`, `x2.txt`  
Pose B: `x3.txt`, `x4.txt`  

---

# **1. Kannala–Brandt Projection & Unprojection Model**

## **1.1 Theory Summary**

This section should describe:

- Mapping from **ray direction → pixel**  
- Non-linear distortion through odd polynomial terms  
- Relationship between the angle θ and distorted angle θ_d  
- Why linear pinhole projection fails in fisheye optics  

---

## **1.2 Implementation Requirements**

You must implement:

### **Projection (3D → pixel)**

Compute the distorted radius:

```markdown
θ_d = θ + k₁ θ³ + k₂ θ⁵ + k₃ θ⁷ + k₄ θ⁹
```

Then map to pixel coordinates using **K**.

---

# **Unprojection (pixel → normalized 3D ray)**

Given pixel coordinates **(u, v)**:

1. Convert pixel coordinates to the normalized plane  
2. Solve for **θ** using numerical inversion of the KB polynomial  
3. Convert **(θ, φ)** back into a unit 3D ray  

---

## **1.3 Validation Using Virtual Points**

Use the virtual test points provided in the PDF to verify:

unproject(project(X)) ≈ X / ||X||

Insert figure placeholder:

![KBModelValidation](figs/kb_validation.png)

---

# **2. Stereo Triangulation Using Fisheye Rays**

## **2.1 Unprojection to Rays**

For each pixel correspondence in **x1.txt**, **x2.txt**:

- Apply **Kannala–Brandt unprojection**  
- Express rays in world coordinates:

```python
ray_world = R_wc @ ray_cam + t_wc  
```
## **2.2 Plane-Based Triangulation**

Because the stereo rig is fully calibrated:

- Compute the **epipolar plane** defined by the two rays  
- Intersect both ray-defined planes to estimate the 3D point  

This is equivalent to computing the **closest point between two rays** expressed in the world frame.

---

## **2.3 Results for Pose A**

Insert placeholder for 3D point cloud:

![TriangulatedPointsA](figs/triangulation_poseA.png)

---

# **3. Bundle Adjustment with Fisheye Stereo (Optional)**

## **3.1 Optimization Parameters**

Optimize over:

- Pose transformation **T_wAwB**  
- 3D coordinates of all reconstructed points  

Treat as known parameters:

- **K₁**, **K₂**  
- Distortion parameters **D₁**, **D₂**  
- Stereo extrinsics **T_wc1**, **T_wc2**  

---

## **3.2 Residual Definition**

Residual = reprojection error under the **Kannala–Brandt model**:


## **3.2 Residual Definition**

Residual = reprojection error under the **Kannala–Brandt model**:

**Insert Python code here:**  
(residual = x_measured – project_KB(X_world, camera_pose, K, D))

---

## **3.3 Final Reconstruction**

Insert BA result placeholder here:

![BAReconstruction](figs/ba_reconstruction.png)

---

# **Appendix A — Rotation Representation in SO(3)**

Useful for parameterizing **T_wAwB** in bundle adjustment.

### **Skew-symmetric operator**

**Insert Python code here:**  
(definition of crossMatrix(x))

### **Concepts to explain**

- Vector **θ** and the skew-symmetric matrix **[θ]×**  
- Exponential map:  
  **R = exp([θ]×)**  
- Logarithmic map:  
  **θ = log(R)** via `scipy.linalg.logm`  
- Why **float64** is required for numerical stability in BA  

---

# **References**

- Laboratory Session 5 PDF  
- *Multiple View Geometry in Computer Vision* — Hartley & Zisserman  
- SciPy documentation — `least_squares`  
- Kannala & Brandt (2006) — *A Generic Camera Model for Fisheye Lenses*


