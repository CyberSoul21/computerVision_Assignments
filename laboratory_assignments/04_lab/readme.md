# **Laboratory Session 4 — Bundle Adjustment & Multiview Geometry**  
*(Computer Vision — UNIZAR)*  
:contentReference[oaicite:0]{index=0}

---

## **Table of Contents**
1. [General Description](#general-description)  
2. [Objectives of the Laboratory](#objectives-of-the-laboratory)  
3. [Prerequisites and Provided Material](#prerequisites-and-provided-material)  
4. [1. Line Fitting with Least Squares](#1-line-fitting-with-least-squares)  
   - [1.1 Theoretical Concepts](#11-theoretical-concepts)  
   - [1.2 Implementation](#12-implementation)  
   - [1.3 Results and Visualizations](#13-results-and-visualizations)  
5. [2. Multiview Geometry and Pose Estimation](#2-multiview-geometry-and-pose-estimation)  
   - [2.1 Initial Reconstruction from Two Views](#21-initial-reconstruction-from-two-views)  
   - [2.2 Residual Function](#22-residual-function)  
   - [2.3 Bundle Adjustment (Two Views)](#23-bundle-adjustment-two-views)  
   - [2.4 Comparison with Ground Truth](#24-comparison-with-ground-truth)  
6. [3. Perspective-n-Point (PnP) for View 3](#3-perspective-n-point-pnp-for-view-3)  
7. [4. Bundle Adjustment with Three Views](#4-bundle-adjustment-with-three-views)  
8. [Appendix A — Rotation Representation in SO(3)](#appendix-a--rotation-representation-in-so3)  
9. [References](#references)

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
 [-0.30234655, -0.71224975, 100.]
]
```

Script `plotGroundTruth.py` for 3D visualization.

Code from Laboratory Session 2 for initial pose estimation.

Ground truth camera poses, 3D points, and scale information.

---

# **1. Line Fitting with Least Squares**

## **1.1 Theoretical Concepts**

This section should explain:

- What residuals are in an optimization problem  
- Why the least-squares method minimizes **resᵀ res**  
- The idea of "best fit" through squared error  
- Minimal number of parameters to describe a 2D line  
- Geometric interpretation of the residual as point–line distance  

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

---

## **2.2 Residual Function**

Template:

```python
def resBundleProjection(Op, x1Data, x2Data, K_c, nPoints):
    """
    Residuals between predicted projections and measured 2D points.
    """
```



