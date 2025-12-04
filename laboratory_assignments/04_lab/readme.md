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
