# 🧠 Computer Vision – Laboratory 1  
## 2D–3D Geometry in Homogeneous Coordinates and Camera Projection  

**Course:** Computer Vision (MRGCV – University of Zaragoza)  
**Authors:** 
Wilson Javier Almario Rodriguez & Diego Méndez Carter  
**Based on materials by:** J. Bermúdez, R. Elvira, J. Lamarca, J.M.M. Montiel  
**Date:** October 2025  

---

## 🎯 Objectives

The goal of this lab is to introduce and practice fundamental concepts of **projective geometry** and **camera modeling**.  
Specifically, the session focuses on understanding how 3D points are represented and projected onto 2D image planes through **homogeneous coordinates** and **camera projection matrices**.

### Main objectives:
1. Practice with **homogeneous coordinates** to represent 2D and 3D geometrical entities.  
2. Understand and implement **perspective projection** of 3D points into 2D images.  
3. Use **Singular Value Decomposition (SVD)** to solve homogeneous systems of linear equations.  
4. Apply these tools to compute **2D lines, vanishing points, and planes in 3D**.

---

## 🧩 Part 1 – 3D to 2D Projection

- The calibration matrix **K**, rotation **R**, and translation **t** of two cameras are provided.  
- Projection matrices **P₁** and **P₂** are computed as:

  \[
  P = K [R|t]
  \]

- The 3D points **A, B, C, D, E** are projected onto both camera images.  
- The resulting 2D coordinates are plotted on each image using Matplotlib.

**Result:** Correct projection of 3D world points onto both camera views.  

---

## 🧮 Part 2 – 2D Lines and Vanishing Points

- Using the projected points, the lines \( l_{AB} \) and \( l_{CD} \) are computed as:
  \[
  l = x_1 \times x_2
  \]
- The intersection \( p_{12} = l_{AB} \times l_{CD} \) gives the **vanishing point**.
- The **3D direction** between points A and B is projected to obtain the **vanishing point at infinity**.

**Result:**  
- Visualization of both lines and their intersection on the image plane.  
- Consistent vanishing point aligned with the 3D direction AB.

---

## 📉 Part 3 – Line Fitting with SVD

- The **SVD** method is applied to fit a line to a set of 2D points.

### 3.1 Using 2 extreme points
- The matrix \( A = x^T \) (2×3) is decomposed using SVD.
- The last column of \( V \) gives the line coefficients.

### 3.2 Using 5 perfect points
- The matrix \( A = x_{GT}^T \) (5×3) is used with perfect points.
- The resulting line coincides with the ground truth \( l_{GT} \).

### 3.3 Singular values interpretation
- The smallest singular value ≈ 0 → indicates perfect collinearity (no noise).

### 3.4 Forcing smallest singular value to 0
- Recomposition with \( s[-1] = 0 \) removes perpendicular noise.  
- The reconstructed matrix represents **points projected exactly onto the fitted line**.

---

## 🌍 Part 4 – Plane Fitting in 3D

- The **3D plane equation** passing through points A, B, C, D is found using SVD:
  \[
  \pi = [a, b, c, d], \quad aX + bY + cZ + d = 0
  \]
- The result corresponds to a horizontal plane \( z = 0.82 \, m \).  
- The distance from each point to the plane is computed as:
  \[
  d_i = \frac{|a x_i + b y_i + c z_i + d|}{\sqrt{a^2 + b^2 + c^2}}
  \]

**Result:**  
- Points A, B, C, D lie exactly on the plane (distance = 0).  
- Point E is ≈ 0.39 m above the plane.

---

## 📷 Tools and Libraries

- **Python 3.10+**
- **NumPy** – matrix and vector operations  
- **Matplotlib** – data visualization and image plotting  
- **OpenCV (cv2)** – image loading and color conversion  
- **SciPy.linalg** – diagonal SVD handling and reconstruction  

---

## 🧩 Key Learning Outcomes

- How to build and manipulate **camera projection matrices**.  
- Understanding of **homogeneous transformations** between 2D and 3D spaces.  
- Use of **SVD for solving linear systems** and **data fitting (lines, planes)**.  
- Practical understanding of **geometric relationships** (points, lines, vanishing points, planes).

---

## 🖼️ Example Outputs

| Section | Output Description |
|----------|--------------------|
| Part 1 | Projection of 3D points A–E on Image 1 & Image 2 |
| Part 2 | Lines \( l_{AB}, l_{CD} \) and intersection \( p_{12} \) |
| Part 3 | Line fitting comparison: Ground truth vs SVD |
| Part 4 | Computed plane \( \pi = [0, 0, -1, 0.82] \) and point distances |

---

## 🧾 References

- MRGCV Course Notes (Computer Vision, University of Zaragoza)  
- Multiple View Geometry – R. Hartley & A. Zisserman  
- Practical exercises adapted from Unizar Vision Lab (JMM Montiel et al.)

---

> 🧩 *"Homogeneous coordinates are the language of geometry —  
understanding them means seeing 3D structure behind every image."*

