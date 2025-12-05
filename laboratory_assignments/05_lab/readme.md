#  Laboratory 5 — Kannala-Brandt Fisheye Model  
**MRGCV Unizar — Computer Vision**  
**Authors:** Wilson Javier Almario (962449), Diego Méndez (960616)

---

##  Overview  

This module implements the **projection** and **unprojection** for fisheye lenses using the  
**Kannala-Brandt distortion model**, enabling accurate omnidirectional geometry.

Functions included:

- `projectKannalaBrandt`
- `_newton_solve_theta`
- `unprojectKannalaBrandt`
- `testingKannalaBrandt`

---

#  1. `projectKannalaBrandt(Pc, K, D)`

### **Purpose**  
Projects 3D points in the camera reference frame into distorted fisheye pixel coordinates.

---

## 🔧 Pipeline  

### 1. Normalized coordinates  

  
$$
a = \frac{X}{Z}, \qquad b = \frac{Y}{Z}
$$
  

  
$$
r = \sqrt{a^2 + b^2}
$$
  


### 2. Fisheye angle  

  
$$
\theta = \arctan(r)
$$
  


### 3. Apply Kannala-Brandt distortion  

  
$$
\theta_d = \theta\left(1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8\right)
$$
  


### 4. Distorted normalized coordinates  

  
$$
x' = \frac{\theta_d}{r} a,  
\qquad  
y' = \frac{\theta_d}{r} b
$$
  


### 5. Pixel coordinates  

  
$$
u = f_x x' + c_x, \qquad v = f_y y' + c_y
$$
  


---

## Inputs  

| Parameter | Shape | Description |
|----------|--------|-------------|
| `Pc` | `(3,)` or `(N,3)` | 3D points in camera frame |
| `K` | `(3,3)` | Intrinsic matrix |
| `D` | `(4,)` | Distortion coefficients |

---

## Output  

| Output | Shape | Description |
|--------|--------|-------------|
| `uv` | `(2,)` or `(N,2)` | Pixel coordinates |

---

## Image Placeholder  

```
![Projection](images/projection.png)
```


---

## Source Code  

```python
def projectKannalaBrandt(Pc, K, D):
    ...
```

---

#  2. `_newton_solve_theta(rd, k1, k2, k3, k4)`

### **Purpose**  
Solves the inverse distortion problem using Newton–Raphson.

---

## 🔧 Pipeline  

Equation to solve:

  
$$
rd = \theta(1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8)
$$
  

Newton update:

  
$$
\theta_{t+1} = \theta_t - \frac{f(\theta_t)}{f'(\theta_t)}
$$
  


---

##  Inputs  

| Parameter | Description |
|----------|-------------|
| `rd` | Distorted radius |
| `k1..k4` | KB distortion coefficients |
| `max_iter` | Max iterations |
| `eps` | Convergence tolerance |

---

##  Output  

| Output | Description |
|--------|-------------|
| `theta` | Undistorted polar angle |

---

## Source Code  

```python
def _newton_solve_theta(rd, k1, k2, k3, k4, max_iter=10, eps=1e-9):
    ...
```

---

# 3. `unprojectKannalaBrandt(uv, K, D, ...)`

### **Purpose**  
Converts pixel coordinates into 3D camera-frame rays.

---

## 🔧 Pipeline  

### 1. Distorted normalized coords  

  
$$
x_d = \frac{u - c_x}{f_x},  
\qquad  
y_d = \frac{v - c_y}{f_y}
$$
  

  
$$
rd = \sqrt{x_d^2 + y_d^2}
$$
  


### 2. Recover undistorted angle  

  
$$
\theta = \text{NewtonSolve}(rd)
$$
  


### 3. Undistorted radius  

  
$$
r = \tan(\theta)
$$
  


### 4. Ray construction  

  
$$
a = x_d \frac{r}{rd},  
\qquad  
b = y_d \frac{r}{rd}
$$
  

  
$$
\text{ray} = (a, b, 1)
$$
  


### 5. Optional normalization  

  
$$
\hat{v} = \frac{v}{\|v\|}
$$
  


---

##  Inputs  

| Parameter | Shape | Description |
|----------|--------|-------------|
| `uv` | `(2,)` or `(N,2)` | Pixel coordinates |
| `K` | `(3,3)` | Camera intrinsics |
| `D` | `(4,)` | KB coefficients |
| `normalize` | bool | Normalize rays |

---

## Output  

| Output | Shape | Description |
|--------|--------|-------------|
| `rays` | `(3,)` or `(N,3)` | 3D direction vectors |

---

## Source Code  

```python
def unprojectKannalaBrandt(uv, K, D, max_iter=10, eps=1e-9, normalize=True):
    ...
```

---

#  4. `testingKannalaBrandt()`

### **Purpose**  
Runs a full validation pipeline with:

- Projection test  
- Unprojection test  
- Angular error computation  

---

## 🔧 Pipeline  

### 1. Load calibration (K, D)  

### 2. Load virtual 3D points  

### 3. Load ground-truth pixel projections  

### 4. Projection error  

  
$$
e_i = \| u_i^{pred} - u_i^{gt} \|
$$
  


### 5. Unprojection angular error  

  
$$
\alpha = \arccos(\langle \hat{X}, \hat{r} \rangle)
$$
  


---

## Source Code  

```python
def testingKannalaBrandt():
    ...
```

---
```
Kannala-Brandt Camera Model
K =
 [[283.98181152   0.         421.60400391]
 [  0.         284.94570923 395.45230103]
 [  0.           0.           1.        ]]
D = [-0.00233686  0.037855   -0.03575607  0.00599863]

=== Given pixels (ground truth) ===
[[503.387  450.1594]
 [267.9465 580.4671]
 [441.0609 493.0671]]

=== Projected pixels (our KB implementation) ===
[[503.38703145 450.1593793 ]
 [267.94653137 580.46712542]
 [441.06092003 493.06708726]]

=== Pixel reprojection error per point (in pixels) ===
Point X1: error = 0.000038 px
Point X2: error = 0.000040 px
Point X3: error = 0.000024 px

=== Unprojected rays (unit length) ===
[[ 0.28221615  0.18814425  0.94072089]
 [-0.47673139  0.57207747  0.66742381]
 [ 0.06711554  0.33557807  0.93961847]]

=== Angular error between original X direction and unprojected ray ===
Point 1: 0.00000754 degrees
Point 2: 0.00000714 degrees
Point 3: 0.00000468 degrees
==================================================

This module implements projection and unprojection for fisheye cameras.
```
----

#  References  

Kannala, J., & Brandt, S. (2006).  
_A generic camera model and calibration method for conventional, wide-angle, and fish-eye lenses._

