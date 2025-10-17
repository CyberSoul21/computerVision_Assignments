# lab1_utils.py
# Helpers for "Lab Session 1: 2D–3D Geometry in Homogeneous Coordinates & Camera Projection"
# Drop this file next to K.txt, R_w_c*.txt, t_w_c*.txt, Image*.jpg if you want.

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Iterable
import numpy as np


# ---------- I/O ----------
def load_txt_matrix(path: str, expected_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load a whitespace- or comma-separated numeric matrix from a .txt file."""
    arr = np.loadtxt(path, delimiter=None)
    if expected_shape and arr.shape != expected_shape:
        raise ValueError(f"{path}: expected shape {expected_shape}, got {arr.shape}")
    return arr.astype(float)


def maybe(path: str) -> Optional[np.ndarray]:
    """Try load; return None if missing."""
    try:
        return np.loadtxt(path, delimiter=None).astype(float)
    except Exception:
        return None


# ---------- Homogeneous helpers ----------
def to_h2(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p).reshape(-1, 2)
    return np.hstack([p, np.ones((p.shape[0], 1))])


def from_h2(ph: np.ndarray) -> np.ndarray:
    ph = np.asarray(ph)
    if ph.ndim == 1:
        ph = ph[None, :]
    w = ph[:, -1:]
    out = np.zeros((ph.shape[0], 2), dtype=float)
    mask = np.abs(w) > 1e-12
    out[mask[:, 0]] = ph[mask[:, 0], :2] / w[mask[:, 0]]
    out[~mask[:, 0]] = ph[~mask[:, 0], :2]
    return out


def normalize_line(l: np.ndarray) -> np.ndarray:
    l = np.asarray(l).reshape(-1)
    n = np.linalg.norm(l[:2])
    return l / n if n > 1e-12 else l


def normalize_plane(pi: np.ndarray) -> np.ndarray:
    pi = np.asarray(pi).reshape(-1)
    n = np.linalg.norm(pi[:3])
    return pi / n if n > 1e-12 else pi


# ---------- Camera & projection ----------
def K_from_params(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=float)


def P_from_KRt(K: np.ndarray, R_cw: np.ndarray, t_cw: np.ndarray) -> np.ndarray:
    """P = K [R|t]. Assumes X_c = R_cw X_w + t_cw (world->camera)."""
    Rt = np.hstack([R_cw, t_cw.reshape(3, 1)])
    return (K @ Rt).astype(float)


def project(P: np.ndarray, XYZ: np.ndarray) -> np.ndarray:
    XYZ = np.asarray(XYZ, dtype=float)
    if XYZ.shape[1] == 3:
        XYZ = np.hstack([XYZ, np.ones((XYZ.shape[0], 1))])
    xh = (P @ XYZ.T).T
    return from_h2(xh)


# ---------- Lines & vanishing points ----------
def line_through(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    p1h = p1 if p1.shape[-1] == 3 else np.append(p1, 1.0)
    p2h = p2 if p2.shape[-1] == 3 else np.append(p2, 1.0)
    return normalize_line(np.cross(p1h, p2h))


def intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    p = np.cross(l1, l2)
    return p / p[-1] if abs(p[-1]) > 1e-12 else p


def vanishing_point(P: np.ndarray, v_world: Iterable[float]) -> np.ndarray:
    v = np.asarray(list(v_world), dtype=float).reshape(3)
    X_inf = np.r_[v, 0.0]  # (vx,vy,vz,0)
    p = P @ X_inf
    return p / p[-1] if abs(p[-1]) > 1e-12 else p


# ---------- Line fitting by SVD ----------
def fit_line_svd(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    A = np.hstack([pts, np.ones((pts.shape[0], 1))])  # rows [x y 1]
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    l = Vt[-1]  # last row of V^T
    return normalize_line(l)


# ---------- Plane via SVD & distances ----------
def plane_from_points(points_3d: np.ndarray) -> np.ndarray:
    P3 = np.asarray(points_3d, dtype=float).reshape(-1, 3)
    M = np.hstack([P3, np.ones((P3.shape[0], 1))])
    U, S, Vt = np.linalg.svd(M, full_matrices=True)
    pi = Vt[-1]  # a,b,c,d
    return normalize_plane(pi)


def point_plane_distance(X: np.ndarray, pi: np.ndarray) -> float:
    X = np.asarray(X, dtype=float).reshape(3)
    pi = np.asarray(pi, dtype=float).reshape(4)
    num = abs(pi[0]*X[0] + pi[1]*X[1] + pi[2]*X[2] + pi[3])
    den = np.linalg.norm(pi[:3])
    return float(num / (den + 1e-12))


# ---------- Convenience loader for your files ----------
@dataclass
class CameraSet:
    K: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    R1: np.ndarray
    t1: np.ndarray
    R2: np.ndarray
    t2: np.ndarray


def load_cameras(folder: str = ".") -> CameraSet:
    """Reads K.txt, R_w_c1.txt, t_w_c1.txt, R_w_c2.txt, t_w_c2.txt (world->camera)."""
    import os
    K = load_txt_matrix(os.path.join(folder, "K.txt"), expected_shape=(3, 3))
    R1 = load_txt_matrix(os.path.join(folder, "R_w_c1.txt"), expected_shape=(3, 3))
    t1 = load_txt_matrix(os.path.join(folder, "t_w_c1.txt")).reshape(3, 1)
    R2 = load_txt_matrix(os.path.join(folder, "R_w_c2.txt"), expected_shape=(3, 3))
    t2 = load_txt_matrix(os.path.join(folder, "t_w_c2.txt")).reshape(3, 1)
    P1 = P_from_KRt(K, R1, t1)
    P2 = P_from_KRt(K, R2, t2)
    return CameraSet(K, P1, P2, R1, t1, R2, t2)
