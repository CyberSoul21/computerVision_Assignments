import numpy as np

def to_h(x):
    return np.hstack([x, np.ones((x.shape[0],1))])

def sampson_error(F, x1h, x2h):
    Fx1  = (F @ x1h.T).T
    Ftx2 = (F.T @ x2h.T).T
    num = np.sum(x2h * Fx1, axis=1)**2
    den = Fx1[:,0]**2 + Fx1[:,1]**2 + Ftx2[:,0]**2 + Ftx2[:,1]**2
    return num / den

def symmetric_epipolar_sqdist(F, x1h, x2h):
    l2 = (F  @ x1h.T).T
    l1 = (F.T@ x2h.T).T
    num = np.abs(np.sum(x2h * (F @ x1h.T).T, axis=1))
    d2 = num / np.sqrt(l2[:,0]**2 + l2[:,1]**2)
    d1 = num / np.sqrt(l1[:,0]**2 + l1[:,1]**2)
    return d1**2 + d2**2  # squared symmetric distance

def evaluate_F(F, x1, x2, sampson_thresh_px2=4.0, sym_thresh_px=2.0):
    x1h, x2h = to_h(x1), to_h(x2)

    # Normalize F for algebraic residual comparability (optional):
    Fn = F / np.linalg.norm(F)

    # Metrics
    samps = sampson_error(F, x1h, x2h)  # ~ px^2
    sym2  = symmetric_epipolar_sqdist(F, x1h, x2h)  # px^2
    alg   = np.abs(np.sum(x2h * ((Fn @ x1h.T).T), axis=1))  # scale-normalized residual

    # Summaries
    def stats(arr):
        return dict(mean=float(np.mean(arr)),
                    median=float(np.median(arr)),
                    p90=float(np.percentile(arr,90)))

    samps_stats = stats(samps)
    sym_stats   = stats(np.sqrt(sym2))  # report in px
    alg_stats   = stats(alg)

    inlier_ratio_samps = float(np.mean(samps < sampson_thresh_px2))
    inlier_ratio_sym   = float(np.mean(np.sqrt(sym2) < sym_thresh_px))

    return {
        "sampson_px2": samps_stats,
        "symmetric_px": sym_stats,
        "algebraic_normed": alg_stats,
        "inlier_ratio_sampson@%.1fpx2"%sampson_thresh_px2: inlier_ratio_samps,
        "inlier_ratio_symmetric@%.1fpx"%sym_thresh_px: inlier_ratio_sym,
    }