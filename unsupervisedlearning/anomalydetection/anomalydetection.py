import numpy as np

# ============================================================
# STEP 1: Estimate Gaussian parameters (mean, var) per feature
# ============================================================
def estimate_gaussian(X):
    """
    X: shape (m, n) — m examples, n features (assumed mostly normal)
    Returns:
        mu:  shape (n,)
        var: shape (n,)
    """
    mu = np.mean(X, axis=0) # calculate mean of each feature
    var = np.var(X, axis=0) # calculate variance of each feature
    return mu, var


# ============================================================
# STEP 2: Compute p(x) — in log-space to avoid underflow
# ============================================================
def multivariate_gaussian_logp(X, mu, var):
    """
    Log probability density of X under the product-of-1D-Gaussians model.

    log p(x) = sum_j [ -0.5*log(2*pi*var_j) - 0.5*(x_j - mu_j)^2 / var_j ]
    """
    return np.sum(
        -0.5 * np.log(2 * np.pi * var)
        - 0.5 * ((X - mu) ** 2) / var,
        axis=1
    )


def multivariate_gaussian_p(X, mu, var):
    """Actual probability p(x) = exp(log p(x))"""
    return np.exp(multivariate_gaussian_logp(X, mu, var))


# ============================================================
# STEP 3: Select epsilon by maximizing F1 on the CV set
# ============================================================
def compute_metrics(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)

    return precision, recall, f1, tp, fp, fn, tn


def select_epsilon(p_val, y_val, n_steps=1000):
    """
    Sweep thresholds on CV probabilities, keep the one with best F1.
    p_val: p(x) for CV examples (or log p(x) — both work)
    y_val: 1 = anomaly, 0 = normal
    """
    best_eps, best_f1 = 0.0, 0.0

    for eps in np.linspace(np.min(p_val), np.max(p_val), n_steps):
        preds = (p_val < eps).astype(int)          # True = flag as anomaly
        _, _, f1, tp, fp, fn, _ = compute_metrics(y_val, preds)

        if tp + fp == 0 or tp + fn == 0:           # skip degenerate thresholds
            continue

        if f1 > best_f1:
            best_f1, best_eps = f1, eps

    return best_eps, best_f1


# ============================================================
# STEP 4: Feature engineering helpers (Gaussianize + ratios)
# ============================================================
def transform_feature(x, kind="log", c=1e-3, p=0.5):
    """Concave transforms to make skewed features roughly Gaussian."""
    if kind == "log":
        return np.log(x + c)
    if kind == "pow":
        return x ** p
    raise ValueError("kind must be 'log' or 'pow'")


def make_ratio_feature(num, denom):
    """
    Ratio feature to capture correlated anomalies
    e.g. x5 = CPU load / network traffic
    """
    denom = np.where(denom == 0, 1e-8, denom)
    return num / denom


# ============================================================
# STEP 5: Demo: anomaly detection on engine data
# ============================================================
if __name__ == "__main__":
    # ----------------------------------------------------
    # Demo data: 2 features — heat and vibration of engines
    # Mostly normal engines + a few anomalies
    # ----------------------------------------------------
    rng = np.random.default_rng(42)
    m_normal = 600

    # Normal engines: heat ~ N(200, 15), vibration ~ N(50, 8), slightly correlated
    heat = rng.normal(200, 15, m_normal)
    vib = 0.3 * heat + rng.normal(-10, 6, m_normal)

    X_normal = np.column_stack([heat, vib])

    # A few anomalous engines (way off in vibration or heat)
    X_anom = np.array([
        [200, 120],   # high vibration
        [320, 50],    # overheating
        [210, 5],     # suspiciously low vibration
    ])

    # ----------------------------------------------------
    # 1) Split: train on normal-only (unlabeled)
    # ----------------------------------------------------
    X_train = X_normal[:400]
    X_cv = np.vstack([X_normal[400:], X_anom])          # CV: normal + anomalies
    y_cv = np.array([0] * 200 + [1] * 3)                 # labels used ONLY for tuning

    # ----------------------------------------------------
    # 2) Fit Gaussian parameters on training data
    # ----------------------------------------------------
    mu, var = estimate_gaussian(X_train)
    print("mu  :", mu)
    print("var :", var)

    # ----------------------------------------------------
    # 3) Compute p(x) on CV set (log-space, then exp)
    # ----------------------------------------------------
    p_cv = multivariate_gaussian_p(X_cv, mu, var)

    # ----------------------------------------------------
    # 4) Tune epsilon with F1
    # ----------------------------------------------------
    epsilon, best_f1 = select_epsilon(p_cv, y_cv)
    print(f"\nBest epsilon : {epsilon:.6e}")
    print(f"Best F1      : {best_f1:.4f}")

    # ----------------------------------------------------
    # 5) Predict & evaluate
    # ----------------------------------------------------
    y_pred = (p_cv < epsilon).astype(int)
    prec, rec, f1, tp, fp, fn, tn = compute_metrics(y_cv, y_pred)

    print(f"\nPrecision: {prec:.3f}  ({tp}/{tp+fp} flagged were real)")
    print(f"Recall   : {rec:.3f}  ({tp}/{tp+fn} anomalies caught)")
    print(f"F1       : {f1:.3f}")

    # ----------------------------------------------------
    # 6) Predict a brand-new engine
    # ----------------------------------------------------
    x_test = np.array([[210, 130]])   # should look anomalous
    p_test = multivariate_gaussian_p(x_test, mu, var)[0]
    print(f"\nNew engine p(x) = {p_test:.3e}  ->  "
          f"{'ANOMALY (inspect!)' if p_test < epsilon else 'normal'}")
