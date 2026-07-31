

import numpy as np
from scipy.optimize import minimize


def _recursion(r, omega, alpha, beta):
    """Filter the return series to conditional variances v_t. v_0 seeded at
    the sample variance, which is the usual choice."""
    T = len(r)
    v = np.empty(T)
    v[0] = np.var(r)
    for t in range(1, T):
        v[t] = omega + alpha * r[t - 1] ** 2 + beta * v[t - 1]
    return v


def _unpack(params):
    """Map 3 unconstrained reals -> valid (omega, alpha, beta).

    omega via softplus so it's >0. alpha, beta via a softmax-style map into the
    simplex interior scaled by 0.999, so alpha+beta<1 (stationarity) always holds
    and the optimiser can't wander into an explosive region.
    """
    raw_omega, a, b = params
    omega = np.log1p(np.exp(raw_omega)) + 1e-12
    ea, eb = np.exp(a), np.exp(b)
    denom = 1 + ea + eb
    alpha = 0.999 * ea / denom
    beta = 0.999 * eb / denom
    return omega, alpha, beta


def _neg_loglik(params, r):
    omega, alpha, beta = _unpack(params)
    v = _recursion(r, omega, alpha, beta)
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(v) + r ** 2 / v)
    return -ll


def fit_garch(r, restarts=4, seed=0):
    """MLE fit. A few random restarts because the surface can have flat spots;
    keep the best. Returns dict with params and the per-step in-sample variance."""
    rng = np.random.default_rng(seed)
    r = np.asarray(r, dtype=float)

    best = None
    for i in range(restarts):
        x0 = rng.normal(0, 1, size=3) if i else np.array([-6.0, -1.0, 2.0])
        res = minimize(_neg_loglik, x0, args=(r,), method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res

    omega, alpha, beta = _unpack(best.x)
    v = _recursion(r, omega, alpha, beta)
    return {
        "omega": omega, "alpha": alpha, "beta": beta,
        "persistence": alpha + beta,
        "neg_loglik": best.fun,
        "v_in_sample": v,
    }


def forecast_garch(r, fit):
    """One-step-ahead conditional variance for every t, using fitted params.
    This is what gets scored against the true v_t in the diagnostic.

    Note forecast[t] uses r[t-1] and v[t-1], so it's a genuine one-step-ahead
    prediction made from information available at t-1 - no peeking."""
    return _recursion(np.asarray(r, dtype=float),
                      fit["omega"], fit["alpha"], fit["beta"])


if __name__ == "__main__":
    from simulators_1 import sim_garch, sim_heston, sim_regime

    print("fitting on GARCH data (should recover omega~2e-6, alpha~0.09, beta~0.90)")
    r, v = sim_garch(60_000, seed=1)
    fit = fit_garch(r)
    print(f"   omega={fit['omega']:.3e}  alpha={fit['alpha']:.4f}  "
          f"beta={fit['beta']:.4f}  persist={fit['persistence']:.4f}")

    # correlation between fitted and true variance - the honest accuracy measure
    vf = forecast_garch(r, fit)
    corr = np.corrcoef(vf, v)[0, 1]
    print(f"   corr(fitted v, true v) = {corr:.4f}")

    print("\nfitting on Heston data (misspecified - no true GARCH params exist)")
    r, v = sim_heston(60_000, seed=1)
    fit = fit_garch(r)
    vf = forecast_garch(r, fit)
    print(f"   persist={fit['persistence']:.4f}  corr(fitted, true)={np.corrcoef(vf, v)[0,1]:.4f}")

    print("\nfitting on Regime data (misspecified - switching, not single GARCH)")
    r, v, s = sim_regime(60_000, seed=1)
    fit = fit_garch(r)
    vf = forecast_garch(r, fit)
    print(f"   persist={fit['persistence']:.4f}  corr(fitted, true)={np.corrcoef(vf, v)[0,1]:.4f}")
