

import numpy as np


def sim_garch(n, omega=2e-6, alpha=0.09, beta=0.90, seed=0, burn=500):
    """GARCH(1,1): v_t = omega + alpha r_{t-1}^2 + beta v_{t-1}.

    Best case for the GARCH baseline since the data obeys its assumed form.
    alpha+beta=0.99 is the realistic high persistence.
    """
    rng = np.random.default_rng(seed)
    total = n + burn

    r = np.zeros(total)
    v = np.zeros(total)
    v[0] = omega / (1 - alpha - beta)  # unconditional variance

    z = rng.standard_normal(total)
    for t in range(1, total):
        v[t] = omega + alpha * r[t - 1] ** 2 + beta * v[t - 1]
        r[t] = np.sqrt(v[t]) * z[t]

    return r[burn:], v[burn:]


def sim_heston(n, mu_v=2e-6, kappa=0.02, theta=2e-6, xi=3e-4, rho=-0.7,
               seed=0, burn=500):
    """Discrete Heston stochastic vol. Variance has its own noise, correlated
    with return noise by rho.

        v_t = v_{t-1} + kappa(theta - v_{t-1}) + xi sqrt(v_{t-1}) w_t
        r_t = sqrt(v_t) z_t,   corr(w, z) = rho

    rho<0 is the leverage effect (down moves raise vol). Neither model has the
    true form here, but we still know v_t. Variance floored so the sqrt stays real.
    """
    rng = np.random.default_rng(seed)
    total = n + burn

    z = rng.standard_normal(total)
    eps = rng.standard_normal(total)
    w = rho * z + np.sqrt(1 - rho ** 2) * eps  # corr(w, z) = rho

    r = np.zeros(total)
    v = np.zeros(total)
    v[0] = theta

    vmin = theta * 1e-3
    for t in range(1, total):
        v[t] = v[t - 1] + kappa * (theta - v[t - 1]) + xi * np.sqrt(v[t - 1]) * w[t]
        if v[t] < vmin:
            v[t] = vmin
        r[t] = np.sqrt(v[t]) * z[t]

    return r[burn:], v[burn:]


def sim_regime(n, seed=0, burn=500,
               p_stay_calm=0.995, p_stay_turb=0.97,
               vol_calm=0.008, vol_turb=0.03,
               garch_alpha=0.08, garch_beta=0.90):
    """Two GARCH regimes (calm, turbulent) with a Markov chain switching between
    them. Calm is sticky, turbulent is shorter and higher vol.

    Manufactures the sudden regime shifts neural forecasters fumble; we know the
    switch times. Returns (r, v, state), state being the regime label for binning
    errors by time-since-switch.
    """
    rng = np.random.default_rng(seed)
    total = n + burn

    persist = garch_alpha + garch_beta
    omega = np.array([vol_calm ** 2 * (1 - persist),   # per-regime baseline
                      vol_turb ** 2 * (1 - persist)])

    r = np.zeros(total)
    v = np.zeros(total)
    state = np.zeros(total, dtype=int)  # 0 calm, 1 turbulent

    v[0] = vol_calm ** 2
    u = rng.random(total)
    z = rng.standard_normal(total)

    for t in range(1, total):
        s = state[t - 1]
        if s == 0:
            state[t] = 0 if u[t] < p_stay_calm else 1
        else:
            state[t] = 1 if u[t] < p_stay_turb else 0

        v[t] = omega[state[t]] + garch_alpha * r[t - 1] ** 2 + garch_beta * v[t - 1]
        r[t] = np.sqrt(v[t]) * z[t]

    return r[burn:], v[burn:], state[burn:]


def summary(r, v, name=""):
    """Stylised-fact check. Wrong here means the downstream comparison is
    measuring the wrong thing."""
    r = np.asarray(r)
    ac1_r = np.corrcoef(r[1:], r[:-1])[0, 1]
    ac1_r2 = np.corrcoef(r[1:] ** 2, r[:-1] ** 2)[0, 1]
    kurt = ((r - r.mean()) ** 4).mean() / r.var() ** 2

    print(f"[{name}] n={len(r)}")
    print(f"   mean return        {r.mean():+.2e}")
    print(f"   vol per step       {r.std():.4f}")
    print(f"   autocorr(r)        {ac1_r:+.4f}   (want ~0)")
    print(f"   autocorr(r^2)      {ac1_r2:+.4f}   (want >0, clustering)")
    print(f"   kurtosis           {kurt:.2f}   (want >3, heavy tails)")
    print(f"   true vol range     [{np.sqrt(v).min():.4f}, {np.sqrt(v).max():.4f}]")


if __name__ == "__main__":
    n = 100_000

    r, v = sim_garch(n, seed=1)
    summary(r, v, "GARCH")

    r, v = sim_heston(n, seed=1)
    summary(r, v, "Heston")

    r, v, s = sim_regime(n, seed=1)
    summary(r, v, "Regime")
    print(f"   turbulent fraction {s.mean():.3f}")
    switches = np.sum(np.abs(np.diff(s)))
    print(f"   regime switches    {switches}  (avg spell {len(s)/max(switches,1):.0f} steps)")
