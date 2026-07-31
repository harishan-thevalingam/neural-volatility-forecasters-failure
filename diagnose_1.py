

import numpy as np


def qlike(v_true, v_hat):
    """QLIKE divergence between true and predicted variance, elementwise.
    Zero iff equal, always nonnegative. Robust and standard for vol forecasts."""
    ratio = v_true / v_hat
    return ratio - np.log(ratio) - 1.0


def by_vol_decile(v_true, preds, n_bins=10):
    """Mean QLIKE per decile of TRUE volatility, for each model in `preds`
    (a dict name -> v_hat). Returns a table (list of dict rows)."""
    vol = np.sqrt(v_true)
    edges = np.quantile(vol, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-12
    binid = np.clip(np.digitize(vol, edges) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        m = binid == b
        row = {"decile": b + 1,
               "vol_lo": edges[b], "vol_hi": edges[b + 1],
               "n": int(m.sum())}
        for name, vh in preds.items():
            row[name] = float(np.mean(qlike(v_true[m], vh[m])))
        rows.append(row)
    return rows


def since_switch(v_true, preds, state, max_lag=15):
    """Mean QLIKE as a function of bars since the last regime change. Lag 0 is the
    switch bar itself. This is where a model that reacts slowly gets exposed."""
    switch = np.zeros(len(state), dtype=bool)
    switch[1:] = state[1:] != state[:-1]
    switch_idx = np.where(switch)[0]

    # for each step, bars since the most recent switch (capped at max_lag)
    since = np.full(len(state), max_lag + 1, dtype=int)
    last = -10**9
    for t in range(len(state)):
        if switch[t]:
            last = t
        since[t] = min(t - last, max_lag + 1)

    rows = []
    for lag in range(0, max_lag + 1):
        m = since == lag
        if m.sum() == 0:
            continue
        row = {"bars_since_switch": lag, "n": int(m.sum())}
        for name, vh in preds.items():
            row[name] = float(np.mean(qlike(v_true[m], vh[m])))
        rows.append(row)
    return rows, len(switch_idx)


def print_table(rows, cols, title):
    print(f"\n{title}")
    header = "  ".join(f"{c:>12s}" for c in cols)
    print("  " + header)
    for r in rows:
        cells = []
        for c in cols:
            val = r[c]
            if isinstance(val, float):
                cells.append(f"{val:12.4f}" if abs(val) < 1e4 else f"{val:12.2e}")
            else:
                cells.append(f"{val:>12}")
        print("  " + "  ".join(cells))


if __name__ == "__main__":
    from simulators_1 import sim_garch, sim_heston, sim_regime
    from garch_1 import fit_garch, forecast_garch
    from mlp_1 import train_mlp, forecast_mlp

    def prep(gen, seed=1):
        out = gen(80_000, seed=seed)
        r = out[0]; v = out[1]
        state = out[2] if len(out) > 2 else None
        return r, v, state

    # ---- Heston: where does the MLP's win come from? ----
    print("=" * 60)
    print("HESTON - MLP beats GARCH overall; which vol levels?")
    print("=" * 60)
    r, v, _ = prep(sim_heston)
    r_tr, r_va, r_te = r[:50000], r[50000:60000], r[60000:]
    v_te = v[60000:]

    model, scale = train_mlp(r_tr, r_va, epochs=60, seed=0, verbose=False)
    mlp_v = forecast_mlp(model, r_te, scale)
    gfit = fit_garch(r_tr)
    g_v = forecast_garch(r_te, gfit)

    preds = {"MLP": mlp_v, "GARCH": g_v}
    rows = by_vol_decile(v_te, preds)
    print_table(rows, ["decile", "vol_lo", "vol_hi", "n", "MLP", "GARCH"],
                "mean QLIKE by true-vol decile (lower = better):")
    tot_m = np.mean(qlike(v_te, mlp_v)); tot_g = np.mean(qlike(v_te, g_v))
    print(f"\n  overall QLIKE:  MLP {tot_m:.4f}   GARCH {tot_g:.4f}")

    # ---- Regime: does GARCH's loss live right after switches? ----
    print("\n" + "=" * 60)
    print("REGIME - near tie overall; is the action post-switch?")
    print("=" * 60)
    r, v, state = prep(sim_regime)
    r_tr, r_va, r_te = r[:50000], r[50000:60000], r[60000:]
    v_te = v[60000:]; state_te = state[60000:]

    model, scale = train_mlp(r_tr, r_va, epochs=60, seed=0, verbose=False)
    mlp_v = forecast_mlp(model, r_te, scale)
    gfit = fit_garch(r_tr)
    g_v = forecast_garch(r_te, gfit)

    preds = {"MLP": mlp_v, "GARCH": g_v}
    rows, nsw = since_switch(v_te, preds, state_te, max_lag=12)
    print(f"\n  {nsw} regime switches in test set")
    print_table(rows, ["bars_since_switch", "n", "MLP", "GARCH"],
                "mean QLIKE by bars since last switch:")
    tot_m = np.mean(qlike(v_te, mlp_v)); tot_g = np.mean(qlike(v_te, g_v))
    print(f"\n  overall QLIKE:  MLP {tot_m:.4f}   GARCH {tot_g:.4f}")
