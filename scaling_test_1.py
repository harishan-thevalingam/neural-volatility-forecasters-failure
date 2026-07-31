"""
The falsifiable core: is the MLP's high-vol edge on Heston data-limited or structural?

High-vol states are rare, so a fixed training set has few tail examples. Two hypotheses:

  statistical  - the model just lacks tail examples. Prediction: grow the training
                 set and the top-decile QLIKE keeps falling toward the noise floor.
  structural   - the architecture can't represent the tail mapping regardless of data.
                 Prediction: top-decile QLIKE plateaus above the floor.

Simulation is what makes this answerable: we generate training sets across orders of
magnitude, hold the test set fixed, and watch the top-decile error. Decay vs plateau
is the whole result.

Noise floor: even a perfect forecaster scores QLIKE>0 here only from finite-sample
estimation - with the TRUE variance the QLIKE is 0 by construction, so the floor we
compare against is 0, and "closes" means trends toward 0.
"""

import numpy as np
from simulators_1 import sim_heston
from mlp_1 import train_mlp, forecast_mlp
from diagnose_1 import qlike


def top_decile_qlike(v_true, v_hat):
    vol = np.sqrt(v_true)
    thr = np.quantile(vol, 0.9)
    m = vol >= thr
    return float(np.mean(qlike(v_true[m], v_hat[m])))


def run(train_sizes=(5000, 15000, 40000, 100000, 250000), seed=0):
    # fixed, large test set from a held-out seed - same target throughout so the
    # only thing changing is training size
    rte, vte = sim_heston(60_000, seed=999)
    rte, vte = rte[10000:], vte[10000:]   # drop warmup region

    # a big validation set, also fixed
    rva, _ = sim_heston(15_000, seed=888)

    print(f"{'train_size':>12}  {'overall_QL':>12}  {'top10%_QL':>12}")
    results = []
    for D in train_sizes:
        rtr, _ = sim_heston(D + 2000, seed=seed)
        model, scale = train_mlp(rtr, rva, epochs=60, seed=seed, verbose=False)
        vhat = forecast_mlp(model, rte, scale)
        overall = float(np.mean(qlike(vte, vhat)))
        top = top_decile_qlike(vte, vhat)
        results.append((D, overall, top))
        print(f"{D:12d}  {overall:12.4f}  {top:12.4f}")

    print("\ninterpretation:")
    tops = [t for _, _, t in results]
    drop = (tops[0] - tops[-1]) / tops[0] * 100
    tail_drop = (tops[-2] - tops[-1]) / tops[-2] * 100
    print(f"  top-decile QLIKE fell {drop:.0f}% from smallest to largest train set")
    print(f"  last doubling changed it by {tail_drop:+.1f}%")
    if abs(tail_drop) < 3:
        print("  -> flattening: consistent with a STRUCTURAL ceiling")
    else:
        print("  -> still falling: consistent with a DATA-LIMITED regime")


if __name__ == "__main__":
    run()
