

import numpy as np
import torch
import torch.nn as nn


HALFLIVES = (2, 5, 10, 25, 50, 120)


def ewma_features(r, halflives=HALFLIVES):
    """log-EWMAs of squared returns at several half-lives, shifted so the feature
    at t uses returns only up to t-1 (one-step-ahead, no peeking).

    Short half-lives react fast to recent shocks; long ones carry the slow
    persistent component. Together they span the memory GARCH gets from beta.
    """
    r2 = np.asarray(r, dtype=np.float64) ** 2
    feats = []
    for hl in halflives:
        lam = 0.5 ** (1.0 / hl)          # decay giving this half-life
        out = np.empty_like(r2)
        acc = r2[0]
        for t in range(len(r2)):
            acc = lam * acc + (1 - lam) * r2[t]
            out[t] = acc
        feats.append(out)
    F = np.stack(feats, axis=1)
    F = np.vstack([F[:1], F[:-1]])       # shift down one: feature_t uses up to t-1
    return np.log(F + 1e-12).astype(np.float32)


class VolMLP(nn.Module):
    def __init__(self, n_feat, hidden=64, depth=2):
        super().__init__()
        layers = [nn.Linear(n_feat, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def gaussian_nll(log_v, r):
    """log v + r^2/v, averaged. Minimised in expectation at the true conditional
    variance, so training on returns targets the unobserved variance correctly."""
    return (log_v + r ** 2 * torch.exp(-log_v)).mean()


def train_mlp(r_train, r_val, halflives=HALFLIVES, hidden=64, depth=2,
              epochs=60, batch_size=256, lr=1e-3, scale=None,
              device="cpu", seed=0, verbose=True):
    """Train on EWMA features. Returns (model, scale). Returns are standardised by
    train std; the same scale is reused everywhere so units stay consistent."""
    torch.manual_seed(seed)
    if scale is None:
        scale = float(np.std(r_train))

    Ftr = ewma_features(np.asarray(r_train) / scale, halflives)
    Fva = ewma_features(np.asarray(r_val) / scale, halflives)
    ytr = (np.asarray(r_train, dtype=np.float32) / scale)
    yva = (np.asarray(r_val, dtype=np.float32) / scale)

    Ftr = torch.tensor(Ftr).to(device); ytr = torch.tensor(ytr).to(device)
    Fva = torch.tensor(Fva).to(device); yva = torch.tensor(yva).to(device)

    model = VolMLP(Ftr.shape[1], hidden, depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(Ftr)
    best_val = float("inf"); best_state = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = gaussian_nll(model(Ftr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            vloss = gaussian_nll(model(Fva), yva).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"   epoch {ep+1:2d}  train_nll={tot/n:+.4f}  val_nll={vloss:+.4f}"
                  f"{'  *' if vloss==best_val else ''}")

    model.load_state_dict(best_state)
    return model, scale


def forecast_mlp(model, r, scale, halflives=HALFLIVES, device="cpu"):
    """One-step-ahead conditional variance in original units, for every step."""
    model.eval()
    F = ewma_features(np.asarray(r) / scale, halflives)
    with torch.no_grad():
        log_v = model(torch.tensor(F).to(device)).cpu().numpy()
    return np.exp(log_v) * scale ** 2


if __name__ == "__main__":
    import time
    from simulators_1 import sim_garch, sim_heston, sim_regime
    from garch_1 import fit_garch, forecast_garch

    for name, gen in [("GARCH", sim_garch), ("Heston", sim_heston),
                      ("Regime", sim_regime)]:
        out = gen(80_000, seed=1)
        r, v = out[0], out[1]
        r_tr, r_va, r_te = r[:50000], r[50000:60000], r[60000:]
        v_te = v[60000:]

        t0 = time.time()
        model, scale = train_mlp(r_tr, r_va, epochs=60, seed=0, verbose=False)
        vhat = forecast_mlp(model, r_te, scale)
        mlp_corr = np.corrcoef(vhat, v_te)[0, 1]

        gfit = fit_garch(r_tr)
        gv = forecast_garch(r_te, gfit)
        g_corr = np.corrcoef(gv, v_te)[0, 1]

        win = "MLP" if mlp_corr > g_corr else "GARCH"
        print(f"[{name:6s}]  MLP corr={mlp_corr:.4f}   GARCH corr={g_corr:.4f}   "
              f"-> {win} wins   ({time.time()-t0:.0f}s)")
