# When and why do neural volatility forecasters break?

A controlled diagnosis, not a horse race. The recurring footnote across the
deep-learning-vs-GARCH volatility literature is that neural models fail in
high-volatility / regime-shift periods. This isolates it.

## Idea

Volatility is latent - on real data you never see the true conditional variance,
only one noisy return draw, so you can't score a forecast against truth. So we
simulate processes where we set sigma_t^2 ourselves, forecast with both a GARCH
baseline and a feedforward neural model given the same information, and measure
error against the known truth conditioned on the latent state.

## Files (run in this order)

    simulators_1.py    GARCH / Heston / regime-switching, each with known variance
    garch_1.py         GARCH(1,1) MLE baseline
    mlp_1.py           feedforward forecaster on EWMA features, Gaussian-NLL trained
    diagnose_1.py      error binned by true-vol decile and by bars-since-switch
    scaling_test_1.py  is the high-vol gap data-limited or architecture-limited?

    pip install -r requirements.txt
    python simulators_1.py
    python garch_1.py
    python mlp_1.py
    python diagnose_1.py
    python scaling_test_1.py

CPU-friendly throughout - seconds to a few minutes per step.

## Findings

Correlation with true variance:

    GARCH data:  MLP 0.998   GARCH 1.000   (MLP matches the true model)
    Heston:      MLP 0.829   GARCH 0.797   (MLP wins - GARCH misspecified)
    Regime:      MLP 0.981   GARCH 0.982   (tie on average)

Binning past the averages:

  - Heston: the MLP's win is entirely in high-vol deciles, where the leverage
    effect (down moves raise vol) is strongest and GARCH's symmetric r^2 term
    is blind to sign.
  - Regime: the on-average tie hides a real MLP advantage in the ~10 bars after
    a switch, cancelled by a small penalty in calm periods.

Falsifiable step: on Heston, scaling training data 50x leaves the top-decile
error flat (0.110 -> 0.114). The high-vol gap is a STRUCTURAL ceiling, not a data
shortage - the fix is a better model class, not more data.

## Notes

- Feedforward, not LSTM, for CPU speed. EWMAs of squared returns at several
  half-lives give the geometric-decay memory GARCH gets from its beta term, so
  the comparison stays fair - same information set, different functional form.
- QLIKE against the true variance is the error metric.
- Numbers are seed-dependent; rerun across seeds and quote the pattern, not any
  single decimal.
