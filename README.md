
#Files

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



#Findings

Correlation with true variance:

    GARCH data:  MLP 0.998   GARCH 1.000   (MLP matches the true model)
    Heston:      MLP 0.829   GARCH 0.797   (MLP wins - GARCH misspecified)
    Regime:      MLP 0.981   GARCH 0.982   (tie on average)


  - Heston: the MLP's win is entirely in high-vol deciles, where the leverage
    effect (down moves raise vol) is strongest and GARCH's symmetric r^2 term
    is blind to sign.
  - Regime: the on-average tie hides a real MLP advantage in the ~10 bars after
    a switch, cancelled by a small penalty in calm periods.

