# validate/calibration_test.py

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import sem
from sklearn.utils import check_array
from validate.results import CalibrationEvaluationResults

# Module-level storage for latest regression model

_last_model_reg = None

def test_calibration(model, X, T, y, n_bins=5):
    global _last_model_reg

    X = check_array(X, accept_sparse=True)
    cate_preds = model.effect(X)

    if len(cate_preds) != len(T) or len(T) != len(y):
        raise ValueError("Mismatch in lengths of input arrays.")

    bins = pd.qcut(cate_preds, q=n_bins, labels=False, duplicates="drop")
    df = pd.DataFrame({"bin": bins, "cate_pred": cate_preds, "T": T, "Y": y})

    gate_vals, gate_se, mean_preds, diff_preds = [], [], [], []

    for b in range(n_bins):
        sub = df[df["bin"] == b]
        if sub.empty or sub["Y"].isnull().any() or sub["T"].isnull().any():
            continue

        treated = sub[sub["T"] == 1]["Y"]
        control = sub[sub["T"] == 0]["Y"]
        if len(treated) == 0 or len(control) == 0:
            continue

        ate = treated.mean() - control.mean()

        try:
            se_treat = sem(treated.dropna())
            se_control = sem(control.dropna())
            se = se_treat + se_control
        except ValueError:
            continue  # Skip bin if sem can't be computed

        if not np.isfinite(se) or se == 0:
            continue  # Skip if standard error is problematic

        mean_pred = sub["cate_pred"].mean()
        gate_vals.append(ate)
        gate_se.append(se)
        mean_preds.append(mean_pred)
        diff_preds.append((sub["cate_pred"] - mean_pred).mean())

    if len(gate_vals) == 0 or len(mean_preds) == 0 or len(diff_preds) == 0:
        raise ValueError("Insufficient data for calibration regression; check input data or binning.")

    gate_vals = np.array(gate_vals)
    gate_se = np.array(gate_se)
    mean_preds = np.array(mean_preds)
    diff_preds = np.array(diff_preds)

    valid_indices = ~(
        np.isnan(gate_vals) | np.isnan(mean_preds) | np.isnan(diff_preds) |
        np.isinf(gate_vals) | np.isinf(mean_preds) | np.isinf(diff_preds)
    )
    gate_vals = gate_vals[valid_indices]
    mean_preds = mean_preds[valid_indices]
    diff_preds = diff_preds[valid_indices]

    if len(gate_vals) < 2:
        raise ValueError("Not enough valid data to perform calibration regression after cleaning.")

    X_reg = sm.add_constant(pd.DataFrame({
        "mean.forest.prediction": mean_preds,
        "differential.forest.prediction": diff_preds
    }))
    model_reg = sm.OLS(gate_vals, X_reg).fit(cov_type='HC3')
    _last_model_reg = model_reg

    print("\nCalibration Regression Summary (R-style):")
    coef_names = ['Intercept', 'mean.forest.prediction', 'differential.forest.prediction']
    for idx, name in enumerate(coef_names):
        estimate = model_reg.params[idx]
        stderr = model_reg.bse[idx]
        tval = model_reg.tvalues[idx]
        pval = model_reg.pvalues[idx]
        signif = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else '.' if pval < 0.1 else ''
        print(f"{name:30s} {estimate:>10.6f} {stderr:>10.6f} {tval:>10.4f} {pval:>10.4e} {signif}")

    coef_data = {
        "Coefficient": coef_names,
        "Estimate": model_reg.params,
        "Std.Error": model_reg.bse,
        "t-value": model_reg.tvalues,
        "p-value": model_reg.pvalues,
        "Significance": [
            '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '.' if p < 0.1 else ''
            for p in model_reg.pvalues
        ]
    }
    coef_df = pd.DataFrame(coef_data)

    plot_df = pd.DataFrame({
        "g_cate": mean_preds + diff_preds,
        "gate": gate_vals,
        "se_gate": gate_se[valid_indices]
    })

    return CalibrationEvaluationResults(
        cal_r_squared=np.array([model_reg.rsquared]),
        plot_data_dict={1: plot_df},
        treatments=np.array([0, 1])
    )

def get_last_model_reg():
    return _last_model_reg