def test_calibration(model, X, T, y, n_bins=5):
    from sklearn.linear_model import LinearRegression
    from sklearn.utils import check_array
    from scipy.stats import sem
    import pandas as pd
    import numpy as np

    X = check_array(X, accept_sparse=True)
    cate_preds = model.effect(X)

    bins = pd.qcut(cate_preds, q=n_bins, labels=False, duplicates='drop')
    df = pd.DataFrame({
        'bin': bins,
        'cate_pred': cate_preds,
        'T': T,
        'Y': y
    })

    gate_vals = []
    gate_se = []
    pred_vals = []

    for b in range(n_bins):
        sub = df[df['bin'] == b]
        if sub.empty:
            continue
        ate = sub[sub['T'] == 1]['Y'].mean() - sub[sub['T'] == 0]['Y'].mean()
        se = sem(sub[sub['T'] == 1]['Y']) + sem(sub[sub['T'] == 0]['Y'])
        gate_vals.append(ate)
        gate_se.append(se)
        pred_vals.append(sub['cate_pred'].mean())

    reg = LinearRegression().fit(np.array(pred_vals).reshape(-1, 1), gate_vals)
    r_squared = reg.score(np.array(pred_vals).reshape(-1, 1), gate_vals)

    from validate.results import CalibrationEvaluationResults
    plot_df = pd.DataFrame({
        'g_cate': pred_vals,
        'gate': gate_vals,
        'se_gate': gate_se
    })

    return CalibrationEvaluationResults(
        cal_r_squared=np.array([r_squared]),
        plot_data_dict={1: plot_df},
        treatments=np.array([0, 1])
    )
