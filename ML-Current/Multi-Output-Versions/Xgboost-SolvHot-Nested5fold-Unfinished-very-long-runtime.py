import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

###############################################################################
# Extended Parity Plot Function
###############################################################################
def plot_parity_with_metrics(
    target_name, data, metrics, save_path=None
):
    """
    Creates a parity (observed vs. predicted) plot for train and test sets,
    annotated with metrics: MSE, R², etc.
    """
    observed_train  = data["train_observed"]
    predicted_train = data["train_predicted"]
    observed_test   = data["test_observed"]
    predicted_test  = data["test_predicted"]

    train_mse = metrics["train_mse"]
    test_mse  = metrics["test_mse"]
    test_r2   = metrics["test_r2"]

    plt.figure(figsize=(6, 5))
    plt.scatter(observed_train, predicted_train, label="Train", color="blue", alpha=0.6)
    plt.scatter(observed_test,  predicted_test,  label="Test",  color="red",  alpha=0.6)

    # Perfect-fit line
    all_values = np.concatenate([observed_train, observed_test, predicted_train, predicted_test])
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", label="Perfect Fit")

    # Metrics text
    metrics_text = (
        f"Train MSE: {train_mse:.2f}\n"
        f"Test MSE:  {test_mse:.2f}\n"
        f"Test R²:   {test_r2:.2f}"
    )
    plt.text(
        0.05, 0.95, metrics_text,
        transform=plt.gca().transAxes,
        fontsize=9, verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

    plt.xlabel(f"Observed {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"Parity Plot for {target_name}")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches="tight")
    plt.close()



###############################################################################
polystyrene_path = r"C:\Users\micha\OneDrive\Documents\GitHub\LCCC-ML\polystyrene-imputated-solvents-hot-encoded-2-23-25.xlsx"
#polystyrene_path = r"C:\Users\Dillo\OneDrive\Documents\GitHub\LCCC-ML\polystyrene-imputated-solvents-hot-encoded-2-23-25.xlsx"
polystyrene = pd.read_excel(polystyrene_path)

# 20 target columns (one for each solvent ratio)
solvent_cols = [
    'Hexane', 'Tetrahydrofuran', 'Dichloromethane', 'Acetonitrile',
    'Tetrachloromethane', 'Ethyl Acetate', 'Toluene', 'Carbon Dioxide',
    'Xylene', 'Cyclohexane', 'Dimethylacetamide', 'Heptane', 'Decalin',
    'Dimethylformamide', 'Water', 'Chloroform', 'Methanol',
    '2,2,4-Trimethylpentane', 'Methyl Ethyl Ketone', 'Cyclohexanone'
]
y = polystyrene[solvent_cols].values  # shape (n_samples, 20)

# Feature columns
columns_to_drop = ["Polymer"] + solvent_cols
feature_columns = polystyrene.drop(columns=columns_to_drop).columns
X = polystyrene.drop(columns=columns_to_drop).values  # shape (n_samples, n_features)

print("Feature matrix X shape:", X.shape)
print("Target y shape:", y.shape)


###############################################################################
# Nested Cross‐Validation Setup
# We'll do a 5‐fold outer CV for final performance estimation
# and a 3‐fold inner CV in hyperparameter search.
###############################################################################
outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)

# We'll store per-fold, per-column metrics in a list of dicts
metrics_list = []

# You may want to store parity plots in a subdirectory
os.makedirs("nested_cv_parity_plots", exist_ok=True)

###############################################################################
# Helper Functions
###############################################################################
def train_single_output_with_early_stopping(X_tr, y_tr, X_val, y_val, params):
    """Train a single XGB model for one column with early stopping on (X_val, y_val)."""
    model = xgb.XGBRegressor(
        **params,
        objective='reg:squarederror',
        early_stopping_rounds=10,
        verbosity=1  
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    y_pred_val = model.predict(X_val)
    return mean_squared_error(y_val, y_pred_val)

def inner_cv_objective(params, X_train_inner, y_train_inner, n_splits=3):
    """
    Perform an inner CV on (X_train_inner, y_train_inner) for a single set of hyperparams.
    We'll train 20 single-output models per fold, average MSE across columns -> fold MSE,
    then average across folds -> return that average MSE.
    """
    # Convert discrete types
    params['n_estimators']      = int(params['n_estimators'])
    params['max_depth']         = int(params['max_depth'])
    params['learning_rate']     = float(params['learning_rate'])
    params['gamma']             = float(params['gamma'])
    params['min_child_weight']  = int(params['min_child_weight'])
    params['subsample']         = float(params['subsample'])
    params['colsample_bytree']  = float(params['colsample_bytree'])
    params['reg_alpha']         = float(params['reg_alpha'])
    params['reg_lambda']        = float(params['reg_lambda'])
    params['max_bin']           = int(params['max_bin'])
    params['max_cat_threshold'] = int(params['max_cat_threshold'])

    # Force GPU usage in XGBoost
    params['tree_method'] = 'hist'
    params['device']      = 'cuda'

    kf_inner = KFold(n_splits=n_splits, shuffle=True, random_state=123)
    fold_mses = []

    for tr_idx, val_idx in kf_inner.split(X_train_inner):
        X_tr_fold, X_val_fold = X_train_inner[tr_idx], X_train_inner[val_idx]
        y_tr_fold, y_val_fold = y_train_inner[tr_idx], y_train_inner[val_idx]

        mses_for_fold = []
        for col_i in range(y_tr_fold.shape[1]):
            mse_col = train_single_output_with_early_stopping(
                X_tr_fold, y_tr_fold[:, col_i],
                X_val_fold, y_val_fold[:, col_i],
                params
            )
            mses_for_fold.append(mse_col)
        fold_mses.append(np.mean(mses_for_fold))

    return np.mean(fold_mses)

def hyperopt_objective(params):
    """
    This is called by hyperopt with a single candidate param set.
    We'll do an inner CV on the global (X_train_inner, y_train_inner) 
    to compute the average MSE, returned as the 'loss'.
    """
    global X_train_inner, y_train_inner
    avg_mse = inner_cv_objective(params, X_train_inner, y_train_inner, n_splits=3)
    return {'loss': avg_mse, 'status': STATUS_OK}


###############################################################################
# Outer Loop: 5‐fold
###############################################################################
outer_fold_num = 1

for outer_train_idx, outer_test_idx in outer_kf.split(X):
    print(f"\n=== Outer Fold {outer_fold_num} ===")

    # Prepare train/test for this outer fold
    X_train_outer = X[outer_train_idx]
    y_train_outer = y[outer_train_idx]
    X_test_outer  = X[outer_test_idx]
    y_test_outer  = y[outer_test_idx]

    # -------------------------------------------------------------------------
    # (A) Inner Hyperopt Search
    # -------------------------------------------------------------------------
    # We'll define the data as global for the hyperopt_objective function
    X_train_inner = X_train_outer
    y_train_inner = y_train_outer

    search_space = {
        'n_estimators':       hp.quniform('n_estimators', 50, 500, 50),
        'max_depth':          hp.quniform('max_depth', 3, 6, 1),
        'learning_rate':      hp.loguniform('learning_rate', -4, -1),
        'subsample':          hp.uniform('subsample', 0.7, 1.0),
        'colsample_bytree':   hp.uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha':          hp.loguniform('reg_alpha', -5, 2),
        'reg_lambda':         hp.loguniform('reg_lambda', -5, 2),
        'gamma':              hp.uniform('gamma', 0, 7),
        'min_child_weight':   hp.quniform('min_child_weight', 1, 5, 1),
        'max_bin':            hp.quniform('max_bin', 128, 512, 64),
        'max_cat_threshold':  hp.quniform('max_cat_threshold', 16, 128, 16)
    }

    trials_inner = Trials()
    best_params_dict = fmin(
        fn=hyperopt_objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100,  
        trials=trials_inner
    )

    # Convert best_params from hyperopt
    best_params_dict['n_estimators']      = int(best_params_dict['n_estimators'])
    best_params_dict['max_depth']         = int(best_params_dict['max_depth'])
    best_params_dict['learning_rate']     = float(best_params_dict['learning_rate'])
    best_params_dict['gamma']             = float(best_params_dict['gamma'])
    best_params_dict['min_child_weight']  = int(best_params_dict['min_child_weight'])
    best_params_dict['subsample']         = float(best_params_dict['subsample'])
    best_params_dict['colsample_bytree']  = float(best_params_dict['colsample_bytree'])
    best_params_dict['reg_alpha']         = float(best_params_dict['reg_alpha'])
    best_params_dict['reg_lambda']        = float(best_params_dict['reg_lambda'])
    best_params_dict['max_bin']           = int(best_params_dict['max_bin'])
    best_params_dict['max_cat_threshold'] = int(best_params_dict['max_cat_threshold'])
    best_params_dict['tree_method']       = 'hist'
    best_params_dict['device']            = 'cuda'
    best_params_dict['objective']         = 'reg:squarederror'

    print("Best Inner Hyperparams:", best_params_dict)

    # -------------------------------------------------------------------------
    # (B) Retrain single‐output models on entire outer train fold
    # -------------------------------------------------------------------------
    final_models = []
    for col_i in range(y_train_outer.shape[1]):
        model_i = xgb.XGBRegressor(**best_params_dict)
        # NOTE: We won't do further early stopping here, we just fit on all outer train.
        model_i.fit(X_train_outer, y_train_outer[:, col_i])
        final_models.append(model_i)

    # -------------------------------------------------------------------------
    # (C) Evaluate all 20 columns on outer test
    #     We'll also store parity plots + extended metrics (MSE, MAE, MAPE, R²).
    # -------------------------------------------------------------------------
    # We'll iterate through each column, get predictions, store in "metrics_list"
    for col_i, model_i in enumerate(final_models):
        col_name = solvent_cols[col_i]

        # Predictions
        y_pred_test = model_i.predict(X_test_outer)
        y_pred_train = model_i.predict(X_train_outer)

        # Compute metrics for test
        mse_test = mean_squared_error(y_test_outer[:, col_i], y_pred_test)
        mae_test = mean_absolute_error(y_test_outer[:, col_i], y_pred_test)
        r2_test  = r2_score(y_test_outer[:, col_i], y_pred_test)

        # MAPE (Mean Absolute Percentage Error), watch for zeros
        y_true_test = y_test_outer[:, col_i]
        epsilon = 1e-9
        mape_test = np.mean(np.abs((y_true_test - y_pred_test) / (y_true_test + epsilon))) * 100

        # Similarly for train
        mse_train = mean_squared_error(y_train_outer[:, col_i], y_pred_train)
        mae_train = mean_absolute_error(y_train_outer[:, col_i], y_pred_train)
        r2_train  = r2_score(y_train_outer[:, col_i], y_pred_train)
        y_true_train = y_train_outer[:, col_i]
        mape_train = np.mean(np.abs((y_true_train - y_pred_train) / (y_true_train + epsilon))) * 100

        # Log it
        metrics_dict = {
            "outer_fold": outer_fold_num,
            "solvent_col": col_name,
            "mse_train": mse_train,
            "mae_train": mae_train,
            "mape_train": mape_train,
            "r2_train": r2_train,
            "mse_test": mse_test,
            "mae_test": mae_test,
            "mape_test": mape_test,
            "r2_test": r2_test
        }
        metrics_list.append(metrics_dict)

        # Create a parity plot for this column in this outer fold
        parity_data = {
            "train_observed":   y_true_train,
            "train_predicted":  y_pred_train,
            "test_observed":    y_true_test,
            "test_predicted":   y_pred_test,
        }
        parity_metrics = {
            "train_mse": mse_train,
            "test_mse":  mse_test,
            "test_r2":   r2_test
        }
        plot_filename = f"nested_cv_parity_plots/outerFold{outer_fold_num}_{col_name}.png"
        plot_parity_with_metrics(
            target_name=f"{col_name}_Fold{outer_fold_num}",
            data=parity_data,
            metrics=parity_metrics,
            save_path=plot_filename
        )

    outer_fold_num += 1


###############################################################################
# All Metrics
###############################################################################
metrics_df = pd.DataFrame(metrics_list)
print("\nAll Metrics (per fold, per column):")
print(metrics_df.head(40))  # Show first 40 rows for reference

# Overall average across folds for each column
summary_by_col = metrics_df.groupby("solvent_col").mean(numeric_only=True)
print("\n=== Average Metrics by Solvent Column (across 5 outer folds) ===")
print(summary_by_col[["mse_test", "mae_test", "mape_test", "r2_test"]])

# Overall average across columns and folds
overall_means = metrics_df[["mse_test", "mae_test", "mape_test", "r2_test"]].mean()
print("\n=== Overall Averages (across all folds and columns) ===")
print(overall_means)

# You may also export metrics_df to CSV
metrics_df.to_csv("nested_cv_metrics.csv", index=False)
print("\nMetrics saved to nested_cv_metrics.csv.")







# Notes
# Jeez, I am noticing only like 0.5 improvement in the Inner fold - validation_0-rsme after 60 rounds. 
# For a minute I thought it wasn't going to improve from 28 best loss at all. Currently 27.7.