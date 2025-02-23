# %%
import pandas as pd
import numpy as np
import cupy as cp  # CuPy for GPU arrays 
import matplotlib.pyplot as plt
import os

# from rdkit import Chem
# from rdkit.Chem import Descriptors
# from rdkit.ML.Descriptors import MoleculeDescriptors

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt.plotting

# %%
# Parity plotter 
# ===========================================================================================
def plot_parity_with_metrics(target_name, data, metrics, save_path=None):
    """
    Creates a parity (observed vs. predicted) plot for train and test sets,
    annotated with MSE and R² metrics.
    """
    observed_train = data["train_observed"]
    predicted_train = data["train_predicted"]
    observed_test = data["test_observed"]
    predicted_test = data["test_predicted"]

    train_mse = metrics["train_mse"]
    test_mse = metrics["test_mse"]
    test_r2 = metrics["test_r2"]

    plt.figure(figsize=(8, 6))
    plt.scatter(observed_train, predicted_train, label="Train", color="blue", alpha=0.6)
    plt.scatter(observed_test, predicted_test, label="Test", color="red", alpha=0.6)

    # Plot perfect-fit line
    min_val = min(observed_train.min(), observed_test.min())
    max_val = max(observed_train.max(), observed_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", label="Perfect Fit")

    # Text box with metrics
    metrics_text = (
        f"Train MSE: {train_mse:.2f}\n"
        f"Test MSE: {test_mse:.2f}\n"
        f"Test R²: {test_r2:.2f}"
    )
    plt.text(
        0.05, 0.95, metrics_text,
        transform=plt.gca().transAxes,
        fontsize=12, verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

    plt.xlabel(f"Observed {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"Parity Plot for {target_name}")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=1200)
    plt.close()
    

# %%
# ============================================================================
# Note that this a new excel sheet I generated 
# from the preprocessing script that is read in.
# ============================================================================

#polystyrene_path = r"C:\Users\Dillo\OneDrive\Documents\GitHub\ActiveProjectsPrivate\ActiveFiles\Excels\polystyrene-imputated.xlsx"
polystyrene_path = r"C:\Users\micha\OneDrive\Documents\GitHub\ActiveProjectsPrivate\ActiveFiles\Scripts\PolystyreneSubset\Excels\polystyrene-imputated-molecular-descriptor-includes-inj-volume-2-15-25.xlsx"
polystyrene       = pd.read_excel(polystyrene_path)

# %%
# Predict the 1D target "Solvent Ratio_1"  -- this only works for two pairs of solvents
ratio_list = polystyrene['Solvent Ratio_1'].values    # target: 1D array
meta_data  = polystyrene[["Solvents_1", "Solvents_2"]]  # for tracking solvent pairs only

# %%
# Drop columns that are not features (e.g., Polymer and solvents used for meta)
columns_to_drop = ["Polymer", "Solvents_1", "Solvents_2", "Solvent Ratio_1"]
feature_columns  = polystyrene.drop(columns=columns_to_drop).columns

X = polystyrene.drop(columns=columns_to_drop).values
y = polystyrene["Solvent Ratio_1"].values

# %%
print("Feature matrix X shape:", X.shape)
print("Target y shape:", y.shape)

# %%
# ============================================================================
# Split the data (80% train, 10% validation, 10% test)
# ============================================================================

# Split into train and temporary sets while returning row indices
X_train_np, X_temp_np, y_train_np, y_temp_np, idx_train, idx_temp = train_test_split(
    X, y, np.arange(len(X)), test_size=0.20, random_state=42
)

# Split the temporary set equally into validation and test sets (10% each)
X_val_np, X_test_np, y_val_np, y_test_np, idx_val, idx_test = train_test_split(
    X_temp_np, y_temp_np, idx_temp, test_size=0.5, random_state=42
)

# %%
print(f"Train size: {X_train_np.shape[0]}")
print(f"Validation size: {X_val_np.shape[0]}")
print(f"Test size: {X_test_np.shape[0]}")

# %%
# ============================================================================
# Convert arrays to CuPy arrays for GPU training
# Note: With only ~200 samples, GPU training probably not be necessary.
# ============================================================================

X_train = cp.asarray(X_train_np)
y_train = cp.asarray(y_train_np)
X_val   = cp.asarray(X_val_np)
y_val   = cp.asarray(y_val_np)
X_test  = cp.asarray(X_test_np)
y_test  = cp.asarray(y_test_np)

# %%
# Define the search space for hyperparameters
search_space = {
    'n_estimators':       hp.quniform('n_estimators', 50, 500, 10),
    'max_depth':          hp.quniform('max_depth', 3, 5, 1),
    'learning_rate':      hp.loguniform('learning_rate', -4, -1),
    'subsample':          hp.uniform('subsample', 0.7, 1.0),
    'colsample_bytree':   hp.uniform('colsample_bytree', 0.5, 1.0),
    # 'colsample_bylevel':  hp.uniform('colsample_bylevel', 0.5, 1.0),  # 50% to 100% per level
    # 'colsample_bynode':   hp.uniform('colsample_bynode', 0.5, 1.0),
    'reg_alpha':          hp.loguniform('reg_alpha', -5, 2),
    'reg_lambda':         hp.loguniform('reg_lambda', -5, 2),
    'gamma':              hp.uniform('gamma', 0, 7),
    'min_child_weight':   hp.quniform('min_child_weight', 1, 5, 1),
    'max_bin':            hp.quniform('max_bin', 128, 1024, 16),
    'max_cat_threshold':  hp.quniform('max_cat_threshold', 16, 1024, 16)
}

# %%
def nested_cross_validation(
    X, y, 
    outer_splits=5, 
    inner_splits=3, 
    hyperopt_search_space=None, 
    hyperopt_max_evals=1000,
    random_state=42
):
    """
    Perform nested cross-validation with an outer CV for final performance estimation
    and an inner CV (Hyperopt) for hyperparameter tuning on each outer fold.
    
    X, y: numpy arrays of features/labels
    outer_splits: number of splits in outer CV
    inner_splits: number of splits in inner CV for hyperparameter tuning
    hyperopt_search_space: the search space to use with Hyperopt
    hyperopt_max_evals: maximum evaluations for each inner loop
    random_state: random seed for reproducibility
    """
    
    # Outer cross-validation for final performance metric
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    
    outer_mse_scores = []
    outer_r2_scores = []
    
    # Loop over outer folds
    for outer_train_idx, outer_test_idx in outer_cv.split(X):
        X_train_outer, X_test_outer = X[outer_train_idx], X[outer_test_idx]
        y_train_outer, y_test_outer = y[outer_train_idx], y[outer_test_idx]

        ############################################################################
        # (A) Hyperparameter Tuning on the (outer) training split using an INNER CV
        ############################################################################
        
        # Define an objective function that does an inner CV on (X_train_outer, y_train_outer)
        def objective(params):
            # Convert integer parameters to int type
            params['n_estimators']           = int(params['n_estimators'])
            params['max_depth']              = int(params['max_depth'])
            params['learning_rate']          = float(params['learning_rate'])
            params['gamma']                  = float(params['gamma'])
            params['min_child_weight']       = int(params['min_child_weight'])
            params['subsample']              = float(params['subsample'])
            params['colsample_bytree']       = float(params['colsample_bytree'])
            # params['colsample_bylevel']      = float(params['colsample_bylevel'])
            # params['colsample_bynode']       = float(params['colsample_bynode'])
            params['reg_alpha']              = float(params['reg_alpha'])
            params['reg_lambda']             = float(params['reg_lambda'])
    
            # Force GPU usage in XGBoost 2.1.4 -- my current version
            params['tree_method']            = 'hist'
            params['device']                 = 'cuda'
            params['max_bin']                = int(params['max_bin'])
            params['max_cat_threshold']      = int(params['max_cat_threshold'])
            
            # Create XGBRegressor with these hyperparameters
            model = xgb.XGBRegressor(
            **params,
            objective='reg:squarederror',
            #early_stopping_rounds=10
            )
            inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
            
            
            mse_list = []
            for train_idx, val_idx in inner_cv.split(X_train_outer):
                X_train_inner, X_val_inner = X_train_outer[train_idx], X_train_outer[val_idx]
                y_train_inner, y_val_inner = y_train_outer[train_idx], y_train_outer[val_idx]
                
                # Fit
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                # Predict
                y_pred_inner = model.predict(X_val_inner)
                
                # Collect MSE for this fold
                mse_fold = mean_squared_error(y_val_inner, y_pred_inner)
                mse_list.append(mse_fold)
            
            # Return average MSE across inner folds
            avg_mse = np.mean(mse_list)
            return {'loss': avg_mse, 'status': STATUS_OK}
                # Run Hyperopt on the *inner* CV
        trials = Trials()
        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=1000,
            trials=trials
        )
        
        # Convert best_params to the correct types
        # Convert certain hyperopt parameters to int
        best_params['n_estimators']           = int(best_params['n_estimators'])
        best_params['max_depth']              = int(best_params['max_depth'])
        best_params['learning_rate']          = float(best_params['learning_rate'])
        best_params['gamma']                  = float(best_params['gamma'])
        best_params['min_child_weight']       = int(best_params['min_child_weight'])
        best_params['subsample']              = float(best_params['subsample'])
        best_params['colsample_bytree']       = float(best_params['colsample_bytree'])
        # best_params['colsample_bylevel']      = float(best_params['colsample_bylevel'])
        # best_params['colsample_bynode']       = float(best_params['colsample_bynode'])
        best_params['reg_alpha']              = float(best_params['reg_alpha'])
        best_params['reg_lambda']             = float(best_params['reg_lambda'])

        # Force GPU usage in XGBoost 2.1.4 -- my current version
        best_params['tree_method']            = 'hist'
        best_params['device']                 = 'cuda'
        best_params['max_bin']                = int(best_params['max_bin'])
        best_params['max_cat_threshold']      = int(best_params['max_cat_threshold'])
        best_params["objective"]              = "reg:squarederror"
        best_params["early_stopping_rounds"]  = 10
        
        
        ############################################################################
        # (B) Train a final model on the entire outer_train split with best_params
        ############################################################################
        print("Best Hyperparameters:", best_params)
        
        final_model = xgb.XGBRegressor(
            **best_params,
            objective='reg:squarederror'
        )
        final_model.fit(X_train_outer, y_train_outer)  # single fit on the entire outer_train fold
        
        ############################################################################
        # (C) Evaluate on the outer_test split to get an unbiased metric
        ############################################################################
        
        y_pred_outer = final_model.predict(X_test_outer)
        mse_outer = mean_squared_error(y_test_outer, y_pred_outer)
        r2_outer  = r2_score(y_test_outer, y_pred_outer)
        
        outer_mse_scores.append(mse_outer)
        outer_r2_scores.append(r2_outer)
    
    # Summarize performance across outer folds
    results = {
        "mean_mse": np.mean(outer_mse_scores),
        "std_mse":  np.std(outer_mse_scores),
        "mean_r2":  np.mean(outer_r2_scores),
        "std_r2":   np.std(outer_r2_scores)
    }
    
    return results

# %%
# 2. Run nested CV
results = nested_cross_validation(
    X, y,
    outer_splits=5,
    inner_splits=3,
    hyperopt_search_space=search_space,
    hyperopt_max_evals=50,
    random_state=42
)

# %%
# ============================================================================
# Train the final model on GPU arrays with the tuned parameters
# ============================================================================

final_model = xgb.XGBRegressor(**best_params)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=0
)

# Evaluate the final model on the test set
y_train_np_conv = cp.asnumpy(y_train)
y_test_np_conv  = cp.asnumpy(y_test)

# Train predictions
y_pred_train_cp = final_model.predict(X_train)
y_pred_train_np = cp.asnumpy(y_pred_train_cp)

# Test predictions
y_pred_test_cp = final_model.predict(X_test)
y_pred_test_np = cp.asnumpy(y_pred_test_cp)




train_mse = mean_squared_error(y_train_np_conv, y_pred_train_np)
test_mse  = mean_squared_error(y_test_np_conv,  y_pred_test_np)
test_r2   = r2_score(y_test_np_conv, y_pred_test_np)

parity_data = {
    "train_observed":   y_train_np_conv,
    "train_predicted":  y_pred_train_np,
    "test_observed":    y_test_np_conv,
    "test_predicted":   y_pred_test_np
}
parity_metrics = {
    "train_mse": train_mse,
    "test_mse":  test_mse,
    "test_r2":   test_r2
}

print(f"Final Model - (Tuned, GPU) Test MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

# %%
importances = final_model.feature_importances_  # 1D numpy array of importances
feature_importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": importances
})

feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)

print(feature_importance_df)

# %%
# Feature importance

save_dir1 = r"C:\Users\micha\OneDrive\Documents\GitHub\ActiveProjectsPrivate\ActiveFiles\Scripts\PolystyreneSubset\Current\Polystyrene-features"
#save_dir = r"C:\Users\Dillo\OneDrive\Documents\GitHub\ActiveProjectsPrivate\ActiveFiles\Scripts\PolystyreneSubset\Current\Polystyrene-Polarity-Plots"
os.makedirs(save_dir1, exist_ok=True)

top_n = 35
top_features = feature_importance_df.head(top_n)

plt.figure(figsize=(8, 6))
plt.barh(top_features["feature"], top_features["importance"], color="steelblue")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title(f"Top {top_n} Feature Importances (XGBoost)")
plt.gca().invert_yaxis()  # so that the highest importance is at the top
plt.tight_layout()
save_path1 = os.path.join(save_dir1, "feature_importances2-17-25-attempt2.png")
plt.savefig(save_path1, dpi=1200, bbox_inches="tight")
plt.show()

# %%
# ============================================================================
# Saving polarity plots and showing predictions with their associated meta data
# ============================================================================

# Create a directory to save your plot (optional)
save_dir = r"C:\Users\micha\OneDrive\Documents\GitHub\ActiveProjectsPrivate\ActiveFiles\Scripts\PolystyreneSubset\Current\Polystyrene-Polarity-Plots"
#save_dir = r"C:\Users\Dillo\OneDrive\Documents\GitHub\ActiveProjectsPrivate\ActiveFiles\Scripts\PolystyreneSubset\Current\Polystyrene-Polarity-Plots"
os.makedirs(save_dir, exist_ok=True)

# Call the plotting function
plot_parity_with_metrics(
    target_name="Solvent Ratio_1",
    data=parity_data,
    metrics=parity_metrics,
    save_path=os.path.join(save_dir, "solvent_ratio_parity_plot2500-2-17-25-attempt2-molecular-desc.png")
)

# %%
# Keeps track of the original solvents along with predictions:

meta_test = meta_data.iloc[idx_test].reset_index(drop=True)

results_test = pd.DataFrame({
    "Solvents_1": meta_test["Solvents_1"],
    "Solvents_2": meta_test["Solvents_2"],
    "Observed": y_test_np_conv,
    "Predicted": y_pred_test_np
})

print(results_test)

# %%


# Plot how the loss function changed over the course of trials
hyperopt.plotting.main_plot_history(trials)
plt.savefig("hyperopt_history.png", dpi=1200, bbox_inches="tight")
plt.show()

# Plot a histogram of the loss
hyperopt.plotting.main_plot_histogram(trials)
plt.savefig("hyperopt_histogram.png", dpi=1200, bbox_inches="tight")
plt.show()

# Plot a summary of the variables that Hyperopt searched
hyperopt.plotting.main_plot_vars(trials)
plt.savefig("hyperopt_vars.png", dpi=1200, bbox_inches="tight")
plt.show()

# %%



