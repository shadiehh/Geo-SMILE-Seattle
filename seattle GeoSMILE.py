# ============================================================
# Geo-SMILE Geo-Importance Pipeline
# Corrected version:
# - Base black-box model is NOT changed
# - UTM_X and UTM_Y remain inside the model features
# - Spatial groups are created using UTM_X and UTM_Y
# - Perturbation uses realistic median baseline values
# - Wasserstein distance measures prediction shift
# - Ridge surrogate learns point-level geo-importance
# ============================================================

# ---------------------- Step 0: Imports ----------------------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from scipy.stats import wasserstein_distance

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import xgboost as xgb
import lightgbm as lgb
from flaml import AutoML


# ============================================================
# Step 1: Load and Prepare Data
# ============================================================

data = pd.read_csv(
    "/kaggle/input/datasets/shadiemohammadi/ziqi-seattle/seattle_sample_1k.csv"
)

# ------------------------------------------------------------
# Full black-box feature set
# IMPORTANT:
# This keeps the original black-box model unchanged.
# UTM_X and UTM_Y are included in the predictive model.
# ------------------------------------------------------------

features = [
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "grade",
    "condition",
    "waterfront",
    "view",
    "age",
    "UTM_X",
    "UTM_Y"
]

spatial_features = ["UTM_X", "UTM_Y"]
target = "log_price"

required_cols = features + [target]
missing_cols = [c for c in required_cols if c not in data.columns]

if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

data = data.dropna(subset=required_cols).reset_index(drop=True)

X = data[features].copy()
y = data[target].copy()


# ============================================================
# Step 2: Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Black-box model features:", features)


# ============================================================
# Step 3: Create Spatial Groups Using UTM Coordinates
# ============================================================

coords_train = X_train[spatial_features].copy()

coords_scaler = StandardScaler()
coords_train_scaled = coords_scaler.fit_transform(coords_train)

# For around 750 training points, 50-100 groups is more stable than 300
n_groups = 100

kmeans = KMeans(
    n_clusters=n_groups,
    random_state=42,
    n_init=10
)

spatial_group_labels = kmeans.fit_predict(coords_train_scaled)

print(f"\nNumber of spatial groups: {n_groups}")
print("Unique spatial groups:", len(np.unique(spatial_group_labels)))

group_size_df = pd.Series(spatial_group_labels).value_counts().sort_index()

print("\nSpatial group size summary:")
print(group_size_df.describe())


# ============================================================
# Step 4: Train Base Black-Box AutoML Model
# ============================================================

automl = AutoML()

automl.fit(
    X_train,
    y_train,
    task="regression",
    time_budget=20,
    metric="r2",
    verbose=0
)

train_preds = automl.predict(X_train)
test_preds = automl.predict(X_test)

base_train_r2 = r2_score(y_train, train_preds)
base_test_r2 = r2_score(y_test, test_preds)

print("\nBase black-box model performance:")
print(f"Train R²: {base_train_r2:.4f}")
print(f"Test R² : {base_test_r2:.4f}")

# Original predictions used as baseline for perturbation analysis
original_preds = train_preds.copy()


# ============================================================
# Step 5: Generate Geo-SMILE Spatial Perturbations
# ============================================================

n_points = X_train.shape[0]

n_subsets = 3000
min_groups = 5
max_groups = 10

# M is the perturbation mask matrix
# Rows = perturbation samples
# Columns = training points / houses
# M[j, i] = 1 means point i was perturbed in perturbation j
M = np.zeros((n_subsets, n_points), dtype=int)

# Stores Wasserstein prediction shift for each perturbation
y_shifts = []

# ------------------------------------------------------------
# Important correction:
# Use median baseline values for ALL model features.
# This includes UTM_X and UTM_Y, but avoids unrealistic zero coordinates.
# ------------------------------------------------------------

baseline_values = X_train.median()

for j in tqdm(range(n_subsets), desc="Generating Geo-SMILE perturbations"):

    # Randomly select spatial groups
    g = np.random.randint(min_groups, max_groups + 1)

    selected_group_ids = np.random.choice(
        n_groups,
        size=g,
        replace=False
    )

    # Find points belonging to selected spatial groups
    perturbed_indices = np.where(
        np.isin(spatial_group_labels, selected_group_ids)
    )[0]

    if len(perturbed_indices) == 0:
        y_shifts.append(0.0)
        continue

    # Create perturbed copy
    X_pert = X_train.copy()

    # Neutralise selected spatial regions using realistic baseline values
    X_pert.iloc[perturbed_indices, :] = baseline_values

    # Predict using the SAME unchanged black-box model
    perturbed_preds = automl.predict(X_pert)

    # Wasserstein shift inside affected region
    shift = wasserstein_distance(
        original_preds[perturbed_indices],
        perturbed_preds[perturbed_indices]
    )

    y_shifts.append(shift)

    # Store perturbation mask
    M[j, perturbed_indices] = 1

y_shifts = np.array(y_shifts)

print("\nPerturbation matrix shape:", M.shape)
print("Shift vector shape:", y_shifts.shape)
print("Mean Wasserstein shift:", y_shifts.mean())
print("Std Wasserstein shift:", y_shifts.std())


# ============================================================
# Step 6: Train/Test Split for Surrogate Fidelity
# ============================================================

M_train, M_test, shift_train, shift_test = train_test_split(
    M,
    y_shifts,
    test_size=0.25,
    random_state=42
)


# ============================================================
# Step 7: Ridge Surrogate for Point-Level Geo-Importance
# ============================================================

ridge_surrogate = Ridge(alpha=1.0)

ridge_surrogate.fit(M_train, shift_train)

fitted_shifts_train = ridge_surrogate.predict(M_train)
fitted_shifts_test = ridge_surrogate.predict(M_test)

ridge_train_r2 = r2_score(shift_train, fitted_shifts_train)
ridge_test_r2 = r2_score(shift_test, fitted_shifts_test)

ridge_train_mse = mean_squared_error(shift_train, fitted_shifts_train)
ridge_test_mse = mean_squared_error(shift_test, fitted_shifts_test)

print("\nGeo-SMILE Ridge surrogate performance:")
print(f"Train R²: {ridge_train_r2:.4f}")
print(f"Test R² : {ridge_test_r2:.4f}")
print(f"Train MSE: {ridge_train_mse:.6f}")
print(f"Test MSE : {ridge_test_mse:.6f}")

# Point-level geo-importance scores
geo_importance = pd.Series(
    ridge_surrogate.coef_,
    index=X_train.index,
    name="geo_importance"
)

geo_importance_abs = geo_importance.abs()


# ============================================================
# Step 8: Fidelity Metrics
# ============================================================

r2_fidelity = r2_score(shift_test, fitted_shifts_test)
mae = mean_absolute_error(shift_test, fitted_shifts_test)
mse = mean_squared_error(shift_test, fitted_shifts_test)
l1 = mae
l2 = np.sqrt(mse)

print("\n--- Geo-SMILE Fidelity Metrics ---")
print(f"R² Fidelity: {r2_fidelity:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"L1: {l1:.4f}")
print(f"L2: {l2:.4f}")

fidelity_metrics_df = pd.DataFrame({
    "Category": ["Fidelity", "Fidelity", "Fidelity", "Fidelity", "Fidelity"],
    "Metric": ["R² Fidelity", "MAE", "MSE", "L1", "L2"],
    "Value": [r2_fidelity, mae, mse, l1, l2]
})

display(fidelity_metrics_df)


# ============================================================
# Step 9: Prepare GeoDataFrame for Full Geo-SMILE Map
# ============================================================

data_train = data.loc[X_train.index].copy()

data_train["geo_importance"] = geo_importance
data_train["geo_importance_abs"] = geo_importance_abs

data_train["geometry"] = gpd.points_from_xy(
    data_train["UTM_X"],
    data_train["UTM_Y"]
)

gdf_train = gpd.GeoDataFrame(
    data_train,
    geometry="geometry",
    crs="EPSG:32610"
)

print("\nGeo-importance summary:")
print(gdf_train["geo_importance_abs"].describe())


# ============================================================
# Step 10: Map 1 - Absolute Geo-SMILE Importance
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))

gdf_train.plot(
    column="geo_importance_abs",
    cmap="plasma",
    legend=True,
    markersize=50,
    ax=ax
)

ax.set_title("Geo-SMILE Point-Level Importance Map", fontsize=15)
ax.set_xlabel("UTM_X")
ax.set_ylabel("UTM_Y")
ax.axis("equal")

plt.tight_layout()
plt.show()


# ============================================================
# Step 11: Map 2 - Signed Geo-SMILE Importance
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))

gdf_train.plot(
    column="geo_importance",
    cmap="Spectral",
    legend=True,
    markersize=50,
    ax=ax
)

ax.set_title("Geo-SMILE Signed Point-Level Importance Map", fontsize=15)
ax.set_xlabel("UTM_X")
ax.set_ylabel("UTM_Y")
ax.axis("equal")

plt.tight_layout()
plt.show()


# ============================================================
# Step 12: Distribution of Wasserstein Shifts
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    y_shifts,
    bins=30,
    kde=True
)

plt.title("Distribution of Wasserstein Prediction Shifts")
plt.xlabel("Wasserstein Shift")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ============================================================
# Step 13: Observed vs Predicted Shifts
# ============================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    shift_test,
    fitted_shifts_test,
    alpha=0.7
)

min_val = min(shift_test.min(), fitted_shifts_test.min())
max_val = max(shift_test.max(), fitted_shifts_test.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

plt.title(f"Geo-SMILE Surrogate Fidelity: Test R² = {r2_fidelity:.4f}")
plt.xlabel("Observed Wasserstein Shift")
plt.ylabel("Predicted Wasserstein Shift")

plt.tight_layout()
plt.show()


# ============================================================
# Step 14: Surrogate Model Comparison
# ============================================================

surrogate_models = {
    "Ridge": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    ),
    "SVR": SVR(kernel="rbf", C=1.0),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective="reg:squarederror",
        verbosity=0,
        random_state=42
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
}

r2_scores = {}
mse_scores = {}

for name, model in surrogate_models.items():

    model.fit(M_train, shift_train)

    pred_test = model.predict(M_test)

    r2_scores[name] = r2_score(shift_test, pred_test)
    mse_scores[name] = mean_squared_error(shift_test, pred_test)

comparison_df = pd.DataFrame({
    "Model": list(r2_scores.keys()),
    "Test_R2": list(r2_scores.values()),
    "Test_MSE": list(mse_scores.values())
}).sort_values(by="Test_R2", ascending=False)

print("\nSurrogate model comparison:")
display(comparison_df)


plt.figure(figsize=(10, 6))

plt.bar(
    comparison_df["Model"],
    comparison_df["Test_R2"]
)

plt.ylabel("Test R² Score")
plt.title("Geo-SMILE Shift Prediction: Surrogate Model Comparison")
plt.xticks(rotation=45)
plt.grid(axis="y")

plt.tight_layout()
plt.show()


# ============================================================
# Step 15: Proxy Ground-Truth ATT Evaluation
# Random seed = 20, Top 20%
# ============================================================

np.random.seed(20)

n_reference_points = 100
top_percent = 20

n_available_points = X_train.shape[0]
n_reference_points = min(n_reference_points, n_available_points)

reference_positions = np.random.choice(
    np.arange(n_available_points),
    size=n_reference_points,
    replace=False
)

reference_indices = X_train.index[reference_positions]

direct_effects = []

for pos, idx in zip(reference_positions, reference_indices):

    X_single_pert = X_train.copy()

    # Neutralise this one point only using the same full feature baseline
    X_single_pert.iloc[pos, :] = baseline_values

    perturbed_preds_single = automl.predict(X_single_pert)

    effect = abs(original_preds[pos] - perturbed_preds_single[pos])

    direct_effects.append(effect)

direct_effects = np.array(direct_effects)

proxy_gt_df = pd.DataFrame({
    "original_index": reference_indices,
    "train_position": reference_positions,
    "direct_perturbation_effect": direct_effects
})

# Top 20% direct effects = proxy-important
gt_threshold = np.percentile(
    proxy_gt_df["direct_perturbation_effect"],
    100 - top_percent
)

proxy_gt_df["y_true"] = (
    proxy_gt_df["direct_perturbation_effect"] >= gt_threshold
).astype(int)

# Geo-SMILE score for the same sampled points
proxy_gt_df["geo_smile_score"] = geo_importance_abs.loc[
    proxy_gt_df["original_index"]
].values

# Top 20% Geo-SMILE scores = predicted important
pred_threshold = np.percentile(
    proxy_gt_df["geo_smile_score"],
    100 - top_percent
)

proxy_gt_df["y_pred"] = (
    proxy_gt_df["geo_smile_score"] >= pred_threshold
).astype(int)

y_true = proxy_gt_df["y_true"].values
y_pred = proxy_gt_df["y_pred"].values
y_score = proxy_gt_df["geo_smile_score"].values

att_acc = accuracy_score(y_true, y_pred)
att_f1 = f1_score(y_true, y_pred)

if len(np.unique(y_true)) > 1:
    att_auroc = roc_auc_score(y_true, y_score)
else:
    att_auroc = np.nan
    print("Warning: ATT AUROC could not be computed because y_true has only one class.")

print("\n--- Proxy Ground-Truth ATT Metrics ---")
print(f"Random seed: 20")
print(f"Top threshold: {top_percent}%")
print(f"ATT ACC: {att_acc:.4f}")
print(f"ATT F1: {att_f1:.4f}")
print(f"ATT AUROC: {att_auroc:.4f}")

att_metrics_df = pd.DataFrame({
    "Category": ["Ground-truth ATT", "Ground-truth ATT", "Ground-truth ATT"],
    "Metric": ["ATT ACC", "ATT F1", "ATT AUROC"],
    "Value": [att_acc, att_f1, att_auroc]
})

display(att_metrics_df)


# ============================================================
# Step 16: Combined Evaluation Table
# ============================================================

evaluation_metrics_df = pd.concat(
    [fidelity_metrics_df, att_metrics_df],
    ignore_index=True
)

print("\n--- Combined Geo-SMILE Evaluation Metrics ---")
display(evaluation_metrics_df)


# ============================================================
# Step 17: Prepare Proxy Ground-Truth GeoDataFrame
# ============================================================

proxy_map_df = data.loc[proxy_gt_df["original_index"]].copy()

proxy_map_df["direct_perturbation_effect"] = proxy_gt_df[
    "direct_perturbation_effect"
].values

proxy_map_df["geo_smile_score"] = proxy_gt_df[
    "geo_smile_score"
].values

proxy_map_df["y_true"] = proxy_gt_df["y_true"].values
proxy_map_df["y_pred"] = proxy_gt_df["y_pred"].values

proxy_map_df["geometry"] = gpd.points_from_xy(
    proxy_map_df["UTM_X"],
    proxy_map_df["UTM_Y"]
)

proxy_gdf = gpd.GeoDataFrame(
    proxy_map_df,
    geometry="geometry",
    crs="EPSG:32610"
)

# Agreement categories
def classify_agreement(row):
    if row["y_true"] == 1 and row["y_pred"] == 1:
        return "TP"
    elif row["y_true"] == 0 and row["y_pred"] == 0:
        return "TN"
    elif row["y_true"] == 0 and row["y_pred"] == 1:
        return "FP"
    elif row["y_true"] == 1 and row["y_pred"] == 0:
        return "FN"

proxy_gdf["agreement"] = proxy_gdf.apply(classify_agreement, axis=1)


# ============================================================
# Step 18: Map 3 - Direct Perturbation Effect
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))

gdf_train.plot(
    ax=ax,
    markersize=15,
    alpha=0.20,
    color="lightgrey"
)

proxy_gdf.plot(
    column="direct_perturbation_effect",
    cmap="viridis",
    legend=True,
    markersize=75,
    ax=ax
)

ax.set_title("Proxy Ground Truth: Direct Perturbation Effect", fontsize=15)
ax.set_xlabel("UTM_X")
ax.set_ylabel("UTM_Y")
ax.axis("equal")

plt.tight_layout()
plt.show()


# ============================================================
# Step 19: Map 4 - Proxy Ground-Truth Important Points
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))

gdf_train.plot(
    ax=ax,
    markersize=15,
    alpha=0.20,
    color="lightgrey"
)

proxy_gdf.plot(
    column="y_true",
    categorical=True,
    legend=True,
    markersize=75,
    ax=ax
)

ax.set_title("Proxy Ground-Truth Important Points", fontsize=15)
ax.set_xlabel("UTM_X")
ax.set_ylabel("UTM_Y")
ax.axis("equal")

plt.tight_layout()
plt.show()


# ============================================================
# Step 20: Map 5 - Geo-SMILE Predicted Important Points
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))

gdf_train.plot(
    ax=ax,
    markersize=15,
    alpha=0.20,
    color="lightgrey"
)

proxy_gdf.plot(
    column="y_pred",
    categorical=True,
    legend=True,
    markersize=75,
    ax=ax
)

ax.set_title("Geo-SMILE Predicted Important Points", fontsize=15)
ax.set_xlabel("UTM_X")
ax.set_ylabel("UTM_Y")
ax.axis("equal")

plt.tight_layout()
plt.show()


# ============================================================
# Step 21: Map 6 - Agreement / Disagreement Map
# ============================================================

agreement_colors = {
    "TP": "green",
    "TN": "lightgrey",
    "FP": "orange",
    "FN": "red"
}

fig, ax = plt.subplots(figsize=(12, 8))

gdf_train.plot(
    ax=ax,
    markersize=15,
    alpha=0.15,
    color="lightgrey"
)

for label, color in agreement_colors.items():
    subset = proxy_gdf[proxy_gdf["agreement"] == label]
    if len(subset) > 0:
        subset.plot(
            ax=ax,
            color=color,
            markersize=80,
            label=label,
            alpha=0.85
        )

ax.legend(title="Agreement Type")
ax.set_title("Geo-SMILE Proxy Attribution Agreement Map", fontsize=15)
ax.set_xlabel("UTM_X")
ax.set_ylabel("UTM_Y")
ax.axis("equal")

plt.tight_layout()
plt.show()


# ============================================================
# Step 22: Save Main Outputs
# ============================================================

geo_smile_results = {
    "base_train_r2": base_train_r2,
    "base_test_r2": base_test_r2,
    "M": M,
    "y_shifts": y_shifts,
    "ridge_surrogate": ridge_surrogate,
    "geo_importance": geo_importance,
    "geo_importance_abs": geo_importance_abs,
    "gdf_train": gdf_train,
    "fidelity_metrics_df": fidelity_metrics_df,
    "att_metrics_df": att_metrics_df,
    "evaluation_metrics_df": evaluation_metrics_df,
    "comparison_df": comparison_df,
    "proxy_gt_df": proxy_gt_df,
    "proxy_gdf": proxy_gdf
}

print("\nGeo-SMILE point-based geospatial pipeline completed successfully.")

# ============================================================
# Geo-SMILE Additional Evaluation Metrics
# Spearman Consistency, Pearson Consistency, Stability,
# Sparsity, and Entropy
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import wasserstein_distance
from itertools import combinations
from tqdm import tqdm


# ============================================================
# Safety Checks
# ============================================================

required_vars = [
    "X_train",
    "data",
    "automl",
    "original_preds",
    "baseline_values",
    "spatial_group_labels",
    "n_groups",
    "geo_importance",
    "geo_importance_abs",
    "proxy_gt_df",
    "gdf_train"
]

missing_vars = [v for v in required_vars if v not in globals()]

if missing_vars:
    raise ValueError(f"Missing required variables from previous Geo-SMILE pipeline: {missing_vars}")


# ============================================================
# Step 1: Consistency with Proxy Direct-Perturbation Reference
# ============================================================
# This evaluates whether Geo-SMILE point-level importance scores
# are consistent with direct single-point perturbation effects.
#
# direct_perturbation_effect = proxy empirical attribution strength
# geo_smile_score = Geo-SMILE attribution magnitude
# ============================================================

if "direct_perturbation_effect" not in proxy_gt_df.columns:
    raise ValueError("proxy_gt_df must contain 'direct_perturbation_effect'.")

if "geo_smile_score" not in proxy_gt_df.columns:
    proxy_gt_df["geo_smile_score"] = geo_importance_abs.loc[
        proxy_gt_df["original_index"]
    ].values

direct_effects = proxy_gt_df["direct_perturbation_effect"].values
geo_scores = proxy_gt_df["geo_smile_score"].values

# Spearman rank consistency
spearman_corr, spearman_p = spearmanr(direct_effects, geo_scores)

# Pearson linear consistency
pearson_corr, pearson_p = pearsonr(direct_effects, geo_scores)

print("\n--- Geo-SMILE Consistency Metrics ---")
print(f"Spearman Consistency: {spearman_corr:.4f}")
print(f"Spearman p-value    : {spearman_p:.6f}")
print(f"Pearson Consistency : {pearson_corr:.4f}")
print(f"Pearson p-value     : {pearson_p:.6f}")


consistency_metrics_df = pd.DataFrame({
    "Category": ["Consistency", "Consistency", "Consistency", "Consistency"],
    "Metric": [
        "Spearman Consistency",
        "Spearman p-value",
        "Pearson Consistency",
        "Pearson p-value"
    ],
    "Value": [
        spearman_corr,
        spearman_p,
        pearson_corr,
        pearson_p
    ]
})

display(consistency_metrics_df)


# ============================================================
# Step 2: Stability Across Repeated Geo-SMILE Runs
# ============================================================
# This repeats the Geo-SMILE perturbation + Ridge surrogate stage.
#
# Important:
# - The base black-box model is NOT retrained.
# - The same trained automl model is reused.
# - Only perturbation sampling and surrogate fitting are repeated.
# ============================================================

n_repeats = 10
n_subsets_stability = 1000

min_groups = 5
max_groups = 10
top_percent_stability = 20

n_points = X_train.shape[0]

importance_runs = []
fidelity_runs = []

repeat_seeds = list(range(10, 10 + n_repeats))

for seed in tqdm(repeat_seeds, desc="Repeated Geo-SMILE stability runs"):
    
    np.random.seed(seed)
    
    M_rep = np.zeros((n_subsets_stability, n_points), dtype=int)
    y_shifts_rep = []

    for j in range(n_subsets_stability):
        
        g = np.random.randint(min_groups, max_groups + 1)
        
        selected_group_ids = np.random.choice(
            n_groups,
            size=g,
            replace=False
        )
        
        perturbed_indices = np.where(
            np.isin(spatial_group_labels, selected_group_ids)
        )[0]
        
        if len(perturbed_indices) == 0:
            y_shifts_rep.append(0.0)
            continue
        
        X_pert = X_train.copy()
        
        # Same correction as main pipeline:
        # full feature vector replaced with realistic median baseline
        X_pert.iloc[perturbed_indices, :] = baseline_values
        
        perturbed_preds = automl.predict(X_pert)
        
        shift = wasserstein_distance(
            original_preds[perturbed_indices],
            perturbed_preds[perturbed_indices]
        )
        
        y_shifts_rep.append(shift)
        M_rep[j, perturbed_indices] = 1

    y_shifts_rep = np.array(y_shifts_rep)
    
    M_rep_train, M_rep_test, shift_rep_train, shift_rep_test = train_test_split(
        M_rep,
        y_shifts_rep,
        test_size=0.25,
        random_state=42
    )
    
    ridge_rep = Ridge(alpha=1.0)
    ridge_rep.fit(M_rep_train, shift_rep_train)
    
    shift_rep_pred = ridge_rep.predict(M_rep_test)
    rep_r2 = r2_score(shift_rep_test, shift_rep_pred)
    
    fidelity_runs.append(rep_r2)
    
    importance_rep = pd.Series(
        ridge_rep.coef_,
        index=X_train.index
    ).abs()
    
    importance_runs.append(importance_rep)


# Convert repeated importance scores to matrix
importance_runs_df = pd.DataFrame(importance_runs).T
importance_runs_df.columns = [f"run_{i+1}" for i in range(n_repeats)]

print("\nRepeated importance matrix shape:")
print(importance_runs_df.shape)


# ============================================================
# Step 3: Pairwise Stability Metrics Across Runs
# ============================================================
# We compute:
# 1. Pairwise Spearman correlation across importance maps
# 2. Pairwise Pearson correlation across importance maps
# 3. Pairwise Jaccard overlap between top-k important points
# ============================================================

pairwise_spearman = []
pairwise_pearson = []
pairwise_jaccard = []

for run_a, run_b in combinations(importance_runs_df.columns, 2):
    
    scores_a = importance_runs_df[run_a].values
    scores_b = importance_runs_df[run_b].values
    
    # Spearman stability
    sp_corr, _ = spearmanr(scores_a, scores_b)
    pairwise_spearman.append(sp_corr)
    
    # Pearson stability
    pr_corr, _ = pearsonr(scores_a, scores_b)
    pairwise_pearson.append(pr_corr)
    
    # Top-k Jaccard stability
    threshold_a = np.percentile(scores_a, 100 - top_percent_stability)
    threshold_b = np.percentile(scores_b, 100 - top_percent_stability)
    
    top_a = set(importance_runs_df.index[scores_a >= threshold_a])
    top_b = set(importance_runs_df.index[scores_b >= threshold_b])
    
    jaccard = len(top_a.intersection(top_b)) / len(top_a.union(top_b))
    pairwise_jaccard.append(jaccard)


stability_spearman_mean = np.nanmean(pairwise_spearman)
stability_spearman_std = np.nanstd(pairwise_spearman)

stability_pearson_mean = np.nanmean(pairwise_pearson)
stability_pearson_std = np.nanstd(pairwise_pearson)

stability_jaccard_mean = np.nanmean(pairwise_jaccard)
stability_jaccard_std = np.nanstd(pairwise_jaccard)

fidelity_repeat_mean = np.mean(fidelity_runs)
fidelity_repeat_std = np.std(fidelity_runs)

print("\n--- Geo-SMILE Stability Across Repeated Runs ---")
print(f"Repeated-run Fidelity R² Mean : {fidelity_repeat_mean:.4f}")
print(f"Repeated-run Fidelity R² Std  : {fidelity_repeat_std:.4f}")
print(f"Spearman Stability Mean       : {stability_spearman_mean:.4f}")
print(f"Spearman Stability Std        : {stability_spearman_std:.4f}")
print(f"Pearson Stability Mean        : {stability_pearson_mean:.4f}")
print(f"Pearson Stability Std         : {stability_pearson_std:.4f}")
print(f"Top-{top_percent_stability}% Jaccard Stability Mean: {stability_jaccard_mean:.4f}")
print(f"Top-{top_percent_stability}% Jaccard Stability Std : {stability_jaccard_std:.4f}")


stability_metrics_df = pd.DataFrame({
    "Category": [
        "Stability", "Stability",
        "Stability", "Stability",
        "Stability", "Stability",
        "Stability", "Stability"
    ],
    "Metric": [
        "Repeated-run Fidelity R² Mean",
        "Repeated-run Fidelity R² Std",
        "Spearman Stability Mean",
        "Spearman Stability Std",
        "Pearson Stability Mean",
        "Pearson Stability Std",
        f"Top-{top_percent_stability}% Jaccard Stability Mean",
        f"Top-{top_percent_stability}% Jaccard Stability Std"
    ],
    "Value": [
        fidelity_repeat_mean,
        fidelity_repeat_std,
        stability_spearman_mean,
        stability_spearman_std,
        stability_pearson_mean,
        stability_pearson_std,
        stability_jaccard_mean,
        stability_jaccard_std
    ]
})

display(stability_metrics_df)


# ============================================================
# Step 4: Point-Level Stability Summary
# ============================================================
# For each point, compute mean and standard deviation of
# Geo-SMILE importance across repeated runs.
# ============================================================

gdf_train["importance_repeat_mean"] = importance_runs_df.mean(axis=1).loc[gdf_train.index]
gdf_train["importance_repeat_std"] = importance_runs_df.std(axis=1).loc[gdf_train.index]

# Coefficient of variation
eps = 1e-12
gdf_train["importance_repeat_cv"] = (
    gdf_train["importance_repeat_std"] /
    (gdf_train["importance_repeat_mean"] + eps)
)

point_stability_summary_df = gdf_train[[
    "geo_importance_abs",
    "importance_repeat_mean",
    "importance_repeat_std",
    "importance_repeat_cv"
]].describe()

print("\n--- Point-Level Stability Summary ---")
display(point_stability_summary_df)


# ============================================================
# Step 5: Sparsity Metrics
# ============================================================
# Sparsity evaluates whether the explanation is concentrated
# in a small number of points or spread across many points.
# ============================================================

scores = geo_importance_abs.values.astype(float)
scores = np.nan_to_num(scores, nan=0.0)

n = len(scores)
eps = 1e-12

# Top-10% active ratio
q90 = np.percentile(scores, 90)
active_top10 = scores >= q90
top10_active_ratio = active_top10.sum() / n

# Hoyer sparsity:
# 0 = dense explanation
# 1 = maximally sparse explanation
l1_norm = np.sum(np.abs(scores))
l2_norm = np.sqrt(np.sum(scores ** 2))

if l2_norm > eps:
    hoyer_sparsity = (
        np.sqrt(n) - (l1_norm / l2_norm)
    ) / (np.sqrt(n) - 1)
else:
    hoyer_sparsity = np.nan

# Gini-style concentration
sorted_scores = np.sort(scores)
if np.sum(sorted_scores) > eps:
    cumulative = np.cumsum(sorted_scores)
    gini_concentration = (
        n + 1 - 2 * np.sum(cumulative) / cumulative[-1]
    ) / n
else:
    gini_concentration = np.nan

print("\n--- Geo-SMILE Sparsity Metrics ---")
print(f"Top-10% Active Ratio : {top10_active_ratio:.4f}")
print(f"Hoyer Sparsity       : {hoyer_sparsity:.4f}")
print(f"Gini Concentration   : {gini_concentration:.4f}")


sparsity_metrics_df = pd.DataFrame({
    "Category": ["Sparsity", "Sparsity", "Sparsity"],
    "Metric": [
        "Top-10% Active Ratio",
        "Hoyer Sparsity",
        "Gini Concentration"
    ],
    "Value": [
        top10_active_ratio,
        hoyer_sparsity,
        gini_concentration
    ]
})

display(sparsity_metrics_df)


# ============================================================
# Step 6: Entropy Metrics
# ============================================================
# Entropy evaluates how distributed or concentrated the
# importance scores are across points.
#
# Lower normalized entropy = more concentrated explanation
# Higher normalized entropy = more diffuse explanation
# ============================================================

score_sum = np.sum(scores)

if score_sum > eps:
    p = scores / score_sum
    p = p[p > 0]
    
    entropy = -np.sum(p * np.log(p))
    normalized_entropy = entropy / np.log(n)
else:
    entropy = np.nan
    normalized_entropy = np.nan

# Effective number of important points
# This converts entropy into an interpretable count-like measure
effective_num_points = np.exp(entropy) if not np.isnan(entropy) else np.nan
effective_point_ratio = effective_num_points / n if not np.isnan(entropy) else np.nan

print("\n--- Geo-SMILE Entropy Metrics ---")
print(f"Entropy                         : {entropy:.4f}")
print(f"Normalized Entropy              : {normalized_entropy:.4f}")
print(f"Effective Number of Points      : {effective_num_points:.2f}")
print(f"Effective Point Ratio           : {effective_point_ratio:.4f}")


entropy_metrics_df = pd.DataFrame({
    "Category": ["Entropy", "Entropy", "Entropy", "Entropy"],
    "Metric": [
        "Entropy",
        "Normalized Entropy",
        "Effective Number of Points",
        "Effective Point Ratio"
    ],
    "Value": [
        entropy,
        normalized_entropy,
        effective_num_points,
        effective_point_ratio
    ]
})

display(entropy_metrics_df)


# ============================================================
# Step 7: Combined Additional Evaluation Table
# ============================================================

additional_metrics_df = pd.concat([
    consistency_metrics_df,
    stability_metrics_df,
    sparsity_metrics_df,
    entropy_metrics_df
], ignore_index=True)

print("\n--- Additional Geo-SMILE Evaluation Metrics ---")
display(additional_metrics_df)


# ============================================================
# Step 8: Optional: Combine with Existing Evaluation Metrics
# ============================================================

if "evaluation_metrics_df" in globals():
    full_evaluation_metrics_df = pd.concat([
        evaluation_metrics_df,
        additional_metrics_df
    ], ignore_index=True)
    
    print("\n--- Full Geo-SMILE Evaluation Metrics ---")
    display(full_evaluation_metrics_df)
else:
    full_evaluation_metrics_df = additional_metrics_df


# ============================================================
# Step 9: Map - Repeated-Run Mean Importance
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))

gdf_train.plot(
    column="importance_repeat_mean",
    cmap="plasma",
    legend=True,
    markersize=50,
    ax=ax
)

ax.set_title("Geo-SMILE Repeated-Run Mean Importance", fontsize=15)
ax.set_xlabel("UTM_X")
ax.set_ylabel("UTM_Y")
ax.axis("equal")

plt.tight_layout()
plt.show()


# ============================================================
# Step 10: Map - Repeated-Run Importance Instability
# ============================================================
# High values indicate points whose importance varies more
# across repeated Geo-SMILE runs.
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))

gdf_train.plot(
    column="importance_repeat_std",
    cmap="magma",
    legend=True,
    markersize=50,
    ax=ax
)

ax.set_title("Geo-SMILE Repeated-Run Importance Instability", fontsize=15)
ax.set_xlabel("UTM_X")
ax.set_ylabel("UTM_Y")
ax.axis("equal")

plt.tight_layout()
plt.show()


# ============================================================
# Step 11: Map - Top 10% Sparse Important Points
# ============================================================

gdf_train["top10_sparse_important"] = active_top10.astype(int)

fig, ax = plt.subplots(figsize=(12, 8))

gdf_train.plot(
    ax=ax,
    markersize=18,
    alpha=0.20,
    color="lightgrey"
)

gdf_train[gdf_train["top10_sparse_important"] == 1].plot(
    ax=ax,
    markersize=70,
    color="red",
    alpha=0.85,
    label="Top 10% Geo-SMILE importance"
)

ax.legend()
ax.set_title("Geo-SMILE Sparse Important Points: Top 10%", fontsize=15)
ax.set_xlabel("UTM_X")
ax.set_ylabel("UTM_Y")
ax.axis("equal")

plt.tight_layout()
plt.show()


# ============================================================
# Step 12: Save Outputs
# ============================================================

geo_smile_additional_evaluation_results = {
    "consistency_metrics_df": consistency_metrics_df,
    "stability_metrics_df": stability_metrics_df,
    "sparsity_metrics_df": sparsity_metrics_df,
    "entropy_metrics_df": entropy_metrics_df,
    "additional_metrics_df": additional_metrics_df,
    "full_evaluation_metrics_df": full_evaluation_metrics_df,
    "importance_runs_df": importance_runs_df,
    "point_stability_summary_df": point_stability_summary_df,
    "gdf_train_updated": gdf_train
}

print("\nAdditional Geo-SMILE evaluation metrics completed successfully.")

