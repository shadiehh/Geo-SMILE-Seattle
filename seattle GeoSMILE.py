# ============================================================
# Geo-SMILE: Geo + Feature + Cell Explainability Pipeline
# Extension of SMILE (Aslansefat) to spatial cases
# Local explainer: feature importance computed per point
# ============================================================

# ============================================================
# Step 0: Imports
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from tqdm import tqdm
from scipy.stats import wasserstein_distance, spearmanr, pearsonr
from itertools import combinations

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

spatial_features = ["UTM_X", "UTM_Y"]
target = "log_price"

# geo_dist replaces separate UTM_X / UTM_Y in the model: a single scalar
# (standardised Euclidean distance from the dataset centroid) that encodes
# spatial location as a first-class feature.  This lets the feature branch
# produce a truly local per-point geo importance for each property —
# consistent with the Chicago dataset where UTM_X + UTM_Y were combined into
# one composite geo feature.
features = [
    "bathrooms", "sqft_living", "sqft_lot", "grade",
    "condition", "waterfront", "view", "age",
    "geo_dist"
]

required_cols = spatial_features + [f for f in features if f != "geo_dist"] + [target]
missing_cols = [c for c in required_cols if c not in data.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

data = data.dropna(subset=required_cols).reset_index(drop=True)

# Build geo_dist from all rows (centroid computed on full dataset).
# UTM_X and UTM_Y are retained in `data` for KMeans (Step 3) and mapping.
_cx = data["UTM_X"].mean()
_cy = data["UTM_Y"].mean()
_sc = np.sqrt(data["UTM_X"].var() + data["UTM_Y"].var())
data["geo_dist"] = (
    np.sqrt((data["UTM_X"] - _cx) ** 2 + (data["UTM_Y"] - _cy) ** 2) / (_sc + 1e-12)
)

X = data[features].copy()
y = data[target].copy()


# ============================================================
# Step 2: Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ============================================================
# Step 3: Spatial Groups via KMeans (Geo branch)
# ============================================================

coords_scaler = StandardScaler()
coords_train_scaled = coords_scaler.fit_transform(data.loc[X_train.index, spatial_features])

n_groups = 100
kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
spatial_group_labels = kmeans.fit_predict(coords_train_scaled)

print(f"\nSpatial groups: {n_groups}")
print("Group size summary:")
print(pd.Series(spatial_group_labels).value_counts().describe())


# ============================================================
# Step 4: Train AutoML Black-Box Model
# ============================================================

automl = AutoML()
automl.fit(
    X_train, y_train,
    task="regression",
    time_budget=20,
    metric="r2",
    verbose=0
)

train_preds = automl.predict(X_train)
test_preds  = automl.predict(X_test)

base_train_r2 = r2_score(y_train, train_preds)
base_test_r2  = r2_score(y_test,  test_preds)

print(f"\nBlack-box — Train R2: {base_train_r2:.4f} | Test R2: {base_test_r2:.4f}")

original_preds  = train_preds.copy()
baseline_values = X_train.median()


# ============================================================
# Step 5: Kernel Utilities
# ============================================================

KERNELS = ["rbf", "exponential", "inverse", "linear", "laplacian"]


def apply_kernel(distances, kernel_type, sigma=None):
    d = np.asarray(distances, dtype=float)
    if sigma is None:
        sigma = np.median(d) + 1e-12
    if kernel_type == "rbf":
        return np.exp(-(d ** 2) / (sigma ** 2))
    elif kernel_type == "exponential":
        return np.exp(-d / sigma)
    elif kernel_type == "inverse":
        return 1.0 / (1.0 + d / sigma)
    elif kernel_type == "linear":
        return np.maximum(0.0, 1.0 - d / (d.max() + 1e-12))
    elif kernel_type == "laplacian":
        return np.exp(-np.abs(d) / sigma)
    raise ValueError(f"Unknown kernel: {kernel_type}")


def select_best_kernel(M_tr, M_te, input_dist_tr, response_tr, response_te):
    """
    input_dist_tr : input-space distances -> locality weights (SMILE: perturbation distance)
    response_tr/te: output-space shifts   -> surrogate target (SMILE: prediction difference)
    Keeping these quantities separate is required by the SMILE locality principle.
    """
    results = {}
    for k in KERNELS:
        w = np.clip(apply_kernel(input_dist_tr, k), 1e-8, None)
        ridge = Ridge(alpha=1.0)
        ridge.fit(M_tr, response_tr, sample_weight=w)
        results[k] = r2_score(response_te, ridge.predict(M_te))
    best = max(results, key=results.get)
    return best, results


# ============================================================
# Step 6: Geo Perturbations (S sets, spatial group masking)
# ============================================================

n_points   = X_train.shape[0]
n_subsets  = 3000
min_groups = 5
max_groups = 10

M_geo        = np.zeros((n_subsets, n_points), dtype=np.int8)
y_shifts_geo = []

for j in tqdm(range(n_subsets), desc="Geo perturbations"):
    g   = np.random.randint(min_groups, max_groups + 1)
    sel = np.random.choice(n_groups, size=g, replace=False)
    idx = np.where(np.isin(spatial_group_labels, sel))[0]

    if len(idx) == 0:
        y_shifts_geo.append(0.0)
        continue

    X_pert = X_train.copy()
    X_pert.iloc[idx] = baseline_values.values
    pert_preds = automl.predict(X_pert)
    y_shifts_geo.append(wasserstein_distance(original_preds[idx], pert_preds[idx]))
    M_geo[j, idx] = 1

y_shifts_geo = np.array(y_shifts_geo)
print(f"\nGeo shifts — mean: {y_shifts_geo.mean():.4f}, std: {y_shifts_geo.std():.4f}")


# ============================================================
# Step 7: Kernel Selection — Geo Branch
# ============================================================
# Input distance = proportion of training points perturbed.
# This measures how far each perturbation is from the original
# spatial configuration in INPUT space, independently of the
# Wasserstein shift (the OUTPUT response).
# ============================================================

# Proportion of training points neutralised in each perturbation
geo_input_dist = M_geo.sum(axis=1).astype(float) / n_points

M_geo_tr, M_geo_te, sg_tr, sg_te, gid_tr, gid_te = train_test_split(
    M_geo, y_shifts_geo, geo_input_dist, test_size=0.25, random_state=42
)

best_k_geo, k_scores_geo = select_best_kernel(
    M_geo_tr, M_geo_te,
    gid_tr,           # INPUT distance -> weight
    sg_tr, sg_te      # OUTPUT Wasserstein shift -> response
)

print("\nKernel R2 — Geo branch:")
for k, v in sorted(k_scores_geo.items(), key=lambda x: -x[1]):
    marker = " <- best" if k == best_k_geo else ""
    print(f"  {k:<14} {v:.4f}{marker}")

geo_w_tr = np.clip(apply_kernel(gid_tr, best_k_geo), 1e-8, None)


# ============================================================
# Step 8: Weighted Ridge Surrogate — Geo Branch
# ============================================================

ridge_geo = Ridge(alpha=1.0)
ridge_geo.fit(M_geo_tr, sg_tr, sample_weight=geo_w_tr)  # weight=INPUT dist, response=OUTPUT Wasserstein

pred_geo_tr = ridge_geo.predict(M_geo_tr)
pred_geo_te = ridge_geo.predict(M_geo_te)

geo_fidelity = {
    "R2 Fidelity": r2_score(sg_te, pred_geo_te),
    "MAE":         mean_absolute_error(sg_te, pred_geo_te),
    "MSE":         mean_squared_error(sg_te, pred_geo_te),
    "L1":          mean_absolute_error(sg_te, pred_geo_te),
    "L2":          np.sqrt(mean_squared_error(sg_te, pred_geo_te)),
}

print(f"\nGeo surrogate fidelity (kernel={best_k_geo}):")
for k, v in geo_fidelity.items():
    print(f"  {k}: {v:.6f}")

geo_importance     = pd.Series(ridge_geo.coef_, index=X_train.index, name="geo_importance")
geo_importance_abs = geo_importance.abs()


# ============================================================
# Step 9: Feature Perturbations
# ============================================================
# K random binary masks over features.
# For each perturbation, store the full prediction vector so
# that local per-point feature importance can be computed
# without extra model calls in Step 12.
# ============================================================

n_feats = len(features)
K_feat  = 3000

M_feat               = np.zeros((K_feat, n_feats),  dtype=np.int8)
all_pert_preds_feat  = np.zeros((K_feat, n_points), dtype=np.float32)
y_shifts_feat        = []

for j in tqdm(range(K_feat), desc="Feature perturbations"):
    mask = np.random.randint(0, 2, size=n_feats)
    M_feat[j] = mask

    X_pert = X_train.copy()
    for ki, feat in enumerate(features):
        if mask[ki] == 0:
            X_pert[feat] = baseline_values[feat]

    pert_preds = automl.predict(X_pert)
    all_pert_preds_feat[j] = pert_preds
    y_shifts_feat.append(wasserstein_distance(original_preds, pert_preds))

y_shifts_feat = np.array(y_shifts_feat)
print(f"\nFeature shifts — mean: {y_shifts_feat.mean():.4f}, std: {y_shifts_feat.std():.4f}")


# ============================================================
# Step 10: Kernel Selection — Feature Branch
# ============================================================
# Input distance = normalised Hamming distance = proportion of
# features masked to baseline in each perturbation.
# This is computed from M_feat (INPUT space) independently of
# the Wasserstein shift (OUTPUT space), fixing the circularity
# present when the same quantity is used for both weight and
# response.
# ============================================================

# Proportion of features masked in each perturbation (input distance)
feat_hamming_dist = (M_feat == 0).sum(axis=1).astype(float) / n_feats

M_feat_tr, M_feat_te, sf_tr, sf_te, fhd_tr, fhd_te = train_test_split(
    M_feat, y_shifts_feat, feat_hamming_dist, test_size=0.25, random_state=42
)

best_k_feat, k_scores_feat = select_best_kernel(
    M_feat_tr, M_feat_te,
    fhd_tr,           # INPUT Hamming distance -> weight
    sf_tr, sf_te      # OUTPUT Wasserstein shift -> response
)

print("\nKernel R2 — Feature branch:")
for k, v in sorted(k_scores_feat.items(), key=lambda x: -x[1]):
    marker = " <- best" if k == best_k_feat else ""
    print(f"  {k:<14} {v:.4f}{marker}")

feat_w_tr = np.clip(apply_kernel(fhd_tr, best_k_feat), 1e-8, None)


# ============================================================
# Step 11: Global Weighted Ridge — Feature Branch (fidelity)
# ============================================================
# A single global surrogate is kept solely to report fidelity
# metrics comparable to the Geo branch. The actual feature
# importance used for explanation comes from Step 12 (local).
# ============================================================

ridge_feat_global = Ridge(alpha=1.0)
ridge_feat_global.fit(M_feat_tr, sf_tr, sample_weight=feat_w_tr)  # weight=INPUT Hamming, response=OUTPUT Wasserstein

pred_feat_tr = ridge_feat_global.predict(M_feat_tr)
pred_feat_te = ridge_feat_global.predict(M_feat_te)

feat_fidelity = {
    "R2 Fidelity": r2_score(sf_te, pred_feat_te),
    "MAE":         mean_absolute_error(sf_te, pred_feat_te),
    "MSE":         mean_squared_error(sf_te, pred_feat_te),
    "L1":          mean_absolute_error(sf_te, pred_feat_te),
    "L2":          np.sqrt(mean_squared_error(sf_te, pred_feat_te)),
}

print(f"\nFeature surrogate fidelity (kernel={best_k_feat}):")
for k, v in feat_fidelity.items():
    print(f"  {k}: {v:.6f}")


# ============================================================
# Step 12: Local Feature Importance per Point
# ============================================================
# For each training point i:
#   y_i[j] = predicted_preds[j][i] - original_preds[i]
#          = how much perturbation j changed THIS point's output
#   w_i[j] = kernel(|y_i[j]|)
#          = weight by local prediction sensitivity
#   Ridge(M_feat, y_i, w_i) -> coef_ = local feature importance for i
#
# Result: local_feat_imp_df  shape (n_points x n_features)
# ============================================================

local_feat_imp = np.zeros((n_points, n_feats), dtype=np.float32)

# Locality weights from INPUT Hamming distance — identical for every point.
# Only the response y_i (prediction change at point i) varies per point.
# This properly separates: INPUT distance -> weight, OUTPUT change -> response.
local_weights = np.clip(apply_kernel(feat_hamming_dist, best_k_feat), 1e-8, None)

for i in tqdm(range(n_points), desc="Local feature importance per point"):
    y_i = all_pert_preds_feat[:, i] - original_preds[i]   # OUTPUT response for point i

    ridge_i = Ridge(alpha=1.0)
    ridge_i.fit(M_feat, y_i, sample_weight=local_weights)  # INPUT weight, OUTPUT response
    local_feat_imp[i] = ridge_i.coef_

local_feat_imp_df     = pd.DataFrame(local_feat_imp,     index=X_train.index, columns=features)
local_feat_imp_abs_df = pd.DataFrame(np.abs(local_feat_imp), index=X_train.index, columns=features)

# Global summary = mean local importance across all points
feature_importance     = local_feat_imp_df.mean(axis=0)
feature_importance_abs = local_feat_imp_abs_df.mean(axis=0)

print("\nGlobal feature importance (mean of local):")
display(feature_importance_abs.sort_values(ascending=False).to_frame("mean_local_importance"))


# ============================================================
# Step 13: Cell Explainability
# ============================================================
# geo_norm[i]           — how geographically important is point i
# local_feat_norm[i, j] — how important is feature j at point i
# cell[i, j] = geo_norm[i] * local_feat_norm[i, j]
# Both dimensions vary spatially.
# ============================================================

_eps = 1e-12

geo_norm = (geo_importance_abs - geo_importance_abs.min()) / \
           (geo_importance_abs.max() - geo_importance_abs.min() + _eps)

# Normalize local feature importance per feature column to [0, 1]
feat_col_min = local_feat_imp_abs_df.min(axis=0)
feat_col_max = local_feat_imp_abs_df.max(axis=0)
local_feat_norm = (local_feat_imp_abs_df - feat_col_min) / \
                  (feat_col_max - feat_col_min + _eps)

cell_matrix = geo_norm.values[:, np.newaxis] * local_feat_norm.values
cell_df     = pd.DataFrame(cell_matrix, index=geo_norm.index, columns=features)

print("\nCell importance matrix shape:", cell_df.shape)


# ============================================================
# Step 14: GeoDataFrame
# ============================================================

data_train = data.loc[X_train.index].copy()
data_train["geo_importance"]     = geo_importance
data_train["geo_importance_abs"] = geo_importance_abs
data_train["geo_norm"]           = geo_norm

for feat in features:
    data_train[f"local_feat_{feat}"]        = local_feat_imp_abs_df[feat]
    data_train[f"local_feat_signed_{feat}"] = local_feat_imp_df[feat]
    data_train[f"cell_{feat}"]              = cell_df[feat]
    data_train[f"cell_signed_{feat}"]       = cell_df[feat] * np.sign(local_feat_imp_df[feat])

data_train["dominant_feature"] = local_feat_imp_abs_df.idxmax(axis=1)

data_train["geometry"] = gpd.points_from_xy(data_train["UTM_X"], data_train["UTM_Y"])
gdf_train = gpd.GeoDataFrame(data_train, geometry="geometry", crs="EPSG:32610")


# ============================================================
# Step 15: Map — Geo Importance (Absolute)
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))
gdf_train.plot(column="geo_importance_abs", cmap="plasma", legend=True, markersize=50, ax=ax)
ax.set_title("Geo-SMILE: Distributional Geo Importance (Geo Branch)", fontsize=15)
ax.set_xlabel("UTM_X"); ax.set_ylabel("UTM_Y"); ax.axis("equal")
plt.tight_layout(); plt.show()


# ============================================================
# Step 16: Map — Signed Distributional Geo Importance (GeoSHAPLY Fig 10a style)
# ============================================================

_geo_abs_max = geo_importance.abs().max()
fig, ax = plt.subplots(figsize=(12, 8))
gdf_train.plot(
    column="geo_importance", cmap="RdBu_r", legend=True, markersize=50, ax=ax,
    vmin=-_geo_abs_max, vmax=_geo_abs_max
)
ax.set_title("Geo-SMILE: Signed Distributional Geo Importance (Geo Branch)", fontsize=15)
ax.set_xlabel("UTM_X"); ax.set_ylabel("UTM_Y"); ax.axis("equal")
plt.tight_layout(); plt.show()


# ============================================================
# Step 16b: Map — Local Geo Importance per Point (geo_dist)
# ============================================================
# geo_dist is a first-class feature in the feature branch.
# Its local importance at each point i = how much that property's
# predicted price changes when its distance-from-centroid is masked —
# i.e. a truly per-point local geo importance, not a distributional one.
# ============================================================

_local_geo_abs_max = gdf_train["local_feat_signed_geo_dist"].abs().max()
fig, ax = plt.subplots(figsize=(12, 8))
gdf_train.plot(
    column="local_feat_signed_geo_dist", cmap="RdBu_r", legend=True, markersize=50, ax=ax,
    vmin=-_local_geo_abs_max, vmax=_local_geo_abs_max
)
ax.set_title("Geo-SMILE: Local Geo Importance per Point (GeoSHAPLY Fig 10a style)", fontsize=15)
ax.set_xlabel("UTM_X"); ax.set_ylabel("UTM_Y"); ax.axis("equal")
plt.tight_layout(); plt.show()

# Figure 10b equivalent: signed cell map for the strongest interacting feature
_strongest_cell_feat = cell_df.drop(columns=["geo_dist"]).mean(axis=0).idxmax()
_cell_abs_max = gdf_train[f"cell_signed_{_strongest_cell_feat}"].abs().max()
fig, ax = plt.subplots(figsize=(12, 8))
gdf_train.plot(
    column=f"cell_signed_{_strongest_cell_feat}", cmap="RdBu_r", legend=True,
    markersize=50, ax=ax, vmin=-_cell_abs_max, vmax=_cell_abs_max
)
ax.set_title(
    f"Geo-SMILE: {_strongest_cell_feat} × GEO Interaction (GeoSHAPLY Fig 10b style)",
    fontsize=15
)
ax.set_xlabel("UTM_X"); ax.set_ylabel("UTM_Y"); ax.axis("equal")
plt.tight_layout(); plt.show()


# ============================================================
# Step 17: Bar Chart — Global Feature Importance (mean local)
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))
feature_importance_abs.sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Geo-SMILE: Global Feature Importance (mean of local)", fontsize=14)
ax.set_xlabel("Mean Local Importance Score")
plt.tight_layout(); plt.show()


# ============================================================
# Step 17b: Beeswarm Summary — GeoSHAPLY Figure 8 Style
# ============================================================
# Y-axis: features sorted by mean |local importance| (ascending).
# X-axis: signed local importance for each training point.
# Dot colour: standardised feature value (red=high, blue=low).
# ============================================================

_sorted_feats = feature_importance_abs.sort_values(ascending=True).index.tolist()

fig, ax = plt.subplots(figsize=(12, max(6, len(_sorted_feats) * 0.55 + 1.5)))

np.random.seed(0)
for i, feat in enumerate(_sorted_feats):
    imp_vals  = local_feat_imp_df[feat].values
    raw_vals  = X_train[feat].values
    feat_norm = (raw_vals - raw_vals.min()) / (raw_vals.max() - raw_vals.min() + 1e-12)
    y_jitter  = i + np.random.uniform(-0.35, 0.35, size=len(imp_vals))
    ax.scatter(imp_vals, y_jitter, c=feat_norm, cmap="RdBu_r",
               alpha=0.55, s=10, vmin=0, vmax=1, linewidths=0)

ax.set_yticks(range(len(_sorted_feats)))
ax.set_yticklabels(_sorted_feats, fontsize=11)
ax.axvline(0, color="black", lw=0.8, ls="--")
ax.set_xlabel("GeoSMILE value (impact on model prediction)", fontsize=12)
ax.set_title("Feature Contribution Ranking — GeoSHAPLY Fig 8 Style", fontsize=14)

_sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
_sm.set_array([])
_cbar = fig.colorbar(_sm, ax=ax, pad=0.02, fraction=0.03)
_cbar.set_label("Feature value", fontsize=10)
_cbar.set_ticks([0, 1]); _cbar.set_ticklabels(["Low", "High"])
plt.tight_layout(); plt.show()


# ============================================================
# Step 18: Maps — Local Feature Importance per Feature
# ============================================================

ncols = 3
nrows = int(np.ceil(n_feats / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 5))
axes_flat = axes.flatten()

for idx, feat in enumerate(features):
    ax = axes_flat[idx]
    col = f"local_feat_signed_{feat}"
    _abs_max = gdf_train[col].abs().max()
    gdf_train.plot(
        column=col,
        cmap="RdBu_r",
        legend=True,
        markersize=30,
        ax=ax,
        vmin=-_abs_max,
        vmax=_abs_max,
    )
    ax.set_title(feat, fontsize=12)
    ax.axis("equal")
    ax.set_xlabel(""); ax.set_ylabel("")

for idx in range(n_feats, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle("Local Feature Importance — Signed Spatial Distribution (GeoSHAPLY Fig 9 Style)", fontsize=15, y=1.01)
plt.tight_layout(); plt.show()


# ============================================================
# Step 19: Map — Dominant Feature per Point
# ============================================================

unique_feats = list(local_feat_imp_abs_df.idxmax(axis=1).unique())
palette      = plt.cm.get_cmap("tab10", len(unique_feats))
color_map    = {f: palette(i) for i, f in enumerate(unique_feats)}

fig, ax = plt.subplots(figsize=(13, 9))

for feat, color in color_map.items():
    subset = gdf_train[gdf_train["dominant_feature"] == feat]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, markersize=55, label=feat, alpha=0.85)

ax.legend(title="Dominant Feature", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.set_title("Geo-SMILE: Dominant Feature Driver per Point", fontsize=15)
ax.set_xlabel("UTM_X"); ax.set_ylabel("UTM_Y"); ax.axis("equal")
plt.tight_layout(); plt.show()


# ============================================================
# Step 20: Heatmap — Cell Explainability (Top-30 Points)
# ============================================================

top30_idx = geo_importance_abs.nlargest(30).index

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(
    cell_df.loc[top30_idx],
    cmap="YlOrRd",
    ax=ax,
    xticklabels=True,
    yticklabels=False,
    cbar_kws={"label": "Cell Importance"}
)
ax.set_title("Cell Explainability: Top-30 Geo-Important Points x Features", fontsize=14)
ax.set_xlabel("Feature")
ax.set_ylabel("Point (top-30 by geo importance)")
plt.tight_layout(); plt.show()


# ============================================================
# Step 21: Map — Cell Importance for Top Feature
# ============================================================

top_feature = feature_importance_abs.idxmax()
print(f"\nMost spatially influential feature: {top_feature}")

fig, ax = plt.subplots(figsize=(12, 8))
gdf_train.plot(column=f"cell_{top_feature}", cmap="hot_r", legend=True, markersize=50, ax=ax)
ax.set_title(f"Cell Importance Map: {top_feature}", fontsize=15)
ax.set_xlabel("UTM_X"); ax.set_ylabel("UTM_Y"); ax.axis("equal")
plt.tight_layout(); plt.show()


# ============================================================
# Step 22: Wasserstein Shift Distributions
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(y_shifts_geo,  bins=30, kde=True, ax=axes[0])
axes[0].set_title("Wasserstein Shifts — Geo Branch"); axes[0].set_xlabel("Shift")
sns.histplot(y_shifts_feat, bins=30, kde=True, ax=axes[1], color="darkorange")
axes[1].set_title("Wasserstein Shifts — Feature Branch"); axes[1].set_xlabel("Shift")
plt.tight_layout(); plt.show()


# ============================================================
# Step 23: Observed vs Predicted Shifts (global surrogates)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, obs, pred, title in [
    (axes[0], sg_te, pred_geo_te,
     f"Geo Surrogate (R2={geo_fidelity['R2 Fidelity']:.3f}, kernel={best_k_geo})"),
    (axes[1], sf_te, pred_feat_te,
     f"Feature Surrogate — global (R2={feat_fidelity['R2 Fidelity']:.3f}, kernel={best_k_feat})")
]:
    ax.scatter(obs, pred, alpha=0.6)
    lo, hi = min(obs.min(), pred.min()), max(obs.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="red")
    ax.set_title(title); ax.set_xlabel("Observed Shift"); ax.set_ylabel("Predicted Shift")

plt.tight_layout(); plt.show()


# ============================================================
# Step 24: Surrogate Model Comparison
# ============================================================

SURROGATES = {
    "Ridge":         Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
    "SVR":           SVR(kernel="rbf", C=1.0),
    "KNN":           KNeighborsRegressor(n_neighbors=5),
    "XGBoost":       xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, verbosity=0, random_state=42),
    "LightGBM":      lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1),
}
WEIGHT_SUPPORT = {"Ridge", "Decision Tree", "Random Forest", "XGBoost", "LightGBM"}

geo_r2s  = {}
feat_r2s = {}

for name, model in SURROGATES.items():
    kw_geo  = {"sample_weight": geo_w_tr}  if name in WEIGHT_SUPPORT else {}
    kw_feat = {"sample_weight": feat_w_tr} if name in WEIGHT_SUPPORT else {}

    model.fit(M_geo_tr,  sg_tr, **kw_geo);  geo_r2s[name]  = r2_score(sg_te, model.predict(M_geo_te))
    model.fit(M_feat_tr, sf_tr, **kw_feat); feat_r2s[name] = r2_score(sf_te, model.predict(M_feat_te))

comparison_df = pd.DataFrame({
    "Model":      list(geo_r2s.keys()),
    "Geo R2":     list(geo_r2s.values()),
    "Feature R2": list(feat_r2s.values()),
}).sort_values("Geo R2", ascending=False).reset_index(drop=True)

print("\nSurrogate model comparison:")
display(comparison_df)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(comparison_df)); w = 0.35
ax.bar(x - w/2, comparison_df["Geo R2"],     w, label="Geo Branch",     color="steelblue")
ax.bar(x + w/2, comparison_df["Feature R2"], w, label="Feature Branch", color="darkorange")
ax.set_xticks(x); ax.set_xticklabels(comparison_df["Model"], rotation=45, ha="right")
ax.set_ylabel("Test R2"); ax.set_title("Surrogate Model Comparison — Geo vs Feature")
ax.legend(); ax.grid(axis="y")
plt.tight_layout(); plt.show()


# ============================================================
# Step 25: Stability — Geo Branch
# ============================================================

n_repeats_stab = 10
n_subsets_stab = 1000
top_pct_geo    = 20

geo_runs     = []
geo_fid_runs = []

for seed in tqdm(range(10, 10 + n_repeats_stab), desc="Geo stability runs"):
    np.random.seed(seed)
    M_r = np.zeros((n_subsets_stab, n_points), dtype=np.int8); ys = []

    for j in range(n_subsets_stab):
        g   = np.random.randint(min_groups, max_groups + 1)
        sel = np.random.choice(n_groups, size=g, replace=False)
        idx = np.where(np.isin(spatial_group_labels, sel))[0]
        if len(idx) == 0: ys.append(0.0); continue
        X_p = X_train.copy(); X_p.iloc[idx] = baseline_values.values
        pp  = automl.predict(X_p)
        ys.append(wasserstein_distance(original_preds[idx], pp[idx])); M_r[j, idx] = 1

    ys   = np.array(ys)
    gid_r = M_r.sum(axis=1).astype(float) / n_points   # input distance for this repeat
    Mt, Me, st, se, gid_tr_r, _ = train_test_split(
        M_r, ys, gid_r, test_size=0.25, random_state=42
    )
    w  = np.clip(apply_kernel(gid_tr_r, best_k_geo), 1e-8, None)  # INPUT dist → weight
    rr = Ridge(alpha=1.0); rr.fit(Mt, st, sample_weight=w)
    geo_fid_runs.append(r2_score(se, rr.predict(Me)))
    geo_runs.append(pd.Series(rr.coef_, index=X_train.index).abs())

geo_runs_df = pd.DataFrame(geo_runs).T
geo_runs_df.columns = [f"run_{i+1}" for i in range(n_repeats_stab)]

geo_sp, geo_pr, geo_jc = [], [], []
for a, b in combinations(geo_runs_df.columns, 2):
    sa, sb = geo_runs_df[a].values, geo_runs_df[b].values
    geo_sp.append(spearmanr(sa, sb)[0]); geo_pr.append(pearsonr(sa, sb)[0])
    ta = np.percentile(sa, 100 - top_pct_geo); tb = np.percentile(sb, 100 - top_pct_geo)
    A  = set(geo_runs_df.index[sa >= ta]); B = set(geo_runs_df.index[sb >= tb])
    geo_jc.append(len(A & B) / len(A | B) if (A | B) else 1.0)

geo_stability = {
    "Fidelity R2 Mean":                 np.mean(geo_fid_runs),
    "Fidelity R2 Std":                  np.std(geo_fid_runs),
    "Spearman Stability Mean":          np.nanmean(geo_sp),
    "Spearman Stability Std":           np.nanstd(geo_sp),
    "Pearson Stability Mean":           np.nanmean(geo_pr),
    "Pearson Stability Std":            np.nanstd(geo_pr),
    f"Jaccard Top-{top_pct_geo}% Mean": np.nanmean(geo_jc),
    f"Jaccard Top-{top_pct_geo}% Std":  np.nanstd(geo_jc),
}
print("\n--- Geo Stability ---")
for k, v in geo_stability.items(): print(f"  {k}: {v:.4f}")


# ============================================================
# Step 26: Stability — Feature Branch (local importance)
# ============================================================
# Each repeat re-runs K feature perturbations with a new seed,
# computes local importance per point, then compares the
# resulting (n_points x n_features) matrices pairwise.
# ============================================================

top_pct_feat   = 50
feat_runs      = []
feat_fid_runs  = []

for seed in tqdm(range(10, 10 + n_repeats_stab), desc="Feature stability runs"):
    np.random.seed(seed)
    M_r   = np.zeros((n_subsets_stab, n_feats),  dtype=np.int8)
    pp_r  = np.zeros((n_subsets_stab, n_points), dtype=np.float32)
    ys    = []

    for j in range(n_subsets_stab):
        mask = np.random.randint(0, 2, size=n_feats); M_r[j] = mask
        X_p  = X_train.copy()
        for ki, feat in enumerate(features):
            if mask[ki] == 0: X_p[feat] = baseline_values[feat]
        pp   = automl.predict(X_p); pp_r[j] = pp
        ys.append(wasserstein_distance(original_preds, pp))

    ys = np.array(ys)
    hd_r = (M_r == 0).sum(axis=1).astype(float) / n_feats  # input Hamming dist for this repeat
    Mt, Me, st, se, hd_tr_r, _ = train_test_split(M_r, ys, hd_r, test_size=0.25, random_state=42)
    w  = np.clip(apply_kernel(hd_tr_r, best_k_feat), 1e-8, None)
    rr = Ridge(alpha=1.0); rr.fit(Mt, st, sample_weight=w)
    feat_fid_runs.append(r2_score(se, rr.predict(Me)))

    # Local feature importance: INPUT Hamming weights, OUTPUT response per point
    loc_w = np.clip(apply_kernel(hd_r, best_k_feat), 1e-8, None)
    loc_r = np.zeros((n_points, n_feats), dtype=np.float32)
    for i in range(n_points):
        y_i = pp_r[:, i] - original_preds[i]
        rri = Ridge(alpha=1.0); rri.fit(M_r, y_i, sample_weight=loc_w)
        loc_r[i] = rri.coef_
    feat_runs.append(np.abs(loc_r).flatten())

feat_runs_arr = np.array(feat_runs)   # (n_repeats x (n_points * n_feats))

feat_sp, feat_pr, feat_jc = [], [], []
for a, b in combinations(range(n_repeats_stab), 2):
    sa, sb = feat_runs_arr[a], feat_runs_arr[b]
    feat_sp.append(spearmanr(sa, sb)[0]); feat_pr.append(pearsonr(sa, sb)[0])
    ta = np.percentile(sa, 100 - top_pct_feat); tb = np.percentile(sb, 100 - top_pct_feat)
    A  = set(np.where(sa >= ta)[0]); B = set(np.where(sb >= tb)[0])
    feat_jc.append(len(A & B) / len(A | B) if (A | B) else 1.0)

feat_stability = {
    "Fidelity R2 Mean":                  np.mean(feat_fid_runs),
    "Fidelity R2 Std":                   np.std(feat_fid_runs),
    "Spearman Stability Mean":           np.nanmean(feat_sp),
    "Spearman Stability Std":            np.nanstd(feat_sp),
    "Pearson Stability Mean":            np.nanmean(feat_pr),
    "Pearson Stability Std":             np.nanstd(feat_pr),
    f"Jaccard Top-{top_pct_feat}% Mean": np.nanmean(feat_jc),
    f"Jaccard Top-{top_pct_feat}% Std":  np.nanstd(feat_jc),
}
print("\n--- Feature Stability ---")
for k, v in feat_stability.items(): print(f"  {k}: {v:.4f}")


# ============================================================
# Step 27: Sparsity & Entropy — helper
# ============================================================

def sparsity_entropy_metrics(scores_raw, label=""):
    s = np.nan_to_num(np.asarray(scores_raw, dtype=float), nan=0.0)
    n = len(s); eps = 1e-12

    q90         = np.percentile(s, 90)
    top10_ratio = (s >= q90).sum() / n
    l1_n = np.sum(np.abs(s)); l2_n = np.sqrt(np.sum(s ** 2))
    hoyer = (np.sqrt(n) - l1_n / (l2_n + eps)) / (np.sqrt(n) - 1) if l2_n > eps else np.nan
    sorted_s = np.sort(s); cumul = np.cumsum(sorted_s)
    gini  = (n + 1 - 2 * np.sum(cumul) / (cumul[-1] + eps)) / n if cumul[-1] > eps else np.nan

    total = s.sum()
    if total > eps:
        p = s[s > 0] / total; ent = -np.sum(p * np.log(p))
        norm_ent = ent / np.log(n); eff_n = np.exp(ent); eff_r = eff_n / n
    else:
        ent = norm_ent = eff_n = eff_r = np.nan

    metrics = {
        "Top-10% Active Ratio": top10_ratio,
        "Hoyer Sparsity":       hoyer,
        "Gini Concentration":   gini,
        "Entropy":              ent,
        "Normalized Entropy":   norm_ent,
        "Effective N":          eff_n,
        "Effective Ratio":      eff_r,
    }
    print(f"\n--- Sparsity & Entropy ({label}) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if not (v is None or np.isnan(v)) else f"  {k}: NaN")
    return metrics


geo_sp_ent   = sparsity_entropy_metrics(geo_importance_abs.values,         label="Geo")
feat_sp_ent  = sparsity_entropy_metrics(feature_importance_abs.values,      label="Feature (mean local)")
local_sp_ent = sparsity_entropy_metrics(local_feat_imp_abs_df.values.flatten(), label="Local Feature (all)")
cell_sp_ent  = sparsity_entropy_metrics(cell_matrix.flatten(),               label="Cell")


# ============================================================
# Step 28: Combined Evaluation Table
# ============================================================

def _rows(metrics, branch, category):
    return [{"Branch": branch, "Category": category, "Metric": k, "Value": v}
            for k, v in metrics.items()]

rows = (
    _rows(geo_fidelity,  "Geo",     "Fidelity")        +
    _rows(feat_fidelity, "Feature", "Fidelity")        +
    _rows(geo_stability,  "Geo",    "Stability")       +
    _rows(feat_stability, "Feature","Stability")       +
    _rows(geo_sp_ent,    "Geo",     "Sparsity/Entropy") +
    _rows(feat_sp_ent,   "Feature", "Sparsity/Entropy") +
    _rows(cell_sp_ent,   "Cell",    "Sparsity/Entropy")
)

eval_df = pd.DataFrame(rows)
print("\n--- Full Geo-SMILE Evaluation Metrics ---")
display(eval_df)


# ============================================================
# Step 29: Stability Maps — Geo Branch
# ============================================================

gdf_train["geo_imp_mean"] = geo_runs_df.mean(axis=1).reindex(gdf_train.index)
gdf_train["geo_imp_std"]  = geo_runs_df.std(axis=1).reindex(gdf_train.index)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
gdf_train.plot(column="geo_imp_mean", cmap="plasma", legend=True, markersize=40, ax=axes[0])
axes[0].set_title("Geo Importance — Repeated-Run Mean", fontsize=13); axes[0].axis("equal")
gdf_train.plot(column="geo_imp_std",  cmap="magma",   legend=True, markersize=40, ax=axes[1])
axes[1].set_title("Geo Importance — Instability (Std)", fontsize=13); axes[1].axis("equal")
plt.tight_layout(); plt.show()


# ============================================================
# Step 30: Top-10% Sparse Important Points Map
# ============================================================

q90_geo = np.percentile(geo_importance_abs.values, 90)
gdf_train["top10_geo"] = (geo_importance_abs >= q90_geo).astype(int)

fig, ax = plt.subplots(figsize=(12, 8))
gdf_train[gdf_train["top10_geo"] == 0].plot(ax=ax, markersize=15, alpha=0.20, color="lightgrey")
gdf_train[gdf_train["top10_geo"] == 1].plot(ax=ax, markersize=70, color="red", alpha=0.85,
                                             label="Top 10% Geo Importance")
ax.legend()
ax.set_title("Geo-SMILE Sparse Important Points: Top 10%", fontsize=15)
ax.set_xlabel("UTM_X"); ax.set_ylabel("UTM_Y"); ax.axis("equal")
plt.tight_layout(); plt.show()


print("\nGeo-SMILE full pipeline completed successfully.")
