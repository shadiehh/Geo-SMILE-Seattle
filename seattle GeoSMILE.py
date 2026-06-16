# ============================================================
# Geo-SMILE: Geo + Feature + Geo-Player Co-Salience Explainability Pipeline
# Extension of SMILE (Aslansefat) to spatial cases
# Local explainer: player importance computed per property
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
import seaborn as sns

try:
    import contextily as ctx
except ImportError:
    import subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "pip", "install", "-q", "contextily"], check=True)
    import contextily as ctx

from tqdm import tqdm
from scipy.stats import wasserstein_distance, spearmanr, pearsonr
from statsmodels.nonparametric.smoothers_lowess import lowess
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

# All model features. UTM_X and UTM_Y enter the model as separate coordinates;
# they are treated as ONE joint "location" player during SMILE explanation
# (matching the Chicago design: location = [x_coord, y_coord] as a joint player).
features = [
    "bathrooms", "sqft_living", "sqft_lot", "grade",
    "condition", "waterfront", "view", "age",
    "UTM_X", "UTM_Y"
]

# SMILE players: non-spatial features are individual players;
# UTM_X + UTM_Y form a single joint "location" player.
non_spatial_feats = [f for f in features if f not in spatial_features]
players   = non_spatial_feats + ["location"]
n_players = len(players)

required_cols = features + [target]
missing_cols  = [c for c in required_cols if c not in data.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

data = data.dropna(subset=required_cols).reset_index(drop=True)
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

# baseline_values derived from training data only — no leakage
baseline_values = X_train.median()


# ============================================================
# Step 3: Spatial Groups via KMeans (Geo branch)
# ============================================================

coords_scaler       = StandardScaler()
coords_train_scaled = coords_scaler.fit_transform(data.loc[X_train.index, spatial_features])

n_groups = 100
kmeans   = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
spatial_group_labels = kmeans.fit_predict(coords_train_scaled)

print(f"\nSpatial groups: {n_groups}")
print(pd.Series(spatial_group_labels).value_counts().describe())


# ============================================================
# Step 4: Train AutoML Black-Box Model
# ============================================================

automl = AutoML()
automl.fit(X_train, y_train, task="regression", time_budget=20, metric="r2", verbose=0, seed=42)

train_preds   = automl.predict(X_train)
test_preds    = automl.predict(X_test)
original_preds = train_preds.copy()

print(f"\nBlack-box — Train R2: {r2_score(y_train, train_preds):.4f} | "
      f"Test R2: {r2_score(y_test, test_preds):.4f}")


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


def select_best_kernel(M_tr, M_va, input_dist_tr, response_tr, response_va):
    """
    Kernel selected on validation set only; caller reports fidelity on a
    separate test set so there is no selection bias in the final metric.
    input_dist_tr : input-space distances  → locality weights
    response_tr/va: output-space shifts    → surrogate target
    """
    results = {}
    for k in KERNELS:
        w = np.clip(apply_kernel(input_dist_tr, k), 1e-8, None)
        ridge = Ridge(alpha=1.0)
        ridge.fit(M_tr, response_tr, sample_weight=w)
        results[k] = r2_score(response_va, ridge.predict(M_va))
    best = max(results, key=results.get)
    return best, results


# ============================================================
# Step 6: Geo Perturbations — Group-Level Mask
# ============================================================
# M_geo ∈ {0,1}^(S×G): each column is one KMeans spatial group.
# Group-level mask ensures each group has its own identifiable Ridge
# coefficient (properties within the same cluster always share the
# same column, so using n_points columns would be non-identifiable).
#
# ONLY the spatial coordinates (UTM_X, UTM_Y) are neutralised for
# properties in selected groups; all other features remain at their
# real observed values. This isolates the geographic contribution
# rather than measuring complete-profile replacement sensitivity.
# ============================================================

np.random.seed(42)  # reproducibility of the main (non-stability) perturbation draws

n_points   = X_train.shape[0]
n_subsets  = 3000
min_groups = 5
max_groups = 10

M_geo        = np.zeros((n_subsets, n_groups), dtype=np.int8)
y_shifts_geo = []

for j in tqdm(range(n_subsets), desc="Geo perturbations"):
    g   = np.random.randint(min_groups, max_groups + 1)
    sel = np.random.choice(n_groups, size=g, replace=False)
    idx = np.where(np.isin(spatial_group_labels, sel))[0]

    if len(idx) == 0:
        y_shifts_geo.append(0.0)
        continue

    X_pert = X_train.copy()
    X_pert.loc[X_train.index[idx], spatial_features] = baseline_values[spatial_features].values
    pert_preds = automl.predict(X_pert)
    y_shifts_geo.append(wasserstein_distance(original_preds[idx], pert_preds[idx]))
    M_geo[j, sel] = 1

y_shifts_geo = np.array(y_shifts_geo)
print(f"\nGeo shifts — mean: {y_shifts_geo.mean():.4f}, std: {y_shifts_geo.std():.4f}")


# ============================================================
# Step 7: Kernel Selection — Geo Branch (3-way split)
# ============================================================
# geo_input_dist = proportion of spatial GROUPS selected per perturbation
# (not proportion of points — the group is the perturbation unit). This is a
# coalition-size / mask-space distance (how much of the group-mask vector
# differs), NOT a geographic proximity measure — it carries no notion of how
# close two selected groups are to each other on the map.
# Three-way split: kernel chosen on validation, fidelity on test.
# ============================================================

geo_input_dist = M_geo.sum(axis=1).astype(float) / n_groups

all_geo_idx  = np.arange(n_subsets)
geo_tr_idx, geo_tmp_idx = train_test_split(all_geo_idx, test_size=0.40, random_state=42)
geo_va_idx, geo_te_idx  = train_test_split(geo_tmp_idx, test_size=0.50, random_state=42)

M_geo_tr  = M_geo[geo_tr_idx];  M_geo_va  = M_geo[geo_va_idx];  M_geo_te  = M_geo[geo_te_idx]
sg_tr     = y_shifts_geo[geo_tr_idx]; sg_va = y_shifts_geo[geo_va_idx]; sg_te = y_shifts_geo[geo_te_idx]
gid_tr    = geo_input_dist[geo_tr_idx]; gid_va = geo_input_dist[geo_va_idx]

best_k_geo, k_scores_geo = select_best_kernel(
    M_geo_tr, M_geo_va, gid_tr, sg_tr, sg_va
)

print("\nKernel R2 — Geo branch (validation set):")
for k, v in sorted(k_scores_geo.items(), key=lambda x: -x[1]):
    print(f"  {k:<14} {v:.4f}{' <- best' if k == best_k_geo else ''}")

geo_w_tr = np.clip(apply_kernel(gid_tr, best_k_geo), 1e-8, None)


# ============================================================
# Step 8: Weighted Ridge Surrogate — Geo Branch
# ============================================================
# n_groups coefficients, one per spatial cluster.
# Each training property is assigned its cluster's coefficient →
# "property-indexed distributional geo score" (not a locally fitted value).
# The kernel was chosen on the validation split (Step 7); once it is fixed,
# the validation perturbations carry no further selection risk, so the FINAL
# coefficients are fit on train+validation combined (80% of n_subsets) rather
# than train alone (60%). Fidelity is still reported on the untouched test
# set, evaluated exactly once.
# ============================================================

M_geo_trva = np.vstack([M_geo_tr, M_geo_va])
sg_trva    = np.concatenate([sg_tr, sg_va])
gid_trva   = np.concatenate([gid_tr, gid_va])
geo_w_trva = np.clip(apply_kernel(gid_trva, best_k_geo), 1e-8, None)

ridge_geo = Ridge(alpha=1.0)
ridge_geo.fit(M_geo_trva, sg_trva, sample_weight=geo_w_trva)

pred_geo_te = ridge_geo.predict(M_geo_te)

geo_fidelity = {
    "R2 Fidelity": r2_score(sg_te, pred_geo_te),
    "MAE":         mean_absolute_error(sg_te, pred_geo_te),
    "MSE":         mean_squared_error(sg_te, pred_geo_te),
    "L1":          mean_absolute_error(sg_te, pred_geo_te),
    "L2":          np.sqrt(mean_squared_error(sg_te, pred_geo_te)),
}
print(f"\nGeo surrogate fidelity (kernel={best_k_geo}, test set):")
for k, v in geo_fidelity.items():
    print(f"  {k}: {v:.6f}")

geo_group_importance     = ridge_geo.coef_              # shape (n_groups,)
geo_group_importance_abs = np.abs(geo_group_importance)  # MAIN geo result, group-level

# y_shifts_geo (the surrogate target) is a Wasserstein DISTANCE — strictly
# non-negative. A Ridge fit on a non-negative target can still produce signed
# coefficients, but the sign only reflects how the linear model partitions the
# magnitude of the shift across groups; it does NOT indicate whether a group
# raises or lowers price. Only the ABSOLUTE coefficient — the "spatial-group
# distributional sensitivity" — is used as the geo result everywhere downstream
# (geo_norm, co-salience, stability, sparsity/entropy). The signed series below
# is kept only for reference and is never interpreted directionally.
geo_importance = pd.Series(
    geo_group_importance[spatial_group_labels],          # map group → property (reference only)
    index=X_train.index,
    name="geo_importance"
)
geo_importance_abs = pd.Series(
    geo_group_importance_abs[spatial_group_labels],      # map group → property
    index=X_train.index,
    name="geo_importance_abs"
)


# ============================================================
# Step 9: Feature Perturbations — Player-Based Masking
# ============================================================
# Players: non-spatial features individual; UTM_X+UTM_Y as "location" player.
# Sign convention (z_k = 1 → player k REMOVED, consistent with Chicago):
#   removal[k] = 1 → player k neutralised to baseline
#   removal[k] = 0 → player k at real value
# Response: r_ik = f(x_i) - f(x_i^(k))   (original minus perturbed).
#   Positive Ridge coefficient → removing this player tends to lower the
#   predicted price → the player contributes positively at this property.
# ============================================================

K_feat = 3000

M_feat               = np.zeros((K_feat, n_players), dtype=np.int8)
all_pert_preds_feat  = np.zeros((K_feat, n_points),  dtype=np.float32)
y_shifts_feat        = []

for j in tqdm(range(K_feat), desc="Feature perturbations"):
    removal = np.random.randint(0, 2, size=n_players)   # 1 = removed
    M_feat[j] = removal

    X_pert = X_train.copy()
    for ki, player in enumerate(players):
        if removal[ki] == 1:
            if player == "location":
                X_pert[spatial_features] = baseline_values[spatial_features].values
            else:
                X_pert[player] = baseline_values[player]

    pert_preds = automl.predict(X_pert)
    all_pert_preds_feat[j] = pert_preds
    y_shifts_feat.append(wasserstein_distance(original_preds, pert_preds))

y_shifts_feat = np.array(y_shifts_feat)
print(f"\nFeature shifts — mean: {y_shifts_feat.mean():.4f}, std: {y_shifts_feat.std():.4f}")


# ============================================================
# Step 10: Kernel Selection — Feature Branch (3-way split)
# ============================================================
# Hamming distance = proportion of players REMOVED per perturbation.
# ============================================================

feat_hamming_dist = M_feat.sum(axis=1).astype(float) / n_players

all_feat_idx = np.arange(K_feat)
feat_tr_idx, feat_tmp_idx = train_test_split(all_feat_idx, test_size=0.40, random_state=42)
feat_va_idx, feat_te_idx  = train_test_split(feat_tmp_idx, test_size=0.50, random_state=42)

M_feat_tr = M_feat[feat_tr_idx]; M_feat_va = M_feat[feat_va_idx]; M_feat_te = M_feat[feat_te_idx]
sf_tr  = y_shifts_feat[feat_tr_idx]; sf_va = y_shifts_feat[feat_va_idx]; sf_te = y_shifts_feat[feat_te_idx]
fhd_tr = feat_hamming_dist[feat_tr_idx]; fhd_va = feat_hamming_dist[feat_va_idx]

best_k_feat, k_scores_feat = select_best_kernel(
    M_feat_tr, M_feat_va, fhd_tr, sf_tr, sf_va
)

print("\nKernel R2 — Feature branch (validation set):")
for k, v in sorted(k_scores_feat.items(), key=lambda x: -x[1]):
    print(f"  {k:<14} {v:.4f}{' <- best' if k == best_k_feat else ''}")

feat_w_tr = np.clip(apply_kernel(fhd_tr, best_k_feat), 1e-8, None)


# ============================================================
# Step 11: Global Weighted Ridge — Feature Branch (distributional fidelity)
# ============================================================
# Kept for Wasserstein-level fidelity comparison with the geo branch.
# Fidelity reported on the held-out test set (not the validation set).
# ============================================================

ridge_feat_global = Ridge(alpha=1.0)
ridge_feat_global.fit(M_feat_tr, sf_tr, sample_weight=feat_w_tr)

pred_feat_tr = ridge_feat_global.predict(M_feat_tr)
pred_feat_te = ridge_feat_global.predict(M_feat_te)

feat_fidelity = {
    "R2 Fidelity": r2_score(sf_te, pred_feat_te),
    "MAE":         mean_absolute_error(sf_te, pred_feat_te),
    "MSE":         mean_squared_error(sf_te, pred_feat_te),
    "L1":          mean_absolute_error(sf_te, pred_feat_te),
    "L2":          np.sqrt(mean_squared_error(sf_te, pred_feat_te)),
}
print(f"\nGlobal feature surrogate fidelity (kernel={best_k_feat}, test set):")
for k, v in feat_fidelity.items():
    print(f"  {k}: {v:.6f}")


# ============================================================
# Step 12: Local Player Importance per Property + Local Fidelity
# ============================================================
# For each training property i, a separate Ridge is fitted using
# TRAINING perturbations only, then evaluated on TEST perturbations.
# Response: r_ik = f(x_i) - f(x_i^(k))   (original minus perturbed).
# Locality weights from INPUT Hamming distance of training perturbations.
# Local fidelity: per-property R² on the held-out test perturbations.
# If a property's held-out responses are (numerically) constant — e.g. every
# test perturbation produced the same shift for that property — R² is
# undefined, not "perfect" or "zero". Such properties are flagged NaN rather
# than silently scored, and excluded from the summary stats below.
# ============================================================

_ZERO_VAR_TOL = 1e-10

local_feat_imp   = np.zeros((n_points, n_players), dtype=np.float32)
local_r2_scores  = np.full(n_points, np.nan, dtype=np.float32)

local_weights_tr = np.clip(apply_kernel(fhd_tr, best_k_feat), 1e-8, None)

n_zero_var_skipped = 0
for i in tqdm(range(n_points), desc="Local player importance per property"):
    y_i_tr = original_preds[i] - all_pert_preds_feat[feat_tr_idx, i]  # original − perturbed
    y_i_te = original_preds[i] - all_pert_preds_feat[feat_te_idx, i]

    ridge_i = Ridge(alpha=1.0)
    ridge_i.fit(M_feat_tr, y_i_tr, sample_weight=local_weights_tr)
    local_feat_imp[i] = ridge_i.coef_

    if np.var(y_i_te) > _ZERO_VAR_TOL:
        local_r2_scores[i] = r2_score(y_i_te, ridge_i.predict(M_feat_te))
    else:
        n_zero_var_skipped += 1

local_feat_imp_df     = pd.DataFrame(local_feat_imp,          index=X_train.index, columns=players)
local_feat_imp_abs_df = pd.DataFrame(np.abs(local_feat_imp),  index=X_train.index, columns=players)
local_r2_series       = pd.Series(local_r2_scores, index=X_train.index)

feature_importance     = local_feat_imp_df.mean(axis=0)
feature_importance_abs = local_feat_imp_abs_df.mean(axis=0)

print("\nGlobal player importance (mean of local):")
display(feature_importance_abs.sort_values(ascending=False).to_frame("mean_local_importance"))

print(f"\nLocal fidelity (per-property R² on held-out test perturbations):")
print(f"  Properties with zero-variance held-out response (R² undefined, excluded): {n_zero_var_skipped}")
print(f"  Median : {np.nanmedian(local_r2_scores):.4f}")
print(f"  Mean   : {np.nanmean(local_r2_scores):.4f}  Std: {np.nanstd(local_r2_scores):.4f}")
print(f"  IQR    : {np.nanpercentile(local_r2_scores,25):.4f} – "
      f"{np.nanpercentile(local_r2_scores,75):.4f}")
_valid_r2 = local_r2_scores[~np.isnan(local_r2_scores)]
print(f"  Prop > 0.50 R²: {(_valid_r2 > 0.50).mean():.2%}  (of properties with defined R²)")


# ============================================================
# Step 13: Geo–Player Co-Salience Score
# ============================================================
# geo_norm[i]           — normalised spatial-group distributional sensitivity [0,1]
# local_feat_norm[i, j] — normalised local player importance at property i [0,1]
# cosalience[i, j]      = geo_norm[i] × local_feat_norm[i, j]
# This is a multiplicative score: a cell is high only when BOTH the property's
# geo sensitivity AND the player's local importance are high. It is a derived
# co-salience score, NOT a statistically estimated interaction effect.
# ============================================================

_eps = 1e-12

geo_norm = (geo_importance_abs - geo_importance_abs.min()) / \
           (geo_importance_abs.max() - geo_importance_abs.min() + _eps)

feat_col_min = local_feat_imp_abs_df.min(axis=0)
feat_col_max = local_feat_imp_abs_df.max(axis=0)
local_feat_norm = (local_feat_imp_abs_df - feat_col_min) / \
                  (feat_col_max - feat_col_min + _eps)

cell_matrix = geo_norm.values[:, np.newaxis] * local_feat_norm.values
cell_df     = pd.DataFrame(cell_matrix, index=geo_norm.index, columns=players)

print("\nGeo–player co-salience matrix shape:", cell_df.shape)


# ============================================================
# Step 14: GeoDataFrame
# ============================================================

data_train = data.loc[X_train.index].copy()
data_train["geo_importance"]     = geo_importance
data_train["geo_importance_abs"] = geo_importance_abs
data_train["geo_norm"]           = geo_norm
data_train["local_fidelity_r2"]  = local_r2_series

for player in players:
    data_train[f"local_{player}"]        = local_feat_imp_abs_df[player]
    data_train[f"local_signed_{player}"] = local_feat_imp_df[player]
    data_train[f"cell_{player}"]         = cell_df[player]
    # cell_signed inherits its sign ENTIRELY from the local player coefficient.
    # The geo factor in cell_df is abs/magnitude-only (geo_norm, Step 13) and
    # therefore contributes no direction of its own — it can only scale the
    # magnitude of the (already-signed) local contribution up or down.
    data_train[f"cell_signed_{player}"]  = cell_df[player] * np.sign(local_feat_imp_df[player])

data_train["dominant_player"] = local_feat_imp_abs_df.idxmax(axis=1)

data_train["geometry"] = gpd.points_from_xy(data_train["UTM_X"], data_train["UTM_Y"])
gdf_train = gpd.GeoDataFrame(data_train, geometry="geometry", crs="EPSG:32610")


def add_basemap(ax, source=None):
    """Add a real city basemap (OpenStreetMap / CARTO tiles) beneath the
    plotted points, reprojected on the fly to the GeoDataFrame's CRS
    (inspired by GeoShapley-style location maps). Fails silently if no
    internet access."""
    if source is None:
        source = ctx.providers.CartoDB.Positron
    try:
        ctx.add_basemap(ax, crs=gdf_train.crs.to_string(), source=source, attribution_size=6)
    except Exception as e:
        print(f"  (basemap unavailable — {e})")


# ============================================================
# Step 15: Map — Spatial-Group Distributional Sensitivity
# ============================================================
# Each property displays the |Ridge coefficient| of its KMeans spatial group.
# The score measures each group's contribution to the (non-negative)
# Wasserstein shift when ONLY the spatial coordinates are neutralised.
# Only the ABSOLUTE value is shown — see the note in Step 8 on why the sign
# of this coefficient is not directionally interpretable.
# This is a GROUP-level score broadcast to member properties, NOT a value
# locally fitted per property.
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))
gdf_train.plot(column="geo_importance_abs", cmap="plasma", legend=True,
               markersize=50, alpha=0.85, ax=ax)
add_basemap(ax)
ax.set_title("Geo-SMILE: Spatial-Group Distributional Sensitivity", fontsize=14)
ax.set_aspect("equal"); ax.set_axis_off()
plt.tight_layout(); plt.show()


# ============================================================
# Step 16b: Map — Local Location Player Importance
# ============================================================
# "location" player = joint UTM_X + UTM_Y treated as one SMILE player.
# Its importance at property i is the Ridge coefficient from the LOCAL
# surrogate fitted separately for each property (truly locally fitted,
# unlike the spatial-group sensitivity score in Step 15). Unlike that score,
# this one IS signed meaningfully: the response here is
# original − perturbed prediction (not a non-negative distance).
# ============================================================

_loc_abs_max = gdf_train["local_signed_location"].abs().max()
fig, ax = plt.subplots(figsize=(12, 8))
gdf_train.plot(
    column="local_signed_location", cmap="RdBu_r", legend=True, markersize=50, alpha=0.85, ax=ax,
    vmin=-_loc_abs_max, vmax=_loc_abs_max
)
add_basemap(ax)
ax.set_title("Geo-SMILE: Local Location Player Importance per Property", fontsize=14)
ax.set_aspect("equal"); ax.set_axis_off()
plt.tight_layout(); plt.show()

# Geo–player co-salience map for the player with the strongest mean co-salience with location
_non_loc = [p for p in players if p != "location"]
_strongest_cell = cell_df[_non_loc].mean(axis=0).idxmax()
_cell_abs_max   = gdf_train[f"cell_signed_{_strongest_cell}"].abs().max()
fig, ax = plt.subplots(figsize=(12, 8))
gdf_train.plot(
    column=f"cell_signed_{_strongest_cell}", cmap="RdBu_r", legend=True,
    markersize=50, alpha=0.85, ax=ax, vmin=-_cell_abs_max, vmax=_cell_abs_max
)
add_basemap(ax)
ax.set_title(f"Geo-SMILE: {_strongest_cell} × Location Co-Salience", fontsize=14)
ax.set_aspect("equal"); ax.set_axis_off()
plt.tight_layout(); plt.show()


# ============================================================
# Step 17: Bar Chart — Global Player Importance (mean local)
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))
feature_importance_abs.sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Geo-SMILE: Global Player Importance (mean of local)", fontsize=14)
ax.set_xlabel("Mean Local Importance Score")
plt.tight_layout(); plt.show()


# ============================================================
# Step 17b: Beeswarm Summary of Local Player Contributions
# ============================================================
# Inspired by GeoShapley-style beeswarm summary plots.
# Y-axis: players sorted by mean |local importance| (ascending).
# X-axis: signed local importance value for each training property.
# Dot colour: standardised player value (red=high, blue=low).
# "location" is 2-D (UTM_X + UTM_Y jointly) and has no single scalar value to
# colour by — UTM_X alone would be an arbitrary, misleading proxy (UTM_Y or a
# distance-from-center measure would colour the same row differently). That
# row is therefore drawn in a neutral grey instead of the value colormap.
# ============================================================

_sorted_players = feature_importance_abs.sort_values(ascending=True).index.tolist()
fig, ax = plt.subplots(figsize=(12, max(6, len(_sorted_players) * 0.55 + 1.5)))

np.random.seed(0)
for i, player in enumerate(_sorted_players):
    imp_vals  = local_feat_imp_df[player].values
    y_jitter  = i + np.random.uniform(-0.35, 0.35, size=len(imp_vals))
    if player == "location":
        ax.scatter(imp_vals, y_jitter, color="grey", alpha=0.55, s=10, linewidths=0)
        continue
    raw_vals  = X_train[player].values
    feat_norm = (raw_vals - raw_vals.min()) / (raw_vals.max() - raw_vals.min() + 1e-12)
    ax.scatter(imp_vals, y_jitter, c=feat_norm, cmap="RdBu_r",
               alpha=0.55, s=10, vmin=0, vmax=1, linewidths=0)

ax.set_yticks(range(len(_sorted_players)))
ax.set_yticklabels(_sorted_players, fontsize=11)
ax.axvline(0, color="black", lw=0.8, ls="--")
ax.set_xlabel("GeoSMILE value (impact on model prediction)", fontsize=12)
ax.set_title("Player Contribution Ranking", fontsize=14)

_sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
_sm.set_array([])
_cbar = fig.colorbar(_sm, ax=ax, pad=0.02, fraction=0.03)
_cbar.set_label("Feature value", fontsize=10)
_cbar.set_ticks([0, 1]); _cbar.set_ticklabels(["Low", "High"])
plt.tight_layout(); plt.show()


# ============================================================
# Step 17c: Local Player Removal Contributions Across Feature Values
# ============================================================
# Inspired by GeoShapley-style dependence plots.
# These show how the LOCAL baseline-removal contribution (Step 12's Ridge
# coefficient for that player at that property) varies with the property's
# observed feature value. They are NOT marginal/causal effect estimates —
# each point is a per-property local-surrogate coefficient, not a controlled
# perturbation of that single feature in isolation.
# For each non-spatial player: x-axis = raw feature value, y-axis = estimated
# price change associated with neutralising that player
# (100 * (exp(local_signed_importance) - 1), since target = log_price).
# Red dashed line = LOWESS trend through the points; grey band = 95%
# bootstrap CI of that trend.
# "location" is 2-D and is shown as a map (Steps 16b/18), not a 1-D plot here.
# ============================================================

def _dependence_plot(ax, raw_vals, signed_local_imp, label, n_boot=80, frac=0.5):
    raw_vals = np.asarray(raw_vals, dtype=float)
    pct_change = 100.0 * (np.exp(np.asarray(signed_local_imp, dtype=float)) - 1.0)

    order   = np.argsort(raw_vals)
    x_sorted = raw_vals[order]
    y_sorted = pct_change[order]
    n = len(x_sorted)

    fit  = lowess(y_sorted, x_sorted, frac=frac, return_sorted=True)
    grid = fit[:, 0]

    boot_curves = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        bx, by = x_sorted[idx], y_sorted[idx]
        s = np.argsort(bx)
        try:
            bf = lowess(by[s], bx[s], frac=frac, return_sorted=True)
            boot_curves.append(np.interp(grid, bf[:, 0], bf[:, 1]))
        except Exception:
            continue

    ax.scatter(x_sorted, y_sorted, s=8, alpha=0.35, color="#3477b5", linewidths=0)
    if boot_curves:
        boot_curves = np.array(boot_curves)
        lower = np.percentile(boot_curves, 2.5, axis=0)
        upper = np.percentile(boot_curves, 97.5, axis=0)
        ax.fill_between(grid, lower, upper, color="grey", alpha=0.3)
    ax.plot(grid, fit[:, 1], color="red", linestyle="--", lw=1.5)
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Est. price change from\nplayer neutralisation (%)", fontsize=10)


_dep_players = [p for p in players if p != "location"]
_ncols_dep   = 4
_nrows_dep   = int(np.ceil(len(_dep_players) / _ncols_dep))

np.random.seed(0)
fig, axes = plt.subplots(_nrows_dep, _ncols_dep, figsize=(22, _nrows_dep * 4.5))
axes_flat = np.atleast_1d(axes).flatten()

for idx, player in enumerate(_dep_players):
    _dependence_plot(axes_flat[idx], X_train[player].values, local_feat_imp_df[player].values, player)

for idx in range(len(_dep_players), len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle(
    "Local Player Removal Contributions Across Feature Values",
    fontsize=15, y=1.02
)
plt.tight_layout(); plt.show()


# ============================================================
# Step 18: Maps — Local Player Importance
# ============================================================

ncols = 3
nrows = int(np.ceil(n_players / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 5))
axes_flat = axes.flatten()

for idx, player in enumerate(players):
    ax   = axes_flat[idx]
    col  = f"local_signed_{player}"
    _amax = gdf_train[col].abs().max()
    gdf_train.plot(column=col, cmap="RdBu_r", legend=True, alpha=0.85,
                   markersize=30, ax=ax, vmin=-_amax, vmax=_amax)
    add_basemap(ax)
    ax.set_title(player, fontsize=12)
    ax.set_aspect("equal"); ax.set_axis_off()

for idx in range(n_players, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle(
    "Local Player Importance — Signed Spatial Distribution",
    fontsize=15, y=1.01
)
plt.tight_layout(); plt.show()


# ============================================================
# Step 18b: Map — Local Surrogate Fidelity per Property
# ============================================================
# Local R² can be NEGATIVE (the per-property local linear surrogate fits worse
# on held-out test perturbations than simply predicting the mean shift). The
# colour scale is intentionally NOT clipped at 0 — vmin is set to the true
# minimum so negative-fidelity properties stay visually distinguishable
# instead of being flattened to the same colour as R²=0.
# ============================================================

_r2_vmin = min(0.0, float(gdf_train["local_fidelity_r2"].min()))
fig, ax = plt.subplots(figsize=(12, 8))
gdf_train.plot(column="local_fidelity_r2", cmap="viridis", legend=True,
               markersize=50, alpha=0.85, ax=ax, vmin=_r2_vmin, vmax=1)
add_basemap(ax)
ax.set_title("Geo-SMILE: Local Surrogate Fidelity R² per Property", fontsize=14)
ax.set_aspect("equal"); ax.set_axis_off()
plt.tight_layout(); plt.show()


# ============================================================
# Step 19: Map — Dominant Player per Property
# ============================================================

unique_players = list(local_feat_imp_abs_df.idxmax(axis=1).unique())
palette        = plt.cm.get_cmap("tab10", len(unique_players))
color_map      = {p: palette(i) for i, p in enumerate(unique_players)}

fig, ax = plt.subplots(figsize=(13, 9))
for player, color in color_map.items():
    subset = gdf_train[gdf_train["dominant_player"] == player]
    if len(subset):
        subset.plot(ax=ax, color=color, markersize=55, label=player, alpha=0.85)
add_basemap(ax)

ax.legend(title="Dominant Player", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.set_title("Geo-SMILE: Dominant Player Driver per Property", fontsize=15)
ax.set_aspect("equal"); ax.set_axis_off()
plt.tight_layout(); plt.show()


# ============================================================
# Step 20: Heatmap — Geo–Player Co-Salience (Top-30 Properties)
# ============================================================

top30_idx = geo_importance_abs.nlargest(30).index

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(
    cell_df.loc[top30_idx], cmap="YlOrRd", ax=ax,
    xticklabels=True, yticklabels=False,
    cbar_kws={"label": "Co-Salience Score"}
)
ax.set_title("Geo–Player Co-Salience: Top-30 Geo-Sensitivity Properties × Players", fontsize=14)
ax.set_xlabel("Player"); ax.set_ylabel("Property (top-30 by geo sensitivity)")
plt.tight_layout(); plt.show()


# ============================================================
# Step 21: Map — Signed Geo–Player Co-Salience for Top Player
# ============================================================

top_player    = feature_importance_abs.idxmax()
_ctop_abs_max = gdf_train[f"cell_signed_{top_player}"].abs().max()
print(f"\nMost important player overall: {top_player}")

fig, ax = plt.subplots(figsize=(12, 8))
gdf_train.plot(
    column=f"cell_signed_{top_player}", cmap="RdBu_r", legend=True,
    markersize=50, alpha=0.85, ax=ax, vmin=-_ctop_abs_max, vmax=_ctop_abs_max
)
add_basemap(ax)
ax.set_title(f"Geo–Player Co-Salience Map: {top_player}", fontsize=15)
ax.set_aspect("equal"); ax.set_axis_off()
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
# Evaluated on the VALIDATION split, not the test split — the test set is
# reserved exclusively for the one-time fidelity numbers already reported in
# Steps 8/11 for the selected Ridge surrogate. Comparing candidate models on
# test here would mean reading the test set a second time for a decision
# (which model "looks best"), which is exactly the kind of repeated peeking
# the train/val/test split is meant to prevent.
# ============================================================

SURROGATES = {
    "Ridge":         Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
    "SVR":           SVR(kernel="rbf", C=1.0),
    "KNN":           KNeighborsRegressor(n_neighbors=5),
    "XGBoost":       xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                       verbosity=0, random_state=42),
    "LightGBM":      lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                        random_state=42, verbose=-1),
}
WEIGHT_SUPPORT = {"Ridge", "Decision Tree", "Random Forest", "XGBoost", "LightGBM"}

geo_r2s = {}; feat_r2s = {}
for name, model in SURROGATES.items():
    kw_geo  = {"sample_weight": geo_w_tr}  if name in WEIGHT_SUPPORT else {}
    kw_feat = {"sample_weight": feat_w_tr} if name in WEIGHT_SUPPORT else {}
    model.fit(M_geo_tr,  sg_tr, **kw_geo);   geo_r2s[name]  = r2_score(sg_va, model.predict(M_geo_va))
    model.fit(M_feat_tr, sf_tr, **kw_feat);  feat_r2s[name] = r2_score(sf_va, model.predict(M_feat_va))

comparison_df = pd.DataFrame({
    "Model": list(geo_r2s), "Geo R2": list(geo_r2s.values()),
    "Feature R2": list(feat_r2s.values()),
}).sort_values("Geo R2", ascending=False).reset_index(drop=True)
print("\nSurrogate model comparison (validation set):"); display(comparison_df)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(comparison_df)); w = 0.35
ax.bar(x - w/2, comparison_df["Geo R2"],     w, label="Geo Branch",     color="steelblue")
ax.bar(x + w/2, comparison_df["Feature R2"], w, label="Feature Branch", color="darkorange")
ax.set_xticks(x); ax.set_xticklabels(comparison_df["Model"], rotation=45, ha="right")
ax.set_ylabel("Validation R2"); ax.set_title("Surrogate Model Comparison — Geo vs Feature (Validation Set)")
ax.legend(); ax.grid(axis="y")
plt.tight_layout(); plt.show()


# ============================================================
# Step 25: Stability — Geo Branch
# ============================================================

n_repeats_stab = 10
n_subsets_stab = 1000
top_pct_geo    = 20

geo_runs = []; geo_fid_runs = []

for seed in tqdm(range(10, 10 + n_repeats_stab), desc="Geo stability runs"):
    np.random.seed(seed)
    M_r = np.zeros((n_subsets_stab, n_groups), dtype=np.int8); ys = []

    for j in range(n_subsets_stab):
        g   = np.random.randint(min_groups, max_groups + 1)
        sel = np.random.choice(n_groups, size=g, replace=False)
        idx = np.where(np.isin(spatial_group_labels, sel))[0]
        if len(idx) == 0: ys.append(0.0); continue
        X_p = X_train.copy()
        X_p.loc[X_train.index[idx], spatial_features] = baseline_values[spatial_features].values
        pp  = automl.predict(X_p)
        ys.append(wasserstein_distance(original_preds[idx], pp[idx]))
        M_r[j, sel] = 1

    ys    = np.array(ys)
    gid_r = M_r.sum(axis=1).astype(float) / n_groups
    Mt, Me, st, se, gd_tr, _ = train_test_split(M_r, ys, gid_r, test_size=0.25, random_state=42)
    w  = np.clip(apply_kernel(gd_tr, best_k_geo), 1e-8, None)
    rr = Ridge(alpha=1.0); rr.fit(Mt, st, sample_weight=w)
    geo_fid_runs.append(r2_score(se, rr.predict(Me)))
    geo_runs.append(np.abs(rr.coef_))   # GROUP-level coefficients (n_groups,) — no property duplication

geo_runs_df = pd.DataFrame(geo_runs).T          # index = spatial group id (0..n_groups-1)
geo_runs_df.columns = [f"run_{i+1}" for i in range(n_repeats_stab)]

geo_rank_consist, geo_score_consist, geo_set_stab = [], [], []
for a, b in combinations(geo_runs_df.columns, 2):
    sa, sb = geo_runs_df[a].values, geo_runs_df[b].values
    geo_rank_consist.append(spearmanr(sa, sb)[0])
    geo_score_consist.append(pearsonr(sa, sb)[0])
    ta = np.percentile(sa, 100 - top_pct_geo); tb = np.percentile(sb, 100 - top_pct_geo)
    A  = set(geo_runs_df.index[sa >= ta]); B = set(geo_runs_df.index[sb >= tb])
    geo_set_stab.append(len(A & B) / len(A | B) if (A | B) else 1.0)

geo_stability = {
    "Fidelity R2 Mean":                                  np.mean(geo_fid_runs),
    "Fidelity R2 Std":                                   np.std(geo_fid_runs),
    "Rank Consistency Mean":                             np.nanmean(geo_rank_consist),
    "Rank Consistency Std":                              np.nanstd(geo_rank_consist),
    "Score Consistency Mean":                            np.nanmean(geo_score_consist),
    "Score Consistency Std":                             np.nanstd(geo_score_consist),
    f"Important-Set Stability Top-{top_pct_geo}% Mean":  np.nanmean(geo_set_stab),
    f"Important-Set Stability Top-{top_pct_geo}% Std":   np.nanstd(geo_set_stab),
}
print("\n--- Geo Stability (computed on the 100 group-level coefficients, not duplicated property scores) ---")
for k, v in geo_stability.items(): print(f"  {k}: {v:.4f}")


# ============================================================
# Step 26: Stability — Feature Branch (local importance)
# ============================================================

top_pct_feat  = 20
feat_runs     = []; feat_fid_runs = []

for seed in tqdm(range(10, 10 + n_repeats_stab), desc="Feature stability runs"):
    np.random.seed(seed)
    M_r  = np.zeros((n_subsets_stab, n_players), dtype=np.int8)
    pp_r = np.zeros((n_subsets_stab, n_points),  dtype=np.float32)
    ys   = []

    for j in range(n_subsets_stab):
        removal = np.random.randint(0, 2, size=n_players); M_r[j] = removal
        X_p = X_train.copy()
        for ki, player in enumerate(players):
            if removal[ki] == 1:
                if player == "location":
                    X_p[spatial_features] = baseline_values[spatial_features].values
                else:
                    X_p[player] = baseline_values[player]
        pp = automl.predict(X_p); pp_r[j] = pp
        ys.append(wasserstein_distance(original_preds, pp))

    ys   = np.array(ys)
    hd_r = M_r.sum(axis=1).astype(float) / n_players
    Mt, Me, st, se, hd_tr_r, _ = train_test_split(M_r, ys, hd_r, test_size=0.25, random_state=42)
    w  = np.clip(apply_kernel(hd_tr_r, best_k_feat), 1e-8, None)
    rr = Ridge(alpha=1.0); rr.fit(Mt, st, sample_weight=w)
    feat_fid_runs.append(r2_score(se, rr.predict(Me)))

    loc_w = np.clip(apply_kernel(hd_r, best_k_feat), 1e-8, None)
    loc_r = np.zeros((n_points, n_players), dtype=np.float32)
    for i in range(n_points):
        y_i = original_preds[i] - pp_r[:, i]           # original − perturbed
        rri = Ridge(alpha=1.0); rri.fit(M_r, y_i, sample_weight=loc_w)
        loc_r[i] = rri.coef_
    feat_runs.append(np.abs(loc_r).flatten())

feat_runs_arr = np.array(feat_runs)

feat_rank_consist, feat_score_consist, feat_set_stab = [], [], []
for a, b in combinations(range(n_repeats_stab), 2):
    sa, sb = feat_runs_arr[a], feat_runs_arr[b]
    feat_rank_consist.append(spearmanr(sa, sb)[0])
    feat_score_consist.append(pearsonr(sa, sb)[0])
    ta = np.percentile(sa, 100 - top_pct_feat); tb = np.percentile(sb, 100 - top_pct_feat)
    A  = set(np.where(sa >= ta)[0]); B = set(np.where(sb >= tb)[0])
    feat_set_stab.append(len(A & B) / len(A | B) if (A | B) else 1.0)

feat_stability = {
    "Fidelity R2 Mean":                                   np.mean(feat_fid_runs),
    "Fidelity R2 Std":                                    np.std(feat_fid_runs),
    "Rank Consistency Mean":                              np.nanmean(feat_rank_consist),
    "Rank Consistency Std":                               np.nanstd(feat_rank_consist),
    "Score Consistency Mean":                             np.nanmean(feat_score_consist),
    "Score Consistency Std":                              np.nanstd(feat_score_consist),
    f"Important-Set Stability Top-{top_pct_feat}% Mean":  np.nanmean(feat_set_stab),
    f"Important-Set Stability Top-{top_pct_feat}% Std":   np.nanstd(feat_set_stab),
}
print("\n--- Feature Stability ---")
for k, v in feat_stability.items(): print(f"  {k}: {v:.4f}")


# ============================================================
# Step 27: Sparsity & Entropy
# ============================================================

def sparsity_entropy_metrics(scores_raw, label=""):
    s = np.nan_to_num(np.asarray(scores_raw, dtype=float), nan=0.0)
    n = len(s); eps = 1e-12
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
        "Hoyer Sparsity": hoyer,
        "Gini Concentration": gini, "Entropy": ent,
        "Normalized Entropy": norm_ent, "Effective N": eff_n, "Effective Ratio": eff_r,
    }
    print(f"\n--- Sparsity & Entropy ({label}) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if not (v is None or (isinstance(v, float) and np.isnan(v)))
              else f"  {k}: NaN")
    return metrics


# Geo sparsity/entropy is computed on the 100 GROUP-level coefficients (not the
# duplicated per-property broadcast), for the same reason as Step 25's stability metrics.
geo_sp_ent   = sparsity_entropy_metrics(geo_group_importance_abs,               label="Geo (group-level)")
feat_sp_ent  = sparsity_entropy_metrics(feature_importance_abs.values,          label="Feature (mean local)")
local_sp_ent = sparsity_entropy_metrics(local_feat_imp_abs_df.values.flatten(), label="Local Feature (all)")
cell_sp_ent  = sparsity_entropy_metrics(cell_matrix.flatten(),                  label="Geo–Player Co-Salience")


# ============================================================
# Step 28: Combined Evaluation Table
# ============================================================

def _rows(metrics, branch, category):
    return [{"Branch": branch, "Category": category, "Metric": k, "Value": v}
            for k, v in metrics.items()]

rows = (
    _rows(geo_fidelity,   "Geo",     "Fidelity")         +
    _rows(feat_fidelity,  "Feature", "Fidelity")         +
    _rows(geo_stability,  "Geo",     "Stability")        +
    _rows(feat_stability, "Feature", "Stability")        +
    _rows(geo_sp_ent,     "Geo",     "Sparsity/Entropy")       +
    _rows(feat_sp_ent,    "Feature", "Sparsity/Entropy")       +
    _rows(local_sp_ent,   "Feature", "Sparsity/Entropy (Local)") +
    _rows(cell_sp_ent,    "Co-Salience", "Sparsity/Entropy")
)
eval_df = pd.DataFrame(rows)
print("\n--- Full Geo-SMILE Evaluation Metrics ---")
display(eval_df)


# ============================================================
# Step 29: Stability Maps — Geo Branch
# ============================================================
# geo_runs_df is indexed by spatial GROUP id (Step 25). Broadcast each
# group's mean/std back to its member properties purely for visualization —
# the stability metrics themselves were already computed at the group level.
# ============================================================

_geo_group_mean = geo_runs_df.mean(axis=1).values   # length n_groups
_geo_group_std  = geo_runs_df.std(axis=1).values     # length n_groups
gdf_train["geo_imp_mean"] = _geo_group_mean[spatial_group_labels]
gdf_train["geo_imp_std"]  = _geo_group_std[spatial_group_labels]

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
gdf_train.plot(column="geo_imp_mean", cmap="plasma", legend=True, markersize=40, alpha=0.85, ax=axes[0])
add_basemap(axes[0])
axes[0].set_title("Spatial-Group Sensitivity — Repeated-Run Mean", fontsize=13)
axes[0].set_aspect("equal"); axes[0].set_axis_off()
gdf_train.plot(column="geo_imp_std",  cmap="magma",  legend=True, markersize=40, alpha=0.85, ax=axes[1])
add_basemap(axes[1])
axes[1].set_title("Spatial-Group Sensitivity — Instability (Std)", fontsize=13)
axes[1].set_aspect("equal"); axes[1].set_axis_off()
plt.tight_layout(); plt.show()


# ============================================================
# Step 30: Top-10% High Geo Score Properties
# ============================================================
# Threshold computed on the 100 GROUP-level coefficients, not the duplicated
# per-property broadcast — group sizes differ, so percentiling the broadcast
# array would let larger clusters dominate the threshold and the resulting
# count of highlighted properties. Groups above the threshold are selected
# first; membership is then mapped back to properties only for display.
# ============================================================

group_threshold = np.percentile(geo_group_importance_abs, 90)
top_groups = np.where(geo_group_importance_abs >= group_threshold)[0]
gdf_train["top10_geo"] = np.isin(spatial_group_labels, top_groups).astype(int)

fig, ax = plt.subplots(figsize=(12, 8))
gdf_train[gdf_train["top10_geo"] == 0].plot(ax=ax, markersize=15, alpha=0.20, color="lightgrey")
gdf_train[gdf_train["top10_geo"] == 1].plot(ax=ax, markersize=70, color="red",
                                             alpha=0.85, label="Top 10% Geo Score")
add_basemap(ax)
ax.legend()
ax.set_title("Geo-SMILE: Top-10% High Spatial-Group Distributional Sensitivity", fontsize=15)
ax.set_aspect("equal"); ax.set_axis_off()
plt.tight_layout(); plt.show()


print("\nGeo-SMILE full pipeline completed successfully.")
