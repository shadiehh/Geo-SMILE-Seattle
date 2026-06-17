!pip install geopandas xgboost scikit-learn scipy matplotlib pandas numpy tqdm

# ============================================================
# FULL PIPELINE: XGBoost + Geo-SMILE-F (SMILE-correct)
# on Ziqi Li Chicago Ride-Hailing Dataset
#
# Algorithm:
#   1. Downloads GeoJSON dataset
#   2. Prepares target log_trip_demand = log1p(TripCount)
#   3. Trains XGBoost black-box model (train/val/test split)
#   4. Generates a GLOBAL perturbation pool applied to all tracts
#   5. SMILE response = Wasserstein distance W1(f(X), f(X_pert))
#   6. Selects best kernel shape on validation split
#   7. Fits one global weighted Ridge surrogate → global importances
#   8. Derives local per-tract effects from pool (signed, for maps)
#   9. Evaluates fidelity, faithfulness, stability, consistency
#  10. Creates Ziqi-style plots and spatial maps
#  11. Optional SHAP comparison
#  12. Saves outputs
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from tqdm import tqdm
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr, wasserstein_distance


# ============================================================
# 0. USER SETTINGS
# ============================================================

N_PERTURBATIONS     = 3000      # global pool size
RANDOM_SEED         = 42

N_CONSISTENCY_RUNS  = 5
CONSISTENCY_SEEDS   = [42, 101, 202, 303, 404]

N_STABILITY_PERTURBATIONS  = 1000
STABILITY_NOISE_SCALE      = 0.01
TOP_K_FEATURES_FOR_STABILITY = 3

KERNELS = ["rbf", "exponential", "inverse", "linear", "laplacian"]

RUN_SHAP_COMPARISON = True


# ============================================================
# 1. HELPER FUNCTIONS
# ============================================================

def safe_display(df, n=5):
    try:
        display(df.head(n))
    except NameError:
        print(df.head(n))


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


def select_best_kernel(M_tr, M_va, dist_tr, resp_tr, resp_va):
    """Select kernel shape on validation set; caller reports fidelity on test."""
    results = {}
    for k in KERNELS:
        w = np.clip(apply_kernel(dist_tr, k), 1e-8, None)
        ridge = Ridge(alpha=1.0)
        ridge.fit(M_tr, resp_tr, sample_weight=w)
        results[k] = r2_score(resp_va, ridge.predict(M_va))
    best = max(results, key=results.get)
    return best, results


def weighted_r2_score(y_true, y_pred, sample_weight):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sample_weight = np.asarray(sample_weight)

    if np.sum(sample_weight) <= 0:
        return np.nan

    weighted_mean = np.average(y_true, weights=sample_weight)
    ss_res = np.sum(sample_weight * (y_true - y_pred) ** 2)
    ss_tot = np.sum(sample_weight * (y_true - weighted_mean) ** 2)

    if ss_tot <= 1e-12:
        return np.nan

    return 1 - (ss_res / ss_tot)


def weighted_mse(y_true, y_pred, sample_weight):
    return np.average((y_true - y_pred) ** 2, weights=sample_weight)


def weighted_mae(y_true, y_pred, sample_weight):
    return np.average(np.abs(y_true - y_pred), weights=sample_weight)


def weighted_rmse(y_true, y_pred, sample_weight):
    return np.sqrt(weighted_mse(y_true, y_pred, sample_weight))


def jaccard_index(list_a, list_b):
    set_a = set(list_a)
    set_b = set(list_b)
    union = set_a.union(set_b)

    if len(union) == 0:
        return np.nan

    return len(set_a.intersection(set_b)) / len(union)


# ============================================================
# 2. DOWNLOAD DATASET
# ============================================================

print("Downloading Ziqi Li Chicago ride-hailing GeoJSON...")

geojson_url = (
    "https://raw.githubusercontent.com/Ziqi-Li/"
    "SHAP_spatial_data_paper/main/Notebooks/chicago_tnp.geojson"
)

gdf = gpd.read_file(geojson_url)

print("Dataset downloaded.")
print("Raw shape:", gdf.shape)
print("Raw CRS:", gdf.crs)
print("\nColumns:")
print(gdf.columns.tolist())

print("\nPreview:")
safe_display(gdf, n=5)


# ============================================================
# 3. PREPARE GEOMETRY, TARGET, AND COORDINATES
# ============================================================

print("\nPreparing geometry, target, and coordinate variables...")

if "TripCount" not in gdf.columns:
    raise ValueError("Expected column 'TripCount' was not found.")

# Create target if it does not exist
target_col = "log_trip_demand"

if target_col not in gdf.columns:
    gdf[target_col] = np.log1p(gdf["TripCount"])

# Create log population density
if "population_den" not in gdf.columns:
    raise ValueError("Expected column 'population_den' was not found.")

gdf["log_population_den"] = np.log1p(gdf["population_den"])

# The file may report EPSG:4326 although coordinates behave as projected.
# This follows the earlier working logic.
gdf = gdf.set_crs("EPSG:3435", allow_override=True)
gdf_model = gdf.to_crs("EPSG:32616").copy()

gdf_model["x_coord"] = gdf_model.geometry.centroid.x
gdf_model["y_coord"] = gdf_model.geometry.centroid.y

if "Pickup Census Tract" in gdf_model.columns:
    gdf_model["tract_id"] = gdf_model["Pickup Census Tract"].astype(str)
else:
    gdf_model["tract_id"] = np.arange(len(gdf_model)).astype(str)

print("Coordinate columns created: x_coord, y_coord")


# ============================================================
# 4. DEFINE ZIQI-COMPARABLE FEATURES
# ============================================================

print("\nDefining Ziqi-comparable feature set...")

feature_groups = {
    # Ziqi-style explanatory variables
    "pct_18to34": ["pct_18to34"],
    "pct_white": ["pct_white"],
    "pct_bachelorORhigher": ["pct_bachelorORhigher"],
    "pct_no_vehicle": ["pct_no_vehicle"],
    "log_population_den": ["log_population_den"],
    "job_entropy": ["job_entropy"],
    "network_den": ["network_den"],
    "TripMiles_mean": ["TripMiles_mean"],
    "pct_share": ["pct_share"],

    # Geo-SMILE / GeoShapley-style grouped location player
    "location": ["x_coord", "y_coord"]
}

group_names = list(feature_groups.keys())

feature_cols = []
for group_name, cols in feature_groups.items():
    for col in cols:
        if col not in feature_cols:
            feature_cols.append(col)

required_cols = feature_cols + [target_col, "tract_id", "geometry"]

missing_cols = [col for col in required_cols if col not in gdf_model.columns]
if len(missing_cols) > 0:
    raise ValueError(f"Missing required columns: {missing_cols}")

gdf_model = (
    gdf_model[required_cols]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .reset_index(drop=True)
    .copy()
)

X = gdf_model[feature_cols].copy()
y = gdf_model[target_col].copy()

print("Target column:", target_col)
print("Feature groups:", group_names)
print("Model feature columns:", feature_cols)
print("Final modelling dataset shape:", gdf_model.shape)
print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

print("\nTarget summary:")
print(y.describe())


# ============================================================
# 5. TRAIN XGBOOST BLACK-BOX MODEL
# ============================================================

print("\nTraining XGBoost black-box model...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_SEED
)

xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=RANDOM_SEED,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

blackbox_r2 = r2_score(y_test, y_pred)
blackbox_mae = mean_absolute_error(y_test, y_pred)
blackbox_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nXGBoost black-box performance:")
print(f"Hold-out R2:   {blackbox_r2:.4f}")
print(f"Hold-out MAE:  {blackbox_mae:.4f}")
print(f"Hold-out RMSE: {blackbox_rmse:.4f}")

gdf_model["Predicted_Log_Demand"] = xgb_model.predict(X)

xgb_importance_df = pd.DataFrame({
    "feature": feature_cols,
    "xgb_importance": xgb_model.feature_importances_
}).sort_values("xgb_importance", ascending=False)

print("\nXGBoost feature importance:")
safe_display(xgb_importance_df, n=20)


# ============================================================
# 6. GEO-SMILE SETUP
# ============================================================

gdf_geo = gdf_model.reset_index(drop=True).copy()
X_geo   = gdf_geo[feature_cols].copy()
n_obs   = len(X_geo)
n_grps  = len(group_names)

# baseline from training data only — prevents leakage into perturbations
baseline_values = X_train.median()

print("\nGeo-SMILE (SMILE-correct) setup:")
print("Number of tracts:", n_obs)
print("Number of feature groups:", n_grps)
print("Feature groups:", group_names)
print("Perturbation pool size:", N_PERTURBATIONS)
print("Kernel candidates:", KERNELS)


# ============================================================
# 7. GLOBAL PERTURBATION POOL  (SMILE)
# Response = W1(f(X), f(X_pert)) — distributional Wasserstein distance.
# Each coalition mask is applied to ALL tracts simultaneously so the
# response is a population-level statistic, not a per-tract scalar.
# ============================================================

print("\nGenerating global perturbation pool...")

rng            = np.random.default_rng(RANDOM_SEED)
M              = np.zeros((N_PERTURBATIONS, n_grps), dtype=np.int8)
y_wd           = np.zeros(N_PERTURBATIONS, dtype=np.float64)
all_pert_preds = np.zeros((N_PERTURBATIONS, n_obs), dtype=np.float32)

original_preds = xgb_model.predict(X_geo).astype(np.float32)

for j in tqdm(range(N_PERTURBATIONS), desc="Perturbations"):
    k       = int(rng.integers(1, n_grps))
    removed = rng.choice(n_grps, size=k, replace=False)
    mask    = np.zeros(n_grps, dtype=np.int8)
    mask[removed] = 1
    M[j]    = mask

    X_pert = X_geo.copy()
    for gi in removed:
        for col in feature_groups[group_names[gi]]:
            X_pert[col] = baseline_values[col]

    pp = xgb_model.predict(X_pert).astype(np.float32)
    all_pert_preds[j] = pp
    y_wd[j] = wasserstein_distance(original_preds, pp)

hamming_dist = M.sum(axis=1).astype(float) / n_grps

print(f"Wasserstein response — mean: {y_wd.mean():.4f}  std: {y_wd.std():.4f}")


# ============================================================
# 8. 3-WAY SPLIT + KERNEL SELECTION  (SMILE)
# Kernel shape selected on validation; fidelity reported on
# the untouched test set exactly once.
# ============================================================

all_idx = np.arange(N_PERTURBATIONS)
tr_idx, tmp_idx = train_test_split(all_idx, test_size=0.40, random_state=RANDOM_SEED)
va_idx, te_idx  = train_test_split(tmp_idx, test_size=0.50, random_state=RANDOM_SEED)

M_tr = M[tr_idx];  M_va = M[va_idx];  M_te = M[te_idx]
wd_tr = y_wd[tr_idx]; wd_va = y_wd[va_idx]; wd_te = y_wd[te_idx]
hd_tr = hamming_dist[tr_idx]; hd_va = hamming_dist[va_idx]; hd_te = hamming_dist[te_idx]

best_kernel, kernel_scores = select_best_kernel(M_tr, M_va, hd_tr, wd_tr, wd_va)

print("\nKernel R² on validation set:")
for k, v in sorted(kernel_scores.items(), key=lambda x: -x[1]):
    print(f"  {k:<14} {v:.4f}{' <- best' if k == best_kernel else ''}")


# ============================================================
# 9. GLOBAL SMILE SURROGATE
# Final fit on train+val; evaluate on test set.
# Coefficients = global player importances (contribution of each
# feature group to the Wasserstein distributional shift).
# ============================================================

M_trva  = np.vstack([M_tr, M_va])
wd_trva = np.concatenate([wd_tr, wd_va])
hd_trva = np.concatenate([hd_tr, hd_va])
w_trva  = np.clip(apply_kernel(hd_trva, best_kernel), 1e-8, None)

ridge_global = Ridge(alpha=1.0)
ridge_global.fit(M_trva, wd_trva, sample_weight=w_trva)

w_te    = np.clip(apply_kernel(hd_te, best_kernel), 1e-8, None)
pred_te = ridge_global.predict(M_te)

global_fidelity = {
    "R2":          r2_score(wd_te, pred_te),
    "Weighted R2": weighted_r2_score(wd_te, pred_te, w_te),
    "MAE":         mean_absolute_error(wd_te, pred_te),
    "WMAE":        weighted_mae(wd_te, pred_te, w_te),
    "RMSE":        np.sqrt(mean_squared_error(wd_te, pred_te)),
    "WRMSE":       weighted_rmse(wd_te, pred_te, w_te),
}

print(f"\nGlobal SMILE surrogate fidelity (kernel={best_kernel}, test set):")
for k, v in global_fidelity.items():
    print(f"  {k}: {v:.6f}")

global_importances     = pd.Series(ridge_global.coef_, index=group_names, name="smile_importance")
global_importances_abs = global_importances.abs()

print("\nGlobal SMILE feature importances (|coef|):")
safe_display(global_importances_abs.sort_values(ascending=False).to_frame(), n=20)

plt.figure(figsize=(10, 5))
global_importances_abs.sort_values().plot(kind="barh", color="steelblue")
plt.title("Global SMILE Feature Importances (|Ridge coef|, Wasserstein response)")
plt.xlabel("|Coefficient|")
plt.tight_layout()
plt.show()


# ============================================================
# 10. LOCAL PER-TRACT EFFECTS  (signed, for spatial maps)
# For each tract i, fit a Ridge on TRAINING perturbations with
# response = f(x_i) - f(x_i_pert).  Sign is meaningful here
# (positive = removing this group lowers the predicted demand at i).
# Fidelity evaluated on TEST perturbations.
# ============================================================

print("\nFitting local per-tract effects...")

_ZERO_VAR_TOL     = 1e-10
local_effects     = np.zeros((n_obs, n_grps), dtype=np.float32)
local_r2          = np.full(n_obs, np.nan,    dtype=np.float32)
w_tr_local        = np.clip(apply_kernel(hd_tr, best_kernel), 1e-8, None)

n_zero_var = 0
for i in tqdm(range(n_obs), desc="Local per-tract Ridge"):
    y_i_tr = original_preds[i] - all_pert_preds[tr_idx, i]
    y_i_te = original_preds[i] - all_pert_preds[te_idx, i]

    ridge_i = Ridge(alpha=1.0)
    ridge_i.fit(M_tr, y_i_tr, sample_weight=w_tr_local)
    local_effects[i] = ridge_i.coef_

    if np.var(y_i_te) > _ZERO_VAR_TOL:
        local_r2[i] = r2_score(y_i_te, ridge_i.predict(M_te))
    else:
        n_zero_var += 1

local_effects_df     = pd.DataFrame(local_effects,          columns=group_names)
local_effects_abs_df = pd.DataFrame(np.abs(local_effects),  columns=group_names)
local_r2_series      = pd.Series(local_r2)

print(f"\nLocal fidelity (per-tract R² on held-out test perturbations):")
print(f"  Zero-variance tracts excluded: {n_zero_var}")
print(f"  Median: {np.nanmedian(local_r2):.4f}")
print(f"  Mean:   {np.nanmean(local_r2):.4f}  Std: {np.nanstd(local_r2):.4f}")
_valid_r2 = local_r2[~np.isnan(local_r2)]
print(f"  Prop > 0.50 R²: {(_valid_r2 > 0.50).mean():.2%}")


# ============================================================
# 11. FAITHFULNESS
# Spearman / Pearson between mask cardinality and Wasserstein
# response: larger coalitions should produce larger shifts.
# ============================================================

faith_spearman, faith_sp = spearmanr(hamming_dist, y_wd)
faith_pearson,  faith_pp = pearsonr(hamming_dist,  y_wd)

faithfulness_summary_df = pd.DataFrame({
    "Metric": ["Spearman (cardinality vs WD)", "Pearson (cardinality vs WD)"],
    "Value":  [faith_spearman, faith_pearson],
    "p":      [faith_sp, faith_pp]
})

print("\nFaithfulness (mask cardinality vs Wasserstein response):")
safe_display(faithfulness_summary_df, n=5)

plt.figure(figsize=(8, 4))
plt.scatter(hamming_dist, y_wd, s=4, alpha=0.4, c="black")
plt.xlabel("Mask cardinality (proportion of groups removed)")
plt.ylabel("Wasserstein distance W₁(f(X), f(X_pert))")
plt.title(f"Faithfulness  Spearman={faith_spearman:.3f}  Pearson={faith_pearson:.3f}")
plt.tight_layout()
plt.show()


# ============================================================
# 12. ADD MAP-READY COLUMNS
# ============================================================

for gname in group_names:
    gdf_geo[f"GeoSMILE_{gname}_Effect"]        = local_effects_df[gname].values
    gdf_geo[f"GeoSMILE_{gname}_AbsImportance"] = local_effects_abs_df[gname].values

gdf_geo["local_fidelity_r2"]    = local_r2
gdf_geo["Predicted_Log_Demand"] = original_preds

print("\nMap-ready Geo-SMILE columns:")
print([c for c in gdf_geo.columns if c.startswith("GeoSMILE_")])


# ============================================================
# 13. CONSISTENCY ASSESSMENT
# Re-run local effects with different random seeds; check
# Jaccard overlap of top-k features across runs.
# ============================================================

print("\nRunning consistency assessment...")

def _run_local_effects(seed, n_perturb):
    rng_c = np.random.default_rng(seed)
    M_c   = np.zeros((n_perturb, n_grps), dtype=np.int8)
    pp_c  = np.zeros((n_perturb, n_obs),  dtype=np.float32)

    for j in range(n_perturb):
        k       = int(rng_c.integers(1, n_grps))
        removed = rng_c.choice(n_grps, size=k, replace=False)
        mask    = np.zeros(n_grps, dtype=np.int8)
        mask[removed] = 1
        M_c[j] = mask
        X_p = X_geo.copy()
        for gi in removed:
            for col in feature_groups[group_names[gi]]:
                X_p[col] = baseline_values[col]
        pp_c[j] = xgb_model.predict(X_p).astype(np.float32)

    hd_c = M_c.sum(axis=1).astype(float) / n_grps
    w_c  = np.clip(apply_kernel(hd_c, best_kernel), 1e-8, None)

    loc = np.zeros((n_obs, n_grps), dtype=np.float32)
    for i in range(n_obs):
        y_i = original_preds[i] - pp_c[:, i]
        r   = Ridge(alpha=1.0)
        r.fit(M_c, y_i, sample_weight=w_c)
        loc[i] = r.coef_
    return loc


N_CONSIST_PERTURB = max(500, N_PERTURBATIONS // 3)
consistency_arrays = []

for run_seed in CONSISTENCY_SEEDS[:N_CONSISTENCY_RUNS]:
    loc = _run_local_effects(run_seed, N_CONSIST_PERTURB)
    consistency_arrays.append(loc)

consistency_arrays = np.array(consistency_arrays)   # (N_runs, n_obs, n_grps)

coef_mean = consistency_arrays.mean(axis=0)
coef_std  = consistency_arrays.std(axis=0)
coef_var  = consistency_arrays.var(axis=0)
coef_cv   = coef_std / (np.abs(coef_mean) + 1e-8)

topk_consistency_rows = []
for tract_pos in range(n_obs):
    top_sets = []
    for run_idx in range(len(consistency_arrays)):
        abs_eff   = np.abs(consistency_arrays[run_idx, tract_pos])
        top_feats = [group_names[i] for i in np.argsort(abs_eff)[::-1][:TOP_K_FEATURES_FOR_STABILITY]]
        top_sets.append(top_feats)
    for i in range(len(top_sets)):
        for j in range(i + 1, len(top_sets)):
            topk_consistency_rows.append({
                "target_idx": tract_pos,
                "run_i": i, "run_j": j,
                "top_k": TOP_K_FEATURES_FOR_STABILITY,
                "jaccard_consistency": jaccard_index(top_sets[i], top_sets[j])
            })

topk_consistency_df = pd.DataFrame(topk_consistency_rows)

consistency_metrics_df = pd.DataFrame({
    "Category": ["Geo-SMILE Consistency"] * 6,
    "Metric": [
        "Mean Coefficient Variance",
        "Mean Coefficient Std",
        "Mean Coefficient CV",
        "Median Coefficient CV",
        f"Mean Top-{TOP_K_FEATURES_FOR_STABILITY} Jaccard Consistency",
        f"Median Top-{TOP_K_FEATURES_FOR_STABILITY} Jaccard Consistency"
    ],
    "Value": [
        np.nanmean(coef_var),
        np.nanmean(coef_std),
        np.nanmean(coef_cv),
        np.nanmedian(coef_cv),
        topk_consistency_df["jaccard_consistency"].mean(),
        topk_consistency_df["jaccard_consistency"].median()
    ]
})

print("\nConsistency metrics:")
safe_display(consistency_metrics_df, n=10)


# ============================================================
# 14. STABILITY ASSESSMENT
# Jitter features with small noise, rerun local effects,
# check Top-k Jaccard overlap with original.
# ============================================================

print("\nRunning stability assessment with small feature noise...")

rng_stab   = np.random.default_rng(999)
X_jittered = X_geo.copy()
feat_std   = X_geo.std(numeric_only=True)

for col in feature_cols:
    X_jittered[col] += rng_stab.normal(
        loc=0.0,
        scale=STABILITY_NOISE_SCALE * (feat_std[col] + 1e-8),
        size=n_obs
    )

rng_stab2   = np.random.default_rng(777)
M_stab      = np.zeros((N_STABILITY_PERTURBATIONS, n_grps), dtype=np.int8)
pp_stab     = np.zeros((N_STABILITY_PERTURBATIONS, n_obs),  dtype=np.float32)
orig_jitter = xgb_model.predict(X_jittered).astype(np.float32)

for j in range(N_STABILITY_PERTURBATIONS):
    k       = int(rng_stab2.integers(1, n_grps))
    removed = rng_stab2.choice(n_grps, size=k, replace=False)
    mask    = np.zeros(n_grps, dtype=np.int8)
    mask[removed] = 1
    M_stab[j] = mask
    X_p = X_jittered.copy()
    for gi in removed:
        for col in feature_groups[group_names[gi]]:
            X_p[col] = baseline_values[col]
    pp_stab[j] = xgb_model.predict(X_p).astype(np.float32)

hd_stab = M_stab.sum(axis=1).astype(float) / n_grps
w_stab  = np.clip(apply_kernel(hd_stab, best_kernel), 1e-8, None)

jitter_effects = np.zeros((n_obs, n_grps), dtype=np.float32)
for i in range(n_obs):
    y_i = orig_jitter[i] - pp_stab[:, i]
    r   = Ridge(alpha=1.0)
    r.fit(M_stab, y_i, sample_weight=w_stab)
    jitter_effects[i] = r.coef_

jitter_abs = np.abs(jitter_effects)
orig_abs   = local_effects_abs_df.values

stability_rows = []
for i in range(n_obs):
    orig_top   = [group_names[x] for x in np.argsort(orig_abs[i])[::-1][:TOP_K_FEATURES_FOR_STABILITY]]
    jitter_top = [group_names[x] for x in np.argsort(jitter_abs[i])[::-1][:TOP_K_FEATURES_FOR_STABILITY]]
    stability_rows.append({
        "target_idx": i,
        "top_k": TOP_K_FEATURES_FOR_STABILITY,
        "jaccard_stability": jaccard_index(orig_top, jitter_top)
    })

stability_detail_df = pd.DataFrame(stability_rows)

stability_metrics_df = pd.DataFrame({
    "Category": ["Geo-SMILE Stability"] * 3,
    "Metric": [
        f"Mean Top-{TOP_K_FEATURES_FOR_STABILITY} Jaccard Stability",
        f"Median Top-{TOP_K_FEATURES_FOR_STABILITY} Jaccard Stability",
        "Feature Noise Scale"
    ],
    "Value": [
        stability_detail_df["jaccard_stability"].mean(),
        stability_detail_df["jaccard_stability"].median(),
        STABILITY_NOISE_SCALE
    ]
})

print("\nStability metrics:")
safe_display(stability_metrics_df, n=10)





# ============================================================
# 16. FINAL EVALUATION TABLE
# ============================================================

fidelity_metrics_df = pd.DataFrame({
    "Category": [
        "Black-box Prediction",
        "Black-box Prediction",
        "Black-box Prediction",
        "Geo-SMILE Setup",
        "Geo-SMILE Setup",
        "Geo-SMILE Setup",
        "Geo-SMILE Global Fidelity",
        "Geo-SMILE Global Fidelity",
        "Geo-SMILE Global Fidelity",
        "Geo-SMILE Global Fidelity",
        "Geo-SMILE Global Fidelity",
        "Geo-SMILE Global Fidelity",
        "Geo-SMILE Local Fidelity",
        "Geo-SMILE Local Fidelity",
        "Geo-SMILE Local Fidelity",
    ],
    "Metric": [
        "XGBoost Hold-out R2",
        "XGBoost Hold-out MAE",
        "XGBoost Hold-out RMSE",
        "Best Kernel",
        "Response",
        "Perturbation Pool Size",
        "Global R2",
        "Global Weighted R2",
        "Global MAE",
        "Global WMAE",
        "Global RMSE",
        "Global WRMSE",
        "Mean Local R2 (per-tract)",
        "Median Local R2 (per-tract)",
        "Prop Local R2 > 0.50",
    ],
    "Value": [
        blackbox_r2,
        blackbox_mae,
        blackbox_rmse,
        best_kernel,
        "Wasserstein W1",
        N_PERTURBATIONS,
        global_fidelity["R2"],
        global_fidelity["Weighted R2"],
        global_fidelity["MAE"],
        global_fidelity["WMAE"],
        global_fidelity["RMSE"],
        global_fidelity["WRMSE"],
        np.nanmean(local_r2),
        np.nanmedian(local_r2),
        float((_valid_r2 > 0.50).mean()),
    ]
})

faithfulness_metrics_df = pd.DataFrame({
    "Category": ["Geo-SMILE Faithfulness", "Geo-SMILE Faithfulness"],
    "Metric":   ["Spearman (cardinality vs WD)", "Pearson (cardinality vs WD)"],
    "Value":    [faith_spearman, faith_pearson]
})

accuracy_metrics_df = pd.DataFrame({
    "Category": ["Geo-SMILE Accuracy"] * 3,
    "Metric":   ["ATT Accuracy", "ATT F1", "ATT AUROC"],
    "Value":    ["Not computed: no attribution ground truth"] * 3
})

final_metric_assessment_df = pd.concat([
    fidelity_metrics_df,
    faithfulness_metrics_df,
    stability_metrics_df,
    consistency_metrics_df,
    accuracy_metrics_df
], ignore_index=True)

print("\nFinal Geo-SMILE evaluation table:")
safe_display(final_metric_assessment_df, n=100)

# ============================================================
# 17. geoshapley FEATURE EFFECT PLOTS
# ============================================================

def bootstrap_ci_by_bins(x, y, n_bins=20, n_boot=300, ci=95, seed=42):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) < n_bins:
        return None

    df["bin"] = pd.qcut(df["x"], q=min(n_bins, df["x"].nunique()), duplicates="drop")

    rows = []

    for _, sub in df.groupby("bin"):
        if len(sub) < 5:
            continue

        x_mid = sub["x"].median()
        boot_means = []

        y_vals = sub["y"].values

        for _ in range(n_boot):
            sample = rng.choice(y_vals, size=len(y_vals), replace=True)
            boot_means.append(np.mean(sample))

        lower = np.percentile(boot_means, (100 - ci) / 2)
        upper = np.percentile(boot_means, 100 - ((100 - ci) / 2))
        mean_y = np.mean(y_vals)

        rows.append({
            "x_mid": x_mid,
            "mean_y": mean_y,
            "lower": lower,
            "upper": upper
        })

    return pd.DataFrame(rows).sort_values("x_mid")


def plot_ziqi_style_effects(gdf, feature_groups, n_cols=3):
    group_names = list(feature_groups.keys())
    n_rows = int(np.ceil(len(group_names) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.8 * n_rows))
    axes = axes.flatten()

    for idx, group_name in enumerate(group_names):
        ax = axes[idx]

        if group_name == "location":
            x = np.sqrt(
                ((gdf["x_coord"] - gdf["x_coord"].median()) ** 2) +
                ((gdf["y_coord"] - gdf["y_coord"].median()) ** 2)
            )
            x_label = "location_distance_from_median"
        else:
            x = gdf[feature_groups[group_name][0]]
            x_label = group_name

        y = gdf[f"GeoSMILE_{group_name}_Effect"]

        ax.scatter(x, y, s=8, alpha=0.65, c="black")
        ax.axhline(0, linestyle="--", linewidth=1)

        ci_df = bootstrap_ci_by_bins(
            x,
            y,
            n_bins=20,
            n_boot=300,
            ci=95,
            seed=RANDOM_SEED
        )

        if ci_df is not None and len(ci_df) > 1:
            ax.plot(ci_df["x_mid"], ci_df["mean_y"], linewidth=2)
            ax.fill_between(
                ci_df["x_mid"],
                ci_df["lower"],
                ci_df["upper"],
                alpha=0.25
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Geo-SMILE effect")
        ax.set_title(group_name)

    for j in range(len(group_names), len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"Geo-SMILE-F Feature Effect Plots\nSelected kernel: {best_kernel}",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


plot_ziqi_style_effects(gdf_geo, feature_groups, n_cols=3)


# ============================================================
# 18. SPATIAL MAPS
# ============================================================

def plot_effect_map(gdf, column, title, cmap="RdBu_r"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    values = gdf[column].values
    max_abs = np.nanmax(np.abs(values))

    if max_abs == 0 or np.isnan(max_abs):
        max_abs = 1.0

    gdf.plot(
        column=column,
        cmap=cmap,
        vmin=-max_abs,
        vmax=max_abs,
        linewidth=0.05,
        edgecolor="none",
        legend=True,
        ax=ax
    )

    ax.set_title(title, fontsize=13)
    ax.axis("off")
    plt.show()


def plot_abs_map(gdf, column, title, cmap="OrRd"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    gdf.plot(
        column=column,
        cmap=cmap,
        linewidth=0.05,
        edgecolor="none",
        legend=True,
        ax=ax
    )

    ax.set_title(title, fontsize=13)
    ax.axis("off")
    plt.show()


#plot_effect_map(
    #gdf_geo,
    #"GeoSMILE_location_Effect",
    #f"Geo-SMILE-F Location Effect\nkernel={best_kernel}"
#)

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

geoshapley_cmap = LinearSegmentedColormap.from_list(
    "GeoShapleyBluePink",
    ["#1E88E5", "white", "#FF0051"]
)

def plot_effect_map(gdf, column, title, cmap=geoshapley_cmap):
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    values = gdf[column].replace([np.inf, -np.inf], np.nan)

    vmin = values.min()
    vmax = values.max()

    norm = TwoSlopeNorm(
        vmin=vmin,
        vcenter=0,
        vmax=vmax
    )

    gdf.plot(
        column=column,
        cmap=cmap,
        norm=norm,
        linewidth=0.05,
        edgecolor="none",
        legend=True,
        ax=ax
    )

    ax.set_title(title, fontsize=13)
    ax.axis("off")
    plt.show()

plot_effect_map(
    gdf_geo,
    "GeoSMILE_location_Effect",
    f"Geo-SMILE-F Location Effect\nkernel={best_kernel}"
)


plot_effect_map(
    gdf_geo,
    "GeoSMILE_log_population_den_Effect",
    f"Geo-SMILE-F Log Population Density Effect\nkernel={best_kernel}"
)

plot_effect_map(
    gdf_geo,
    "GeoSMILE_pct_no_vehicle_Effect",
    f"Geo-SMILE-F No-Vehicle Effect\nkernel={best_kernel}"
)

plot_effect_map(
    gdf_geo,
    "GeoSMILE_TripMiles_mean_Effect",
    f"Geo-SMILE-F Trip Miles Effect\nkernel={best_kernel}"
)

plot_abs_map(
    gdf_geo,
    "GeoSMILE_location_AbsImportance",
    f"Geo-SMILE-F Absolute Location Importance\nkernel={best_kernel}"
)


n_cols = 3
n_rows = int(np.ceil(len(group_names) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()

for idx, group_name in enumerate(group_names):
    col = f"GeoSMILE_{group_name}_Effect"
    values = gdf_geo[col].values
    max_abs = np.nanmax(np.abs(values))

    if max_abs == 0 or np.isnan(max_abs):
        max_abs = 1.0

    gdf_geo.plot(
        column=col,
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
        linewidth=0.05,
        edgecolor="none",
        legend=True,
        ax=axes[idx]
    )

    axes[idx].set_title(f"{group_name} Effect")
    axes[idx].axis("off")

for j in range(len(group_names), len(axes)):
    axes[j].axis("off")

plt.suptitle(
    f"Geo-SMILE-F Spatial Feature Effect Maps\nSelected kernel: {best_kernel}",
    fontsize=16
)
plt.tight_layout()
plt.show()

# ============================================================
# ZIQI-STYLE SHAP EFFECT PLOTS WITH FEATURE-SPECIFIC BANDWIDTHS
# ============================================================

import xgboost as xgb
from statsmodels.nonparametric.smoothers_lowess import lowess

# ------------------------------------------------------------
# 1. Get exact model features
# ------------------------------------------------------------
try:
    model_features = list(xgb_model.get_booster().feature_names)
except:
    model_features = list(xgb_model.feature_names_in_)

X_shap = gdf_geo[model_features].copy()

# ------------------------------------------------------------
# 2. Compute native XGBoost SHAP values
# ------------------------------------------------------------
dmatrix = xgb.DMatrix(X_shap, feature_names=model_features)

shap_raw = xgb_model.get_booster().predict(
    dmatrix,
    pred_contribs=True
)

shap_values = shap_raw[:, :-1]

shap_df = pd.DataFrame(
    shap_values,
    columns=model_features,
    index=gdf_geo.index
)

# ------------------------------------------------------------
# 3. Feature-specific bandwidths
# Larger = smoother, smaller = more local
# Adjust these to match Ziqi's Chicago visual style
# ------------------------------------------------------------
feature_bandwidths = {
    "pct_18to34": 0.18,
    "pct_white": 0.12,
    "pct_bachelorORhigher": 0.18,
    "pct_no_vehicle": 0.16,
    "log_population_den": 0.18,
    "job_entropy": 0.14,
    "network_den": 0.16,
    "TripMiles_mean": 0.18,
    "pct_share": 0.14,
}

default_bandwidth = 0.16

# ------------------------------------------------------------
# 4. Plot all SHAP feature effects
# ------------------------------------------------------------
def plot_ziqi_shap_with_bandwidths(
    gdf,
    shap_df,
    model_features,
    feature_bandwidths,
    n_cols=3
):
    n_features = len(model_features)
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18, 4.5 * n_rows)
    )

    axes = axes.flatten()

    for idx, feature in enumerate(model_features):
        ax = axes[idx]

        df = pd.DataFrame({
            "x": gdf[feature],
            "y": shap_df[feature]
        }).replace([np.inf, -np.inf], np.nan).dropna()

        df = df.sort_values("x")

        x = df["x"].values
        y = df["y"].values

        bw = feature_bandwidths.get(feature, default_bandwidth)

        smoothed = lowess(
            endog=y,
            exog=x,
            frac=bw,
            return_sorted=True
        )

        ax.scatter(x, y, s=8, alpha=0.65, c="black")
        ax.plot(
            smoothed[:, 0],
            smoothed[:, 1],
            linewidth=2
        )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1,
            color="red"
        )

        ax.set_xlabel(feature)
        ax.set_ylabel("SHAP value")
        ax.set_title(feature)

    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        "Ziqi-style SHAP Feature Effect Plots with Feature-specific Bandwidths",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


plot_ziqi_shap_with_bandwidths(
    gdf_geo,
    shap_df,
    model_features,
    feature_bandwidths,
    n_cols=3
)


# ============================================================
# 19. OPTIONAL SHAP COMPARISON
# ============================================================

if RUN_SHAP_COMPARISON:
    try:
        import shap

        print("\nComputing SHAP values for comparison...")

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_geo)

        shap_df = pd.DataFrame(shap_values, columns=feature_cols)
        shap_df["target_idx"] = np.arange(len(X_geo))
        shap_df["tract_id"] = gdf_geo["tract_id"].values

        shap_long_rows = []

        for group_name, cols in feature_groups.items():
            group_shap = shap_df[cols].sum(axis=1).values

            gdf_geo[f"SHAP_{group_name}_Effect"] = group_shap
            gdf_geo[f"SHAP_{group_name}_AbsImportance"] = np.abs(group_shap)

            shap_long_rows.append(pd.DataFrame({
                "target_idx": np.arange(len(X_geo)),
                "tract_id": gdf_geo["tract_id"].values,
                "feature_group": group_name,
                "shap_effect": group_shap,
                "shap_abs_importance": np.abs(group_shap)
            }))

        shap_grouped_long_df = pd.concat(shap_long_rows, ignore_index=True)

        _long_rows = []
        for _gname in group_names:
            _long_rows.append(pd.DataFrame({
                "target_idx":             np.arange(n_obs),
                "tract_id":               gdf_geo["tract_id"].values,
                "feature_group":          _gname,
                "geosmile_effect":        local_effects_df[_gname].values,
                "geosmile_abs_importance": local_effects_abs_df[_gname].values,
            }))
        geosmile_feature_effects_df = pd.concat(_long_rows, ignore_index=True)

        geosmile_shap_comparison_df = geosmile_feature_effects_df.merge(
            shap_grouped_long_df,
            on=["target_idx", "tract_id", "feature_group"],
            how="inner"
        )

        comparison_rows = []

        for group_name in group_names:
            sub = geosmile_shap_comparison_df[
                geosmile_shap_comparison_df["feature_group"] == group_name
            ].copy()

            if len(sub) > 2:
                signed_pearson_corr, signed_pearson_p = pearsonr(
                    sub["geosmile_effect"],
                    sub["shap_effect"]
                )

                signed_spearman_corr, signed_spearman_p = spearmanr(
                    sub["geosmile_effect"],
                    sub["shap_effect"]
                )

                abs_pearson_corr, abs_pearson_p = pearsonr(
                    sub["geosmile_abs_importance"],
                    sub["shap_abs_importance"]
                )

                abs_spearman_corr, abs_spearman_p = spearmanr(
                    sub["geosmile_abs_importance"],
                    sub["shap_abs_importance"]
                )
            else:
                signed_pearson_corr, signed_pearson_p = np.nan, np.nan
                signed_spearman_corr, signed_spearman_p = np.nan, np.nan
                abs_pearson_corr, abs_pearson_p = np.nan, np.nan
                abs_spearman_corr, abs_spearman_p = np.nan, np.nan

            comparison_rows.append({
                "feature_group": group_name,
                "signed_pearson_corr": signed_pearson_corr,
                "signed_pearson_p": signed_pearson_p,
                "signed_spearman_corr": signed_spearman_corr,
                "signed_spearman_p": signed_spearman_p,
                "absolute_pearson_corr": abs_pearson_corr,
                "absolute_pearson_p": abs_pearson_p,
                "absolute_spearman_corr": abs_spearman_corr,
                "absolute_spearman_p": abs_spearman_p
            })

        geosmile_shap_correlation_df = pd.DataFrame(comparison_rows)

        print("\nGeo-SMILE-F vs SHAP comparison:")
        safe_display(geosmile_shap_correlation_df, n=20)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        shared_abs = max(
            np.nanmax(np.abs(gdf_geo["GeoSMILE_location_Effect"].values)),
            np.nanmax(np.abs(gdf_geo["SHAP_location_Effect"].values))
        )

        gdf_geo.plot(
            column="GeoSMILE_location_Effect",
            cmap="RdBu_r",
            vmin=-shared_abs,
            vmax=shared_abs,
            linewidth=0.05,
            edgecolor="none",
            legend=True,
            ax=axes[0]
        )
        axes[0].set_title("Geo-SMILE-F Location Effect")
        axes[0].axis("off")

        gdf_geo.plot(
            column="SHAP_location_Effect",
            cmap="RdBu_r",
            vmin=-shared_abs,
            vmax=shared_abs,
            linewidth=0.05,
            edgecolor="none",
            legend=True,
            ax=axes[1]
        )
        axes[1].set_title("SHAP Location Effect")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("\nSHAP comparison skipped.")
        print("Reason:", str(e))

        shap_df = None
        shap_grouped_long_df = None
        geosmile_shap_comparison_df = None
        geosmile_shap_correlation_df = None


# ============================================================
# 20. SAVE OUTPUTS
# ============================================================

xgb_importance_df.to_csv("xgboost_feature_importance.csv", index=False)

global_importances.to_frame().to_csv("geosmile_global_importances.csv")
faithfulness_summary_df.to_csv("geosmile_f_faithfulness_summary.csv", index=False)

local_effects_df.to_csv("geosmile_f_local_effects.csv", index=False)
local_r2_series.to_frame(name="local_r2").to_csv("geosmile_f_local_fidelity.csv", index=False)
final_metric_assessment_df.to_csv("geosmile_f_final_metric_assessment.csv", index=False)

stability_detail_df.to_csv("geosmile_f_stability_details.csv", index=False)
stability_metrics_df.to_csv("geosmile_f_stability_summary.csv", index=False)
consistency_metrics_df.to_csv("geosmile_f_consistency_summary.csv", index=False)
topk_consistency_df.to_csv("geosmile_f_topk_consistency_details.csv", index=False)

if RUN_SHAP_COMPARISON and "geosmile_shap_correlation_df" in globals() and geosmile_shap_correlation_df is not None:
    geosmile_shap_correlation_df.to_csv("geosmile_f_vs_shap_correlation.csv", index=False)
    geosmile_shap_comparison_df.to_csv("geosmile_f_vs_shap_long_comparison.csv", index=False)

gdf_geo.to_crs("EPSG:4326").to_file(
    "geosmile_f_best_results.geojson",
    driver="GeoJSON"
)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n================ Geo-SMILE-F Final Summary ================")
print("Explanation type:               Feature-level Geo-SMILE-F (SMILE algorithm)")
print("Comparable baseline:            Ziqi Li-style XGBoost + SHAP spatial effect maps")
print("Spatial unit:                   Chicago census tract")
print("Target column:                  ", target_col)
print("Feature groups:                 ", group_names)
print("Model features:                 ", feature_cols)
print("Selected kernel:                ", best_kernel)
print("Response type:                  Wasserstein W1 (distributional)")
print("Perturbation pool size:          ", N_PERTURBATIONS)
print("XGBoost hold-out R2:             ", blackbox_r2)
print("XGBoost hold-out MAE:            ", blackbox_mae)
print("XGBoost hold-out RMSE:           ", blackbox_rmse)
print("Global surrogate R2 (test):      ", global_fidelity["R2"])
print("Mean local R2 (per-tract):       ", np.nanmean(local_r2))
print("Median local R2 (per-tract):     ", np.nanmedian(local_r2))
print("Mean stability Jaccard:          ", stability_detail_df["jaccard_stability"].mean())
print("Mean coefficient CV:             ", np.nanmean(coef_cv))
print("Median coefficient CV:           ", np.nanmedian(coef_cv))
print(f"Mean Top-{TOP_K_FEATURES_FOR_STABILITY} Jaccard Consistency: ",
      topk_consistency_df["jaccard_consistency"].mean())
print(f"Median Top-{TOP_K_FEATURES_FOR_STABILITY} Jaccard Consistency: ",
      topk_consistency_df["jaccard_consistency"].median())
print("ATT / attribution accuracy:      Not computed because no attribution ground truth exists")
print("============================================================")

print("\nSaved files:")
print("xgboost_feature_importance.csv")
print("geosmile_global_importances.csv")
print("geosmile_f_faithfulness_summary.csv")
print("geosmile_f_local_effects.csv")
print("geosmile_f_local_fidelity.csv")
print("geosmile_f_final_metric_assessment.csv")
print("geosmile_f_stability_details.csv")
print("geosmile_f_stability_summary.csv")
print("geosmile_f_consistency_summary.csv")
print("geosmile_f_topk_consistency_details.csv")
print("geosmile_f_best_results.geojson")

if RUN_SHAP_COMPARISON and "geosmile_shap_correlation_df" in globals() and geosmile_shap_correlation_df is not None:
    print("geosmile_f_vs_shap_correlation.csv")
    print("geosmile_f_vs_shap_long_comparison.csv")

print("\nTop 15 Geo-SMILE-F location absolute importance tracts:")
safe_display(
    gdf_geo[[
        "tract_id",
        target_col,
        "Predicted_Log_Demand",
        "GeoSMILE_location_Effect",
        "GeoSMILE_location_AbsImportance"
    ]].sort_values("GeoSMILE_location_AbsImportance", ascending=False),
    n=15
)