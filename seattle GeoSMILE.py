# ============================================================
# Geo-SMILE: Geo + Feature + Cell Explainability Pipeline
# Extension of SMILE (Aslansefat) to spatial cases
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

features = [
    "bathrooms", "sqft_living", "sqft_lot", "grade",
    "condition", "waterfront", "view", "age",
    "UTM_X", "UTM_Y"
]

spatial_features = ["UTM_X", "UTM_Y"]
target = "log_price"

required_cols = features + [target]
missing_cols = [c for c in required_cols if c not in data.columns]
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


# ============================================================
# Step 3: Spatial Groups via KMeans (Geo branch)
# ============================================================

coords_scaler = StandardScaler()
coords_train_scaled = coords_scaler.fit_transform(X_train[spatial_features])

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

print(f"\nBlack-box — Train R²: {base_train_r2:.4f} | Test R²: {base_test_r2:.4f}")

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


def select_best_kernel(M_tr, M_te, shifts_tr, shifts_te):
    results = {}
    for k in KERNELS:
        w = np.clip(apply_kernel(shifts_tr, k), 1e-8, None)
        ridge = Ridge(alpha=1.0)
        ridge.fit(M_tr, shifts_tr, sample_weight=w)
        results[k] = r2_score(shifts_te, ridge.predict(M_te))
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
    shift = wasserstein_distance(original_preds[idx], pert_preds[idx])
    y_shifts_geo.append(shift)
    M_geo[j, idx] = 1

y_shifts_geo = np.array(y_shifts_geo)
print(f"\nGeo shifts — mean: {y_shifts_geo.mean():.4f}, std: {y_shifts_geo.std():.4f}")


# ============================================================
# Step 7: Kernel Selection — Geo Branch
# ============================================================

M_geo_tr, M_geo_te, sg_tr, sg_te = train_test_split(
    M_geo, y_shifts_geo, test_size=0.25, random_state=42
)

best_k_geo, k_scores_geo = select_best_kernel(M_geo_tr, M_geo_te, sg_tr, sg_te)

print("\nKernel R² — Geo branch:")
for k, v in sorted(k_scores_geo.items(), key=lambda x: -x[1]):
    marker = " <- best" if k == best_k_geo else ""
    print(f"  {k:<14} {v:.4f}{marker}")

geo_w_tr = np.clip(apply_kernel(sg_tr, best_k_geo), 1e-8, None)


# ============================================================
# Step 8: Weighted Ridge Surrogate — Geo Branch
# ============================================================

ridge_geo = Ridge(alpha=1.0)
ridge_geo.fit(M_geo_tr, sg_tr, sample_weight=geo_w_tr)

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

print("\nGeo importance summary:")
print(geo_importance_abs.describe())


# ============================================================
# Step 9: Feature Perturbations (K sets, feature masking)
# ============================================================

n_feats = len(features)
K_feat  = 3000

M_feat        = np.zeros((K_feat, n_feats), dtype=np.int8)
y_shifts_feat = []

for j in tqdm(range(K_feat), desc="Feature perturbations"):
    mask = np.random.randint(0, 2, size=n_feats)
    M_feat[j] = mask

    X_pert = X_train.copy()
    for ki, feat in enumerate(features):
        if mask[ki] == 0:
            X_pert[feat] = baseline_values[feat]

    pert_preds = automl.predict(X_pert)
    # Wasserstein over all points (all are affected by feature masking)
    shift = wasserstein_distance(original_preds, pert_preds)
    y_shifts_feat.append(shift)

y_shifts_feat = np.array(y_shifts_feat)
print(f"\nFeature shifts — mean: {y_shifts_feat.mean():.4f}, std: {y_shifts_feat.std():.4f}")


# ============================================================
# Step 10: Kernel Selection — Feature Branch
# ============================================================

M_feat_tr, M_feat_te, sf_tr, sf_te = train_test_split(
    M_feat, y_shifts_feat, test_size=0.25, random_state=42
)

best_k_feat, k_scores_feat = select_best_kernel(M_feat_tr, M_feat_te, sf_tr, sf_te)

print("\nKernel R² — Feature branch:")
for k, v in sorted(k_scores_feat.items(), key=lambda x: -x[1]):
    marker = " <- best" if k == best_k_feat else ""
    print(f"  {k:<14} {v:.4f}{marker}")

feat_w_tr = np.clip(apply_kernel(sf_tr, best_k_feat), 1e-8, None)


# ============================================================
# Step 11: Weighted Ridge Surrogate — Feature Branch
# ============================================================

ridge_feat = Ridge(alpha=1.0)
ridge_feat.fit(M_feat_tr, sf_tr, sample_weight=feat_w_tr)

pred_feat_tr = ridge_feat.predict(M_feat_tr)
pred_feat_te = ridge_feat.predict(M_feat_te)

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

feature_importance     = pd.Series(ridge_feat.coef_, index=features, name="feature_importance")
feature_importance_abs = feature_importance.abs()

print("\nFeature importance ranking:")
display(feature_importance_abs.sort_values(ascending=False).to_frame())


# ============================================================
# Step 12: Cell Explainability
# ============================================================
# Both importance vectors are min-max normalized to [0, 1]
# so that geo (n_points) and feature (n_features) scales are
# equalized before combining.
# cell[i, j] = geo_norm[i] * feat_norm[j]
# Result: (n_points x n_features) matrix
# ============================================================

_eps = 1e-12

geo_norm  = (geo_importance_abs  - geo_importance_abs.min())  / (geo_importance_abs.max()  - geo_importance_abs.min()  + _eps)
feat_norm = (feature_importance_abs - feature_importance_abs.min()) / (feature_importance_abs.max() - feature_importance_abs.min() + _eps)

cell_matrix = np.outer(geo_norm.values, feat_norm.values)
cell_df     = pd.DataFrame(cell_matrix, index=geo_norm.index, columns=features)

print("\nCell importance matrix shape:", cell_df.shape)
print("Cell importance summary:")
display(cell_df.describe())


# ============================================================
# Step 13: GeoDataFrame
# ============================================================

data_train = data.loc[X_train.index].copy()
data_train["geo_importance"]     = geo_importance
data_train["geo_importance_abs"] = geo_importance_abs
data_train["geo_norm"]           = geo_norm

for feat in features:
    data_train[f"cell_{feat}"] = cell_df[feat]

data_train["geometry"] = gpd.points_from_xy(data_train["UTM_X"], data_train["UTM_Y"])
gdf_train = gpd.GeoDataFrame(data_train, geometry="geometry", crs="EPSG:32610")


# ============================================================
# Step 14: Map — Geo Importance (Absolute)
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))
gdf_train.plot(column="geo_importance_abs", cmap="plasma", legend=True, markersize=50, ax=ax)
ax.set_title("Geo-SMILE: Point-Level Geo Importance", fontsize=15)
ax.set_xlabel("UTM_X"); ax.set_ylabel("UTM_Y"); ax.axis("equal")
plt.tight_layout(); plt.show()


# ============================================================
# Step 15: Map — Geo Importance (Signed)
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))
gdf_train.plot(column="geo_importance", cmap="Spectral", legend=True, markersize=50, ax=ax)
ax.set_title("Geo-SMILE: Signed Geo Importance", fontsize=15)
ax.set_xlabel("UTM_X"); ax.set_ylabel("UTM_Y"); ax.axis("equal")
plt.tight_layout(); plt.show()


# ============================================================
# Step 16: Bar Chart — Global Feature Importance
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))
feature_importance_abs.sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Geo-SMILE: Global Feature Importance", fontsize=14)
ax.set_xlabel("Importance Score")
plt.tight_layout(); plt.show()


# ============================================================
# Step 17: Heatmap — Cell Explainability (Top-30 Points)
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
# Step 18: Map — Cell Importance for Top Feature
# ============================================================

top_feature = feature_importance_abs.idxmax()
print(f"\nMost spatially influential feature: {top_feature}")

fig, ax = plt.subplots(figsize=(12, 8))
gdf_train.plot(column=f"cell_{top_feature}", cmap="hot_r", legend=True, markersize=50, ax=ax)
ax.set_title(f"Cell Importance Map: {top_feature}", fontsize=15)
ax.set_xlabel("UTM_X"); ax.set_ylabel("UTM_Y"); ax.axis("equal")
plt.tight_layout(); plt.show()


# ============================================================
# Step 19: Wasserstein Shift Distributions
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(y_shifts_geo,  bins=30, kde=True, ax=axes[0])
axes[0].set_title("Wasserstein Shifts — Geo Branch")
axes[0].set_xlabel("Shift")

sns.histplot(y_shifts_feat, bins=30, kde=True, ax=axes[1], color="darkorange")
axes[1].set_title("Wasserstein Shifts — Feature Branch")
axes[1].set_xlabel("Shift")

plt.tight_layout(); plt.show()


# ============================================================
# Step 20: Observed vs Predicted Shifts
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, obs, pred, title in [
    (axes[0], sg_te, pred_geo_te,
     f"Geo Surrogate (R2={geo_fidelity['R2 Fidelity']:.3f}, kernel={best_k_geo})"),
    (axes[1], sf_te, pred_feat_te,
     f"Feature Surrogate (R2={feat_fidelity['R2 Fidelity']:.3f}, kernel={best_k_feat})")
]:
    ax.scatter(obs, pred, alpha=0.6)
    lo = min(obs.min(), pred.min())
    hi = max(obs.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="red")
    ax.set_title(title)
    ax.set_xlabel("Observed Shift")
    ax.set_ylabel("Predicted Shift")

plt.tight_layout(); plt.show()


# ============================================================
# Step 21: Surrogate Model Comparison (both branches)
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

    model.fit(M_geo_tr,  sg_tr, **kw_geo)
    geo_r2s[name] = r2_score(sg_te, model.predict(M_geo_te))

    model.fit(M_feat_tr, sf_tr, **kw_feat)
    feat_r2s[name] = r2_score(sf_te, model.predict(M_feat_te))

comparison_df = pd.DataFrame({
    "Model":      list(geo_r2s.keys()),
    "Geo R2":     list(geo_r2s.values()),
    "Feature R2": list(feat_r2s.values()),
}).sort_values("Geo R2", ascending=False).reset_index(drop=True)

print("\nSurrogate model comparison:")
display(comparison_df)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(comparison_df))
w = 0.35
ax.bar(x - w/2, comparison_df["Geo R2"],     w, label="Geo Branch",     color="steelblue")
ax.bar(x + w/2, comparison_df["Feature R2"], w, label="Feature Branch", color="darkorange")
ax.set_xticks(x)
ax.set_xticklabels(comparison_df["Model"], rotation=45, ha="right")
ax.set_ylabel("Test R2")
ax.set_title("Surrogate Model Comparison — Geo vs Feature Branch")
ax.legend(); ax.grid(axis="y")
plt.tight_layout(); plt.show()


# ============================================================
# Step 22: Stability — Geo Branch
# ============================================================

n_repeats_stab = 10
n_subsets_stab = 1000
top_pct_geo    = 20   # top-20% for geo (many points)
top_pct_feat   = 50   # top-50% for feature (only 10 features)

geo_runs     = []
geo_fid_runs = []

for seed in tqdm(range(10, 10 + n_repeats_stab), desc="Geo stability runs"):
    np.random.seed(seed)
    M_r = np.zeros((n_subsets_stab, n_points), dtype=np.int8)
    ys  = []

    for j in range(n_subsets_stab):
        g   = np.random.randint(min_groups, max_groups + 1)
        sel = np.random.choice(n_groups, size=g, replace=False)
        idx = np.where(np.isin(spatial_group_labels, sel))[0]
        if len(idx) == 0:
            ys.append(0.0); continue
        X_p = X_train.copy()
        X_p.iloc[idx] = baseline_values.values
        pp  = automl.predict(X_p)
        ys.append(wasserstein_distance(original_preds[idx], pp[idx]))
        M_r[j, idx] = 1

    ys = np.array(ys)
    Mt, Me, st, se = train_test_split(M_r, ys, test_size=0.25, random_state=42)
    w  = np.clip(apply_kernel(st, best_k_geo), 1e-8, None)
    rr = Ridge(alpha=1.0)
    rr.fit(Mt, st, sample_weight=w)
    geo_fid_runs.append(r2_score(se, rr.predict(Me)))
    geo_runs.append(pd.Series(rr.coef_, index=X_train.index).abs())

geo_runs_df = pd.DataFrame(geo_runs).T
geo_runs_df.columns = [f"run_{i+1}" for i in range(n_repeats_stab)]

geo_sp, geo_pr, geo_jc = [], [], []

for a, b in combinations(geo_runs_df.columns, 2):
    sa, sb = geo_runs_df[a].values, geo_runs_df[b].values
    geo_sp.append(spearmanr(sa, sb)[0])
    geo_pr.append(pearsonr(sa, sb)[0])
    ta = np.percentile(sa, 100 - top_pct_geo)
    tb = np.percentile(sb, 100 - top_pct_geo)
    A  = set(geo_runs_df.index[sa >= ta])
    B  = set(geo_runs_df.index[sb >= tb])
    geo_jc.append(len(A & B) / len(A | B) if (A | B) else 1.0)

geo_stability = {
    "Fidelity R2 Mean":              np.mean(geo_fid_runs),
    "Fidelity R2 Std":               np.std(geo_fid_runs),
    "Spearman Stability Mean":       np.nanmean(geo_sp),
    "Spearman Stability Std":        np.nanstd(geo_sp),
    "Pearson Stability Mean":        np.nanmean(geo_pr),
    "Pearson Stability Std":         np.nanstd(geo_pr),
    f"Jaccard Top-{top_pct_geo}% Mean": np.nanmean(geo_jc),
    f"Jaccard Top-{top_pct_geo}% Std":  np.nanstd(geo_jc),
}

print("\n--- Geo Stability ---")
for k, v in geo_stability.items():
    print(f"  {k}: {v:.4f}")


# ============================================================
# Step 23: Stability — Feature Branch
# ============================================================

feat_runs     = []
feat_fid_runs = []

for seed in tqdm(range(10, 10 + n_repeats_stab), desc="Feature stability runs"):
    np.random.seed(seed)
    M_r = np.zeros((n_subsets_stab, n_feats), dtype=np.int8)
    ys  = []

    for j in range(n_subsets_stab):
        mask = np.random.randint(0, 2, size=n_feats)
        M_r[j] = mask
        X_p = X_train.copy()
        for ki, feat in enumerate(features):
            if mask[ki] == 0:
                X_p[feat] = baseline_values[feat]
        pp = automl.predict(X_p)
        ys.append(wasserstein_distance(original_preds, pp))

    ys = np.array(ys)
    Mt, Me, st, se = train_test_split(M_r, ys, test_size=0.25, random_state=42)
    w  = np.clip(apply_kernel(st, best_k_feat), 1e-8, None)
    rr = Ridge(alpha=1.0)
    rr.fit(Mt, st, sample_weight=w)
    feat_fid_runs.append(r2_score(se, rr.predict(Me)))
    feat_runs.append(pd.Series(rr.coef_, index=features).abs())

feat_runs_df = pd.DataFrame(feat_runs).T
feat_runs_df.columns = [f"run_{i+1}" for i in range(n_repeats_stab)]

feat_sp, feat_pr, feat_jc = [], [], []

for a, b in combinations(feat_runs_df.columns, 2):
    sa, sb = feat_runs_df[a].values, feat_runs_df[b].values
    feat_sp.append(spearmanr(sa, sb)[0])
    feat_pr.append(pearsonr(sa, sb)[0])
    ta = np.percentile(sa, 100 - top_pct_feat)
    tb = np.percentile(sb, 100 - top_pct_feat)
    A  = set(feat_runs_df.index[sa >= ta])
    B  = set(feat_runs_df.index[sb >= tb])
    feat_jc.append(len(A & B) / len(A | B) if (A | B) else 1.0)

feat_stability = {
    "Fidelity R2 Mean":              np.mean(feat_fid_runs),
    "Fidelity R2 Std":               np.std(feat_fid_runs),
    "Spearman Stability Mean":       np.nanmean(feat_sp),
    "Spearman Stability Std":        np.nanstd(feat_sp),
    "Pearson Stability Mean":        np.nanmean(feat_pr),
    "Pearson Stability Std":         np.nanstd(feat_pr),
    f"Jaccard Top-{top_pct_feat}% Mean": np.nanmean(feat_jc),
    f"Jaccard Top-{top_pct_feat}% Std":  np.nanstd(feat_jc),
}

print("\n--- Feature Stability ---")
for k, v in feat_stability.items():
    print(f"  {k}: {v:.4f}")


# ============================================================
# Step 24: Sparsity & Entropy — helper function
# ============================================================

def sparsity_entropy_metrics(scores_raw, label=""):
    s   = np.nan_to_num(np.asarray(scores_raw, dtype=float), nan=0.0)
    n   = len(s)
    eps = 1e-12

    # Sparsity
    q90          = np.percentile(s, 90)
    top10_ratio  = (s >= q90).sum() / n

    l1_n = np.sum(np.abs(s))
    l2_n = np.sqrt(np.sum(s ** 2))
    hoyer = (np.sqrt(n) - l1_n / (l2_n + eps)) / (np.sqrt(n) - 1) if l2_n > eps else np.nan

    sorted_s = np.sort(s)
    cumul    = np.cumsum(sorted_s)
    gini     = (n + 1 - 2 * np.sum(cumul) / (cumul[-1] + eps)) / n if cumul[-1] > eps else np.nan

    # Entropy
    total = s.sum()
    if total > eps:
        p        = s[s > 0] / total
        ent      = -np.sum(p * np.log(p))
        norm_ent = ent / np.log(n)
        eff_n    = np.exp(ent)
        eff_r    = eff_n / n
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
        val_str = f"{v:.4f}" if (v is not None and not np.isnan(v)) else "NaN"
        print(f"  {k}: {val_str}")

    return metrics


geo_sp_ent  = sparsity_entropy_metrics(geo_importance_abs.values,     label="Geo")
feat_sp_ent = sparsity_entropy_metrics(feature_importance_abs.values,  label="Feature")
cell_sp_ent = sparsity_entropy_metrics(cell_matrix.flatten(),           label="Cell (flattened)")


# ============================================================
# Step 25: Combined Evaluation Table
# ============================================================

def _make_rows(metrics, branch, category):
    return [
        {"Branch": branch, "Category": category, "Metric": k, "Value": v}
        for k, v in metrics.items()
    ]

rows = (
    _make_rows(geo_fidelity,  "Geo",     "Fidelity")  +
    _make_rows(feat_fidelity, "Feature", "Fidelity")  +
    _make_rows(geo_stability,  "Geo",    "Stability") +
    _make_rows(feat_stability, "Feature","Stability") +
    _make_rows(geo_sp_ent,   "Geo",     "Sparsity/Entropy") +
    _make_rows(feat_sp_ent,  "Feature", "Sparsity/Entropy") +
    _make_rows(cell_sp_ent,  "Cell",    "Sparsity/Entropy")
)

eval_df = pd.DataFrame(rows)
print("\n--- Full Geo-SMILE Evaluation Metrics ---")
display(eval_df)


# ============================================================
# Step 26: Stability Maps — Geo Branch
# ============================================================

gdf_train["geo_imp_mean"] = geo_runs_df.mean(axis=1).reindex(gdf_train.index)
gdf_train["geo_imp_std"]  = geo_runs_df.std(axis=1).reindex(gdf_train.index)
eps_cv = 1e-12
gdf_train["geo_imp_cv"] = gdf_train["geo_imp_std"] / (gdf_train["geo_imp_mean"] + eps_cv)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

gdf_train.plot(column="geo_imp_mean", cmap="plasma", legend=True, markersize=40, ax=axes[0])
axes[0].set_title("Geo Importance — Repeated-Run Mean", fontsize=13)
axes[0].axis("equal")

gdf_train.plot(column="geo_imp_std", cmap="magma", legend=True, markersize=40, ax=axes[1])
axes[1].set_title("Geo Importance — Instability (Std)", fontsize=13)
axes[1].axis("equal")

plt.tight_layout(); plt.show()


# ============================================================
# Step 27: Top-10% Sparse Important Points Map
# ============================================================

q90_geo = np.percentile(geo_importance_abs.values, 90)
gdf_train["top10_geo"] = (geo_importance_abs >= q90_geo).astype(int)

fig, ax = plt.subplots(figsize=(12, 8))

gdf_train[gdf_train["top10_geo"] == 0].plot(
    ax=ax, markersize=15, alpha=0.20, color="lightgrey"
)
gdf_train[gdf_train["top10_geo"] == 1].plot(
    ax=ax, markersize=70, color="red", alpha=0.85, label="Top 10% Geo Importance"
)

ax.legend()
ax.set_title("Geo-SMILE Sparse Important Points: Top 10%", fontsize=15)
ax.set_xlabel("UTM_X"); ax.set_ylabel("UTM_Y"); ax.axis("equal")
plt.tight_layout(); plt.show()


print("\nGeo-SMILE full pipeline completed successfully.")
