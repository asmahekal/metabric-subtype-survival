#!/usr/bin/env python3

import argparse, json, os, platform, re, shutil, sys, time, traceback, warnings
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix, roc_curve,
                             precision_recall_curve)

from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import (RandomSurvivalForest, ExtraSurvivalTrees, GradientBoostingSurvivalAnalysis,
                             ComponentwiseGradientBoostingSurvivalAnalysis)
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc,
                            integrated_brier_score, brier_score)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False
try:
    import statsmodels.api as sm
    from statsmodels.duration.hazard_regression import PHReg
    HAS_SM = True
except Exception:
    HAS_SM = False

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)


SUBTYPE_COL = "pam50_+_claudin-low_subtype"
OUTCOME_COLS = ["overall_survival_months", "overall_survival", "death_from_cancer"]
ID_LIKE = ["patient_id", "cohort", "cancer_type", "oncotree_code", "tumor_other_histologic_subtype"]
CLIN_NUM = ["age_at_diagnosis", "lymph_nodes_examined_positive", "mutation_count", "nottingham_prognostic_index",
            "tumor_size", "tumor_stage", "neoplasm_histologic_grade"]
CLIN_CAT = ["type_of_breast_surgery", "cellularity", "chemotherapy", "er_status", "pr_status", "her2_status",
            "hormone_therapy", "radio_therapy", "inferred_menopausal_state", SUBTYPE_COL, "integrative_cluster",
            "3-gene_classifier_subtype", "cancer_type_detailed", "primary_tumor_laterality",
            "her2_status_measured_by_snp6", "er_status_measured_by_ihc"]
TREATMENT_COLS = ["type_of_breast_surgery", "chemotherapy", "hormone_therapy", "radio_therapy"]
HORIZON = 60.0
TD_TIMES = [36.0, 60.0, 120.0]
TAU = 120.0
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})


class Logger:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.t0 = time.time()

    def __call__(self, *msg):
        s = f"[{time.time() - self.t0:8.1f}s] " + " ".join(str(m) for m in msg)
        print(s, flush=True)
        self.f.write(s + "\n"); self.f.flush()


def norm_col(c):
    return re.sub(r"\s+", "_", str(c).strip().lower())


def savefig(fig, outdir, name):
    fig.savefig(os.path.join(outdir, "figures", name + ".png"))
    fig.savefig(os.path.join(outdir, "figures", "pdf", name + ".pdf"))
    plt.close(fig)


def ci95(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = se * stats.t.ppf(0.975, len(x) - 1)
    return (m - h, m + h)


def corrected_ttest(diffs, n_train, n_test):
    d = np.asarray(diffs, float); d = d[~np.isnan(d)]
    k = len(d)
    if k < 3 or np.allclose(d.std(ddof=1), 0):
        return np.nan, np.nan, (np.nan, np.nan)
    var = d.var(ddof=1) * (1.0 / k + n_test / max(n_train, 1))
    t = d.mean() / np.sqrt(var)
    p = 2 * stats.t.sf(abs(t), k - 1)
    h = stats.t.ppf(0.975, k - 1) * np.sqrt(var)
    return t, p, (d.mean() - h, d.mean() + h)


def load_data(path, log, exclude_treatment=False):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [norm_col(c) for c in df.columns]
    log(f"Loaded {path}: {df.shape[0]} rows x {df.shape[1]} columns")
    for c in ["overall_survival_months", "death_from_cancer"]:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found.")


    dfc = df["death_from_cancer"].astype(str).str.strip().str.lower()
    df["os_time"] = pd.to_numeric(df["overall_survival_months"], errors="coerce")
    df["os_event"] = np.where(dfc.isin(["died of disease", "died of other causes"]), 1,
                              np.where(dfc == "living", 0, np.nan))
    df["dss_event"] = np.where(dfc == "died of disease", 1,
                               np.where(dfc.isin(["living", "died of other causes"]), 0, np.nan))
    df["cause"] = np.where(dfc == "died of disease", 1, np.where(dfc == "died of other causes", 2,
                           np.where(dfc == "living", 0, np.nan)))
    if "overall_survival" in df.columns:
        ok = df["os_event"].notna() & df["overall_survival"].notna()
        agree = (df.loc[ok, "os_event"] == 1 - df.loc[ok, "overall_survival"]).mean()
        log(f"Agreement between death_from_cancer-derived OS event and 'overall_survival' flag: {agree:.3f}")
    n0 = len(df)
    df = df[df["os_time"].notna() & (df["os_time"] > 0) & df["os_event"].notna()].reset_index(drop=True)
    log(f"Excluded {n0 - len(df)} patients with missing/zero follow-up or unknown vital status -> n = {len(df)}")
    df["os_time"] = df["os_time"].clip(lower=0.1)


    if SUBTYPE_COL in df.columns:
        df[SUBTYPE_COL] = df[SUBTYPE_COL].astype(str).str.strip().replace({"nan": np.nan, "NC": np.nan})
    for c in ["er_status", "pr_status", "her2_status"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.capitalize().replace({"Nan": np.nan})


    mut_cols = [c for c in df.columns if c.endswith("_mut")]
    for c in mut_cols:
        s = df[c].astype(str).str.strip()
        df[c] = (~s.isin(["0", "0.0", "nan", "", "None", "NaN"])).astype(int)
    clin_num = [c for c in CLIN_NUM if c in df.columns]
    clin_cat = [c for c in CLIN_CAT if c in df.columns]
    if exclude_treatment:
        clin_cat = [c for c in clin_cat if c not in TREATMENT_COLS]
    for c in clin_num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in clin_cat:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
    derived = {"os_time", "os_event", "dss_event", "cause"}
    reserved = set(clin_num) | set(CLIN_CAT) | set(OUTCOME_COLS) | set(ID_LIKE) | set(mut_cols) | derived
    gene_cols = [c for c in df.columns if c not in reserved and pd.api.types.is_numeric_dtype(df[c])]
    log(f"Features: {len(clin_num)} clinical numeric, {len(clin_cat)} clinical categorical, "
        f"{len(gene_cols)} mRNA genes, {len(mut_cols)} mutation flags")
    return df, dict(clin_num=clin_num, clin_cat=clin_cat, genes=gene_cols, muts=mut_cols)


def define_subgroups(df):
    st = df.get(SUBTYPE_COL, pd.Series(np.nan, index=df.index))
    g = {"All": np.ones(len(df), bool)}
    if {"er_status", "pr_status", "her2_status"} <= set(df.columns):
        g["TNBC"] = ((df.er_status == "Negative") & (df.pr_status == "Negative") & (df.her2_status == "Negative")).values
    g["Basal"] = (st == "Basal").values
    g["HER2-enriched"] = (st == "Her2").values
    g["Claudin-low"] = (st == "claudin-low").values
    g["Aggressive (Basal/HER2/CL)"] = st.isin(["Basal", "Her2", "claudin-low"]).values
    g["LumB"] = (st == "LumB").values
    g["LumA"] = (st == "LumA").values
    return {k: v for k, v in g.items() if v.sum() >= 40}


class FoldPreprocessor:

    def __init__(self, feats, groups, n_genes=50, min_mut_freq=0.02):
        self.feats, self.groups, self.n_genes, self.min_mut_freq = feats, groups, n_genes, min_mut_freq

    def fit(self, X, y):
        num = [c for c in self.groups["clin_num"] if c in self.feats]
        cat = [c for c in self.groups["clin_cat"] if c in self.feats]
        genes = [c for c in self.groups["genes"] if c in self.feats]
        muts = [c for c in self.groups["muts"] if c in self.feats]
        cat = [c for c in cat if X[c].nunique(dropna=True) > 1]
        num = [c for c in num if X[c].notna().sum() > 10 and X[c].nunique() > 1]
        if genes:
            ev, tm = y["event"], y["time"]
            scores = {}
            for g in genes:
                v = X[g].values.astype(float)
                ok = ~np.isnan(v)
                if ok.sum() < 20 or np.nanstd(v) == 0:
                    continue
                try:
                    scores[g] = abs(concordance_index_censored(ev[ok], tm[ok], v[ok])[0] - 0.5)
                except Exception:
                    pass
            genes = sorted(scores, key=scores.get, reverse=True)[: self.n_genes]
        if muts:
            muts = [c for c in muts if X[c].mean() >= self.min_mut_freq]
        self.num_, self.cat_, self.genes_, self.muts_ = num, cat, genes, muts
        tr = []
        if num + genes:
            tr.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num + genes))
        if cat:
            try:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10)
            except TypeError:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            tr.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)]), cat))
        if muts:
            tr.append(("mut", "passthrough", muts))
        self.ct_ = ColumnTransformer(tr, remainder="drop", verbose_feature_names_out=False)
        self.ct_.fit(X)
        self.names_ = [re.sub(r"[\[\]<>,]", "_", str(n)) for n in self.ct_.get_feature_names_out()]
        return self

    def transform(self, X):
        return np.asarray(self.ct_.transform(X), dtype=float)


def breslow(risk_lin, time_, event):
    order = np.argsort(time_)
    t, e, r = time_[order], event[order].astype(bool), np.exp(np.clip(risk_lin[order], -30, 30))
    uniq = np.unique(t[e])
    at_risk = np.array([r[t >= u].sum() for u in uniq])
    d = np.array([((t == u) & e).sum() for u in uniq])
    return uniq, np.cumsum(d / at_risk)


class XGBCox(BaseEstimator):
    def __init__(self, n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                 min_child_weight=5, reg_lambda=1.0, random_state=0, n_jobs=1):
        self.n_estimators, self.learning_rate, self.max_depth = n_estimators, learning_rate, max_depth
        self.subsample, self.colsample_bytree, self.min_child_weight = subsample, colsample_bytree, min_child_weight
        self.reg_lambda, self.random_state, self.n_jobs = reg_lambda, random_state, n_jobs

    def fit(self, X, y):
        lab = np.where(y["event"], y["time"], -y["time"])
        self.model_ = xgb.XGBRegressor(objective="survival:cox", n_estimators=self.n_estimators,
                                       learning_rate=self.learning_rate, max_depth=self.max_depth,
                                       subsample=self.subsample, colsample_bytree=self.colsample_bytree,
                                       min_child_weight=self.min_child_weight, reg_lambda=self.reg_lambda,
                                       random_state=self.random_state, n_jobs=self.n_jobs, tree_method="hist")
        self.model_.fit(X, lab)
        self.bt_, self.bH_ = breslow(self.predict(X), y["time"], y["event"])
        return self

    def predict(self, X):
        return self.model_.predict(X, output_margin=True)

    def surv_at(self, X, times):
        H0 = np.interp(times, self.bt_, self.bH_, left=0.0)
        return np.exp(-np.outer(np.exp(np.clip(self.predict(X), -30, 30)), H0))


class SignGuardSVM(BaseEstimator):

    def __init__(self, alpha=1.0, random_state=0):
        self.alpha, self.random_state = alpha, random_state

    def fit(self, X, y):
        self.m_ = FastSurvivalSVM(alpha=self.alpha, rank_ratio=1.0, max_iter=100, optimizer="avltree",
                                  random_state=self.random_state).fit(X, y)
        c = concordance_index_censored(y["event"], y["time"], self.m_.predict(X))[0]
        self.sign_ = 1.0 if c >= 0.5 else -1.0
        return self

    def predict(self, X):
        return self.sign_ * self.m_.predict(X)


class TunedCoxnet(BaseEstimator):

    def __init__(self, l1_ratio=0.5, n_alphas=8, random_state=0):
        self.l1_ratio, self.n_alphas, self.random_state = l1_ratio, n_alphas, random_state

    def fit(self, X, y):
        path = CoxnetSurvivalAnalysis(l1_ratio=self.l1_ratio, alpha_min_ratio=0.01, n_alphas=30, max_iter=5000).fit(X, y)
        alphas = path.alphas_[np.linspace(3, len(path.alphas_) - 1, self.n_alphas).astype(int)]
        try:
            gs = GridSearchCV(CoxnetSurvivalAnalysis(l1_ratio=self.l1_ratio, max_iter=5000),
                              {"alphas": [[a] for a in alphas]},
                              cv=KFold(3, shuffle=True, random_state=self.random_state), error_score=0.5, n_jobs=1)
            gs.fit(X, y)
            best = gs.best_params_["alphas"]
        except Exception:
            best = [alphas[len(alphas) // 2]]
        self.m_ = CoxnetSurvivalAnalysis(l1_ratio=self.l1_ratio, alphas=best, fit_baseline_model=True,
                                         max_iter=5000).fit(X, y)
        self.coef_ = self.m_.coef_[:, 0]
        return self

    def predict(self, X):
        return self.m_.predict(X)

    def predict_survival_function(self, X, return_array=False):
        return self.m_.predict_survival_function(X, return_array=return_array)


def make_models(quick, n_jobs, seed):
    ntree = 100 if quick else 300
    m = {
        "CoxPH": lambda: CoxPHSurvivalAnalysis(alpha=1.0, n_iter=200, ties="efron"),
        "CoxNet": lambda: TunedCoxnet(random_state=seed),
        "RSF": lambda: RandomSurvivalForest(n_estimators=ntree, min_samples_leaf=15, max_features="sqrt",
                                            n_jobs=n_jobs, random_state=seed),
        "ExtraSurvTrees": lambda: ExtraSurvivalTrees(n_estimators=ntree, min_samples_leaf=15, max_features="sqrt",
                                                     n_jobs=n_jobs, random_state=seed),
        "GBSA": lambda: GradientBoostingSurvivalAnalysis(n_estimators=100 if quick else 200, learning_rate=0.05,
                                                         max_depth=3, subsample=0.8, random_state=seed),
        "CW-GBSA": lambda: ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=300, learning_rate=0.1,
                                                                         subsample=0.8, random_state=seed),
        "SurvSVM": lambda: SignGuardSVM(alpha=1.0, random_state=seed),
    }
    if HAS_XGB:
        m["XGB-Cox"] = lambda: XGBCox(n_estimators=150 if quick else 300, random_state=seed, n_jobs=n_jobs)
    if quick:
        m = {k: v for k, v in m.items() if k in ["CoxPH", "CoxNet", "RSF", "XGB-Cox", "GBSA"]}
    return m


def surv_matrix(model, X, times):
    if hasattr(model, "surv_at"):
        return model.surv_at(X, times)
    if not hasattr(model, "predict_survival_function"):
        return None
    try:
        fns = model.predict_survival_function(X)
        out = np.empty((len(fns), len(times)))
        for i, f in enumerate(fns):
            tt = np.clip(times, f.x[0], f.x[-1])
            out[i] = f(tt)
        return out
    except Exception:
        return None


def youden_threshold(risk, y, horizon=HORIZON):
    lab, ok = binary_labels(y, horizon)
    if ok.sum() < 10 or len(np.unique(lab[ok])) < 2:
        return np.median(risk)
    fpr, tpr, thr = roc_curve(lab[ok], risk[ok])
    return thr[np.argmax(tpr - fpr)]


def binary_labels(y, horizon=HORIZON):
    ev, tm = y["event"].astype(bool), y["time"]
    lab = np.where(ev & (tm <= horizon), 1, 0)
    known = (ev & (tm <= horizon)) | (tm > horizon)
    return lab, known


def safe(fn, default=np.nan):
    try:
        return fn()
    except Exception:
        return default


def compute_metrics(y_tr, y_te, risk, S=None, grid=None, thr=None):
    r = {"n": len(y_te), "events": int(y_te["event"].sum())}
    if r["events"] < 3 or len(y_te) < 10:
        return r
    ev, tm = y_te["event"], y_te["time"]
    r["harrell_c"] = safe(lambda: concordance_index_censored(ev, tm, risk)[0])

    keep = tm < y_tr["time"].max()
    yk, rk = y_te[keep], risk[keep]
    tau = min(TAU, np.percentile(yk["time"], 90))
    r["uno_c"] = safe(lambda: concordance_index_ipcw(y_tr, yk, rk, tau=tau)[0])
    lo, hi = yk["time"].min(), yk["time"].max()
    for t in TD_TIMES:
        if lo < t < hi and ((yk["time"] <= t) & yk["event"]).sum() >= 3:
            r[f"auc_{int(t)}m"] = safe(lambda: cumulative_dynamic_auc(y_tr, yk, rk, [t])[0][0])
    if grid is not None:
        g = grid[(grid > lo) & (grid < hi)]
        if len(g) >= 3:
            r["mean_td_auc"] = safe(lambda: cumulative_dynamic_auc(y_tr, yk, rk, g)[1])
            if S is not None:
                Sk = S[keep][:, np.isin(grid, g)]
                r["ibs"] = safe(lambda: integrated_brier_score(y_tr, yk, Sk, g))
        if S is not None and lo < HORIZON < hi:
            j = int(np.argmin(np.abs(grid - HORIZON)))
            r["brier_5y"] = safe(lambda: brier_score(y_tr, yk, S[keep][:, j], [grid[j]])[1][0])

    lab, known = binary_labels(y_te)
    if known.sum() >= 10 and len(np.unique(lab[known])) == 2:
        L, R = lab[known], risk[known]
        r["n_5y_known"] = int(known.sum())
        r["auc_roc_5y_binary"] = safe(lambda: roc_auc_score(L, R))
        r["pr_auc_5y"] = safe(lambda: average_precision_score(L, R))
        pred = (R >= (thr if thr is not None else np.median(risk))).astype(int)
        tn, fp, fn, tp = confusion_matrix(L, pred, labels=[0, 1]).ravel()
        r.update(tp=tp, fp=fp, tn=tn, fn=fn,
                 accuracy=accuracy_score(L, pred), balanced_accuracy=balanced_accuracy_score(L, pred),
                 sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
                 ppv=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1),
                 f1=f1_score(L, pred, zero_division=0), mcc=matthews_corrcoef(L, pred),
                 kappa=cohen_kappa_score(L, pred))
        if S is not None and grid is not None:
            j = int(np.argmin(np.abs(grid - HORIZON)))
            r["brier_5y_binary"] = np.mean((1 - S[known, j] - L) ** 2)
    return r


METRIC_ORDER = ["harrell_c", "uno_c", "auc_36m", "auc_60m", "auc_120m", "mean_td_auc", "ibs", "brier_5y",
                "auc_roc_5y_binary", "pr_auc_5y", "accuracy", "balanced_accuracy", "sensitivity", "specificity",
                "ppv", "npv", "f1", "mcc", "kappa", "brier_5y_binary"]


def feature_sets(groups):
    clin = groups["clin_num"] + groups["clin_cat"]
    fs = {"NPI": ["nottingham_prognostic_index"] if "nottingham_prognostic_index" in groups["clin_num"] else [],
          "Clinical": clin, "Genes": groups["genes"], "Mutations": groups["muts"],
          "Clin+Genes": clin + groups["genes"], "Clin+Genes+Mut": clin + groups["genes"] + groups["muts"]}
    return {k: v for k, v in fs.items() if len(v) > 0}


def make_folds(df, n_splits, n_repeats, seed):
    lab = df[SUBTYPE_COL].fillna("NA").astype(str) + "_" + df["os_event"].astype(int).astype(str)\
        if SUBTYPE_COL in df.columns else df["os_event"].astype(int).astype(str)
    vc = lab.value_counts()
    lab = lab.where(~lab.isin(vc[vc < n_splits * 2].index), "rare_" + df["os_event"].astype(int).astype(str))
    out = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for k, (tr, te) in enumerate(skf.split(df, lab)):
            out.append((r, k, tr, te))
    return out


def fit_predict(model_fn, Xtr, ytr, Xte, grid):
    m = model_fn()
    t0 = time.time()
    m.fit(Xtr, ytr)
    fit_s = time.time() - t0
    risk_te, risk_tr = m.predict(Xte), m.predict(Xtr)
    S_te = surv_matrix(m, Xte, grid)
    return m, risk_te, risk_tr, S_te, fit_s


def run_cv(df, groups, subgroups, cfg, log):
    FS = feature_sets(groups)
    MODELS = make_models(cfg.quick, cfg.n_jobs, cfg.seed)
    spec_fs = [f for f in cfg.specific_feature_sets if f in FS]
    folds = make_folds(df, cfg.folds, cfg.repeats, cfg.seed)
    endpoints = {"OS": "os_event", "DSS": "dss_event"}
    rec, oof, coefs = [], [], []
    X_all = df
    total = len(folds)
    log(f"CV: {cfg.repeats} repeat(s) x {cfg.folds} folds | models: {list(MODELS)} | feature sets: {list(FS)} | "
        f"subtype-specific feature sets: {spec_fs} | subgroups: {list(subgroups)}")
    for fi, (r, k, tr_idx, te_idx) in enumerate(folds):
        tf = time.time()
        for ep, evcol in endpoints.items():
            valid = df[evcol].notna().values
            tr = tr_idx[valid[tr_idx]]; te = te_idx[valid[te_idx]]
            y = Surv.from_arrays(df[evcol].fillna(0).astype(bool).values, df["os_time"].values)
            ev_t = y["time"][tr][y["event"][tr]]
            grid = np.unique(np.percentile(ev_t, np.linspace(10, 80, 15)).round(2))


            for fs_name, feats in FS.items():
                try:
                    prep = FoldPreprocessor(feats, groups, cfg.n_genes).fit(X_all.iloc[tr], y[tr])
                    Xtr, Xte = prep.transform(X_all.iloc[tr]), prep.transform(X_all.iloc[te])
                except Exception as e:
                    log(f"  prep failed {ep}/{fs_name}: {e}"); continue
                for mname, mfn in MODELS.items():
                    if fs_name == "NPI" and mname != "CoxPH":
                        continue
                    try:
                        m, rte, rtr, Ste, fit_s = fit_predict(mfn, Xtr, y[tr], Xte, grid)
                    except Exception as e:
                        log(f"  fit failed global {ep}/{fs_name}/{mname}: {str(e)[:120]}"); continue
                    j5 = int(np.argmin(np.abs(grid - HORIZON)))
                    oof.append(pd.DataFrame({"repeat": r, "fold": k, "endpoint": ep, "strategy": "global",
                                             "trained_on": "All", "model": mname, "feature_set": fs_name,
                                             "idx": te, "risk": rte,
                                             "risk_pct": stats.rankdata(rte) / len(rte),
                                             "p_event_5y": (1 - Ste[:, j5]) if Ste is not None else np.nan}))
                    if mname == "CoxNet" and hasattr(m, "coef_"):
                        nz = [prep.names_[i] for i in np.flatnonzero(np.abs(m.coef_) > 1e-8)]
                        coefs.append({"repeat": r, "fold": k, "endpoint": ep, "strategy": "global", "subgroup": "All",
                                      "feature_set": fs_name, "selected": nz})
                    for sg, mask in subgroups.items():
                        mtr, mte = mask[tr], mask[te]
                        if mte.sum() < 10:
                            continue
                        thr = youden_threshold(rtr[mtr], y[tr][mtr])
                        met = compute_metrics(y[tr][mtr], y[te][mte], rte[mte],
                                              Ste[mte] if Ste is not None else None, grid, thr)
                        rec.append({"repeat": r, "fold": k, "endpoint": ep, "strategy": "global", "subgroup": sg,
                                    "model": mname, "feature_set": fs_name, "n_train": int(mtr.sum()),
                                    "n_train_total": len(tr), "fit_sec": fit_s, **met})


            for sg, mask in subgroups.items():
                if sg == "All":
                    continue
                str_, ste_ = tr[mask[tr]], te[mask[te]]
                if y["event"][str_].sum() < cfg.min_events or len(ste_) < 10:
                    continue
                for fs_name in spec_fs:
                    try:
                        prep = FoldPreprocessor(FS[fs_name], groups, cfg.n_genes_specific).fit(X_all.iloc[str_], y[str_])
                        Xtr, Xte = prep.transform(X_all.iloc[str_]), prep.transform(X_all.iloc[ste_])
                    except Exception as e:
                        log(f"  prep failed specific {ep}/{sg}/{fs_name}: {e}"); continue
                    for mname, mfn in MODELS.items():
                        try:
                            m, rte, rtr, Ste, fit_s = fit_predict(mfn, Xtr, y[str_], Xte, grid)
                        except Exception as e:
                            log(f"  fit failed specific {ep}/{sg}/{fs_name}/{mname}: {str(e)[:120]}"); continue
                        j5 = int(np.argmin(np.abs(grid - HORIZON)))
                        oof.append(pd.DataFrame({"repeat": r, "fold": k, "endpoint": ep, "strategy": "specific",
                                                 "trained_on": sg, "model": mname, "feature_set": fs_name,
                                                 "idx": ste_, "risk": rte,
                                                 "risk_pct": stats.rankdata(rte) / len(rte),
                                                 "p_event_5y": (1 - Ste[:, j5]) if Ste is not None else np.nan}))
                        if mname == "CoxNet" and hasattr(m, "coef_"):
                            nz = [prep.names_[i] for i in np.flatnonzero(np.abs(m.coef_) > 1e-8)]
                            coefs.append({"repeat": r, "fold": k, "endpoint": ep, "strategy": "specific",
                                          "subgroup": sg, "feature_set": fs_name, "selected": nz})
                        thr = youden_threshold(rtr, y[str_])
                        met = compute_metrics(y[str_], y[ste_], rte, Ste, grid, thr)
                        rec.append({"repeat": r, "fold": k, "endpoint": ep, "strategy": "specific", "subgroup": sg,
                                    "model": mname, "feature_set": fs_name, "n_train": len(str_),
                                    "n_train_total": len(str_), "fit_sec": fit_s, **met})
        log(f"Fold {fi + 1}/{total} (repeat {r + 1}, fold {k + 1}) done in {time.time() - tf:.0f}s")
    return pd.DataFrame(rec), pd.concat(oof, ignore_index=True), coefs


def summarize(met):
    keys = ["endpoint", "strategy", "subgroup", "model", "feature_set"]
    cols = [c for c in METRIC_ORDER if c in met.columns]
    rows = []
    for k, g in met.groupby(keys, sort=False):
        row = dict(zip(keys, k))
        row["n_folds"] = len(g); row["n_test_mean"] = g["n"].mean(); row["events_test_mean"] = g["events"].mean()
        for c in cols:
            v = g[c].astype(float)
            lo, hi = ci95(v)
            row[f"{c}_mean"], row[f"{c}_sd"], row[f"{c}_ci_low"], row[f"{c}_ci_high"] = v.mean(), v.std(ddof=1), lo, hi
        row["fit_sec_mean"] = g["fit_sec"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def pretty_table(summ, metrics):
    t = summ[["endpoint", "strategy", "subgroup", "model", "feature_set", "n_folds"]].copy()
    for m in metrics:
        if f"{m}_mean" in summ:
            t[m] = [f"{a:.3f} ± {b:.3f}" if pd.notna(a) else "" for a, b in zip(summ[f"{m}_mean"], summ[f"{m}_sd"])]
    return t


def specific_vs_global(met, spec_fs, subgroups_n, n_total):
    keys = ["repeat", "fold", "endpoint", "subgroup", "model", "feature_set"]
    g = met[(met.strategy == "global") & met.feature_set.isin(spec_fs) & (met.subgroup != "All")]
    s = met[met.strategy == "specific"]
    mm = g.merge(s, on=keys, suffixes=("_glob", "_spec"))
    rows = []
    for k, d in mm.groupby(["endpoint", "subgroup", "model", "feature_set"]):
        row = dict(zip(["endpoint", "subgroup", "model", "feature_set"], k))
        n_sg = subgroups_n.get(k[1], np.nan)
        for metric, better in [("uno_c", 1), ("harrell_c", 1), ("auc_60m", 1), ("mean_td_auc", 1), ("ibs", -1),
                               ("mcc", 1), ("balanced_accuracy", 1)]:
            a, b = f"{metric}_spec", f"{metric}_glob"
            if a not in d or b not in d:
                continue
            diff = (d[a] - d[b]).astype(float).values
            diff = diff[~np.isnan(diff)]
            if len(diff) < 2:
                continue
            nf = d["repeat"].nunique() and len(d) / d["repeat"].nunique()
            n_test = n_sg / nf if nf else np.nan
            t, p, (lo, hi) = corrected_ttest(diff, n_sg - n_test, n_test)
            try:
                pw = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            except Exception:
                pw = np.nan
            row.update({f"{metric}_global": d[b].mean(), f"{metric}_specific": d[a].mean(),
                        f"delta_{metric}": diff.mean(), f"delta_{metric}_ci_low": lo, f"delta_{metric}_ci_high": hi,
                        f"p_corrected_t_{metric}": p, f"p_wilcoxon_{metric}": pw,
                        f"specific_win_rate_{metric}": np.mean(better * diff > 0)})
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out) and "p_corrected_t_uno_c" in out:
        out["q_bh_uno_c"] = np.nan
        for ep, d in out.groupby("endpoint"):
            p = d["p_corrected_t_uno_c"].values
            ok = ~np.isnan(p)
            if ok.sum():
                q = np.full(len(p), np.nan)
                q[ok] = bh(p[ok])
                out.loc[d.index, "q_bh_uno_c"] = q
    return out


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = p[o] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1)
    return out


def km(time_, event):
    t, s, ci = kaplan_meier_estimator(event.astype(bool), time_, conf_type="log-log")
    return t, s, ci


def km_at(time_, event, t0):
    if len(time_) == 0:
        return np.nan
    t, s = kaplan_meier_estimator(event.astype(bool), time_)
    return s[t <= t0][-1] if np.any(t <= t0) else 1.0


def logrank(time_, event, groups):
    try:
        y = Surv.from_arrays(event.astype(bool), time_)
        chi, p = compare_survival(y, np.asarray(groups))
        return chi, p
    except Exception:
        return np.nan, np.nan


def cox_table(df_, time_col, event_col, covars, label):
    if not HAS_SM:
        return pd.DataFrame()
    d = df_[[time_col, event_col] + covars].dropna()
    X = pd.get_dummies(d[covars], drop_first=True, dtype=float)
    X = X.loc[:, X.std() > 0]
    try:
        res = PHReg(d[time_col].values, X.values, status=d[event_col].values.astype(int), ties="efron").fit()
    except Exception:
        return pd.DataFrame()
    se = np.sqrt(np.diag(res.cov_params()))
    return pd.DataFrame({"model": label, "variable": X.columns, "HR": np.exp(res.params),
                         "HR_ci_low": np.exp(res.params - 1.96 * se), "HR_ci_high": np.exp(res.params + 1.96 * se),
                         "p_value": res.pvalues, "n": len(d), "events": int(d[event_col].sum())})


def aalen_johansen(time_, cause):
    o = np.argsort(time_); t, c = time_[o], cause[o]
    ut = np.unique(t[c > 0]); S, cif1, cif2 = 1.0, [0.0], [0.0]; n = len(t)
    for u in ut:
        nrisk = (t >= u).sum(); d1 = ((t == u) & (c == 1)).sum(); d2 = ((t == u) & (c == 2)).sum()
        cif1.append(cif1[-1] + S * d1 / nrisk); cif2.append(cif2[-1] + S * d2 / nrisk)
        S *= 1 - (d1 + d2) / nrisk
    return np.r_[0, ut], np.array(cif1), np.array(cif2)


def table1(df, groups, out):
    by = SUBTYPE_COL
    d = df[df[by].notna()]
    levels = [l for l in ["LumA", "LumB", "Her2", "Basal", "claudin-low", "Normal"] if l in d[by].unique()]
    rows = []
    rows.append({"variable": "N", **{l: str((d[by] == l).sum()) for l in levels}, "Overall": str(len(df)), "p_value": ""})
    for c in groups["clin_num"] + ["os_time"]:
        vals = {l: d.loc[d[by] == l, c].dropna() for l in levels}
        f = lambda v: f"{v.median():.1f} [{v.quantile(.25):.1f}-{v.quantile(.75):.1f}]" if len(v) else ""
        p = safe(lambda: stats.kruskal(*[v for v in vals.values() if len(v) > 1]).pvalue)
        rows.append({"variable": f"{c}, median [IQR]", **{l: f(v) for l, v in vals.items()},
                     "Overall": f(df[c].dropna()), "p_value": f"{p:.3g}" if pd.notna(p) else ""})
    for c in [x for x in groups["clin_cat"] if x != by] + ["os_event", "dss_event"]:
        ct = pd.crosstab(d[c], d[by])
        p = safe(lambda: stats.chi2_contingency(ct)[1]) if ct.shape[0] > 1 else np.nan
        rows.append({"variable": c, "p_value": f"{p:.3g}" if pd.notna(p) else ""})
        for lvl in ct.index[:12]:
            r = {"variable": f"   {lvl}, n (%)"}
            for l in levels:
                n = ct.loc[lvl, l] if l in ct.columns else 0
                r[l] = f"{n} ({100 * n / max((d[by] == l).sum(), 1):.1f})"
            n_all = (df[c] == lvl).sum(); r["Overall"] = f"{n_all} ({100 * n_all / len(df):.1f})"; r["p_value"] = ""
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "tables", "T01_baseline_characteristics_by_PAM50.csv"), index=False)
    return t


def descriptive_figures(df, groups, subgroups, out, log):

    rows = []
    for sg, m in subgroups.items():
        d = df[m]
        rows.append({"subgroup": sg, "n": int(m.sum()), "os_events": int(d.os_event.sum()),
                     "dss_events": int(d.dss_event.sum()), "other_cause_deaths": int((d.cause == 2).sum()),
                     "median_follow_up_months": float(np.median(d.os_time)),
                     "os_5y_KM": km_at(d.os_time.values, d.os_event.values, 60),
                     "dss_5y_KM": km_at(d.os_time.values, d.dss_event.fillna(0).values, 60),
                     "os_10y_KM": km_at(d.os_time.values, d.os_event.values, 120),
                     "dss_10y_KM": km_at(d.os_time.values, d.dss_event.fillna(0).values, 120)})
    sgt = pd.DataFrame(rows)
    sgt.to_csv(os.path.join(out, "tables", "T02_subgroups_events_survival.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(sgt))
    ax.bar(x - 0.2, sgt.n, 0.4, label="patients", color="#9ecae1"); ax.bar(x + 0.2, sgt.dss_events, 0.4, label="cancer deaths", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(sgt.subgroup, rotation=30, ha="right"); ax.legend(); ax.set_title("Analysed subgroups")
    savefig(fig, out, "F01_subgroup_sizes")


    cols = groups["clin_num"] + groups["clin_cat"]
    miss = df[cols].isna().mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(miss) + 1))
    ax.barh(miss.index, 100 * miss.values, color="#7f7f7f"); ax.set_xlabel("% missing"); ax.set_title("Missing clinical data")
    savefig(fig, out, "F02_missingness")
    miss.rename("fraction_missing").to_csv(os.path.join(out, "tables", "T03_missingness.csv"))


    lr_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for i, (ep, col) in enumerate([("Overall survival", "os_event"), ("Disease-specific survival", "dss_event")]):
        for j, (gname, gser) in enumerate([("PAM50 + claudin-low", df[SUBTYPE_COL]),
                                           ("Receptor status", np.where(subgroups.get("TNBC", np.zeros(len(df), bool)), "TNBC", "non-TNBC"))]):
            ax = axes[i, j]; gs = pd.Series(gser, index=df.index)
            ok = gs.notna() & df[col].notna()
            for c_i, lvl in enumerate(sorted(gs[ok].unique())):
                m = ok & (gs == lvl)
                if m.sum() < 10:
                    continue
                t, s, ci = km(df.os_time[m].values, df[col][m].values)
                ax.step(t / 12, s, where="post", color=PALETTE[c_i % 10], label=f"{lvl} (n={m.sum()})")
                ax.fill_between(t / 12, ci[0], ci[1], step="post", alpha=0.12, color=PALETTE[c_i % 10])
            chi, p = logrank(df.os_time[ok].values, df[col][ok].values, gs[ok].values)
            lr_rows.append({"endpoint": ep, "grouping": gname, "chi2": chi, "p_value": p})
            ax.set_title(f"{ep} by {gname}\nlog-rank p = {p:.2e}"); ax.set_xlabel("Years"); ax.set_ylabel("Survival probability")
            ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
    savefig(fig, out, "F03_KM_by_subtype_OS_DSS")
    pd.DataFrame(lr_rows).to_csv(os.path.join(out, "tables", "T04_logrank_by_subtype.csv"), index=False)


    names = [s for s in subgroups if s != "All"]
    nc = 4; nr = int(np.ceil((len(names) + 1) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3.3 * nr), squeeze=False)
    cif_rows = []
    for a, sg in zip(axes.flat, ["All"] + names):
        d = df[subgroups[sg] & df.cause.notna()]
        t, c1, c2 = aalen_johansen(d.os_time.values, d.cause.values)
        a.step(t / 12, c1, where="post", color="#d62728", label="Breast-cancer death")
        a.step(t / 12, c2, where="post", color="#7f7f7f", label="Other-cause death")
        a.set_title(f"{sg} (n={len(d)})", fontsize=9); a.set_ylim(0, 0.8); a.set_xlabel("Years")
        for yr in [5, 10]:
            k = np.searchsorted(t, yr * 12, side="right") - 1
            cif_rows.append({"subgroup": sg, "year": yr, "CIF_cancer_death": c1[k], "CIF_other_death": c2[k]})
    axes.flat[0].legend(fontsize=8)
    for a in list(axes.flat)[len(names) + 1:]:
        a.axis("off")
    fig.suptitle("Competing risks: cumulative incidence (Aalen-Johansen)")
    savefig(fig, out, "F04_competing_risks_CIF")
    pd.DataFrame(cif_rows).to_csv(os.path.join(out, "tables", "T05_competing_risks_CIF.csv"), index=False)
    return sgt


def fig_benchmark_heatmaps(summ, out):
    for ep in summ.endpoint.unique():
        d = summ[(summ.endpoint == ep) & (summ.strategy == "global") & (summ.subgroup == "All")]
        for metric in ["uno_c", "harrell_c", "ibs", "mean_td_auc"]:
            col = f"{metric}_mean"
            if col not in d or d[col].isna().all():
                continue
            pv = d.pivot_table(index="model", columns="feature_set", values=col)
            order = [c for c in ["NPI", "Clinical", "Genes", "Mutations", "Clin+Genes", "Clin+Genes+Mut"] if c in pv.columns]
            pv = pv[order]
            fig, ax = plt.subplots(figsize=(1.3 * len(order) + 2, 0.5 * len(pv) + 1.5))
            cmap = "viridis_r" if metric == "ibs" else "viridis"
            im = ax.imshow(pv.values, cmap=cmap, aspect="auto")
            ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=30, ha="right")
            ax.set_yticks(range(len(pv))); ax.set_yticklabels(pv.index)
            for i in range(pv.shape[0]):
                for j in range(pv.shape[1]):
                    v = pv.values[i, j]
                    if pd.notna(v):
                        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                                color="white" if (v < np.nanmean(pv.values)) ^ (metric == "ibs") else "black")
            plt.colorbar(im, ax=ax, fraction=0.04)
            ax.set_title(f"{ep}: {metric} (mean over CV folds), whole cohort")
            savefig(fig, out, f"F05_benchmark_heatmap_{ep}_{metric}")


def fig_model_bars(summ, out):
    for ep in summ.endpoint.unique():
        d = summ[(summ.endpoint == ep) & (summ.strategy == "global") & (summ.subgroup == "All")].copy()
        metrics = [m for m in ["harrell_c", "uno_c", "auc_60m", "ibs", "auc_roc_5y_binary", "mcc"] if f"{m}_mean" in d]
        fss = [f for f in ["Clinical", "Genes", "Clin+Genes", "Clin+Genes+Mut"] if f in d.feature_set.unique()]
        models = [m for m in d.model.unique()]
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.5), squeeze=False)
        for a, met in zip(axes.flat, metrics):
            w = 0.8 / max(len(fss), 1)
            for i, fs in enumerate(fss):
                dd = d[d.feature_set == fs].set_index("model").reindex(models)
                a.bar(np.arange(len(models)) + i * w, dd[f"{met}_mean"], w,
                      yerr=[dd[f"{met}_mean"] - dd[f"{met}_ci_low"], dd[f"{met}_ci_high"] - dd[f"{met}_mean"]],
                      capsize=2, label=fs, color=PALETTE[i])
            npi = d[d.feature_set == "NPI"]
            if len(npi) and pd.notna(npi[f"{met}_mean"].iloc[0]):
                a.axhline(npi[f"{met}_mean"].iloc[0], ls="--", color="k", lw=1, label="NPI (CoxPH)")
            a.set_xticks(np.arange(len(models)) + 0.4 - w / 2); a.set_xticklabels(models, rotation=45, ha="right")
            a.set_title(met)
            lo = np.nanmin(d[f"{met}_ci_low"]) if met != "mcc" else np.nanmin(d[f"{met}_ci_low"])
            if met in ["harrell_c", "uno_c", "auc_60m", "auc_roc_5y_binary"]:
                a.set_ylim(max(0.4, lo - 0.03), min(1, np.nanmax(d[f"{met}_ci_high"]) + 0.03))
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"{ep}: model x feature-set benchmark (mean, 95% CI across folds)")
        savefig(fig, out, f"F06_model_benchmark_bars_{ep}")


def fig_specific_vs_global(cmp, out):
    if not len(cmp) or "delta_uno_c" not in cmp:
        return
    for ep in cmp.endpoint.unique():
        for fs in cmp.feature_set.unique():
            d = cmp[(cmp.endpoint == ep) & (cmp.feature_set == fs)]
            if not len(d):
                continue
            pv = d.pivot_table(index="subgroup", columns="model", values="delta_uno_c")
            pp = d.pivot_table(index="subgroup", columns="model", values="p_corrected_t_uno_c").reindex_like(pv)
            fig, ax = plt.subplots(figsize=(1.1 * pv.shape[1] + 3, 0.55 * pv.shape[0] + 1.8))
            lim = np.nanmax(np.abs(pv.values)) if np.isfinite(np.nanmax(np.abs(pv.values))) else 0.05
            im = ax.imshow(pv.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            for i in range(pv.shape[0]):
                for j in range(pv.shape[1]):
                    v, p = pv.values[i, j], pp.values[i, j]
                    if pd.notna(v):
                        star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                        ax.text(j, i, f"{v:+.3f}{star}", ha="center", va="center", fontsize=8)
            ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns, rotation=40, ha="right")
            ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index)
            plt.colorbar(im, ax=ax, fraction=0.04, label="Δ Uno C (specific − global)")
            ax.set_title(f"{ep} | {fs}: subtype-specific minus global model\n(* p<0.05, ** p<0.01, corrected resampled t-test)")
            savefig(fig, out, f"F07_specific_vs_global_heatmap_{ep}_{fs}")


            rows = []
            for sg, dd in d.groupby("subgroup"):
                b = dd.loc[(dd[["uno_c_global", "uno_c_specific"]].max(axis=1)).idxmax()]
                rows.append(b)
            fr = pd.DataFrame(rows).sort_values("delta_uno_c")
            fig, ax = plt.subplots(figsize=(7, 0.5 * len(fr) + 1.5))
            yy = np.arange(len(fr))
            ax.errorbar(fr.delta_uno_c, yy, xerr=[fr.delta_uno_c - fr.delta_uno_c_ci_low, fr.delta_uno_c_ci_high - fr.delta_uno_c],
                        fmt="s", color="k", capsize=3)
            ax.axvline(0, color="grey", ls="--")
            ax.set_yticks(yy); ax.set_yticklabels([f"{s} [{m}]" for s, m in zip(fr.subgroup, fr.model)])
            for y_, (dl, p) in enumerate(zip(fr.delta_uno_c, fr.p_corrected_t_uno_c)):
                ax.text(ax.get_xlim()[1], y_, f"  p={p:.3f}", va="center", fontsize=8)
            ax.set_xlabel("Δ Uno C-index (subtype-specific − global), 95% CI")
            ax.set_title(f"{ep} | {fs}: does training within the subtype help?")
            savefig(fig, out, f"F08_specific_vs_global_forest_{ep}_{fs}")


def fig_subgroup_performance(summ, out):
    for ep in summ.endpoint.unique():
        d = summ[(summ.endpoint == ep) & (summ.feature_set != "NPI")]
        best = d.loc[d.groupby(["subgroup", "strategy"])["uno_c_mean"].idxmax().dropna()]
        npi = summ[(summ.endpoint == ep) & (summ.feature_set == "NPI")].set_index("subgroup")
        sgs = list(dict.fromkeys(best.subgroup))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(sgs)); w = 0.27
        for i, stg in enumerate(["global", "specific"]):
            dd = best[best.strategy == stg].set_index("subgroup").reindex(sgs)
            ax.bar(x + (i - 0.5) * w, dd.uno_c_mean, w, yerr=[dd.uno_c_mean - dd.uno_c_ci_low, dd.uno_c_ci_high - dd.uno_c_mean],
                   capsize=2, color=PALETTE[i], label=f"best {stg} model")
        nn = npi.reindex(sgs)
        ax.bar(x + 1.5 * w, nn.uno_c_mean, w, color="#bbbbbb", label="NPI alone")
        ax.set_xticks(x + 0.25 * w); ax.set_xticklabels(sgs, rotation=30, ha="right"); ax.set_ylim(0.45, None)
        ax.axhline(0.5, color="k", lw=0.8); ax.set_ylabel("Uno C-index"); ax.legend(fontsize=8)
        ax.set_title(f"{ep}: discrimination by subgroup (best configuration per strategy)")
        savefig(fig, out, f"F09_subgroup_best_models_{ep}")


def oof_agg(oof, ep, strategy, trained_on, model, fs):
    d = oof[(oof.endpoint == ep) & (oof.strategy == strategy) & (oof.trained_on == trained_on) &
            (oof.model == model) & (oof.feature_set == fs)]
    return d.groupby("idx").agg(risk=("risk_pct", "mean"), p5y=("p_event_5y", "mean"))


def pick_best(summ, ep, subgroup="All", strategy="global", need_sf=False, fs=None):
    d = summ[(summ.endpoint == ep) & (summ.subgroup == subgroup) & (summ.feature_set != "NPI")]
    if strategy:
        d = d[d.strategy == strategy]
    if fs:
        d = d[d.feature_set == fs]
    if need_sf:
        d = d[d["ibs_mean"].notna()] if "ibs_mean" in d else d.iloc[0:0]
    d = d.dropna(subset=["uno_c_mean"])
    if not len(d):
        return None
    b = d.loc[d.uno_c_mean.idxmax()]
    return dict(strategy=b.strategy, trained_on="All" if b.strategy == "global" else subgroup, model=b.model,
                feature_set=b.feature_set, label=f"{b.model} [{b.feature_set}]" + ("" if b.strategy == "global" else " (subtype-specific)"))


def boot_c(y, risk, ref=None, B=500, seed=0):
    rng = np.random.default_rng(seed); n = len(y); cs, ds = [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y["event"][i].sum() < 3:
            continue
        c = concordance_index_censored(y["event"][i], y["time"][i], risk[i])[0]; cs.append(c)
        if ref is not None:
            ds.append(c - concordance_index_censored(y["event"][i], y["time"][i], ref[i])[0])
    out = {"c_boot_ci_low": np.percentile(cs, 2.5), "c_boot_ci_high": np.percentile(cs, 97.5)}
    if ref is not None:
        ds = np.array(ds)
        out.update(delta_vs_npi=np.mean(ds), delta_ci_low=np.percentile(ds, 2.5), delta_ci_high=np.percentile(ds, 97.5),
                   p_delta_le_0=np.mean(ds <= 0))
    return out


def dca(p, t_, e_, thresholds, horizon=HORIZON):
    n = len(p); rows = []
    s_all = km_at(t_, e_, horizon)
    for pt in thresholds:
        tr = p >= pt; x = tr.mean()
        if tr.sum() > 0:
            s = km_at(t_[tr], e_[tr], horizon)
            nb = (1 - s) * x - s * x * pt / (1 - pt)
        else:
            nb = 0.0
        rows.append({"threshold": pt, "net_benefit_model": nb,
                     "net_benefit_treat_all": (1 - s_all) - s_all * pt / (1 - pt), "net_benefit_treat_none": 0.0})
    return pd.DataFrame(rows)


def oof_analyses(df, summ, oof, subgroups, out, log, cfg):
    res = {}
    for ep, evcol in [("OS", "os_event"), ("DSS", "dss_event")]:
        valid = df[evcol].notna().values
        y_all = Surv.from_arrays(df[evcol].fillna(0).astype(bool).values, df["os_time"].values)
        best = pick_best(summ, ep)
        if best is None:
            continue
        best_sf = pick_best(summ, ep, need_sf=True) or best
        configs = {"Best: " + best["label"]: best,
                   "Clinical CoxPH": dict(strategy="global", trained_on="All", model="CoxPH", feature_set="Clinical"),
                   "NPI alone": dict(strategy="global", trained_on="All", model="CoxPH", feature_set="NPI")}
        bg = pick_best(summ, ep, fs="Genes")
        if bg:
            configs["Best genes-only: " + bg["label"]] = bg
        if best_sf["label"] != best["label"]:
            configs["Best calibrated-capable: " + best_sf["label"]] = best_sf
        agg = {k: oof_agg(oof, ep, v["strategy"], v["trained_on"], v["model"], v["feature_set"]) for k, v in configs.items()}
        agg = {k: v for k, v in agg.items() if len(v)}
        res[ep] = dict(best=best, best_sf=best_sf)


        rows = []
        npi_a = agg.get("NPI alone")
        for k, a in agg.items():
            idx = a.index.values[valid[a.index.values]]
            yy, rr = y_all[idx], a.loc[idx, "risk"].values
            ref = npi_a.reindex(idx)["risk"].values if npi_a is not None and k != "NPI alone" else None
            if ref is not None and np.isnan(ref).any():
                ref = None
            row = {"endpoint": ep, "configuration": k, "n": len(idx),
                   "harrell_c_pooled_oof": concordance_index_censored(yy["event"], yy["time"], rr)[0]}
            row.update(boot_c(yy, rr, ref, B=cfg.n_boot, seed=cfg.seed))
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(out, "tables", f"T10_pooled_OOF_cindex_bootstrap_{ep}.csv"), index=False)


        fig, ax = plt.subplots(figsize=(7, 4.5)); tdrows = []
        times = np.arange(12, 181, 6.0)
        for i, (k, a) in enumerate(agg.items()):
            idx = a.index.values[valid[a.index.values]]
            yy = y_all[idx]; tt = times[(times > yy["time"].min()) & (times < np.percentile(yy["time"], 95))]
            try:
                aucs, mean_auc = cumulative_dynamic_auc(y_all[valid], yy, a.loc[idx, "risk"].values, tt)
            except Exception:
                continue
            ax.plot(tt / 12, aucs, marker="o", ms=3, color=PALETTE[i], label=f"{k} (mean {mean_auc:.3f})")
            tdrows += [{"endpoint": ep, "configuration": k, "months": t_, "td_auc": a_} for t_, a_ in zip(tt, aucs)]
        ax.set_xlabel("Years since diagnosis"); ax.set_ylabel("Cumulative/dynamic AUC"); ax.legend(fontsize=7)
        ax.set_title(f"{ep}: time-dependent AUC (pooled out-of-fold predictions)")
        savefig(fig, out, f"F10_time_dependent_AUC_{ep}")
        pd.DataFrame(tdrows).to_csv(os.path.join(out, "tables", f"T11_time_dependent_auc_{ep}.csv"), index=False)


        lab, known = binary_labels(y_all)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
        for i, (k, a) in enumerate(agg.items()):
            idx = a.index.values[valid[a.index.values] & known[a.index.values]]
            L, R = lab[idx], a.loc[idx, "risk"].values
            fpr, tpr, _ = roc_curve(L, R); pr, rc, _ = precision_recall_curve(L, R)
            a1.plot(fpr, tpr, color=PALETTE[i], label=f"{k} AUC={roc_auc_score(L, R):.3f}")
            a2.plot(rc, pr, color=PALETTE[i], label=f"{k} AP={average_precision_score(L, R):.3f}")
        a1.plot([0, 1], [0, 1], "k--", lw=0.8); a1.set_xlabel("1 − specificity"); a1.set_ylabel("Sensitivity"); a1.legend(fontsize=6)
        a2.set_xlabel("Recall"); a2.set_ylabel("Precision"); a2.legend(fontsize=6)
        fig.suptitle(f"{ep}: 5-year death prediction (patients censored before 5 y excluded)")
        savefig(fig, out, f"F11_ROC_PR_5y_{ep}")
        a = agg[[k for k in agg if k.startswith("Best:")][0]]
        idx = a.index.values[valid[a.index.values] & known[a.index.values]]
        L, R = lab[idx], a.loc[idx, "risk"].values
        fpr, tpr, thr = roc_curve(L, R); th = thr[np.argmax(tpr - fpr)]
        cm = confusion_matrix(L, (R >= th).astype(int))
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({100 * cm[i, j] / cm.sum():.1f}%)", ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred alive", "pred death"]); ax.set_yticks([0, 1])
        ax.set_yticklabels(["alive at 5y", "died ≤5y"]); ax.set_title(f"{ep}: {best['label']}\n(Youden threshold)", fontsize=9)
        savefig(fig, out, f"F12_confusion_matrix_5y_{ep}")


        sgs = list(subgroups); nc = 4; nr = int(np.ceil(len(sgs) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(4.4 * nc, 4.3 * nr), squeeze=False); rg_rows = []
        for axx, sg in zip(axes.flat, sgs):
            b = pick_best(summ, ep, subgroup=sg, strategy=None)
            if b is None:
                axx.axis("off"); continue
            a = oof_agg(oof, ep, b["strategy"], b["trained_on"], b["model"], b["feature_set"])
            idx = a.index.values[subgroups[sg][a.index.values] & valid[a.index.values]]
            if len(idx) < 30:
                axx.axis("off"); continue
            r = a.loc[idx, "risk"].values
            grp = pd.qcut(stats.rankdata(r), 3, labels=["Low", "Intermediate", "High"]).astype(str)
            tt, ee = df.os_time.values[idx], df[evcol].values[idx]
            for i, gname in enumerate(["Low", "Intermediate", "High"]):
                m = grp == gname
                t, s, _ = km(tt[m], ee[m])
                axx.step(t / 12, s, where="post", color=["#2ca02c", "#ff7f0e", "#d62728"][i], label=f"{gname} (n={m.sum()})")
            chi, p = logrank(tt, ee, grp)
            dd = pd.DataFrame({"t": tt, "e": ee, "high_vs_low": np.where(grp == "High", 1, np.where(grp == "Low", 0, np.nan))})
            ct = cox_table(dd.dropna(), "t", "e", ["high_vs_low"], "high vs low")
            hr = ct.iloc[0] if len(ct) else None
            rg_rows.append({"endpoint": ep, "subgroup": sg, "configuration": b["label"], "n": len(idx), "logrank_chi2": chi,
                            "logrank_p": p, "HR_high_vs_low": hr.HR if hr is not None else np.nan,
                            "HR_ci_low": hr.HR_ci_low if hr is not None else np.nan,
                            "HR_ci_high": hr.HR_ci_high if hr is not None else np.nan,
                            **{f"surv_5y_{g}": km_at(tt[grp == g], ee[grp == g], 60) for g in ["Low", "Intermediate", "High"]}})
            axx.set_title(f"{sg}\n{b['label']}\nlog-rank p={p:.1e}" + (f", HR={hr.HR:.2f}" if hr is not None else ""), fontsize=7.5)
            axx.set_ylim(0, 1.02); axx.set_xlabel("Years"); axx.legend(fontsize=6.5)
        for axx in list(axes.flat)[len(sgs):]:
            axx.axis("off")
        fig.suptitle(f"{ep}: out-of-fold risk tertiles within each subgroup", y=1.01); fig.tight_layout()
        savefig(fig, out, f"F13_KM_risk_groups_by_subgroup_{ep}")
        pd.DataFrame(rg_rows).to_csv(os.path.join(out, "tables", f"T12_risk_groups_logrank_HR_{ep}.csv"), index=False)


        sfk = [k for k in agg if agg[k]["p5y"].notna().all()]
        cal_rows, dca_rows = [], []
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
        for i, k in enumerate(sfk):
            a = agg[k]; idx = a.index.values[valid[a.index.values]]
            p = a.loc[idx, "p5y"].values; tt, ee = df.os_time.values[idx], df[evcol].values[idx]
            bins = pd.qcut(p, 10, labels=False, duplicates="drop")
            pr, ob = [], []
            for b_ in np.unique(bins):
                m = bins == b_
                pr.append(p[m].mean()); ob.append(1 - km_at(tt[m], ee[m], HORIZON))
                cal_rows.append({"endpoint": ep, "configuration": k, "decile": int(b_), "n": int(m.sum()),
                                 "predicted_5y_risk": pr[-1], "observed_5y_risk_KM": ob[-1]})
            pr, ob = np.array(pr), np.array(ob)
            slope, intercept = np.polyfit(pr, ob, 1) if len(pr) > 2 else (np.nan, np.nan)
            ece = np.mean(np.abs(pr - ob))
            a1.plot(pr, ob, "o-", color=PALETTE[i], label=f"{k}\nslope={slope:.2f}, ECE={ece:.3f}")
            th = np.linspace(0.02, 0.8, 40)
            d_ = dca(p, tt, ee, th); d_["endpoint"] = ep; d_["configuration"] = k; dca_rows.append(d_)
            a2.plot(th, d_.net_benefit_model, color=PALETTE[i], label=k)
        if dca_rows:
            a2.plot(th, dca_rows[0].net_benefit_treat_all, "k--", lw=1, label="Treat all")
            a2.axhline(0, color="grey", lw=1, label="Treat none")
            a2.set_ylim(-0.05, max(0.05, np.nanmax(dca_rows[0].net_benefit_treat_all) + 0.05))
        a1.plot([0, 1], [0, 1], "k:", lw=1); a1.set_xlabel("Predicted 5-year risk"); a1.set_ylabel("Observed (1 − KM)")
        a1.set_title("Calibration at 5 years (deciles)"); a1.legend(fontsize=6)
        a2.set_xlabel("Threshold probability"); a2.set_ylabel("Net benefit"); a2.set_title("Decision curve analysis (5 y)"); a2.legend(fontsize=6)
        fig.suptitle(ep)
        savefig(fig, out, f"F14_calibration_DCA_5y_{ep}")
        pd.DataFrame(cal_rows).to_csv(os.path.join(out, "tables", f"T13_calibration_5y_{ep}.csv"), index=False)
        if dca_rows:
            pd.concat(dca_rows).to_csv(os.path.join(out, "tables", f"T14_decision_curve_{ep}.csv"), index=False)


        a = agg[[k for k in agg if k.startswith("Best:")][0]]
        dd = df.loc[a.index].copy(); dd["ML_risk_score_per_SD"] = (a["risk"] - a["risk"].mean()) / a["risk"].std()
        dd = dd[dd[evcol].notna()]
        tabs = []
        adj = [c for c in ["age_at_diagnosis", "nottingham_prognostic_index", "er_status", "her2_status"] if c in dd]
        tabs.append(cox_table(dd, "os_time", evcol, ["ML_risk_score_per_SD"], "univariable | All"))
        tabs.append(cox_table(dd, "os_time", evcol, ["ML_risk_score_per_SD"] + adj + [SUBTYPE_COL], "multivariable | All"))
        tabs.append(cox_table(dd, "os_time", evcol, adj + [SUBTYPE_COL], "clinical-only reference | All"))
        for sg in [s for s in subgroups if s not in ("All", "LumA")]:
            m = subgroups[sg][dd.index.values]
            tabs.append(cox_table(dd[m], "os_time", evcol, ["ML_risk_score_per_SD", "age_at_diagnosis", "nottingham_prognostic_index"],
                                  f"multivariable | {sg}"))
        ct = pd.concat([t for t in tabs if len(t)], ignore_index=True) if any(len(t) for t in tabs) else pd.DataFrame()
        if len(ct):
            ct.insert(0, "endpoint", ep)
            ct.to_csv(os.path.join(out, "tables", f"T15_cox_independent_value_{ep}.csv"), index=False)
            f = ct[ct.model == "multivariable | All"]
            fig, ax = plt.subplots(figsize=(7, 0.4 * len(f) + 1.5))
            yy = np.arange(len(f))[::-1]
            ax.errorbar(f.HR, yy, xerr=[f.HR - f.HR_ci_low, f.HR_ci_high - f.HR], fmt="s", color="k", capsize=3)
            ax.axvline(1, color="grey", ls="--"); ax.set_xscale("log")
            ax.set_yticks(yy); ax.set_yticklabels([f"{v}  (p={p:.1e})" for v, p in zip(f.variable, f.p_value)], fontsize=8)
            ax.set_xlabel("Hazard ratio (95% CI)"); ax.set_title(f"{ep}: multivariable Cox – ML risk score adjusted for clinical factors")
            savefig(fig, out, f"F15_forest_multivariable_cox_{ep}")
    return res


def shap_analysis(df, groups, subgroups, out, log, cfg):
    if not (HAS_SHAP and HAS_XGB):
        log("SHAP/XGBoost not available – skipping interpretability."); return {}
    FS = feature_sets(groups)
    fs_name = "Clin+Genes+Mut" if "Clin+Genes+Mut" in FS else "Clin+Genes"
    tops, rows = {}, []
    for ep, evcol in [("OS", "os_event"), ("DSS", "dss_event")]:
        valid = df[evcol].notna().values
        y = Surv.from_arrays(df[evcol].fillna(0).astype(bool).values, df["os_time"].values)
        targets = [("Global model | All", subgroups["All"], cfg.n_genes)] +\
                  [(f"Specific | {sg}", m, cfg.n_genes_specific) for sg, m in subgroups.items() if sg != "All"]
        nplot = len(targets); nc = 4; nr = int(np.ceil(nplot / nc))
        figb, axb = plt.subplots(nr, nc, figsize=(4.6 * nc, 4.2 * nr), squeeze=False)
        for pi, (name, mask, ng) in enumerate(targets):
            idx = np.flatnonzero(mask & valid)
            if y["event"][idx].sum() < cfg.min_events:
                axb.flat[pi].axis("off"); continue
            try:
                prep = FoldPreprocessor(FS[fs_name], groups, ng).fit(df.iloc[idx], y[idx])
                X = prep.transform(df.iloc[idx])
                m = XGBCox(n_estimators=300, random_state=cfg.seed, n_jobs=1).fit(X, y[idx])
                sv = shap.TreeExplainer(m.model_).shap_values(X)
            except Exception as e:
                log(f"  SHAP failed for {name}: {e}"); continue
            imp = pd.Series(np.abs(sv).mean(0), index=prep.names_).sort_values(ascending=False)
            tops[(ep, name)] = list(imp.index[:20])
            for rank, (f, v) in enumerate(imp.head(30).items(), 1):
                corr = np.corrcoef(X[:, prep.names_.index(f)], sv[:, prep.names_.index(f)])[0, 1] if np.std(X[:, prep.names_.index(f)]) > 0 else np.nan
                rows.append({"endpoint": ep, "model": name, "rank": rank, "feature": f, "mean_abs_shap": v,
                             "direction_(corr_value_vs_shap)": corr,
                             "feature_type": "gene" if f in groups["genes"] else ("mutation" if f in groups["muts"] else "clinical")})
            a = axb.flat[pi]; top = imp.head(12)[::-1]
            a.barh(top.index, top.values, color=["#d62728" if f in groups["genes"] else ("#9467bd" if f in groups["muts"] else "#1f77b4") for f in top.index])
            a.set_title(f"{name} (n={len(idx)})", fontsize=8.5); a.tick_params(labelsize=7)
            if name.startswith("Global") or any(s in name for s in ["TNBC", "HER2", "Claudin"]):
                plt.figure()
                shap.summary_plot(sv, X, feature_names=prep.names_, max_display=20, show=False)
                f = plt.gcf(); f.suptitle(f"{ep}: SHAP – {name}", fontsize=10)
                savefig(f, out, f"F17_SHAP_beeswarm_{ep}_{re.sub(r'[^A-Za-z0-9]+', '_', name)}")
        for a in list(axb.flat)[nplot:]:
            a.axis("off")
        figb.suptitle(f"{ep}: top SHAP features (red = gene, purple = mutation, blue = clinical)", y=1.0)
        figb.tight_layout(); savefig(figb, out, f"F16_SHAP_top_features_by_subgroup_{ep}")


        names = [n for (e, n) in tops if e == ep]
        if len(names) > 2:
            J = np.array([[len(set(tops[(ep, a)]) & set(tops[(ep, b)])) / len(set(tops[(ep, a)]) | set(tops[(ep, b)]))
                           for b in names] for a in names])
            fig, ax = plt.subplots(figsize=(0.7 * len(names) + 3, 0.6 * len(names) + 2))
            im = ax.imshow(J, cmap="magma", vmin=0, vmax=1)
            for i in range(len(names)):
                for j in range(len(names)):
                    ax.text(j, i, f"{J[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if J[i, j] < 0.5 else "black")
            ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.04, label="Jaccard index of top-20 SHAP features")
            ax.set_title(f"{ep}: how different are the prognostic drivers between subtypes?")
            savefig(fig, out, f"F18_SHAP_driver_overlap_jaccard_{ep}")
            pd.DataFrame(J, index=names, columns=names).to_csv(os.path.join(out, "tables", f"T17_SHAP_jaccard_{ep}.csv"))
    pd.DataFrame(rows).to_csv(os.path.join(out, "tables", "T16_SHAP_top_features.csv"), index=False)
    return tops


def stability_analysis(coefs, groups, out):
    if not coefs:
        return
    rows = []
    for c in coefs:
        for f in c["selected"]:
            rows.append({k: c[k] for k in ["endpoint", "strategy", "subgroup", "feature_set"]} | {"feature": f})
    if not rows:
        return
    d = pd.DataFrame(rows)
    nf = pd.DataFrame(coefs).groupby(["endpoint", "strategy", "subgroup", "feature_set"]).size().rename("n_folds")
    fr = d.groupby(["endpoint", "strategy", "subgroup", "feature_set", "feature"]).size().rename("times_selected").reset_index()
    fr = fr.merge(nf.reset_index(), on=["endpoint", "strategy", "subgroup", "feature_set"])
    fr["selection_frequency"] = fr.times_selected / fr.n_folds
    fr["feature_type"] = np.where(fr.feature.isin(groups["genes"]), "gene", np.where(fr.feature.isin(groups["muts"]), "mutation", "clinical"))
    fr = fr.sort_values(["endpoint", "strategy", "subgroup", "feature_set", "selection_frequency"], ascending=[True, True, True, True, False])
    fr.to_csv(os.path.join(out, "tables", "T18_CoxNet_selection_stability.csv"), index=False)
    for ep in fr.endpoint.unique():
        g = fr[(fr.endpoint == ep) & (fr.feature_set == "Clin+Genes") & (fr.feature_type == "gene")]
        if not len(g):
            continue
        g = g.assign(col=np.where(g.strategy == "global", "Global", g.subgroup))
        top = g.groupby("feature").selection_frequency.max().sort_values(ascending=False).head(30).index
        pv = g[g.feature.isin(top)].pivot_table(index="feature", columns="col", values="selection_frequency").fillna(0)
        pv = pv.loc[pv.max(axis=1).sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(0.8 * pv.shape[1] + 3, 0.3 * len(pv) + 1.5))
        im = ax.imshow(pv.values, cmap="Greens", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(pv))); ax.set_yticklabels(pv.index, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.04, label="selection frequency across CV folds")
        ax.set_title(f"{ep}: stability of elastic-net gene selection (Clin+Genes)")
        savefig(fig, out, f"F19_gene_selection_stability_{ep}")


def fmt(summ, ep, sg, strategy, model, fs, m):
    d = summ[(summ.endpoint == ep) & (summ.subgroup == sg) & (summ.strategy == strategy) & (summ.model == model) & (summ.feature_set == fs)]
    if not len(d) or f"{m}_mean" not in d or pd.isna(d[f"{m}_mean"].iloc[0]):
        return "NA"
    r = d.iloc[0]
    return f"{r[f'{m}_mean']:.3f} (95% CI {r[f'{m}_ci_low']:.3f}–{r[f'{m}_ci_high']:.3f})"


def md_table(d):
    try:
        return d.round(3).to_markdown(index=False)
    except Exception:
        return "```\n" + d.round(3).to_string(index=False) + "\n```"


def write_summary(df, summ, cmp, sgt, tops, res, cfg, out, runtime):
    L = ["# Results summary (auto-generated)", "",
         f"Run: {datetime.now():%Y-%m-%d %H:%M} | runtime {runtime / 60:.1f} min | "
         f"{cfg.repeats}x{cfg.folds}-fold repeated stratified CV | seed {cfg.seed} | quick={cfg.quick}", "",
         f"Cohort: n = {len(df)} | OS events = {int(df.os_event.sum())} | breast-cancer deaths = {int(df.dss_event.sum())} | "
         f"other-cause deaths = {int((df.cause == 2).sum())} | median follow-up = {df.os_time.median():.1f} months", "",
         "## Subgroups", "", md_table(sgt), ""]
    for ep in ["OS", "DSS"]:
        if ep not in res:
            continue
        b = res[ep]["best"]
        L += [f"## {ep}", "",
              f"* Best global configuration (whole cohort): **{b['label']}**",
              f"  * Uno C = {fmt(summ, ep, 'All', 'global', b['model'], b['feature_set'], 'uno_c')}",
              f"  * Harrell C = {fmt(summ, ep, 'All', 'global', b['model'], b['feature_set'], 'harrell_c')}",
              f"  * AUC(5y) = {fmt(summ, ep, 'All', 'global', b['model'], b['feature_set'], 'auc_60m')}",
              f"  * IBS = {fmt(summ, ep, 'All', 'global', b['model'], b['feature_set'], 'ibs')}",
              f"  * 5-y binary: balanced accuracy = {fmt(summ, ep, 'All', 'global', b['model'], b['feature_set'], 'balanced_accuracy')}, "
              f"MCC = {fmt(summ, ep, 'All', 'global', b['model'], b['feature_set'], 'mcc')}",
              f"* NPI alone: Uno C = {fmt(summ, ep, 'All', 'global', 'CoxPH', 'NPI', 'uno_c')}",
              f"* Clinical CoxPH: Uno C = {fmt(summ, ep, 'All', 'global', 'CoxPH', 'Clinical', 'uno_c')}", ""]
        if len(cmp):
            c = cmp[cmp.endpoint == ep].copy()
            if len(c) and "delta_uno_c" in c:
                L += ["### Subtype-specific vs global (best model per subgroup & feature set, ΔUno C)", ""]
                for (sg, fs), d in c.groupby(["subgroup", "feature_set"]):
                    r = d.loc[d[["uno_c_global", "uno_c_specific"]].max(axis=1).idxmax()]
                    L.append(f"* {sg} | {fs} | {r.model}: global {r.uno_c_global:.3f} vs specific {r.uno_c_specific:.3f}; "
                             f"Δ = {r.delta_uno_c:+.3f} (95% CI {r.delta_uno_c_ci_low:+.3f} to {r.delta_uno_c_ci_high:+.3f}), "
                             f"p = {r.p_corrected_t_uno_c:.3g}, specific wins {100 * r.specific_win_rate_uno_c:.0f}% of folds")
                L.append("")
        for (e, name), feats in tops.items():
            if e == ep:
                L.append(f"* Top SHAP features – {name}: {', '.join(feats[:10])}")
        L.append("")
    L += ["## Notes for the manuscript", "",
          "* All preprocessing (imputation, scaling, one-hot encoding, mutation filtering, univariate gene selection) and "
          "hyper-parameter tuning (CoxNet inner CV) were fitted inside training folds only.",
          "* Global and subtype-specific models were evaluated on the identical held-out patients of each fold; paired "
          "differences were tested with the Nadeau–Bengio corrected resampled t-test and Wilcoxon signed-rank test, "
          "with Benjamini–Hochberg FDR (q_bh_uno_c).",
          "* 5-year binary metrics exclude patients censored before 60 months; thresholds were chosen by Youden's J on training folds.",
          "* Pooled out-of-fold figures (KM tertiles, calibration, DCA, ROC) use the configuration with the best CV Uno C – "
          "report fold-level CV metrics (T06/T07) as the primary, unbiased estimates.",
          "* SHAP models were fitted on the full (sub)cohort for explanation only; they are not used for performance estimates."]
    with open(os.path.join(out, "RESULTS_SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def write_excel(out):
    try:
        path = os.path.join(out, "ALL_TABLES.xlsx")
        with pd.ExcelWriter(path) as w:
            for fn in sorted(os.listdir(os.path.join(out, "tables"))):
                if fn.endswith(".csv"):
                    d = pd.read_csv(os.path.join(out, "tables", fn))
                    if len(d) > 200000:
                        continue
                    d.to_excel(w, sheet_name=fn[:-4][:31], index=False)
    except Exception as e:
        print("Excel export skipped:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="METABRIC_RNA_Mutation.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-genes", type=int, default=50)
    ap.add_argument("--n-genes-specific", type=int, default=30)
    ap.add_argument("--specific-feature-sets", default="Clinical,Clin+Genes")
    ap.add_argument("--min-events", type=int, default=15)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-treatment", action="store_true")
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--quick", action="store_true")
    cfg = ap.parse_args()
    if not os.path.exists(cfg.data):
        sys.exit(f"Data file not found: {cfg.data}")
    if cfg.quick:
        cfg.folds, cfg.repeats, cfg.n_boot = 3, 1, 100
    cfg.specific_feature_sets = [s.strip() for s in cfg.specific_feature_sets.split(",")]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = cfg.out or f"results_METABRIC_{'quick_' if cfg.quick else ''}{stamp}"
    for sub in ["", "tables", "figures", "figures/pdf", "predictions"]:
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    log = Logger(os.path.join(out, "run_log.txt"))
    t0 = time.time()
    import sklearn, sksurv
    env = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
           "sklearn": sklearn.__version__, "sksurv": sksurv.__version__, "xgboost": xgb.__version__ if HAS_XGB else None,
           "shap": shap.__version__ if HAS_SHAP else None, "statsmodels": sm.__version__ if HAS_SM else None}
    json.dump({"config": vars(cfg), "environment": env}, open(os.path.join(out, "config_and_environment.json"), "w"), indent=2)
    log("Environment:", env)

    df, groups = load_data(cfg.data, log, cfg.exclude_treatment)
    subgroups = define_subgroups(df)
    log("Subgroups:", {k: int(v.sum()) for k, v in subgroups.items()})

    steps = {}
    try:
        table1(df, groups, out); sgt = descriptive_figures(df, groups, subgroups, out, log); steps["descriptive"] = "ok"
    except Exception:
        log(traceback.format_exc()); sgt = pd.DataFrame(); steps["descriptive"] = "failed"

    met, oof, coefs = run_cv(df, groups, subgroups, cfg, log)
    met.to_csv(os.path.join(out, "tables", "T06_cv_metrics_per_fold_RAW.csv"), index=False)
    oof.to_csv(os.path.join(out, "predictions", "out_of_fold_predictions.csv.gz"), index=False, compression="gzip")
    summ = summarize(met)
    summ.to_csv(os.path.join(out, "tables", "T07_cv_metrics_summary_mean_sd_ci.csv"), index=False)
    pretty_table(summ, ["harrell_c", "uno_c", "auc_36m", "auc_60m", "auc_120m", "ibs", "brier_5y", "auc_roc_5y_binary",
                        "pr_auc_5y", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "ppv", "npv", "f1", "mcc", "kappa"]
                 ).to_csv(os.path.join(out, "tables", "T08_cv_metrics_pretty_mean_sd.csv"), index=False)
    cmp = specific_vs_global(met, cfg.specific_feature_sets, {k: int(v.sum()) for k, v in subgroups.items()}, len(df))
    cmp.to_csv(os.path.join(out, "tables", "T09_specific_vs_global_paired_tests.csv"), index=False)
    log("CV finished; producing figures")

    for name, fn in [("benchmark heatmaps", lambda: fig_benchmark_heatmaps(summ, out)),
                     ("model bars", lambda: fig_model_bars(summ, out)),
                     ("specific vs global", lambda: fig_specific_vs_global(cmp, out)),
                     ("subgroup performance", lambda: fig_subgroup_performance(summ, out)),
                     ("stability", lambda: stability_analysis(coefs, groups, out))]:
        try:
            fn(); steps[name] = "ok"
        except Exception:
            log(f"{name} failed:\n" + traceback.format_exc()); steps[name] = "failed"
    res, tops = {}, {}
    try:
        res = oof_analyses(df, summ, oof, subgroups, out, log, cfg); steps["oof analyses"] = "ok"
    except Exception:
        log("OOF analyses failed:\n" + traceback.format_exc()); steps["oof analyses"] = "failed"
    if not cfg.no_shap:
        try:
            tops = shap_analysis(df, groups, subgroups, out, log, cfg); steps["shap"] = "ok"
        except Exception:
            log("SHAP failed:\n" + traceback.format_exc()); steps["shap"] = "failed"
    runtime = time.time() - t0
    try:
        write_summary(df, summ, cmp, sgt, tops, res, cfg, out, runtime)
    except Exception:
        log("summary failed:\n" + traceback.format_exc())
    write_excel(out)
    log("Step status:", steps)
    log(f"Total runtime {runtime / 60:.1f} min")
    zip_path = shutil.make_archive(out, "zip", root_dir=os.path.dirname(os.path.abspath(out)) or ".",
                                   base_dir=os.path.basename(os.path.abspath(out)))
    log(f"Results: {zip_path}")


if __name__ == "__main__":
    main()
