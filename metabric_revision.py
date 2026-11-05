#!/usr/bin/env python3
import argparse, io, json, os, re, sys, time, traceback, urllib.request, warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis

import metabric_subtype_survival as M

warnings.filterwarnings("ignore")

SPEC_SG = ["TNBC", "Basal", "HER2-enriched", "Claudin-low", "Aggressive (Basal/HER2/CL)", "LumB", "LumA"]
NONOVERLAP = ["Basal", "HER2-enriched", "Claudin-low", "LumB", "LumA"]
EP = {"DSS": "dss_event", "OS": "os_event"}
MOLECULAR_CLIN = ["pam50_+_claudin-low_subtype", "integrative_cluster", "3-gene_classifier_subtype",
                  "her2_status_measured_by_snp6", "mutation_count"]
TCGA_BASE = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/"
GENE_ALIAS = {"cxcl8": "il8", "kmt2c": "mll3", "kmt2d": "mll2", "tubb4a": "tubb4", "tubb4b": "tubb2c",
              "adgra2": "gpr124", "afdn": "mllt4", "agmo": "tmem195", "large1": "large", "ackr3": "cxcr7",
              "izumo1r": "folr4", "rad51d": "rad51l3"}


class Clock:
    def __init__(self, budget_min):
        self.t0, self.budget = time.time(), budget_min * 60

    def left(self):
        return self.budget - (time.time() - self.t0)

    def ok(self, need_min):
        return self.left() > need_min * 60


def fast_cindex(t, e, r):
    t, e, r = np.asarray(t, float), np.asarray(e, bool), np.asarray(r, float)
    if e.sum() == 0:
        return np.nan
    ti, ri = t[e][:, None], r[e][:, None]
    comp = t[None, :] > ti
    conc = ((ri > r[None, :]) & comp).sum() + 0.5 * ((ri == r[None, :]) & comp).sum()
    n = comp.sum()
    return conc / n if n else np.nan


def paired_test(d_spec, d_ref, keys, metric, n_sg):
    mm = d_spec.merge(d_ref, on=keys, suffixes=("_a", "_b"))
    diff = (mm[f"{metric}_a"] - mm[f"{metric}_b"]).astype(float).dropna().values
    if len(diff) < 3:
        return None
    nf = len(mm) / max(mm["repeat"].nunique(), 1)
    n_test = n_sg / nf
    _, p, (lo, hi) = M.corrected_ttest(diff, n_sg - n_test, n_test)
    return dict(a=mm[f"{metric}_a"].mean(), b=mm[f"{metric}_b"].mean(), delta=diff.mean(), ci_low=lo, ci_high=hi, p=p,
                win_rate=np.mean(diff > 0), n_pairs=len(diff))


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.maximum.accumulate((n - np.arange(n)) * p[o])
    out = np.empty(n); out[o] = np.minimum(adj, 1)
    return out


# ---------------------------------------------------------------------------------------------------------------- P1
def part_descriptives(df, groups, subgroups, folds, main_dir, out, log):
    st = df[M.SUBTYPE_COL].fillna("unclassified")
    tn = np.where(subgroups["TNBC"], "TNBC", "non-TNBC")
    ct = pd.crosstab(st, tn, margins=True)
    ct["in_pooled_aggressive"] = [("yes" if s in ("Basal", "Her2", "claudin-low") else "no") if s != "All" else "" for s in ct.index]
    ct.to_csv(os.path.join(out, "R01_subgroup_overlap_PAM50_by_TNBC.csv"))
    rows = []
    for a in SPEC_SG:
        for b in SPEC_SG:
            rows.append({"subgroup_a": a, "subgroup_b": b, "n_overlap": int((subgroups[a] & subgroups[b]).sum())})
    pd.DataFrame(rows).pivot(index="subgroup_a", columns="subgroup_b", values="n_overlap").to_csv(
        os.path.join(out, "R01b_subgroup_overlap_matrix.csv"))

    rows = []
    for ep, ev in EP.items():
        e = df[ev].fillna(0).values.astype(bool)
        for r, k, tr, te in folds:
            for sg in ["All"] + SPEC_SG:
                m = subgroups[sg]
                rows.append({"endpoint": ep, "subgroup": sg, "repeat": r, "fold": k, "n_train": int(m[tr].sum()),
                             "events_train": int(e[tr][m[tr]].sum()), "n_test": int(m[te].sum()),
                             "events_test": int(e[te][m[te]].sum())})
    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join(out, "R02a_events_per_fold.csv"), index=False)
    s = ev.groupby(["endpoint", "subgroup"]).agg(n_train_median=("n_train", "median"),
                                                  events_train_min=("events_train", "min"),
                                                  events_train_median=("events_train", "median"),
                                                  events_train_max=("events_train", "max"),
                                                  events_test_min=("events_test", "min"),
                                                  events_test_median=("events_test", "median")).reset_index()
    s.to_csv(os.path.join(out, "R02_training_and_test_events_by_subgroup.csv"), index=False)

    cols = groups["clin_num"] + groups["clin_cat"]
    miss = pd.DataFrame({sg: df.loc[subgroups[sg], cols].isna().mean() * 100 for sg in ["All"] + SPEC_SG}).round(1)
    miss.to_csv(os.path.join(out, "R03_missingness_pct_by_subgroup.csv"))
    st_sub = pd.crosstab(df[M.SUBTYPE_COL].fillna("unclassified"), df["tumor_stage"].isna().map({True: "stage_missing", False: "stage_recorded"}))
    st_sub.to_csv(os.path.join(out, "R03b_tumour_stage_missing_by_subtype.csv"))

    rows = []
    for ep, ev_ in EP.items():
        valid = df[ev_].notna()
        t, e = df.os_time, df[ev_].fillna(0).astype(bool)
        for sg in ["All"] + SPEC_SG:
            m = subgroups[sg] & valid.values
            cens = m & (t.values < M.HORIZON) & (~e.values)
            rows.append({"endpoint": ep, "subgroup": sg, "n": int(m.sum()),
                         "excluded_censored_before_5y": int(cens.sum()),
                         "pct_excluded": 100 * cens.sum() / max(m.sum(), 1),
                         "events_within_5y": int((m & (t.values <= M.HORIZON) & e.values).sum()),
                         "event_free_beyond_5y": int((m & (t.values > M.HORIZON)).sum())})
    pd.DataFrame(rows).to_csv(os.path.join(out, "R04_five_year_classification_exclusions.csv"), index=False)
    log("P1 descriptives done")


# ---------------------------------------------------------------------------------------------------------------- P2
def part_multiplicity(main_dir, out, log):
    cmp = pd.read_csv(os.path.join(main_dir, "tables", "T09_specific_vs_global_paired_tests.csv"))
    cmp["holm_uno_c"] = np.nan
    cmp["q_bh_nonoverlapping_uno_c"] = np.nan
    for ep, d in cmp.groupby("endpoint"):
        cmp.loc[d.index, "holm_uno_c"] = holm(d.p_corrected_t_uno_c.values)
        dn = d[d.subgroup.isin(NONOVERLAP)]
        cmp.loc[dn.index, "q_bh_nonoverlapping_uno_c"] = M.bh(dn.p_corrected_t_uno_c.values)
    cmp.to_csv(os.path.join(out, "R05_paired_tests_with_alternative_multiplicity.csv"), index=False)
    rows = []
    for ep, d in cmp.groupby("endpoint"):
        for name, sel in [("all 112", d), ("non-overlapping 80", d[d.subgroup.isin(NONOVERLAP)])]:
            for fs in ["Clinical", "Clin+Genes", "all"]:
                x = sel if fs == "all" else sel[sel.feature_set == fs]
                rows.append({"endpoint": ep, "family": name, "feature_set": fs, "n_tests": len(x),
                             "specific_lower": int((x.delta_uno_c < 0).sum()),
                             "nominal_p<0.05_favour_pooled": int(((x.delta_uno_c < 0) & (x.p_corrected_t_uno_c < 0.05)).sum()),
                             "nominal_p<0.05_favour_specific": int(((x.delta_uno_c > 0) & (x.p_corrected_t_uno_c < 0.05)).sum()),
                             "BH_q<0.05_favour_pooled": int(((x.delta_uno_c < 0) & (x.q_bh_uno_c < 0.05)).sum()) if name == "all 112" else
                             int(((x.delta_uno_c < 0) & (x.q_bh_nonoverlapping_uno_c < 0.05)).sum()),
                             "Holm<0.05_favour_pooled": int(((x.delta_uno_c < 0) & (x.holm_uno_c < 0.05)).sum()),
                             "sign_test_p_two_sided": stats.binomtest(int((x.delta_uno_c < 0).sum()), len(x)).pvalue if len(x) else np.nan})
    pd.DataFrame(rows).to_csv(os.path.join(out, "R05b_multiplicity_summary.csv"), index=False)
    log("P2 multiplicity done")


def part_patient_bootstrap(df, folds, subgroups, main_dir, out, log, B=500, seed=42):
    oof = pd.read_csv(os.path.join(main_dir, "predictions", "out_of_fold_predictions.csv.gz"))
    summ = pd.read_csv(os.path.join(main_dir, "tables", "T07_cv_metrics_summary_mean_sd_ci.csv"))
    rng = np.random.default_rng(seed)
    rows = []
    for ep, ev in EP.items():
        e_all = df[ev].fillna(0).values.astype(bool); t_all = df.os_time.values
        for sg in SPEC_SG:
            for fs in ["Clinical", "Clin+Genes"]:
                s = summ[(summ.endpoint == ep) & (summ.subgroup == sg) & (summ.feature_set == fs)]
                if not len(s):
                    continue
                best = s.loc[s.uno_c_mean.idxmax(), "model"]
                g = oof[(oof.endpoint == ep) & (oof.model == best) & (oof.feature_set == fs) &
                        (((oof.strategy == "global") & (oof.trained_on == "All")) | ((oof.strategy == "specific") & (oof.trained_on == sg)))]
                per_fold = []
                for (r, k), d in g.groupby(["repeat", "fold"]):
                    gl = d[d.strategy == "global"].set_index("idx")["risk"]
                    sp = d[d.strategy == "specific"].set_index("idx")["risk"]
                    idx = np.intersect1d(sp.index.values, gl.index.values)
                    idx = idx[subgroups[sg][idx] & ~df[ev].isna().values[idx]]
                    if len(idx) < 10:
                        continue
                    per_fold.append((idx, gl.loc[idx].values, sp.loc[idx].values))
                if len(per_fold) < 5:
                    continue
                obs = np.mean([fast_cindex(t_all[i], e_all[i], b) - fast_cindex(t_all[i], e_all[i], a) for i, a, b in per_fold])
                boots = []
                for _ in range(B):
                    vals = []
                    for i, a, b in per_fold:
                        j = rng.integers(0, len(i), len(i))
                        ii = i[j]
                        if e_all[ii].sum() < 2:
                            continue
                        vals.append(fast_cindex(t_all[ii], e_all[ii], b[j]) - fast_cindex(t_all[ii], e_all[ii], a[j]))
                    if vals:
                        boots.append(np.mean(vals))
                boots = np.array(boots)
                rows.append({"endpoint": ep, "subgroup": sg, "feature_set": fs, "learner": best,
                             "delta_harrell_c_specific_minus_pooled": obs,
                             "boot_ci_low": np.percentile(boots, 2.5), "boot_ci_high": np.percentile(boots, 97.5),
                             "boot_p_two_sided": 2 * min(np.mean(boots >= 0), np.mean(boots <= 0)), "B": len(boots)})
    pd.DataFrame(rows).to_csv(os.path.join(out, "R06_patient_bootstrap_delta_c_best_learner.csv"), index=False)
    log("P2b patient-level bootstrap done")


# ---------------------------------------------------------------------------------------------------------------- P3
def part_incremental(main_dir, out, log):
    met = pd.read_csv(os.path.join(main_dir, "tables", "T06_cv_metrics_per_fold_RAW.csv"))
    g = met[(met.strategy == "global") & (met.subgroup == "All")]
    keys = ["repeat", "fold", "endpoint", "model"]
    rows = []
    for ep in ["DSS", "OS"]:
        for mdl in g.model.unique():
            for added, base in [("Clin+Genes", "Clinical"), ("Clin+Genes+Mut", "Clin+Genes"), ("Clin+Genes+Mut", "Clinical")]:
                a = g[(g.endpoint == ep) & (g.model == mdl) & (g.feature_set == added)]
                b = g[(g.endpoint == ep) & (g.model == mdl) & (g.feature_set == base)]
                for metric in ["uno_c", "harrell_c", "auc_60m", "ibs"]:
                    res = paired_test(a[keys + [metric]], b[keys + [metric]], keys, metric, 1902)
                    if res:
                        rows.append({"endpoint": ep, "learner": mdl, "comparison": f"{added} vs {base}", "metric": metric,
                                     "value_with_added": res["a"], "value_base": res["b"], "delta": res["delta"],
                                     "ci_low": res["ci_low"], "ci_high": res["ci_high"], "p_corrected_t": res["p"]})
    pd.DataFrame(rows).to_csv(os.path.join(out, "R07_incremental_value_of_molecular_features.csv"), index=False)
    log("P3 incremental value done")


# ---------------------------------------------------------------------------------------------------------------- P4
def cloglog(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(-np.log(1 - p))


def cox_cal_slope(p, t, e):
    try:
        m = CoxPHSurvivalAnalysis(alpha=1e-6).fit(cloglog(p)[:, None], Surv.from_arrays(e.astype(bool), t))
        return float(m.coef_[0])
    except Exception:
        return np.nan


def recalibrate_crossfit(p, t, e, k=10, seed=0):
    out = np.full(len(p), np.nan)
    kf = StratifiedKFold(k, shuffle=True, random_state=seed)
    x = cloglog(p)[:, None]
    for tr, te in kf.split(x, e.astype(int)):
        m = CoxPHSurvivalAnalysis(alpha=1e-6).fit(x[tr], Surv.from_arrays(e[tr].astype(bool), t[tr]))
        fns = m.predict_survival_function(x[te])
        out[te] = [1 - f(min(M.HORIZON, f.x[-1])) for f in fns]
    return out


def aj_at(t, cause, t0):
    tt, c1, _ = M.aalen_johansen(t, cause)
    return c1[tt <= t0][-1] if np.any(tt <= t0) else 0.0


def nb_curve(p, t, e, cause, thresholds, competing):
    n = len(p); res = []
    for pt in thresholds:
        sel = p >= pt; x = sel.mean()
        if sel.sum() == 0:
            res.append(0.0); continue
        r = aj_at(t[sel], cause[sel], M.HORIZON) if competing else 1 - M.km_at(t[sel], e[sel], M.HORIZON)
        res.append(r * x - (1 - r) * x * pt / (1 - pt))
    return np.array(res)


def part_calibration(df, subgroups, main_dir, out, log, B=500, seed=42):
    oof = pd.read_csv(os.path.join(main_dir, "predictions", "out_of_fold_predictions.csv.gz"))
    summ = pd.read_csv(os.path.join(main_dir, "tables", "T07_cv_metrics_summary_mean_sd_ci.csv"))
    rng = np.random.default_rng(seed)
    cal_rows, nb_rows, sum_rows = [], [], []
    thresholds = np.round(np.arange(0.05, 0.41, 0.05), 2)
    for ep, ev in EP.items():
        best = M.pick_best(summ, ep, need_sf=True)
        configs = {f"Best: {best['label']}": best,
                   "Clinical Cox": dict(strategy="global", trained_on="All", model="CoxPH", feature_set="Clinical"),
                   "NPI": dict(strategy="global", trained_on="All", model="CoxPH", feature_set="NPI")}
        valid = df[ev].notna().values
        preds = {}
        for name, c in configs.items():
            a = M.oof_agg(oof, ep, c["strategy"], c["trained_on"], c["model"], c["feature_set"])
            preds[name] = a["p5y"]
        idx = np.array(sorted(set.intersection(*[set(v.dropna().index) for v in preds.values()])))
        idx = idx[valid[idx]]
        t, e = df.os_time.values[idx], df[ev].values[idx].astype(int)
        cause = df["cause"].values[idx].astype(int) if ep == "DSS" else np.where(e == 1, 1, 0)
        P = {}
        for name, s in preds.items():
            p = s.loc[idx].values
            P[name] = p
            P[name + " (recalibrated)"] = recalibrate_crossfit(p, t, e, seed=seed)
        for name, p in P.items():
            bins = pd.qcut(p, 10, labels=False, duplicates="drop")
            pr, ob_km, ob_aj = [], [], []
            for b_ in np.unique(bins):
                m = bins == b_
                pr.append(p[m].mean()); ob_km.append(1 - M.km_at(t[m], e[m], M.HORIZON))
                ob_aj.append(aj_at(t[m], cause[m], M.HORIZON) if ep == "DSS" else np.nan)
                cal_rows.append({"endpoint": ep, "configuration": name, "decile": int(b_), "n": int(m.sum()),
                                 "predicted_5y_risk": pr[-1], "observed_KM_net_risk": ob_km[-1],
                                 "observed_AJ_absolute_risk": ob_aj[-1]})
            pr, ob_km, ob_aj = map(np.array, (pr, ob_km, ob_aj))
            row = {"endpoint": ep, "configuration": name, "n": len(p), "mean_predicted": p.mean(),
                   "observed_KM_net": 1 - M.km_at(t, e, M.HORIZON),
                   "observed_AJ_absolute": aj_at(t, cause, M.HORIZON) if ep == "DSS" else np.nan,
                   "decile_slope_vs_KM": np.polyfit(pr, ob_km, 1)[0], "ECE_vs_KM": np.mean(np.abs(pr - ob_km)),
                   "cox_calibration_slope": cox_cal_slope(p, t, e)}
            if ep == "DSS":
                row.update(decile_slope_vs_AJ=np.polyfit(pr, ob_aj, 1)[0], ECE_vs_AJ=np.mean(np.abs(pr - ob_aj)))
            sum_rows.append(row)
        for competing in ([False, True] if ep == "DSS" else [False]):
            point = {name: nb_curve(p, t, e, cause, thresholds, competing) for name, p in P.items()}
            r_all = aj_at(t, cause, M.HORIZON) if competing else 1 - M.km_at(t, e, M.HORIZON)
            treat_all = r_all - (1 - r_all) * thresholds / (1 - thresholds)
            boots = {name: [] for name in P}; boots_all = []
            for _ in range(B):
                j = rng.integers(0, len(t), len(t))
                for name, p in P.items():
                    boots[name].append(nb_curve(p[j], t[j], e[j], cause[j], thresholds, competing))
                ra = aj_at(t[j], cause[j], M.HORIZON) if competing else 1 - M.km_at(t[j], e[j], M.HORIZON)
                boots_all.append(ra - (1 - ra) * thresholds / (1 - thresholds))
            boots = {k: np.array(v) for k, v in boots.items()}; boots_all = np.array(boots_all)
            ref = "Clinical Cox"
            for name in P:
                for i, pt in enumerate(thresholds):
                    d_ref = boots[name][:, i] - boots[ref][:, i]
                    d_all = boots[name][:, i] - boots_all[:, i]
                    nb_rows.append({"endpoint": ep, "risk_scale": "absolute (Aalen-Johansen)" if competing else "net (Kaplan-Meier)",
                                    "configuration": name, "threshold": pt, "net_benefit": point[name][i],
                                    "nb_ci_low": np.percentile(boots[name][:, i], 2.5), "nb_ci_high": np.percentile(boots[name][:, i], 97.5),
                                    "treat_all": treat_all[i],
                                    "delta_vs_clinical_cox": point[name][i] - point[ref][i],
                                    "delta_vs_clinical_cox_ci_low": np.percentile(d_ref, 2.5),
                                    "delta_vs_clinical_cox_ci_high": np.percentile(d_ref, 97.5),
                                    "delta_vs_treat_all": point[name][i] - treat_all[i],
                                    "delta_vs_treat_all_ci_low": np.percentile(d_all, 2.5),
                                    "delta_vs_treat_all_ci_high": np.percentile(d_all, 97.5)})
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
        cr = pd.DataFrame(cal_rows); cr = cr[cr.endpoint == ep]
        for i, name in enumerate(P):
            d = cr[cr.configuration == name]
            axs[0].plot(d.predicted_5y_risk, d.observed_KM_net_risk, "o-" if "recal" not in name else "s--",
                        color=M.PALETTE[i % len(M.PALETTE)], label=name, ms=4)
        axs[0].plot([0, 1], [0, 1], "k:"); axs[0].set_xlabel("Predicted 5-year risk"); axs[0].set_ylabel("Observed (1 - KM)")
        axs[0].legend(fontsize=6); axs[0].set_title(f"{ep}: calibration before/after cross-fitted recalibration")
        nb = pd.DataFrame(nb_rows); nb = nb[(nb.endpoint == ep) & (nb.risk_scale.str.startswith("net"))]
        for i, name in enumerate(P):
            d = nb[nb.configuration == name]
            axs[1].plot(d.threshold, d.net_benefit, color=M.PALETTE[i % len(M.PALETTE)], label=name)
            axs[1].fill_between(d.threshold, d.nb_ci_low, d.nb_ci_high, color=M.PALETTE[i % len(M.PALETTE)], alpha=0.12)
        axs[1].plot(thresholds, nb[nb.configuration == list(P)[0]].treat_all, "k--", label="Treat all")
        axs[1].axhline(0, color="grey"); axs[1].legend(fontsize=6); axs[1].set_xlabel("Threshold probability")
        axs[1].set_ylabel("Net benefit"); axs[1].set_title("Decision curves with bootstrap 95% CI")
        M.savefig(fig, out, f"RF1_calibration_recalibration_DCA_{ep}")
    pd.DataFrame(cal_rows).to_csv(os.path.join(out, "R08_calibration_deciles_net_vs_absolute.csv"), index=False)
    pd.DataFrame(sum_rows).to_csv(os.path.join(out, "R08b_calibration_summary.csv"), index=False)
    pd.DataFrame(nb_rows).to_csv(os.path.join(out, "R09_decision_curves_bootstrap_CI.csv"), index=False)
    log("P4 calibration / recalibration / competing risks / DCA done")


# ---------------------------------------------------------------------------------------------------------------- P5
def learners(seed, n_jobs):
    return {
        "CoxNet": lambda: M.TunedCoxnet(random_state=seed),
        "RSF": lambda: RandomSurvivalForest(n_estimators=300, min_samples_leaf=15, max_features="sqrt", n_jobs=n_jobs, random_state=seed),
        "GBSA": lambda: GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=seed),
    }


def candidates(seed, n_jobs):
    c = {"CoxNet|auto": lambda: M.TunedCoxnet(random_state=seed)}
    for leaf in (5, 15, 30):
        c[f"RSF|leaf={leaf}"] = (lambda leaf=leaf: RandomSurvivalForest(n_estimators=200, min_samples_leaf=leaf, max_features="sqrt",
                                                                       n_jobs=n_jobs, random_state=seed))
    c["GBSA|lr=0.05,d=2,n=200"] = lambda: GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=seed)
    c["GBSA|lr=0.1,d=3,n=100"] = lambda: GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=seed)
    return c


def fit_eval(mfn, Xtr, ytr, Xte, yte):
    m = mfn(); m.fit(Xtr, ytr)
    r = m.predict(Xte)
    return r


def metrics_simple(ytr, yte, r):
    out = {"n": len(yte), "events": int(yte["event"].sum())}
    if out["events"] < 3:
        return out
    out["harrell_c"] = M.safe(lambda: concordance_index_censored(yte["event"], yte["time"], r)[0])
    keep = yte["time"] < ytr["time"].max()
    tau = min(M.TAU, np.percentile(yte["time"][keep], 90))
    out["uno_c"] = M.safe(lambda: concordance_index_ipcw(ytr, yte[keep], r[keep], tau=tau)[0])
    return out


def inner_select(cands, Xtr, ytr, seed, k=3):
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    scores = {c: [] for c in cands}
    for itr, ite in skf.split(Xtr, ytr["event"].astype(int)):
        if ytr["event"][ite].sum() < 2:
            continue
        for c, fn in cands.items():
            try:
                r = fit_eval(fn, Xtr[itr], ytr[itr], Xtr[ite], ytr[ite])
                scores[c].append(concordance_index_censored(ytr["event"][ite], ytr["time"][ite], r)[0])
            except Exception:
                scores[c].append(np.nan)
    return {c: np.nanmean(v) if len(v) else np.nan for c, v in scores.items()}


def part_sensitivity(df, groups, subgroups, folds, cfg, out, log, clock, variants, endpoints, nested):
    FS = M.feature_sets(groups)
    core = [c for c in groups["clin_num"] + groups["clin_cat"] if c not in M.TREATMENT_COLS + MOLECULAR_CLIN]
    fsets = {"Clinical": FS["Clinical"], "Clin+Genes": FS["Clin+Genes"], "Core": core, "Core+Genes": core + groups["genes"]}
    L = learners(cfg.seed, cfg.n_jobs)
    C = candidates(cfg.seed, cfg.n_jobs)
    rng = np.random.default_rng(cfg.seed)
    rec, sel = [], []
    path_rec = os.path.join(out, "R10_sensitivity_per_fold.csv")
    for fi, (r, k, tr_idx, te_idx) in enumerate(folds):
        tf = time.time()
        for ep in endpoints:
            ev = EP[ep]
            valid = df[ev].notna().values
            tr, te = tr_idx[valid[tr_idx]], te_idx[valid[te_idx]]
            y = Surv.from_arrays(df[ev].fillna(0).astype(bool).values, df.os_time.values)
            prep_cache = {}

            def prep(feats_name, ng, idx_tr, key):
                kk = (feats_name, ng, key)
                if kk not in prep_cache:
                    p = M.FoldPreprocessor(fsets[feats_name], groups, ng).fit(df.iloc[idx_tr], y[idx_tr])
                    prep_cache[kk] = p
                return prep_cache[kk]

            def record(variant, strategy, trained_on, sg, fsn, model, idx_tr, idx_te, risk, extra=None):
                m = metrics_simple(y[idx_tr], y[idx_te], risk)
                rec.append({"repeat": r, "fold": k, "endpoint": ep, "variant": variant, "strategy": strategy,
                            "trained_on": trained_on, "subgroup": sg, "feature_set": fsn, "model": model,
                            "n_train": len(idx_tr), "events_train": int(y["event"][idx_tr].sum()), **m, **(extra or {})})

            jobs = []
            if "matched_genes" in variants:
                jobs += [("genes30_both", "Clin+Genes", 30, 30), ("genes50_both", "Clin+Genes", 50, 50), ("clinical", "Clinical", 30, 30)]
            if "core" in variants:
                jobs += [("core_no_treatment_no_molecular", "Core", 30, 30), ("core_plus_genes30", "Core+Genes", 30, 30)]
            for variant, fsn, ng_pool, ng_spec in jobs:
                p = prep(fsn, ng_pool, tr, "pooled")
                Xtr, Xte = p.transform(df.iloc[tr]), p.transform(df.iloc[te])
                for mn, mf in L.items():
                    try:
                        risk = fit_eval(mf, Xtr, y[tr], Xte, y[te])
                    except Exception as e_:
                        log(f"   fail pooled {variant} {mn}: {e_}"); continue
                    for sg in SPEC_SG:
                        mte = subgroups[sg][te]
                        if mte.sum() >= 10:
                            record(variant, "pooled", "All", sg, fsn, mn, tr[subgroups[sg][tr]], te[mte], risk[mte],
                                   {"events_train_pooled_total": int(y["event"][tr].sum())})
                for sg in SPEC_SG:
                    s_tr, s_te = tr[subgroups[sg][tr]], te[subgroups[sg][te]]
                    if y["event"][s_tr].sum() < cfg.min_events or len(s_te) < 10:
                        continue
                    try:
                        ps = M.FoldPreprocessor(fsets[fsn], groups, ng_spec).fit(df.iloc[s_tr], y[s_tr])
                        Xs, Xst = ps.transform(df.iloc[s_tr]), ps.transform(df.iloc[s_te])
                    except Exception as e_:
                        log(f"   prep fail specific {variant} {sg}: {e_}"); continue
                    for mn, mf in L.items():
                        try:
                            record(variant, "specific", sg, sg, fsn, mn, s_tr, s_te, fit_eval(mf, Xs, y[s_tr], Xst, y[s_te]))
                        except Exception as e_:
                            log(f"   fail specific {variant} {sg} {mn}: {e_}")
                    if "downsampled" in variants and variant in ("genes30_both", "clinical"):
                        n_s = len(s_tr)
                        d_tr = np.sort(rng.choice(tr, size=n_s, replace=False))
                        if y["event"][d_tr].sum() < 5:
                            continue
                        try:
                            pd_ = M.FoldPreprocessor(fsets[fsn], groups, ng_spec).fit(df.iloc[d_tr], y[d_tr])
                            Xd, Xdt = pd_.transform(df.iloc[d_tr]), pd_.transform(df.iloc[s_te])
                        except Exception as e_:
                            log(f"   prep fail downsampled {sg}: {e_}"); continue
                        for mn, mf in L.items():
                            try:
                                record(variant, "pooled_downsampled", "All (random n = subgroup n)", sg, fsn, mn, d_tr, s_te,
                                       fit_eval(mf, Xd, y[d_tr], Xdt, y[s_te]),
                                       {"subgroup_share_in_downsample": float(subgroups[sg][d_tr].mean())})
                            except Exception as e_:
                                log(f"   fail downsampled {sg} {mn}: {e_}")
            if nested and ep in nested:
                for fsn in ["Clinical", "Clin+Genes"]:
                    units = [("pooled", "All", tr)] + [("specific", sg, tr[subgroups[sg][tr]]) for sg in SPEC_SG]
                    for strategy, on, u_tr in units:
                        if strategy == "specific" and y["event"][u_tr].sum() < cfg.min_events:
                            continue
                        u_te = te if strategy == "pooled" else te[subgroups[on][te]]
                        try:
                            pp = M.FoldPreprocessor(fsets[fsn], groups, 30).fit(df.iloc[u_tr], y[u_tr])
                            Xu, Xut = pp.transform(df.iloc[u_tr]), pp.transform(df.iloc[u_te])
                        except Exception as e_:
                            log(f"   prep fail nested {on}: {e_}"); continue
                        sc = inner_select(C, Xu, y[u_tr], cfg.seed + fi)
                        best_overall = max(sc, key=lambda c: -np.inf if np.isnan(sc[c]) else sc[c])
                        per_learner = {}
                        for c, v in sc.items():
                            ln = c.split("|")[0]
                            if not np.isnan(v) and (ln not in per_learner or v > sc[per_learner[ln]]):
                                per_learner[ln] = c
                        chosen = {f"tuned:{ln}": c for ln, c in per_learner.items()}
                        chosen["nested_selected"] = best_overall
                        fitted = {}
                        for label, c in chosen.items():
                            if c not in fitted:
                                try:
                                    fitted[c] = fit_eval(C[c], Xu, y[u_tr], Xut, y[u_te])
                                except Exception as e_:
                                    log(f"   fail nested fit {on} {c}: {e_}"); fitted[c] = None
                            risk = fitted[c]
                            if risk is None:
                                continue
                            sgs = SPEC_SG if strategy == "pooled" else [on]
                            for sg in sgs:
                                mte = subgroups[sg][u_te] if strategy == "pooled" else np.ones(len(u_te), bool)
                                if mte.sum() < 10:
                                    continue
                                record("nested_tuning", strategy, on, sg, fsn, label, u_tr, u_te[mte], risk[mte], {"config": c})
                        sel.append({"repeat": r, "fold": k, "endpoint": ep, "strategy": strategy, "trained_on": on, "feature_set": fsn,
                                    "selected": best_overall, **{f"innerC_{c}": v for c, v in sc.items()}})
        pd.DataFrame(rec).to_csv(path_rec, index=False)
        if sel:
            pd.DataFrame(sel).to_csv(os.path.join(out, "R11_nested_selection_choices.csv"), index=False)
        log(f"  sensitivity fold {fi + 1}/{len(folds)} [{','.join(endpoints)} | {','.join(variants)} | nested={nested}] "
            f"{time.time() - tf:.0f}s; {clock.left() / 60:.0f} min left")
        if not clock.ok(5):
            log("  time budget reached – stopping sensitivity loop early"); break
    return pd.DataFrame(rec), pd.DataFrame(sel)


def sensitivity_tests(rec, out, log, n_sg, tag):
    if rec is None or not len(rec):
        return
    keys = ["repeat", "fold", "endpoint", "subgroup", "feature_set", "model"]
    rows = []
    for (variant, ep, sg, fsn, mdl), d in rec.groupby(["variant", "endpoint", "subgroup", "feature_set", "model"]):
        piv = {s: d[d.strategy == s][keys + ["uno_c", "harrell_c"]] for s in d.strategy.unique()}
        for a, b in [("specific", "pooled"), ("pooled_downsampled", "pooled"), ("specific", "pooled_downsampled")]:
            if a in piv and b in piv:
                for metric in ["uno_c", "harrell_c"]:
                    res = paired_test(piv[a], piv[b], keys, metric, n_sg[sg])
                    if res:
                        rows.append({"variant": variant, "endpoint": ep, "subgroup": sg, "feature_set": fsn, "model": mdl,
                                     "comparison": f"{a} - {b}", "metric": metric, "mean_a": res["a"], "mean_b": res["b"],
                                     "delta": res["delta"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                                     "p_corrected_t": res["p"], "win_rate_a": res["win_rate"], "n_pairs": res["n_pairs"]})
    t = pd.DataFrame(rows)
    if not len(t):
        return
    t["q_bh"] = np.nan
    for _, d in t.groupby(["variant", "endpoint", "comparison", "metric"]):
        ok = d.p_corrected_t.notna()
        t.loc[d.index[ok], "q_bh"] = M.bh(d.p_corrected_t[ok].values)
    t.to_csv(os.path.join(out, f"R12_sensitivity_paired_tests{tag}.csv"), index=False)
    s = t[t.metric == "uno_c"].groupby(["variant", "endpoint", "feature_set", "comparison"]).agg(
        n=("delta", "size"), a_lower=("delta", lambda x: int((x < 0).sum())), median_delta=("delta", "median"),
        nominal_a_better=("p_corrected_t", lambda x: 0), q_lt_005_a_worse=("q_bh", lambda x: 0)).reset_index()
    for i, row in s.iterrows():
        d = t[(t.metric == "uno_c") & (t.variant == row.variant) & (t.endpoint == row.endpoint) &
              (t.feature_set == row.feature_set) & (t.comparison == row.comparison)]
        s.loc[i, "nominal_a_better"] = int(((d.delta > 0) & (d.p_corrected_t < 0.05)).sum())
        s.loc[i, "nominal_a_worse"] = int(((d.delta < 0) & (d.p_corrected_t < 0.05)).sum())
        s.loc[i, "q_lt_005_a_better"] = int(((d.delta > 0) & (d.q_bh < 0.05)).sum())
        s.loc[i, "q_lt_005_a_worse"] = int(((d.delta < 0) & (d.q_bh < 0.05)).sum())
    s.to_csv(os.path.join(out, f"R12b_sensitivity_summary{tag}.csv"), index=False)
    log(f"sensitivity tests written ({tag})")


# ---------------------------------------------------------------------------------------------------------------- P6
def part_shap_stability(df, groups, subgroups, cfg, out, log, B=20):
    import shap
    FS = M.feature_sets(groups)
    fsn = "Clin+Genes+Mut"
    rng = np.random.default_rng(cfg.seed)
    rows, tops = [], {}
    ep, ev = "DSS", "dss_event"
    valid = df[ev].notna().values
    y = Surv.from_arrays(df[ev].fillna(0).astype(bool).values, df.os_time.values)
    targets = [("Pooled", subgroups["All"], cfg.n_genes)] + [(sg, subgroups[sg], cfg.n_genes_specific) for sg in SPEC_SG]
    for name, mask, ng in targets:
        idx0 = np.flatnonzero(mask & valid)
        tops[name] = []
        for b in range(B):
            idx = rng.choice(idx0, len(idx0), replace=True)
            try:
                p = M.FoldPreprocessor(FS[fsn], groups, ng).fit(df.iloc[idx], y[idx])
                X = p.transform(df.iloc[idx])
                m = M.XGBCox(n_estimators=300, random_state=cfg.seed + b, n_jobs=1).fit(X, y[idx])
                sv = shap.TreeExplainer(m.model_).shap_values(X)
                imp = pd.Series(np.abs(sv).mean(0), index=p.names_).sort_values(ascending=False)
                tops[name].append(list(imp.index[:20]))
            except Exception as e_:
                log(f"   SHAP bootstrap fail {name} b={b}: {e_}")
        sets = [set(x) for x in tops[name]]
        if len(sets) < 2:
            continue
        within = [len(a & b_) / len(a | b_) for i, a in enumerate(sets) for b_ in sets[i + 1:]]
        freq = pd.Series([f for s in sets for f in s]).value_counts() / len(sets)
        for f, v in freq.items():
            rows.append({"endpoint": ep, "model": name, "feature": f, "top20_frequency": v,
                         "feature_type": "gene" if f in groups["genes"] else ("mutation" if f in groups["muts"] else "clinical")})
        rows.append({"endpoint": ep, "model": name, "feature": "__WITHIN_MODEL_MEAN_JACCARD__", "top20_frequency": np.mean(within)})
        log(f"   SHAP stability {name}: mean within-model Jaccard {np.mean(within):.2f}; "
            f"features in >=80% of bootstraps: {int((freq >= 0.8).sum())}")
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(out, "R13_SHAP_bootstrap_stability.csv"), index=False)
    names = [n for n in tops if len(tops[n]) > 1]
    stable = {n: set(d[(d.model == n) & (d.top20_frequency >= 0.5) & (d.feature != "__WITHIN_MODEL_MEAN_JACCARD__")].feature) for n in names}
    J = pd.DataFrame([[len(stable[a] & stable[b]) / max(len(stable[a] | stable[b]), 1) for b in names] for a in names], index=names, columns=names)
    J.to_csv(os.path.join(out, "R13b_SHAP_between_model_jaccard_stable_features.csv"))
    log("P6 SHAP stability done")


# ---------------------------------------------------------------------------------------------------------------- P7
def fetch(rel, cache):
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, rel.replace("/", "__"))
    if not os.path.exists(path):
        with urllib.request.urlopen(TCGA_BASE + rel, timeout=600) as r, open(path, "wb") as f:
            f.write(r.read())
    return path


def t_cat_from_size(mm):
    return pd.cut(mm, [-np.inf, 20, 50, np.inf], labels=["T1", "T2", "T3+"]).astype(object)


def n_cat_from_count(n):
    return pd.cut(n, [-np.inf, 0, 3, 9, np.inf], labels=["N0", "N1", "N2", "N3"]).astype(object)


def harmonise_metabric(df, genes):
    h = pd.DataFrame(index=df.index)
    h["age"] = df.age_at_diagnosis
    h["t_cat"] = t_cat_from_size(df.tumor_size)
    h["n_cat"] = n_cat_from_count(df.lymph_nodes_examined_positive)
    h["er"] = df.er_status; h["pr"] = df.pr_status; h["her2"] = df.her2_status
    h["menopause"] = df.inferred_menopausal_state.map({"Post": "Post", "Pre": "Pre"})
    h["subtype"] = df[M.SUBTYPE_COL]
    G = df[genes].astype(float)
    G = (G - G.mean()) / G.std(ddof=0)
    return pd.concat([h, G], axis=1)


def load_tcga(cache, genes, log):
    pat = pd.read_csv(fetch("brca_tcga_pan_can_atlas_2018/data_clinical_patient.txt", cache), sep="\t", comment="#")
    leg = pd.read_csv(fetch("brca_tcga/data_clinical_patient.txt", cache), sep="\t", comment="#", low_memory=False)
    expr = pd.read_csv(fetch("brca_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt", cache), sep="\t")
    pat = pat[pat.SEX.astype(str).str.lower() == "female"].copy()
    d = pd.DataFrame({"patient": pat.PATIENT_ID.values})
    d["age"] = pd.to_numeric(pat.AGE, errors="coerce").values
    ts = pat.PATH_T_STAGE.astype(str).str.upper().str.extract(r"(T[0-4])")[0]
    d["t_cat"] = ts.map({"T1": "T1", "T2": "T2", "T3": "T3+", "T4": "T3+"}).values
    ns = pat.PATH_N_STAGE.astype(str).str.upper().str.extract(r"(N[0-3])")[0]
    d["n_cat"] = ns.values
    d["subtype"] = pat.SUBTYPE.astype(str).str.replace("BRCA_", "", regex=False).replace({"nan": np.nan}).values
    d["os_time"] = pd.to_numeric(pat.OS_MONTHS, errors="coerce").values
    d["os_event"] = pat.OS_STATUS.astype(str).str.startswith("1").astype(float).values
    d["dss_time"] = pd.to_numeric(pat.DSS_MONTHS, errors="coerce").values
    dss = pat.DSS_STATUS.astype(str)
    d["dss_event"] = np.where(dss.str.startswith("1"), 1.0, np.where(dss.str.startswith("0"), 0.0, np.nan))
    lg = leg.set_index("PATIENT_ID")
    def rec(col):
        return lg.reindex(d.patient)[col].astype(str).str.strip().str.capitalize().where(
            lambda s: s.isin(["Positive", "Negative"])).values if col in lg else np.nan
    d["er"] = rec("ER_STATUS_BY_IHC"); d["pr"] = rec("PR_STATUS_BY_IHC")
    ihc = lg.reindex(d.patient)["IHC_HER2"].astype(str).str.capitalize() if "IHC_HER2" in lg else pd.Series(np.nan, index=d.patient)
    fish = lg.reindex(d.patient)["HER2_FISH_STATUS"].astype(str).str.capitalize() if "HER2_FISH_STATUS" in lg else pd.Series(np.nan, index=d.patient)
    her2 = np.where((ihc.values == "Positive") | (fish.values == "Positive"), "Positive",
                    np.where((ihc.values == "Negative") | (fish.values == "Negative"), "Negative", "NA"))
    d["her2"] = pd.Series(her2, dtype=object).replace({"NA": np.nan}).values
    mp = lg.reindex(d.patient)["MENOPAUSE_STATUS"].astype(str) if "MENOPAUSE_STATUS" in lg else pd.Series("", index=d.patient)
    d["menopause"] = pd.Series(np.where(mp.str.startswith("Post").values, "Post",
                                        np.where(mp.str.startswith("Pre").values, "Pre", "NA")), dtype=object).replace({"NA": np.nan}).values
    expr["sym"] = expr.Hugo_Symbol.astype(str).str.lower()
    expr = expr.drop_duplicates("sym").set_index("sym").drop(columns=["Hugo_Symbol", "Entrez_Gene_Id"], errors="ignore")
    expr.columns = [c[:12] for c in expr.columns]
    expr = expr.loc[:, ~pd.Index(expr.columns).duplicated()]
    rows = {}
    for g in genes:
        src = g if g in expr.index else GENE_ALIAS.get(g)
        if src is not None and src in expr.index:
            v = np.log2(expr.loc[src].astype(float).clip(lower=0) + 1)
            rows[g] = (v - v.mean()) / (v.std(ddof=0) if v.std(ddof=0) > 0 else 1)
    E = pd.DataFrame(rows)
    d = d.set_index("patient").join(E, how="inner").reset_index()
    log(f"TCGA: {len(d)} female patients with RNA-seq; genes matched {E.shape[1]}/{len(genes)}; "
        f"OS events {int(d.os_event.sum())}, DSS events {int(np.nansum(d.dss_event))}")
    return d, [g for g in genes if g in E.columns]


def tcga_subgroups(d):
    st = d.subtype
    tn = (d.er == "Negative") & (d.pr == "Negative") & (d.her2 == "Negative")
    return {"All": np.ones(len(d), bool), "TNBC": tn.values, "Basal": (st == "Basal").values,
            "HER2-enriched": (st == "Her2").values, "Aggressive (Basal/HER2)": st.isin(["Basal", "Her2"]).values,
            "LumB": (st == "LumB").values, "LumA": (st == "LumA").values}


def part_tcga(df, groups, subgroups, cfg, out, log, B=1000):
    genes_m = groups["genes"]
    T, genes = load_tcga(os.path.join(out, "_tcga_cache"), genes_m, log)
    Hm = harmonise_metabric(df, genes)
    Hm["subtype"] = Hm["subtype"].replace({"claudin-low": "Claudin-low"})
    clin_num, clin_cat = ["age"], ["t_cat", "n_cat", "er", "pr", "her2", "menopause"]
    sg_m = {"All": subgroups["All"], "TNBC": subgroups["TNBC"], "Basal": subgroups["Basal"], "HER2-enriched": subgroups["HER2-enriched"],
            "Aggressive (Basal/HER2)": df[M.SUBTYPE_COL].isin(["Basal", "Her2"]).values, "LumB": subgroups["LumB"], "LumA": subgroups["LumA"]}
    sg_t = tcga_subgroups(T)
    desc = []
    for sg in sg_t:
        for ep, (tc, ec) in {"OS": ("os_time", "os_event"), "DSS": ("dss_time", "dss_event")}.items():
            m = sg_t[sg] & T[ec].notna().values & T[tc].notna().values & (T[tc].values > 0)
            desc.append({"cohort": "TCGA", "subgroup": sg, "endpoint": ep, "n": int(m.sum()), "events": int(T[ec].values[m].sum()),
                         "events_within_5y": int(((T[tc].values <= 60) & (T[ec].values == 1) & m).sum()),
                         "median_followup_months": float(np.median(T[tc].values[m])) if m.sum() else np.nan})
    pd.DataFrame(desc).to_csv(os.path.join(out, "R14a_TCGA_cohort_events.csv"), index=False)
    L = learners(cfg.seed, cfg.n_jobs)
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for ep, (tc, ec, mev) in {"DSS": ("dss_time", "dss_event", "dss_event"), "OS": ("os_time", "os_event", "os_event")}.items():
        vm = df[mev].notna().values
        ym = Surv.from_arrays(df[mev].fillna(0).astype(bool).values, df.os_time.values)
        vt = T[ec].notna().values & T[tc].notna().values & (T[tc].values > 0)
        Tt = T[vt].reset_index(drop=True)
        yt = Surv.from_arrays(Tt[ec].astype(bool).values, Tt[tc].values)
        sgt = {k: v[vt] for k, v in sg_t.items()}
        for fsn, feats_extra in [("Harmonised clinical", []), ("Harmonised clinical + genes", genes)]:
            gm = dict(clin_num=clin_num, clin_cat=clin_cat + ["subtype"], genes=list(feats_extra), muts=[])
            gs = dict(clin_num=clin_num, clin_cat=clin_cat, genes=list(feats_extra), muts=[])
            pooled_feats = clin_num + clin_cat + ["subtype"] + list(feats_extra)
            spec_feats = clin_num + clin_cat + list(feats_extra)
            itr = np.flatnonzero(vm)
            pp = M.FoldPreprocessor(pooled_feats, gm, 30).fit(Hm.iloc[itr], ym[itr])
            Xp, Xpt = pp.transform(Hm.iloc[itr]), pp.transform(Tt)
            pooled_risk = {mn: fit_eval(mf, Xp, ym[itr], Xpt, yt) for mn, mf in L.items()}
            for sg in ["TNBC", "Basal", "HER2-enriched", "Aggressive (Basal/HER2)", "LumB", "LumA"]:
                s_tr = np.flatnonzero(vm & sg_m[sg])
                mt = sgt[sg]
                if mt.sum() < 20 or yt["event"][mt].sum() < 5:
                    continue
                ps = M.FoldPreprocessor(spec_feats, gs, 30).fit(Hm.iloc[s_tr], ym[s_tr])
                Xs, Xst = ps.transform(Hm.iloc[s_tr]), ps.transform(Tt[mt])
                tt, ee = yt["time"][mt], yt["event"][mt]
                tau = min(60.0, np.percentile(tt, 95))
                for mn, mf in L.items():
                    try:
                        rs = fit_eval(mf, Xs, ym[s_tr], Xst, yt[mt])
                    except Exception as e_:
                        log(f"   TCGA fail {sg} {mn}: {e_}"); continue
                    rp = pooled_risk[mn][mt]
                    cp, cs = fast_cindex(tt, ee, rp), fast_cindex(tt, ee, rs)
                    up = M.safe(lambda: concordance_index_ipcw(yt[mt], yt[mt], rp, tau=tau)[0])
                    us = M.safe(lambda: concordance_index_ipcw(yt[mt], yt[mt], rs, tau=tau)[0])
                    bd = []
                    for _ in range(B):
                        j = rng.integers(0, len(tt), len(tt))
                        if ee[j].sum() < 2:
                            continue
                        bd.append(fast_cindex(tt[j], ee[j], rs[j]) - fast_cindex(tt[j], ee[j], rp[j]))
                    bd = np.array(bd)
                    rows.append({"endpoint": ep, "feature_set": fsn, "subgroup": sg, "learner": mn,
                                 "n_train_metabric_pooled": len(itr), "n_train_metabric_specific": len(s_tr),
                                 "events_train_specific": int(ym["event"][s_tr].sum()),
                                 "n_tcga": int(mt.sum()), "events_tcga": int(ee.sum()),
                                 "harrell_c_pooled": cp, "harrell_c_specific": cs, "delta_harrell_c": cs - cp,
                                 "delta_ci_low": np.percentile(bd, 2.5), "delta_ci_high": np.percentile(bd, 97.5),
                                 "boot_p_two_sided": 2 * min(np.mean(bd >= 0), np.mean(bd <= 0)),
                                 "uno_c_pooled_tau": up, "uno_c_specific_tau": us, "tau_months": tau})
            r_all = {mn: fast_cindex(yt["time"], yt["event"], pooled_risk[mn]) for mn in L}
            for mn, c in r_all.items():
                rows.append({"endpoint": ep, "feature_set": fsn, "subgroup": "All", "learner": mn, "n_tcga": len(Tt),
                             "events_tcga": int(yt["event"].sum()), "harrell_c_pooled": c})
            log(f"   TCGA {ep} {fsn}: done")
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "R14_TCGA_external_validation_pooled_vs_specific.csv"), index=False)
    log("P7 TCGA external validation done")


# ---------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="METABRIC_RNA_Mutation.csv")
    ap.add_argument("--main", required=True)
    ap.add_argument("--out", default="results_revision")
    ap.add_argument("--budget-min", type=float, default=420)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true")
    cfg = ap.parse_args()
    cfg.folds, cfg.repeats, cfg.min_events, cfg.n_genes, cfg.n_genes_specific = 5, 3, 15, 50, 30
    for sub_ in ["", "figures", "figures/pdf"]:
        os.makedirs(os.path.join(cfg.out, sub_), exist_ok=True)
    log = M.Logger(os.path.join(cfg.out, "revision_log.txt"))
    clock = Clock(cfg.budget_min)
    df, groups = M.load_data(cfg.data, log)
    subgroups = M.define_subgroups(df)
    folds = M.make_folds(df, cfg.folds, cfg.repeats, cfg.seed)
    if cfg.quick:
        folds = folds[:2]
    n_sg = {k: int(v.sum()) for k, v in subgroups.items()}
    status = {}

    def run(name, fn, need_min=1):
        if not clock.ok(need_min):
            status[name] = "skipped (time)"; log(f"SKIP {name}: not enough time left"); return None
        t = time.time()
        try:
            res = fn(); status[name] = f"ok ({(time.time() - t) / 60:.1f} min)"; return res
        except Exception:
            status[name] = "failed"; log(f"{name} failed:\n" + traceback.format_exc()); return None

    run("P1 descriptives", lambda: part_descriptives(df, groups, subgroups, folds, cfg.main, cfg.out, log))
    run("P2 multiplicity", lambda: part_multiplicity(cfg.main, cfg.out, log))
    run("P2b patient bootstrap", lambda: part_patient_bootstrap(df, folds, subgroups, cfg.main, cfg.out, log, B=100 if cfg.quick else 500))
    run("P3 incremental value", lambda: part_incremental(cfg.main, cfg.out, log))
    run("P4 calibration/DCA", lambda: part_calibration(df, subgroups, cfg.main, cfg.out, log, B=50 if cfg.quick else 500))
    run("P7 TCGA", lambda: part_tcga(df, groups, subgroups, cfg, cfg.out, log, B=100 if cfg.quick else 1000), need_min=20)
    recA = run("P5a sensitivity (matched genes, core, downsampled; DSS+OS)",
               lambda: part_sensitivity(df, groups, subgroups, folds, cfg, cfg.out, log, clock,
                                        ["matched_genes", "core", "downsampled"], ["DSS", "OS"], None)[0], need_min=60)
    if recA is not None:
        recA.to_csv(os.path.join(cfg.out, "R10_sensitivity_per_fold.csv"), index=False)
        run("P5a tests", lambda: sensitivity_tests(recA, cfg.out, log, n_sg, ""))
    out_b = os.path.join(cfg.out, "nested"); os.makedirs(out_b, exist_ok=True)
    recB = run("P5b nested tuning + model selection (DSS)",
               lambda: part_sensitivity(df, groups, subgroups, folds, cfg, out_b, log, clock, [], ["DSS"], ["DSS"])[0], need_min=60)
    if recB is not None:
        run("P5b tests", lambda: sensitivity_tests(recB, cfg.out, log, n_sg, "_nested_DSS"))
    run("P6 SHAP stability", lambda: part_shap_stability(df, groups, subgroups, cfg, cfg.out, log, B=5 if cfg.quick else 20), need_min=25)
    out_c = os.path.join(cfg.out, "nested_OS"); os.makedirs(out_c, exist_ok=True)
    recC = run("P5c nested tuning + model selection (OS)",
               lambda: part_sensitivity(df, groups, subgroups, folds, cfg, out_c, log, clock, [], ["OS"], ["OS"])[0], need_min=90)
    if recC is not None:
        run("P5c tests", lambda: sensitivity_tests(recC, cfg.out, log, n_sg, "_nested_OS"))
    json.dump(status, open(os.path.join(cfg.out, "revision_status.json"), "w"), indent=2)
    log("Revision status:", status)
    try:
        with pd.ExcelWriter(os.path.join(cfg.out, "REVISION_TABLES.xlsx")) as xw:
            for f in sorted(os.listdir(cfg.out)):
                if f.endswith(".csv") and f != "R10_sensitivity_per_fold.csv":
                    try:
                        pd.read_csv(os.path.join(cfg.out, f)).to_excel(xw, sheet_name=f[:31], index=False)
                    except Exception:
                        pass
    except Exception:
        log("excel failed:\n" + traceback.format_exc())
    log(f"Revision finished in {(time.time() - clock.t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
