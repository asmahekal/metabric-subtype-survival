import sys, os, numpy as np, pandas as pd
import metabric_subtype_survival as M, metabric_revision as R
from statsmodels.duration.hazard_regression import PHReg
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from statsmodels.stats.multitest import multipletests

data, main, revdir, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
log = M.Logger(os.path.join(out, "riley_log.txt"))
df, groups = M.load_data(data, log)
sub = M.define_subgroups(df)
FS = M.feature_sets(groups)
folds = M.make_folds(df, 5, 3, 42)
oof = pd.read_csv(os.path.join(main, "predictions", "out_of_fold_predictions.csv.gz"))
summ = pd.read_csv(os.path.join(main, "tables", "T07_cv_metrics_summary_mean_sd_ci.csv"))
evf = pd.read_csv(os.path.join(revdir, "R02a_events_per_fold.csv"))
evf["subgroup"] = evf.subgroup.replace({"Aggressive": "Aggressive (Basal/HER2/CL)"})
S_, D_ = 0.9, 0.05


def r2cs(t, e, x):
    m = PHReg(t, x.reshape(len(t), -1), status=e, ties="efron")
    f = m.fit(disp=0)
    lr = 2 * (f.llf - m.loglike(np.zeros(x.reshape(len(t), -1).shape[1])))
    return 1 - np.exp(-lr / len(t))


def max_r2(t, e):
    E, T = e.sum(), t.sum() / 12.0
    return 1 - np.exp(2 * (E * np.log(E / T) - E) / len(t))


def n_t2_t3(P, r2, mx):
    n2 = P / ((S_ - 1) * np.log(1 - r2 / S_))
    s3 = r2 / (r2 + D_ * mx)
    n3 = P / ((s3 - 1) * np.log(1 - r2 / s3))
    return n2, n3


def n_t1(t, e, horizons=(60, 120)):
    lam = e.sum() / (t.sum() / 12.0); fu = t.mean() / 12.0
    req = 0
    for h in horizons:
        ty = h / 12.0; F = 1 - np.exp(-lam * ty)
        for n in range(10, 200000, 5):
            E = n * lam * fu
            lo = 1 - np.exp(-lam * np.exp(-1.96 / np.sqrt(E)) * ty); hi = 1 - np.exp(-lam * np.exp(1.96 / np.sqrt(E)) * ty)
            if max(hi - F, F - lo) <= 0.05:
                req = max(req, n); break
    return req


def p_max(n, r2, mx):
    a = n * (S_ - 1) * np.log(1 - r2 / S_)
    s3 = r2 / (r2 + D_ * mx)
    b = n * (s3 - 1) * np.log(1 - r2 / s3)
    return min(a, b)


def compact(d):
    x = pd.DataFrame(index=d.index)
    x["npi"] = d.nottingham_prognostic_index
    x["age"] = d.age_at_diagnosis
    x["er"] = (d.er_status == "Positive").astype(float)
    x["her2"] = (d.her2_status == "Positive").astype(float)
    x["pr"] = (d.pr_status == "Positive").astype(float)
    x["log_nodes"] = np.log1p(d.lymph_nodes_examined_positive)
    x["size"] = d.tumor_size
    x["grade"] = d.neoplasm_histologic_grade
    return x


ORDER = ["npi", "age", "er", "her2", "pr", "log_nodes", "size", "grade"]
KS = [1, 2, 4, 6, 8]
XC = compact(df)


def fitpred(tr, te, cols, y):
    a = XC.iloc[tr][cols]; b = XC.iloc[te][cols]
    med = a.median(); a = a.fillna(med); b = b.fillna(med)
    keep = [c for c in cols if a[c].std() > 0]
    mu, sd = a[keep].mean(), a[keep].std()
    m = CoxPHSurvivalAnalysis(alpha=1e-4, ties="efron").fit(((a[keep] - mu) / sd).values, y[tr])
    return m.predict(((b[keep] - mu) / sd).values), len(keep)


rows, oofc = [], {}
for ep, ev in R.EP.items():
    valid = df[ev].notna().values
    y = Surv.from_arrays(df[ev].fillna(0).astype(bool).values, df.os_time.values)
    lp = np.full(len(df), np.nan); cnt = np.zeros(len(df))
    for r, k, tri, tei in folds:
        tr, te = tri[valid[tri]], tei[valid[tei]]
        p, _ = fitpred(tr, te, ORDER, y)
        lp[te] = np.nansum([lp[te] * cnt[te], p], axis=0) / (cnt[te] + 1); cnt[te] += 1
    oofc[ep] = lp
t_all = df.os_time.values.astype(float)
for ep, ev in R.EP.items():
    valid = df[ev].notna().values
    e_all = df[ev].fillna(0).values.astype(int)
    y = Surv.from_arrays(e_all.astype(bool), t_all)
    for fs in ["Clinical", "Clin+Genes", "Compact (8 parameters)"]:
        if fs != "Compact (8 parameters)":
            b = M.pick_best(summ, ep, fs=fs)
            a = M.oof_agg(oof, ep, "global", "All", b["model"], fs)
        for sg in ["All"] + R.SPEC_SG:
            idx = np.flatnonzero(sub[sg] & valid)
            if fs == "Compact (8 parameters)":
                x = oofc[ep][idx]; ii = idx; P = 8; Pc = 8
            else:
                ng = 50 if sg == "All" else 30
                pp = M.FoldPreprocessor(FS[fs], groups, ng).fit(df.iloc[idx], y[idx])
                P = pp.transform(df.iloc[idx]).shape[1]
                Pc = P + (len(groups["genes"]) - ng if fs == "Clin+Genes" else 0)
                ii = np.intersect1d(idx, a.index.values); x = a.loc[ii, "risk"].values.astype(float)
            x = (x - np.nanmean(x)) / np.nanstd(x)
            r2 = r2cs(t_all[ii], e_all[ii], x); mx = max_r2(t_all[idx], e_all[idx])
            n2, n3 = n_t2_t3(Pc, r2, mx); n1 = n_t1(t_all[idx], e_all[idx])
            f = evf[(evf.endpoint == ep) & (evf.subgroup == sg)]
            ntr, etr = f.n_train.mean(), f.events_train.mean()
            req = max(n1, n2, n3)
            rows.append(dict(endpoint=ep, feature_set=fs, subgroup=sg, n=len(idx), events=int(e_all[idx].sum()),
                             parameters_retained=P, parameters_considered=Pc, r2_cs_anticipated=r2, r2_cs_max=mx,
                             n_T1_overall_risk=n1, n_T2_shrinkage=n2, n_T3_optimism=n3, n_required=req,
                             events_required=req * e_all[idx].mean(), n_train_per_fold=ntr, events_train_per_fold=etr,
                             pct_of_requirement=100 * ntr / req, max_parameters_supported=p_max(ntr, r2, mx), T1_met=ntr >= n1))
ss = pd.DataFrame(rows)
ss.to_csv(os.path.join(out, "R17_minimum_sample_size_Riley_T1_T3.csv"), index=False)
log("sample size table done")

cmp_rows, fold_rows = [], []
for ep, ev in R.EP.items():
    valid = df[ev].notna().values
    y = Surv.from_arrays(df[ev].fillna(0).astype(bool).values, df.os_time.values)
    s8 = ss[(ss.endpoint == ep) & (ss.feature_set == "Compact (8 parameters)")].set_index("subgroup")
    for sg in R.SPEC_SG:
        pm = s8.loc[sg, "max_parameters_supported"]
        ok = [k for k in KS if k <= pm]
        k_spec = max(ok) if ok else 1
        kp = s8.loc["All", "max_parameters_supported"]
        k_pool = max([k for k in KS if k <= kp])
        for r, kk, tri, tei in folds:
            tr, te = tri[valid[tri]], tei[valid[tei]]
            s_tr, s_te = tr[sub[sg][tr]], te[sub[sg][te]]
            if len(s_te) < 10:
                continue
            for design, kpool in [("same predictors", k_spec), ("each at its supported complexity", k_pool)]:
                rp, _ = fitpred(tr, s_te, ORDER[:kpool], y)
                rs, used = fitpred(s_tr, s_te, ORDER[:k_spec], y)
                mp = R.metrics_simple(y[s_tr], y[s_te], rp); ms = R.metrics_simple(y[s_tr], y[s_te], rs)
                fold_rows.append(dict(repeat=r, fold=kk, endpoint=ep, subgroup=sg, design=design, k_pooled=kpool, k_specific=k_spec,
                                      specific_compliant=bool(ok), uno_c_pooled=mp.get("uno_c"), uno_c_specific=ms.get("uno_c"),
                                      harrell_c_pooled=mp.get("harrell_c"), harrell_c_specific=ms.get("harrell_c")))
fr = pd.DataFrame(fold_rows)
fr.to_csv(os.path.join(out, "R18_riley_compliant_per_fold.csv"), index=False)
for (ep, sg, des), g in fr.groupby(["endpoint", "subgroup", "design"]):
    dlt = (g.uno_c_specific - g.uno_c_pooled).dropna().values
    t, p, (lo, hi) = M.corrected_ttest(dlt, 4.0, 1.0)
    cmp_rows.append(dict(endpoint=ep, subgroup=sg, design=des, k_pooled=g.k_pooled.iloc[0], k_specific=g.k_specific.iloc[0],
                         specific_compliant=g.specific_compliant.iloc[0], uno_c_pooled=g.uno_c_pooled.mean(), uno_c_specific=g.uno_c_specific.mean(),
                         delta=dlt.mean(), ci_low=lo, ci_high=hi, p_corrected_t=p))
cp = pd.DataFrame(cmp_rows)
cp["q_holm"] = np.nan
for (ep, des), g in cp.groupby(["endpoint", "design"]):
    cp.loc[g.index, "q_holm"] = multipletests(g.p_corrected_t, method="holm")[1]
cp.to_csv(os.path.join(out, "R18b_riley_compliant_paired_tests.csv"), index=False)

cal = []
for ep, ev in R.EP.items():
    valid = df[ev].notna().values
    e = df[ev].fillna(0).values.astype(int); m = valid & ~np.isnan(oofc[ep])
    x = oofc[ep][m]
    f = PHReg(t_all[m], x[:, None], status=e[m], ties="efron").fit(disp=0)
    from sksurv.metrics import concordance_index_censored
    hc = concordance_index_censored(e[m].astype(bool), t_all[m], x)[0]
    cal.append(dict(endpoint=ep, model="pooled compact Cox, 8 parameters", n=int(m.sum()), harrell_c_oof=hc, cox_calibration_slope=float(f.params[0]),
                    slope_ci_low=float(f.conf_int()[0][0]), slope_ci_high=float(f.conf_int()[0][1])))
pd.DataFrame(cal).to_csv(os.path.join(out, "R18c_pooled_compact_calibration.csv"), index=False)
log("riley compliant analysis done")
