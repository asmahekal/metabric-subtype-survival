import sys, os, numpy as np, pandas as pd
import metabric_subtype_survival as M, metabric_revision as R
from statsmodels.duration.hazard_regression import PHReg
from sksurv.util import Surv

data, main, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
log = M.Logger(os.path.join(out, "sample_size_log.txt"))
df, groups = M.load_data(data, log)
sub = M.define_subgroups(df)
FS = M.feature_sets(groups)
oof = pd.read_csv(os.path.join(main, "predictions", "out_of_fold_predictions.csv.gz"))
summ = pd.read_csv(os.path.join(main, "tables", "T07_cv_metrics_summary_mean_sd_ci.csv"))
ev_fold = pd.read_csv(os.path.join(sys.argv[4], "R02a_events_per_fold.csv")) if len(sys.argv) > 4 else None


def r2cs_from_score(t, e, x):
    m = PHReg(t, x[:, None], status=e, ties="efron")
    f = m.fit(disp=0)
    lr = 2 * (f.llf - m.loglike(np.zeros(1)))
    return 1 - np.exp(-lr / len(t)), lr


def max_r2cs(t, e):
    E, T = e.sum(), t.sum() / 12.0
    lnl0 = E * np.log(E / T) - E
    return 1 - np.exp(2 * lnl0 / len(t))


def n_required(P, r2, mx, S=0.9, delta=0.05):
    n1 = P / ((S - 1) * np.log(1 - r2 / S))
    s2 = r2 / (r2 + delta * mx)
    n2 = P / ((s2 - 1) * np.log(1 - r2 / s2))
    return n1, n2, s2


rows = []
for ep, ev in R.EP.items():
    valid = df[ev].notna().values
    t_all = df.os_time.values.astype(float); e_all = df[ev].fillna(0).values.astype(int)
    y = Surv.from_arrays(e_all.astype(bool), t_all)
    for fs in ["Clinical", "Clin+Genes"]:
        b = M.pick_best(summ, ep, fs=fs)
        a = M.oof_agg(oof, ep, "global", "All", b["model"], fs)
        for sg in ["All"] + R.SPEC_SG:
            m = sub[sg] & valid
            idx = np.flatnonzero(m)
            ng = 50 if sg == "All" else 30
            p = M.FoldPreprocessor(FS[fs], groups, ng).fit(df.iloc[idx], y[idx])
            P = p.transform(df.iloc[idx]).shape[1]
            P_cand = P + (len(groups["genes"]) - ng if fs == "Clin+Genes" else 0)
            ii = np.intersect1d(idx, a.index.values)
            x = a.loc[ii, "risk"].values.astype(float); x = (x - x.mean()) / x.std()
            r2, lr = r2cs_from_score(t_all[ii], e_all[ii], x)
            mx = max_r2cs(t_all[idx], e_all[idx])
            n1, n2, s2 = n_required(P, r2, mx)
            n1c, n2c, _ = n_required(P_cand, r2, mx)
            ef = e_all[idx].mean()
            req = max(n1, n2); req_c = max(n1c, n2c)
            row = dict(endpoint=ep, feature_set=fs, subgroup=sg, n=len(idx), events=int(e_all[idx].sum()), event_fraction=ef,
                       parameters_in_model=P, candidate_parameters=P_cand, r2_cs_anticipated=r2, r2_cs_max=mx,
                       r2_nagelkerke_anticipated=r2 / mx, n_required_shrinkage=n1, n_required_optimism=n2,
                       n_required=req, events_required=req * ef, n_required_candidates=req_c, events_required_candidates=req_c * ef)
            if ev_fold is not None:
                f = ev_fold[(ev_fold.endpoint == ep) & (ev_fold.subgroup.replace({"Aggressive": "Aggressive (Basal/HER2/CL)"}) == sg)]
                row.update(n_train_per_fold=f.n_train.mean(), events_train_per_fold=f.events_train.mean())
                row["ratio_available_to_required"] = row["n_train_per_fold"] / req
            rows.append(row)
pd.DataFrame(rows).to_csv(os.path.join(out, "R17_minimum_sample_size_Riley.csv"), index=False)
log("sample size done")
