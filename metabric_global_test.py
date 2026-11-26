import sys, os, numpy as np, pandas as pd
import metabric_subtype_survival as M
main, out = sys.argv[1:3]
r = pd.read_csv(os.path.join(main, "tables", "T06_cv_metrics_per_fold_RAW.csv"))
r = r[(r.subgroup != "All") & r.feature_set.isin(["Clinical", "Clin+Genes"])]
k = ["repeat", "fold", "endpoint", "subgroup", "model", "feature_set"]
g = r[r.strategy == "global"][k + ["uno_c"]].merge(r[r.strategy == "specific"][k + ["uno_c"]], on=k, suffixes=("_pooled", "_specific"))
g["delta"] = g.uno_c_specific - g.uno_c_pooled
rows = []
NONOVER = ["Basal", "HER2-enriched", "Claudin-low", "LumB", "LumA"]
for ep in ["DSS", "OS"]:
    for fs in ["Clinical", "Clin+Genes", "both"]:
        for fam, sgs in [("all subgroups", None), ("mutually exclusive PAM50 subtypes", NONOVER)]:
            x = g[g.endpoint == ep]
            if fs != "both":
                x = x[x.feature_set == fs]
            if sgs:
                x = x[x.subgroup.isin(sgs)]
            per = x.groupby(["repeat", "fold"]).delta.mean().values
            t, p, (lo, hi) = M.corrected_ttest(per, 1521.6, 380.4)
            rows.append(dict(endpoint=ep, feature_set=fs, family=fam, comparisons_averaged=int(len(x) / len(per)), partitions=len(per),
                             mean_delta_uno_c=per.mean(), ci_low=lo, ci_high=hi, p_corrected_t=p))
pd.DataFrame(rows).to_csv(os.path.join(out, "R05c_global_dependence_aware_test.csv"), index=False)
print(pd.DataFrame(rows).round(4).to_string())
