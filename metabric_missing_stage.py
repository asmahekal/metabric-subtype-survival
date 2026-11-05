import sys, numpy as np, pandas as pd, types
import metabric_subtype_survival as M, metabric_revision as R
from sksurv.util import Surv
log=M.Logger(sys.argv[2]+'/log.txt')
df,groups=M.load_data(sys.argv[1],log); sub=M.define_subgroups(df); folds=M.make_folds(df,5,3,42)
L=R.learners(42,-1)
clin=groups['clin_num']+groups['clin_cat']; nostage=[c for c in clin if c!='tumor_stage']
cc_mask=df['tumor_stage'].notna().values
rows=[]
for fi,(r,k,tri,tei) in enumerate(folds):
    for ep,ev in R.EP.items():
        valid=df[ev].notna().values; tr,te=tri[valid[tri]],tei[valid[tei]]
        y=Surv.from_arrays(df[ev].fillna(0).astype(bool).values, df.os_time.values)
        for variant,feats,trm,tem in [('with_stage',clin,tr,te),('without_stage',nostage,tr,te),('complete_case_stage',clin,tr[cc_mask[tr]],te[cc_mask[te]])]:
            p=M.FoldPreprocessor(feats,groups,30).fit(df.iloc[trm],y[trm]); Xtr,Xte=p.transform(df.iloc[trm]),p.transform(df.iloc[tem])
            for mn,mf in L.items():
                risk=R.fit_eval(mf,Xtr,y[trm],Xte,y[tem])
                for sg in ['All']+R.SPEC_SG:
                    m=sub[sg][tem]
                    if m.sum()>=10: rows.append(dict(repeat=r,fold=k,endpoint=ep,variant=variant,strategy='pooled',subgroup=sg,model=mn,**R.metrics_simple(y[trm],y[tem][m],risk[m])))
            if variant=='complete_case_stage': continue
            for sg in R.SPEC_SG:
                s_tr,s_te=trm[sub[sg][trm]],tem[sub[sg][tem]]
                ps=M.FoldPreprocessor(feats,groups,30).fit(df.iloc[s_tr],y[s_tr]); Xs,Xst=ps.transform(df.iloc[s_tr]),ps.transform(df.iloc[s_te])
                for mn,mf in L.items():
                    rows.append(dict(repeat=r,fold=k,endpoint=ep,variant=variant,strategy='specific',subgroup=sg,model=mn,**R.metrics_simple(y[s_tr],y[s_te],R.fit_eval(mf,Xs,y[s_tr],Xst,y[s_te]))))
    pd.DataFrame(rows).to_csv(sys.argv[2]+'/R15_missing_stage_sensitivity_per_fold.csv',index=False)
    log(f'fold {fi+1}/15 done')
