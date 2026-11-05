import sys, numpy as np, pandas as pd
import metabric_subtype_survival as M, metabric_revision as R
from sksurv.util import Surv
log=M.Logger(sys.argv[2]+'/log.txt')
df,groups=M.load_data(sys.argv[1],log); folds=M.make_folds(df,5,3,42)
L=R.learners(42,-1)
clin=groups['clin_num']+groups['clin_cat']
core=[c for c in clin if c not in M.TREATMENT_COLS+R.MOLECULAR_CLIN]
fs={'Core':(core,{**groups,'genes':[],'muts':[]},50),'Core+Genes':(core+groups['genes'],groups,50),'Clinical':(clin,{**groups,'genes':[],'muts':[]},50)}
rows=[]
for fi,(r,k,tri,tei) in enumerate(folds):
    for ep,ev in R.EP.items():
        valid=df[ev].notna().values; tr,te=tri[valid[tri]],tei[valid[tei]]
        y=Surv.from_arrays(df[ev].fillna(0).astype(bool).values, df.os_time.values)
        for fsn,(feats,g,ng) in fs.items():
            p=M.FoldPreprocessor(feats,groups,ng).fit(df.iloc[tr],y[tr]); Xtr,Xte=p.transform(df.iloc[tr]),p.transform(df.iloc[te])
            for mn,mf in L.items():
                rows.append(dict(repeat=r,fold=k,endpoint=ep,feature_set=fsn,model=mn,**R.metrics_simple(y[tr],y[te],R.fit_eval(mf,Xtr,y[tr],Xte,y[te]))))
    pd.DataFrame(rows).to_csv(sys.argv[2]+'/R16_core_incremental_per_fold.csv',index=False)
    log(f'fold {fi+1}/15 done')
