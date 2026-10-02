import sys, numpy as np, pandas as pd
sys.path.insert(0,'.')
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.metrics import roc_auc_score
from wildfirevuln.model import Design, CATS
from wildfirevuln import taxonomy as T, product
d=pd.read_csv('data/processed/dins_residential.csv')
rng=np.random.default_rng(0)
def draw_impute(df, ref):
    df=df.copy()
    for c in CATS:
        known=ref[c][ref[c]!='unknown']; vc=known.value_counts(normalize=True)
        m=df[c]=='unknown'; df.loc[m,c]=rng.choice(vc.index, m.sum(), p=vc.values)
    return df
folds=['Camp 2018','Eaton 2025','Palisades 2025','CZU Lightning Cmplx 2020','Caldor 2021','Glass 2020','Dixie 2021','Park 2024']
sp=product.spec()
res={}
for ev in folds:
    tr,te=d[d.event!=ev],d[d.event==ev]; y=tr.burnt; yt=te.burnt
    def gbm(cats,num,imp,data_tr=tr,data_te=te):
        a,b=(draw_impute(data_tr,tr),draw_impute(data_te,tr)) if imp=='draw' else (data_tr,data_te)
        D=Design(cats=cats,use_num=num).fit(a)
        g=H(max_iter=300,learning_rate=0.06,random_state=0).fit(D.transform(a),y)
        return roc_auc_score(yt,g.predict_proba(D.transform(b))[:,1])
    r={}
    r['gbm_all_softimp']=gbm(CATS,True,'soft')
    r['gbm_all_draw']=gbm(CATS,True,'draw')
    r['gbm_survey_draw_nospace']=gbm(CATS,False,'draw')
    r['gbm_space_only']=gbm(['struct'],True,'soft')
    r['gbm_spec_draw_space']=gbm(list(sp),True,'draw',product.apply(tr,sp),product.apply(te,sp))
    res[ev]=r; print(ev,{k:round(v,3) for k,v in r.items()},flush=True)
print(pd.DataFrame(res).T.mean().round(3))
