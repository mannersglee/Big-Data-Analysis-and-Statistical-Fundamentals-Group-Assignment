"""CGSS 2021: personal income position and perceived social fairness."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cgss_matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/CGSS数据合集/CGSS2021/cgss2021stata/CGSS2021.dta'
OUT=Path(__file__).resolve().parent/'output';OUT.mkdir(exist_ok=True)
cols=['A2','A3_1','A7a','A8a','A35','A43e','P2_2','weight_raking']
raw=pd.read_stata(SOURCE,columns=cols,convert_categoricals=False)
raw['age']=(2021-pd.to_numeric(raw.A3_1,errors='coerce')).where(lambda x:x.between(18,100))
raw['female']=raw.A2.map({1:0,2:1})
raw['income']=pd.to_numeric(raw.A8a,errors='coerce').where(lambda x:x.between(0,9_999_996))
raw['fairness_score']=raw.A35.where(raw.A35.isin([1,2,3,4,5]))
raw['fair']=raw.fairness_score.isin([4,5]).where(raw.fairness_score.notna()).astype(float)
raw['education']=pd.cut(raw.A7a,bins=[0,3,4,8,13],labels=['小学及以下','初中','高中/中专','大专及以上'])
raw['subjective_status']=raw.A43e.where(raw.A43e.isin([1,2,3,4,5]))
raw['gap_responsibility']=raw.P2_2.where(raw.P2_2.isin([1,2,3,4,5]))
fields=['age','female','income','fairness_score','fair','education','subjective_status','gap_responsibility']
raw[fields].isna().sum().rename('missing_n').to_csv(OUT/'missingness.csv',encoding='utf-8-sig')
data=raw.dropna(subset=['age','female','income','fair','education']).copy()
assert len(data)>=3000
# Income quintiles are based on positive-income respondents, with a separate zero-income group.
positive=data.loc[data.income.gt(0),'income']
cutoffs=positive.quantile([.2,.4,.6,.8]).to_numpy()
labels=['零收入','正收入Q1','正收入Q2','正收入Q3','正收入Q4','正收入Q5']
def income_group(x):
    if x==0:return '零收入'
    return labels[1+int(np.searchsorted(cutoffs,x,side='left'))]
data['income_group']=pd.Categorical(data.income.map(income_group),categories=labels,ordered=True)
data['log_income']=np.log1p(data.income)
data['income_for_model']=data.income.clip(upper=data.income.quantile(.995))
data['log_income_model']=np.log1p(data.income_for_model)
# Discrete Gini for the observed valid-income sample, zeros included.
values=np.sort(data.income.to_numpy(dtype=float));n=len(values)
gini=(2*np.dot(np.arange(1,n+1),values)/(n*values.sum())-(n+1)/n) if values.sum()>0 else np.nan
pd.DataFrame([{'raw_n':len(raw),'complete_case_n':n,'complete_case_rate':n/len(raw),'income_median':data.income.median(),'income_mean':data.income.mean(),'income_p99_5':data.income.quantile(.995),'sample_gini_unweighted':gini,'positive_income_quintile_cutoffs':';'.join(map(str,cutoffs))}]).to_csv(OUT/'sample_summary.csv',index=False,encoding='utf-8-sig')
data[['age','female','income','fairness_score','fair','education','subjective_status','gap_responsibility','income_group','log_income_model']].to_csv(OUT/'analysis_sample.csv',index=False,encoding='utf-8-sig')
summary=data.groupby('income_group',observed=True).agg(n=('fair','size'),median_income=('income','median'),fair_rate=('fair','mean'),mean_fairness_score=('fairness_score','mean'))
summary.to_csv(OUT/'income_group_summary.csv',encoding='utf-8-sig')
dist=[]
for name in ['age','income','fairness_score']:
 dist.append({'variable':name,'level':'continuous','n':n,'percent':100,'mean':data[name].mean(),'sd':data[name].std()})
for name in ['female','fair','education','income_group']:
 for level,count in data[name].value_counts().items():dist.append({'variable':name,'level':str(level),'n':int(count),'percent':100*count/n,'mean':None,'sd':None})
pd.DataFrame(dist).to_csv(OUT/'variable_distribution.csv',index=False,encoding='utf-8-sig')
model=smf.logit('fair ~ log_income_model + age + female + C(education)',data=data).fit(disp=False)
model.summary2().tables[1].to_csv(OUT/'logit_coefficients.csv',encoding='utf-8-sig')
# Survey-position contrast: income group categories rather than assuming log-linearity.
group_model=smf.logit('fair ~ C(income_group) + age + female + C(education)',data=data).fit(disp=False)
group_model.summary2().tables[1].to_csv(OUT/'group_model_coefficients.csv',encoding='utf-8-sig')
scenarios=[]
for income in [0,10000,30000,60000,100000]:
 frame=data.copy();frame['log_income_model']=np.log1p(income)
 scenarios.append({'annual_income_yuan':income,'mean_predicted_fair_probability':model.predict(frame).mean()})
pd.DataFrame(scenarios).to_csv(OUT/'standardized_predictions.csv',index=False,encoding='utf-8-sig')
# Fixed holdout, training-only preprocessing, same features for both learners.
X=data[['log_income_model','age','female','education']];y=data.fair.astype(int)
train,test=train_test_split(np.arange(n),test_size=.2,random_state=20261109,stratify=y)
pre=ColumnTransformer([('num',StandardScaler(),['log_income_model','age','female']),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['education'])],sparse_threshold=0)
models=[('constant',None),('logistic',LogisticRegression(max_iter=1000)),('hist_gradient_boosting',HistGradientBoostingClassifier(max_iter=150,learning_rate=.05,max_leaf_nodes=15,l2_regularization=10,random_state=20261109))]
results=[]
for name,est in models:
 if est is None:p=np.repeat(y.iloc[train].mean(),len(test))
 else:
  pipe=Pipeline([('preprocess',pre),('model',est)]);pipe.fit(X.iloc[train],y.iloc[train]);p=pipe.predict_proba(X.iloc[test])[:,1]
 results.append({'model':name,'train_n':len(train),'test_n':len(test),'auc':roc_auc_score(y.iloc[test],p) if name!='constant' else .5,'brier':brier_score_loss(y.iloc[test],p),'log_loss':log_loss(y.iloc[test],p)})
pd.DataFrame(results).to_csv(OUT/'holdout_models.csv',index=False,encoding='utf-8-sig')
fig,ax=plt.subplots(figsize=(7,4))
summary.fair_rate.mul(100).plot.bar(ax=ax,color='#2471a3')
ax.set_xticklabels(['Zero','Positive Q1','Q2','Q3','Q4','Q5'],rotation=0)
ax.set_ylabel('Perceives society as fair (%)');ax.set_xlabel('Annual personal income group')
ax.set_title('CGSS 2021 | unweighted complete cases')
fig.tight_layout();fig.savefig(OUT/'fairness_by_income.png',dpi=180);plt.close(fig)
print('raw',len(raw),'complete',n,'gini',gini)
print(summary.to_string());print(model.summary2().tables[1].loc['log_income_model'].to_string())
print(pd.DataFrame(results).to_string(index=False))
# Sensitivity analyses: positive earners only, and adjustment for subjective status.
sensitivity=[]
for name,frame,formula in [
 ('primary',data,'fair ~ log_income_model + age + female + C(education)'),
 ('positive_income_only',data.loc[data.income.gt(0)].copy(),'fair ~ log_income_model + age + female + C(education)'),
 ('with_subjective_status',data.dropna(subset=['subjective_status']).copy(),'fair ~ log_income_model + age + female + C(education) + C(subjective_status)')]:
 fitted=smf.logit(formula,data=frame).fit(disp=False)
 ci=fitted.conf_int().loc['log_income_model']
 sensitivity.append({'specification':name,'n':len(frame),'log_income_coef':fitted.params['log_income_model'],'odds_ratio_per_log_unit':np.exp(fitted.params['log_income_model']),'ci_low_or':np.exp(ci.iloc[0]),'ci_high_or':np.exp(ci.iloc[1]),'p':fitted.pvalues['log_income_model']})
pd.DataFrame(sensitivity).to_csv(OUT/'sensitivity_models.csv',index=False,encoding='utf-8-sig')
clipped_values=np.sort(data.income.clip(upper=data.income.quantile(.995)).to_numpy(dtype=float))
clipped_gini=2*np.dot(np.arange(1,n+1),clipped_values)/(n*clipped_values.sum())-(n+1)/n
pd.DataFrame([{'gini_raw_income':gini,'gini_top_0_5_percent_winsorized':clipped_gini,'median_income':data.income.median(),'mean_income':data.income.mean()}]).to_csv(OUT/'inequality_sensitivity.csv',index=False,encoding='utf-8-sig')
