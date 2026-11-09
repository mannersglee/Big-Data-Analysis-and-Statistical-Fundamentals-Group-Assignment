"""Holdout prediction and exploratory propensity weighting for the three papers."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CONFIG = [
 ('01_数字鸿沟与老年健康','good_health','internet',['internet','age','female','residence'],['residence'],'classification',['age','female','residence']),
 ('02_家庭背景与教育公平','math_pct','computer_internet',['family_economy','books','computer_internet'],['family_economy','books'],'regression',['family_economy','books']),
 ('03_流动人口城市融入','local_identity','local_ties',['years_local','local_ties','age','female','education'],['education'],'classification',['years_local','age','female','education'])]

def transformer(features, categorical):
 numeric=[x for x in features if x not in categorical]
 return ColumnTransformer([
  ('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),
  ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical)],sparse_threshold=0)

for dirname,outcome,treatment,features,categorical,task,confounders in CONFIG:
 out=ROOT/dirname/'output'; data=pd.read_csv(out/'analysis_sample.csv'); X=data[features]; y=data[outcome].astype(float)
 train,test=train_test_split(np.arange(len(data)),test_size=.2,random_state=20261109,stratify=y if task=='classification' else None)
 estimators=[('constant',None),('linear',LogisticRegression(max_iter=1000) if task=='classification' else Ridge(alpha=10)),
  ('hist_gradient_boosting',HistGradientBoostingClassifier(max_iter=150,learning_rate=.05,max_leaf_nodes=15,l2_regularization=10,random_state=20261109) if task=='classification' else HistGradientBoostingRegressor(max_iter=150,learning_rate=.05,max_leaf_nodes=15,l2_regularization=10,random_state=20261109))]
 scores=[]; predictions=[]
 for name,est in estimators:
  if est is None: pred=np.repeat(y.iloc[train].mean(),len(test))
  else:
   pipe=Pipeline([('preprocess',transformer(features,categorical)),('model',est)])
   pipe.fit(X.iloc[train],y.iloc[train]);pred=pipe.predict_proba(X.iloc[test])[:,1] if task=='classification' else pipe.predict(X.iloc[test])
  row={'model':name,'train_n':len(train),'test_n':len(test)}
  if task=='classification': row.update(auc=roc_auc_score(y.iloc[test],pred) if name!='constant' else .5,brier=brier_score_loss(y.iloc[test],pred),log_loss=float(-np.mean(y.iloc[test]*np.log(np.clip(pred,1e-12,1))+(1-y.iloc[test])*np.log(np.clip(1-pred,1e-12,1)))))
  else:row.update(rmse=mean_squared_error(y.iloc[test],pred)**.5,mae=mean_absolute_error(y.iloc[test],pred),r2=r2_score(y.iloc[test],pred))
  scores.append(row)
  predictions.extend({'model':name,'row_index':int(i),'observed':float(obs),'prediction':float(p)} for i,obs,p in zip(test,y.iloc[test],pred))
 pd.DataFrame(scores).to_csv(out/'holdout_models.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(predictions).to_csv(out/'holdout_predictions.csv',index=False,encoding='utf-8-sig')
 # Exploratory IPW; confidence interval treats estimated propensity as fixed.
 t=data[treatment].astype(int).to_numpy(); conf=data[confounders].copy()
 if 'residence' in conf: conf['residence']=conf['residence'].astype(str)
 Z=pd.get_dummies(conf,drop_first=True,dtype=float);Z=sm.add_constant(Z.astype(float))
 propensity=np.asarray(sm.Logit(t,Z).fit(disp=False).predict(Z)); clipped=np.clip(propensity,.05,.95); weights=t/clipped+(1-t)/(1-clipped)
 fit=sm.WLS(y.to_numpy(),sm.add_constant(t),weights=weights).fit(cov_type='HC3');ci=fit.conf_int()[1]
 diagnostic={'n':len(data),'treated_n':int(t.sum()),'control_n':int((1-t).sum()),'propensity_p01':float(np.quantile(propensity,.01)),'propensity_p99':float(np.quantile(propensity,.99)),'clipped_n':int(((propensity<.05)|(propensity>.95)).sum()),'effective_n_treated':float(weights[t==1].sum()**2/(weights[t==1]**2).sum()),'effective_n_control':float(weights[t==0].sum()**2/(weights[t==0]**2).sum()),'weighted_difference':float(fit.params[1]),'ci_low_hc3':float(ci[0]),'ci_high_hc3':float(ci[1]),'p_hc3':float(fit.pvalues[1])}
 (out/'ipw_diagnostics.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2))
 balance=[]
 for column in Z.columns.drop('const'):
  z=Z[column].to_numpy();den=np.sqrt((z[t==1].var()+z[t==0].var())/2)
  pre=abs(z[t==1].mean()-z[t==0].mean())/den if den else 0
  post=abs(np.average(z[t==1],weights=weights[t==1])-np.average(z[t==0],weights=weights[t==0]))/den if den else 0
  balance.append({'covariate':column,'smd_before':pre,'smd_after':post})
 pd.DataFrame(balance).to_csv(out/'ipw_balance.csv',index=False,encoding='utf-8-sig')
 print(dirname,scores,diagnostic)
