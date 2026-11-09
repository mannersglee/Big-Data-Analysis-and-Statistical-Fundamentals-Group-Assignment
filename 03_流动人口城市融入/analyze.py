"""CMDS 2017 C questionnaire: local identity, residence duration and local ties."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cmds_matplotlib')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/CMDS中国流动人口动态监测（STATA+SPSS）/2017 CMDS/2017 CMDS STATA数据-适用14及以上版本/2017_Migrant_C_desensitized.dta'
OUT = Path(__file__).resolve().parent / 'output'
OUT.mkdir(exist_ok=True)
cols = ['newID', 'q101c1y', 'q101b1', 'q101e1', 'q101m1y', 'Q309', 'Q313', 'Q314', 'Q503H', 'w_l']
raw = pd.read_stata(SOURCE, columns=cols)
assert raw.newID.is_unique
raw['age'] = (2017 - pd.to_numeric(raw.q101c1y, errors='coerce')).where(lambda x: x.between(15, 80))
raw['years_local'] = (2017 - pd.to_numeric(raw.q101m1y, errors='coerce')).where(lambda x: x.between(0, 70))
raw['duration_group'] = pd.cut(raw.years_local, bins=[-1, 2, 5, 10, 70], labels=['0–2年', '3–5年', '6–10年', '11年以上'])
raw['local_identity'] = raw.Q503H.map({'基本同意': 1, '完全同意': 1, '不同意': 0, '完全不同意': 0})
raw['local_ties'] = raw.Q309.isin(['其他本地人', '同乡（户口迁至本地）', '同乡（户口迁至本地与老家以来的其他地区）']).where(raw.Q309.notna()).astype(float)
raw['female'] = raw.q101b1.map({'男': 0, '女': 1, 'Ů': 1})
raw['education'] = raw.q101e1.astype(str).replace({'Сѧ': '小学'}).where(raw.q101e1.notna())
raw['stay_intent'] = raw.Q314.map({'是': 1, '否': 0})
raw['settle_intent'] = raw.Q313.map({'愿意': 1, '不愿意': 0})
fields = ['age', 'years_local', 'duration_group', 'local_identity', 'local_ties', 'female', 'education', 'stay_intent', 'settle_intent']
raw[fields].isna().sum().rename('missing_n').to_csv(OUT / 'missingness.csv', encoding='utf-8-sig')
data = raw.dropna(subset=['age', 'years_local', 'duration_group', 'local_identity', 'local_ties', 'female', 'education']).copy()
assert len(data) >= 3000
# Retain raw ancillary outcomes and weights for further descriptive work.
data[['newID'] + fields + ['w_l']].to_csv(OUT / 'analysis_sample.csv', index=False, encoding='utf-8-sig')
summary = data.groupby('duration_group', observed=True).agg(n=('newID', 'size'), identity_rate=('local_identity', 'mean'), local_ties_rate=('local_ties', 'mean'))
summary.to_csv(OUT / 'duration_summary.csv', encoding='utf-8-sig')
ties = data.groupby('local_ties').agg(n=('newID', 'size'), identity_rate=('local_identity', 'mean'))
ties.to_csv(OUT / 'ties_summary.csv', encoding='utf-8-sig')
model = smf.logit('local_identity ~ years_local + local_ties + age + female + C(education)', data=data).fit(disp=False)
model.summary2().tables[1].to_csv(OUT / 'logit_coefficients.csv', encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=(6, 4))
summary.identity_rate.mul(100).plot.bar(ax=ax, color='#2471a3')
ax.set_xticklabels(['0–2 years', '3–5 years', '6–10 years', '11+ years'], rotation=0)
ax.set_ylabel('Feels like a local (%)'); ax.set_xlabel('Years since current migration')
ax.set_title('CMDS 2017 C questionnaire')
fig.tight_layout(); fig.savefig(OUT / 'identity_by_duration.png', dpi=180); plt.close(fig)
print(f'Raw n={len(raw)}; complete-case n={len(data)}')
print(summary.to_string())
print(ties.to_string())
for x in ['years_local', 'local_ties']:
    print(f'{x}: OR={np.exp(model.params[x]):.3f}, p={model.pvalues[x]:.4g}')

# Report-ready distributions, in-sample fit, and standardized predictions.
from sklearn.metrics import roc_auc_score, brier_score_loss
n = len(data)
rows = [{'variable': name, 'level': 'continuous', 'n': n, 'percent': 100, 'mean': data[name].mean(), 'sd': data[name].std()} for name in ['age','years_local']]
for name in ['duration_group','local_identity','local_ties','female','education']:
    for level, count in data[name].value_counts().items():
        rows.append({'variable': name, 'level': str(level), 'n': int(count), 'percent': 100*count/n, 'mean': None, 'sd': None})
pd.DataFrame(rows).to_csv(OUT / 'variable_distribution.csv', index=False, encoding='utf-8-sig')
pred = model.predict(data)
pd.DataFrame([{'n': n, 'event_rate': data.local_identity.mean(), 'pseudo_r2_mcfadden': model.prsquared,
               'aic': model.aic, 'auc_in_sample': roc_auc_score(data.local_identity, pred),
               'brier_in_sample': brier_score_loss(data.local_identity, pred)}]).to_csv(OUT / 'model_fit.csv', index=False, encoding='utf-8-sig')
scenarios = []
for name, values in [('years_local',[2,5,10,15]),('local_ties',[0,1])]:
    for value in values:
        frame = data.copy(); frame[name] = value
        scenarios.append({'variable': name, 'scenario': value, 'mean_predicted_probability': model.predict(frame).mean()})
pd.DataFrame(scenarios).to_csv(OUT / 'standardized_predictions.csv', index=False, encoding='utf-8-sig')
cal = pd.DataFrame({'observed': data.local_identity, 'predicted': pred})
cal['decile'] = pd.qcut(cal.predicted, 10, labels=False, duplicates='drop') + 1
cal.groupby('decile').agg(n=('observed','size'), observed_rate=('observed','mean'), predicted_rate=('predicted','mean')).to_csv(OUT / 'calibration.csv', encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=(5,4))
c = pd.read_csv(OUT / 'calibration.csv')
ax.plot(c.predicted_rate, c.observed_rate, 'o-', label='Observed by predicted decile')
ax.plot([0,1], [0,1], '--', color='gray', label='Perfect calibration')
ax.set(xlim=(0.3,1), ylim=(0.3,1), xlabel='Mean fitted probability', ylabel='Observed event rate', title='In-sample calibration')
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / 'calibration.png', dpi=180); plt.close(fig)
