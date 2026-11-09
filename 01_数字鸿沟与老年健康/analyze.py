"""CHARLS 2020: internet use and self-reported health among adults aged 60+."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/charls_matplotlib')
import zipfile
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/CHARLS数据合集/CHARLS 2020年全国追踪调查/所有数据的压缩包CHARLS2020r.zip'
OUT = Path(__file__).resolve().parent / 'output'
OUT.mkdir(exist_ok=True)

with zipfile.ZipFile(SOURCE) as archive:
    def read(name, columns):
        with archive.open(name + '.dta') as stream:
            return pd.read_stata(stream, columns=columns, convert_categoricals=False)
    health = read('Health_Status_and_Functioning', ['ID', 'xrage', 'da001', 'da040'])
    demographic = read('Demographic_Background', ['ID', 'xrgender', 'ba008'])

assert health.ID.is_unique and demographic.ID.is_unique
raw = health.merge(demographic, on='ID', how='left', validate='one_to_one')
raw = raw.loc[raw.xrage.ge(60)].copy()
raw['age'] = raw.xrage.where(raw.xrage.between(60, 110))
raw['internet'] = raw.da040.map({1: 1, 2: 0})
raw['health_score'] = raw.da001.where(raw.da001.isin([1, 2, 3, 4, 5]))
raw['good_health'] = raw.health_score.isin([1, 2]).where(raw.health_score.notna()).astype(float)
raw['female'] = raw.xrgender.map({1: 0, 2: 1})
raw['residence'] = raw.ba008.where(raw.ba008.isin([1, 2, 3])).astype('category')
cols = ['age', 'internet', 'health_score', 'good_health', 'female', 'residence']
raw[cols].isna().sum().rename('missing_n').to_csv(OUT / 'missingness.csv', encoding='utf-8-sig')
data = raw.dropna(subset=['age', 'internet', 'good_health', 'female', 'residence']).copy()
assert len(data) >= 3000, f'有效样本不足3000: {len(data)}'
data[['ID'] + cols].to_csv(OUT / 'analysis_sample.csv', index=False, encoding='utf-8-sig')
summary = data.groupby('internet').agg(n=('ID', 'size'), mean_age=('age', 'mean'), good_health_rate=('good_health', 'mean'), mean_health_score=('health_score', 'mean'))
summary.to_csv(OUT / 'group_summary.csv', encoding='utf-8-sig')
model = smf.logit('good_health ~ internet + age + female + C(residence)', data=data).fit(disp=False)
model.summary2().tables[1].to_csv(OUT / 'logit_coefficients.csv', encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=(6, 4))
summary.good_health_rate.mul(100).plot.bar(ax=ax, color=['#8395a7', '#2471a3'])
ax.set_xticklabels(['No internet', 'Uses internet'], rotation=0)
ax.set_ylabel('Good self-rated health (%)')
ax.set_xlabel('')
ax.set_title('CHARLS 2020 | age 60+')
fig.tight_layout(); fig.savefig(OUT / 'health_by_internet.png', dpi=180); plt.close(fig)
print(f'Raw age 60+ n={len(raw)}; complete-case n={len(data)}')
print(summary.to_string())
print('Adjusted internet odds ratio:', round(float(__import__('numpy').exp(model.params['internet'])), 3))

# Report-ready distributions, in-sample fit, and standardized predictions.
from sklearn.metrics import roc_auc_score, brier_score_loss
n = len(data)
distribution = pd.DataFrame([
    {'variable': 'age', 'level': 'continuous', 'n': n, 'percent': 100, 'mean': data.age.mean(), 'sd': data.age.std()},
    *({'variable': name, 'level': str(level), 'n': int(count), 'percent': 100*count/n, 'mean': None, 'sd': None}
      for name in ['internet', 'good_health', 'female', 'residence']
      for level, count in data[name].value_counts().sort_index().items())
])
distribution.to_csv(OUT / 'variable_distribution.csv', index=False, encoding='utf-8-sig')
pred = model.predict(data)
pd.DataFrame([{'n': n, 'event_rate': data.good_health.mean(), 'pseudo_r2_mcfadden': model.prsquared,
               'aic': model.aic, 'auc_in_sample': roc_auc_score(data.good_health, pred),
               'brier_in_sample': brier_score_loss(data.good_health, pred)}]).to_csv(OUT / 'model_fit.csv', index=False, encoding='utf-8-sig')
scenarios = []
for value in [0, 1]:
    frame = data.copy(); frame['internet'] = value
    scenarios.append({'scenario': f'internet={value}', 'mean_predicted_probability': model.predict(frame).mean()})
pd.DataFrame(scenarios).to_csv(OUT / 'standardized_predictions.csv', index=False, encoding='utf-8-sig')
cal = pd.DataFrame({'observed': data.good_health, 'predicted': pred})
cal['decile'] = pd.qcut(cal.predicted, 10, labels=False, duplicates='drop') + 1
cal.groupby('decile').agg(n=('observed','size'), observed_rate=('observed','mean'), predicted_rate=('predicted','mean')).to_csv(OUT / 'calibration.csv', encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=(5,4))
c = pd.read_csv(OUT / 'calibration.csv')
ax.plot(c.predicted_rate, c.observed_rate, 'o-', label='Observed by predicted decile')
ax.plot([0,1], [0,1], '--', color='gray', label='Perfect calibration')
ax.set(xlim=(0,0.5), ylim=(0,0.5), xlabel='Mean fitted probability', ylabel='Observed event rate', title='In-sample calibration')
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / 'calibration.png', dpi=180); plt.close(fig)
