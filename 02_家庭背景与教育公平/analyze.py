"""CEPS 2014-15 follow-up: family resources and standardized math scores."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ceps_matplotlib')
import zipfile
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/CEPS数据合集/CEPS14-15追访/学生问卷数据（中文） .zip'
OUT = Path(__file__).resolve().parent / 'output'
OUT.mkdir(exist_ok=True)
columns = ['ids', 'schids', 'w2mat', 'w2upmat', 'w2a09', 'w2a11', 'w2a12', 'w2sweight']
with zipfile.ZipFile(SOURCE) as archive:
    with archive.open('cepsw2studentCN.dta') as stream:
        raw = pd.read_stata(stream, columns=columns)

def chinese(series):
    def repair(value):
        if not isinstance(value, str): return value
        try: return value.encode('latin1').decode('gbk')
        except (UnicodeError, ValueError): return value
    return series.map(repair)

for name in ['w2a09', 'w2a11', 'w2a12']:
    raw[name] = chinese(raw[name])
raw['math_pct'] = (pd.to_numeric(raw.w2mat, errors='coerce') / pd.to_numeric(raw.w2upmat, errors='coerce') * 100).where(lambda x: x.between(0, 100))
raw['family_economy'] = pd.Categorical(raw.w2a09, categories=['非常困难', '比较困难', '中等', '比较富裕', '很富裕'], ordered=True)
raw['books'] = pd.Categorical(raw.w2a11, categories=['很少', '比较少', '一般', '比较多', '很多'], ordered=True)
raw['computer_internet'] = raw.w2a12.map({'有电脑和网络': 1, '有电脑，无网络': 0, '都没有': 0})
raw['weight'] = pd.to_numeric(raw.w2sweight, errors='coerce').where(lambda x: x.gt(0))
raw[['math_pct', 'family_economy', 'books', 'computer_internet', 'weight']].isna().sum().rename('missing_n').to_csv(OUT / 'missingness.csv', encoding='utf-8-sig')
data = raw.dropna(subset=['math_pct', 'family_economy', 'books', 'computer_internet']).copy()
assert len(data) >= 3000, f'有效样本不足3000: {len(data)}'
data[['ids', 'schids', 'math_pct', 'family_economy', 'books', 'computer_internet', 'weight']].to_csv(OUT / 'analysis_sample.csv', index=False, encoding='utf-8-sig')
summary = data.groupby('family_economy', observed=True).agg(n=('ids', 'size'), mean_math_pct=('math_pct', 'mean'), internet_access_rate=('computer_internet', 'mean'))
summary.to_csv(OUT / 'group_summary.csv', encoding='utf-8-sig')
model = smf.ols('math_pct ~ C(family_economy) + C(books) + computer_internet', data=data).fit(cov_type='HC3')
model.summary2().tables[1].to_csv(OUT / 'ols_coefficients.csv', encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=(7, 4))
summary.mean_math_pct.plot.bar(ax=ax, color='#2471a3')
ax.set_xticklabels(['Very difficult', 'Difficult', 'Average', 'Well-off', 'Very well-off'], rotation=20, ha='right')
ax.set_ylabel('Math score / full score (%)'); ax.set_xlabel('')
ax.set_title('CEPS 2014–15 | family economy')
fig.tight_layout(); fig.savefig(OUT / 'math_by_family_economy.png', dpi=180); plt.close(fig)
print(f'Raw n={len(raw)}; complete-case n={len(data)}')
print(summary.to_string())

# Report-ready distributions, in-sample fit, and standardized predictions.
import numpy as np
n = len(data)
rows = [{'variable': 'math_pct', 'level': 'continuous', 'n': n, 'percent': 100, 'mean': data.math_pct.mean(), 'sd': data.math_pct.std()}]
for name in ['family_economy', 'books', 'computer_internet']:
    for level, count in data[name].value_counts().items():
        rows.append({'variable': name, 'level': str(level), 'n': int(count), 'percent': 100*count/n, 'mean': None, 'sd': None})
pd.DataFrame(rows).to_csv(OUT / 'variable_distribution.csv', index=False, encoding='utf-8-sig')
pred = model.predict(data)
pd.DataFrame([{'n': n, 'mean_outcome': data.math_pct.mean(), 'r_squared': model.rsquared,
               'adjusted_r_squared': model.rsquared_adj, 'aic': model.aic,
               'rmse_in_sample': np.sqrt(np.mean((data.math_pct-pred)**2)),
               'mae_in_sample': np.mean(np.abs(data.math_pct-pred))}]).to_csv(OUT / 'model_fit.csv', index=False, encoding='utf-8-sig')
scenarios = []
for name, values in [('family_economy', data.family_economy.cat.categories), ('computer_internet', [0,1])]:
    for value in values:
        frame = data.copy(); frame[name] = value
        scenarios.append({'variable': name, 'scenario': str(value), 'mean_predicted_math_pct': model.predict(frame).mean()})
pd.DataFrame(scenarios).to_csv(OUT / 'standardized_predictions.csv', index=False, encoding='utf-8-sig')
fit = pd.DataFrame({'observed': data.math_pct, 'predicted': pred})
fit['quintile'] = pd.qcut(fit.predicted, 5, labels=False, duplicates='drop') + 1
fit.groupby('quintile').agg(n=('observed','size'), observed_mean=('observed','mean'), predicted_mean=('predicted','mean')).to_csv(OUT / 'fitted_groups.csv', encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=(5,4))
f = pd.read_csv(OUT / 'fitted_groups.csv')
ax.scatter(f.predicted_mean, f.observed_mean)
lo, hi = min(f.predicted_mean.min(),f.observed_mean.min()), max(f.predicted_mean.max(),f.observed_mean.max())
ax.plot([lo,hi],[lo,hi],'--',color='gray')
ax.set(xlabel='Mean fitted math score (%)', ylabel='Observed mean math score (%)', title='In-sample fitted groups')
fig.tight_layout(); fig.savefig(OUT / 'fitted_groups.png', dpi=180); plt.close(fig)
