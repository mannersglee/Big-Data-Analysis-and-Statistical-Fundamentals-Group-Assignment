"""Publication-style figures for the four research papers.

Run after each topic's analyze.py and src/advanced_analysis.py. PNG and SVG are
written together; individual records stay in ignored output/analysis_sample.csv.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/academic_figures_matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
NAVY='#24374A'; TEAL='#2D7180'; BLUE='#6386A1'; ORANGE='#B87549'; GRID='#E5E9EC'; GRAY='#64717D'
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','font.size':10,'axes.titlesize':13,'axes.titleweight':'semibold','axes.labelsize':10,'xtick.color':NAVY,'ytick.color':NAVY,'text.color':NAVY,'axes.labelcolor':NAVY,'axes.edgecolor':GRID,'grid.color':GRID,'grid.linewidth':.7,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})

def read(folder,name): return pd.read_csv(ROOT/folder/'output'/name,encoding='utf-8-sig')
def save(fig,folder,name):
 out=ROOT/folder/'output'
 fig.savefig(out/f'{name}.png',dpi=220,bbox_inches='tight',pad_inches=.15)
 fig.savefig(out/f'{name}.svg',bbox_inches='tight',pad_inches=.15)
 plt.close(fig)
def axes(title,ylabel='',xlabel='',figsize=(7,4.3)):
 fig,ax=plt.subplots(figsize=figsize)
 ax.set_title(title,loc='left',pad=16)
 ax.set_ylabel(ylabel);ax.set_xlabel(xlabel)
 ax.grid(axis='y',zorder=0);ax.set_axisbelow(True)
 return fig,ax
def note(ax,text): ax.text(0,-.25,text,transform=ax.transAxes,fontsize=8,color=GRAY,va='top')
def wilson(success,n):
 p=success/n;z=1.96;den=1+z*z/n
 center=(p+z*z/(2*n))/den
 delta=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return center-delta,center+delta

def rate_chart(folder,name,labels,counts,rates,title,footnote,ylim=None):
 n=np.asarray(counts,dtype=float);p=np.asarray(rates,dtype=float)
 low,high=wilson(p*n,n)
 fig,ax=axes(title,'比例（%）')
 x=np.arange(len(labels));colors=[BLUE]*len(labels);colors[-1]=TEAL
 ax.bar(x,p*100,width=.62,color=colors,zorder=2)
 ax.errorbar(x,p*100,yerr=np.vstack(((p-low)*100,(high-p)*100)),fmt='none',ecolor=NAVY,capsize=4,lw=1.1,zorder=3)
 for xi,pi,ni in zip(x,p,n):ax.text(xi,pi*100+5.0,f'{pi:.1%}\nn={int(ni):,}',ha='center',va='bottom',fontsize=8.5)
 ax.set_xticks(x,labels)
 ax.set_ylim(0,ylim or min(100,max(p*100)+17))
 note(ax,footnote);fig.tight_layout();save(fig,folder,name)

def forest(folder,name,coef_file,terms,title,logit=True):
 df=read(folder,coef_file).set_index('Unnamed: 0')
 fig,ax=axes(title,'','',figsize=(7,3.6));ax.grid(axis='x');ax.grid(axis='y',visible=False)
 y=np.arange(len(terms))[::-1]
 for yi,(term,label) in zip(y,terms):
  row=df.loc[term];v=float(row['Coef.']);lo=float(row['[0.025']);hi=float(row['0.975]'])
  if logit:v,lo,hi=np.exp([v,lo,hi])
  ax.errorbar(v,yi,xerr=np.array([[v-lo],[hi-v]]),fmt='o',color=TEAL,ecolor=NAVY,capsize=4,markersize=6,lw=1.2)
  ax.text(hi+(0.03 if logit else .5),yi,f'{v:.2f} [{lo:.2f}, {hi:.2f}]',va='center',fontsize=8.5,color=GRAY)
 ax.set_yticks(y,[label for _,label in terms]);ax.axvline(1 if logit else 0,color=ORANGE,lw=1,ls='--')
 ax.set_xlabel('优势比（95% 置信区间）' if logit else '成绩百分点差（95% 置信区间）')
 ax.set_xlim(left=max(0,ax.get_xlim()[0]) if logit else ax.get_xlim()[0])
 note(ax,'模型条件关联；区间基于当前模型，未作因果解释。')
 fig.tight_layout();save(fig,folder,name)

def holdout(folder,metric,title,better='lower'):
 df=read(folder,'holdout_models.csv')
 labels={'constant':'常数基线','linear':'线性模型','logistic':'Logistic','hist_gradient_boosting':'梯度提升'}
 vals=df[metric].to_numpy(float);x=np.arange(len(df))
 best=np.argmin(vals) if better=='lower' else np.argmax(vals)
 fig,ax=axes(title,metric.upper() if metric!='brier' else 'Brier 分数',figsize=(6.5,4.1))
 ax.bar(x,vals,color=[TEAL if i==best else BLUE for i in x],width=.58)
 for i,v in enumerate(vals):ax.text(i,v+max(vals)*.015,f'{v:.3f}',ha='center',fontsize=9)
 ax.set_xticks(x,[labels[m] for m in df.model]);ax.set_ylim(0,max(vals)*1.17)
 note(ax,f'固定 80/20 划分；仅测试集，n={int(df.test_n.iloc[0]):,}；越{"低" if better=="lower" else "高"}越好。')
 fig.tight_layout();save(fig,folder,'holdout_performance')

def calibration(folder):
 df=read(folder,'calibration.csv')
 fig,ax=axes('样本内概率校准','实际事件率','平均预测概率',figsize=(5.7,4.7))
 ax.grid(axis='both');ax.plot([0,1],[0,1],ls='--',color=GRAY,lw=1,label='理想校准')
 ax.plot(df.predicted_rate,df.observed_rate,'o-',color=TEAL,lw=1.6,label='预测值十分位')
 low=max(0,min(df.predicted_rate.min(),df.observed_rate.min())-.08);high=min(1,max(df.predicted_rate.max(),df.observed_rate.max())+.08)
 ax.set(xlim=(low,high),ylim=(low,high));ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
 ax.legend(frameon=False,loc='upper left',fontsize=8);note(ax,'同一样本拟合与评价；不代表样本外校准。')
 fig.tight_layout();save(fig,folder,'calibration')

def fitted_groups(folder):
 df=read(folder,'fitted_groups.csv')
 fig,ax=axes('样本内分组拟合','各组观察均分（%）','各组拟合均分（%）',figsize=(5.7,4.7))
 ax.grid(axis='both');lo=min(df.predicted_mean.min(),df.observed_mean.min())-3;hi=max(df.predicted_mean.max(),df.observed_mean.max())+3
 ax.plot([lo,hi],[lo,hi],ls='--',color=GRAY,lw=1,label='完全一致')
 ax.plot(df.predicted_mean,df.observed_mean,'o',color=TEAL,markersize=7,label='按拟合值分五组')
 for _,row in df.iterrows():ax.annotate(f"Q{int(row['quintile'])}",(row.predicted_mean,row.observed_mean),xytext=(5,5),textcoords='offset points',fontsize=8)
 ax.set(xlim=(lo,hi),ylim=(lo,hi));ax.legend(frameon=False,fontsize=8)
 note(ax,'同一样本拟合与分组；不代表个体预测误差或外部验证。')
 fig.tight_layout();save(fig,folder,'fitted_groups')

f='01_数字鸿沟与老年健康';d=read(f,'group_summary.csv')
rate_chart(f,'health_by_internet',['未上网','上网'],d.n,d.good_health_rate,'互联网使用与健康较好比例','误差线：未加权 Wilson 95% 区间；未纳入复杂抽样设计。',ylim=40)
calibration(f)
forest(f,'model_effects','logit_coefficients.csv',[('internet','上网（是 vs 否）'),('age','年龄（每岁）'),('female','女性（vs 男性）')],'自评健康模型：条件优势比')
holdout(f,'brier','健康结局：测试集预测误差')

f='02_家庭背景与教育公平';d=read(f,'group_summary.csv');sample=read(f,'analysis_sample.csv')
labels=['非常困难','比较困难','中等','比较富裕','很富裕']; means=[];se=[];ns=[]
for label in labels:
 x=sample.loc[sample.family_economy==label,'math_pct'].dropna();means.append(x.mean());ns.append(len(x));se.append(x.std(ddof=1)/np.sqrt(len(x)))
fig,ax=axes('家庭经济条件与数学成绩','数学成绩占满分比例（%）',figsize=(7.3,4.6));x=np.arange(5)
ax.bar(x,means,color=[BLUE]*4+[TEAL],width=.62)
ax.errorbar(x,means,yerr=1.96*np.asarray(se),fmt='none',ecolor=NAVY,capsize=4,lw=1.1)
for i,(m,n) in enumerate(zip(means,ns)):ax.text(i,m+1.96*se[i]+2.5,f'{m:.1f}\nn={n:,}',ha='center',fontsize=8.5)
ax.set_xticks(x,labels);ax.set_ylim(0,90)
note(ax,'误差线：未加权均值 ± 1.96×标准误；未纳入学校聚类与复杂抽样。')
fig.tight_layout();save(fig,f,'math_by_family_economy')
fitted_groups(f)
forest(f,'model_effects','ols_coefficients.csv',[('computer_internet','家有电脑和网络'),('C(books)[T.比较多]','藏书比较多（vs 很少）'),('C(books)[T.很多]','藏书很多（vs 很少）')],'数学成绩模型：条件分差',logit=False)
holdout(f,'rmse','数学成绩：测试集预测误差')

f='03_流动人口城市融入';d=read(f,'duration_summary.csv')
rate_chart(f,'identity_by_duration',['0–2年','3–5年','6–10年','11年以上'],d.n,d.identity_rate,'现居地年限与本地身份认同','误差线：未加权 Wilson 95% 区间；未纳入复杂抽样设计。',ylim=100)
calibration(f)
forest(f,'model_effects','logit_coefficients.csv',[('years_local','现居地年限（每年）'),('local_ties','本地交往（是 vs 否）'),('age','年龄（每岁）'),('female[T.1]','女性（vs 男性）')],'本地身份认同模型：条件优势比')
holdout(f,'brier','身份认同：测试集预测误差')

f='04_收入差距与公平感';d=read(f,'income_group_summary.csv')
rate_chart(f,'fairness_by_income',['零收入','Q1','Q2','Q3','Q4','Q5'],d.n,d.fair_rate,'个人收入位置与社会公平认同','正收入按样本五分位分组；误差线为未加权 Wilson 95% 区间。',ylim=85)
forest(f,'model_effects','logit_coefficients.csv',[('log_income_model','对数收入（每单位）'),('age','年龄（每岁）'),('female','女性（vs 男性）')],'社会公平感模型：条件优势比')
holdout(f,'brier','公平感：测试集预测误差')
sample=read(f,'analysis_sample.csv');income=np.sort(sample.income.to_numpy(dtype=float));cum=np.r_[0,np.cumsum(income)/income.sum()];pop=np.linspace(0,1,len(cum))
fig,ax=axes('样本个人收入的洛伦兹曲线','累计收入占比','累计受访者占比',figsize=(5.7,5.2))
ax.grid(axis='both');ax.plot(pop,pop,ls='--',color=GRAY,lw=1,label='完全均等')
ax.plot(pop,cum,color=TEAL,lw=2,label='CGSS 2021 完整案例')
ax.set(xlim=(0,1),ylim=(0,1));ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.legend(frameon=False,fontsize=8);note(ax,'个人自报收入；未加权，受极端值影响；不代表全国官方基尼。')
fig.tight_layout();save(fig,f,'income_lorenz')
print('Generated 16 PNG and 16 SVG figures.')
