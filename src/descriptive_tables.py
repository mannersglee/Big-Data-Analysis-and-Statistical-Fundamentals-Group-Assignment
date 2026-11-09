"""Create comparable numeric descriptive-statistics tables for all four papers."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
 '01_数字鸿沟与老年健康': [('age','年龄（岁）'),('internet','使用互联网（0/1）'),('health_score','自评健康原分（1–5）'),('good_health','健康较好（0/1）'),('female','女性（0/1）')],
 '02_家庭背景与教育公平': [('math_pct','数学成绩百分比'),('family_economy_level','家庭经济等级（1–5）'),('books_level','家庭藏书等级（1–5）'),('computer_internet','家有电脑和网络（0/1）')],
 '03_流动人口城市融入': [('age','年龄（岁）'),('years_local','现居地年限（年）'),('local_identity','本地身份认同（0/1）'),('local_ties','本地交往（0/1）'),('female','女性（0/1）')],
 '04_收入差距与公平感': [('age','年龄（岁）'),('income','个人年收入（元）'),('fairness_score','社会公平感原分（1–5）'),('fair','认同社会公平（0/1）'),('female','女性（0/1）')],
}
for dirname, fields in SPECS.items():
 out=ROOT/dirname/'output'
 data=pd.read_csv(out/'analysis_sample.csv',low_memory=False)
 if dirname.startswith('02_'):
  data['family_economy_level']=data['family_economy'].map({x:i for i,x in enumerate(['非常困难','比较困难','中等','比较富裕','很富裕'],1)})
  data['books_level']=data['books'].map({x:i for i,x in enumerate(['很少','比较少','一般','比较多','很多'],1)})
 rows=[]
 for field,label in fields:
  x=pd.to_numeric(data[field],errors='coerce').dropna()
  rows.append({'variable':field,'label':label,'n':len(x),'mean':x.mean(),'std':x.std(ddof=1),'min':x.min(),'p25':x.quantile(.25),'median':x.median(),'p75':x.quantile(.75),'max':x.max()})
 pd.DataFrame(rows).to_csv(out/'numeric_descriptives.csv',index=False,encoding='utf-8-sig')
 print(dirname, len(data))
