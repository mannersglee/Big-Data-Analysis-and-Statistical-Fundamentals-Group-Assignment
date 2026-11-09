# 流动人口城市融入

使用本仓库 CMDS 2017 年 C 卷脱敏 Stata 数据，分析流动人口的本地身份认同与在现居地居住年限、本地交往之间的关联。原始数据有 13,998 条、387 个变量；当前分析有效样本 13,994 人。

在项目根目录运行：

```bash
/opt/anaconda3/envs/myenv/bin/python 03_流动人口城市融入/analyze.py
```

脚本输出到 `output/`：分析样本、缺失统计、按居住年限和交往分组的汇总表、Logistic 回归系数、图表。数据源为 `../data/CMDS中国流动人口动态监测（STATA+SPSS）/2017 CMDS/2017 CMDS STATA数据-适用14及以上版本/2017_Migrant_C_desensitized.dta`。变量定义、结果和局限见 [报告.md](报告.md)。
