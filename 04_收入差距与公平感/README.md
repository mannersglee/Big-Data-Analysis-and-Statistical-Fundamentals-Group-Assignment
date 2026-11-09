# 收入差距与公平感

[完整论文](报告.md)使用 CGSS 2021 数据，分析个人收入分布、收入位置与社会公平感。原始数据 8,148 人、700 个变量；完整案例 7,264 人。论文包含变量说明、分布表、基尼敏感性、分组图、Logistic 模型、预测比较及识别边界。

运行：

```bash
/opt/anaconda3/envs/myenv/bin/python 04_收入差距与公平感/analyze.py
```

原始数据需位于 `../data/CGSS数据合集/CGSS2021/cgss2021stata/CGSS2021.dta`。`output/` 保存汇总、图表和模型结果；逐人 `analysis_sample.csv` 被 `.gitignore` 排除，不上传公共仓库。
