import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================== 1. 读取 CSV ==================
# CSV 格式示例：filename,corruption_types,severities,num_corruptions,original_annotations,transformed_annotations,mAP@0.5,mAP@0.5-0.95
df = pd.read_csv("corruption_info_with_map.csv")  # 你可以把你的 mAP 数据加到 csv

# ================== 2. 柱状图：按干扰类型 ==================
# 将多种干扰拆开
df_types = df.copy()
df_types = df_types.assign(corruption_type_split=df_types['corruption_types'].str.split('+'),
                           severity_split=df_types['severities'].str.split('+'))

# 展开多行
df_types = df_types.explode(['corruption_type_split', 'severity_split'])
df_types['severity_split'] = df_types['severity_split'].astype(int)

# 计算每种干扰类型的平均 mAP
type_map = df_types.groupby('corruption_type_split')['mAP@0.5'].mean().sort_values()

plt.figure(figsize=(10,6))
type_map.plot(kind='bar', color='skyblue')
plt.ylabel("mAP@0.5 (%)")
plt.title("模型在不同干扰类型下的鲁棒性")
plt.xticks(rotation=45, ha="right")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("robustness_by_corruption.pdf")
plt.show()

# ================== 3. 折线图：按严重等级 ==================
severity_map = df_types.groupby('severity_split')['mAP@0.5'].mean()
plt.figure(figsize=(8,5))
plt.plot(severity_map.index, severity_map.values, marker='o', linestyle='-', color='red')
plt.xticks(range(1,6))
plt.xlabel("干扰严重等级")
plt.ylabel("mAP@0.5 (%)")
plt.title("模型鲁棒性随干扰严重等级变化")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("robustness_by_severity.pdf")
plt.show()

# ================== 4. 热力图：干扰类型 × 严重等级 ==================
heatmap_data = df_types.pivot_table(index='corruption_type_split', columns='severity_split', values='mAP@0.5', aggfunc='mean')
plt.figure(figsize=(10,6))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("不同干扰类型和严重等级下的 mAP@0.5")
plt.xlabel("Severity")
plt.ylabel("Corruption Type")
plt.tight_layout()
plt.savefig("robustness_heatmap.pdf")
plt.show()