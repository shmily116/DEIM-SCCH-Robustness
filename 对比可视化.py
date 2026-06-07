import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小（放大3倍）
plt.rcParams['font.size'] = 24  # 默认字体大小
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['legend.fontsize'] = 20  # 图例字体
plt.rcParams['axes.labelsize'] = 22   # 坐标轴标签
plt.rcParams['axes.titlesize'] = 28   # 标题
plt.rcParams['xtick.labelsize'] = 18  # X轴刻度
plt.rcParams['ytick.labelsize'] = 18  # Y轴刻度

# ==================== 您的原始数据 ====================
models = ['DEIM-PECH (Ours)', 'DEIM', 'RT-DETR-r18', 'YOLOv8-n', 'YOLOv11-n']
models_short = ['DEIM-SCCH', 'DEIM', 'RT-DETR', 'YOLOv8', 'YOLOv11']

# 原始数据（来自您的实验）
clean_map = [84, 80.9, 83.5, 76.4, 75.5]  # 干净 mAP@0.5
corrupted_map = [65.9, 58.5, 62.3, 48.4, 46.9]  # 脏 mAP@0.5
drop_rate = [21.5, 27.7, 25.7, 36.6, 37.9]  # 相对下降率
clean_ar = [63.9, 58.9, 63.3, 53.5, 53.5]  # 干净 AR@0.5-0.95
corrupted_ar = [52.3, 47.1, 49.8, 37.2, 35.5]  # 脏 AR@0.5-0.95


# ==================== 正确的归一化方法 ====================
def normalize_by_dimension(data_list):
    """
    对单个维度进行归一化（越高越好）
    公式: (x - min) / (max - min)
    结果范围: 0-1
    """
    min_val = min(data_list)
    max_val = max(data_list)
    if max_val - min_val == 0:
        return [0.5] * len(data_list)
    return [(x - min_val) / (max_val - min_val) for x in data_list]


# 计算各维度的鲁棒性得分（脏/干净）
robustness_score = [corrupted_map[i] / clean_map[i] for i in range(len(models))]

# 对各维度分别归一化
clean_map_norm = normalize_by_dimension(clean_map)
corrupted_map_norm = normalize_by_dimension(corrupted_map)
robustness_norm = normalize_by_dimension(robustness_score)
clean_ar_norm = normalize_by_dimension(clean_ar)
corrupted_ar_norm = normalize_by_dimension(corrupted_ar)

# ==================== 打印归一化结果（验证用） ====================
print("=" * 70)
print("归一化后的各维度数值（越接近1越好）")
print("=" * 70)
print(f"{'Model':<18} {'Clean mAP':<12} {'Corrupted mAP':<14} {'Robustness':<12} {'Clean AR':<12} {'Corrupted AR':<12}")
print("-" * 70)
for i, model in enumerate(models_short):
    print(
        f"{model:<18} {clean_map_norm[i]:.3f}       {corrupted_map_norm[i]:.3f}          {robustness_norm[i]:.3f}       {clean_ar_norm[i]:.3f}       {corrupted_ar_norm[i]:.3f}")
print("=" * 70)

# 输出DEIM-PECH的优势
print(f"\n✅ DEIM-PECH 在各维度的归一化得分:")
print(f"   - Clean mAP: {clean_map_norm[0]:.3f} (第1名)")
print(f"   - Corrupted mAP: {corrupted_map_norm[0]:.3f} (第1名)")
print(f"   - Robustness: {robustness_norm[0]:.3f} (第1名)")
print(f"   - Clean AR: {clean_ar_norm[0]:.3f} (第1名)")
print(f"   - Corrupted AR: {corrupted_ar_norm[0]:.3f} (第1名)")
print(f"\n结论: DEIM-PECH 在全部5个维度上均排名第一！")

# ==================== 绘制雷达图（所有区域填充） ====================
fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(projection='polar'))  # 图幅放大

# 定义5个评估维度标签（字体通过rcParams控制）
metrics = [
    'Clean\nAP@0.5',
    'Corrupted\nAP@0.5',
    'Robustness\n(1-Drop Rate)',
    'Clean\nAR@0.5-0.95',
    'Corrupted\nAR@0.5-0.95'
]

# 归一化后的数据矩阵 [模型 x 维度]
normalized_data = []
for i in range(len(models)):
    values = [
        clean_map_norm[i],
        corrupted_map_norm[i],
        robustness_norm[i],
        clean_ar_norm[i],
        corrupted_ar_norm[i]
    ]
    normalized_data.append(values)

# 雷达图角度设置
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # 闭合

# 颜色方案（半透明填充）
colors_line = ['#2E8B57', '#A23B72', '#F18F01', '#4A90D9', '#D9A04A']  # 线条颜色
colors_fill = ['#2E8B57', '#A23B72', '#F18F01', '#4A90D9', '#D9A04A']  # 填充颜色
alphas = [0.25, 0.2, 0.2, 0.2, 0.2]  # 填充透明度（DEIM-PECH 稍高）

# 绘制雷达图（先填充后画线，避免线条被遮挡）
for i, model in enumerate(models_short):
    values = normalized_data[i]
    values += values[:1]  # 闭合

    # 线条样式（线宽也放大）
    linewidth = 6 if i == 0 else 3  # 原3/1.5 → 6/3
    linestyle = '-' if i == 0 else '--'
    color_line = colors_line[i]

    # 填充区域（所有模型都填充）
    ax.fill(angles, values, alpha=alphas[i], color=colors_fill[i])

    # 绘制边界线
    ax.plot(angles, values, 'o-', linewidth=linewidth, linestyle=linestyle,
            label=model, color=color_line, alpha=1.0, markersize=12 if i == 0 else 8,  # 原6/4 → 12/8
            markerfacecolor=color_line, markeredgecolor='white', markeredgewidth=1.5)

# 设置刻度标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=24, fontweight='bold')  # 原11 → 33

# 设置y轴刻度（0到1）
ax.set_ylim(0, 1.1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=18)  # 原9 → 27

# 添加网格和标题
ax.grid(True, alpha=0.3, linestyle='--')

# 图例（放在右侧，字体放大）
ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.1), fontsize=22, frameon=True)  # 原10 → 30

plt.tight_layout()
plt.savefig('radar_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('radar_chart.pdf', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 输出雷达图解读 ====================
print("\n" + "=" * 70)
print("📊 雷达图解读说明")
print("=" * 70)
print("""
┌─────────────────────────────────────────────────────────────────────┐
│                        雷达图使用指南                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   【坐标轴含义】                                                      │
│   • 放射状线条 = 5个评估维度                                          │
│   • 同心圆 = 归一化得分（0→内圈，1→外圈）                              │
│   • 得分越高 = 越靠近外圈 = 性能越好                                   │
│                                                                     │
│   【如何判断最优模型】                                                 │
│   • 面积最大的多边形 = 综合性能最好的模型                               │
│   • DEIM-PECH (绿色区域) 覆盖了其他所有模型                           │
│   • 在全部5个维度上均位于最外圈                                       │
│                                                                     │
│   【您的数据结论】                                                     │
│   • DEIM-PECH (绿色实线+绿色填充) 在5个维度上均得分最高                │
│   • 其他模型 (彩色虚线+半透明填充) 的五边形面积明显更小                 │
│   • 雷达图清晰展示: DEIM-PECH 具有全面的性能优势                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")