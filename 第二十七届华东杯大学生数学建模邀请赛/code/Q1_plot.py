import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 色号
COLORS = [
    "#ec347c", # 粉色
    "#00aae3", # 天蓝色
    "#dc5e31", # 橙色
    "#79bb56", # 浅绿色
    "#1f4e9f", # 深蓝色
]


df1 = pd.read_excel('data/产能和利润信息.xlsx')
df2 = pd.read_excel('data/各工序故障损失.xlsx')
df3 = pd.read_excel('data/工人分配方案.xlsx')
df4 = pd.read_excel('data/培训方案.xlsx')


print(df1.head())
print(df2.head())
print(df3.head())
print(df4.head())


def plot1_capacity_bar(df):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['工序'], df['单线日产能(件/天)'], color=COLORS[1])
    # plt.title('各工序单线日产能对比', fontsize=18)
    plt.xlabel('工序', fontsize=14)
    plt.ylabel('单线日产能(件/天)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=11)

    min_capacity_idx = df['单线日产能(件/天)'].astype(float).idxmin()
    bars[min_capacity_idx].set_color(COLORS[4])
    plt.tight_layout()
    plt.savefig('img/q1_各工序产能对比.png', dpi=300)
    plt.close()

def plot2_fault_pie(df):
    plt.figure(figsize=(10, 8))
    explode = [0.1 if i == df['总损失金额(元)'].astype(float).idxmax() else 0 for i in range(len(df))]
    plt.pie(df['总损失金额(元)'].astype(float), 
            labels=df['工序'], 
            autopct='%1.1f%%',
            startangle=90, 
            explode=explode,
            colors=COLORS,
            shadow=False,
            textprops={'fontsize': 12})
    # plt.title('各工序故障损失占比', fontsize=18)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('img/q1_各工序故障损失占比.png', dpi=300)
    plt.close()

def plot3_worker_heatmap(df):
    plt.figure(figsize=(10, 8))
    custom_cmap = plt.cm.colors.LinearSegmentedColormap.from_list("custom_blue_cmap", [COLORS[1], COLORS[4]], N=256)
    sns.heatmap(df.set_index('技工等级'), annot=True, cmap=custom_cmap, fmt='d', linewidths=.5, cbar_kws={'label': '人数'})
    # plt.title('工人分配方案热力图', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('img/q1_工人分配方案热力图.png', dpi=300)
    plt.close()

# def plot4_fault_scatter(df):
#     plt.figure(figsize=(10, 6))
#     scatter = plt.scatter(df['平均故障率(次/小时)'].astype(float), 
#                df['工人数'].astype(int), 
#                s=df['总损失金额(元)'].astype(float)/500,
#                alpha=0.7, c=range(len(df)), cmap='viridis')

#     for i, txt in enumerate(df['工序']):
#         plt.annotate(txt, 
#                     (float(df.iloc[i]['平均故障率(次/小时)']), int(df.iloc[i]['工人数'])),
#                     xytext=(5, 5), textcoords='offset points', fontsize=10)

#     plt.title('故障率与工人数量的关系', fontsize=18)
#     plt.xlabel('平均故障率(次/小时)', fontsize=14)
#     plt.ylabel('工人数（人）', fontsize=14)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     cbar = plt.colorbar(scatter)
#     cbar.set_label('工序索引', fontsize=12)
#     plt.tight_layout()
#     plt.savefig('img/q1_故障率与工人数量关系.png', dpi=300)
#     plt.close()

# def plot5_training_bar(df):
#     if df['培训人数'].astype(int).sum() == 0:
#         return
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(df['培训类型'], df['培训人数'].astype(int), color=COLORS[5])
#     for bar in bars:
#         height = bar.get_height()
#         if height > 0:
#             plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
#                     f'{height}', ha='center', va='bottom', fontsize=11)
#     plt.title('各级技工培训人数', fontsize=18)
#     plt.xlabel('培训类型', fontsize=14)
#     plt.ylabel('培训人数', fontsize=14)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.savefig('img/q1_各级技工培训人数.png', dpi=300)
#     plt.close()

def plot6_combo(df1, df2):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel('工序', fontsize=14)
    ax1.set_ylabel('单线日产能(件/天)', color=COLORS[0], fontsize=14)
    ax1.bar(df1['工序'], df1['单线日产能(件/天)'].astype(float), color=COLORS[0], alpha=0.7, label='单线日产能')
    ax1.tick_params(axis='y', labelcolor=COLORS[0])

    ax2 = ax1.twinx()
    ax2.set_ylabel('故障损失金额(元)', color=COLORS[4], fontsize=14)
    ax2.plot(df2['工序'], df2['总损失金额(元)'].astype(float), 'o-', color=COLORS[4], linewidth=2, label='故障损失金额')
    ax2.tick_params(axis='y', labelcolor=COLORS[4])

    for i, txt in enumerate(df2['平均故障率(次/小时)']):
        ax2.annotate(f'故障率: {float(txt)*100:.1f} %', 
                    xy=(i, float(df2.iloc[i]['总损失金额(元)'])),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

    plt.title('各工序产能与故障损失关系', fontsize=18)
    plt.tight_layout()
    plt.savefig('img/q1_各工序产能与故障损失关系.png', dpi=300)
    plt.close()

# 调用函数绘图
plot1_capacity_bar(df1)
plot2_fault_pie(df2)
plot3_worker_heatmap(df3)
# plot4_fault_scatter(df2)
# plot5_training_bar(df4)
plot6_combo(df1, df2)