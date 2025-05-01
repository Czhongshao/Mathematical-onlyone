import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义全局颜色组
COLORS = [
    "#ffcc94",  # 浅橙色 - 适合饼图和主要分类
    "#f3f3f3",  # 浅灰色 - 适合背景和次要元素
    "#e867a7",  # 粉红色 - 适合强调和对比
    "#86a5fe",  # 淡蓝色 - 适合技能和效率相关图表
    "#ff914e",  # 橙色 - 适合警示和故障相关图表
    "#947ff2",  # 紫色 - 适合培训和成本相关图表
    "#726869"   # 深灰色 - 适合辅助和边界元素
]

# 定义数据
# 技工等级和工序数量
num_levels = 5  # 技工等级数
num_steps = 5   # 工序数

# 技工现有人数
N = [7, 10, 15, 33, 35]  # N_i，第i级技工人数

# 每道工序所需总人数（三条流水线总需求）
R = [24, 30, 15, 12, 9]  # R_j，第j道工序所需人数（裁剪、缝制、水洗、熨烫、包装）

# 技能匹配矩阵（1表示第i级技工可以从事第j道工序）
skill_matrix = np.array([
    [0, 0, 0, 0, 1],  # 1级技工只能做包装
    [0, 0, 0, 1, 1],  # 2级技工能做熨烫和包装
    [0, 0, 1, 1, 1],  # 3级技工能做水洗、熨烫和包装
    [1, 0, 1, 1, 1],  # 4级技工能做裁剪、水洗、熨烫和包装
    [1, 1, 1, 1, 1]   # 5级技工能做所有工序
])

# 技工在各工序的效率（件/天）
e = np.zeros((num_levels, num_steps))
e[0, 4] = 30  # 1级技工包装效率
e[1, 3:5] = [14, 30]  # 2级技工熨烫、包装效率
e[2, 2:5] = [12, 14, 35]  # 3级技工水洗、熨烫、包装效率
e[3, [0, 2, 3, 4]] = [9, 13, 16, 40]  # 4级技工裁剪、水洗、熨烫、包装效率
e[4, :] = [10, 9, 14, 16, 40]  # 5级技工所有工序效率

# 技工在各工序的故障率（次/小时）
f = np.zeros((num_levels, num_steps))
f[0, :] = 0.5  # 1级技工故障率
f[1, :] = 0.5  # 2级技工故障率
f[2, :] = 0.4  # 3级技工故障率
f[3, :] = 0.2  # 4级技工故障率
f[4, :] = 0.2  # 5级技工故障率

# 单次故障损失（元/次）
L = [50, 50, 30, 30, 10]  # L_j，各工序单次故障损失

# 故障排除时间（分钟/次）
T_repair = [4, 3, 6, 1, 1]  # 各工序排除故障时间

# 每件产品利润
p = 40  # 元/件

# 培训费用（元/人）
c = [100, 150, 100, 300]  # c_i，第i级技工提升到i+1级的费用

# 工序名称和技工等级名称
process_names = ["裁剪", "缝制", "水洗", "熨烫", "包装"]
worker_levels = ["1级技工", "2级技工", "3级技工", "4级技工", "5级技工"]

def plot_worker_distribution():
    """绘制技工人数分布饼图"""
    plt.figure(figsize=(8, 6))
    plt.pie(N, labels=worker_levels, autopct='%1.1f%%', colors=COLORS[:5])
    plt.title('技工人数分布')
    plt.savefig('img/技工人数分布.png', dpi=300)
    plt.close()

def plot_process_requirements():
    """绘制各工序所需人数条形图"""
    plt.figure(figsize=(8, 6))
    plt.bar(process_names, R, color=COLORS[3])
    plt.title('各工序所需人数')
    plt.ylabel('人数')
    plt.savefig('img/各工序所需人数.png', dpi=300)
    plt.close()

def plot_skill_matrix():
    """绘制技能匹配矩阵热力图"""
    plt.figure(figsize=(8, 6))
    plt.imshow(skill_matrix, cmap='Blues')
    plt.colorbar(label='匹配情况')
    plt.xticks(range(num_steps), process_names)
    plt.yticks(range(num_levels), worker_levels)
    plt.title('技能匹配矩阵')
    for i in range(num_levels):
        for j in range(num_steps):
            text_color = 'white' if skill_matrix[i, j] == 1 else 'black'
            plt.text(j, i, str(skill_matrix[i, j]), ha='center', va='center', color=text_color)
    plt.savefig('img/技能匹配矩阵.png', dpi=300)
    plt.close()

def plot_efficiency_matrix():
    """绘制技工效率热力图"""
    plt.figure(figsize=(8, 6))
    plt.imshow(e, cmap='YlGnBu')
    plt.colorbar(label='效率(件/天)')
    plt.xticks(range(num_steps), process_names)
    plt.yticks(range(num_levels), worker_levels)
    plt.title('技工效率 (件/天)')
    for i in range(num_levels):
        for j in range(num_steps):
            if skill_matrix[i, j] == 1:  # 只在技能匹配的地方显示数值
                plt.text(j, i, str(e[i, j]), ha='center', va='center', color='black')
    plt.savefig('img/技工效率.png', dpi=300)
    plt.close()

def plot_failure_rate_matrix():
    """绘制技工故障率热力图"""
    plt.figure(figsize=(8, 6))
    plt.imshow(f, cmap='YlOrRd_r')
    plt.colorbar(label='故障率(次/小时)')
    plt.xticks(range(num_steps), process_names)
    plt.yticks(range(num_levels), worker_levels)
    plt.title('技工故障率 (次/小时)')
    for i in range(num_levels):
        for j in range(num_steps):
            if skill_matrix[i, j] == 1:  # 只在技能匹配的地方显示数值
                plt.text(j, i, str(f[i, j]), ha='center', va='center', color='black')
    plt.savefig('img/技工故障率.png', dpi=300)
    plt.close()

def plot_failure_loss():
    """绘制各工序单次故障损失条形图"""
    plt.figure(figsize=(8, 6))
    plt.bar(process_names, L, color=COLORS[4])
    plt.title('各工序单次故障损失')
    plt.ylabel('损失(元/次)')
    plt.savefig('img/各工序单次故障损失.png', dpi=300)
    plt.close()

def plot_repair_time():
    """绘制各工序故障排除时间条形图"""
    plt.figure(figsize=(8, 6))
    plt.bar(process_names, T_repair, color=COLORS[2])
    plt.title('各工序故障排除时间')
    plt.ylabel('时间(分钟/次)')
    plt.savefig('img/各工序故障排除时间.png', dpi=300)
    plt.close()

def plot_training_cost():
    """绘制各级技工培训费用条形图"""
    plt.figure(figsize=(8, 6))
    plt.bar([f"{i+1}→{i+2}" for i in range(len(c))], c, color=COLORS[5])
    plt.title('各级技工培训费用')
    plt.ylabel('培训费用(元/人)')
    plt.xlabel('技工等级提升')
    plt.savefig('img/各级技工培训费用.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_worker_distribution() # 绘制各级技工人数分布饼图
    plot_process_requirements() # 绘制各工序所需人数条形图
    plot_skill_matrix() # 绘制技能匹配矩阵热力图
    plot_efficiency_matrix() # 绘制技工在各工序的效率热力图
    plot_failure_rate_matrix() # 绘制技工在各工序的故障率热力图
    plot_failure_loss() # 绘制各工序单次故障损失条形图
    plot_repair_time() # 绘制各工序故障排除时间条形图
    plot_training_cost() # 绘制各级技工培训费用条形图