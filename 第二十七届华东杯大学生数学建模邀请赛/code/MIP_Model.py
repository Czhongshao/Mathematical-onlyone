import pulp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

def cloth_mip(hours_per_day, days_per_week, weeks, N, R, skill_matrix, e, f, L, T_repair, p, c, num_lines):
    """
    混合整数规划模型
    
    参数:
    hours_per_day: 每天工作小时数
    days_per_week: 每周工作天数
    weeks: 总工作周数
    N: 各级技工人数
    R: 各工序所需人数
    skill_matrix: 技能匹配
    e: 效率
    f: 故障率
    L: 各工序单次故障损失
    T_repair: 各工序故障排除时间
    p: 每件产品利润
    c: 培训费用
    num_lines: 流水线数量
    
    返回:
    求解结果字典
    """
    
    # 计算总工作时间
    total_days = days_per_week * weeks
    T = hours_per_day * total_days  # 总工作小时数
    
    # 输出参数设置信息
    print(f"\n工作参数设置:")
    print(f"每天工作: {hours_per_day} 小时")
    print(f"每周工作: {days_per_week} 天")
    print(f"计算周期: {weeks} 周")
    print(f"总工作天数: {total_days} 天")
    print(f"总工作小时数: {T} 小时\n")
    
    # 输出初始条件信息
    print(f"技工人数: {N}")
    print(f"工序需求: {R}")
    print(f"产品利润: {p} 元/件")
    
    # 创建问题实例
    prob = pulp.LpProblem("Worker_Assignment_Problem", pulp.LpMaximize)
    
    # 创建决策变量
    # x[i][j] 表示第i级技工分配到第j道工序的人数
    x = {}
    for i in range(num_levels):
        for j in range(num_steps):
            x[i, j] = pulp.LpVariable(f"x_{i+1}_{j+1}", lowBound=0, cat='Integer')
    
    # y[i] 表示第i级技工提升到i+1级的人数
    y = {}
    for i in range(num_levels-1):
        y[i] = pulp.LpVariable(f"y_{i+1}", lowBound=0, cat='Integer')
    
    # 添加约束条件
    # 1. 技能匹配约束
    for i in range(num_levels):
        for j in range(num_steps):
            if skill_matrix[i, j] == 0:
                prob += x[i, j] == 0, f"Skill_Match_{i+1}_{j+1}"
    
    # 2. 人员总数约束
    for i in range(num_levels-1):
        prob += pulp.lpSum(x[i, j] for j in range(num_steps)) + y[i] <= N[i], f"Worker_Count_{i+1}"
    
    # 最高级技工数量约束（包括从4级提升上来的）
    prob += pulp.lpSum(x[num_levels-1, j] for j in range(num_steps)) <= N[num_levels-1] + y[num_levels-2], "Worker_Count_5"
    
    # 3. 工序需求约束
    for j in range(num_steps):
        prob += pulp.lpSum(x[i, j] for i in range(num_levels)) >= R[j], f"Process_Requirement_{j+1}"
    
    # 定义辅助变量和表达式
    # 每道工序的总产量
    Q_j = {}
    for j in range(num_steps):
        Q_j[j] = pulp.lpSum(x[i, j] * e[i, j] for i in range(num_levels) if skill_matrix[i, j] == 1)
    
    # 定义产能瓶颈变量
    Q = pulp.LpVariable("Q", lowBound=0) # 总产量
    for j in range(num_steps):
        prob += Q <= Q_j[j], f"Bottleneck_{j+1}"
    
    # 计算故障损失
    D_j = {}
    for j in range(num_steps):
        # 计算每道工序的平均故障率
        weighted_f = pulp.lpSum(x[i, j] * f[i, j] for i in range(num_levels) if skill_matrix[i, j] == 1)
        total_workers = pulp.lpSum(x[i, j] for i in range(num_levels) if skill_matrix[i, j] == 1)
        
        # 故障次数 = 故障率 * 工作时间 * 人数
        # 故障损失 = 故障次数 * 单次损失
        D_j[j] = weighted_f * T * L[j]
    
    # 总故障损失
    D = pulp.lpSum(D_j[j] for j in range(num_steps))
    
    # 培训费用
    C = pulp.lpSum(y[i] * c[i] for i in range(num_levels-1))
    
    # 设置目标函数：最大化总利润 = 产品收益 - 故障损失 - 培训费用
    prob += Q * p * total_days - D - C, "Total_Profit"
    
    # 求解问题
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    # 判断是否得到最优解
    if prob.status == pulp.LpStatusOptimal:
        print("成功找到最优解!")
    
    # 处理结果
    results = {}
    
    # 收集工人分配方案
    assignment_data = []
    for i in range(num_levels):
        row = [f"技工{i+1}级"]
        for j in range(num_steps):
            row.append(int(x[i, j].value()))
        assignment_data.append(row)
    
    assignment_df = pd.DataFrame(assignment_data, 
                    columns=["技工等级", "裁剪", "缝制", "水洗", "熨烫", "包装"])
    results['assignment'] = assignment_df
    
    # 收集产能和利润信息
    capacity_data = []
    min_capacity = float('inf')
    bottleneck_process = ""
    
    for j in range(num_steps):
        process_name = ["裁剪", "缝制", "水洗", "熨烫", "包装"][j]
        
        # 计算有效工作时间
        total_workers = sum(x[i, j].value() for i in range(num_levels) if skill_matrix[i, j] == 1)
        if total_workers > 0:
            avg_fault_rate = sum(x[i, j].value() * f[i, j] for i in range(num_levels) if skill_matrix[i, j] == 1) / total_workers
        else:
            avg_fault_rate = 0
            
        fault_times = avg_fault_rate * T * total_workers
        repair_hours = fault_times * T_repair[j] / 60
        effective_hours = T - repair_hours
        
        # 计算每条流水线的产能
        raw_capacity = sum(x[i, j].value() * e[i, j] for i in range(num_levels) if skill_matrix[i, j] == 1)
        per_line_capacity = (effective_hours / total_days) * raw_capacity / num_lines
        total_capacity = per_line_capacity * num_lines
        
        if per_line_capacity < min_capacity:
            min_capacity = per_line_capacity
            bottleneck_process = process_name
            
        capacity_data.append([process_name, f"{per_line_capacity:.2f}", f"{total_capacity:.2f}"])
    
    capacity_df = pd.DataFrame(capacity_data, columns=["工序", "单线日产能(件/天)", "总日产能(件/天)"])
    results['capacity'] = capacity_df
    results['bottleneck'] = {'process': bottleneck_process, 'capacity_per_line': min_capacity, 'total_capacity': min_capacity*num_lines}
    
    total_production = min_capacity * total_days * num_lines
    total_revenue = total_production * p
    
    # 收集各工序故障损失
    fault_data = []
    total_fault_loss = 0
    
    for j in range(num_steps):
        process_name = ["裁剪", "缝制", "水洗", "熨烫", "包装"][j]
        total_workers = sum(x[i, j].value() for i in range(num_levels) if skill_matrix[i, j] == 1)
        
        if total_workers > 0: 
            avg_fault_rate = sum(x[i, j].value() * f[i, j] for i in range(num_levels) if skill_matrix[i, j] == 1) / total_workers
        else: # 若是无技工分配到该工序，那么就将故障率设为0
            avg_fault_rate = 0
            
        fault_times = avg_fault_rate * T * total_workers
        fault_loss = fault_times * L[j]
        total_fault_loss += fault_loss
        
        fault_data.append([
            process_name, 
            f"{avg_fault_rate:.4f}", 
            int(total_workers), 
            f"{fault_times:.2f}",
            L[j],
            f"{fault_loss:.2f}"
        ])
    
    fault_df = pd.DataFrame(fault_data, 
                columns=["工序", "平均故障率(次/小时)", "工人数", "总故障次数", "单次损失(元)", "总损失金额(元)"])
    results['fault'] = fault_df
    results['total_fault_loss'] = total_fault_loss
    
    # 收集培训方案
    training_data = []
    for i in range(num_levels-1):
        training_data.append([f"技工{i+1}级提升到{i+2}级", int(y[i].value()), c[i], int(y[i].value() * c[i])])
    
    training_df = pd.DataFrame(training_data, 
                columns=["培训类型", "培训人数", "单位培训费用(元)", "总培训费用(元)"])
    total_training_cost = sum(y[i].value() * c[i] for i in range(num_levels-1))
    results['training'] = training_df
    results['total_training_cost'] = total_training_cost
    
    # 计算总利润
    total_profit = total_revenue - total_fault_loss - total_training_cost
    
    # 输出结果
    print("\n产能和利润信息:")
    capacity_df.to_excel('excel/产能和利润信息.xlsx', index=False)
    print('已经保存到 - excel/产能和利润信息.xlsx')
    
    print("\n工人分配方案:")
    assignment_df.to_excel('excel/工人分配方案.xlsx', index=False)
    print('已经保存到 - excel/工人分配方案.xlsx')
    
    print("\n各工序故障损失:")
    fault_df.to_excel('excel/各工序故障损失.xlsx', index=False)
    print('已经保存到 - excel/各工序故障损失.xlsx')

    print("\n培训方案:")
    training_df.to_excel('excel/培训方案.xlsx', index=False)
    print('已经保存到 - excel/培训方案.xlsx')


    print(f"\n{weeks}周期间:")
    print(f"瓶颈工序: {bottleneck_process}, 单线日产能: {min_capacity:.2f} 件/天, 总日产能: {min_capacity*num_lines:.2f} 件/天")
    print(f"总故障损失: {total_fault_loss:.2f} 元")
    print(f"总培训费用: {total_training_cost:.2f} 元")
    print(f"总产量: {total_production:.2f} 件")
    print(f"总收入: {total_revenue:.2f} 元")
    print(f"总利润: {total_profit:.2f} 元")
    print(f"平均每周利润: {total_profit/weeks:.2f} 元/周")
    
    results['total_production'] = total_production
    results['total_revenue'] = total_revenue
    results['total_profit'] = total_profit
    results['weekly_profit'] = total_profit/weeks
    
    return results

# 主程序
if __name__ == "__main__":
    hours_per_day = int(input('每天工作小时数: ')) # 每天工作小时数 8
    days_per_week = int(input('每周工作天数: ')) # 每周工作天数 5
    weeks = int(input('总工作周数: ')) # 总工作周数 4

    # 定义固定参数和变量
    # 技工等级 工序数量 流水线数量
    num_levels = 5 # 技工等级数
    num_steps = 5  # 工序数
    num_lines = 3  # 流水线数

    # 技工现有人数
    N = [7, 10, 15, 33, 35] # N_i，第i级技工人数

    # 每道工序所需总人数（三条流水线总需求）
    R = [24, 30, 15, 12, 9] # R_j，第j道工序所需人数（裁剪、缝制、水洗、熨烫、包装）

    # 技能匹配矩阵（1表示第i级技工可以从事第j道工序）
    skill_matrix = np.array([
      [0, 0, 0, 0, 1], # 1级技工只能做包装
      [0, 0, 0, 1, 1], # 2级技工能做熨烫和包装
      [0, 0, 1, 1, 1], # 3级技工能做水洗、熨烫和包装
      [1, 0, 1, 1, 1], # 4级技工能做裁剪、水洗、熨烫和包装
      [1, 1, 1, 1, 1]  # 5级技工能做所有工序
    ])

    # 技工在各工序的效率（件/天）
    e = np.zeros((num_levels, num_steps))
    e[0, 4] = 30 # 1级技工包装效率
    e[1, 3:5] = [14, 30] # 2级技工熨烫、包装效率
    e[2, 2:5] = [12, 14, 35] # 3级技工水洗、熨烫、包装效率
    e[3, [0, 2, 3, 4]] = [9, 13, 16, 40] # 4级技工裁剪、水洗、熨烫、包装效率
    e[4, :] = [10, 9, 14, 16, 40] # 5级技工所有工序效率

    # 技工在各工序的故障率（次/小时）
    f = np.zeros((num_levels, num_steps))
    f[0, :] = 0.5 # 1级技工故障率
    f[1, :] = 0.5 # 2级技工故障率
    f[2, :] = 0.4 # 3级技工故障率
    f[3, :] = 0.2 # 4级技工故障率
    f[4, :] = 0.2 # 5级技工故障率

    # 单次故障损失（元/次）
    L = [50, 50, 30, 30, 10] # L_j，各工序单次故障损失

    # 故障排除时间（分钟/次）
    T_repair = [4, 3, 6, 1, 1] # 各工序排除故障时间

    # 每件产品利润
    p = 40 # 元/件

    # 培训费用（元/人）
    c = [100, 150, 100, 300] # c_i，第i级技工提升到i+1级的费用

    
    # 调用优化函数
    results = cloth_mip(hours_per_day, days_per_week, weeks, N, R, skill_matrix, e, f, L, T_repair, p, c, num_lines)