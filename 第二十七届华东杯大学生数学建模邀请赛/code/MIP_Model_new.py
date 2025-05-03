import pulp
import numpy as np
import pandas as pd

def cloth_mip_new(hours_per_day, days_per_week, weeks, N, R, skill_matrix, e, f, L, T_repair, p, c, num_lines):
    num_levels = len(N)
    num_steps = len(R)
    total_days = hours_per_day * days_per_week * weeks
    total_hours = total_days

    # 定义模型
    model = pulp.LpProblem("Clothing_Production_Optimization", pulp.LpMaximize)

    # 决策变量
    x = pulp.LpVariable.dicts("assign", ((i, j) for i in range(num_levels) for j in range(num_steps)),
                              lowBound=0, cat='Integer')  # 安排从事某工序的技工人数
    t = pulp.LpVariable.dicts("train", (i for i in range(num_levels - 1)), cat='Integer', lowBound=0)  # 培训人数

    # 约束1：不能分配超过现有+培训后人数
    for i in range(num_levels):
        if i == 0:
            model += pulp.lpSum([x[i, j] for j in range(num_steps)]) <= N[i] - t[i]
        elif i < num_levels - 1:
            model += pulp.lpSum([x[i, j] for j in range(num_steps)]) <= N[i] - t[i] + t[i - 1]
        else:
            model += pulp.lpSum([x[i, j] for j in range(num_steps)]) <= N[i] + t[i - 1]

    # 约束2：各工序总人数满足需求
    for j in range(num_steps):
        model += pulp.lpSum([x[i, j] for i in range(num_levels) if skill_matrix[i, j] == 1]) == R[j]

    # 约束3：非技能匹配则不能从事该工序
    for i in range(num_levels):
        for j in range(num_steps):
            if skill_matrix[i, j] == 0:
                model += x[i, j] == 0

    # 目标函数：最大化总利润 = 产品利润 - 故障损失 - 培训成本
    total_production = pulp.lpSum(
        [x[i, j] * e[i][j] * weeks * days_per_week for i in range(num_levels) for j in range(num_steps)]
    )

    total_fault_loss = pulp.lpSum([
        x[i, j] * f[i][j] * (total_hours) * (L[j]) for i in range(num_levels) for j in range(num_steps)
    ])

    total_training_cost = pulp.lpSum([t[i] * c[i] for i in range(num_levels - 1)])

    profit = total_production * p - total_fault_loss - total_training_cost
    model += profit

    # 求解
    model.solve()

    # 结果整理
    assign_result = {(i, j): x[i, j].varValue for i in range(num_levels) for j in range(num_steps)}
    train_result = {i: t[i].varValue for i in range(num_levels - 1)}
    final_profit = pulp.value(profit)

    # 输出Excel
    assign_df = pd.DataFrame(np.zeros((num_levels, num_steps)), columns=[f"工序{j+1}" for j in range(num_steps)],
                             index=[f"技工{i+1}级" for i in range(num_levels)])
    for (i, j), val in assign_result.items():
        assign_df.iloc[i, j] = int(val)

    train_df = pd.DataFrame({
        "提升对象": [f"{i+1}级→{i+2}级" for i in range(num_levels - 1)],
        "培训人数": [int(train_result[i]) for i in range(num_levels - 1)],
        "培训单价": [c[i] for i in range(num_levels - 1)],
        "培训总费用": [int(train_result[i]) * c[i] for i in range(num_levels - 1)]
    })

    assign_df.to_excel("data/new_人员分配方案.xlsx")
    train_df.to_excel("data/new_培训方案.xlsx")
    
    return {
        "人员分配": assign_df,
        "培训方案": train_df,
        "总利润": final_profit
    }

# 主程序
if __name__ == "__main__":
    hours_per_day = 8
    days_per_week = 5
    weeks = 24

    num_levels = 5
    num_steps = 5
    num_lines = 3

    N = [7, 10, 15, 33, 35]
    R = [24, 30, 15, 12, 9]

    skill_matrix = np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ])

    e = np.zeros((num_levels, num_steps))
    e[0, 4] = 30
    e[1, 3:5] = [14, 30]
    e[2, 2:5] = [12, 14, 35]
    e[3, [0, 2, 3, 4]] = [9, 13, 16, 40]
    e[4, :] = [10, 9, 14, 16, 40]

    f = np.zeros((num_levels, num_steps))
    f[0, :] = 0.5
    f[1, :] = 0.5
    f[2, :] = 0.4
    f[3, :] = 0.2
    f[4, :] = 0.2

    L = [50, 50, 30, 30, 10]
    T_repair = [4, 3, 6, 1, 1]

    p = 40
    c = [100, 150, 100, 300]

    results = cloth_mip_new(hours_per_day, days_per_week, weeks, N, R, skill_matrix, e, f, L, T_repair, p, c, num_lines)

    print("总利润：", results["总利润"])
    print("人员分配：")
    print(results["人员分配"])
    print("培训方案：")
    print(results["培训方案"])
