import pulp
import numpy as np
import pandas as pd

def cloth_mip_new(hours_per_day, days_per_week, weeks, N, R, skill_matrix, e, f, L, T_repair, p, c, num_lines):
    import pulp
    import numpy as np
    import pandas as pd
    import os

    # 设置工序名称列表
    process_names = ["裁剪", "缝制", "水洗", "熨烫", "包装"]

    num_levels = len(N)         # 技工级别数量
    num_steps = len(R)          # 工序数量
    total_days = hours_per_day * days_per_week * weeks
    total_hours = total_days    # 总工作小时数（用于计算故障损失）

    # 定义优化模型（目标为最大化利润）
    model = pulp.LpProblem("Clothing_Production_Optimization", pulp.LpMaximize)

    # 定义决策变量：x为每级技工在每道工序上的分配人数，t为每一级向上培训的人数
    x = pulp.LpVariable.dicts("assign", ((i, j) for i in range(num_levels) for j in range(num_steps)),
                              lowBound=0, cat='Integer')
    t = pulp.LpVariable.dicts("train", (i for i in range(num_levels - 1)), cat='Integer', lowBound=0)

    # 每级工人数量限制约束（考虑培训后的增加和减少）
    for i in range(num_levels):
        if i == 0:
            model += pulp.lpSum([x[i, j] for j in range(num_steps)]) <= N[i] - t[i]
        elif i < num_levels - 1:
            model += pulp.lpSum([x[i, j] for j in range(num_steps)]) <= N[i] - t[i] + t[i - 1]
        else:
            model += pulp.lpSum([x[i, j] for j in range(num_steps)]) <= N[i] + t[i - 1]

    # 工序人数需求约束（确保每道工序达到R[j]要求）
    for j in range(num_steps):
        model += pulp.lpSum([x[i, j] for i in range(num_levels) if skill_matrix[i, j] == 1]) == R[j]

    # 不具备技能的技工不得分配到该工序
    for i in range(num_levels):
        for j in range(num_steps):
            if skill_matrix[i, j] == 0:
                model += x[i, j] == 0

    # 定义目标函数：利润 = 收入 - 故障损失 - 培训费用
    total_production_expr = pulp.lpSum(
        [x[i, j] * e[i][j] * weeks * days_per_week for i in range(num_levels) for j in range(num_steps)]
    )
    total_fault_loss_expr = pulp.lpSum([
        x[i, j] * f[i][j] * total_hours * L[j] for i in range(num_levels) for j in range(num_steps)
    ])
    total_training_cost_expr = pulp.lpSum([t[i] * c[i] for i in range(num_levels - 1)])

    profit = total_production_expr * p - total_fault_loss_expr - total_training_cost_expr
    model += profit

    # 求解模型
    model.solve()

    # 获取解结果
    assign_result = {(i, j): x[i, j].varValue for i in range(num_levels) for j in range(num_steps)}
    train_result = {i: t[i].varValue for i in range(num_levels - 1)}

    # 工人分配结果表
    assign_df = pd.DataFrame(np.zeros((num_levels, num_steps)),
                             columns=process_names,
                             index=[f"技工{i+1}级" for i in range(num_levels)])
    for (i, j), val in assign_result.items():
        assign_df.iloc[i, j] = int(val)

    # 培训方案表
    train_df = pd.DataFrame({
        "提升对象": [f"{i+1}级→{i+2}级" for i in range(num_levels - 1)],
        "培训人数": [int(train_result[i]) for i in range(num_levels - 1)],
        "培训单价": [c[i] for i in range(num_levels - 1)],
        "培训总费用": [int(train_result[i]) * c[i] for i in range(num_levels - 1)]
    })

    # 各工序产能计算
    capacity = []
    for j in range(num_steps):
        total = sum(x[i, j].varValue * e[i][j] for i in range(num_levels))
        capacity.append(total)

    bottleneck_index = np.argmin(capacity)
    bottleneck_process = process_names[bottleneck_index]
    min_capacity = capacity[bottleneck_index]

    total_production = min_capacity * num_lines * days_per_week * weeks
    total_revenue = total_production * p
    total_fault_loss = sum(
        x[i, j].varValue * f[i][j] * total_hours * L[j]
        for i in range(num_levels) for j in range(num_steps)
    )
    total_training_cost = sum(train_result[i] * c[i] for i in range(num_levels - 1))
    total_profit = total_revenue - total_fault_loss - total_training_cost
    weekly_profit = total_profit / weeks

    # 创建输出目录
    os.makedirs("data", exist_ok=True)

    # 保存工人分配方案
    assign_df.to_excel("data/new_工人分配方案.xlsx")

    # 保存培训方案
    train_df.to_excel("data/new_培训方案.xlsx")

    # 保存各工序故障损失
    fault_df = pd.DataFrame([
        {
            "工序": process_names[j],
            "总人数": int(sum(x[i, j].varValue for i in range(num_levels))),
            "总故障损失": sum(x[i, j].varValue * f[i][j] * total_hours * L[j] for i in range(num_levels))
        }
        for j in range(num_steps)
    ])
    fault_df.to_excel("data/new_各工序故障损失.xlsx", index=False)

    # 保存产能信息
    capacity_df = pd.DataFrame({
        "工序": process_names,
        "单线日产能(件/天)": [round(val, 2) for val in capacity],
        "总日产能(件/天)": [round(val * num_lines, 2) for val in capacity]
    })
    capacity_df.to_excel("data/new_产能和利润信息.xlsx", index=False)

    # 每周方案保存
    weekly_df = pd.DataFrame({
        "周数": [f"第{i+1}周" for i in range(weeks)],
        "产量(件)": [total_production / weeks] * weeks,
        "周利润(元)": [weekly_profit] * weeks,
        "单件利润(元)": [weekly_profit / (total_production / weeks)] * weeks
    })
    weekly_df.to_excel("data/new_每周方案.xlsx", index=False)

    # 控制台打印结果
    print("\n产能和利润信息:")
    print(capacity_df)
    print('已保存到 data/new_产能和利润信息.xlsx')

    print("\n工人分配方案:")
    print(assign_df)
    print('已保存到 data/new_工人分配方案.xlsx')

    print("\n各工序故障损失:")
    print(fault_df)
    print('已保存到 data/new_各工序故障损失.xlsx')

    print("\n培训方案:")
    print(train_df)
    print(f"总培训费用: {total_training_cost:.2f} 元")
    print('已保存到 data/new_培训方案.xlsx')

    print(f"\n{weeks} 周期间:")
    print(f"瓶颈工序: {bottleneck_process}, 单线日产能: {min_capacity:.2f} 件/天, 总日产能: {min_capacity * num_lines:.2f} 件/天")
    print(f"总故障损失: {total_fault_loss:.2f} 元")
    print(f"总培训费用: {total_training_cost:.2f} 元")
    print(f"总产量: {total_production:.2f} 件")
    print(f"总收入: {total_revenue:.2f} 元")
    print(f"总利润: {total_profit:.2f} 元")
    print(f"平均每周利润: {weekly_profit:.2f} 元/周")
    print("每周方案已保存到 data/new_每周方案.xlsx")

    # 返回结构化结果
    return {
        "人员分配": assign_df,
        "培训方案": train_df,
        "总利润": total_profit,
        "每周方案": weekly_df
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
