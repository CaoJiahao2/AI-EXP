# -*- coding: utf-8 -*-
"""
魔方表面涂色问题求解脚本

任务描述：
给定一个标准的三阶魔方（3x3x3），对魔方的外表面进行着色，
选择颜色（红、黄、蓝、绿、白、橙）满足相邻魔方块颜色不同，
并使总资源消耗最低。每种颜色的资源消耗分别为：
红：1，黄：2，蓝：3，绿：4，白：5，橙：6。

实验要求：
1. 定义魔方结构并创建图结构表示相邻关系。
2. 采用贪心算法进行求解。
3. 输出最小资源消耗。
4. 调用Python语言的库，输出每个魔方块的颜色的可视化结果（保存为6个JPG图像）。

作者：曹佳豪
日期：2025-01-08
"""

import sys
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import rcParams

# 设置中文字体，示例使用 SimHei
rcParams['font.sans-serif'] = ['WenQuanYi']  # 使用黑体
rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 定义颜色及其对应的资源消耗
COLOR_COSTS = {
    '红': 1,
    '黄': 2,
    '蓝': 3,
    '绿': 4,
    '白': 5,
    '橙': 6
}

# 定义颜色在matplotlib中的映射
COLOR_MAP = {
    '红': 'red',
    '黄': 'yellow',
    '蓝': 'blue',
    '绿': 'green',
    '白': 'white',
    '橙': 'orange'
}

# 定义魔方六个外表面的名称
FACES = ['U', 'D', 'F', 'B', 'L', 'R']  # 上, 下, 前, 后, 左, 右

# 为每个魔方块分配一个唯一的ID（0-53）
# 使用(face, row, col)来表示每个魔方块的位置
# face: 'U', 'D', 'F', 'B', 'L', 'R'
# row, col: 0,1,2 从上到下、从左到右
STICKERS = {}
sticker_id = 0
for face in FACES:
    for row in range(3):
        for col in range(3):
            STICKERS[sticker_id] = {
                'face': face,
                'row': row,
                'col': col
            }
            sticker_id += 1

# 定义相邻关系
# 使用邻接表表示，每个魔方块对应一个集合，存储其相邻的魔方块ID
ADJACENCY = defaultdict(set)

# 定义每个面与其他面的相邻关系及对应边
# 每个元组为 (相邻面, 相邻面对应的边, 是否需要翻转)
# 边的定义：'top', 'bottom', 'left', 'right'
ADJACENT_FACES = {
    'U': [('B', 'top', True),
          ('R', 'top', False),
          ('F', 'top', False),
          ('L', 'top', True)],
    'D': [('F', 'bottom', False),
          ('R', 'bottom', False),
          ('B', 'bottom', True),
          ('L', 'bottom', False)],
    'F': [('U', 'bottom', False),
          ('R', 'left', True),
          ('D', 'top', False),
          ('L', 'right', False)],
    'B': [('U', 'top', True),
          ('L', 'left', True),
          ('D', 'bottom', True),
          ('R', 'right', False)],
    'L': [('U', 'left', True),
          ('F', 'left', False),
          ('D', 'left', False),
          ('B', 'right', True)],
    'R': [('U', 'right', False),
          ('B', 'left', True),
          ('D', 'right', False),
          ('F', 'right', True)]
}

# 边对应的行列映射
def get_edge_indices(face, edge):
    """
    获取指定面某一边的所有魔方块的ID列表。
    edge: 'top', 'bottom', 'left', 'right'
    """
    indices = []
    for col in range(3):
        for row in range(3):
            if face == 'U' or face == 'D':
                if edge == 'top' and row == 0:
                    indices.append(face_id(face) * 9 + row * 3 + col)
                elif edge == 'bottom' and row == 2:
                    indices.append(face_id(face) * 9 + row * 3 + col)
            elif face == 'F' or face == 'B':
                if edge == 'top' and row == 0:
                    indices.append(face_id(face) * 9 + row * 3 + col)
                elif edge == 'bottom' and row == 2:
                    indices.append(face_id(face) * 9 + row * 3 + col)
                elif edge == 'left' and col == 0:
                    indices.append(face_id(face) * 9 + row * 3 + col)
                elif edge == 'right' and col == 2:
                    indices.append(face_id(face) * 9 + row * 3 + col)
            elif face == 'L' or face == 'R':
                if edge == 'top' and row == 0:
                    indices.append(face_id(face) * 9 + row * 3 + col)
                elif edge == 'bottom' and row == 2:
                    indices.append(face_id(face) * 9 + row * 3 + col)
                elif edge == 'left' and col == 0:
                    indices.append(face_id(face) * 9 + row * 3 + col)
                elif edge == 'right' and col == 2:
                    indices.append(face_id(face) * 9 + row * 3 + col)
    return indices

def face_id(face):
    """
    返回面名称对应的索引。
    FACES = ['U', 'D', 'F', 'B', 'L', 'R']
    'U' -> 0, 'D' ->1, 'F'->2, 'B'->3, 'L'->4, 'R'->5
    """
    return FACES.index(face)

# 手动定义边的魔方块ID
# 为简化，手动列出每个面的边对应的魔方块ID
# 例如，U面的top边对应B面的top边，需要考虑翻转

# 定义每个面的边对应的魔方块ID
def get_face_edge(face, edge):
    """
    获取指定面某一边的魔方块ID列表。
    edge: 'top', 'bottom', 'left', 'right'
    返回列表按照从左到右的顺序
    """
    face_idx = face_id(face)
    stickers = []
    if edge == 'top':
        row = 0
        for col in range(3):
            stickers.append(face_idx * 9 + row * 3 + col)
    elif edge == 'bottom':
        row = 2
        for col in range(3):
            stickers.append(face_idx * 9 + row * 3 + col)
    elif edge == 'left':
        col = 0
        for row in range(3):
            stickers.append(face_idx * 9 + row * 3 + col)
    elif edge == 'right':
        col = 2
        for row in range(3):
            stickers.append(face_idx * 9 + row * 3 + col)
    return stickers

def adjacent_edge(face, adj_face):
    """
    根据当前面和相邻面，返回当前面对应的边。
    """
    mapping = {
        ('U', 'B'): 'top',
        ('U', 'R'): 'right',
        ('U', 'F'): 'bottom',
        ('U', 'L'): 'left',
        ('D', 'F'): 'bottom',
        ('D', 'R'): 'right',
        ('D', 'B'): 'bottom',
        ('D', 'L'): 'left',
        ('F', 'U'): 'top',
        ('F', 'R'): 'left',
        ('F', 'D'): 'bottom',
        ('F', 'L'): 'right',
        ('B', 'U'): 'top',
        ('B', 'L'): 'left',
        ('B', 'D'): 'bottom',
        ('B', 'R'): 'right',
        ('L', 'U'): 'left',
        ('L', 'F'): 'left',
        ('L', 'D'): 'left',
        ('L', 'B'): 'right',
        ('R', 'U'): 'right',
        ('R', 'B'): 'left',
        ('R', 'D'): 'right',
        ('R', 'F'): 'right'
    }
    return mapping.get((face, adj_face), None)

def corresponding_edge(face, adj_face):
    """
    根据当前面和相邻面，返回相邻面对应的边。
    """
    mapping = {
        ('U', 'B'): 'top',
        ('U', 'R'): 'top',
        ('U', 'F'): 'top',
        ('U', 'L'): 'top',
        ('D', 'F'): 'bottom',
        ('D', 'R'): 'bottom',
        ('D', 'B'): 'bottom',
        ('D', 'L'): 'bottom',
        ('F', 'U'): 'bottom',
        ('F', 'R'): 'left',
        ('F', 'D'): 'top',
        ('F', 'L'): 'right',
        ('B', 'U'): 'top',
        ('B', 'L'): 'left',
        ('B', 'D'): 'bottom',
        ('B', 'R'): 'right',
        ('L', 'U'): 'left',
        ('L', 'F'): 'left',
        ('L', 'D'): 'left',
        ('L', 'B'): 'right',
        ('R', 'U'): 'right',
        ('R', 'B'): 'left',
        ('R', 'D'): 'right',
        ('R', 'F'): 'right'
    }
    return mapping.get((face, adj_face), None)

# 构建相邻关系
for face in FACES:
    face_idx = face_id(face)
    # 首先，定义同一面的内部相邻关系（上下左右）
    for row in range(3):
        for col in range(3):
            current_id = face_idx * 9 + row * 3 + col
            # 上
            if row > 0:
                neighbor_id = face_idx * 9 + (row - 1) * 3 + col
                ADJACENCY[current_id].add(neighbor_id)
            # 下
            if row < 2:
                neighbor_id = face_idx * 9 + (row + 1) * 3 + col
                ADJACENCY[current_id].add(neighbor_id)
            # 左
            if col > 0:
                neighbor_id = face_idx * 9 + row * 3 + (col - 1)
                ADJACENCY[current_id].add(neighbor_id)
            # 右
            if col < 2:
                neighbor_id = face_idx * 9 + row * 3 + (col + 1)
                ADJACENCY[current_id].add(neighbor_id)
    # 然后，定义跨面相邻关系
    for adjacent in ADJACENT_FACES[face]:
        adj_face, adj_edge, is_reversed = adjacent
        current_edges = get_face_edge(face, adjacent_edge(face, adj_face))
        adj_face_edges = get_face_edge(adj_face, corresponding_edge(face, adj_face))
        if is_reversed:
            adj_face_edges = adj_face_edges[::-1]
        for i in range(3):
            ADJACENCY[current_edges[i]].add(adj_face_edges[i])
            ADJACENCY[adj_face_edges[i]].add(current_edges[i])

# 修正相邻关系，确保每对相邻魔方块互相包含对方
for sticker, neighbors in ADJACENCY.items():
    for neighbor in neighbors:
        ADJACENCY[neighbor].add(sticker)

# 贪心算法实现
def greedy_coloring(adjacency, color_costs, colors):
    """
    使用贪心算法为魔方表面着色。
    adjacency: 邻接表，字典类型，键为魔方块ID，值为相邻的魔方块ID集合
    color_costs: 颜色及其对应的资源消耗字典
    colors: 可用颜色列表
    返回: (颜色分配字典, 总资源消耗)
    """
    # 按照魔方块的度数从高到低排序
    degree_sorted = sorted(adjacency.keys(), key=lambda x: len(adjacency[x]), reverse=True)
    assignment = {}
    total_cost = 0

    for sticker in degree_sorted:
        used_colors = set()
        for neighbor in adjacency[sticker]:
            if neighbor in assignment:
                used_colors.add(assignment[neighbor])
        # 从可用颜色中选择资源消耗最低的颜色
        available_colors = [color for color in colors if color not in used_colors]
        if not available_colors:
            # 如果没有可用颜色，无法完成涂色
            print(f"魔方块 {sticker} 无可用颜色，无法完成涂色。")
            sys.exit(1)
        # 选择资源消耗最低的颜色
        selected_color = min(available_colors, key=lambda c: color_costs[c])
        assignment[sticker] = selected_color
        total_cost += color_costs[selected_color]

    return assignment, total_cost

# 可视化函数
def visualize_faces(assignments, face_names, color_map, save=True):
    """
    可视化每个面的颜色分配结果，并保存为JPG图像。
    assignments: 颜色分配字典，键为魔方块ID，值为颜色名
    face_names: 面的名称列表
    color_map: 颜色名到matplotlib颜色的映射字典
    save: 是否保存为图片
    """
    for face in face_names:
        fig, ax = plt.subplots(figsize=(3,3))
        face_idx = face_id(face)
        # 创建一个3x3的网格
        for row in range(3):
            for col in range(3):
                sticker = face_idx * 9 + row * 3 + col
                color = color_map.get(assignments.get(sticker, '灰'), 'gray')
                rect = Rectangle((col, 2 - row), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
                ax.add_patch(rect)
                # 添加魔方块ID文本
                ax.text(col + 0.5, 2 - row + 0.5, f"{sticker}", 
                        horizontalalignment='center', verticalalignment='center', fontsize=8)
        # 设置图形属性
        ax.set_xlim(0,3)
        ax.set_ylim(0,3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"{face} face color", fontsize=14)
        if save:
            plt.savefig(f"/home/jiahaocao/AI-EXP/exps/vis_mofan/{face}.jpg")
        plt.show()

# 主程序
def main():
    start_time = time.time()
    # 使用贪心算法进行着色
    assignment, total_cost = greedy_coloring(ADJACENCY, COLOR_COSTS, list(COLOR_COSTS.keys()))
    end_time = time.time()

    # 输出最小资源消耗
    print(f"最小总资源消耗: {total_cost}")

    # 输出每个魔方块的颜色分配
    for face in FACES:
        print(f"=== {face} 面 ===")
        face_idx = face_id(face)
        for row in range(3):
            row_colors = []
            for col in range(3):
                sticker = face_idx * 9 + row * 3 + col
                color = assignment.get(sticker, '无')
                row_colors.append(color)
            print(' '.join(row_colors))
        print()

    # 可视化每个面并保存为JPG图像
    visualize_faces(assignment, FACES, COLOR_MAP, save=True)

    # 输出程序运行时间
    print(f"程序运行时间: {end_time - start_time:.4f} 秒")

if __name__ == "__main__":
    main()
