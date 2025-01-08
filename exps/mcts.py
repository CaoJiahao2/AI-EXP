import math
import random
import matplotlib.pyplot as plt
import networkx as nx
import time
from collections import deque

# 定义动作（上，下，左，右）
ACTIONS = [(1, 0), (-1, 0), (0, -1), (0, 1)]

def manhattan_distance(state1, state2):
    """计算曼哈顿距离"""
    return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])

def generate_grid(rows, cols, obstacle_density):
    """
    生成带有随机障碍物的二维网格地图
    :param rows: 网格行数
    :param cols: 网格列数
    :param obstacle_density: 障碍物密度（0到1之间）
    :return: 二维网格地图
    """
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if random.random() < obstacle_density:
                grid[i][j] = 1  # 1 表示障碍物
    return grid

def get_valid_actions(state, grid):
    """获取当前状态下的所有有效动作"""
    valid_actions = []
    for action in ACTIONS:
        new_state = (state[0] + action[0], state[1] + action[1])
        if (0 <= new_state[0] < len(grid)) and (0 <= new_state[1] < len(grid[0])) and (grid[new_state[0]][new_state[1]] == 0):
            valid_actions.append(action)
    return valid_actions

def move(state, action, grid):
    """根据动作移动，确保不越界且不进入障碍物"""
    new_state = (state[0] + action[0], state[1] + action[1])
    if (0 <= new_state[0] < len(grid)) and (0 <= new_state[1] < len(grid[0])) and (grid[new_state[0]][new_state[1]] == 0):
        return new_state
    return state  # 如果移动无效，保持原地

def is_goal(state, goal):
    """判断是否到达目标"""
    return state == goal

def get_neighbors(state, grid):
    """获取当前状态的邻居节点（上下左右）"""
    neighbors = []
    for action in ACTIONS:
        neighbor = move(state, action, grid)
        if neighbor != state:
            neighbors.append(neighbor)
    return neighbors

def is_reachable(map_grid, start, goal):
    """
    使用深度优先搜索（DFS）检查起点和终点是否可达。

    :param map_grid: 地图二维数组
    :param start: 起点坐标 (x, y)
    :param goal: 终点坐标 (x, y)
    :return: 可达返回True，否则返回False
    """
    stack = [start]
    visited = set()
    while stack:
        current = stack.pop()
        if current == goal:
            return True
        if current in visited:
            continue
        visited.add(current)
        neighbors = get_neighbors(current, map_grid)
        for neighbor in neighbors:
            if neighbor not in visited:
                stack.append(neighbor)
    return False

class Node:
    """MCTS 树的节点"""
    def __init__(self, state, parent=None):
        self.state = state          # 当前状态 (row, column)
        self.parent = parent        # 父节点
        self.children = []          # 子节点
        self.visits = 0             # 访问次数
        self.reward = 0             # 累积奖励
        self.untried_actions = []   # 未尝试的动作

    def is_fully_expanded(self):
        """判断节点是否完全扩展"""
        return len(self.untried_actions) == 0

    def best_child(self, c_param=math.sqrt(2)):
        """选择最佳子节点（使用UCB1公式）"""
        choices_weights = [
            (child.reward / child.visits) + c_param * math.sqrt((2 * math.log(self.visits)) / child.visits)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self, grid):
        """扩展节点，生成一个子节点"""
        action = self.untried_actions.pop()
        next_state = move(self.state, action, grid)
        child_node = Node(next_state, parent=self)
        child_node.untried_actions = get_valid_actions(next_state, grid)
        self.children.append(child_node)
        return child_node

    def __repr__(self):
        return f"Node(state={self.state}, visits={self.visits}, reward={self.reward})"

def simulate(state, goal, grid):
    """
    模拟函数：从当前状态开始，随机选择动作直到达到目标或达到最大步数
    引入一定的启发式策略和随机性，避免陷入循环
    """
    current_state = state
    steps = 0
    max_steps = 50  # 避免无限循环
    visited = set()
    visited.add(current_state)

    while not is_goal(current_state, goal) and steps < max_steps:
        possible_actions = get_valid_actions(current_state, grid)
        if not possible_actions:
            break  # 无可行动作

        # 80% 概率选择最优动作，20% 随机动作
        if random.random() < 0.8:
            # 启发式：选择能最小化与目标距离的动作
            best_actions = []
            min_distance = float('inf')
            for action in possible_actions:
                next_state = move(current_state, action, grid)
                dist = manhattan_distance(next_state, goal)
                if dist < min_distance:
                    best_actions = [action]
                    min_distance = dist
                elif dist == min_distance:
                    best_actions.append(action)
            chosen_action = random.choice(best_actions)
        else:
            # 随机选择动作
            chosen_action = random.choice(possible_actions)

        next_state = move(current_state, chosen_action, grid)

        if next_state in visited:
            break  # 避免循环
        current_state = next_state
        visited.add(current_state)
        steps += 1

    return 1 if is_goal(current_state, goal) else 0

def mcts(root, grid, goal, iterations=10000):
    """执行蒙特卡洛树搜索"""
    success_simulations = 0  # 成功到达目标的模拟次数

    for _ in range(iterations):
        node = root

        # 选择阶段
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        # 扩展阶段
        if not node.is_fully_expanded():
            node = node.expand(grid)

        # 模拟阶段
        reward = simulate(node.state, goal, grid)
        if reward == 1:
            success_simulations += 1

        # 回溯与更新
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    print(f"模拟次数: {iterations}, 成功次数: {success_simulations}")
    return root

def extract_path(root, goal):
    """
    提取路径：从根节点开始，选择访问次数最多的子节点，直到达到目标
    """
    path = [root.state]
    node = root
    visited_states = set()
    visited_states.add(node.state)

    while True:
        if not node.children:
            break
        # 选择访问次数最多的子节点
        node = max(node.children, key=lambda c: c.visits)
        if node.state in visited_states:
            break  # 防止进入循环
        path.append(node.state)
        visited_states.add(node.state)
        if is_goal(node.state, goal):
            break

    return path

def visualize_grid(grid, path=None, search_tree=None, start=(0,0), goal=(0,0), filename="grid_path.jpg", obstacle_density=0.2):
    """
    可视化网格、路径和搜索树，并保存为JPG文件
    :param grid: 二维网格地图
    :param path: 规划出的路径
    :param search_tree: MCTS搜索树，使用networkx绘制
    :param start: 起点坐标
    :param goal: 终点坐标
    :param filename: 保存的JPG文件名
    """
    rows = len(grid)
    cols = len(grid[0])
    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.grid(True)

    # 绘制障碍物
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='black')
                ax.add_patch(rect)

    # 绘制路径
    if path:
        path_x = [state[1] for state in path]
        path_y = [state[0] for state in path]
        ax.plot(path_x, path_y, color='blue', linewidth=2, marker='o', label='Path')

    # 绘制起点和终点
    ax.plot(start[1], start[0], marker='s', color='green', markersize=10, label='Start')
    ax.plot(goal[1], goal[0], marker='*', color='red', markersize=15, label='Goal')

    # 绘制搜索树（如果提供）
    if search_tree:
        G = nx.DiGraph()
        for parent, children in search_tree.items():
            for child in children:
                G.add_edge(parent, child)
        pos = {state: (state[1], state[0]) for state in G.nodes()}
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, arrows=False)
        # 可选：只绘制部分搜索树以避免过于复杂
        # nx.draw_networkx_nodes(G, pos, ax=ax, node_size=10, node_color='gray', alpha=0.3)

    ax.legend()
    plt.title(f"Grid Path Planning (Size: {rows}x{cols}, Obstacle Density: {int(obstacle_density*100)}%)")
    plt.savefig(filename, format='jpg')
    plt.close()

def validate_path(path, grid, start, goal):
    """
    验证路径中是否包含障碍物，并且路径是否从起点到终点
    """
    if not path:
        return False
    if path[0] != start or path[-1] != goal:
        return False
    for state in path:
        if state != start and state != goal:
            row, col = state
            if grid[row][col] != 0:
                return False
    return True

def build_search_tree(node, tree):
    """
    构建搜索树的数据结构，用于可视化
    """
    if node.state not in tree:
        tree[node.state] = []
    for child in node.children:
        tree[node.state].append(child.state)
        build_search_tree(child, tree)

def run_experiment(rows=10, cols=10, obstacle_density=0.2, iterations=10000, max_attempts=20, save_visualization=True):
    """
    运行一次MCTS路径规划实验
    :param rows: 网格行数
    :param cols: 网格列数
    :param obstacle_density: 障碍物密度（0到1之间）
    :param iterations: MCTS迭代次数
    :param max_attempts: 最大尝试次数生成可行路径的地图
    :param save_visualization: 是否保存可视化结果
    """
    attempts = 0
    while attempts < max_attempts:
        # 生成网格地图
        grid = generate_grid(rows, cols, obstacle_density)

        # 确保起点和终点不被障碍物占据
        start = (0, 0)                # 左下角
        goal = (rows-1, cols-1)      # 右上角
        grid[start[0]][start[1]] = 0
        grid[goal[0]][goal[1]] = 0

        # 检查起点和终点是否可达
        if is_reachable(grid, start, goal):
            print(f"地图生成成功，尝试次数：{attempts + 1}")
            break
        else:
            attempts += 1
            print(f"尝试次数：{attempts} - 未生成可行路径的地图，重新生成...")

    if attempts >= max_attempts:
        print("无法生成可行路径的地图，请调整参数。")
        return

    # 初始化MCTS根节点
    root = Node(start)
    root.untried_actions = get_valid_actions(start, grid)

    # 记录搜索树
    search_tree = {}

    # 执行MCTS
    start_time = time.time()
    root = mcts(root, grid, goal, iterations)
    end_time = time.time()

    # 构建搜索树数据结构（仅限可视化）
    build_search_tree(root, search_tree)

    # 提取路径
    path = extract_path(root, goal)

    # 验证路径
    is_valid = validate_path(path, grid, start, goal)

    # 可视化并保存结果
    if save_visualization:
        filename = f"grid_path_{rows}x{cols}_density{int(obstacle_density*100)}.jpg"
        visualize_grid(grid, path, search_tree, start, goal, filename, obstacle_density)

    # 输出结果
    print(f"地图规模: {rows}x{cols}, 障碍物密度: {obstacle_density}")
    print(f"模拟次数: {iterations}, 成功次数: {root.reward}")
    print(f"搜索时间: {end_time - start_time:.2f} 秒")
    if is_valid:
        print("找到的路径是有效的。")
        print("路径长度:", len(path))
    else:
        print("未能找到有效的路径。")
        print("当前最佳路径:")
        print(path)

def main():
    """
    主函数：运行多个实验，测试MCTS在不同障碍物密度和地图规模下的表现
    """
    experiments = [
        {'rows': 5, 'cols': 5, 'obstacle_density': 0.2},
        {'rows': 5, 'cols': 5, 'obstacle_density': 0.3},
        {'rows': 5, 'cols': 5, 'obstacle_density': 0.4},
        {'rows': 8, 'cols': 8, 'obstacle_density': 0.2},
        {'rows': 8, 'cols': 8, 'obstacle_density': 0.3},
        {'rows': 8, 'cols': 8, 'obstacle_density': 0.4},
        {'rows': 10, 'cols': 10, 'obstacle_density': 0.2},
        {'rows': 10, 'cols': 10, 'obstacle_density': 0.3},
        {'rows': 10, 'cols': 10, 'obstacle_density': 0.4},
        {'rows': 15, 'cols': 15, 'obstacle_density': 0.3},
        {'rows': 15, 'cols': 15, 'obstacle_density': 0.4},
        {'rows': 15, 'cols': 15, 'obstacle_density': 0.5},
    ]

    for exp in experiments:
        print("\n" + "="*50)
        print(f"运行实验: {exp}")
        run_experiment(
            rows=exp['rows'],
            cols=exp['cols'],
            obstacle_density=exp['obstacle_density'],
            iterations=500000,
            max_attempts=20,
            save_visualization=True
        )

if __name__ == "__main__":
    main()
