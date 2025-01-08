# 基于蒙特卡洛树搜索的机器人路径规划
# 完整的Python实现，包括地图生成、MCTS算法、路径提取和可视化
# 可视化结果将保存为jpg文件

import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import math
import time

# 设置随机种子以保证结果可重复
random.seed(42)
np.random.seed(42)

def generate_map(width, height, obstacle_density, start, goal):
    """
    生成带有随机障碍物的二维地图。

    :param width: 地图宽度
    :param height: 地图高度
    :param obstacle_density: 障碍物密度（0到1之间）
    :param start: 起点坐标 (x, y)
    :param goal: 终点坐标 (x, y)
    :return: 二维numpy数组表示的地图
    """
    map_grid = np.zeros((height, width), dtype=int)
    
    # 计算障碍物数量
    num_obstacles = int(obstacle_density * width * height)
    obstacles = set()
    
    # 随机放置障碍物，确保起点和终点不被覆盖
    while len(obstacles) < num_obstacles:
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        if (x, y) != start and (x, y) != goal:
            obstacles.add((x, y))
    for (x, y) in obstacles:
        map_grid[y][x] = 1  # 1表示障碍物，0表示空地
    
    return map_grid

class MCTSNode:
    """
    MCTS节点类，表示机器人在某一位置的状态。
    """
    def __init__(self, position, parent=None):
        self.position = position          # 当前节点的位置 (x, y)
        self.parent = parent              # 父节点
        self.children = []                # 子节点列表
        self.visits = 0                   # 被访问次数
        self.reward = 0.0                 # 累计奖励
        self.untried_actions = []         # 未尝试的动作列表

    def is_fully_expanded(self, map_grid):
        """
        判断当前节点是否已经完全扩展（所有可能动作都已尝试）。
        """
        if not self.untried_actions:
            self.untried_actions = get_neighbors(self.position, map_grid)
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        """
        使用UCB1公式选择最佳子节点。
        """
        choices_weights = [
            (child.reward / child.visits) +
            c_param * math.sqrt((2 * math.log(self.visits)) / child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def add_child(self, child_position):
        """
        添加子节点。
        """
        child = MCTSNode(child_position, parent=self)
        self.children.append(child)
        self.untried_actions.remove(child_position)
        return child

def get_neighbors(position, map_grid):
    """
    获取当前位置的所有合法邻居（上下左右四个方向）。

    :param position: 当前坐标 (x, y)
    :param map_grid: 地图二维数组
    :return: 邻居坐标列表
    """
    neighbors = []
    x, y = position
    directions = [(-1,0), (1,0), (0,-1), (0,1)]  # 左, 右, 上, 下
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < map_grid.shape[1] and 0 <= ny < map_grid.shape[0]:
            if map_grid[ny][nx] == 0:
                neighbors.append((nx, ny))
    return neighbors

def simulate_policy(position, goal, map_grid, max_steps=100):
    """
    从当前位置开始进行模拟，采用贪心策略朝向目标移动，直到达到目标或达到最大步数。

    :param position: 当前坐标 (x, y)
    :param goal: 目标坐标 (x, y)
    :param map_grid: 地图二维数组
    :param max_steps: 模拟的最大步数
    :return: 奖励值（到达目标为1，否则为0）
    """
    current = position
    steps = 0
    while current != goal and steps < max_steps:
        neighbors = get_neighbors(current, map_grid)
        if not neighbors:
            break  # 死路
        # 采用贪心策略选择离目标最近的邻居
        neighbors.sort(key=lambda pos: euclidean_distance(pos, goal))
        current = neighbors[0]
        steps += 1
    if current == goal:
        return 1.0  # 成功到达
    else:
        return 0.0  # 未到达

def euclidean_distance(a, b):
    """
    计算两个点之间的欧几里得距离。

    :param a: 点a的坐标 (x, y)
    :param b: 点b的坐标 (x, y)
    :return: 两点之间的距离
    """
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def mcts_search(map_grid, start, goal, iterations=1000):
    """
    使用MCTS搜索最优路径。

    :param map_grid: 地图二维数组
    :param start: 起点坐标 (x, y)
    :param goal: 终点坐标 (x, y)
    :param iterations: MCTS迭代次数
    :return: MCTS根节点
    """
    root = MCTSNode(start)
    root.untried_actions = get_neighbors(start, map_grid)
    
    for _ in range(iterations):
        node = root
        # 选择阶段
        while node.is_fully_expanded(map_grid) and node.children:
            node = node.best_child()
        
        # 扩展阶段
        if not node.is_fully_expanded(map_grid):
            action = random.choice(node.untried_actions)
            node = node.add_child(action)
            node.untried_actions = get_neighbors(node.position, map_grid)
        
        # 模拟阶段
        reward = simulate_policy(node.position, goal, map_grid)
        
        # 回传阶段
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent
    
    return root

def extract_path(root, goal):
    """
    从MCTS树中提取路径。

    :param root: MCTS根节点
    :param goal: 终点坐标 (x, y)
    :return: 路径列表，从起点到终点，如果未找到则返回None
    """
    path = []
    node = root
    while node.position != goal:
        if not node.children:
            break  # 无法继续
        # 选择访问次数最多的子节点
        node = max(node.children, key=lambda n: n.visits)
        path.append(node.position)
    return path if node.position == goal else None

def visualize(map_grid, start, goal, path=None, search_nodes=None, filename='mcts_path_planning.jpg'):
    """
    可视化地图、搜索过程和路径，并保存为jpg文件。

    :param map_grid: 二维地图
    :param start: 起点坐标
    :param goal: 终点坐标
    :param path: 最终路径
    :param search_nodes: 被搜索的节点
    :param filename: 保存的文件名
    """
    plt.figure(figsize=(8,8))
    plt.imshow(map_grid, cmap='Greys', origin='upper')
    
    # 标记起点和终点
    plt.scatter(start[0], start[1], marker='o', color='green', label='起点')
    plt.scatter(goal[0], goal[1], marker='x', color='red', label='终点')
    
    # 绘制搜索过的节点
    if search_nodes:
        xs, ys = zip(*search_nodes)
        plt.scatter(xs, ys, marker='.', color='blue', alpha=0.1, label='搜索节点')
    
    # 绘制路径
    if path:
        px, py = zip(*path)
        plt.plot(px, py, color='yellow', linewidth=2, label='路径')
    
    plt.legend(loc='upper right')
    plt.title('蒙特卡洛树搜索路径规划')
    plt.gca().invert_yaxis()  # 反转Y轴以匹配地图坐标
    plt.grid(False)
    plt.savefig(filename, format='jpg')
    plt.show()
    print(f"可视化结果已保存为 {filename}")

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

def collect_search_nodes(node, nodes_set):
    """
    收集MCTS树中的所有搜索节点。

    :param node: 当前节点
    :param nodes_set: 存储节点坐标的集合
    """
    nodes_set.add(node.position)
    for child in node.children:
        collect_search_nodes(child, nodes_set)

def main():
    # 地图参数
    width = 50               # 地图宽度
    height = 50              # 地图高度
    obstacle_density = 0.2   # 障碍物密度（20%）

    # 起点和终点
    start = (0, 0)
    goal = (width - 1, height - 1)

    # 生成地图
    map_grid = generate_map(width, height, obstacle_density, start, goal)
    
    # 确保起点和终点不是障碍物
    map_grid[start[1]][start[0]] = 0
    map_grid[goal[1]][goal[0]] = 0

    # 检查是否有可行路径，如果没有则重新生成地图
    attempts = 0
    max_attempts = 20
    while not is_reachable(map_grid, start, goal):
        map_grid = generate_map(width, height, obstacle_density, start, goal)
        map_grid[start[1]][start[0]] = 0
        map_grid[goal[1]][goal[0]] = 0
        attempts += 1
        if attempts >= max_attempts:
            print("无法生成可行路径的地图，请调整参数。")
            return
    print(f"地图生成成功，尝试次数：{attempts + 1}")

    # MCTS搜索
    iterations = 5000  # MCTS迭代次数，根据需要调整
    start_time = time.time()
    root = mcts_search(map_grid, start, goal, iterations=iterations)
    end_time = time.time()
    print(f"MCTS搜索完成，迭代次数：{iterations}，耗时：{end_time - start_time:.2f}秒")

    # 提取路径
    path = extract_path(root, goal)
    if path:
        print(f"找到路径，长度：{len(path)}")
    else:
        print("未找到路径。")

    # 收集被搜索的节点用于可视化
    search_nodes = set()
    collect_search_nodes(root, search_nodes)

    # 可视化结果并保存为jpg文件
    visualize(map_grid, start, goal, path, search_nodes, filename='mcts_path_planning.jpg')

if __name__ == "__main__":
    main()
