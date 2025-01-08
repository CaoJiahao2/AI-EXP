import math
import random

# 定义网格地图（左下角为原点，行从下到上，列从左到右）
# 0 表示可通行区域，1 表示障碍物
GRID_MAP = [
    [0, 0, 0, 1, 0],  # Row 4 (Top)
    [1, 1, 0, 1, 0],  # Row 3
    [0, 0, 0, 1, 0],  # Row 2
    [0, 1, 1, 1, 0],  # Row 1
    [0, 0, 0, 0, 0]   # Row 0 (Bottom)
]

# 起点和终点
START = (0, 0)  # 左下角 (row, column)
GOAL = (4, 4)   # 右上角 (row, column)

# 可能的动作（上，下，左，右）
ACTIONS = [ (1, 0), (-1, 0), (0, -1), (0, 1) ]

def manhattan_distance(state1, state2):
    """计算曼哈顿距离"""
    return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])

def get_valid_actions(state):
    """获取当前状态下的所有有效动作"""
    valid_actions = []
    for action in ACTIONS:
        new_state = (state[0] + action[0], state[1] + action[1])
        if (0 <= new_state[0] < len(GRID_MAP)) and (0 <= new_state[1] < len(GRID_MAP[0])) and (GRID_MAP[new_state[0]][new_state[1]] == 0):
            valid_actions.append(action)
    return valid_actions

def move(state, action):
    """根据动作移动，确保不越界且不进入障碍物"""
    new_state = (state[0] + action[0], state[1] + action[1])
    if (0 <= new_state[0] < len(GRID_MAP)) and (0 <= new_state[1] < len(GRID_MAP[0])) and (GRID_MAP[new_state[0]][new_state[1]] == 0):
        return new_state
    return state  # 如果移动无效，保持原地

def is_goal(state):
    """判断是否到达目标"""
    return state == GOAL

class Node:
    """MCTS 树的节点"""
    def __init__(self, state, parent=None):
        self.state = state          # 当前状态
        self.parent = parent        # 父节点
        self.children = []          # 子节点
        self.visits = 0             # 访问次数
        self.reward = 0             # 累积奖励
        self.untried_actions = get_valid_actions(state).copy()  # 未尝试的动作

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

    def expand(self):
        """扩展节点，生成一个子节点"""
        action = self.untried_actions.pop()
        next_state = move(self.state, action)
        child_node = Node(next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def __repr__(self):
        return f"Node(state={self.state}, visits={self.visits}, reward={self.reward})"

def simulate(state):
    """
    模拟函数：从当前状态开始，随机选择动作直到达到目标或达到最大步数
    引入一定的启发式策略和随机性，避免陷入循环
    """
    current_state = state
    steps = 0
    max_steps = 20  # 避免无限循环
    visited = set()
    visited.add(current_state)

    while not is_goal(current_state) and steps < max_steps:
        possible_actions = get_valid_actions(current_state)
        if not possible_actions:
            break  # 无可行动作

        # 80% 概率选择最优动作，20% 随机动作
        if random.random() < 0.8:
            # 启发式：选择能最小化与目标距离的动作
            best_actions = []
            min_distance = float('inf')
            for action in possible_actions:
                next_state = move(current_state, action)
                dist = manhattan_distance(next_state, GOAL)
                if dist < min_distance:
                    best_actions = [action]
                    min_distance = dist
                elif dist == min_distance:
                    best_actions.append(action)
            chosen_action = random.choice(best_actions)
        else:
            # 随机选择动作
            chosen_action = random.choice(possible_actions)

        next_state = move(current_state, chosen_action)

        if next_state in visited:
            break  # 避免循环
        current_state = next_state
        visited.add(current_state)
        steps += 1

    return 1 if is_goal(current_state) else 0

def mcts(root, iterations=10000):
    """执行蒙特卡洛树搜索"""
    success_simulations = 0  # 成功到达目标的模拟次数

    for _ in range(iterations):
        node = root

        # 选择阶段
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        # 扩展阶段
        if not node.is_fully_expanded():
            node = node.expand()

        # 模拟阶段
        reward = simulate(node.state)
        if reward == 1:
            success_simulations += 1

        # 回溯与更新
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    print(f"模拟次数: {iterations}, 成功次数: {success_simulations}")
    return root

def extract_path(root):
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
        if is_goal(node.state):
            break

    return path

def visualize_grid(path=None):
    """可视化网格和路径"""
    grid_visual = []
    # 从最高行到最低行进行可视化
    for i in reversed(range(len(GRID_MAP))):
        row = ""
        for j in range(len(GRID_MAP[0])):
            if (i, j) == START:
                row += "S "  # Start
            elif (i, j) == GOAL:
                row += "G "  # Goal
            elif path and (i, j) in path:
                row += "* "  # Path
            elif GRID_MAP[i][j] == 1:
                row += "X "  # Obstacle
            else:
                row += ". "  # Free space
        grid_visual.append(row)
    print("\n网格地图（S: 起点, G: 目标, X: 障碍, *: 路径）:")
    for row in grid_visual:
        print(row)

def validate_path(path):
    """
    验证路径中是否包含障碍物
    """
    for state in path:
        if state != START and state != GOAL:
            row, col = state
            if GRID_MAP[row][col] != 0:
                return False
    return True

def main():
    root = Node(START)
    root = mcts(root, iterations=10000)
    path = extract_path(root)

    # 验证路径是否有效
    is_valid = validate_path(path)

    visualize_grid(path)

    if path[-1] == GOAL and is_valid:
        print("\n找到的路径:")
        print(path)
    else:
        print("\n未能找到到达目标的路径。")
        print("当前最佳路径:")
        print(path)

if __name__ == "__main__":
    main()
