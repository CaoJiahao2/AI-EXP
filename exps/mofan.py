import sys
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 定义颜色及其对应的资源消耗
COLOR_COSTS = {
    '红': 1,
    '黄': 2,
    '蓝': 3,
    '绿': 4,
    '白': 5,
    '橙': 6
}

COLORS = list(COLOR_COSTS.keys())

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

# 定义每个面的位置编号（1到9）
# 位置编号从左上角到右下角依次为1到9
# 1 2 3
# 4 5 6
# 7 8 9
FACE_POSITIONS = {
    'U': [(0, 1, 0), (1, 1, 0), (2, 1, 0),
          (0, 0, 0), (1, 0, 0), (2, 0, 0),
          (0, -1, 0), (1, -1, 0), (2, -1, 0)],
    'D': [(0, 1, 2), (1, 1, 2), (2, 1, 2),
          (0, 0, 2), (1, 0, 2), (2, 0, 2),
          (0, -1, 2), (1, -1, 2), (2, -1, 2)],
    'F': [(2, 1, 1), (2, 1, 0), (2, 1, -1),
          (2, 0, 1), (2, 0, 0), (2, 0, -1),
          (2, -1, 1), (2, -1, 0), (2, -1, -1)],
    'B': [(0, 1, -1), (0, 1, 0), (0, 1, 1),
          (0, 0, -1), (0, 0, 0), (0, 0, 1),
          (0, -1, -1), (0, -1, 0), (0, -1, 1)],
    'L': [(1, 1, -1), (1, 1, 0), (1, 1, 1),
          (1, 0, -1), (1, 0, 0), (1, 0, 1),
          (1, -1, -1), (1, -1, 0), (1, -1, 1)],
    'R': [(1, 1, 1), (1, 1, 0), (1, 1, -1),
          (1, 0, 1), (1, 0, 0), (1, 0, -1),
          (1, -1, 1), (1, -1, 0), (1, -1, -1)]
}

# 为每个魔方块分配一个唯一的ID
# 使用“面名+位置编号”作为唯一标识，例如“U1”, “F5”
BLOCKS = {}
block_id = 1
for face in FACES:
    for pos_num in range(1, 10):
        BLOCKS[block_id] = {
            'id': block_id,
            'face': face,
            'position_num': pos_num,
            'coords': FACE_POSITIONS[face][pos_num - 1],
            'type': 'center' if pos_num == 5 else 'edge' if pos_num in [2,4,6,8] else 'corner'
        }
        block_id += 1

# 构建坐标到块ID的映射
COORD_TO_BLOCK = {}
for block in BLOCKS.values():
    COORD_TO_BLOCK[tuple(block['coords'])] = block['id']

# 定义邻接关系
ADJACENCY = defaultdict(set)

# 定义每个魔方块的邻接关系
for block in BLOCKS.values():
    x, y, z = block['coords']
    neighbors_coords = [
        (x-1, y, z), (x+1, y, z),  # 左右
        (x, y-1, z), (x, y+1, z),  # 前后
        (x, y, z-1), (x, y, z+1)   # 上下
    ]
    for coord in neighbors_coords:
        neighbor_id = COORD_TO_BLOCK.get(coord)
        if neighbor_id:
            ADJACENCY[block['id']].add(neighbor_id)

# 为每个块添加同一面的相邻块（例如，同一面上位置1与位置2相邻）
# 这样可以确保同一面上的块也被视为相邻
FACE_NEIGHBORS = {
    'U': [(1,2), (2,3), (3,6), (6,9), (9,8), (8,7), (7,4), (4,1)],
    'D': [(10,11), (11,12), (12,15), (15,18), (18,17), (17,16), (16,13), (13,10)],
    'F': [(19,20), (20,21), (21,24), (24,27), (27,26), (26,25), (25,22), (22,19)],
    'B': [(28,29), (29,30), (30,33), (33,36), (36,35), (35,34), (34,31), (31,28)],
    'L': [(37,38), (38,39), (39,42), (42,45), (45,44), (44,43), (43,40), (40,37)],
    'R': [(46,47), (47,48), (48,51), (51,54), (54,53), (53,52), (52,49), (49,46)]
}

# 根据FACE_NEIGHBORS添加同一面内的邻接
for face, pairs in FACE_NEIGHBORS.items():
    for a, b in pairs:
        # 计算块ID
        # 面的起始块ID
        face_start_id = (FACES.index(face)) * 9 + 1
        block_a = face_start_id + a - 1
        block_b = face_start_id + b - 1
        ADJACENCY[block_a].add(block_b)
        ADJACENCY[block_b].add(block_a)

# 确保邻接关系无重复
for block in ADJACENCY:
    ADJACENCY[block] = list(ADJACENCY[block])

# 回溯法实现
class RubiksCubeColoringBacktracking:
    def __init__(self, blocks, adjacency, color_costs, colors):
        self.blocks = blocks
        self.adjacency = adjacency
        self.color_costs = color_costs
        self.colors = colors
        self.assignment = {}
        self.min_cost = sys.maxsize
        self.best_assignment = {}
    
    # 检查当前颜色是否可以安全赋值
    def is_safe(self, block, color):
        for neighbor in self.adjacency[block]:
            if neighbor in self.assignment and self.assignment[neighbor] == color:
                return False
        return True
    
    # 回溯算法核心
    def backtrack(self, block_order, index, current_cost):
        # 所有魔方块都已赋值
        if index == len(block_order):
            if current_cost < self.min_cost:
                self.min_cost = current_cost
                self.best_assignment = self.assignment.copy()
            return
        
        block = block_order[index]
        
        # 按照资源消耗从低到高的顺序尝试颜色
        sorted_colors = sorted(self.colors, key=lambda c: self.color_costs[c])
        for color in sorted_colors:
            if self.is_safe(block, color):
                self.assignment[block] = color
                new_cost = current_cost + self.color_costs[color]
                
                # 剪枝：如果当前成本已经超过已知最小成本，停止探索
                if new_cost < self.min_cost:
                    self.backtrack(block_order, index + 1, new_cost)
                
                # 回溯
                del self.assignment[block]
    
    # 解决问题
    def solve(self):
        # 按照邻接数从多到少排序，提高回溯效率
        block_order = sorted(self.blocks.keys(), key=lambda b: len(self.adjacency[b]), reverse=True)
        self.backtrack(block_order, 0, 0)
    
    # 打印解决方案
    def print_solution(self):
        if not self.best_assignment:
            print("没有找到可行的颜色分配方案。")
            return
        print("=== 回溯法解决方案 ===")
        print(f"最小总资源消耗: {self.min_cost}")
        print("魔方块颜色分配:")
        for block in sorted(self.best_assignment.keys()):
            face = self.blocks[block]['face']
            pos_num = self.blocks[block]['position_num']
            color = self.best_assignment[block]
            print(f"魔方块 {block} ({face}{pos_num}): {color}")
        print()
    
    # 获取最佳分配方案
    def get_best_assignment(self):
        return self.best_assignment, self.min_cost

# 贪心算法实现
class RubiksCubeColoringGreedy:
    def __init__(self, blocks, adjacency, color_costs, colors):
        self.blocks = blocks
        self.adjacency = adjacency
        self.color_costs = color_costs
        self.colors = colors
        self.assignment = {}
        self.total_cost = 0
    
    # 获取可用颜色（不与相邻魔方块相同）
    def get_available_colors(self, block):
        used_colors = set()
        for neighbor in self.adjacency[block]:
            if neighbor in self.assignment:
                used_colors.add(self.assignment[neighbor])
        available = [color for color in self.colors if color not in used_colors]
        return available
    
    # 解决问题
    def solve(self):
        # 按照魔方块的度数从高到低排序
        block_order = sorted(self.blocks.keys(), key=lambda b: len(self.adjacency[b]), reverse=True)
        for block in block_order:
            available_colors = self.get_available_colors(block)
            if not available_colors:
                # 无可用颜色，无法完成涂色
                print(f"魔方块 {block} ({self.blocks[block]['face']}{self.blocks[block]['position_num']}) 无可用颜色，无法完成涂色。")
                return
            # 选择资源消耗最低的可用颜色
            selected_color = min(available_colors, key=lambda c: self.color_costs[c])
            self.assignment[block] = selected_color
            self.total_cost += self.color_costs[selected_color]
    
    # 打印解决方案
    def print_solution(self):
        if not self.assignment:
            print("没有找到可行的颜色分配方案。")
            return
        print("=== 贪心算法解决方案 ===")
        print(f"总资源消耗: {self.total_cost}")
        print("魔方块颜色分配:")
        for block in sorted(self.assignment.keys()):
            face = self.blocks[block]['face']
            pos_num = self.blocks[block]['position_num']
            color = self.assignment[block]
            print(f"魔方块 {block} ({face}{pos_num}): {color}")
        print()
    
    # 获取分配方案
    def get_assignment(self):
        return self.assignment, self.total_cost

# 图形化展示函数
def visualize_cube(assignments, title):
    """
    使用matplotlib绘制魔方的展开图，显示颜色分配结果。
    assignments: 字典，键为块ID，值为颜色名。
    title: 图表标题。
    """
    # 魔方展开图的布局
    # 上 (U) 面位于中间上方
    # 下 (D) 面位于中间下方
    # 前 (F) 面位于中间
    # 后 (B) 面位于中间后方（在展开图中通常不显示或放在适当位置）
    # 左 (L) 面位于前面左侧
    # 右 (R) 面位于前面右侧

    # 为简化，按照以下布局绘制展开图
    #    U
    # L  F  R  B
    #    D

    face_positions_2d = {
        'U': (3, 6),
        'L': (0, 3),
        'F': (3, 3),
        'R': (6, 3),
        'B': (9, 3),
        'D': (3, 0)
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制每个面的9个方块
    for face, (start_x, start_y) in face_positions_2d.items():
        for pos_num in range(1, 10):
            block_id = (FACES.index(face)) * 9 + pos_num
            color_name = assignments.get(block_id, '灰色')  # 默认灰色表示未分配
            color = COLOR_MAP.get(color_name, 'gray')
            # 计算方块的位置
            row = (pos_num - 1) // 3
            col = (pos_num - 1) % 3
            x = start_x + col
            y = start_y + (2 - row)  # 翻转y轴，使得位置1在左上角
            # 添加方块
            rect = Rectangle((x, y), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            # 添加文本标签
            ax.text(x + 0.5, y + 0.5, f"{face}{pos_num}", 
                    horizontalalignment='center', verticalalignment='center', fontsize=8)
    
    # 设置图形属性
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16)
    plt.show()

# 主程序
def main():
    # 初始化回溯法解决方案
    cube_coloring_bt = RubiksCubeColoringBacktracking(BLOCKS, ADJACENCY, COLOR_COSTS, COLORS)
    
    # 记录回溯法的开始时间
    start_time_bt = time.time()
    cube_coloring_bt.solve()
    end_time_bt = time.time()
    
    # 打印回溯法的解决方案
    cube_coloring_bt.print_solution()
    print(f"回溯法求解时间: {end_time_bt - start_time_bt:.4f} 秒\n")
    
    # 获取回溯法的分配方案
    bt_assignment, bt_cost = cube_coloring_bt.get_best_assignment()
    
    # 初始化贪心算法解决方案
    cube_coloring_greedy = RubiksCubeColoringGreedy(BLOCKS, ADJACENCY, COLOR_COSTS, COLORS)
    
    # 记录贪心算法的开始时间
    start_time_greedy = time.time()
    cube_coloring_greedy.solve()
    end_time_greedy = time.time()
    
    # 打印贪心算法的解决方案
    cube_coloring_greedy.print_solution()
    print(f"贪心算法求解时间: {end_time_greedy - start_time_greedy:.4f} 秒\n")
    
    # 获取贪心算法的分配方案
    greedy_assignment, greedy_cost = cube_coloring_greedy.get_assignment()
    
    # 图形化展示回溯法结果
    visualize_cube(bt_assignment, "回溯法颜色分配结果")
    
    # 图形化展示贪心算法结果
    visualize_cube(greedy_assignment, "贪心算法颜色分配结果")

if __name__ == "__main__":
    main()
