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

# 定义唯一的魔方块（26个）
# 角块：8个，边块：12个，中心块：6个
UNIQUE_BLOCKS = {
    # 角块
    1: {'coords': (0, 1, 0), 'faces': ['U', 'F', 'L']},
    2: {'coords': (0, 1, 2), 'faces': ['U', 'F', 'R']},
    3: {'coords': (2, 1, 0), 'faces': ['U', 'B', 'L']},
    4: {'coords': (2, 1, 2), 'faces': ['U', 'B', 'R']},
    5: {'coords': (0, -1, 0), 'faces': ['D', 'F', 'L']},
    6: {'coords': (0, -1, 2), 'faces': ['D', 'F', 'R']},
    7: {'coords': (2, -1, 0), 'faces': ['D', 'B', 'L']},
    8: {'coords': (2, -1, 2), 'faces': ['D', 'B', 'R']},
    # 边块
    9: {'coords': (1, 1, 0), 'faces': ['U', 'L']},
    10: {'coords': (1, 1, 2), 'faces': ['U', 'R']},
    11: {'coords': (0, 1, 1), 'faces': ['U', 'F']},
    12: {'coords': (2, 1, 1), 'faces': ['U', 'B']},
    13: {'coords': (1, -1, 0), 'faces': ['D', 'L']},
    14: {'coords': (1, -1, 2), 'faces': ['D', 'R']},
    15: {'coords': (0, -1, 1), 'faces': ['D', 'F']},
    16: {'coords': (2, -1, 1), 'faces': ['D', 'B']},
    17: {'coords': (0, 0, 1), 'faces': ['F', 'L']},
    18: {'coords': (0, 0, -1), 'faces': ['F', 'R']},
    19: {'coords': (2, 0, 1), 'faces': ['B', 'L']},
    20: {'coords': (2, 0, -1), 'faces': ['B', 'R']},
    # 中心块
    21: {'coords': (1, 1, 1), 'faces': ['U']},  # U中心
    22: {'coords': (1, -1, 1), 'faces': ['D']}, # D中心
    23: {'coords': (1, 0, 2), 'faces': ['F']},  # F中心
    24: {'coords': (1, 0, 0), 'faces': ['B']},  # B中心
    25: {'coords': (0, 0, 0), 'faces': ['L']},  # L中心
    26: {'coords': (2, 0, 0), 'faces': ['R']}   # R中心
}

# 为每个魔方块分配一个唯一的ID
BLOCKS = UNIQUE_BLOCKS.copy()

# 构建坐标到块ID的映射
COORD_TO_BLOCK = {}
for block_id, block in BLOCKS.items():
    COORD_TO_BLOCK[tuple(block['coords'])] = block_id

# 定义邻接关系
ADJACENCY = defaultdict(set)

# 定义每个魔方块的邻接关系
for block_id, block in BLOCKS.items():
    x, y, z = block['coords']
    neighbors_coords = [
        (x-1, y, z), (x+1, y, z),  # 左右
        (x, y-1, z), (x, y+1, z),  # 前后
        (x, y, z-1), (x, y, z+1)   # 上下
    ]
    for coord in neighbors_coords:
        neighbor_id = COORD_TO_BLOCK.get(coord)
        if neighbor_id:
            ADJACENCY[block_id].add(neighbor_id)

# 确保邻接关系无重复
for block in ADJACENCY:
    ADJACENCY[block] = list(ADJACENCY[block])

# 着色算法实现与之前相同，基于重新定义的块和邻接关系
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
            faces = self.blocks[block]['faces']
            color = self.best_assignment[block]
            print(f"魔方块 {block} (Faces: {','.join(faces)}): {color}")
        print()

    # 获取最佳分配方案
    def get_best_assignment(self):
        return self.best_assignment, self.min_cost

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
                print(f"魔方块 {block} (Faces: {','.join(self.blocks[block]['faces'])}) 无可用颜色，无法完成涂色。")
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
            faces = self.blocks[block]['faces']
            color = self.assignment[block]
            print(f"魔方块 {block} (Faces: {','.join(faces)}): {color}")
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
    # 后 (B) 面位于前面的右侧
    # 左 (L) 面位于前面的左侧
    # 右 (R) 面位于前面的右侧

    face_positions_2d = {
        'U': (3, 6),
        'L': (0, 3),
        'F': (3, 3),
        'R': (6, 3),
        'B': (9, 3),
        'D': (3, 0)
    }

    fig, ax = plt.subplots(figsize=(12, 8))

    for face, (start_x, start_y) in face_positions_2d.items():
        # 获取属于该面的块
        face_blocks = [block_id for block_id, block in BLOCKS.items() if face in block['faces']]
        # 对于每个面，按照位置编号（假设左上到右下顺序）
        # 需要手动排序或定义块的位置
        # 这里假设块按三维坐标的某种顺序排列
        # 具体位置需要根据物理坐标映射

        # 例如，对于U面，块的坐标有固定的y和z值
        # 排序方式：x从0到2， z从0到2

        # 获取U面块
        if face == 'U':
            sorted_face_blocks = sorted(face_blocks, key=lambda b: (BLOCKS[b]['coords'][0], BLOCKS[b]['coords'][2]))
        elif face == 'D':
            sorted_face_blocks = sorted(face_blocks, key=lambda b: (BLOCKS[b]['coords'][0], BLOCKS[b]['coords'][2]))
        elif face == 'F':
            sorted_face_blocks = sorted(face_blocks, key=lambda b: (BLOCKS[b]['coords'][1], BLOCKS[b]['coords'][0]))
        elif face == 'B':
            sorted_face_blocks = sorted(face_blocks, key=lambda b: (BLOCKS[b]['coords'][1], -BLOCKS[b]['coords'][0]))
        elif face == 'L':
            sorted_face_blocks = sorted(face_blocks, key=lambda b: (BLOCKS[b]['coords'][2], BLOCKS[b]['coords'][1]))
        elif face == 'R':
            sorted_face_blocks = sorted(face_blocks, key=lambda b: (-BLOCKS[b]['coords'][2], BLOCKS[b]['coords'][1]))

        # 假设每个面有3x3的块
        for idx, block_id in enumerate(sorted_face_blocks):
            row = idx // 3
            col = idx % 3
            x = start_x + col
            y = start_y + (2 - row)  # 翻转y轴，使得位置1在左上角
            color_name = assignments.get(block_id, '灰色')  # 默认灰色表示未分配
            color = COLOR_MAP.get(color_name, 'gray')
            rect = Rectangle((x, y), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            # 添加文本标签
            ax.text(x + 0.5, y + 0.5, f"{face}", 
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

    # 记录回溯法的开始时间
    start_time_bt = time.time()
    cube_coloring_bt.solve()
    end_time_bt = time.time()

    # 打印回溯法的解决方案
    cube_coloring_bt.print_solution()
    print(f"回溯法求解时间: {end_time_bt - start_time_bt:.4f} 秒\n")

    # 获取回溯法的分配方案
    bt_assignment, bt_cost = cube_coloring_bt.get_best_assignment()

    # 图形化展示回溯法结果
    visualize_cube(bt_assignment, "回溯法颜色分配结果")

    # 图形化展示贪心算法结果
    visualize_cube(greedy_assignment, "贪心算法颜色分配结果")

if __name__ == "__main__":
    main()
