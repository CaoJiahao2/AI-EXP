import heapq
from collections import deque
from typing import List, Tuple, Optional, Dict, Set
import time
import psutil
import os

# 定义目标状态：1~15顺序排列，最后一个位置(索引15)为0(空格)
GOAL_STATE = (1, 2, 3, 4,
              5, 6, 7, 8,
              9, 10, 11, 12,
              13, 14, 15, 0)  

# 移动方向及其对应的索引变化量
MOVE_DIRECTIONS = {
    'Up': -4,
    'Down': 4,
    'Left': -1,
    'Right': 1
}

class PuzzleState:
    """
    表示十五数码问题的一个状态
    """
    def __init__(self, state: Tuple[int], parent: Optional['PuzzleState'] = None, 
                 move: Optional[str] = None, depth: int = 0):
        self.state = state       # 当前拼图状态（使用元组存储，0 表示空格）
        self.parent = parent     # 指向父状态，用于回溯路径
        self.move = move         # 从父状态移动到当前状态的动作（Up, Down, Left, Right）
        self.depth = depth       # 当前节点的深度（初始状态的深度为0）
        
        self.zero_index = self.state.index(0)  # 空格在拼图中的索引位置
        self.cost = 0            # g(n)：起始状态到当前状态的实际路径代价
        self.heuristic = 0       # h(n)：当前状态到目标状态的启发式估计值

    def __lt__(self, other: 'PuzzleState'):
        """
        定义在优先队列（heapq）中的比较依据：f(n) = g(n) + h(n)
        """
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

    def generate_successors(self) -> List['PuzzleState']:
        """
        生成当前状态的所有可能的后继状态
        """
        successors = []
        for move, delta in MOVE_DIRECTIONS.items():
            # 计算空格移动后的位置
            new_zero_index = self.zero_index + delta
            if self.is_valid_move(move):
                # 复制当前状态，并交换空格与目标位置的数字
                new_state = list(self.state)
                new_state[self.zero_index], new_state[new_zero_index] = (
                    new_state[new_zero_index], new_state[self.zero_index]
                )
                # 创建新的 PuzzleState 对象
                successors.append(PuzzleState(
                    state=tuple(new_state),
                    parent=self,
                    move=move,
                    depth=self.depth + 1
                ))
        return successors

    def is_valid_move(self, move: str) -> bool:
        """
        判断给定方向的移动是否合法（是否会超出边界）
        """
        zero_row, zero_col = divmod(self.zero_index, 4)
        if move == 'Up':
            return zero_row > 0
        elif move == 'Down':
            return zero_row < 3
        elif move == 'Left':
            return zero_col > 0
        elif move == 'Right':
            return zero_col < 3
        return False

    def manhattan_distance(self) -> int:
        """
        计算当前状态与目标状态之间的曼哈顿距离
        """
        distance = 0
        for idx, value in enumerate(self.state):
            if value == 0:
                continue  # 0 表示空格，不计入距离
            # 计算该数字在目标状态中的正确位置
            target_idx = value - 1
            current_row, current_col = divmod(idx, 4)
            target_row, target_col = divmod(target_idx, 4)
            distance += abs(current_row - target_row) + abs(current_col - target_col)
        return distance

    def __str__(self):
        """
        以4x4网格形式返回当前拼图状态的字符串表示
        """
        grid_str = ""
        for i in range(16):
            val = self.state[i]
            if val == 0:
                grid_str += "   "
            else:
                grid_str += f"{val:2d} "
            # 每行4个元素，换行
            if (i + 1) % 4 == 0:
                grid_str += "\n"
        return grid_str


def is_solvable(state: Tuple[int]) -> bool:
    """
    判断给定的初始状态是否可解。
    对于4x4拼图，判断依据：
    - 计算逆序数（inversion_count）
    - 结合空格的行位置（从上到下0~3）
    """
    # 统计逆序数
    inversion_count = 0
    state_list = [num for num in state if num != 0]
    for i in range(len(state_list)):
        for j in range(i+1, len(state_list)):
            if state_list[i] > state_list[j]:
                inversion_count += 1

    # 空格所在的行（从 0 开始计数）
    zero_row = state.index(0) // 4

    # 对4x4拼图的可解性判断
    # 如果空格在偶数行（自上而下），则逆序数必须为奇数；否则为偶数
    if zero_row % 2 == 0: 
        return inversion_count % 2 != 0
    else:
        return inversion_count % 2 == 0


def reconstruct_path(state: PuzzleState) -> List[PuzzleState]:
    """
    从目标状态回溯到初始状态，重建解路径
    """
    path = []
    current = state
    while current:
        path.append(current)
        current = current.parent
    path.reverse()
    return path


def print_puzzle(state: Tuple[int]):
    """
    打印给定拼图状态（4x4）到控制台
    """
    grid_str = ""
    for i in range(16):
        val = state[i]
        if val == 0:
            grid_str += "   "
        else:
            grid_str += f"{val:2d} "
        if (i + 1) % 4 == 0:
            grid_str += "\n"
    print(grid_str)


# ----------------- 三种搜索算法 -----------------

def bfs(start_state: Tuple[int]) -> Optional[PuzzleState]:
    """
    宽度优先搜索（BFS）算法
    保证找到最短路径，但可能消耗大量内存
    """
    start = PuzzleState(start_state)
    queue = deque([start])           # FIFO 队列
    visited: Set[Tuple[int]] = set() # 已访问状态集合
    visited.add(start.state)

    while queue:
        current = queue.popleft()
        
        if current.state == GOAL_STATE:
            return current
        
        for successor in current.generate_successors():
            if successor.state not in visited:
                visited.add(successor.state)
                queue.append(successor)
    
    return None  # 未找到解


def dfs(start_state: Tuple[int], max_depth: int = 50) -> Optional[PuzzleState]:
    """
    深度优先搜索（DFS）算法 + 迭代加深（最大深度 max_depth）
    不保证最短路径，可能因深度过大而超时或找不到解
    """
    start = PuzzleState(start_state)

    def dfs_recursive(state: PuzzleState, depth: int, visited: Set[Tuple[int]]) -> Optional[PuzzleState]:
        if state.state == GOAL_STATE:
            return state
        if depth == 0:
            return None

        for successor in state.generate_successors():
            if successor.state not in visited:
                visited.add(successor.state)
                result = dfs_recursive(successor, depth - 1, visited)
                if result is not None:
                    return result
        return None
    
    # 迭代加深循环
    for current_depth in range(max_depth + 1):
        visited: Set[Tuple[int]] = set()
        visited.add(start.state)
        result = dfs_recursive(start, current_depth, visited)
        if result is not None:
            return result
    
    return None


def a_star(start_state: Tuple[int]) -> Optional[PuzzleState]:
    """
    A* 算法
    使用曼哈顿距离作为启发式函数
    """
    start = PuzzleState(start_state)
    start.cost = 0
    start.heuristic = start.manhattan_distance()

    # 优先队列：根据 f(n) = g(n) + h(n) 的值选取最优节点
    open_list = []
    heapq.heappush(open_list, (start.cost + start.heuristic, start))

    # 存储已访问状态及其对应的最小成本
    closed_set: Dict[Tuple[int], int] = {start.state: start.cost}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current.state == GOAL_STATE:
            return current
        
        # 从当前状态生成后继状态
        for successor in current.generate_successors():
            new_cost = current.cost + 1
            if successor.state not in closed_set or new_cost < closed_set[successor.state]:
                # 如果该后继状态从当前路径获得更优的代价，则更新并加入 open_list
                closed_set[successor.state] = new_cost
                successor.cost = new_cost
                successor.heuristic = successor.manhattan_distance()
                heapq.heappush(open_list, (successor.cost + successor.heuristic, successor))

    return None  # 未找到解


# ----------------- 主函数：测试各算法并记录性能 -----------------

def main():
    
    # 示例初始状态1
    # initial_state = (
    #     5, 1, 2, 4,
    #     9, 6, 3, 8,
    #     13, 10, 7, 12,
    #     0, 14, 11, 15
    # )
    # 示例初始状态2
    # initial_state = (
    #     0, 5, 2, 4,
    #     6, 1, 3, 8,
    #     9, 10, 7, 12,
    #     13, 14, 11, 15
    # )
    # 示例初始状态3
    initial_state = (
        5, 1, 2, 4,
        9, 6, 3, 8,
        13, 10, 7, 12,
        14, 11, 15, 0
    )
    # 随机生成初始状态
    # initial_state = tuple(random.sample(range(16), 16))

    print("初始状态:")
    print_puzzle(initial_state)

    # 判断可解性
    if not is_solvable(initial_state):
        print("该初始状态不可解！")
        return
    else:
        print("该初始状态可解。\n")

    # 要对比的搜索算法
    algorithms = {
        "A*": a_star,
        "BFS": bfs,
        "DFS": dfs
    }

    # 获取当前进程对象，便于后续查看内存使用
    process = psutil.Process(os.getpid())

    for algo_name, algo_func in algorithms.items():
        print(f"开始使用 {algo_name} 算法求解...")

        # 记录开始时的时间和内存使用
        start_time = time.time()
        start_mem = process.memory_info().rss / (1024.0 ** 1)  # KB

        # 针对 DFS 做特殊处理（可设置最大深度）
        if algo_name == "DFS":
            solution = algo_func(initial_state, max_depth=50)
        else:
            solution = algo_func(initial_state)

        # 记录结束时的时间和内存使用
        end_time = time.time()
        end_mem = process.memory_info().rss / (1024.0 ** 1)  # KB

        # 计算用时和内存增量
        elapsed_time = end_time - start_time
        mem_usage = end_mem - start_mem

        if solution is None:
            print(f"【{algo_name}】 未找到解。\n")
        else:
            path = reconstruct_path(solution)
            steps = len(path) - 1  # 移动步数

            # 打印统计信息
            print(f"【{algo_name}】找到解，")
            print(f"  - 用时：{elapsed_time:.4f} 秒")
            print(f"  - 路径长度：{steps} 步")
            print(f"  - 内存使用：{mem_usage:.4f} KB\n")

            # 如需打印具体路径，可在此解注：
            for i, st in enumerate(path):
                print(f"步骤 {i}：移动 {st.move}")
                print_puzzle(st.state)

if __name__ == "__main__":
    main()
