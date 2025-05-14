import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import random
from collections import deque
import heapq
import matplotlib


# 设置页面配置
st.set_page_config(page_title="最短路径算法可视化", layout="wide")

# 设置标题
st.title("最短路径算法可视化")

# 创建侧边栏
st.sidebar.title("算法设置")

# 算法选择
algorithm = st.sidebar.selectbox(
    "选择算法",
    ["深度优先搜索 (DFS)", "宽度优先搜索 (BFS)", "A*算法"]
)

# 节点数量设置
num_nodes = st.sidebar.slider("节点数量", min_value=0, max_value=50, value=30)

# 边的密度设置
edge_density = st.sidebar.slider("边的密度", min_value=0.1, max_value=0.5, value=0.2)

# 起点和终点设置
start_node = st.sidebar.number_input("起点", min_value=0, max_value=num_nodes - 1, value=0)
end_node = st.sidebar.number_input("终点", min_value=0, max_value=num_nodes - 1, value=num_nodes - 1)

# 随机生成图或加载已有图的选择
graph_option = st.sidebar.radio("图形选择", ["随机生成图", "加载已有图"])

# 创建两列布局
col1, col2 = st.columns([2, 1])

# 设置 matplotlib 字体
matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans GB']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 深度优先搜索算法
def dfs(graph, start, end):
    start_time = time.time() # 记录开始时间
    visited = set() # 记录已访问节点
    stack = [(start, [start], 0)]  # (当前节点, 路径, 距离)
    expanded_nodes = 0
    best_path = None
    best_distance = float('inf')
    animation_frames = []

    while stack:
        node, path, distance = stack.pop()
        expanded_nodes += 1
        animation_frames.append(list(path))

        if node == end:
            # 找到终点时，比较路径长度，保留最短的
            if distance < best_distance:
                best_distance = distance
                best_path = path
            continue  # 继续搜索其他可能的路径

        if node not in visited:
            visited.add(node)

            # 获取相邻节点并按照距离排序（启发式）
            neighbors = list(graph.neighbors(node))
            neighbors.sort(key=lambda n: graph[node][n]['weight'] if n not in visited else float('inf'))

            for neighbor in neighbors:
                if neighbor not in visited:
                    new_distance = distance + graph[node][neighbor]['weight']
                    # 如果当前路径已经超过最佳路径，则不再继续
                    if best_path is not None and new_distance >= best_distance:
                        continue
                    new_path = path + [neighbor]
                    stack.append((neighbor, new_path, new_distance))

    end_time = time.time()
    if best_path:
        return best_path, best_distance, expanded_nodes, end_time - start_time, animation_frames

    end_time = time.time()
    return None, 0, expanded_nodes, end_time - start_time, animation_frames


# 宽度优先搜索算法（使用优先队列处理有权图）
def bfs(graph, start, end):
    start_time = time.time()
    # 使用优先队列而不是普通队列，以便在有权图中找到最短路径
    # 实际上使BFS变成类似Dijkstra算法的实现
    priority_queue = [(0, start, [start])]  # (距离, 当前节点, 路径)
    heapq.heapify(priority_queue)

    # 记录到每个节点的最短距离
    distances = {start: 0}
    expanded_nodes = 0
    animation_frames = []

    while priority_queue:
        distance, node, path = heapq.heappop(priority_queue)
        expanded_nodes += 1
        animation_frames.append(list(path))

        # 如果找到终点，返回路径
        if node == end:
            end_time = time.time()
            return path, distance, expanded_nodes, end_time - start_time, animation_frames

        # 如果当前节点的距离大于已知的最短距离，跳过
        if node in distances and distance > distances[node]:
            continue

        # 探索所有邻居
        for neighbor in graph.neighbors(node):
            new_distance = distance + graph[node][neighbor]['weight']

            # 如果找到更短的路径，更新距离并加入队列
            if neighbor not in distances or new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                new_path = path + [neighbor]
                heapq.heappush(priority_queue, (new_distance, neighbor, new_path))

    end_time = time.time()
    return None, 0, expanded_nodes, end_time - start_time, animation_frames


# A*算法
def astar(graph, start, end, positions):
    start_time = time.time()

    # 启发函数：使用欧几里得距离
    def heuristic(node1, node2):
        pos1 = positions[node1]
        pos2 = positions[node2]
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    # 优先队列
    open_set = [(0, start, [start], 0)]  # (f值, 当前节点, 路径, g值)
    heapq.heapify(open_set)

    # 已访问节点集合
    closed_set = set()
    expanded_nodes = 0
    animation_frames = []

    while open_set:
        f, node, path, g = heapq.heappop(open_set)
        expanded_nodes += 1
        animation_frames.append(list(path))

        if node == end:
            end_time = time.time()
            return path, g, expanded_nodes, end_time - start_time, animation_frames

        if node in closed_set:
            continue

        closed_set.add(node)

        for neighbor in graph.neighbors(node):
            if neighbor in closed_set:
                continue

            new_g = g + graph[node][neighbor]['weight']
            new_f = new_g + heuristic(neighbor, end)
            new_path = path + [neighbor]

            heapq.heappush(open_set, (new_f, neighbor, new_path, new_g))

    end_time = time.time()
    return None, 0, expanded_nodes, end_time - start_time, animation_frames


# 生成随机图
def generate_random_graph(num_nodes, edge_density, fixed_positions=None):
    # 基于networkx创建空图
    G = nx.Graph()

    # 添加节点
    for i in range(num_nodes):
        G.add_node(i)

    # 使用固定位置或生成随机位置
    if fixed_positions is not None:
        positions = fixed_positions
    else:
        positions = {i: (random.uniform(0, 10), random.uniform(0, 10)) for i in range(num_nodes)}

    # 添加边
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # 根据边密度随机决定是否添加边
            if random.random() < edge_density:
                # 计算两点之间的欧几里得距离作为权重
                weight = np.sqrt((positions[i][0] - positions[j][0]) ** 2 +
                                 (positions[i][1] - positions[j][1]) ** 2)
                G.add_edge(i, j, weight=round(weight, 2))

    return G, positions


# 保存图到文件
def save_graph_to_file(G, positions, filename="graph_data.csv"):
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append([u, v, data['weight']])

    edges_df = pd.DataFrame(edges, columns=['source', 'target', 'weight'])

    pos_data = []
    for node, pos in positions.items():
        pos_data.append([node, pos[0], pos[1]])

    pos_df = pd.DataFrame(pos_data, columns=['node', 'x', 'y'])

    # 保存到CSV文件
    edges_df.to_csv(f"edges_{filename}", index=False)
    pos_df.to_csv(f"positions_{filename}", index=False)

    return f"edges_{filename}", f"positions_{filename}"


# 从文件加载图
def load_graph_from_file(edges_file, positions_file):
    edges_df = pd.read_csv(edges_file)
    pos_df = pd.read_csv(positions_file)

    G = nx.Graph()

    # 添加节点和位置
    positions = {}
    for _, row in pos_df.iterrows():
        node = int(row['node'])
        G.add_node(node)
        positions[node] = (row['x'], row['y'])

    # 添加边
    for _, row in edges_df.iterrows():
        G.add_edge(int(row['source']), int(row['target']), weight=row['weight'])

    return G, positions


# 可视化图和路径
def visualize_graph(G, positions, path=None, algorithm_name="", distance=0, expanded_nodes=0, time_taken=0):
    plt.figure(figsize=(10, 8))

    # 绘制所有边
    nx.draw_networkx_edges(G, positions, alpha=0.3)

    # 绘制所有节点
    nx.draw_networkx_nodes(G, positions, node_size=300, node_color='lightblue')

    # 绘制起点和终点
    if path and len(path) > 0:
        nx.draw_networkx_nodes(G, positions, nodelist=[path[0]], node_size=500, node_color='green')
        nx.draw_networkx_nodes(G, positions, nodelist=[path[-1]], node_size=500, node_color='red')

    # 绘制路径
    if path and len(path) > 1:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, positions, edgelist=path_edges, width=2, edge_color='red')

    # 绘制节点标签
    nx.draw_networkx_labels(G, positions)

    # 绘制边权重标签
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_labels, font_size=8)

    plt.title(f"{algorithm_name} - 节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    plt.axis('off')

    return plt


# 初始化会话状态
if 'G' not in st.session_state:
    st.session_state.G = None
    st.session_state.positions = None
    st.session_state.prev_num_nodes = None
    st.session_state.prev_edge_density = None
    st.session_state.fixed_positions = None

# 重新生成图形的按钮
if st.sidebar.button("重新生成图形"):
    st.session_state.G = None
    st.session_state.positions = None
    st.session_state.prev_num_nodes = None

# 主程序
if st.session_state.G is None or st.session_state.prev_num_nodes != num_nodes:
    st.session_state.prev_num_nodes = num_nodes
    st.session_state.prev_edge_density = edge_density
    if graph_option == "随机生成图":
        st.session_state.G, st.session_state.positions = generate_random_graph(num_nodes, edge_density)
        st.session_state.fixed_positions = st.session_state.positions.copy()

elif st.session_state.prev_edge_density != edge_density:
    st.session_state.prev_edge_density = edge_density
    if graph_option == "随机生成图":
        st.session_state.G, _ = generate_random_graph(num_nodes, edge_density, st.session_state.fixed_positions)
        st.session_state.positions = st.session_state.fixed_positions.copy()

# 图形选择逻辑
if graph_option == "随机生成图":
    # 添加保存图的按钮
    if st.sidebar.button("保存当前图"):
        edges_file, pos_file = save_graph_to_file(st.session_state.G, st.session_state.positions)
        st.sidebar.success(f"图已保存到文件: {edges_file} 和 {pos_file}")

else:  # 加载已有图
    uploaded_edges = st.sidebar.file_uploader("上传边数据文件", type="csv")
    uploaded_positions = st.sidebar.file_uploader("上传位置数据文件", type="csv")

    if uploaded_edges is not None and uploaded_positions is not None:
        # 保存上传的文件
        with open("temp_edges.csv", "wb") as f:
            f.write(uploaded_edges.getbuffer())
        with open("temp_positions.csv", "wb") as f:
            f.write(uploaded_positions.getbuffer())

        # 加载图
        st.session_state.G, st.session_state.positions = load_graph_from_file("temp_edges.csv", "temp_positions.csv")

        # 更新节点数量
        num_nodes = st.session_state.G.number_of_nodes()
        st.sidebar.write(f"加载的图有 {num_nodes} 个节点和 {st.session_state.G.number_of_edges()} 条边")
    else:
        st.warning("请上传边数据和位置数据文件")
        st.stop()

# 检查图的连通性
if not nx.is_connected(st.session_state.G):
    st.warning("生成的图不是连通的，某些节点之间可能没有路径。请重新生成图或调整边密度。")

# 检查起点和终点是否在图中
if start_node not in st.session_state.G.nodes() or end_node not in st.session_state.G.nodes():
    st.error(f"起点 {start_node} 或终点 {end_node} 不在图中，请重新选择。")
    st.stop()

# 运行选定的算法
path = None
distance = 0
expanded_nodes = 0
time_taken = 0
animation_frames = []

if algorithm == "深度优先搜索 (DFS)":
    path, distance, expanded_nodes, time_taken, animation_frames = dfs(st.session_state.G, start_node, end_node)
    algorithm_name = "深度优先搜索 (DFS)"
elif algorithm == "宽度优先搜索 (BFS)":
    path, distance, expanded_nodes, time_taken, animation_frames = bfs(st.session_state.G, start_node, end_node)
    algorithm_name = "宽度优先搜索 (BFS)"
elif algorithm == "A*算法":
    path, distance, expanded_nodes, time_taken, animation_frames = astar(st.session_state.G, start_node, end_node,
                                                       st.session_state.positions)
    algorithm_name = "A*算法"

# 在第一列显示图形可视化
with col1:
    if animation_frames and len(animation_frames) > 0:
        frame_idx = st.slider("搜索过程帧", min_value=0, max_value=len(animation_frames)-1, value=len(animation_frames)-1)
        # 距离和其他参数仅在最后一帧显示，否则只显示路径
        is_final_frame = (frame_idx == len(animation_frames)-1 and path)
        if is_final_frame:
            fig = visualize_graph(st.session_state.G, st.session_state.positions, animation_frames[frame_idx], algorithm_name, distance,
                                  expanded_nodes, time_taken)
        else:
            fig = visualize_graph(st.session_state.G, st.session_state.positions, animation_frames[frame_idx], algorithm_name)
        st.pyplot(fig)
        if not path:
            st.error(f"未找到从节点 {start_node} 到节点 {end_node} 的路径")
    else:
        fig = visualize_graph(st.session_state.G, st.session_state.positions, None, algorithm_name)
        st.pyplot(fig)
        st.error(f"未找到从节点 {start_node} 到节点 {end_node} 的路径")

# 在第二列显示算法结果
with col2:
    st.subheader("算法结果")

    if path:
        st.success(f"找到从节点 {start_node} 到节点 {end_node} 的路径")
        st.write(f"**路径:** {' -> '.join(map(str, path))}")
        st.write(f"**最短距离:** {distance:.2f}")
    else:
        st.error(f"未找到从节点 {start_node} 到节点 {end_node} 的路径")

    st.write(f"**扩展节点数:** {expanded_nodes}")
    st.write(f"**算法执行时间:** {time_taken:.6f} 秒")

    # 显示算法说明
    st.subheader("算法说明")
    if algorithm == "深度优先搜索 (DFS)":
        st.write("""
        **深度优先搜索 (DFS)** 是一种图遍历算法，它沿着一条路径尽可能深入地搜索，直到无法继续前进时回溯。

        **特点:**
        - 使用栈数据结构
        - 不一定找到最短路径
        - 空间复杂度较低
        - 适合探索所有可能的路径
        """)
    elif algorithm == "宽度优先搜索 (BFS)":
        st.write("""
        **宽度优先搜索 (BFS)** 是一种图遍历算法，它逐层探索图中的节点，先访问所有邻近节点，然后再访问下一层节点。
        本实现使用优先队列对BFS进行了改进，使其能在有权图中找到最短路径。

        **特点:**
        - 使用优先队列数据结构
        - 在有权图和无权图中都能找到最短路径
        - 空间复杂度较高
        - 适合寻找最短路径
        """)
    elif algorithm == "A*算法":
        st.write("""
        **A*算法** 是一种启发式搜索算法，结合了Dijkstra算法和贪心最佳优先搜索的特点。

        **特点:**
        - 使用优先队列和启发函数
        - 通常能找到最短路径
        - 比Dijkstra算法更高效
        - 启发函数的选择影响算法效率
        - f(n) = g(n) + h(n)，其中g(n)是从起点到当前节点的实际距离，h(n)是从当前节点到目标的估计距离
        """)

    # 显示算法比较
    st.subheader("算法比较")
    comparison_data = {
        "算法": ["深度优先搜索 (DFS)", "宽度优先搜索 (BFS)", "A*算法"],
        "保证最短路径": ["否", "是（有权图和无权图）", "是（有权图）"],
        "时间复杂度": ["O(V+E)", "O(E log V)", "O(E log V)"],
        "空间复杂度": ["O(V)", "O(V)", "O(V)"],
        "适用场景": ["探索所有路径", "寻找最短路径", "有权图最短路径"],
    }

    st.table(pd.DataFrame(comparison_data))