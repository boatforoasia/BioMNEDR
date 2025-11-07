from msi.msi import MSI
from diff_prof.diffusion_profiles import DiffusionProfiles
import multiprocessing
import numpy as np
import pickle
import networkx as nx
import scipy.sparse as sp

from tests.msi import test_msi
from tests.diff_prof import test_diffusion_profiles
import copy


# 加载MSI
msi = MSI()
msi.load()
drugs_indications_idx = [msi.node2idx[key] for key in msi.drug_or_indication2proteins.keys()]   # 提取drug和indications的id


# 生成meta-graph
def generate_meta_graph(edges_dict):
    # 创建一个新的空有向图对象
    meta_G = copy.deepcopy(msi.graph)
    meta_G.remove_edges_from(list(meta_G.edges()))

    # 仅保留满足条件的边，edge_dict()中的特定边
    for source, targets in edges_dict.items():
        for target in targets:
            meta_G.add_edge(source, target)

    return meta_G

drugs2proteins_G = generate_meta_graph(msi.drug2proteins)
proteins2proteins_G = generate_meta_graph(msi.protein2proteins)
proteins2indications_G = generate_meta_graph(msi.protein2indications)
proteins2functions_G = generate_meta_graph(msi.protein2functions)
functions2proteins_G = generate_meta_graph(msi.function2proteins)
functions2functions_G = generate_meta_graph(msi.function2functions)


# 生成稀疏矩阵
# 将NetworkX图转换为稀疏矩阵
initial_M = sp.csr_matrix(nx.adjacency_matrix(msi.graph))
drugs2proteins_M = sp.csr_matrix(nx.adjacency_matrix(drugs2proteins_G))
proteins2functions_M = sp.csr_matrix(nx.adjacency_matrix(proteins2functions_G))
functions2proteins_M = sp.csr_matrix(nx.adjacency_matrix(functions2proteins_G))
proteins2proteins_M = sp.csr_matrix(nx.adjacency_matrix(proteins2proteins_G))
proteins2indications_M = sp.csr_matrix(nx.adjacency_matrix(proteins2indications_G))
functions2functions_M = sp.csr_matrix(nx.adjacency_matrix(functions2functions_G))

d2p2p2p2i_M = drugs2proteins_M.dot(proteins2proteins_M).dot(proteins2proteins_M).dot(proteins2indications_M)
d2p2p2p2i = nx.from_scipy_sparse_array(d2p2p2p2i_M)

d2p2f2p2i_M = drugs2proteins_M.dot(proteins2functions_M).dot(functions2proteins_M).dot(proteins2indications_M)
d2p2f2p2i = nx.from_scipy_sparse_array(d2p2f2p2i_M)

d2p2f2f2p2i_M = drugs2proteins_M.dot(proteins2functions_M).dot(functions2functions_M).dot(functions2proteins_M).dot(proteins2indications_M)
d2p2f2f2p2i = nx.from_scipy_sparse_array(d2p2f2f2p2i_M)

d2p2p2i_M = drugs2proteins_M.dot(proteins2proteins_M).dot(proteins2indications_M)
d2p2p2i = nx.from_scipy_sparse_array(d2p2p2i_M)

d2p2i_M = drugs2proteins_M.dot(proteins2indications_M)
d2p2i = nx.from_scipy_sparse_array(d2p2i_M)

# 获取稀疏矩阵中边的权重值
weights = d2p2p2i_M.data
nx.set_edge_attributes(d2p2p2i, values=dict(zip(d2p2p2i.edges(), weights)), name='weight')

weights = d2p2i_M.data
nx.set_edge_attributes(d2p2i, values=dict(zip(d2p2i.edges(), weights)), name='weight')

weights = d2p2p2p2i_M.data
nx.set_edge_attributes(d2p2p2p2i, values=dict(zip(d2p2p2p2i.edges(), weights)), name='weight')

weights = d2p2f2p2i_M.data
nx.set_edge_attributes(d2p2f2p2i, values=dict(zip(d2p2f2p2i.edges(), weights)), name='weight')

weights = d2p2f2f2p2i_M.data
nx.set_edge_attributes(d2p2f2f2p2i, values=dict(zip(d2p2f2f2p2i.edges(), weights)), name='weight')

# weights = d2p2f2p2i_M.data
# nx.set_edge_attributes(d2p2f2p2i, values=dict(zip(d2p2f2p2i.edges(), weights)), name='weight')


# 带权随机游走
# def random_walks(G, start_nodes, num_walks, walk_length):
#     walks = []
#
#     for start_node in start_nodes:
#
#         for _ in range(num_walks):
#             if list(G.neighbors(start_node)):
#                 current_node = start_node
#                 walk = [current_node]
#
#                 for _ in range(walk_length):
#                     neighbors = list(G.neighbors(current_node))  # 可能有些节点没有边，因此加上了连接自己的边
#                     weights = [G[current_node][neighbor]['weight'] for neighbor in neighbors]
#
#                     # 按照边的权重进行随机选择下一个节点
#                     next_node = np.random.choice(neighbors, p=weights / np.sum(weights))
#
#                     walk.append(next_node)
#                     current_node = next_node
#
#             else:
#                 walk = [start_node] * walk_length
#             walks.append(walk)
#
#     return walks

num_walks = 1000  # 游走次数
walk_length = 100  # 每次游走的步长
start_nodes = drugs_indications_idx  # 起始节点列表

from tqdm import trange
import random

def random_walks(G, start_nodes, num_walks, walk_length):
    # non-weighted
    walks = []

    for start_node in start_nodes:
        for _ in range(num_walks):
            if list(G.neighbors(start_node)):
                current_node = start_node
                walk = [current_node]

                for _ in range(walk_length):
                    neighbors = list(G.neighbors(current_node))
                    # weights = [G[current_node][neighbor]['weight'] for neighbor in neighbors]

                    next_node = random.choice(neighbors)

                    walk.append(next_node)
                    current_node = next_node

            else:
                walk = [start_node] * walk_length
            walks.append(walk)

            # 显示进度条
            progress_bar.set_postfix({'Walks': len(walks)})
            progress_bar.update()

    return walks


# weighted random walk
# def random_walks(G, start_nodes, num_walks, walk_length):
#     walks = []
#
#     for start_node in start_nodes:
#         for _ in range(num_walks):
#             if list(G.neighbors(start_node)):
#                 current_node = start_node
#                 walk = [current_node]
#
#                 for _ in range(walk_length):
#                     neighbors = list(G.neighbors(current_node))
#                     weights = [G[current_node][neighbor]['weight'] for neighbor in neighbors]
#
#                     next_node = np.random.choice(neighbors, p=weights / np.sum(weights))
#
#                     walk.append(next_node)
#                     current_node = next_node
#
#             else:
#                 walk = [start_node] * walk_length
#             walks.append(walk)
#
#             # 显示进度条
#             progress_bar.set_postfix({'Walks': len(walks)})
#             progress_bar.update()
#
#     return walks

import numpy as np


def weighted_random_walks(G, start_nodes, num_walks, walk_length):
    walks = []

    # 预先计算并存储概率值
    probabilities_dict = {}
    for node in drugs_indications_idx:
        neighbors = list(G.neighbors(node))
        weights = np.array([G[node][neighbor]['weight'] for neighbor in neighbors], dtype=np.float64)
        total_weight = np.sum(weights)

        if total_weight > 0:
            probabilities = weights / total_weight
        else:
            probabilities = np.zeros_like(weights)

        probabilities_dict[node] = probabilities

    for start_node in start_nodes:
        for _ in range(num_walks):
            if list(G.neighbors(start_node)):
                current_node = start_node
                walk = [current_node]

                for _ in range(walk_length):
                    neighbors = list(G.neighbors(current_node))
                    probabilities = probabilities_dict[current_node]

                    next_node = np.random.choice(neighbors, p=probabilities)

                    walk.append(next_node)
                    current_node = next_node

            else:
                walk = [start_node] * walk_length
            walks.append(walk)

            # 显示进度条
            progress_bar.set_postfix({'Walks': len(walks)})
            progress_bar.update()

    return walks

def softmax_weighted_random_walks(G, start_nodes, num_walks, walk_length):
    walks = []

    # 预先计算并存储概率值
    probabilities_dict = {}
    for node in drugs_indications_idx:
        neighbors = list(G.neighbors(node))
        weights = np.array([G[node][neighbor]['weight'] for neighbor in neighbors], dtype=np.float64)
        total_weight = np.sum(weights)

        if total_weight > 0:
            weights = weights / max(weights)
            probabilities = np.exp(weights) / np.sum(np.exp(weights))
            # probabilities = np.exp(weights) / np.sum(np.exp(weights))
        else:
            probabilities = np.zeros_like(weights)

        probabilities_dict[node] = probabilities

    for start_node in start_nodes:
        for _ in range(num_walks):
            if list(G.neighbors(start_node)):
                current_node = start_node
                walk = [current_node]

                for _ in range(walk_length):
                    neighbors = list(G.neighbors(current_node))
                    probabilities = probabilities_dict[current_node]

                    next_node = np.random.choice(neighbors, p=probabilities)

                    walk.append(next_node)
                    current_node = next_node

            else:
                walk = [start_node] * walk_length
            walks.append(walk)

            # 显示进度条
            progress_bar.set_postfix({'Walks': len(walks)})
            progress_bar.update()

    return walks

def sqrt_weighted_random_walks(G, start_nodes, num_walks, walk_length):
    walks = []

    # 预先计算并存储概率值
    probabilities_dict = {}
    for node in drugs_indications_idx:
        neighbors = list(G.neighbors(node))
        weights = np.array([G[node][neighbor]['weight'] for neighbor in neighbors], dtype=np.float64)
        total_weight = np.sum(weights)

        if total_weight > 0:
            probabilities = np.sqrt(weights) / np.sum(np.sqrt(weights))
        else:
            probabilities = np.zeros_like(weights)

        probabilities_dict[node] = probabilities

    for start_node in start_nodes:
        for _ in range(num_walks):
            if list(G.neighbors(start_node)):
                current_node = start_node
                walk = [current_node]

                for _ in range(walk_length):
                    neighbors = list(G.neighbors(current_node))
                    probabilities = probabilities_dict[current_node]

                    next_node = np.random.choice(neighbors, p=probabilities)

                    walk.append(next_node)
                    current_node = next_node

            else:
                walk = [start_node] * walk_length
            walks.append(walk)

            # 显示进度条
            progress_bar.set_postfix({'Walks': len(walks)})
            progress_bar.update()

    return walks

# 设置进度条
# total_iterations = len(start_nodes) * num_walks
# progress_bar = trange(total_iterations, desc='Progress', unit='iteration')
#
# # 调用random_walks函数
# # walks = random_walks(G, start_nodes, num_walks, walk_length)
# d2p2p2i_walks = random_walks(d2p2p2i, start_nodes, num_walks, walk_length)
# # d2p2f2p2i_walks = random_walks(d2p2f2p2i, start_nodes, num_walks, walk_length)
#
# # 完成进度条
# progress_bar.close()

def path_to_file(walks, file_name = 'walks/demo.txt'):
    output_path = open(file_name, "w")
    for walk in walks:
        outline = " ".join(msi.graph.nodes[msi.idx2node[id]]['type'][0]+'-%s' %id for id in walk)
        print(outline, file=output_path)

# path_to_file(walks = d2p2p2i_walks, file_name = 'walks/d2p2p2iwalks4MSI_1000_100.txt')
# path_to_file(walks = d2p2f2p2i_walks, file_name = 'walks/d2p2f2p2iwalks4MSI_30_50.txt')


# paths_list = [d2p2i, d2p2p2i, d2p2p2p2i, d2p2f2p2i]
# paths_list = [d2p2f2p2i, d2p2f2f2p2i]
paths_dict = {d2p2i: "d2p2i", d2p2p2i: "d2p2p2i", d2p2p2p2i: "d2p2p2p2i", d2p2f2p2i: "d2p2f2p2i", d2p2f2f2p2i: "d2p2f2f2p2i"}
# for meta_path in paths_list:
#     total_iterations = len(start_nodes) * num_walks
#     progress_bar = trange(total_iterations, desc='Progress', unit='iteration')
#
#     # 调用 random_walks 函数
#     walks = random_walks(meta_path, start_nodes, num_walks, walk_length)
#     # walks = weighted_random_walks(meta_path, start_nodes, num_walks, walk_length)
#
#     # 完成进度条
#     progress_bar.close()
#
#     path_to_file(walks=walks, file_name='walks/' + paths_dict[meta_path] + 'randomwalks4MSI_' + str(num_walks) + '_' + str(walk_length) + '.txt')
    # path_to_file(walks=walks, file_name='walks/' + paths_dict[meta_path] + 'walks4MSI_' + str(num_walks) + '_' + str(walk_length) + '.txt')


# paths_list = [d2p2p2i, d2p2i, d2p2p2p2i, d2p2f2p2i, d2p2f2f2p2i]
paths_list = [d2p2p2i, d2p2i, d2p2f2p2i]
for meta_path in paths_list:
    total_iterations = len(start_nodes) * num_walks
    progress_bar = trange(total_iterations, desc='Progress', unit='iteration')

    # 调用 random_walks 函数
    # walks = random_walks(meta_path, start_nodes, num_walks, walk_length)
    # walks = weighted_random_walks(meta_path, start_nodes, num_walks, walk_length)
    walks = sqrt_weighted_random_walks(meta_path, start_nodes, num_walks, walk_length)

    # 完成进度条
    progress_bar.close()

    path_to_file(walks=walks, file_name='walks/' + paths_dict[meta_path] + 'sqrtwalks4MSI_' + str(num_walks) + '_' + str(walk_length) + '.txt')




# paths_list = ['d2p2i', 'd2p2p2i', 'd2p2p2p2i', 'd2p2f2p2i']
# for meta_path in paths_list:
#     total_iterations = len(start_nodes) * num_walks
#     progress_bar = trange(total_iterations, desc='Progress', unit='iteration')
#
#     # 调用random_walks函数
#     # walks = random_walks(G, start_nodes, num_walks, walk_length)
#     walks = random_walks(meta_path, start_nodes, num_walks, walk_length)
#     # d2p2f2p2i_walks = random_walks(d2p2f2p2i, start_nodes, num_walks, walk_length)
#     # 完成进度条
#     progress_bar.close()
#
#     path_to_file(walks=walks, file_name='walks/' + meta_path + 'walks4MSI_' + string(num_walks) + '_' + string(walk_length) + '.txt')
