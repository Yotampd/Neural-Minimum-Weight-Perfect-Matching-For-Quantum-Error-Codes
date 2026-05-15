import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import get_laplacian, to_undirected

ADJ_Z_L5 = np.array(
[[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
 [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
 [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
 [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]]
)

ADJ_X_L5 = np.array(
[[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
 [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]]
)


def generate_stabilizer_adjacency(H_matrix):
    H = H_matrix.astype(int)
    # Matrix multiplication to find shared qubits
    adj = (H @ H.T)
    # Zero out diagonal (no self-loops)
    np.fill_diagonal(adj, 0)
    return adj

def to_dense_laplacian(edge_index, edge_weight, num_nodes):
    sparse_laplacian = torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        torch.Size([num_nodes, num_nodes]),
        dtype=torch.float32,
        device=edge_index.device
    )
    return sparse_laplacian.to_dense()

# ---------------------------------------------------------------------
# Calc Lap PE
# ---------------------------------------------------------------------
def precompute_laplacian_pe(adj_matrix, k_eigenvectors=8):
    G = nx.from_numpy_array(adj_matrix)
    
    num_nodes = G.number_of_nodes()
    
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    lattice_edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    
    laplacian_edge_index, laplacian_edge_weight = get_laplacian(
        lattice_edge_index, normalization='sym', num_nodes=num_nodes
    )
    
    L_sym = to_dense_laplacian(laplacian_edge_index, laplacian_edge_weight, num_nodes)
    
    eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)
    
    num_available = max(0, num_nodes - 1)
    
    if num_available >= k_eigenvectors:
        pe = eigenvectors[:, 1 : k_eigenvectors + 1]
    else:
        pe = eigenvectors[:, 1:]
        pad_amt = k_eigenvectors - num_available
        padding = torch.zeros(num_nodes, pad_amt, device=pe.device)
        pe = torch.cat([pe, padding], dim=1)
    return pe

# ---------------------------------------------------------------------

def precompute_shortest_paths(adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    
    all_paths = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))
    
    dist_map = {}
    edge_path_map = {} 
    
    for u, targets in all_paths.items():
        for v, node_path in targets.items(): # node_path is [u, w, x, v]
            if u >= v:
                continue
            
            edge_path = [] 
            
            for i in range(len(node_path) - 1):
                n1, n2 = node_path[i], node_path[i+1]
                edge_path.append(tuple(sorted((n1, n2)))) 
            
            key = (u, v)
            dist_map[key] = len(edge_path) 
            edge_path_map[key] = edge_path 
            
    return dist_map, edge_path_map


def precompute_boundary_paths(pc_matrix, stab_type, L):
    

    AUG_H = np.zeros((pc_matrix.shape[0] + L - 1, pc_matrix.shape[1]), dtype=int)

    if L == 5:
        if stab_type == "Z":
            AUG_H[0:12, :] = pc_matrix
            AUG_H[12, [0, 1]] = 1
            AUG_H[13, [2, 3]] = 1
            AUG_H[14, [21, 22]] = 1
            AUG_H[15, [23, 24]] = 1
        elif stab_type == "X":
            AUG_H[0:12, :] = pc_matrix
            AUG_H[12, [5, 10]] = 1
            AUG_H[13, [15, 20]] = 1
            AUG_H[14, [4, 9]] = 1
            AUG_H[15, [14, 19]] = 1
        else:
            raise ValueError("incorrect stab type")
    
    elif L == 3:
        if stab_type == "Z":
            AUG_H[0:4, :] = pc_matrix
            AUG_H[4, [0, 1]] = 1
            AUG_H[5, [7, 8]] = 1
        elif stab_type == "X":
            AUG_H[0:4, :] = pc_matrix
            AUG_H[4, [3, 6]] = 1
            AUG_H[5, [2, 5]] = 1

    elif L == 7:
        if stab_type == "Z":
            AUG_H[0:24, :] = pc_matrix
            AUG_H[24, [0, 1]] = 1
            AUG_H[25, [2, 3]] = 1
            AUG_H[26, [4, 5]] = 1
            AUG_H[27, [43, 44]] = 1
            AUG_H[28, [45, 46]] = 1
            AUG_H[29, [47, 48]] = 1
        elif stab_type == "X":
            AUG_H[0:24, :] = pc_matrix
            AUG_H[24, [7, 14]] = 1
            AUG_H[25, [21, 28]] = 1
            AUG_H[26, [35, 42]] = 1
            AUG_H[27, [6, 13]] = 1
            AUG_H[28, [20, 27]] = 1
            AUG_H[29, [34, 41]] = 1
    elif L == 9:
        if stab_type == "Z":
            AUG_H[0:40, :] = pc_matrix
            AUG_H[40, [0, 1]] = 1
            AUG_H[41, [2, 3]] = 1
            AUG_H[42, [4, 5]] = 1
            AUG_H[43, [6, 7]] = 1
            AUG_H[44, [73, 74]] = 1
            AUG_H[45, [75, 76]] = 1
            AUG_H[46, [77, 78]] = 1
            AUG_H[47, [79, 80]] = 1
        elif stab_type == "X":
            AUG_H[0:40, :] = pc_matrix
            AUG_H[40, [9, 18]] = 1
            AUG_H[41, [27, 36]] = 1
            AUG_H[42, [45, 54]] = 1
            AUG_H[43, [63, 72]] = 1
            AUG_H[44, [8, 17]] = 1
            AUG_H[45, [26, 35]] = 1
            AUG_H[46, [44, 53]] = 1
            AUG_H[47, [62, 71]] = 1
        
    Full_Graph = (AUG_H @ AUG_H.T) % 2 
    G = nx.from_numpy_array(Full_Graph)

    num_real_nodes = (L*L - 1)//2 
    num_virtual_nodes = L - 1
    virtual_nodes = set(range(num_real_nodes, num_real_nodes + num_virtual_nodes))
    
    all_paths = {}

    for u in range(num_real_nodes):
        all_paths[u] = nx.shortest_path(G, source = u)

    boundary_dist_map = {}
    boundary_qubit_path_map = {}


    for u in range(num_real_nodes): 
        min_dist = float("inf")
        best_path = None

        for v in virtual_nodes:
            path = all_paths[u][v]
            dist = len(path) - 1 

            if dist < min_dist:
                min_dist = dist
                best_path = path

        # calculate fault id's to boundary
        boundary_dist_map[u] = min_dist
        qubit_path = find_qubit_path_to_bounds(best_path, AUG_H)
        boundary_qubit_path_map[u] = qubit_path

    return boundary_dist_map, boundary_qubit_path_map


def find_qubit_path_to_bounds(path, aug_pc_mat):
    final_qubit_path = []
    edge_path = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_path.append((u, v))
    
    
    for u, v in edge_path:
        row_u = aug_pc_mat[u] 
        row_v = aug_pc_mat[v] 
        
        common_qubit_indices = np.nonzero(row_u & row_v)[0]
        
        if len(common_qubit_indices) == 0:
            raise ValueError(f"No common qubit found for edge ({u}, {v})")
        
        
        common_qubit = common_qubit_indices[0]
        
        # 3. Add the qubit to the path
        final_qubit_path.append(common_qubit)
        
    return final_qubit_path    
        
def generate_rotated_x_coords(L):
    coords = []
    
    for l in range(L - 1):
        for k in range((L - 1) // 2):
            
            if l == 0:
                u = 0.
                v = float(2 * k + 1)
                coords.append([u, v])
            elif l == L - 2:
                u = float(L)
                v = float(2 * k)
                coords.append([u, v])
                
            u = float(l + 1)
            
            if l % 2 == 0:
                v = float(2 * k)     
            else:
                v = float(2 * k + 1) 
                
            coords.append([u, v])
            
    return torch.tensor(coords, dtype=torch.float32)

def generate_rotated_z_coords(L):
    coords = []
    
    for l in range(L - 1):
        u = float(l)
        
        if l % 2 == 0:
            v = 0.
        else:
            v = float(L)
            
        coords.append([u, v])
        
        for k in range((L - 1) // 2):
            if l % 2 == 0:
                v_bulk = float(2 * k + 2)
            else:
                v_bulk = float(2 * k + 1)
                
            coords.append([u, v_bulk])
            
    return torch.tensor(coords, dtype=torch.float32)