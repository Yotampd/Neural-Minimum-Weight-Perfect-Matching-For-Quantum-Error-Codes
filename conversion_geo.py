import networkx as nx
import torch 
from Graph_C import build_comp_graph, build_edges_vector
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_undirected
import torch_sparse
import torch.nn.functional as F
from Data import RotatedSurfaceCode, Get_rotated_surface_Code
import numpy as np 


#x = node feature matrix - every row is a node, every column is a feature for that node
# G.nodes is index:dict  where the dict containes the pos and type vec

def convert_graph_to_torch(G, label_vector, L, z_tensor, syndrome_tensor, stab_t):
    num_nodes = G.number_of_nodes()

    x = torch.zeros(L * L, 5, dtype=torch.float32)  #shape [num nodes, 5], 5 is the number of features

    center = (L // 2, L // 2)

    #### node feature matrix ####
    #node_features = []
    for node_id in G.nodes:
        type_vec = G.nodes[node_id]["type_vec"] # error type
        pos = G.nodes[node_id]["pos"] #cooridantes
        pos_vec = list(pos) #tuple to list
        # euclidean distance to center (L/2, L/2)
        dx = pos[0] - center[0]
        dy = pos[1] - center[1]
        dist_to_center = (dx ** 2 + dy ** 2) ** 0.5

        feature_vec = type_vec + pos_vec + [dist_to_center] #concatantion 
        x[node_id] = torch.tensor(feature_vec, dtype=torch.float32)


    edge_list = []
    edge_features = []
    full_labels = []


    for u, v, attrs in G.edges(data = True): #attrs is the attribute dict for edge, true inculdes the dict in the loop
        # lookup index of (u, v) in label vector
        idx = attrs["edge_index"]  #attrs is the dictionary for u, v - iterating over tuple with 2 vertices and dict for them

        key = (u, v) if u <= v else (v, u)
        
        edge_list.append((u, v))
        edge_list.append((v, u))  # both directions, shape is [2 X num_edges]
        
        pos_u = G.nodes[u]["pos"]
        pos_v = G.nodes[v]["pos"]

        dx = abs(pos_v[0] - pos_u[0])
        dy = abs(pos_v[1] - pos_u[1])
        
        dx = min(dx, L - dx)
        dy = min(dy, L - dy)

        dist = attrs["dist"]

        edge_features.append([dist, dx, dy]) #for each edge add the dist in both directions, the dist idx corresponds to the edge idx
        edge_features.append([dist, dx, dy])

        full_labels.append(label_vector[idx]) #one for every edge, label vector is in length of num edges(not duplicated)
        full_labels.append(label_vector[idx]) #full_labels is in length of 2Xnum_edges

    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() #each column is an edge, each row is a node(sec/trg). shape: [2, 2 X num_edges] 
        edge_attr = torch.tensor(edge_features, dtype=torch.float32) #each row is an edge, colums are features. currently is a vector with distances(only one feature) - edge feature matrix [2 X num_edges]
    
    y = torch.tensor(full_labels, dtype=torch.float32) #final label vector, shape: [2 X num_edges]


    if stab_t == "X":
        stab_flag = 0 # x stabilizers correcting z errors
    elif stab_t == "Z":
        stab_flag = 1 # z stabilizers correcting x errors
    else:
        raise ValueError("unknown stab type")
    
    stab_flag_tensor = torch.tensor(stab_flag, dtype=torch.int64)
    

    # Only check for valid node IDs if there are actually edges
    if edge_index.numel() > 0:
        assert edge_index.max().item() < x.shape[0], \
            f"edge_index contains invalid node id: max={edge_index.max().item()}, num_nodes={x.shape[0]}"
            
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, z=z_tensor, syndrome = syndrome_tensor, stab_flag = stab_flag_tensor) #the data comes with node and edge feature matrices and edge index tensor so the gnn will know the connectivity

#----------------------------------------------------------------------------------------------------
# Global cache for pre-computed PEs
PE_CACHE = {}

def add_laplacian_pe(data, L, k_eigenvectors=8):
    global PE_CACHE
    num_nodes = L * L
    
    key = (L, k_eigenvectors)
    if key in PE_CACHE:
        pe = PE_CACHE[key].to(data.x.device)
    else:
        print(f"Cache miss. Computing fixed Laplacian PE for L={L}...")
        
        edges = []
        for i in range(L):
            for j in range(L):
                node = i * L + j
                neighbor_right = i * L + (j + 1) % L
                neighbor_down = ((i + 1) % L) * L + j
                edges.append([node, neighbor_right])
                edges.append([node, neighbor_down])
        
        # Convert to [2, E] tensor and make it undirected
        lattice_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        lattice_edge_index = to_undirected(lattice_edge_index, num_nodes=num_nodes)

        laplacian_edge_index, laplacian_edge_weight = get_laplacian(
            lattice_edge_index,
            normalization='sym',
            num_nodes=num_nodes
        )

        L_sym = to_dense_laplacian(laplacian_edge_index, laplacian_edge_weight, num_nodes)

        eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)
        
        pe = eigenvectors[:, 1 : k_eigenvectors + 1]

        PE_CACHE[key] = pe
        pe = pe.to(data.x.device)

    data.x = torch.cat([data.x, pe], dim=1)
    
    return data


def to_dense_laplacian(edge_index, edge_weight, num_nodes): #spare to dense
    sparse_laplacian = torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        torch.Size([num_nodes, num_nodes]),
        dtype=torch.float32,
        device=edge_index.device
    )
    return sparse_laplacian.to_dense()


# Rotated laplacian pe

def add_laplacian_pe_rotated(data, L, k_eigenvectors=8, stab_type='Z'):
    global PE_CACHE
    key = (L, k_eigenvectors, 'rotated', stab_type)
    
    if key in PE_CACHE:
        pe = PE_CACHE[key].to(data.x.device)
    else:
        print(f"Cache miss. Computing fixed Laplacian PE for L={L}, code=rotated, stab={stab_type}...")
        lattice_edge_index, num_nodes = get_rotated_stabilizer_graph(L, stab_type)
        
        laplacian_edge_index, laplacian_edge_weight = get_laplacian(
            lattice_edge_index, normalization='sym', num_nodes=num_nodes
        )
        L_sym = to_dense_laplacian(laplacian_edge_index, laplacian_edge_weight, num_nodes)
        eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)
        pe = eigenvectors[:, 1 : k_eigenvectors + 1] # [N, k]
        PE_CACHE[key] = pe
        pe = pe.to(data.x.device)

    # data.x shape is [N*2, 3]
    # pe shape is [N, 8]
    num_stabs_per_type = (L - 1) * (L + 1) // 2
    total_nodes = num_stabs_per_type * 2
    
    full_pe = torch.zeros(total_nodes, k_eigenvectors, device=data.x.device, dtype=data.x.dtype)
    
    if stab_type == 'Z':
        full_pe[:num_stabs_per_type] = pe
    else: # stab_type == 'X'
        full_pe[num_stabs_per_type:] = pe
        
    data.x = torch.cat([data.x, full_pe], dim=1)
    return data


def to_dense_laplacian(edge_index, edge_weight, num_nodes): 
    sparse_laplacian = torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        torch.Size([num_nodes, num_nodes]),
        dtype=torch.float32,
        device=edge_index.device
    )
    return sparse_laplacian.to_dense()


# --- Rotated PE Function ---
def get_rotated_stabilizer_graph(L, stab_type="Z"):
    
    code = RotatedSurfaceCode(L)
    if stab_type == "Z":
        H_matrix = code.H_Z # (num_stabs, num_qubits)
    else: # stab_type == "X"
        H_matrix = code.H_X # (num_stabs, num_qubits)
    
    num_nodes = H_matrix.shape[0]
    
    H_torch = torch.from_numpy(H_matrix).float()
    
    adj_matrix = torch.matmul(H_torch, H_torch.t())
    
    adj_matrix.fill_diagonal_(0)
    
    edge_index = (adj_matrix > 0).nonzero(as_tuple=False).t().contiguous()
    
    lattice_edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    
    return lattice_edge_index, num_nodes
#----------------------------------------------------------------------------------------------------




def combine_graphs(data_Z, data_X, L):
    combined_data = Data()

    # Concatenate node features and labels
    combined_data.x = torch.cat([data_Z.x, data_X.x], dim=0)
    combined_data.y = torch.cat([data_Z.y, data_X.y], dim=0)

    # Combine edge indices and attributes
    # The node indices of data_X need to be shifted
    num_nodes_Z = data_Z.num_nodes
    combined_edge_index = torch.cat(
        [data_Z.edge_index, data_X.edge_index + (L*L)], dim=1
    )
    combined_data.edge_index = combined_edge_index

    # Add a new edge attribute to distinguish between Z and X components
    # The new attribute is at index 3 (fourth column)
    attr_Z = F.pad(data_Z.edge_attr, (0, 1), value=1)
    attr_X = F.pad(data_X.edge_attr, (0, 1), value=0)
    combined_data.edge_attr = torch.cat([attr_Z, attr_X], dim=0)
    
    # Store the original data tensors for decoding
    combined_data.z_Z = data_Z.z
    combined_data.z_X = data_X.z

    combined_data.syndrome_Z = data_Z.syndrome
    combined_data.syndrome_X = data_X.syndrome

    combined_data.stab_t_Z = data_Z.stab_t
    combined_data.stab_t_X = data_X.stab_t
    
    combined_data.num_nodes_Z = num_nodes_Z
    combined_data.L = data_Z.L
    combined_data.id = data_Z.id

    combined_data.y_Z = data_Z.y
    combined_data.y_X = data_X.y

    combined_data.syndrome = torch.cat([data_Z.syndrome, data_X.syndrome], dim=0)
    

    return combined_data



def convert_graph_to_torch_rotated(G, label_vector, L, z_tensor, syndrome_tensor, stab_t, precomputed_data):
    if L == 5: 
        num_stabs_per_type = 12
    else:
        num_stabs_per_type = (L * L - 1) // 2
    virtual_node_start_idx = num_stabs_per_type
    num_nodes_real = G.number_of_nodes()

    num_nodes_total = 1 + num_stabs_per_type #adding one virtual 
    

    num_qubits_per_type = L*L
    num_nodes = num_stabs_per_type

    H_full, _ = Get_rotated_surface_Code(L, full_H=True)
    H_Z = H_full[0:num_stabs_per_type, 0:num_qubits_per_type]
    H_X = H_full[num_stabs_per_type:, num_qubits_per_type:] 

    if stab_t == "Z":
        H_current = H_Z
        type_vec = [0.0, 1.0]

        real_coords = precomputed_data['z_coords']
        u_min, u_max = real_coords[:, 0].min(), real_coords[:, 0].max()
        v_mean = real_coords[:, 1].mean()
        virtual_coord = torch.tensor([(u_min + u_max)/2.0, v_mean])

    elif stab_t == "X":
        H_current = H_X
        type_vec = [1.0, 0.0]

        real_coords = precomputed_data['x_coords']
        v_min, v_max = real_coords[:, 1].min(), real_coords[:, 1].max()
        u_mean = real_coords[:, 0].mean()
        virtual_coord = torch.tensor([u_mean, (v_min + v_max)/2.0])
    else:
        raise ValueError("unknown stab type")

    node_feat_dim = 5
    x = torch.zeros(num_nodes_total, node_feat_dim, dtype=torch.float32) #changed to add one virtual node

    avg_loc_lookup = np.zeros(num_stabs_per_type, dtype=float)
    for i in range(num_stabs_per_type):
        qubit_indices = np.nonzero(H_current[i])[0]
        if len(qubit_indices) > 0:
            avg_loc_lookup[i] = np.mean(qubit_indices)
    
    avg_loc_lookup = torch.tensor(avg_loc_lookup, dtype=torch.float32)


    syndrome_tensor_padded = F.pad(syndrome_tensor, (0, 1), value=0.0)

    node_feat_dim = 5
    x = torch.zeros(num_nodes_total, node_feat_dim, dtype=torch.float32)

    # real nodes
    for i in range(num_stabs_per_type):
        dynamic_avg_loc = syndrome_tensor[i] * avg_loc_lookup[i]
        coord = real_coords[i]
        x[i] = torch.tensor(type_vec + coord.tolist() + [dynamic_avg_loc])
    
    avg_v_node_loc = (num_qubits_per_type - 1) / 2 
    
    x[virtual_node_start_idx] = torch.tensor(type_vec + virtual_coord.tolist() + [avg_v_node_loc])


    # ------ Edges --------------

    edge_list = []
    edge_features = []
    full_labels = []

    for u, v, attrs in G.edges(data = True):
        idx = attrs["edge_index"]
        
        edge_list.append((u, v))
        edge_list.append((v, u))  # both directions

        pos_u = x[u, 2:4]
        pos_v = x[v, 2:4]

        diff = pos_u - pos_v
        du = torch.abs(diff[0])
        dv = torch.abs(diff[1])

        # Edge feature: [distance]
        dist = attrs["dist"]
        
        edge_features.append([dist, du, dv])
        edge_features.append([dist, du, dv])

        full_labels.append(label_vector[idx])
        full_labels.append(label_vector[idx])

    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32) 
    
    
    y = torch.tensor(full_labels, dtype=torch.float32)

    stab_flag_tensor = torch.tensor(1 if stab_t == "Z" else 0, dtype=torch.int64)

    if edge_index.numel() > 0:
        assert edge_index.max().item() < x.shape[0], \
            f"edge_index contains invalid node id: max={edge_index.max().item()}, num_nodes={x.shape[0]}"

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, z=z_tensor, syndrome = syndrome_tensor_padded, stab_flag = stab_flag_tensor)



def combine_graphs_rotated(data_Z, data_X, L):
    combined_data = Data()

    combined_data.x = torch.cat([data_Z.x, data_X.x], dim=0)
    combined_data.y = torch.cat([data_Z.y, data_X.y], dim=0)

    num_nodes_Z = data_Z.x.shape[0]
    
    combined_edge_index = torch.cat(
        [data_Z.edge_index, data_X.edge_index + num_nodes_Z], dim=1
    )
    combined_data.edge_index = combined_edge_index

    attr_Z = F.pad(data_Z.edge_attr, (0,1), value=0.0) 
    attr_Z[:, -1] = 1.0 # component flag: [dist, 0, 0, 1]

    # X-component edges
    attr_X = F.pad(data_X.edge_attr, (0, 1), value=0.0)
    
    combined_data.edge_attr = torch.cat([attr_Z, attr_X], dim=0)
    
    combined_data.z_Z = data_Z.z
    combined_data.z_X = data_X.z

    combined_data.syndrome_Z = data_Z.syndrome
    combined_data.syndrome_X = data_X.syndrome

    combined_data.stab_t_Z = data_Z.stab_t
    combined_data.stab_t_X = data_X.stab_t
    
    combined_data.num_nodes_Z = num_nodes_Z
    combined_data.L = data_Z.L
    combined_data.id = data_Z.id

    combined_data.y_Z = data_Z.y
    combined_data.y_X = data_X.y

    combined_data.syndrome = torch.cat([data_Z.syndrome, data_X.syndrome], dim=0)
    
    return combined_data

def add_precomputed_pe_rotated(combined_data, pe_z, pe_x):
    
    pe_z = pe_z.to(combined_data.x.device)
    pe_x = pe_x.to(combined_data.x.device)
    
    k_eigenvectors = pe_z.shape[1]

    pe_z_virtual = torch.mean(pe_z, dim=0, keepdim=True)
    pe_x_virtual = torch.mean(pe_x, dim=0, keepdim=True)

    full_pe = torch.cat([pe_z, pe_z_virtual, pe_x, pe_x_virtual], dim=0)
    
    combined_data.x = torch.cat([combined_data.x, full_pe], dim=1)
    
    return combined_data