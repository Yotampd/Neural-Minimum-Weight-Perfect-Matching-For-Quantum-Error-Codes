import numpy as np
import networkx as nx
from itertools import combinations
import time
from Data import Get_rotated_surface_Code 

def get_qubits_from_edge_path(edge_path, H_matrix):
    qubit_path = []
    
    for u, v in edge_path:
        row_u = H_matrix[u] 
        row_v = H_matrix[v] 
        
        common_qubit_indices = np.nonzero(row_u & row_v)[0]
        
        if len(common_qubit_indices) == 0:
            raise ValueError(f"No common qubit found for edge ({u}, {v})")
        
        common_qubit = common_qubit_indices[0]
        
        qubit_path.append(common_qubit)
        
    return qubit_path
# ---------------------------------------------------------------
#  Precompute & Checks
# ---------------------------------------------------------------

def get_qubits_from_edge_path_cached(u, v, H_matrix, cache):
    key = (min(u, v), max(u, v))
    if key in cache: return cache[key]

    row_u = H_matrix[u] 
    row_v = H_matrix[v] 
    common = np.nonzero(row_u & row_v)[0]
    
    path = [common[0]] if len(common) > 0 else []
    cache[key] = path
    return path

def precompute_logical_effects(edge_path_map, boundary_edge_path_map, H_matrix, logical_vec, cache):
    edge_logical_map = {}       
    boundary_logical_map = {}   

    #Internal edges
    for (u, v), path_edges in edge_path_map.items():
        qubit_path = []
        for s1, s2 in path_edges:
            q_list = get_qubits_from_edge_path_cached(s1, s2, H_matrix, cache)
            qubit_path.extend(q_list)
        
        if len(qubit_path) > 0:
            parity = np.sum(logical_vec[qubit_path]) % 2
        else:
            parity = 0
        edge_logical_map[(min(u, v), max(u, v))] = parity

    #Boundary edges
    for u, qubit_list in boundary_edge_path_map.items():
        if len(qubit_list) > 0:
            parity = np.sum(logical_vec[qubit_list]) % 2
        else:
            parity = 0
        boundary_logical_map[u] = parity

    return edge_logical_map, boundary_logical_map


def _build_ground_truth_generic(z, syndrome_np, G_real, logical_matrix, final_testing, precomputed_data, L, is_x_type):
    num_stabs_per_type = (L*L - 1) // 2 
    num_qubits_per_type = L * L 
    model_virtual_node_idx = num_stabs_per_type

    # Select relevant maps
    if is_x_type:
        prefix = 'x'
        H_full, _ = Get_rotated_surface_Code(L, full_H=True)
        H_sub = H_full[num_stabs_per_type:, num_qubits_per_type:] 
    else:
        prefix = 'z'
        H_full, _ = Get_rotated_surface_Code(L, full_H=True)
        H_sub = H_full[0:num_stabs_per_type, 0:num_qubits_per_type] 

    dist_map = precomputed_data[f'{prefix}_dist_map']
    boundary_dist_map = precomputed_data[f'{prefix}_boundary_dist_map']
    edge_path_map = precomputed_data[f'{prefix}_edge_path_map']
    boundary_edge_path_map = precomputed_data[f'{prefix}_boundary_edge_path_map']

    defects = np.nonzero(syndrome_np)[0]
    defects_set = set(defects)
    if len(defects) == 0: return []

    logical_vec = logical_matrix.flatten() if logical_matrix.ndim == 2 else logical_matrix
    
    qubit_cache = {}
    edge_log_map, bound_log_map = precompute_logical_effects(
        edge_path_map, boundary_edge_path_map, H_sub, logical_vec, qubit_cache
    )

    error_graph = nx.Graph()
    flipped_qubits_indices = np.nonzero(z.astype(int))[0]
    
    qubit_to_edge_info = {} 

    for q in flipped_qubits_indices:
        connected_stabs = np.nonzero(H_sub[:, q])[0]
        
        if len(connected_stabs) == 2:
            u, v = connected_stabs[0], connected_stabs[1] #add edge between the stabilizers which means the chain
            error_graph.add_edge(u, v) 
            qubit_to_edge_info[q] = {'type': 'int', 'nodes': (u, v)}
            
        elif len(connected_stabs) == 1:
            u = connected_stabs[0] #end of chain at the boundary so we just add node to graph
            error_graph.add_node(u) 
            qubit_to_edge_info[q] = {'type': 'bound', 'nodes': (u,)}

    final_matched_edges = []
    
    # --- process Each Cluster ---
    for cluster_nodes in nx.connected_components(error_graph):
        
        cluster_defects = [n for n in cluster_nodes if n in defects_set] #leave only active nodes from the syndrome in each cluster 
        if len(cluster_defects) == 0: continue

        target_parity = 0
        for q, info in qubit_to_edge_info.items():
            if info['nodes'][0] in cluster_nodes:
                if logical_vec[q] == 1:
                    target_parity ^= 1

        nodes_to_match = list(cluster_defects)
        if len(nodes_to_match) % 2 != 0: #if odd number cluster then we add a virtual node
            nodes_to_match.append(model_virtual_node_idx)

        # local MWPM  
        mwpm_graph = nx.Graph()
        for u, v in combinations(nodes_to_match, 2):
            w = get_grid_weight(u, v, model_virtual_node_idx, dist_map, boundary_dist_map)
            mwpm_graph.add_edge(u, v, weight=-w)
            
        matches = nx.max_weight_matching(mwpm_graph, maxcardinality=True)
        current_matching = [tuple(sorted((u, v))) for u, v in matches]
        
        # verify & fix Logical Errors
        if final_testing:
            append_matches(final_matched_edges, current_matching)
        else:
            current_parity = calculate_matching_parity(current_matching, edge_log_map, bound_log_map, model_virtual_node_idx)

            max_attempts = 20
            attempts = 0
            
            while current_parity != target_parity and attempts < max_attempts:
                new_matching = fix_parity_by_swapping(
                    current_matching, target_parity, current_parity,
                    edge_log_map, bound_log_map, model_virtual_node_idx,
                    dist_map, boundary_dist_map
                )
                
                if new_matching == current_matching:
                    break
                
                current_matching = new_matching
                current_parity = calculate_matching_parity(current_matching, edge_log_map, bound_log_map, model_virtual_node_idx)
                attempts += 1

            if current_parity == target_parity:
                append_matches(final_matched_edges, current_matching)
            else:
                print("ENTER BF SEARCH")
                bf_matches = solve_cluster_brute_force(
                    nodes_to_match, target_parity, 
                    edge_log_map, bound_log_map, model_virtual_node_idx
                )
                if bf_matches is not None:
                    append_matches(final_matched_edges, bf_matches)
                else:
                    return None


    # -------------------------------------------------------------
    #ASSERTION
    # -------------------------------------------------------------
    if not final_testing:
        correction_vector = np.zeros(num_qubits_per_type, dtype=int)
        
        for u, v in final_matched_edges:
            path_qubits = []
            
            #Boundary Match 
            if u == model_virtual_node_idx or v == model_virtual_node_idx:
                real_node = u if v == model_virtual_node_idx else v
                path_qubits = boundary_edge_path_map.get(real_node, [])
                
            #Internal Match 
            else:
                edge_key = (min(u, v), max(u, v))
                if edge_key in edge_path_map:
                    path_of_stabs = edge_path_map[edge_key] # List of (s1, s2)
                    for s1, s2 in path_of_stabs:
                        path_qubits.extend(get_qubits_from_edge_path_cached(s1, s2, H_sub, qubit_cache))

            # apply flips
            for q in path_qubits:
                correction_vector[q] ^= 1

        total_error = (z.astype(int) + correction_vector) % 2
        
        syndrome_check = np.dot(logical_vec, total_error) % 2
        
        if syndrome_check == 1:
            return None

    return final_matched_edges

# ---------------------------------------------------------------

def get_grid_weight(u, v, virt_id, dist_map, b_dist_map):
    if u == virt_id: return b_dist_map.get(v, 9999)
    if v == virt_id: return b_dist_map.get(u, 9999)
    if u in dist_map and v in dist_map[u]: return dist_map[u][v]
    return 9999 

def get_log_val(u, v, virt_id, edge_log, bound_log):
    if u == virt_id: return bound_log.get(v, 0)
    if v == virt_id: return bound_log.get(u, 0)
    return edge_log.get((min(u,v), max(u,v)), 0)

def calculate_matching_parity(matches, edge_log, bound_log, virt_id):
    p = 0
    for u, v in matches:
        p ^= get_log_val(u, v, virt_id, edge_log, bound_log)
    return p

def append_matches(final_list, matches):
    for u, v in matches:
        final_list.append(tuple(sorted((u, v))))

def fix_parity_by_swapping(matching, target_p, current_p, edge_log, bound_log, virt_id, dist_map, b_dist_map):
    edges = list(matching)
    n = len(edges)
    
    best_matching = matching
    min_cost_increase = float('inf')
    found = False
    
    for i in range(n):
        for j in range(i + 1, n):
            u, v = edges[i]
            x, y = edges[j]
            
            curr_w = get_grid_weight(u,v,virt_id,dist_map,b_dist_map) + get_grid_weight(x,y,virt_id,dist_map,b_dist_map)
            curr_l = get_log_val(u,v,virt_id,edge_log,bound_log) ^ get_log_val(x,y,virt_id,edge_log,bound_log)
            
            # Swap A
            w_a = get_grid_weight(u,x,virt_id,dist_map,b_dist_map) + get_grid_weight(v,y,virt_id,dist_map,b_dist_map)
            l_a = get_log_val(u,x,virt_id,edge_log,bound_log) ^ get_log_val(v,y,virt_id,edge_log,bound_log)
            
            if curr_l ^ l_a == 1: 
                cost = w_a - curr_w
                if cost < min_cost_increase:
                    min_cost_increase = cost
                    new_m = edges[:]
                    new_m[i] = tuple(sorted((u,x)))
                    new_m[j] = tuple(sorted((v,y)))
                    best_matching = new_m
                    found = True
            
            # Swap B
            w_b = get_grid_weight(u,y,virt_id,dist_map,b_dist_map) + get_grid_weight(v,x,virt_id,dist_map,b_dist_map)
            l_b = get_log_val(u,y,virt_id,edge_log,bound_log) ^ get_log_val(v,x,virt_id,edge_log,bound_log)
            
            if curr_l ^ l_b == 1: 
                cost = w_b - curr_w
                if cost < min_cost_increase:
                    min_cost_increase = cost
                    new_m = edges[:]
                    new_m[i] = tuple(sorted((u,y)))
                    new_m[j] = tuple(sorted((v,x)))
                    best_matching = new_m
                    found = True
                    
    return best_matching


def solve_cluster_brute_force(nodes, target_p, edge_log, bound_log, virt_id):
    def generate_matchings(pool):
        if not pool:
            yield []
            return
        
        first = pool[0]
        rest = pool[1:]
        
        for i, mate in enumerate(rest):
            pair = tuple(sorted((first, mate)))
            remaining = rest[:i] + rest[i+1:]
            
            for partial in generate_matchings(remaining):
                yield [pair] + partial

    def get_parity(matching):
        p = 0
        for u, v in matching:
            if u == virt_id: p ^= bound_log.get(v, 0)
            elif v == virt_id: p ^= bound_log.get(u, 0)
            else: p ^= edge_log.get((min(u,v), max(u,v)), 0)
        return p

    for m in generate_matchings(nodes):
        if get_parity(m) == target_p:
            return m
            
    return None 

# ---------------------------------------------------------------
# WRAPPERS
# ---------------------------------------------------------------

def build_ground_truth_matching_rotated(z, syndrome_np, G_real, logical_matrix, final_testing, precomputed_data, L):
    return _build_ground_truth_generic(z, syndrome_np, G_real, logical_matrix, final_testing, precomputed_data, L, is_x_type=False)

def build_ground_truth_matching_X_rotated(z, syndrome_np, G_real, logical_matrix, final_testing, precomputed_data, L):
    return _build_ground_truth_generic(z, syndrome_np, G_real, logical_matrix, final_testing, precomputed_data, L, is_x_type=True)