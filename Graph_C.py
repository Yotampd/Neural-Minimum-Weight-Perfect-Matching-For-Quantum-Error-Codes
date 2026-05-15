import networkx as nx
import torch
from itertools import combinations
import numpy as np

def build_comp_graph(defects, L):
    G = nx.Graph()
    edge_to_idx = {}
    edge_idx = 0    
     
    for coord, stab_type, stab_idx in defects: #defects is a tuple with coord stab type and stab index
        if stab_type == "X": # z errors 
            type_vec = [1, 0]
        elif stab_type == "Z": # x errors
            type_vec = [0, 1]
        else:
            raise ValueError("unknown stab_type")
        
        G.add_node(stab_idx, pos=coord, type = stab_type, type_vec = type_vec) 

    for u, v in combinations(G.nodes, 2): #all combinatios of pairs with no repetions(no self loops)
        type_u = G.nodes[u]["type"]
        type_v = G.nodes[v]["type"]
        if type_u != type_v:
            continue # dont build edges between different types of errors
        pos_u = G.nodes[u]["pos"]  
        pos_v = G.nodes[v]["pos"]
        manhattan_distance = min(abs(pos_u[0] - pos_v[0]), L - abs(pos_u[0] - pos_v[0]) ) + min(abs(pos_u[1] - pos_v[1]), L - abs(pos_u[1] - pos_v[1]) )
        G.add_edge(u, v, dist=manhattan_distance, weight=None, edge_index = edge_idx)
        edge_to_idx[(u,v)] = edge_idx
        edge_to_idx[(v,u)] = edge_idx #dict symmetry 
        edge_idx += 1 # u,v and v,u share the same edge index


    return G , edge_to_idx


def build_syndrome_graph_rotated(defect_indices, dist_map, stab_type = "Z"):
    G = nx.Graph()
    edge_to_idx = {}
    edge_idx = 0
    
    if stab_type == "X": # z errors 
        type_vec = [1, 0]
    elif stab_type == "Z": # x errors
        type_vec = [0, 1]
    else:
        raise ValueError(f"unknown stab_type: {stab_type}")
    
    for stab_idx in defect_indices:
        G.add_node(stab_idx, type_vec=type_vec) # [0,1] is Z-stab

    for u, v in combinations(defect_indices, 2):
        key = (min(u, v), max(u, v))
        
        if key not in dist_map:
            raise KeyError(f"Distance for edge {key} not found in precomputed map.")
        
            
        distance = dist_map[key]
        
        G.add_edge(u, v, dist=distance, edge_index=edge_idx)
        edge_to_idx[(u,v)] = edge_idx
        edge_to_idx[(v,u)] = edge_idx
        edge_idx += 1

    return G, edge_to_idx


def syndrome_to_coordinates(syndrome, L, noise_type="independent", stab_type = "Z"):   #returns a list of ((i, j), "type")
     
    if isinstance(syndrome, torch.Tensor):
        syndrome = syndrome.cpu().numpy()

    defects = []

    if noise_type == "independent":  #one type of stabilizer
        for idx in range(len(syndrome)): 
            if syndrome[idx] == 1:
                i , j = divmod(idx, L)
                if stab_type == "Z": #correcting x errors, with plaquette(z) stabilizers 
                    coord = (i + 0.5, j + 0.5) #center of plaquattte 
                elif stab_type == "X":
                    coord = (i , j) #vertex location
                else:
                    raise ValueError("stab_type must be X or Z")
                defects.append((coord, stab_type, idx))
        
        return defects


    elif noise_type == "depolarization":
        assert len(syndrome) == 2 * L * L , "syndrome length must be 2*L^2"

        for idx in range(L*L): #first half of the vector, Z stabilizers - x errors
            if syndrome[idx] == 1:
                i, j = divmod(idx, L)  # floor division and reminder for the mapping (0,0) -> 0, (0,1) -> 1
                coord = (i + 0.5, j + 0.5) #plaquatte location
                defects.append((coord, "Z", idx))
                
        for idx in range(L*L, L*L*2): #2nd half of the vector, X stabilizers - z errors
            if syndrome[idx] == 1: #flipped 
                rel_idx = idx - L*L #only for the mapping
                i, j = divmod(rel_idx, L)
                coord = (i , j) #vertex location
                defects.append((coord, "X", idx))
        
        return defects

    else: 
        raise ValueError("unsupported noise type")
        

     


def build_edges_vector(edge_to_idx, true_edges, num_edges): #true edges will be the edges that indeed in the matching - based on the data
    label_vector = np.zeros(num_edges, dtype = np.float64)

    if len(true_edges) == 0:
        return label_vector

    for u, v in true_edges:
        idx = edge_to_idx.get((u,v))
        if idx is not None:
            label_vector[idx] = 1.0
        else: 
            raise ValueError(f"edge {u}, {v} not in edge_to_idx dict")

    return label_vector