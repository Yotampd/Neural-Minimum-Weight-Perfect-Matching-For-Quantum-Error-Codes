import numpy as np
import networkx as nx
import random
from collections import defaultdict
from itertools import combinations, permutations
from networkx.algorithms.matching import max_weight_matching
from Data import ToricCode
import time
import torch
GLOBAL_BRUTE_FORCE_COUNTER = 0 
# ----------------------------
# MAPPING FUNCTIONS and PATH CONSTRUCTION
# ----------------------------
def qubit_idx_to_coord(q, L):
    stripe = q // L  # qubit's row
    offset = q % L  # qubit's col
    is_horiz = (stripe % 2 == 0)  # qubits on horizontal lines has even stripe
    row = stripe // 2
    col = offset
    return row, col, 'h' if is_horiz else 'v'


def qubit_to_stabs(q, L):
    r, c, ori = qubit_idx_to_coord(q, L)
    if ori == 'h':
        return [(r % L, c % L),
                ((r - 1) % L, c % L)]  # if qubit is on horizontal line it touches the top and bottom stabilizers
    else:
        return [(r % L, c % L), (r % L, (c - 1) % L)]  # qubit is on vertical, touch both left and right stabilizers


def qubit_to_stab_indices(q, L):
    stabs = qubit_to_stabs(q, L)  # stabs is a list with the stabilizers location that the qubit touch - 2 stabilizers
    return [r * L + c for r, c in stabs]  # list with 2 stabilizers indices


def stab_to_coord(stab_idx, L):
    return divmod(stab_idx, L)


def path_between_stabilizers(u, v, L):
    if u > v:
        u, v = v, u
    r1, c1 = divmod(u, L)
    r2, c2 = divmod(v, L)
    path = [(r1, c1)]

    if v > L * L:
        raise (ValueError("invalid stabilizer"))
    if u > L * L:
        raise (ValueError("invalid stabilizer"))

    if r2 > r1:  # stab2 is below stab1
        if (r2 - r1) % L > (r1 - r2) % L:  # wrap the toric code in vertical direction is shorter
            while r1 != r2:
                r1 = (r1 - 1) % L
                path.append((r1, c1))
            if (c2 - c1) % L > (c1 - c2) % L:  # wrap the toric from left is shorter
                while c1 != c2:
                    c1 = (c1 - 1) % L  # going from left
                    path.append((r1, c1))
                return path, calc_path(path, L)
            else:
                while c1 != c2:
                    c1 = (c1 + 1) % L  # going from right
                    path.append((r1, c1))
                return path, calc_path(path, L)
        else:
            while r1 != r2:
                r1 = (r1 + 1) % L  # going down
                path.append((r1, c1))
            if (c2 - c1) % L > (c1 - c2) % L:  # wrap the toric from is shorter
                while c1 != c2:
                    c1 = (c1 - 1) % L  # going from left
                    path.append((r1, c1))
                return path, calc_path(path, L)
            else:
                while c1 != c2:
                    c1 = (c1 + 1) % L  # going from right
                    path.append((r1, c1))
                return path, calc_path(path, L)

    else:  # stab2 is above stab1
        if (c2 - c1) % L > (c1 - c2) % L:  # wrap the toric code in vertical direction is shorter
            while c1 != c2:
                c1 = (c1 - 1) % L  # going up
                path.append((r1, c1))
            if (r2 - r1) % L > (r1 - r2) % L:  # wrap the toric from left is shorter
                while r1 != r2:
                    r1 = (r1 - 1) % L  # going from left
                    path.append((r1, c1))
                return path, calc_path(path, L)
            else:
                while r1 != r2:
                    r1 = (r1 + 1) % L  # going from right
                    path.append((r1, c1))
                return path, calc_path(path, L)
        else:
            while c1 != c2:
                c1 = (c1 + 1) % L  # going down
                path.append((r1, c1))
            if (r2 - r1) % L > (r1 - r2) % L:  # wrap the toric from is shorter
                while r1 != r2:
                    r1 = (r1 - 1) % L  # going from left
                    path.append((r1, c1))
                return path, calc_path(path, L)
            else:
                while r1 != r2:
                    r1 = (r1 + 1) % L  # going from right
                    path.append((r1, c1))
                return path, calc_path(path, L)


def coord_to_qubit_index(s1, s2, L):
    r1, c1 = s1
    r2, c2 = s2
    needToJump = checkIfJump(s1, s2)
    if not needToJump:
        if c2 - c1 == 1:  # going right
            return (L + (2 * r1 * L)) + c2
        if c1 - c2 == 1:  # going left
            return (L + 2 * r1 * L) + c1
        if r2 - r1 == 1:  # going down
            return 2 * (r1 + 1) * L + c1
        if r1 - r2 == 1:  # going ups
            return 2 * r1 * L + c1
    else:
        if r1 == r2:  # going left or right periodically
            return (2 * r1 * L) + L
        else:
            return c1


def checkIfJump(s1, s2):
    r1, c1 = s1
    r2, c2 = s2  # unpack coordinates
    return abs(r2 - r1) != 1 and abs(c2 - c1) != 1


def calc_path(path, L):
    q_path = []
    for i in range(len(path) - 1):
        q = coord_to_qubit_index(path[i], path[i + 1], L)
        q_path.append(q)
    return q_path

# ----------------------------
# FLIPPED QUBIT GRAPH
# ----------------------------
def build_qubit_graph(flipped_qubits, L):
    flipped_set = {i for i, v in enumerate(flipped_qubits) if
                   v == 1}  # put the *index* of the flipped qubit in the flipped set
    G = nx.Graph()
    G.add_nodes_from(flipped_set)  # each flipped qubit becomes a node

    for q1 in flipped_set:
        stabs1 = set(qubit_to_stabs(q1, L))  # the 2 stabilizer's coordinates that qubit1 flipped
        for q2 in flipped_set:  # check relation with  other qubits
            if q1 >= q2:  # no double-checking and self-loops
                continue
            stabs2 = set(qubit_to_stabs(q2, L))
            if stabs1 & stabs2:  # checks if 2 qubits share one stabilizer(same coordinates) - they never share 2
                G.add_edge(q1, q2)  # building connectivity through shared stabilizers
    return G  # flipped qubits that share a stabilizer are neighbors of each other(instead of making a list of the
    # flipped neighbors)

# ----------------------------
# ENDPOINTS + STABILIZERS
# ----------------------------

def get_flipped_stabilizers_from_cluster(clusters, L):
    stab_lists = []
    for cluster in clusters:
        stab_count = defaultdict(int)  # dict for every cluster
        for q in cluster:
            for stab in qubit_to_stab_indices(q, L):  # the 2 stabilizers that this qubit touch(stabilizer index)
                stab_count[stab] += 1
        # keep only stabilizers with odd parity (actually flipped)
        #print(stab_count)
        stabs = sorted([s for s, count in stab_count.items() if count % 2 == 1])
        stab_lists.append(
            stabs)  # list with lists of the endpoints stabilizers for each cluster(the ones that didn't cancel)
    return stab_lists

# ----------------------------
# MWPM + LOGICAL CHECK
# ----------------------------

def run_mwpm_on_clusters(endpoint_stabs, L):  # endpoint stabs are the stabilizers in the clusters that didn't cancel
    matched_edges = []
    for cluster in endpoint_stabs:
        if len(cluster) == 2:  # matching is obvious
            matched_edges.append(tuple(cluster))
        elif len(cluster) > 2:
            G = nx.Graph()
            for u, v in combinations(cluster,
                                     2):  # build compleate graph for this cluster with the stabilizers that didn't cancel
                r1, c1 = stab_to_coord(u, L)  # coordinate of the stabilizer u
                r2, c2 = stab_to_coord(v, L)
                dr = min((r1 - r2) % L, (r2 - r1) % L)
                dc = min((c1 - c2) % L, (c2 - c1) % L)
                dist = dr + dc  # calculate distance for weights (locally)
                G.add_edge(u, v, weight=-dist)  # negative dist so max weight matching will be minimum weight matching
            matches = max_weight_matching(G, maxcardinality=True)
            matched_edges.extend(list(matches))
    return matched_edges


def get_all_perfect_matchings(nodes):
    nodes_list = list(nodes) 
    if not nodes_list:
        yield []
        return
        
    first = nodes_list[0]
    rest = nodes_list[1:]
    
    for i, second in enumerate(rest):
        pair = tuple(sorted((first, second))) 
        remaining = rest[:i] + rest[i+1:]
        
        for matching in get_all_perfect_matchings(remaining):
            yield [pair] + matching


def apply_correction(matched_edges, L, z):
    correction_vector = z.astype(int).copy()  # every time we enter this function we start with original flipped qubits vector  
    for u, v in matched_edges:
        path = path_between_stabilizers(u, v, L)  # qubits indices in the correction path
        #print(f"correction qubit path indices for matching {u} and {v}: ", path[1])
        for q in path[1]:
            correction_vector[q] ^= 1
    return correction_vector


def try_permutations_and_correct(endpoint_stabs, L, matched_edges, z, logical_matrix):
    # endpoint stabs, list of lists with stabilizers that didn't cancel
    for i, cluster in enumerate(endpoint_stabs):
        if len(cluster) <= 2:
            continue


        other_matches = [e for e in matched_edges if e[0] not in cluster and e[
            1] not in cluster]  # keep the matches that are not in the current cluster

        for trial_matching in get_all_perfect_matchings(cluster):
            new_matching = other_matches + trial_matching
            correction = apply_correction(new_matching, L, z=z)
            if not logical_error(correction, logical_matrix=logical_matrix):
                return new_matching
    all_endpoints = [stab for group in endpoint_stabs for stab in group]
    all_endpoints = sorted(all_endpoints)
    start_time = time.time()
    
    global GLOBAL_BRUTE_FORCE_COUNTER    
    GLOBAL_BRUTE_FORCE_COUNTER += 1     

    for perm in permutations(all_endpoints):
        if time.time() - start_time > 10:
            return None
        trial_matching = []
        for j in range(0, len(perm) - 1, 2):  # permute and take adjacent couples
            trial_matching.append((perm[j], perm[j + 1]))  # create new matching
        correction = apply_correction(trial_matching, L, z=z)  # correction for new matching
        if not logical_error(correction, logical_matrix=logical_matrix):
            #print("last match was successful")
            return trial_matching

    return None


def logical_error(correction_vector, logical_matrix):
    result = logical_matrix @ correction_vector % 2
    return np.any(result) 


#------------------------------
# X stabilizers
#------------------------------

#------------------------------
# MAPPING
#------------------------------

def qubit_to_stab_vertex(q, L):
    r, c, ori = qubit_idx_to_coord(q, L)
    if ori == 'h':
        return [(r % L, c % L),
                (r % L, (c + 1) % L)]  # if qubit is on horizontal line it touches the left and right vertex stabilizers
    else:
        return [(r % L, c % L),
                ((r+1) % L, c % L)]  # qubit is on vertical, touch both up and bottom vertex stabilizers


def qubit_to_stab_vertex_indices(q, L):
    stabs = qubit_to_stab_vertex(q,
                                 L)  # stabs is a list with the stabilizers location that the qubit touch - 2 stabilizers
    return [r * L + c for r, c in stabs]  # list with 2 stabilizers indices


def path_between_stabilizers_X(u, v, L):
    if u > v:
        u, v = v, u
    r1, c1 = divmod(u, L)
    r2, c2 = divmod(v, L)
    path = [(r1, c1)]

    if v > L * L:
        raise (ValueError("invalid stabilizer"))
    if u > L * L:
        raise (ValueError("invalid stabilizer"))

    if r2 > r1:  
        if (r2 - r1) % L > (r1 - r2) % L:  
            while r1 != r2:
                r1 = (r1 - 1) % L
                path.append((r1, c1))
            if (c2 - c1) % L > (c1 - c2) % L:  
                while c1 != c2:
                    c1 = (c1 - 1) % L  
                    path.append((r1, c1))
                return path, calc_path_X(path, L)
            else:
                while c1 != c2:
                    c1 = (c1 + 1) % L  
                    path.append((r1, c1))
                return path, calc_path_X(path, L)
        else:
            while r1 != r2:
                r1 = (r1 + 1) % L  
                path.append((r1, c1))
            if (c2 - c1) % L > (c1 - c2) % L:  
                while c1 != c2:
                    c1 = (c1 - 1) % L  
                    path.append((r1, c1))
                return path, calc_path_X(path, L)
            else:
                while c1 != c2:
                    c1 = (c1 + 1) % L 
                    path.append((r1, c1))
                return path, calc_path_X(path, L)
    else:  
        if (c2 - c1) % L > (c1 - c2) % L:  
            while c1 != c2:
                c1 = (c1 - 1) % L  
                path.append((r1, c1))
            if (r2 - r1) % L > (r1 - r2) % L:  
                while r1 != r2:
                    r1 = (r1 - 1) % L  
                    path.append((r1, c1))
                return path, calc_path_X(path, L)
            else:
                while r1 != r2:
                    r1 = (r1 + 1) % L  
                    path.append((r1, c1))
                return path, calc_path_X(path, L)
        else:
            while c1 != c2:
                c1 = (c1 + 1) % L  
                path.append((r1, c1))
            if (r2 - r1) % L > (r1 - r2) % L:  
                while r1 != r2:
                    r1 = (r1 - 1) % L  
                    path.append((r1, c1))
                return path, calc_path_X(path, L)
            else:
                while r1 != r2:
                    r1 = (r1 + 1) % L 
                    path.append((r1, c1))
                return path, calc_path_X(path, L)


def coord_to_qubit_index_X(s1, s2, L):
    r1, c1 = s1
    r2, c2 = s2
    needToJump = checkIfJump(s1, s2)
    if not needToJump:
        if c2 - c1 == 1:  
            return (2 * r1 * L) + c1
        if c1 - c2 == 1:  
            return (2 * r1 * L) + c2
        if r2 - r1 == 1:  
            return L * (r1 + r2) + c1
        if r1 - r2 == 1:  
            return L * (r1 + r2) + c1
    else:
        if r1 == r2:  
            return (2 * r1 * L) + (L-1)
        else: 
            return (2 * L * L) - L + c1


def calc_path_X(path, L):
    q_path = []
    for i in range(len(path) - 1):
        q = coord_to_qubit_index_X(path[i], path[i + 1], L)
        q_path.append(q)
    return q_path

# ----------------------------
# FLIPPED QUBIT GRAPH
# ----------------------------
def build_qubit_graph_X(flipped_qubits, L):
    flipped_set = {i for i, v in enumerate(flipped_qubits) if
                   v == 1}  # put the *index* of the flipped qubit in the flipped set
    G = nx.Graph()
    G.add_nodes_from(flipped_set)  # each flipped qubit becomes a node

    for q1 in flipped_set:
        stabs1 = set(qubit_to_stab_vertex(q1, L))  # the 2 stabilizer's coordinates that qubit1 flipped
        for q2 in flipped_set:  # check relation with  other qubits
            if q1 >= q2:  # no double-checking and self-loops
                continue
            stabs2 = set(qubit_to_stab_vertex(q2, L))
            if stabs1 & stabs2:  # checks if 2 qubits share one stabilizer(same coordinates) - they never share 2
                G.add_edge(q1, q2)  # building connectivity through shared stabilizers
    return G  # flipped qubits that share a stabilizer are neighbors of each other(instead of making a list of the
    # flipped neighbors)


# ----------------------------
# ENDPOINTS + STABILIZERS
# ----------------------------

def get_flipped_stabilizers_from_cluster_X(clusters, L):
    stab_lists = []
    for cluster in clusters:
        stab_count = defaultdict(int)  # dict for every cluster
        for q in cluster:
            for stab in qubit_to_stab_vertex_indices(q, L):  # the 2 stabilizers that this qubit touch(stabilizer index)
                stab_count[stab] += 1
        # keep only stabilizers with odd parity (actually flipped)
        stabs = sorted([s for s, count in stab_count.items() if count % 2 == 1])
        stab_lists.append(
            stabs)  # list with lists of the endpoints stabilizers for each cluster(the ones that didn't cancel)
    return stab_lists


# ----------------------------
# MWPM
# ----------------------------

def run_mwpm_on_clusters_X(endpoint_stabs, L):  
    matched_edges = []
    for cluster in endpoint_stabs:
        if len(cluster) == 2:  # matching is obvious
            matched_edges.append(tuple(cluster))
        elif len(cluster) > 2:
            G = nx.Graph()
            for u, v in combinations(cluster, 2):  # build compleate graph for this cluster with the stabilizers that didn't cancel
                r1, c1 = stab_to_coord(u, L)  
                r2, c2 = stab_to_coord(v, L)
                dr = min((r1 - r2) % L, (r2 - r1) % L)
                dc = min((c1 - c2) % L, (c2 - c1) % L)
                dist = dr + dc  
                G.add_edge(u, v, weight=-dist)  
            matches = max_weight_matching(G, maxcardinality=True)
            matched_edges.extend(list(matches))
    return matched_edges


def apply_correction_X(matched_edges, L, flipped_qubits):
    correction_vector = flipped_qubits.astype(int).copy() 
    for u, v in matched_edges:
        path = path_between_stabilizers_X(u, v, L)  # qubits indices in the correction path
        for q in path[1]:
            correction_vector[q] ^= 1
    return correction_vector


def try_permutations_and_correct_X(endpoint_stabs, L, matched_edges, z, logical_matrix):
    # endpoint stabs, list of lists with stabilizers that didn't cancel
    for i, cluster in enumerate(endpoint_stabs):
        if len(cluster) <= 2:
            continue

        other_matches = [e for e in matched_edges if e[0] not in cluster and e[1] not in cluster] 

        for trial_matching in get_all_perfect_matchings(cluster):
            new_matching = other_matches + trial_matching
            correction = apply_correction_X(new_matching, L, flipped_qubits=z)
            if not logical_error_X(correction, logical_matrix_X=logical_matrix):
                return new_matching

    all_endpoints = [stab for group in endpoint_stabs for stab in group]
    all_endpoints = sorted(all_endpoints)
    start_time = time.time()
    
    global GLOBAL_BRUTE_FORCE_COUNTER    
    GLOBAL_BRUTE_FORCE_COUNTER += 1      
    
    for perm in permutations(all_endpoints):
        if time.time() - start_time > 10:
            return None
        trial_matching = []
        for j in range(0, len(perm) - 1, 2):  # permute and take adjacent couples
            trial_matching.append((perm[j], perm[j + 1]))  # create new matching
        correction = apply_correction_X(trial_matching, L, flipped_qubits=z)  # correction for new matching
        if not logical_error_X(correction, logical_matrix_X=logical_matrix):
            return trial_matching

    return None

def logical_error_X(correction_vector, logical_matrix_X):
    result = logical_matrix_X @ correction_vector % 2
    return np.any(result)  # false if the vector is all zeros - no logical error


# ----------------------------
# MAIN PIPELINE
# ----------------------------

def build_ground_truth_matching(z, syndrome, L, logical_matrix, final_testing):
    if isinstance(syndrome, torch.Tensor):
        num_defects = torch.count_nonzero(syndrome).item()
    else:
        num_defects = np.count_nonzero(syndrome)

    if num_defects == 0:
        return []
    
    G = build_qubit_graph(flipped_qubits=z, L=L)
    clusters = list(nx.connected_components(G))
    endpoint_stabs = get_flipped_stabilizers_from_cluster(clusters, L)

    matched_edges = run_mwpm_on_clusters(endpoint_stabs, L)
    correction = apply_correction(matched_edges=matched_edges, L=L, z=z)


    if final_testing == False: #refine labels only in training
        if logical_error(correction, logical_matrix=logical_matrix): 
            alt_match = try_permutations_and_correct(endpoint_stabs=endpoint_stabs, L=L, matched_edges=matched_edges, z=z, logical_matrix=logical_matrix)
            if alt_match is not None: 
                matched_edges = alt_match
                correction = apply_correction(matched_edges=matched_edges, L=L, z=z)
            if alt_match is None and final_testing == False : #training and no valid matching 
                matched_edges = alt_match # we return None to filter in training

    return matched_edges

def build_ground_truth_matching_X(z, syndrome, L, logical_matrix, final_testing):
    if isinstance(syndrome, torch.Tensor):
        num_defects = torch.count_nonzero(syndrome).item()
    else:
        num_defects = np.count_nonzero(syndrome)

    if num_defects == 0:
        return []

    G = build_qubit_graph_X(flipped_qubits=z, L=L)
    clusters = list(nx.connected_components(G)) 
    endpoint_stabs = get_flipped_stabilizers_from_cluster_X(clusters=clusters, L=L)

    matched_edges = run_mwpm_on_clusters_X(endpoint_stabs, L)
    correction = apply_correction_X(matched_edges=matched_edges, L=L, flipped_qubits=z)

    if final_testing == False:
        if logical_error_X(correction_vector=correction, logical_matrix_X=logical_matrix):
            alt_match = try_permutations_and_correct_X(endpoint_stabs=endpoint_stabs, L=L, matched_edges=matched_edges, z=z, logical_matrix=logical_matrix)
            if alt_match is not None:
                matched_edges = alt_match
                correction = apply_correction_X(matched_edges=matched_edges, L=L, flipped_qubits=z)
            if alt_match is None and final_testing == False:
                matched_edges = alt_match
    return matched_edges