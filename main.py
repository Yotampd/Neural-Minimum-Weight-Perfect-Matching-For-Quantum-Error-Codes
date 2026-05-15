#======================================================================================
# Implementation of Neural Minimum Weight Perfect Matching for Quantum Error Codes 
# ICML 2026
#======================================================================================

from __future__ import print_function
from torch.optim import Adam
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import random
import os
from torch.utils import data
from datetime import datetime
import logging
from Data import *
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
import shutil
from Graph_C import build_comp_graph, build_edges_vector, syndrome_to_coordinates, build_syndrome_graph_rotated
from model import EdgeClassifierTransformer2 
from training import train_model, test_model
from GT_C import build_ground_truth_matching, build_ground_truth_matching_X, GLOBAL_BRUTE_FORCE_COUNTER
from training import plot_test_acc
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from conversion_geo import convert_graph_to_torch, add_laplacian_pe, convert_graph_to_torch_rotated, combine_graphs_rotated, add_precomputed_pe_rotated, combine_graphs 
from precompute_rot import precompute_laplacian_pe, precompute_shortest_paths, ADJ_Z_L5, ADJ_X_L5, precompute_boundary_paths, generate_rotated_x_coords, generate_rotated_z_coords,generate_stabilizer_adjacency
from Rotated_GT import build_ground_truth_matching_rotated, build_ground_truth_matching_X_rotated
GLOBAL_NONE_COUNTER = 0
##################################################################
##################################################################
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

##################################################################
class QECC_Dataset(data.Dataset):
    def __init__(self, code, ps, len, args, final_testing = False):
        self.code = code
        self.ps = ps 
        self.len = len
        self.logic_matrix = code.logic_matrix.cpu() # erased transpose
        self.pc_matrix = code.pc_matrix.clone().cpu()
        self.zero_cw = torch.zeros((self.pc_matrix.shape[1])).long()
        self.noise_method = self.independent_noise if args.noise_type == 'independent' else self.depolarization_noise
        self.args = args
        self.final_testing = final_testing
        self.code_type = args.code_type

        self.num_stabs_total = self.pc_matrix.shape[0]
        if args.noise_type == 'depolarization':
            self.num_stabs_per_type = self.num_stabs_total // 2
        else: # independent noise
            self.num_stabs_per_type = self.num_stabs_total
        
        self.precomputed_data = args.precomputed_data

    def independent_noise(self,pp=None): 
        pp = random.choice(self.ps) if pp is None else pp
        flips = np.random.binomial(1, pp, self.pc_matrix.shape[1])
        return flips
        
    def depolarization_noise(self,pp=None):
        ## See original noise definition in https://github.com/Krastanov/neural-decoder/
        pp = random.choice(self.ps) if pp is None else pp
        out_dimZ = out_dimX = self.pc_matrix.shape[1]//2
        def makeflips(q):
            q = q/3.
            flips = np.zeros((out_dimZ+out_dimX,), dtype=np.dtype('b'))
            rand = np.random.rand(out_dimZ or out_dimX)
            both_flips  = (2*q<=rand) & (rand<3*q)
            ###
            x_flips = rand<  q # z stabilizers syndrome
            flips[:out_dimZ] ^= x_flips
            flips[:out_dimZ] ^= both_flips
            ###
            z_flips = (q<=rand) & (rand<2*q) # x stabilizers syndrome
            flips[out_dimZ:out_dimZ+out_dimX] ^= z_flips
            flips[out_dimZ:out_dimZ+out_dimX] ^= both_flips
            return flips
        flips = makeflips(pp)
        while not np.any(flips):
            flips = makeflips(pp) 
        return flips*1.
        
        
    
    def __getitem__(self, index): #build a graph based on syndrome
        global GLOBAL_NONE_COUNTER
        if self.args.repetitions > 1:
            raise NotImplementedError("")

        MAX_RETRIES = 120 
        start_time = time.time()
        # ------------------------------------
        #           Toric Code
        # ------------------------------------
        if self.code_type == 'toric':
            for i in range(MAX_RETRIES):

                pp = random.choice(self.ps) # sample an error rate
                z = torch.from_numpy(self.noise_method(pp)).float()  # flipped qubits vector 
                L = self.args.code_L
                noise_type = self.args.noise_type
                logical_matrix = self.logic_matrix.cpu().numpy().astype(int)

                if noise_type == "independent":
                    # syndrome from Z stabilizers only
                    syndrome = torch.matmul(self.pc_matrix, z.long()) % 2 
                    syndrome = syndrome.float()
                    syndrome_np = syndrome.numpy()

                    # build coordinates for Z stabilizers
                    defects = syndrome_to_coordinates(syndrome, L, noise_type="independent", stab_type="Z")
                    G, edge_to_idx = build_comp_graph(defects, L)
                    num_edges = len(edge_to_idx) // 2

                    matched_edges = build_ground_truth_matching(z.numpy(), syndrome_np, L, logical_matrix, self.final_testing)
                    if not self.final_testing: # only in training
                        if matched_edges is None: 
                            continue
                    label_vector = build_edges_vector(edge_to_idx, matched_edges, num_edges) 
                    graph_data = convert_graph_to_torch(G, label_vector, L, z, syndrome, stab_t="Z") #with indepedent noise model i only do x errors - z stabilizers
                    end = time.time()
                    graph_data = add_laplacian_pe(graph_data, L=L)
                    num_edges = graph_data.edge_attr.shape[0]
                    component_feature = torch.ones((num_edges, 1), dtype=graph_data.edge_attr.dtype, device=graph_data.edge_attr.device)
                    graph_data.edge_attr = torch.cat([graph_data.edge_attr, component_feature], dim=1)
                    graph_data.stab_t = torch.tensor(1) #Z stabilizer
                    graph_data.L = L

                    return graph_data

                elif noise_type == "depolarization":
                    z_np = z.numpy()
                    z_part = z_np[:2 * L * L]     # x-error flips - z stabilizers
                    x_part = z_np[2 * L * L:]     # z-error flips - x stabilizers 
    
                    # syndrome vector: H_block @ z
                    syndrome = torch.matmul(self.pc_matrix, z.long()) % 2
                    syndrome = syndrome.float()
                    syndrome_np = syndrome.numpy()

                    # Split the full syndrome into Z and X parts
                    syndrome_Z = syndrome_np[:L * L]   
                    syndrome_X = syndrome_np[L * L:]  

                    # Build Z stabilizers graph - corrcting x errors 
                    defects_Z = syndrome_to_coordinates(torch.tensor(syndrome_Z), L, noise_type="independent", stab_type="Z")
                    G_Z, edge_to_idx_Z = build_comp_graph(defects_Z, L)
                    num_edges_Z = len(edge_to_idx_Z) // 2

                    matched_edges_Z = build_ground_truth_matching(z_part, syndrome_Z, L, logical_matrix[:2, :2*L*L], final_testing=self.final_testing) #logical matrix is block matrix we take the logical z operators
                    if not self.final_testing: #only in training
                        if matched_edges_Z is None: #no valid correction
                            continue
                    label_vector_Z = build_edges_vector(edge_to_idx_Z, matched_edges_Z, num_edges_Z)
                    data_Z = convert_graph_to_torch(G_Z, label_vector_Z, L, torch.from_numpy(z_part), torch.from_numpy(syndrome_Z), stab_t = "Z")
                    data_Z = add_laplacian_pe(data_Z, L=L)

                    # Build X stabilizers graph - correcting z errors
                    defects_X = syndrome_to_coordinates(torch.tensor(syndrome_X), L, noise_type="independent", stab_type="X")
                    G_X, edge_to_idx_X = build_comp_graph(defects_X, L)
                    num_edges_X = len(edge_to_idx_X) // 2


                    matched_edges_X = build_ground_truth_matching_X(x_part, syndrome_X, L, logical_matrix[2:, 2*L*L:], self.final_testing)
                    if not self.final_testing: #if false - meaning im in training
                        if matched_edges_X is None: # no valid correction found
                            continue
                    label_vector_X = build_edges_vector(edge_to_idx_X, matched_edges_X, num_edges_X)
                    data_X = convert_graph_to_torch(G_X, label_vector_X, L, torch.from_numpy(x_part), torch.from_numpy(syndrome_X), stab_t = "X")
                    data_X = add_laplacian_pe(data_X, L=L)

                    data_Z.L = L
                    data_X.L = L

                    data_Z.stab_t = torch.tensor(1) # 1 for Z-component
                    data_X.stab_t = torch.tensor(0) # 0 for X-component

                    data_Z.id = index
                    data_X.id = index

                    if data_Z.edge_index.numel() == 0 and data_X.edge_index.numel() == 0:
                        continue
                    
                    #combine to a graph with 2 connected componenets
                    combined_data = combine_graphs(data_Z, data_X, L=L)
                    return combined_data
                else:
                    raise ValueError("Unsupported noise type")
        
        
        elif self.code_type == 'rotated':
            #---------------------------
            #       Rotated 
            #---------------------------
            for i in range(MAX_RETRIES):
                pp = random.choice(self.ps) # sample an error rate
                z = torch.from_numpy(self.noise_method(pp)).float()  # flipped qubits vector 
                L = self.args.code_L
                noise_type = self.args.noise_type

                if noise_type == "independent":
                    # syndrome from Z stabilizers only
                    syndrome = torch.matmul(self.pc_matrix, z.long()) % 2  # [m], syndrome vector, every coordinate is a stabilizer
                    syndrome = syndrome.float()
                    syndrome_np = syndrome.numpy()
                    z_dist_map = self.precomputed_data['z_dist_map']
                    z_boundary_dist_map = self.precomputed_data['z_boundary_dist_map']
                    # Logic Matrix
                    logic_Z_np = self.logic_matrix.cpu().numpy().astype(int)
                    num_stabs_per_type = (L*L - 1) // 2
                    virtual_node_start_idx = num_stabs_per_type
                    # build coordinates for Z stabilizers
                    defects = np.nonzero(syndrome_np)[0]
                    G, edge_to_idx = build_syndrome_graph_rotated(defects, z_dist_map, stab_type="Z")
                    num_edges = len(edge_to_idx) // 2
                    logical_matrix_np = self.logic_matrix.cpu().numpy().astype(int)
                    matched_edges = build_ground_truth_matching_rotated(z.numpy(), syndrome_np, G, logic_Z_np, self.final_testing, self.precomputed_data, L=L)
                    if not self.final_testing: 
                        if matched_edges is None:  
                            continue
                    G.add_node(virtual_node_start_idx, type_vec = [0.0, 1.0]) # Z-type
                    current_edge_idx = num_edges

                    for u in defects:
                        dist_to_boundary = z_boundary_dist_map[u]
                        G.add_edge(u, virtual_node_start_idx, dist=dist_to_boundary, edge_index=current_edge_idx)
                        edge_to_idx[(u, virtual_node_start_idx)] = current_edge_idx
                        edge_to_idx[(virtual_node_start_idx, u)] = current_edge_idx
                        current_edge_idx += 1
                    num_edges = current_edge_idx
                    
                    label_vector = build_edges_vector(edge_to_idx, matched_edges, num_edges) 
                    graph_data = convert_graph_to_torch_rotated(G, label_vector, L, z, syndrome, stab_t="Z", precomputed_data=self.precomputed_data)

                    pe_z = self.precomputed_data['pe_z'].to(graph_data.x.device)
                    pe_z_virtual = torch.mean(pe_z, dim=0, keepdim=True)
                    full_pe = torch.cat([pe_z, pe_z_virtual], dim=0)

                    graph_data.x = torch.cat([graph_data.x, full_pe], dim=1)
                    graph_data.edge_attr = F.pad(graph_data.edge_attr, (0, 1), value=1)

                    graph_data.stab_t = torch.tensor(1) # Z stabilizer
                    graph_data.L = L

                    if graph_data.edge_index.numel() == 0:
                        continue

                    return graph_data

                elif noise_type == "depolarization":
                    z_np = z.numpy()
                    z_part = z_np[:L*L]     # x-error flips - z stabilizers
                    x_part = z_np[L*L:]     # z-error flips - x stabilizers 
                    syndrome = torch.matmul(self.pc_matrix, z.long()) % 2
                    syndrome = syndrome.float()
                    syndrome_np = syndrome.numpy()
                    syndrome_Z = syndrome_np[:self.num_stabs_per_type]
                    syndrome_X = syndrome_np[self.num_stabs_per_type:]

                    logic_Z_np = self.logic_matrix[0:1, :L*L].cpu().numpy().astype(int)
                    logic_X_np = self.logic_matrix[1:2, L*L:].cpu().numpy().astype(int)

                    z_dist_map = self.precomputed_data['z_dist_map']
                    x_dist_map = self.precomputed_data['x_dist_map']

                    num_stabs_per_type = (L*L - 1) //2
                    virtual_node_start_idx = num_stabs_per_type
                    GT_virtual_nodes = list(range(num_stabs_per_type, num_stabs_per_type + L - 1)) 

                    # Build Z stabilizers graph - corrcting x errors 
                    defects_Z = np.nonzero(syndrome_Z)[0]
                    G_Z, edge_to_idx_Z = build_syndrome_graph_rotated(defects_Z, z_dist_map, stab_type = "Z") 
                    num_edges_Z = len(edge_to_idx_Z) // 2

                    matched_edges_Z = build_ground_truth_matching_rotated(z_part, syndrome_Z, G_Z, logic_Z_np, final_testing=self.final_testing, precomputed_data=self.precomputed_data, L=L)
                    if not self.final_testing: # only in training
                        if matched_edges_Z is None: 
                            # --------------------------------------
                            GLOBAL_NONE_COUNTER += 1
                            # --------------------------------------
                            continue
                    # add virtual node
                    G_Z.add_node(virtual_node_start_idx, type_vec = [0.0, 1.0])
                    z_boundary_dist_map = self.precomputed_data['z_boundary_dist_map']
                    current_edge_idx = num_edges_Z

                    for u in defects_Z: # add edge to virtual node
                        dist_to_boundary = z_boundary_dist_map[u] # Use precomputed min dist
                        G_Z.add_edge(u, virtual_node_start_idx, dist=dist_to_boundary, edge_index=current_edge_idx)
                        edge_to_idx_Z[(u, virtual_node_start_idx)] = current_edge_idx
                        edge_to_idx_Z[(virtual_node_start_idx, u)] = current_edge_idx
                        current_edge_idx += 1
                    num_edges_Z = current_edge_idx
                    

                    label_vector_Z = build_edges_vector(edge_to_idx_Z, matched_edges_Z, num_edges_Z)
                    data_Z = convert_graph_to_torch_rotated(G_Z, label_vector_Z, L, torch.from_numpy(z_part), torch.from_numpy(syndrome_Z), stab_t = "Z", precomputed_data=self.precomputed_data)

                    # Build X stabilizers graph - correcting z errors
                    defects_X = np.nonzero(syndrome_X)[0]
                    G_X, edge_to_idx_X = build_syndrome_graph_rotated(defects_X, x_dist_map, stab_type="X")
                    num_edges_X = len(edge_to_idx_X) // 2


                    matched_edges_X = build_ground_truth_matching_X_rotated(
                                x_part, syndrome_X, G_X, logic_X_np, 
                                final_testing=self.final_testing, 
                                precomputed_data=self.precomputed_data, L=L
                            )
                    if not self.final_testing:
                        if matched_edges_X is None: 
                            # -------------------
                            GLOBAL_NONE_COUNTER += 1
                            # ------------------
                            continue

                    G_X.add_node(virtual_node_start_idx, type_vec = [1.0, 0.0])
                    x_boundary_dist_map = self.precomputed_data['x_boundary_dist_map']
                    current_edge_idx = num_edges_X

                    for u in defects_X:
                        dist_to_boundary = x_boundary_dist_map[u]
                        G_X.add_edge(u, virtual_node_start_idx, dist=dist_to_boundary, edge_index=current_edge_idx)
                        edge_to_idx_X[(u, virtual_node_start_idx)] = current_edge_idx
                        edge_to_idx_X[(virtual_node_start_idx, u)] = current_edge_idx
                        current_edge_idx += 1
                    num_edges_X = current_edge_idx
                    label_vector_X = build_edges_vector(edge_to_idx_X, matched_edges_X, num_edges_X)
                    data_X = convert_graph_to_torch_rotated(G_X, label_vector_X, L, torch.from_numpy(x_part), torch.from_numpy(syndrome_X), stab_t = "X", precomputed_data=self.precomputed_data)
                    data_Z.L = L
                    data_X.L = L
                    data_Z.stab_t = torch.tensor(1) # 1 for Z-component
                    data_X.stab_t = torch.tensor(0) # 0 for X-component
                    data_Z.id = index
                    data_X.id = index
                    
                    combined_data = combine_graphs_rotated(data_Z, data_X, L=L)
                    combined_data = add_precomputed_pe_rotated(combined_data, self.precomputed_data['pe_z'], self.precomputed_data['pe_x'])

                    if data_Z.edge_index.numel() == 0 and data_X.edge_index.numel() == 0: 
                        continue 


                    return combined_data
                else:
                    raise ValueError("Unsupported noise type")
        
            raise RuntimeError(f"Failed to generate valid sample {MAX_RETRIES} times")  
    
    def __len__(self):
        return self.len


##################################################################
##################################################################
# ============================================ single GPU  ==============================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logging.info(f"torch sees {torch.cuda.device_count()} GPU(s)")
    for i in range(torch.cuda.device_count()):
        logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    args.code.logic_matrix = args.code.logic_matrix.to(device) 
    args.code.pc_matrix = args.code.pc_matrix.to(device) 
    code = args.code
    assert 0 < args.repetitions 


    model = EdgeClassifierTransformer2(
        node_feat_dim=5+8, # 8 is for the laplacian pe
        edge_feat_dim=4, 
        hidden_dim=args.d_model,
        heads=4,
        num_layers=args.num_layers,
        L = args.code_L,
        noise_type = args.noise_type,
        num_stabs_total = args.code.num_stabs_total,
        code_type = args.code_type
    )

    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs with PyG DataParallel")
        model = DataParallel(model)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #===== checkpoint loading ===========
    start_epoch = 0
    best_loss = float('inf')

    if args.load_model_path is not None and os.path.exists(args.load_model_path):
        checkpoint = torch.load(args.load_model_path, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint.get('best_loss', float('inf'))
            start_epoch = checkpoint.get('epoch', 0) + 1
            logging.info(f"Resuming from epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint)  # Raw state_dict — test only, no optimizer
            logging.info("Loaded raw model state dict (testing only)")
    else:
        logging.info("Starting training from scratch")

  #================== transfer try ========================

    if args.pretrained_transformer_path and os.path.exists(args.pretrained_transformer_path):
        pt_ckpt = torch.load(args.pretrained_transformer_path, map_location=device)
        pt_state = pt_ckpt['model_state_dict'] if 'model_state_dict' in pt_ckpt else pt_ckpt
        model_state = model.state_dict()

        transfer_prefixes = ('transformer_pred_layers', 'predictor_norm', 'output_lin')
        copied, skipped = 0, 0
        for key, param in pt_state.items():
            if key.startswith(transfer_prefixes):
                if key in model_state and model_state[key].shape == param.shape:
                    model_state[key] = param.clone()
                    copied += 1
                else:
                    skipped += 1  # shape mismatch (e.g. different d_model)

        model.load_state_dict(model_state)
        logging.info(
            f"Transferred {copied} transformer-predictor tensors from "
            f"{args.pretrained_transformer_path} ({skipped} skipped — shape mismatch)"
        )

    #=========================================================
    logging.info(f'PC matrix shape {code.pc_matrix.shape}')
    logging.info(model)
    logging.info(f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')
    ps_train = np.linspace(0.01, 0.2, 9)
    if args.noise_type == 'depolarization':
        ps_train = np.linspace(0.05, 0.2, 9)
    if args.repetitions > 1:
        ps_train = np.linspace(0.02, 0.04, 9)

    ps_test = np.linspace(0.01, 0.2, 18)
    if args.noise_type == 'depolarization':
        ps_test = np.linspace(0.05, 0.2, 18)
    if args.repetitions > 1:
        ps_test = np.linspace(0.02, 0.04, 18)
    
    if args.ps_test is not None: #for rebuttle
        ps_test = np.array(args.ps_test)


    train_dataloader = DataListLoader(QECC_Dataset(code, ps_train, len=args.batch_size * 500, args=args, final_testing = False), batch_size=int(args.batch_size), 
                                  shuffle=True, num_workers=0)

    test_dataloader_list = [DataListLoader(QECC_Dataset(code, [ps_test[ii]], len=int(args.test_batch_size * 200),args=args, final_testing = True), 
                                       batch_size=int(args.test_batch_size), shuffle=False, num_workers=4) for ii in range(len(ps_test))] 
    
    #=================scheduler====================
    t_max = args.epochs*1.0
    if args.use_warmup:
        total_steps = args.epochs * len(train_dataloader)
        warmup_steps = int(0.05 * total_steps)

        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-5) 

    #=============================================
    global GLOBAL_NONE_COUNTER
    GLOBAL_NONE_COUNTER = 0
    print("\nCounter reset to 0. Starting training logic")
    # --------------------------------
    return model, train_dataloader, optimizer, scheduler, device, test_dataloader_list, ps_test, start_epoch, best_loss 
# ============================================ single GPU ==============================================

##################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN-based MWPM Decoder for QEC')
    parser.add_argument('--test_only', action='store_true', help='Run only the test phase')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to best model checkpoint')
    parser.add_argument('--epochs', type=int, default=500) # currently 500 
    parser.add_argument('--workers', type=int, default=0) # TODO changed to 0 from 4
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128) # changed from 128
    parser.add_argument('--test_batch_size', type=int, default=512) #changed from 512
    parser.add_argument('--seed', type=int, default=42)

    # Code args
    parser.add_argument('--code_type', type=str, default='toric',choices=['toric', 'rotated'])
    parser.add_argument('--code_L', type=int, default=4,help='Lattice length')
    parser.add_argument('--repetitions', type=int, default=1,help='Number of faulty repetitions. <=1 is equivalent to none.')
    parser.add_argument('--noise_type', type=str,default='independent', choices=['independent','depolarization'],help='Noise model')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')

    # model args
    parser.add_argument('--d_model', type=int, default=128,help='GNN hidden dimension')
    parser.add_argument('--use_warmup', action='store_true', help='Use warmup with cosine scheduler')
    parser.add_argument('--ps_test', type=float, nargs='+', default=None, help='Override test error rates, e.g. --ps_test 0.03 0.23')
    parser.add_argument('--pretrained_transformer_path', type=str, default=None, help='Path to a checkpoint whose transformer predictor weights are transferred ''(GNN layers stay randomly initialised)')

    args = parser.parse_args()



    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    set_seed(args.seed)

    precomputed_data = {}

    if args.code_type == 'rotated':
        
        print("Precomputing data for Rotated Code")

        # Adjacency Matrices
        if args.code_L == 5:
            adj_z = ADJ_Z_L5
            adj_x = ADJ_X_L5
        else:
            print(f"Generating Dynamic Adjacency for L={args.code_L}")
            H_full, _ = Get_rotated_surface_Code(args.code_L, full_H=True)
            num_stabs_per_type = (args.code_L**2 - 1) // 2
            num_qubits_per_type = args.code_L**2
            
            H_Z_matrix = H_full[0:num_stabs_per_type, 0:num_qubits_per_type]
            H_X_matrix = H_full[num_stabs_per_type:, num_qubits_per_type:]
            
            adj_z = generate_stabilizer_adjacency(H_Z_matrix)
            adj_x = generate_stabilizer_adjacency(H_X_matrix)
        
        # PEs
        print("Computing Laplacian PE")
        precomputed_data['pe_z'] = precompute_laplacian_pe(adj_z, k_eigenvectors=8)
        precomputed_data['pe_x'] = precompute_laplacian_pe(adj_x, k_eigenvectors=8)
        
        # compute distance
        print("Computing all-pairs shortest path distances")
        precomputed_data['z_dist_map'], precomputed_data['z_edge_path_map'] = precompute_shortest_paths(adj_z)
        precomputed_data['x_dist_map'], precomputed_data['x_edge_path_map'] = precompute_shortest_paths(adj_x)

        #boundary paths
        print("Computing all-pairs boundary paths")
        
        H_full, _ = Get_rotated_surface_Code(args.code_L, full_H=True)
        num_stabs_per_type = (args.code_L**2 - 1) // 2 # 12 for L=5
        num_qubits_per_type = args.code_L**2 # 25
        
        H_Z_matrix = H_full[0:num_stabs_per_type, 0:num_qubits_per_type]
        H_X_matrix = H_full[num_stabs_per_type:, num_qubits_per_type:]

        precomputed_data['z_boundary_dist_map'], precomputed_data['z_boundary_edge_path_map'] = precompute_boundary_paths(H_Z_matrix, "Z", args.code_L)
        precomputed_data['x_boundary_dist_map'], precomputed_data['x_boundary_edge_path_map'] = precompute_boundary_paths(H_X_matrix, "X", args.code_L)

        #compute coordinates
        print("Computing Topological Coordinates")
        x_coords = generate_rotated_x_coords(args.code_L)
        z_coords = generate_rotated_z_coords(args.code_L)

        precomputed_data['x_coords'] = x_coords
        precomputed_data['z_coords'] = z_coords


    class Code():
        pass
    code = Code()

    if args.code_type == 'toric':
        code_func_name = f'Get_{args.code_type}_Code'
    elif args.code_type == 'rotated':
        code_func_name = f'Get_{args.code_type}_surface_Code'
    else:
        raise ValueError(f"Unknown code_type: {args.code_type}")

    H, Lz = eval(code_func_name)(args.code_L, full_H=args.noise_type == 'depolarization')
    code.logic_matrix = torch.from_numpy(Lz).long() 
    code.pc_matrix = torch.from_numpy(H).long() 
    code.n = code.pc_matrix.shape[1]
    code.k = code.n - code.pc_matrix.shape[0]
    code.code_type = args.code_type
    code.num_stabs_total = code.pc_matrix.shape[0]
    code.num_stabs_per_type = code.num_stabs_total // 2 if args.noise_type == 'depolarization' else code.num_stabs_total
    args.code = code
    args.precomputed_data = precomputed_data
    ####################################################################
    model_dir = os.path.join('Final_Results_GNN_MWPM', args.code_type, 
                             'Code_L_' + str(args.code_L) , 
                             f'noise_model_{args.noise_type}', 
                             f'repetition_{args.repetitions}' , 
                             datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    os.makedirs(model_dir, exist_ok=True)
    args.path = model_dir
    writer = SummaryWriter(log_dir=os.path.join(model_dir, 'tensorboard_logs'))
    handlers = [
        logging.FileHandler(os.path.join(model_dir, 'logging.txt'))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=handlers)
    logging.info(f"Path to model/logs: {model_dir}")
    logging.info(args)

    logging.info("="*50)
    logging.info("Start training GNN for MWPM")
    logging.info("="*50)

    # # setup #
    model, train_dataloader, optimizer, scheduler, device, test_dataloader_list, ps_test, start_epoch, best_loss = main(args) #changed dataloader name
    #===== training GNN model ====== 
    if not args.test_only:
        test_accuracies = train_model(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            args=args,
            test_dataloader_list=test_dataloader_list,
            ps_test=ps_test,
            start_epoch = start_epoch,
            best_loss = best_loss,
            writer=writer
        )
    else:
        assert args.load_model_path is not None, "Must provide --load_model_path when using --test_only"
        logging.info(f"Loading model from {args.load_model_path}")
        checkpoint = torch.load(args.load_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # ===== Load last model before final testing =====
    last_checkpoint_path = os.path.join(args.path, 'last_checkpoint.pt')
    if os.path.exists(last_checkpoint_path):
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded last model from epoch {checkpoint['epoch']} for final testing.")
    else:
        logging.warning("No last_checkpoint.pt found, using in-memory model for final testing.")


    # ===== testing ======
    test_model(
        model=model,
        dataloader_list=test_dataloader_list,
        device=device,
        ps_test=ps_test,
        args=args,
        final_testing=True,
        epoch = None
    )

    writer.close()

    