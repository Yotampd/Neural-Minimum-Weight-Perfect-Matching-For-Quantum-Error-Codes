import torch
import torch.distributed as dist
from torch.nn import BCEWithLogitsLoss
from torch_geometric.loader import DataListLoader
import os
import logging
import argparse as args
import matplotlib.pyplot as plt 
import time
import networkx as nx
import numpy as np
import pymatching
from pymatching import Matching
import torch.nn.functional as F
import GT_C
from GT_C import stab_to_coord, path_between_stabilizers, path_between_stabilizers_X
import math
import copy
import ldpc
import scipy.sparse
from Data import Get_toric_Code, Get_rotated_surface_Code
from Rotated_GT import get_qubits_from_edge_path


H_SPARSE_CACHE = None
L_TORCH_CACHE = None
H_ROTATED_CACHE = {}
TORIC_PATH_CACHE_Z = {} 
TORIC_PATH_CACHE_X = {} 
PATHS_PRECOMPUTED_FOR_L = None
def precompute_toric_paths(L):
    global TORIC_PATH_CACHE_Z, TORIC_PATH_CACHE_X, PATHS_PRECOMPUTED_FOR_L
    
    if PATHS_PRECOMPUTED_FOR_L == L:
        return 

    logging.info(f"Precomputing Toric paths for L={L}...")
    num_stabs = L * L
    
    for u in range(num_stabs):
        for v in range(u + 1, num_stabs):
            _, path_z = path_between_stabilizers(u, v, L=L)
            path_set_z = frozenset(path_z)
            TORIC_PATH_CACHE_Z[(u, v)] = path_set_z
            TORIC_PATH_CACHE_Z[(v, u)] = path_set_z
            
            _, path_x = path_between_stabilizers_X(u, v, L=L)
            path_set_x = frozenset(path_x)
            TORIC_PATH_CACHE_X[(u, v)] = path_set_x
            TORIC_PATH_CACHE_X[(v, u)] = path_set_x
            
    PATHS_PRECOMPUTED_FOR_L = L
    logging.info("Precomputation complete.")

def get_rotated_H_matrices(L): 
    global H_ROTATED_CACHE
    if L in H_ROTATED_CACHE:
        return H_ROTATED_CACHE[L]
    
    logging.info(f"Caching H_Z and H_X matrices for L={L}")
    num_stabs = (L**2 - 1) // 2
    num_qubits = L**2
    H_full, _ = Get_rotated_surface_Code(L, full_H=True)
    
    H_Z = H_full[0:num_stabs, 0:num_qubits]
    H_X = H_full[num_stabs:, num_qubits:]
    
    H_ROTATED_CACHE[L] = (H_Z, H_X)
    return H_Z, H_X

ler_vs_epochs_data = {}


def train_step(model, batch, optimizer, scheduler, device, use_warmup, alpha = 0.5, beta = 0.5): # recives a graph every step  
    model.train()
    batch = batch.to(device)

    optimizer.zero_grad()

    # Forward pass
    edge_logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.syndrome)  # shape: [num_edges]
    label_vector = batch.y

    logits = edge_logits.view(-1) 
    targets = label_vector.view(-1)
    edge_probs = torch.sigmoid(edge_logits)

    L_confidance = F.binary_cross_entropy(edge_probs.view(-1), edge_probs.view(-1), reduction="mean")

    # Loss
    loss_fn = BCEWithLogitsLoss() 
    bce_loss = loss_fn(edge_logits.view(-1), label_vector.view(-1)) 

    loss = bce_loss + (0.01 * L_confidance)

    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()

    if use_warmup and scheduler is not None:
        scheduler.step()
    return loss.item()


def train_model(model, dataloader, optimizer, scheduler, device, args, test_dataloader_list, ps_test, start_epoch, best_loss, writer):
    best_loss = float("inf")

    train_losses = []
    test_accuracies = {}
    learning_rates = []
    
    ber_all_pred_epochs = {}
    ber_all_manhattan_epochs = {}
    ler_all_pred_epochs = {}
    ler_all_manhattan_epochs = {}

    prev_brute_force_count = GT_C.GLOBAL_BRUTE_FORCE_COUNTER

    for epoch in range(start_epoch + 1, args.epochs + 1):
        start_time = time.time()

        model.train()
        total_loss = 0.0
        batch_count = 0
        node_counts = []

        for graph_list in dataloader:
            batch_loss = 0.0
    
            for graph in graph_list:
                graph = graph.to(device) 
                node_counts.append(torch.unique(graph.edge_index).numel())
                loss = train_step(model, graph, optimizer, scheduler, device, args.use_warmup) 
                batch_loss += loss
            avg_batch_loss = batch_loss / len(graph_list)
            
            total_loss += avg_batch_loss
            batch_count += 1

        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)


        end_time = time.time()
        duration = end_time - start_time

        if node_counts:
            avg_nodes = sum(node_counts) / len(node_counts)


        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f} | Time = {duration:.2f} sec")
        logging.info(f"Epoch {epoch}: Average Loss = {avg_loss:.4f} | Time = {duration:.2f} sec | Number of nodes per sample = {avg_nodes}")

        current_brute_force_count = GT_C.GLOBAL_BRUTE_FORCE_COUNTER
        epoch_brute_force_count = current_brute_force_count - prev_brute_force_count
        
        print(f"Epoch {epoch}: Brute Force Searches = {epoch_brute_force_count}")
        logging.info(f"Epoch {epoch}: Brute Force Searches = {epoch_brute_force_count}")
        
        prev_brute_force_count = current_brute_force_count
        
        if not args.use_warmup: 
            scheduler.step() 

        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        logging.info(f"Epoch {epoch}: Learning Rate = {current_lr:.6f}")

        #tensor board
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # Validation evaluation logic
        if epoch >= 700 and epoch % 5 == 0:
            # test "best model"
            best_model_path = os.path.join(args.path, 'best_checkpoint.pt')
            if os.path.exists(best_model_path):

                training_state = copy.deepcopy(model.state_dict())

                checkpoint = torch.load(best_model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"[BEST MODEL] Loaded best model (epoch {checkpoint['epoch']}) for testing at epoch {epoch}.")
                logging.info(f"[BEST MODEL] Loaded best model (epoch {checkpoint['epoch']}) for testing at epoch {epoch}.")

                # Save tagged best model
                tagged_best_path = os.path.join(args.path, f'best_model_epoch_{epoch:03d}.pt')
                torch.save(checkpoint, tagged_best_path)

                # Run test with best model
                test_model(model, test_dataloader_list, device, ps_test, args, final_testing=True, epoch=epoch, writer=writer)

                model.load_state_dict(training_state)
                model.train()
            else:
                print("Warning: best_checkpoint.pt not found during validation.")
                logging.warning("best_checkpoint.pt not found during validation.")

        elif epoch >= 700 and epoch % 11 == 0:
            # test "last model"
            print(f"[LAST MODEL] Testing last model directly at epoch {epoch}.")
            logging.info(f"[LAST MODEL] Testing last model directly at epoch {epoch}.")

            last_model_path = os.path.join(args.path, f'last_model_epoch_{epoch:03d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, last_model_path)
            print(f"Saved last model checkpoint at epoch {epoch}.")
            logging.info(f"Saved last model checkpoint at epoch {epoch}.")
            
            test_model(model, test_dataloader_list, device, ps_test, args, final_testing=True, epoch=epoch, writer=writer)


        # Save full checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }
        torch.save(checkpoint, os.path.join(args.path, 'last_checkpoint.pt'))

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, os.path.join(args.path, 'best_checkpoint.pt'))
            print("Best model saved.")
            logging.info("Best model saved.")
        
    plot_training(train_losses, args.path)
    plot_learning_rate(learning_rates, args.path)

    return test_accuracies


@torch.no_grad()
def test_model(model, dataloader_list, device, ps_test, args, final_testing = False, epoch= None, writer=None):
    if args.code_type == 'toric':
        precompute_toric_paths(args.code_L)
    model.eval()
    collected_weights = []
    results = []
    acc_list = []
    p_dict = {}

    ber_all_pred = []
    ler_all_pred = []
    ps_vals = [float(p) for p in ps_test]
    

    for i, dataloader in enumerate(dataloader_list): # for each noise level 
        predictions_at_p = []
        total = 0
        correct = 0

        ber_list_pred = []
        ler_list_pred = []
        p_val = ps_test[i]
        
        for batch_idx, graph_list in enumerate(dataloader):
            for graph_id, graph in enumerate(graph_list):
                graph = graph.to(device)
                
                edge_logits = model(graph.x, graph.edge_index, graph.edge_attr, graph.syndrome)
                preds = torch.sigmoid(edge_logits)
                
                collected_weights.append(preds.detach().cpu()) 

                # ======= averaging over p both direction to one weight ===========
                edge_index = graph.edge_index.cpu()
                preds_cpu = preds.cpu()
                if args.noise_type == "depolarization":
                    edge_attr_cpu = graph.edge_attr.cpu() 
                    edge_mask_Z = edge_attr_cpu[:, 3] == 1
                    edge_index_Z = edge_index[:, edge_mask_Z]
                    preds_Z = preds_cpu[edge_mask_Z]
                    
                    edge_mask_X = edge_attr_cpu[:, 3] == 0
                    edge_index_X = edge_index[:, edge_mask_X]
                    preds_X = preds_cpu[edge_mask_X]

                    edge_dict_Z = {}
                    for idx in range(edge_index_Z.size(1)):
                        u, v = edge_index_Z[0, idx].item(), edge_index_Z[1, idx].item()
                        key = tuple(sorted((u, v)))
                        if key in edge_dict_Z: edge_dict_Z[key].append(preds_Z[idx].item())
                        else: edge_dict_Z[key] = [preds_Z[idx].item()]
                    edge_list_Z = []
                    weight_list_Z = []
                    for (u, v), weights in edge_dict_Z.items():
                        edge_list_Z.append([u,v])
                        weight_list_Z.append(max(weights))

                    edge_index_clean_Z = torch.tensor(edge_list_Z, dtype=torch.long).T
                    weights_clean_Z = torch.tensor(weight_list_Z, dtype=torch.float32)

                    edge_dict_X = {}
                    for idx in range(edge_index_X.size(1)):
                        u, v = edge_index_X[0, idx].item(), edge_index_X[1, idx].item()
                        key = tuple(sorted((u, v)))
                        if key in edge_dict_X: edge_dict_X[key].append(preds_X[idx].item())
                        else: edge_dict_X[key] = [preds_X[idx].item()]
                    edge_list_X = []
                    weight_list_X = []
                    for (u, v), weights in edge_dict_X.items():
                        edge_list_X.append([u,v])
                        weight_list_X.append(max(weights))
                    
                    edge_index_clean_X = torch.tensor(edge_list_X, dtype=torch.long).T
                    weights_clean_X = torch.tensor(weight_list_X, dtype=torch.float32)

                    if final_testing:
                        if args.code_type == 'rotated':
                            ber_pred_Z, ler_pred_Z = decode_and_evaluate_rotated(
                                graph.z_Z, graph.syndrome_Z, graph.stab_t_Z, 
                                edge_index_clean_Z, weights_clean_Z, args, p_val
                            )
                            ber_pred_X, ler_pred_X = decode_and_evaluate_rotated(
                                graph.z_X, graph.syndrome_X, graph.stab_t_X, 
                                edge_index_clean_X, weights_clean_X, args, p_val
                            )

                        else: #toric
                            ber_pred_Z, ler_pred_Z = decode_and_evaluate(graph.z_Z, graph.syndrome_Z, graph.stab_t_Z, edge_index_clean_Z, weights_clean_Z, args, graph.L, graph.num_nodes_Z, graph.y_Z)
                            ber_pred_X, ler_pred_X = decode_and_evaluate(graph.z_X, graph.syndrome_X, graph.stab_t_X, edge_index_clean_X, weights_clean_X, args, graph.L, graph.num_nodes_Z, graph.y_X)
                        logical_error_pred = max(ler_pred_Z, ler_pred_X)
                        
                        ler_list_pred.append(logical_error_pred)

                        ber_list_pred.append(np.mean([ber_pred_Z, ber_pred_X]))


                else: # independent noise
                    edge_dict = {}
                    for idx in range(edge_index.size(1)):
                        u, v = edge_index[0, idx].item(), edge_index[1, idx].item()
                        key = tuple(sorted((u, v)))
                        if key in edge_dict: edge_dict[key].append(preds_cpu[idx].item())
                        else: edge_dict[key] = [preds_cpu[idx].item()]

                    edge_list = []
                    weight_list = []
                    for (u, v), weights in edge_dict.items():
                        edge_list.append([u, v])
                        weight_list.append(max(weights))
                    
                    edge_index_clean = torch.tensor(edge_list, dtype=torch.long).T
                    weights_clean = torch.tensor(weight_list, dtype=torch.float32)
                    

                # ===== Decoder indepedent =====
                    if final_testing == True:
                        if args.code_type == 'rotated':
                            ber_pred, ler_pred = decode_and_evaluate_rotated(
                                graph.z, graph.syndrome, graph.stab_t, 
                                edge_index_clean, weights_clean, args, p_val)
                        else: # Toric
                            ber_pred, ler_pred = decode_and_evaluate(graph.z, graph.syndrome, graph.stab_t, edge_index_clean, weights_clean, args, graph.L, 0, graph.y)


                        ber_list_pred.append(ber_pred)
                        ler_list_pred.append(ler_pred)
 

        if not final_testing:
            acc = correct / total if total > 0 else 0
            p_dict[ps_test[i]] = acc
            acc_list.append(acc)

        print(f"Test @ p={ps_test[i]:.3f}:")
        if final_testing == True:
            print(f"BER_Pred={np.mean(ber_list_pred):.4f}, LER_Pred={np.mean(ler_list_pred):.4f}")
        else:
            print(f"acc={acc}")
        
        logging.info(f"Test @ p={ps_test[i]:.3f}:")
        if final_testing == True:
            logging.info(f"BER_Pred={np.mean(ber_list_pred):.4f}, LER_Pred={np.mean(ler_list_pred):.4f}")
        else:
            logging.info(f"acc:{acc}:")
        
        ber_all_pred.append(np.mean(ber_list_pred))
        ler_all_pred.append(np.mean(ler_list_pred))

    # Save all predictions
    save_path = os.path.join(args.path, 'predicted_edge_weights.pt')
    torch.save(results, save_path)
    logging.info(f"Predicted edge weights saved to {save_path}")
    print(f"Predicted edge weights saved to {save_path}")

    if epoch is not None: 
        all_weights_flat = torch.cat(collected_weights)
        
        data_save_name = f"weights_epoch_{epoch}.pt"
        torch.save(all_weights_flat, os.path.join(args.path, data_save_name))
        
        plot_save_name = f"weights_hist_epoch_{epoch}.png"
        plot_path = os.path.join(args.path, plot_save_name)
        
        plot_weight_hist(all_weights_flat, plot_path, epoch)
        
        print(f"Saved weight histogram to {plot_path}")
        logging.info(f"Saved weight histogram to {plot_path}")
 
    # plot ber and ler
    if final_testing:
        suffix = f"epoch_{epoch:03d}" if epoch is not None else "final"
        plot_ber_vs_p(ps_vals, ber_all_pred, args.path, suffix, epoch, writer=writer)
        plot_ler_vs_p(ps_vals, ler_all_pred, args.path, suffix, epoch, writer=writer)
        plot_ber_vs_p_log(ps_vals, ber_all_pred, args.path, suffix, epoch, writer=writer)
        plot_ler_vs_p_log(ps_vals, ler_all_pred, args.path, suffix, epoch, writer=writer)

    return 

#plots
def plot_weight_hist(weights_tensor, save_path, epoch): 
    weights_np = weights_tensor.numpy()
    
    plt.figure(figsize=(10, 6))
    plt.hist(weights_np, bins=100, alpha=0.7, color='blue', density=True)
    
    plt.title(f"Weight Distribution - Epoch {epoch}")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density (Log Scale)")
    plt.yscale('log') 
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()

def plot_training(train_losses, save_dir):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss & Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_plot.png"))
    plt.close()

def plot_learning_rate(learning_rates, save_dir):
    import matplotlib.pyplot as plt
    epochs = range(1, len(learning_rates) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, learning_rates, color='tab:red')
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "learning_rate_plot.png"))
    plt.close()

    
def plot_test_acc(test_accuracies, save_dir, final_epoch):
    import matplotlib.pyplot as plt
    import os

    plt.figure(figsize=(10, 6))
    for p in sorted(test_accuracies):
        acc_list = test_accuracies[p]
        # Make the x-axis match the test epochs
        if len(acc_list) == 1:
            epochs = [final_epoch]
        else:
            epochs = [40 * i for i in range(1, len(acc_list))]  
            epochs.append(final_epoch)  # Add final test
        plt.plot(epochs, acc_list, label=f"p={p:.3f}", marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy per Noise Level (p)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_accuracy_per_p.png"))
    plt.close()

def plot_ber_vs_p(ps_vals, ber_pred, save_dir, suffix="", epoch = None, writer=None):
    plt.figure()
    plt.plot(ps_vals, ber_pred, label='BER Predicted', marker = "s")
    plt.xlabel('Physical Error Rate (p)')
    plt.ylabel('Bit Error Rate (BER)')
    title = f'BER vs Physical Error Rate (epoch {epoch})' if epoch is not None else 'BER vs Physical Error Rate (final)'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"ber_vs_p_{suffix}.png" if suffix else "ber_vs_p.png"
    plt.savefig(os.path.join(save_dir, filename))
    if writer is not None and epoch is not None:
        writer.add_figure("BER_vs_p", plt.gcf(), global_step=epoch)
    plt.close()



def plot_ler_vs_p(ps_vals, ler_pred, save_dir, suffix="", epoch = None, writer=None):
    plt.figure()
    plt.plot(ps_vals, ler_pred, label='LER Predicted', marker = "s")
    plt.xlabel('Physical Error Rate (p)')
    plt.ylabel('Logical Error Rate (LER)')
    title = f'LER vs Physical Error Rate (epoch {epoch})' if epoch is not None else 'LER vs Physical Error Rate (final)'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"ler_vs_p_{suffix}.png" if suffix else "ler_vs_p.png"
    plt.savefig(os.path.join(save_dir, filename))
    if writer is not None and epoch is not None:
        writer.add_figure("LER_vs_p", plt.gcf(), global_step=epoch)
    plt.close()

    global ler_vs_epochs_data
    if epoch is not None:
        for i, p_rate in enumerate(ps_vals):
            if p_rate not in ler_vs_epochs_data:
                ler_vs_epochs_data[p_rate] = {'epochs': [], 'lers': []}
            
            ler_vs_epochs_data[p_rate]["epochs"].append(epoch)
            ler_vs_epochs_data[p_rate]['lers'].append(ler_pred[i]) # Using the predicted LER
        plot_ler_vs_epochs(ler_vs_epochs_data, save_dir)
            


def plot_ber_vs_p_log(ps_vals, ber_pred, save_dir, suffix="", epoch = None, writer=None):
    plt.figure()
    plt.plot(ps_vals, ber_pred, label='BER Predicted', marker='s')
    plt.xlabel('Physical Error Rate (p)')
    plt.ylabel('BER (log scale)')
    plt.yscale('log')
    title = f'BER vs p (Log Scale) (epoch {epoch})' if epoch is not None else 'BER vs p (Log Scale) (final)'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"ber_vs_p_log_{suffix}.png" if suffix else "ber_vs_p_log.png"
    plt.savefig(os.path.join(save_dir, filename))
    if writer is not None and epoch is not None:
        writer.add_figure("BER_vs_p_log", plt.gcf(), global_step=epoch)
    plt.close()

def plot_ler_vs_p_log(ps_vals, ler_pred, save_dir, suffix="", epoch = None, writer = None):
    plt.figure()
    plt.plot(ps_vals, ler_pred, label='LER Predicted', marker='s')
    plt.xlabel('Physical Error Rate (p)')
    plt.ylabel('LER (log scale)')
    plt.yscale('log')
    title = f'LER vs p (Log Scale) (epoch {epoch})' if epoch is not None else 'LER vs p (Log Scale) (final)'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"ler_vs_p_log_{suffix}.png" if suffix else "ler_vs_p_log.png"
    plt.savefig(os.path.join(save_dir, filename))
    if writer is not None and epoch is not None:
        writer.add_figure("LER_vs_p_log", plt.gcf(), global_step=epoch)
    plt.close()


def plot_ler_vs_epochs(ler_data, save_dir):
 
    plot_folder = os.path.join(save_dir, 'epoch_plots')
    os.makedirs(plot_folder, exist_ok=True)

    plt.figure(figsize=(12, 7))
    for p_rate, data in ler_data.items():
        # Use a float for the p_rate label for consistency
        p_label = float(p_rate) 
        if data['epochs']: # Only plot if there is data
            plt.plot(data['epochs'], data['lers'], label=f'p = {p_label:.3f}', marker='o')
    
    plt.xlabel("Epoch")
    plt.ylabel("Logical Error Rate (LER)")
    plt.title("LER vs. Epochs for each Physical Error Rate")
    #plt.yscale('log') # Using a log scale is often helpful for error rates
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    
    save_path = os.path.join(plot_folder, "ler_vs_epochs_log.png")
    plt.savefig(save_path)
    plt.close()
    print(f"LER vs. Epochs plot updated and saved to {save_path}")
    logging.info(f"LER vs. Epochs plot updated and saved to {save_path}")

#======== decoder ===========

def decode_and_evaluate(z, syndrome, stab_t, edge_index_clean, predicted_weights, args, L, node_shift, y):  
    syndrome = syndrome.cpu().numpy().astype(np.int32)
    z = z.cpu()
    logical_mat = args.code.logic_matrix
    logical_mat = logical_mat.cpu()
    noise_type = args.noise_type

    if edge_index_clean.numel() == 0:
        ber_pred = 0.0
        ler_pred = 0
        return ber_pred, ler_pred
    used_nodes = set(edge_index_clean.flatten().tolist())
    max_node_idx = max(used_nodes)

    assert np.all(syndrome[max_node_idx + 1:] == 0), \
    f"Syndrome contains active stabilizers beyond used nodes: {np.nonzero(syndrome[max_node_idx + 1:])}"

    if stab_t == 0 : # X stabilizers decoding - z errors
        syndrome_trimmed = syndrome[:max_node_idx + 1 - L*L].astype(np.int32) 
        edge_index_clean = edge_index_clean - (L*L)
    elif stab_t == 1 : # Z stabilizers decoding - x errors
        syndrome_trimmed = syndrome[:max_node_idx + 1].astype(np.int32) 
    else:
        raise ValueError(f"unknown stab_t: {stab_t}")
    



    pred_matching = Matching()
    current_cache = TORIC_PATH_CACHE_X if stab_t == 0 else TORIC_PATH_CACHE_Z
    for i in range(edge_index_clean.shape[1]):
        u, v = int(edge_index_clean[0, i].item()), int(edge_index_clean[1, i].item())  # take an edge

        fault_id_set = current_cache.get((u, v))

        w = predicted_weights[i].item() 
        w_prime = -math.log(max(w, 1e-6))  
        pred_matching.add_edge(u, v, fault_ids=fault_id_set  ,weight=w_prime)


    z_hat_pred_list = pred_matching.decode(syndrome_trimmed)
    z_hat_pred = torch.tensor(z_hat_pred_list, dtype=torch.float32).cpu()

    if len(z_hat_pred) < (2 * L * L):
        pad_len = (2 * L * L) - len(z_hat_pred)
        z_hat_pred = F.pad(z_hat_pred, (0, pad_len), value=0)



    #=====Logic Matrix=====
    if noise_type == "independent":
        final_logic_mat = logical_mat
    if noise_type == "depolarization":
        if stab_t == 1: #Z stabilizers - x errors
            final_logic_mat = logical_mat[:2, :2*L*L]
        if stab_t == 0: #X stabilizers - z errors
            final_logic_mat = logical_mat[2:, 2*L*L:]

    # === BER & LER ===

    # correcting
    corrected_pred = (z + z_hat_pred) % 2 
    
    # BER
    ber_pred = torch.mean(corrected_pred).item()

    # LER
    corrected_pred_long = corrected_pred.long()
    logical_syndrome_pred = torch.matmul(final_logic_mat, corrected_pred_long) % 2
    ler_pred = torch.any(logical_syndrome_pred).item()

    return ber_pred, ler_pred


def decode_and_evaluate_rotated(z, syndrome, stab_t, edge_index_clean, predicted_weights, args, p_val):
    z = z.cpu()
    syndrome = syndrome.cpu().numpy().astype(np.int32)
    stab_t = int(stab_t.item()) 
    L = args.code_L
    L_qubits = L * L
    precomputed_data = args.precomputed_data
    logical_mat = args.code.logic_matrix.cpu()
    
    num_stabs_per_type = (L * L - 1) // 2 
    
    H_Z, H_X = get_rotated_H_matrices(L) 
    virtual_node_idx = num_stabs_per_type

    if edge_index_clean.numel() == 0:
        if not np.any(syndrome):
            return 0.0, 0 

    if stab_t == 1: # Z stabilizers (X errors)
        H_matrix = H_Z
        dist_map = precomputed_data['z_dist_map']
        edge_path_map = precomputed_data['z_edge_path_map']
        boundary_dist_map = precomputed_data['z_boundary_dist_map']
        boundary_edge_path_map = precomputed_data['z_boundary_edge_path_map'] 
        
        final_logic_mat = logical_mat[0:1, :L_qubits]

        max_node_idx = 0
        if edge_index_clean.numel() > 0:
            max_node_idx = edge_index_clean.max().item()
            all_nodes = edge_index_clean.flatten().unique()
            real_nodes = all_nodes[all_nodes != virtual_node_idx]
            max_real_node_idx = real_nodes.max().item()
        
        syndrome_trimmed = syndrome[:max_real_node_idx + 1]
        edge_index_clean_shifted = edge_index_clean

    elif stab_t == 0: # X stabilizers (Z errors)
        H_matrix = H_X
        dist_map = precomputed_data['x_dist_map']
        edge_path_map = precomputed_data['x_edge_path_map']
        boundary_dist_map = precomputed_data['x_boundary_dist_map']
        boundary_edge_path_map = precomputed_data['x_boundary_edge_path_map']
        
        final_logic_mat = logical_mat[1:2, L_qubits:]
        
        shift = num_stabs_per_type + 1 
        edge_index_clean_shifted = edge_index_clean - shift 

        max_node_idx = 0
        if edge_index_clean_shifted.numel() > 0:
            max_node_idx = edge_index_clean_shifted.max().item()
            all_nodes = edge_index_clean_shifted.flatten().unique()
            real_nodes = all_nodes[all_nodes != virtual_node_idx]
            max_real_node_idx = real_nodes.max().item()

        syndrome_trimmed = syndrome[:max_real_node_idx + 1]

    else:
        raise ValueError(f"unknown stab_t: {stab_t}")

    num_detectors = num_stabs_per_type
    pred_matching = Matching()

    for i in range(edge_index_clean_shifted.shape[1]):
        u = edge_index_clean_shifted[0, i].item()
        v = edge_index_clean_shifted[1, i].item()

        if u == virtual_node_idx or v == virtual_node_idx:
            continue

        key = (min(u, v), max(u, v))
        w_pred = predicted_weights[i].item()
        w_prime = -math.log(max(w_pred, 1e-6))
        
        if key not in dist_map:
             continue


        if key not in edge_path_map:
            continue
            
        edge_path = edge_path_map[key]
        qubit_list = get_qubits_from_edge_path(edge_path, H_matrix)
        
        fault_id_set = set(qubit_list)
        pred_matching.add_edge(u, v, fault_ids=fault_id_set, weight=w_prime)
    
    # --- Add Boundary Edges ---
    for i in range(edge_index_clean_shifted.shape[1]):
        u = edge_index_clean_shifted[0, i].item()
        v = edge_index_clean_shifted[1, i].item()
        if u != virtual_node_idx and v != virtual_node_idx:
            continue
        real_node = u if v == virtual_node_idx else v
        w_pred = predicted_weights[i].item()
        w_prime = -math.log(max(w_pred, 1e-9))

        if real_node in boundary_edge_path_map:
            qubits = boundary_edge_path_map[real_node]
            qubits_set = set(qubits)
            pred_matching.add_boundary_edge(real_node, fault_ids=qubits_set, weight=w_prime)

    # --- Decode ---
    z_hat_pred_list = pred_matching.decode(syndrome_trimmed)

    z_hat_pred = torch.tensor(z_hat_pred_list, dtype=torch.float32).cpu()

    pad_len_pred = z.shape[0] - z_hat_pred.shape[0]
    if pad_len_pred > 0:
        z_hat_pred = F.pad(z_hat_pred, (0, pad_len_pred), value=0)
    elif pad_len_pred < 0:
        z_hat_pred = z_hat_pred[:z.shape[0]]

    corrected_pred = (z + z_hat_pred) % 2

    ber_pred = torch.mean(corrected_pred).item()

    logical_syndrome_pred = torch.matmul(final_logic_mat, corrected_pred.long()) % 2
    ler_pred = torch.any(logical_syndrome_pred).item()

    return ber_pred, ler_pred