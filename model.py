import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, TransformerConv, global_mean_pool

class QWP(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, heads=4, num_layers=4, L = 4,dropout=0.1, noise_type = 'depolarization', num_stabs_total = None, code_type = "toric"):
        super().__init__()

        self.L = L
        self.noise_type = noise_type

        if code_type == 'rotated' and noise_type == 'depolarization':
            embedding_num_nodes = num_stabs_total + 2
        elif code_type == 'rotated' and noise_type == 'independent':
            embedding_num_nodes = num_stabs_total + 1
        else:
            embedding_num_nodes = num_stabs_total

        self.num_nodes = num_stabs_total
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.stabilizer_embeddings = nn.Embedding(embedding_num_nodes, hidden_dim)
        self.embeddings_norm = nn.LayerNorm(hidden_dim)

        #-----------------------------------------------------------------------------------------------------------------------------
        self.k_eigenvectors = 8
        self.type_dim = 2
        self.coords_dim = 2
        self.dist_to_center_dim = 1

        embed_dim_per_feature = hidden_dim // 4

        # ====== distance embedding =========
        self.max_distance = 2 * L  
        self.distance_embed_dim = 128  
        self.distance_embedding = nn.Embedding(self.max_distance + 1, self.distance_embed_dim) # maps integer distances to vectors in length embed dim
        self.distance_norm = nn.LayerNorm(self.distance_embed_dim)
        # ===================================

        # ====== seperate node features embed =========
        self.type_embedding = nn.Linear(self.type_dim, embed_dim_per_feature) #one layer for one hot encoded
        self.type_norm = nn.LayerNorm(embed_dim_per_feature)

        self.coords_embedding = nn.Sequential(
            nn.Linear(self.coords_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim_per_feature)
        )

        self.coords_norm = nn.LayerNorm(embed_dim_per_feature)

        self.dist_embedding = nn.Sequential(
            nn.Linear(self.dist_to_center_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim_per_feature)
        )

        self.dist_norm = nn.LayerNorm(embed_dim_per_feature)

        self.pe_embedding = nn.Sequential(
            nn.Linear(self.k_eigenvectors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim_per_feature)
        )
        self.pe_norm = nn.LayerNorm(embed_dim_per_feature)

        #-----------------------------------------------------------------------------------------------------------------------------
        combined_node_feat = (self.hidden_dim * 2)
        self.pre_graph_proj = nn.Sequential(
            nn.Linear(combined_node_feat, hidden_dim),
            nn.LayerNorm(hidden_dim))
        #-----------------------------------------------------------------------------------------------------------------------------

        # ======= edge encoding =============
        self.rem_feats_dim = max(0, edge_feat_dim - 1)
        self.rem_enc_dim = 64 # A size for the encoded features (can be tuned)
        if self.rem_feats_dim > 0:
            self.rem_enc = nn.Sequential(
                nn.Linear(self.rem_feats_dim, self.rem_enc_dim),
                nn.ReLU(),
                nn.LayerNorm(self.rem_enc_dim),
                nn.Linear(self.rem_enc_dim, self.rem_enc_dim),
                nn.ReLU(),
                nn.LayerNorm(self.rem_enc_dim),
            )
        else:
            self.rem_enc = None
        self.GNN_edge_dim = self.distance_embed_dim + self.rem_enc_dim
        #-----------------------------------------------------------------------------------------------------------------------------
        #encoding layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=heads,
                dropout=dropout,
                beta=True,      # internal residual for attention
                concat=False,
            )
            norm1 = nn.LayerNorm(hidden_dim)  # pre-norm for attention
            norm2 = nn.LayerNorm(hidden_dim)  # pre-norm for FFN


            ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4), 
                nn.GELU(),               
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )

            self.layers.append(nn.ModuleDict({
                "conv": conv,
                "norm1": norm1,
                "ffn": ffn,
                "norm2": norm2
            }))
        transformer_input_dim = (2 * hidden_dim) + self.distance_embed_dim + self.rem_enc_dim
        self.predictor_norm = nn.LayerNorm(transformer_input_dim)
        #================= Transformer prediction layers =================================
        self.transformer_pred_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_input_dim,
                nhead=4, # Number of attention heads for the transformer
                dim_feedforward=512, # Dimension of the feedforward network model in the transformer
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        self.output_lin = nn.Linear(transformer_input_dim, 1)
        #================= Transformer prediction layers =================================
    
    
    def forward(self, x, edge_index, edge_attr, syndrome):
        #===== seperate feature embedding ======
        type_vec = x[:, :self.type_dim]
        coords = x[:, self.type_dim:self.type_dim + self.coords_dim]
        dist_to_center = x[:, self.type_dim + self.coords_dim:self.type_dim + self.coords_dim + self.dist_to_center_dim]
        pe = x[:, self.type_dim + self.coords_dim + self.dist_to_center_dim:]


        type_embed = self.type_embedding(type_vec)
        type_embed = self.type_norm(type_embed)

        coords_embed = self.coords_embedding(coords)
        coords_embed = self.coords_norm(coords_embed)

        dist_embed = self.dist_embedding(dist_to_center)
        dist_embed = self.dist_norm(dist_embed)

        pe_embed = self.pe_embedding(pe)
        pe_embed = self.pe_norm(pe_embed)
        #========================================
        num_nodes_in_batch = x.shape[0]
        x = torch.cat([type_embed, coords_embed, dist_embed, pe_embed], dim=1) #hidden dim
        #-----------------------------------------------------------------------------------------------------------------------------
        node_indices = torch.arange(num_nodes_in_batch, device=x.device)
        learnable_embeds = self.stabilizer_embeddings(node_indices)
        learnable_embeds = self.embeddings_norm(learnable_embeds)   
        
        syndrome_pm = (syndrome.float() * 2) - 1.0 #modulate syndrome
        embeds = torch.cat([x, learnable_embeds], dim=1)
        x_from_embeds = embeds * syndrome_pm.unsqueeze(-1)
        
        x = self.pre_graph_proj(x_from_embeds)
        
        # --- Build per-edge features ---
        distance_ids = edge_attr[:, 0].long().clamp(max=self.max_distance)
        distance_embeds = self.distance_embedding(distance_ids)
        distance_embeds = self.distance_norm(distance_embeds) # normalizing the embeddings
        directional_feats = edge_attr[:, 1:]  # dx, dy

        if self.rem_feats_dim > 0:
            rem_enc_features = self.rem_enc(directional_feats)
        else:
            rem_enc_features = torch.empty(directional_feats.size(0), 0).to(x.device)
        
        
        GNN_edge_features = torch.cat([distance_embeds, rem_enc_features], dim=1)
        
        # TransformerConv + FFN (per layer)
        for layer in self.layers:
            # --- Attention block ---
            x_res = x
            x_norm = layer["norm1"](x)                               # Pre-norm
            x_att = layer["conv"]((x_norm, x_norm), edge_index)               
            x = x_att + x_res
            # --- FFN block ---
            ffn_out = layer["ffn"](layer["norm2"](x))           # Pre-norm
            x = x + ffn_out                                     # Residual for FFN only

        src, tgt = edge_index
        x_src, x_tgt = x[src], x[tgt]
        edge_input = torch.cat([x_src, x_tgt, distance_embeds, rem_enc_features], dim=1)
        #edge_input = torch.cat([x_src, x_tgt], dim=1)
        edge_input = self.predictor_norm(edge_input)
        transformer_output = self.transformer_pred_layers(edge_input.unsqueeze(0)).squeeze(0)
        
        logits = self.output_lin(transformer_output)


        return logits.squeeze(-1)