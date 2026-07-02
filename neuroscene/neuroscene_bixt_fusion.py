import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiXTCrossAttention(nn.Module):
    """
    The core Bi-Directional Cross-Attention module from BiXT.
    Computes a single shared attention matrix to update both the latents and the inputs.
    """
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        # Linear projections
        self.proj_r_lat = nn.Linear(d_model, d_model)
        self.proj_r_x = nn.Linear(d_model, d_model)
        self.proj_v_lat = nn.Linear(d_model, d_model)
        self.proj_v_x = nn.Linear(d_model, d_model)

        self.sa_proj_q_lat = nn.Linear(d_model, d_model)
        self.sa_proj_k_lat = nn.Linear(d_model, d_model)
        self.sa_proj_v_lat = nn.Linear(d_model, d_model)

        # MLP Layers
        self.mlp_lat = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.mlp_x = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(d_model * 4, d_model)
        )

        # Norm Layers
        self.norm_lat1 = nn.LayerNorm(d_model)
        self.norm_lat2 = nn.LayerNorm(d_model)
        self.norm_x1 = nn.LayerNorm(d_model)
        self.norm_x2 = nn.LayerNorm(d_model)

        self.out_latent = nn.Linear(d_model, d_model)
        self.out_input = nn.Linear(d_model, d_model)

        # Gated Learnable Params for Training Stability (Unused in this test iteration)
        self.gate_attn_lat = nn.Parameter(torch.zeros(1))
        self.gate_attn_x = nn.Parameter(torch.zeros(1))
        self.gate_ffn_lat = nn.Parameter(torch.zeros(1))
        self.gate_ffn_x = nn.Parameter(torch.zeros(1))

    def forward(self, latents, inputs):
        B, M, D = latents.shape
        _, N_total, _ = inputs.shape

        # 1. Project inputs
        r_lat = self.proj_r_lat(self.norm_lat1(latents))
        r_x = self.proj_r_x(self.norm_x1(inputs))
        v_lat = self.proj_v_lat(self.norm_lat1(latents))
        v_x = self.proj_v_x(self.norm_x1(inputs))

        # 2. Reshape for Multi-Head Attention: [B, SeqLen, D] -> [B, num_heads, SeqLen, head_dim]
        r_lat = r_lat.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        r_x = r_x.view(B, N_total, self.num_heads, self.head_dim).transpose(1, 2)
        v_lat = v_lat.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v_x = v_x.view(B, N_total, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Compute Shared Attention Matrix (BiXT style)
        # Scaled by math.sqrt(head_dim) instead of d_model for multi-head mathematical stability
        a_lat_tok = torch.matmul(r_lat, r_x.transpose(-2, -1)) / math.sqrt(self.head_dim)
        a_tok_lat = a_lat_tok.transpose(-2, -1)

        # 4. Compute Deltas (Ensure dim=-1 is provided to Softmax)
        delta_lat = torch.matmul(torch.softmax(a_lat_tok, dim=-1), v_x)
        delta_tok = torch.matmul(torch.softmax(a_tok_lat, dim=-1), v_lat)

        # 5. Reshape Multi-Head back to [B, SeqLen, D]
        delta_lat = delta_lat.transpose(1, 2).contiguous().view(B, M, D)
        delta_tok = delta_tok.transpose(1, 2).contiguous().view(B, N_total, D)

        # 6. Apply Residuals
        latents = torch.add(latents, delta_lat)
        inputs = torch.add(inputs, delta_tok)

        # 7. Apply MLPs with Residuals
        latents = torch.add(latents, self.mlp_lat(self.norm_lat2(latents)))
        inputs = torch.add(inputs, self.mlp_x(self.norm_x2(inputs)))

        out_l = self.out_latent(latents)
        out_x = self.out_input(inputs)

        # 8. Self-Attention on Latents (Reshaped for Multi-Head)
        q_sa = self.sa_proj_q_lat(out_l).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        k_sa = self.sa_proj_k_lat(out_l).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v_sa = self.sa_proj_v_lat(out_l).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q_sa, k_sa.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        sa_out = torch.matmul(attn_weights, v_sa)
        sa_out = sa_out.transpose(1, 2).contiguous().view(B, M, D)

        # Add residual connection for self-attention
        out_l = torch.add(out_l, sa_out)

        return out_l, out_x


class NeuroSceneBiXTFusion(nn.Module):
    """
    The wrapper that manages the concatenation of the RGB-D spatial and visual features,
    routes them through the BiXT layers, and slices the output to preserve the 3D hash map shape.
    """
    def __init__(self, d_model=512, num_latents=128, num_layers=3, num_heads=8):
        super().__init__()
        self.num_latents = num_latents

        self.latent_queries = nn.Parameter(torch.randn(1, num_latents, d_model))

        self.layers = nn.ModuleList([
            BiXTCrossAttention(d_model=d_model, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, spatial_feats, visual_feats):
        B, N, D = spatial_feats.shape
        latents = self.latent_queries.expand(B, -1, -1)

        combined_inputs = torch.cat([spatial_feats, visual_feats], dim=1)

        for layer in self.layers:
            latents, combined_inputs = layer(latents, combined_inputs)

        updated_spatial = combined_inputs[:, :N, :]
        return self.final_norm(updated_spatial)


# --- Quick Verification ---
if __name__ == "__main__":
    B = 1
    N = 10000    # 10k Voxels and 10k Pixels
    C = 512      # Feature Dimension

    spatial_tensor = torch.randn(B, N, C)
    visual_tensor = torch.randn(B, N, C)

    model = NeuroSceneBiXTFusion(d_model=C, num_latents=128, num_layers=3)

    print("--- Testing BiXT Forward Pass ---")
    output = model(spatial_tensor, visual_tensor)

    print(f"Input Voxel Shape:  {spatial_tensor.shape}")
    print(f"Output Voxel Shape: {output.shape}")
    assert output.shape == (B, N, C), "Output voxel dimensions do not match input!"
    print("Success! Dimensions match perfectly for the 3D Hash Map.")