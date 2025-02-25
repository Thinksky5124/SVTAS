import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "Dimension must be divisible by the number of heads."
        self.scale = self.head_dim ** -0.5

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        HW = H * W

        # Flatten spatial dimensions and project queries, keys, and values
        Q = self.query_proj(x1.view(B, C, HW).permute(0, 2, 1))  # (B, HW, C)
        K = self.key_proj(x2.view(B, C, HW).permute(0, 2, 1))    # (B, HW, C)
        V = self.value_proj(x2.view(B, C, HW).permute(0, 2, 1))  # (B, HW, C)

        Q = Q.view(B, HW, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, HW, head_dim)
        K = K.view(B, HW, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, HW, head_dim)
        V = V.view(B, HW, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, HW, head_dim)

        # Compute attention scores: (B, num_heads, HW, head_dim) x (B, num_heads, head_dim, HW)
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale  # (B, num_heads, HW, HW)
        attention_probs = torch.softmax(attention_scores, dim=-1)             # (B, num_heads, HW, HW)

        # Compute weighted sum of values
        attention_output = torch.matmul(attention_probs, V)  # (B, num_heads, HW, head_dim)

        # Concatenate heads and project back
        attention_output = attention_output.permute(0, 2, 1, 3).reshape(B, HW, -1)  # (B, HW, C)
        output = self.output_proj(attention_output).permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)

        return output
    
# class CrossAttention(nn.Module):
#     def __init__(self, dim, num_heads):
#         super(CrossAttention, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5

#         # Projection layers for queries, keys, and values
#         self.query_proj = nn.Linear(dim, dim)
#         self.key_proj = nn.Linear(dim, dim)
#         self.value_proj = nn.Linear(dim, dim)
#         self.output_proj = nn.Linear(dim, dim)

#     def forward(self, x1, x2):
#         # Flatten spatial dimensions
#         B, C, H, W = x1.shape
#         A_flat = x1.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
#         B_flat = x2.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

#         # Project queries, keys, and values
#         Q_A = self.query_proj(A_flat)  # (B, HW, C)
#         K_B = self.key_proj(B_flat)   # (B, HW, C)
#         V_B = self.value_proj(B_flat) # (B, HW, C)

#         # Compute attention scores
#         attention_scores = torch.bmm(Q_A, K_B.transpose(1, 2)) * self.scale  # (B, HW, HW)
#         attention_probs = torch.softmax(attention_scores, dim=-1)            # (B, HW, HW)

#         # Compute weighted sum of values
#         attention_output = torch.bmm(attention_probs, V_B)  # (B, HW, C)

#         # Project output back and reshape to original dimensions
#         output = self.output_proj(attention_output).permute(0, 2, 1)  # (B, C, HW)
#         return output.view(B, C, H, W)

# # Example usage
# batch, channels, height, width = 1, 2048, 7, 7
# A = torch.randn(batch, channels, height, width)  # Feature map A
# B = torch.randn(batch, channels, height, width)  # Feature map B

# cross_attention = CrossAttention(dim=channels, num_heads=8)
# output_A_to_B = cross_attention(A, B)  # Cross-attention output A -> B
