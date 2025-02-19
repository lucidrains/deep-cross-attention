import torch
from torch import nn, cat, stack
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, RMSNorm

from rotary_embedding_torch import RotaryEmbedding

import einx
from einops import rearrange, einsum
from einops.layers.torch import Rearrange

# ein notation

# b - batch
# n -sequence
# h - heads
# l - logits
# o - number of grn outputs
# y - laYer

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.norm = RMSNorm(dim)

        self.heads = heads
        dim_inner = heads * dim_head

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.to_q = nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim_inner, bias = False))
        self.to_k = nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim_inner, bias = False))
        self.to_v = nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim_inner, bias = False))

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        q_input,
        k_input,
        v_input
    ):

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        q, k, v = map(self.split_heads, (q, k, v))

        # relative positions

        q, k = self.rotary_embed.rotate_queries_with_cached_keys(q, k)

        # attention branch

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal = True
        )

        out = self.merge_heads(out)

        return self.to_out(out)

# feedforward

def FeedForward(dim, expansion_factor = 4.):
    dim_hidden = int(dim * expansion_factor)

    return nn.Sequential(
        Linear(dim, dim_hidden),
        nn.GELU(),
        Linear(dim_hidden, dim)
    )

# GRNv3
# the input dependent one lines up with all the literature, and the winning solution for hyper connections (dynamic)

class GRN(Module):
    def __init__(
        self,
        dim,
        num_layers,
        num_outputs = 1
    ):
        super().__init__()

        self.num_outputs = num_outputs
        self.num_layers = num_layers

        self.to_aggregate = nn.Sequential(
            RMSNorm(dim),
            Linear(dim, num_outputs, bias = False),
            Rearrange('... outputs -> outputs ...')
        )

        self.bias = nn.Parameter(torch.zeros(num_outputs, num_layers))

        nn.init.zeros_(self.to_aggregate[-2].weight)

        with torch.no_grad():
            self.bias[:, -1] = 1.

    def forward(
        self,
        tokens_across_depth # Float['y b n d']
    ):
        assert self.num_layers == tokens_across_depth.shape[0]

        aggregate = self.to_aggregate(tokens_across_depth)

        aggregate = einx.add('o y ..., o y -> o y ...', aggregate, self.bias).relu()

        output = einsum(tokens_across_depth, aggregate, 'y b n d, o y b n -> o b n d')

        if self.num_outputs == 1:
            output = rearrange(output, '1 ... -> ...')

        return output

# DCA Decoder Block

class DCABlock(Module):
    def __init__(
        self,
        dim,
        *,
        grn_num_layers,
        dim_head = 64,
        heads = 8,
        ff_expansion_factor = 4.
    ):
        super().__init__()

        self.qkv_grn = GRN(dim, num_layers = grn_num_layers, num_outputs = 3)

        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads)

        self.pre_ff_norm = RMSNorm(dim)

        self.ff = FeedForward(dim = dim, expansion_factor = ff_expansion_factor)

    def forward(
        self,
        tokens_across_depth # Float['depth b n d']
    ):
        q_input, k_input, v_input = self.qkv_grn(tokens_across_depth)

        residual = q_input

        attn_out = self.attn(q_input, k_input, v_input)

        ff_input = self.pre_ff_norm(attn_out + residual)

        ff_out = self.ff(ff_input)

        return ff_out + attn_out

# classes

class DCAGPT(Module):
    def __init__(
        self,
        num_tokens,
        dim,
        depth,
        past_layers_k = 2,
        dim_head = 64,
        heads = 8,
        ff_expansion_factor = 4.
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        # the `k` hyperparameter, which seems to refer to sub sampling of which layers to include for efficiency
        # but weirdly, they not only do last k layers, but also the first k? also some mention about intermediate layers being pooled? just go with first and last for now

        self.past_layers_k = past_layers_k

        # the proposed DCA blocks

        dca_blocks = []
        for i in range(depth):

            dca = DCABlock(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_expansion_factor = ff_expansion_factor,
                grn_num_layers = min(past_layers_k * 2, i + 1)
            )

            dca_blocks.append(dca)

        self.dca_blocks = ModuleList(dca_blocks)

        # norm and logits

        self.final_grn = GRN(dim, num_layers = min(past_layers_k * 2, depth + 1))

        self.norm = RMSNorm(dim)
        self.to_logits = Linear(dim, num_tokens, bias = False)
 
    def select_layers(
        self,
        all_tokens_stacked
    ):
        k = self.past_layers_k # k in paper

        num_layers = all_tokens_stacked.shape[0]

        # determine which layers to include

        if num_layers < (k * 2):
            return all_tokens_stacked

        output = cat((
            all_tokens_stacked[:k], # first k layers
            all_tokens_stacked[-k:] # last k layers
        ))

        return output

    def forward(
        self,
        ids,
        return_loss = False
    ):

        if return_loss:
            ids, labels = ids[:, :-1], ids[:, 1:]

        tokens = self.token_emb(ids)

        all_tokens = [tokens]

        for dca_block in self.dca_blocks:

            all_tokens_stacked = stack(all_tokens)

            dca_block_input = self.select_layers(all_tokens_stacked)

            dca_out = dca_block(dca_block_input)

            # append dca output for next iteration

            all_tokens.append(dca_out)

        final_grn_input = self.select_layers(stack(all_tokens))

        pooled_tokens = self.final_grn(final_grn_input)

        embed = self.norm(pooled_tokens)

        logits = self.to_logits(embed)

        if not return_loss:
            return logits

        return F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)
