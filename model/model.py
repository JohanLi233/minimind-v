from .LMConfig import LMConfig

import math
from typing import Optional, Tuple, List
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

class MLAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) block - Corrected GQA Handling.
    """
    def __init__(self, args: LMConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Ensure n_kv_heads divides n_heads for GQA
        if self.n_heads % self.n_kv_heads != 0:
             raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        # MLA specific dimensions
        self.kv_latent_dim = args.kv_latent_dim
        self.q_latent_dim = args.q_latent_dim

        # Validate config attributes
        if not hasattr(args, 'kv_latent_dim') or args.kv_latent_dim is None:
             raise ValueError("LMConfig must include valid kv_latent_dim for MLAttention")
        if not hasattr(args, 'q_latent_dim') or args.q_latent_dim is None:
             raise ValueError("LMConfig must include valid q_latent_dim for MLAttention")
        if not hasattr(args, 'max_seq_len') or args.max_seq_len is None:
            raise ValueError("LMConfig must include max_seq_len for MLAttention mask")
        if not hasattr(args, 'rope_theta') or args.rope_theta is None:
            args.rope_theta = 10000.0 # Default value if missing
            print(f"Warning: rope_theta not found in LMConfig, defaulting to {args.rope_theta}")


        # --- MLA Projection Layers ---
        self.w_dkv = nn.Linear(args.dim, self.kv_latent_dim, bias=False)
        self.w_uk = nn.Linear(self.kv_latent_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.w_kr = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.w_uv = nn.Linear(self.kv_latent_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.w_dq = nn.Linear(args.dim, self.q_latent_dim, bias=False)
        self.w_uq = nn.Linear(self.q_latent_dim, self.n_heads * self.head_dim, bias=False)
        self.w_qr = nn.Linear(self.q_latent_dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False) # Output projection matches V shape

        # --- Standard Components ---
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and getattr(args, 'flash_attn', False) # Safely check flash_attn
        self.dropout = args.dropout

        # Causal mask buffer
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

        # Dimension per head for K after concatenation (used for scaling)
        self.k_concat_head_dim = 2 * self.head_dim

    def forward(
        self,
        x: torch.Tensor,
        pos_cis: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        bsz, seq_len, _ = x.shape

        # --- Query Path ---
        c_q = self.w_dq(x)
        q_u = self.w_uq(c_q).view(bsz, seq_len, self.n_heads, self.head_dim)
        q_r_pre_rope = self.w_qr(c_q).view(bsz, seq_len, self.n_heads, self.head_dim)
        q_r, _ = apply_rotary_emb(q_r_pre_rope, torch.zeros_like(q_r_pre_rope), pos_cis=pos_cis)
        xq = torch.cat([q_u, q_r], dim=-1) # Shape: [bsz, seq_len, n_heads, 2 * head_dim]

        # --- Key/Value Path ---
        c_kv_current = self.w_dkv(x)
        k_r_pre_rope_current = self.w_kr(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        _, k_r_current = apply_rotary_emb(torch.zeros_like(k_r_pre_rope_current), k_r_pre_rope_current, pos_cis=pos_cis)

        if past_key_value is not None:
            past_c_kv, past_k_r = past_key_value
            c_kv = torch.cat([past_c_kv, c_kv_current], dim=1)
            k_r = torch.cat([past_k_r, k_r_current], dim=1)
        else:
            c_kv = c_kv_current
            k_r = k_r_current

        k_u = self.w_uk(c_kv).view(bsz, -1, self.n_kv_heads, self.head_dim)
        v_u = self.w_uv(c_kv).view(bsz, -1, self.n_kv_heads, self.head_dim)

        mla_cache = (c_kv, k_r) if use_cache else None

        # Concatenate K components
        xk = torch.cat([k_u, k_r], dim=-1) # Shape: [bsz, total_seq_len, n_kv_heads, 2 * head_dim]
        # V component (NO concatenation)
        xv = v_u                         # Shape: [bsz, total_seq_len, n_kv_heads, head_dim]

        # --- GQA Expansion ---
        # Repeat K/V heads AFTER all K/V components are computed and concatenated/selected
        # Input shapes to repeat_kv:
        # xk: [bsz, total_seq_len, n_kv_heads, 2 * head_dim]
        # xv: [bsz, total_seq_len, n_kv_heads, head_dim]
        xk = repeat_kv(xk, self.n_rep) # Output: [bsz, total_seq_len, n_heads, 2 * head_dim]
        xv = repeat_kv(xv, self.n_rep) # Output: [bsz, total_seq_len, n_heads, head_dim]

        # --- Attention Calculation ---
        # Transpose to shape [bsz, n_heads, seq_len/total_seq_len, head_dim_variant]
        xq = xq.transpose(1, 2) # [bsz, n_heads, seq_len, 2 * head_dim]
        xk = xk.transpose(1, 2) # [bsz, n_heads, total_seq_len, 2 * head_dim]
        xv = xv.transpose(1, 2) # [bsz, n_heads, total_seq_len, head_dim]

        query_len = xq.shape[2]
        key_len = xk.shape[2]
        current_seq_len = key_len if past_key_value is None else key_len - past_key_value[0].shape[1] # Len of current input

        # Use Flash Attention 2 for training (seq_len == key_len) and potentially generation
        if self.flash:
            # Flash Attention requires Pytorch >= 2.0
            # It automatically handles causal masking when is_causal=True
            # It also handles GQA broadcasting correctly if K/V have n_kv_heads and Q has n_heads
            # However, we have already repeated K/V to n_heads, so standard SDPA works.
            dropout_p = self.dropout if self.training else 0.0
            # We need a causal mask only when generating tokens one by one and q_len != k_len
            # But SDPA's is_causal=True handles the standard causal case (q_len == k_len)
            # For generation (q_len=1), is_causal doesn't apply, but the matmul works correctly.
            # Let's rely on is_causal for the training case (q_len == k_len == seq_len)
            is_causal_sdpa = query_len == key_len # Only apply SDPA causal mask if lengths match (training)

            try:
                 # Use causal mask for training, no mask needed for single-token generation (handled by slicing)
                output = F.scaled_dot_product_attention(
                    xq, xk, xv,
                    attn_mask=None, # Masking handled by is_causal or slicing in generation
                    dropout_p=dropout_p,
                    is_causal=is_causal_sdpa # Make causal mask conditional
                )
            except Exception as e:
                 # Fallback if SDPA fails for some reason (e.g., specific dtype issue)
                 print(f"SDPA failed with error: {e}. Falling back to manual attention.")
                 if query_len == key_len: # Manual causal calculation for training
                     scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.k_concat_head_dim)
                     mask_slice = self.mask[:, :, :query_len, :key_len]
                     scores = scores + mask_slice
                     scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                     scores = self.attn_dropout(scores)
                     output = torch.matmul(scores, xv)
                 else: # Manual calculation for generation (q_len=1)
                    scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.k_concat_head_dim)
                    # No causal mask needed here as query_len is 1
                    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                    scores = self.attn_dropout(scores)
                    output = torch.matmul(scores, xv)

        else:
            # Manual calculation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.k_concat_head_dim)

            # Apply causal mask only if needed (standard training case)
            if query_len == key_len:
                # Extract the correct slice of the mask
                # Mask shape is (1, 1, max_seq_len, max_seq_len)
                # We need mask for current query positions interacting with all key positions
                mask_slice = self.mask[:, :, :query_len, :key_len]
                scores = scores + mask_slice
            # For generation (query_len=1), no causal mask is needed for the scores matrix

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # Shape: [bsz, n_heads, seq_len, head_dim]

        # --- Reshape and Output Projection ---
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1) # Shape: [bsz, seq_len, n_heads * head_dim]
        output = self.resid_dropout(self.wo(output)) # Shape: [bsz, seq_len, dim]

        return output, mla_cache

class DynamicTanh(nn.Module):
    """
    实现 Dynamic Tanh (DyT) 层，作为 Normalization 层的替代方案。
    DyT(x) = γ * tanh(αx) + β
    """
    def __init__(self, dim: int, alpha):
        """
        初始化 DyT 层。

        参数:
            dim (int): 输入特征的维度 (对应 Norm 层中的 dim 或 normalized_shape[-1])。
            init_a (float): 可学习标量 alpha 的初始值。默认为 0.5。
        """
        super().__init__()
        self.dim = dim
        # 可学习的标量 alpha
        self.alpha = nn.Parameter(torch.ones(1) * alpha)
        # 可学习的逐通道缩放 gamma (γ)，初始化为 1
        self.gamma = nn.Parameter(torch.ones(dim))
        # 可学习的逐通道平移 beta (β)，初始化为 0
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        DyT 的前向传播。
        期望 x 的形状是 (..., dim)。
        """
        # alpha (标量) 会自动广播
        # gamma 和 beta (形状 dim) 会自动广播到 x 的最后一个维度
        return self.gamma * torch.tanh(self.alpha * x) + self.beta

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: LMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, pos_cis)
        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )
        if self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores += self.mask[:, :, :seq_len, :seq_len]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoEGate(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(config)

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # 训练模式下，重复输入数据
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理模式下，只选择最优专家
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 例如当tokens_per_expert=[6, 15, 20, 26, 33, 38, 46, 52]
        # 当token_idxs=[3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...]
        # 意味着当token_idxs[:6] -> [3,  7, 19, 21, 24, 25,  4]位置的token都由专家0处理，token_idxs[6:15]位置的token都由专家1处理......
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 使用 scatter_add_ 进行 sum 操作
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        # MLA
        self.attention = MLAttention(config)
        # self.attention = Attention(config)

        self.layer_id = layer_id

        # RMSNorm
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

        # DynamicTanh
        # self.attention_norm = DynamicTanh(config.dim, alpha=config.alpha * 1.5)
        # self.ffn_norm = DynamicTanh(config.dim, alpha=config.alpha)

        self.feed_forward = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        h_attn, past_kv = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        h = x + h_attn
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, past_kv


class MiniMindLM(PreTrainedModel):
    config_class = LMConfig

    def __init__(self, params: LMConfig = None):
        self.params = params or LMConfig()
        super().__init__(self.params)
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, params) for l in range(self.n_layers)])

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        # self.norm = DynamicTanh(params.dim, alpha=params.alpha)

        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight
        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim=params.dim // params.n_heads, theta=params.rope_theta),
                             persistent=False)
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args):
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = args.get('start_pos', 0)
        h = self.dropout(self.tok_embeddings(input_ids))
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]
        past_kvs = []
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis,
                past_key_value=past_key_values[l],
                use_cache=use_cache
            )
            past_kvs.append(past_kv)
        logits = self.output(self.norm(h))
        aux_loss = sum(l.feed_forward.aux_loss for l in self.layers if isinstance(l.feed_forward, MOEFeedForward))
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT

    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, pad_token_id=0, **args):
        # 流式生成
        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

        # 直接生成
        generated = []
        for i in range(input_ids.size(0)):
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            out = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
            tokens_list = [tokens[:, -1:] for tokens in out]
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)
        max_length = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat(
                [seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)],
                dim=-1)
            for seq in generated
        ]
        return torch.cat(generated, dim=0)

    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        while input_ids.shape[1] < max_new_tokens - 1:
            if first_seq or not use_cache:
                out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args), False
            else:
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                           start_pos=input_ids.shape[1] - 1, **args)
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            logits /= (temperature + 1e-9)
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            yield input_ids[:, start:]
            if input_ids_next.item() == eos_token_id:
                break
