import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.mistral.modeling_mistral import repeat_kv

# --------------------------------------------
#  1.  generic KV-extractor
# --------------------------------------------


def unpack_kv(pkv, layer_idx: int):
    """
    Return (key, value) tensors for *this* layer, no matter the cache type.

    pkv can be:
      • tuple(key, value)  – legacy format
      • DynamicCache       – and all its subclasses (OffloadedCache, QuantizedCache)

    Both tensors are shaped [B, H_kv, S, D]
    """
    # ---- legacy tuple ------------------------------------------------------
    if isinstance(pkv, tuple):
        return pkv

    # ---- new cache classes -------------------------------------------------
    if isinstance(pkv, DynamicCache):
        # DynamicCache implements  __getitem__(idx)  →  (k, v)
        return pkv[layer_idx]                      # ⚠️ needs the layer index!

    # ---- last-chance fall-back --------------------------------------------
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        return pkv.key_cache[layer_idx], pkv.value_cache[layer_idx]

    # nothing matched
    raise TypeError(f"unpack_kv: unsupported cache type {type(pkv)}")


class ZeroAttention(nn.Module):
    def __init__(self, attn: nn.Module, thresh: float = 0.01):
        super().__init__()
        self.attn = attn
        self.th = thresh

    def forward(self, hidden_states, **kwargs):
        # make sure we get weights, but don't duplicate the kwarg
        kwargs.setdefault("output_attentions", True)

        # run the real attention
        ctx, attn_w, pkv = self.attn(hidden_states, **kwargs)
        if attn_w is None or not isinstance(pkv, tuple):
            # if we’re in SDPA path (attn_w None) or pkv is a Cache,
            # skip pruning – just return what we got
            return ctx, attn_w, pkv

        # ----- pruning mask (B,H,T,S) --------------------------------------
        print("###Masking tokens###")
        mask = (attn_w.mean(1) > self.th).unsqueeze(1)   # (B,1,T,S)
        pruned = attn_w * mask

        # ----- full value tensor (B,H,S,D) ---------------------------------
        v_kv = pkv[1]                                   # tuple (k,v)
        v_all = repeat_kv(v_kv, self.attn.num_key_value_groups)

        # ----- new context --------------------------------------------------
        new_ctx = torch.matmul(pruned, v_all)            # (B,H,T,D)
        B, H, T, D = new_ctx.shape
        new_ctx = new_ctx.transpose(1, 2).reshape(B, T, H * D)
        out = self.attn.o_proj(new_ctx)

        return out, pruned, pkv


# --------------------------------------------
#  2.  pruning wrapper
# --------------------------------------------
class PrunedAttention(nn.Module):
    def __init__(self, base_attn: nn.Module, thresh: float = 0.05, layer_idx: Optional[int] = None):
        """
        Parameters
        ----------
        base_attn   the original `MistralAttention` module
        thresh      keep tokens whose mean-attention ≥ thresh
        layer_idx   *required* when the model uses DynamicCache-style caches
        """
        super().__init__()
        self.attn = base_attn
        self.th = thresh
        self.layer_idx = layer_idx

    # ----------------------------------------
    def forward(self, hidden_states, **kwargs):
        # make sure weights are returned
        kwargs.setdefault("output_attentions", True)

        # run the real attention
        ctx, attn_w, pkv = self.attn(hidden_states, **kwargs)

        # Abort early if we cannot prune (e.g. SDPA path, Flash-Attn)
        if attn_w is None:
            return ctx, attn_w, pkv

        # ------------------ obtain K / V ------------------
        try:
            k, v = unpack_kv(pkv, self.layer_idx or 0)
        except Exception as e:
            print("⚠️ prune-skip:", e)
            return ctx, attn_w, pkv

        # ------------------ build importance mask ---------
        # attn_w: [B, H_q, T, S]
        keep = (attn_w.mean(dim=(1, 2)) > self.th)   # [B, S]  True ↔ keep
        if not keep.any():
            # nothing survives → fall back to original output
            return ctx, attn_w, pkv

        # scatter mask back to full [B, 1, T, S]
        pruned_weights = attn_w * keep[:, None, None, :]

        # ------------------ recompute context -------------
        v_full = repeat_kv(v, self.attn.num_key_value_groups)  # [B, H_q, S, D]
        new_ctx = torch.matmul(pruned_weights, v_full)         # [B, H_q, T, D]
        B, H, T, D = new_ctx.shape
        new_ctx = new_ctx.transpose(1, 2).reshape(B, T, H * D)
        out = self.attn.o_proj(new_ctx)
        # print("PrunedAttention is being used")
        return out, pruned_weights, pkv


class TopKPrunedAttention(nn.Module):
    """
    * shallow layers 0 … prune_start_layer-1 : no pruning
    * deep    layers prune_start_layer … end : keep Top-K tokens by mean-attention
    """

    def __init__(
        self,
        base_attn: nn.Module,
        layer_idx: int,                   # ← pass this when you wrap
        prune_start_layer: int = 16,      # first layer where pruning begins
        k_keep: Union[int, List[int]] = 128,    # constant K or per-layer list
    ):
        super().__init__()
        self.attn = base_attn
        self.layer_idx = layer_idx
        self.prune_start = prune_start_layer

        # allow a per-layer schedule
        if isinstance(k_keep, list):
            self.k_for_layer = k_keep[layer_idx]
        else:
            self.k_for_layer = k_keep

    # ----------------------------------------------------------
    def forward(self, hidden_states, **kwargs):
        kwargs.setdefault("output_attentions", True)
        ctx, attn_w, pkv = self.attn(hidden_states, **kwargs)

        # (1) skip if we’re in a path that doesn’t give attn or before pruning starts
        if attn_w is None or self.layer_idx < self.prune_start:
            return ctx, attn_w, pkv

        # (2) get K,V for *this* layer
        try:
            k, v = unpack_kv(pkv, self.layer_idx)
        except Exception as e:
            print("prune-skip:", e)
            return ctx, attn_w, pkv

        # (3) importance = mean attention over heads + query positions
        #     attn_w: [B, H_q, T_q, S]   →   [B, S]
        importance = attn_w.mean(dim=(1, 2))

        # Top-K indices  (shape: [B, k])
        k_keep = min(self.k_for_layer, importance.shape[1])
        topk_idx = importance.topk(k_keep, dim=-1).indices

        # (4) slice K/V  — note we have to gather batch-wise
        k_new = torch.stack([k[b, :, topk_idx[b], :]
                            for b in range(k.size(0))])
        v_new = torch.stack([v[b, :, topk_idx[b], :]
                            for b in range(v.size(0))])

        # (5) recompute attention with pruned memory
        q = self.attn.q_proj(hidden_states)            # (B, T, D)
        B, T, D = q.shape
        H_q = self.attn.num_heads
        q = q.view(B, T, H_q, -1).transpose(1, 2)      # (B, H_q, T, Dh)

        # ⚠️  NEW — broadcast K/V to all query heads
        # (B, H_q, S', Dh)
        k_all = repeat_kv(k_new, self.attn.num_key_value_groups)
        # (B, H_q, S', Dh)
        v_all = repeat_kv(v_new, self.attn.num_key_value_groups)

        # q  [B,Hq,T,Dh]  ·  k_all^T [B,Hq,Dh,S’]  =  [B,Hq,T,S’]
        attn_scores = torch.matmul(q, k_all.transpose(-2, -1))
        attn_scores /= math.sqrt(q.size(-1))
        attn_probs = attn_scores.softmax(dim=-1)

        ctx_new = torch.matmul(attn_probs, v_all)      # (B,Hq,T,Dh)
        ctx_new = ctx_new.transpose(1, 2).reshape(B, T, -1)
        out = self.attn.o_proj(ctx_new)
        if isinstance(pkv, DynamicCache):
            new_cache = DynamicCache()
            # Copy everything as-is
            new_cache.key_cache = list(pkv.key_cache)
            new_cache.value_cache = list(pkv.value_cache)
            # Replace only this layer
            new_cache.key_cache[self.layer_idx] = k_new
            new_cache.value_cache[self.layer_idx] = v_new
            pkv = new_cache
        # ⬇️  do NOT replace pkv – just pass it through unchanged
        return out, attn_probs, pkv


def layerwise_topk(layer_idx, total_layers=32):
    # Linear schedule: prune less at early layers
    min_k, max_k = 192, 64
    k = int(min_k - (layer_idx / (total_layers - 1)) * (min_k - max_k))
    return k


class ReasoningAwareAttention(nn.Module):
    def __init__(self, attn: nn.Module, tokenizer, prompt_token_ids, layer_idx: int):
        super().__init__()
        self.attn = attn
        self.tokenizer = tokenizer
        self.prompt_token_ids = set(prompt_token_ids)
        self.layer_idx = layer_idx

    def forward(self, hidden_states, **kwargs):
        kwargs.setdefault("output_attentions", True)
        ctx, attn_w, pkv = self.attn(hidden_states, **kwargs)
        if attn_w is None or not isinstance(pkv, tuple):
            return ctx, attn_w, pkv

        last_token_attn = attn_w[:, :, -1, :]  # (B, H, S)
        importance = last_token_attn.mean(1)   # (B, S)

        # Boost reasoning tokens
        for b in range(importance.shape[0]):
            for s in range(importance.shape[1]):
                if s in self.prompt_token_ids:
                    importance[b, s] *= 2.5  # Boosting weight

        # Top-k strategy
        k = layerwise_topk(self.layer_idx)
        topk_indices = torch.topk(importance, k, dim=1).indices  # (B, k)

        # Force-keep reasoning tokens at deeper layers
        if self.layer_idx >= 24:
            for b in range(importance.shape[0]):
                for idx in self.prompt_token_ids:
                    if idx < importance.shape[1]:
                        topk_indices[b] = torch.unique(
                            torch.cat([topk_indices[b], torch.tensor([idx], device=importance.device)]))

        # Build binary mask (B, H, 1, S)
        mask = torch.zeros_like(attn_w)
        for b in range(mask.shape[0]):
            for h in range(mask.shape[1]):
                mask[b, h, -1, topk_indices[b]] = 1.0

        pruned = attn_w * mask
        v_kv = pkv[1]
        v_all = repeat_kv(v_kv, self.attn.num_key_value_groups)

        new_ctx = torch.matmul(pruned, v_all)
        B, H, T, D = new_ctx.shape
        new_ctx = new_ctx.transpose(1, 2).reshape(B, T, H * D)
        out = self.attn.o_proj(new_ctx)

        return out, pruned, pkv


reasoning_cues = ["step", "step:", "therefore",
                  "because", "so", "conclusion", "hence"]
