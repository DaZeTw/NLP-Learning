import inspect
import math
import re
import textwrap
from typing import List, Optional, Set, Tuple, Union

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
            print("New cache")
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


# -------- dependencies you already have -------------
# from transformers.models.mistral.modeling_mistral import repeat_kv
# from customize_attention import unpack_kv
# from transformers.cache_utils import DynamicCache
# ----------------------------------------------------


class ScratchpadPrunedAttention(nn.Module):
    def __init__(self,
                 base_attn: nn.Module,
                 layer_idx: int,
                 prune_from: int = 12,
                 debug: bool = False):
        super().__init__()
        self.attn = base_attn
        self.layer_idx = layer_idx
        self.prune_from = prune_from
        self.debug = debug

        self.keep_mask: torch.BoolTensor | None = None
        self.to_drop:   List[int] = []

        # debug flags (reset after each pad)
        self._printed_request = False
        self._printed_effect = False

    # ---------------- tracker hooks -----------------
    def append_position(self, keep: bool = True):
        if self.keep_mask is None:
            dev = self.attn.o_proj.weight.device
            self.keep_mask = torch.tensor([keep], dtype=torch.bool, device=dev)
        else:
            self.keep_mask = torch.cat(
                [self.keep_mask,
                 torch.tensor([keep], dtype=torch.bool,
                              device=self.keep_mask.device)]
            )

    def mark_positions_dropped(self, idx_list: List[int]):
        # ignore if above cutoff
        if self.prune_from >= 0 and self.layer_idx < self.prune_from:
            return

        # new pad close → reset debug flags
        self._printed_request = False
        self._printed_effect = False

        self.to_drop.extend(idx_list)

        if self.debug and not self._printed_request:
            tail = ", ".join(map(str, idx_list[:8]))
            if len(idx_list) > 8:
                tail += "…"
            # print(f"[L{self.layer_idx:02d}] drop request {len(idx_list)} → {tail}")
            self._printed_request = True

    # ---------------- prune helper ------------------
    def _compress_kv(self, pkv):
        k, v = unpack_kv(pkv, self.layer_idx)
        keep_idx = torch.nonzero(self.keep_mask).squeeze(-1)

        k_new = k[:, :, keep_idx, :].contiguous()
        v_new = v[:, :, keep_idx, :].contiguous()

        hard = isinstance(pkv, DynamicCache)
        if hard:                               # overwrite in-place
            pkv.key_cache[self.layer_idx] = k_new
            pkv.value_cache[self.layer_idx] = v_new
        else:                                 # legacy tuple path
            pkv = (k_new, v_new)

        return pkv, v_new, hard

    def _maybe_prune(self, pkv):
        if not self.to_drop or self.keep_mask is None:
            return pkv, None, False           # nothing to do

        drop_idx = torch.unique(
            torch.tensor(self.to_drop, device=self.keep_mask.device))
        self.keep_mask[drop_idx] = False
        self.to_drop.clear()

        pkv, v_new, hard = self._compress_kv(pkv)

        if self.debug and not self._printed_effect:
            kept = int(self.keep_mask.sum())
            total = len(self.keep_mask)
            mode = "HARD" if hard else "SOFT"
            print(f"[L{self.layer_idx:02d}] {mode} prune → keep {kept}/{total}")
            self._printed_effect = True

        return pkv, v_new, hard

    # ---------------- forward -----------------------
    def forward(self, hidden_states, **kwargs):
        kwargs.setdefault("output_attentions", True)
        ctx, attn_w, pkv = self.attn(hidden_states, **kwargs)

        # prune once right after base attention
        pkv, v_new, _ = self._maybe_prune(pkv)

        # early exit if this layer is not pruned
        if (self.prune_from >= 0 and self.layer_idx < self.prune_from) \
           or self.keep_mask is None or attn_w is None:
            return ctx, attn_w, pkv

        # apply keep-mask to attention weights
        keep = self.keep_mask.to(attn_w.device).view(1, 1, 1, -1)
        attn_w = attn_w * keep
        attn_w = attn_w / (attn_w.sum(-1, keepdim=True) + 1e-6)

        # select the correct V
        if v_new is None:                      # no hard slice happened
            _, v_to_use = unpack_kv(pkv, self.layer_idx)
        else:
            v_to_use = v_new

        v_all = repeat_kv(v_to_use, self.attn.num_key_value_groups)

        new_ctx = torch.matmul(attn_w, v_all)
        B, H, T, D = new_ctx.shape
        new_ctx = new_ctx.transpose(1, 2).reshape(B, T, H * D)
        out = self.attn.o_proj(new_ctx)

        # one-time confirmation that masked forward ran
        if self.debug and self._printed_effect:
            print(f"[L{self.layer_idx:02d}] ✔ masked forward → OK")
            self._printed_effect = False       # avoid duplicates

        return out, attn_w, pkv


# class SingleLayerScratchpadPruner(nn.Module):
#     """
#     Prunes scratch-pad tokens only in ONE designated layer.

#     * Accepts absolute indices from the tracker.
#     * Keeps `shift` = total tokens removed so far, so later
#       index lists are mapped correctly.
#     """

#     def __init__(self, base_attn: nn.Module,
#                  layer_idx: int,
#                  debug: bool = True):
#         super().__init__()
#         self.attn = base_attn
#         self.layer_idx = layer_idx
#         self.debug = debug

#         # ---- state ----
#         self.keep_mask: torch.BoolTensor | None = None
#         self.to_drop:   List[int] = []
#         self.shift = 0                       # how many tokens removed before

#         # debug flags
#         self._printed_effect = False

#     # ---------- tracker hooks -----------------------
#     def append_position(self, keep=True):
#         if self.keep_mask is None:
#             dev = self.attn.o_proj.weight.device
#             self.keep_mask = torch.tensor([keep], dtype=torch.bool, device=dev)
#         else:
#             self.keep_mask = torch.cat(
#                 [self.keep_mask,
#                  torch.tensor([keep], dtype=torch.bool,
#                               device=self.keep_mask.device)]
#             )

#     def mark_positions_dropped(self, idx_list: List[int]):
#         # translate absolute → current positions
#         adj = [i - self.shift for i in idx_list if i - self.shift >= 0]
#         self.to_drop.extend(adj)
#         if self.debug:
#             tail = ", ".join(map(str, adj[:8])) + ("…" if len(adj) > 8 else "")
#             print(
#                 f"[Mark to drop called]-[L{self.layer_idx}] drop request {len(adj)} → {tail}")

#     # ---------- helper to compress KV --------------
#     def _compress_kv(self, pkv):
#         k, v = unpack_kv(pkv, self.layer_idx)
#         keep_idx = torch.nonzero(self.keep_mask).squeeze(-1)
#         k_new = k[:, :, keep_idx, :].contiguous()
#         v_new = v[:, :, keep_idx, :].contiguous()

#         hard = isinstance(pkv, DynamicCache)
#         if hard:
#             pkv.key_cache[self.layer_idx] = k_new
#             pkv.value_cache[self.layer_idx] = v_new
#         else:        # tuple cache
#             pkv = (k_new, v_new)

#         return pkv, v_new, hard

#     # ---------- maybe prune once per pad -----------
#     def _maybe_prune(self, pkv):
#         if not self.to_drop or self.keep_mask is None:
#             return pkv, None, False

#         drop_idx = torch.unique(
#             torch.tensor(self.to_drop, device=self.keep_mask.device))
#         self.keep_mask[drop_idx] = False
#         self.to_drop.clear()

#         pkv, v_new, hard = self._compress_kv(pkv)

#         removed = int((~self.keep_mask).sum())
#         self.shift = removed                # update shift for future pads

#         if self.debug:
#             kept, total = int(self.keep_mask.sum()), len(self.keep_mask)
#             mode = "HARD" if hard else "SOFT"
#             print(f"[L{self.layer_idx}] {mode} prune → keep {kept}/{total}")
#             self._printed_effect = True

#         return pkv, v_new, hard

#     # ---------- forward ----------------------------
#     def forward(self, hidden_states, **kwargs):
#         kwargs.setdefault("output_attentions", True)
#         ctx, attn_w, pkv = self.attn(hidden_states, **kwargs)

#         pkv, v_new, _ = self._maybe_prune(pkv)

#         if self.keep_mask is None or attn_w is None:
#             return ctx, attn_w, pkv

#         keep = self.keep_mask.to(attn_w.device).view(1, 1, 1, -1)
#         attn_w = attn_w * keep
#         attn_w = attn_w / (attn_w.sum(-1, keepdim=True) + 1e-6)

#         if v_new is None:
#             _, v_to_use = unpack_kv(pkv, self.layer_idx)
#         else:
#             v_to_use = v_new

#         v_all = repeat_kv(v_to_use, self.attn.num_key_value_groups)
#         new_ctx = torch.matmul(attn_w, v_all)
#         B, H, T, D = new_ctx.shape
#         new_ctx = new_ctx.transpose(1, 2).reshape(B, T, H * D)
#         out = self.attn.o_proj(new_ctx)

#         if self.debug and self._printed_effect:
#             print(f"[L{self.layer_idx}] ✔ masked forward → OK")
#             self._printed_effect = False

#         return out, attn_w, pkv

class SingleLayerScratchpadPruner(nn.Module):
    """
    Prunes scratch-pad tokens only in ONE designated layer.
    """

    def __init__(self, base_attn: nn.Module, layer_idx: int, debug: bool = True):
        super().__init__()
        self.attn = base_attn
        self.layer_idx = layer_idx
        self.debug = debug

        self.keep_mask: torch.BoolTensor | None = None   # 1 0 0 1 …
        self.to_drop:   set[int] = set()                # absolute indices
        self.shift = 0                                  # tokens removed so far

        self._printed_effect = True                    # one-shot log flag

    def append_position(self, keep: bool = True):
        dev = self.attn.o_proj.weight.device
        bit = torch.tensor([keep], dtype=torch.bool, device=dev)
        self.keep_mask = bit if self.keep_mask is None else torch.cat(
            [self.keep_mask, bit])

    def mark_positions_dropped(self, idx_list: List[int]):
        # Convert ABS indices to RELATIVE indices before storing
        rel_idx = [i - self.shift for i in idx_list if i - self.shift >= 0]
        self.to_drop.update(rel_idx)  # store only relative positions

        if self.debug:
            tail = ", ".join(map(str, list(idx_list)[:8]))
            print(
                f"[Mark]-[L{self.layer_idx}] want drop {len(idx_list)} → {tail}{'…' if len(idx_list) > 8 else ''}")
            print(f"To drop tokens (rel to current KV): {self.to_drop}")

    def _compress_kv(self, pkv, keep_idx: torch.Tensor):
        """slice K/V and write back for DynamicCache."""
        k, v = unpack_kv(pkv, self.layer_idx)
        k_new = k[:, :, keep_idx, :].contiguous()
        v_new = v[:, :, keep_idx, :].contiguous()

        hard = isinstance(pkv, DynamicCache)
        if hard:
            pkv.key_cache[self.layer_idx] = k_new
            pkv.value_cache[self.layer_idx] = v_new
            return pkv, v_new, True
        else:
            return (k_new, v_new), v_new, False

    def _maybe_prune(self, pkv):
        if not self.to_drop or self.keep_mask is None:
            return pkv, None, False, None
        # ─────────────  stash state BEFORE we mutate anything  ──────────────
        shift_before = self.shift                 # << new
        # requested_abs = sorted(self.to_drop)       # << new

        old_len = len(self.keep_mask)
        device = self.keep_mask.device
        # print(f"keep_mask device: {self.keep_mask.device}")
        # print(f"to_drop sample: {self.to_drop}")

        # if self.keep_mask.device.type == "meta":
        #     raise RuntimeError(
        #         "keep_mask is on meta device — ensure model is properly initialized")

        # rel = torch.tensor([i - shift_before for i in requested_abs
        #                     if i - shift_before >= 0],
        #                    device=self.keep_mask.device).unique()
        rel = torch.tensor(sorted(self.to_drop), device=self.keep_mask.device)
        rel = rel[rel < old_len]

        if rel.numel() == 0:
            self.to_drop.clear()
            return pkv, None, False

        # ------------------------------------------------------------
        # flip bits + compress
        self.keep_mask[rel] = False
        keep_idx = self.keep_mask.nonzero().squeeze(-1)
        # Expected: torch.Size([T])
        print(f"[DEBUG] keep_mask.shape : {self.keep_mask.shape}")
        # Expected: torch.Size([K]), where K ≤ T
        print(f"[DEBUG] keep_idx.shape  : {keep_idx.shape}")

        # ─── just before you call _compress_kv ───────────────────────────────
        if self.debug:
            k_pre, v_pre = unpack_kv(pkv, self.layer_idx)
            print(f"[L{self.layer_idx}]   K before prune : {tuple(k_pre.shape)}")
            print(f"[L{self.layer_idx}]   V before prune : {tuple(v_pre.shape)}")

        pkv, v_new, hard = self._compress_kv(pkv, keep_idx)

        # ─── immediately after ------------------------------------------------
        if self.debug:
            k_post, v_post = unpack_kv(pkv, self.layer_idx)
            print(f"[L{self.layer_idx}]   K after  prune : {tuple(k_post.shape)}")
            print(f"[L{self.layer_idx}]   V after  prune : {tuple(v_post.shape)}")

        # trim mask
        self.keep_mask = self.keep_mask[keep_idx].clone()
        removed_now = old_len - len(self.keep_mask)
        keep_idx_new = torch.arange(
            len(self.keep_mask), device=self.keep_mask.device)
        # book-keeping
        self.shift += removed_now
        self.to_drop.clear()

        # ------------------------------------------------------------
        # DEBUG (now with correct numbers)
        if self.debug:
            k, v = unpack_kv(pkv, self.layer_idx)

            # Compute absolute positions for the current keep_mask
            abs_positions = list(
                range(self.shift, self.shift + len(self.keep_mask)))

            # Absolute positions we *wanted* to drop
            requested_abs = sorted([self.shift + i for i in rel.tolist()])

            # Absolute positions that actually got dropped
            removed_abs = list(range(shift_before, shift_before + removed_now))

            # Difference: tokens that were removed but NOT explicitly requested
            surprise = sorted(set(removed_abs) - set(requested_abs))

            print(f"\n[L{self.layer_idx}] 🔪 pruning triggered")
            print(f"  • rel size             : {rel.numel()}")
            print(f"  • removed_now          : {removed_now}")
            if surprise:
                print(
                    f"    ↳ from trailing hole : {surprise[:10]}{' …' if len(surprise) > 10 else ''}")
            print(
                f"  • keep                 : {int(self.keep_mask.sum())}/{len(self.keep_mask) + removed_now}")
            print(
                f"  • KV shape             : k={tuple(k.shape)}, v={tuple(v.shape)}")
            print(f"  • mode                 : {'HARD' if hard else 'SOFT'}\n")

        return pkv, v_new, hard, keep_idx_new

    def forward(self, hidden_states, **kwargs):
        kwargs["output_attentions"] = True
        kwargs["use_cache"] = True
        ctx, attn_w = self.attn(hidden_states, **kwargs)
        pkv = kwargs.get("past_key_value", None)
        # print(f"pkv type:{type(pkv)}")
        # ctx, attn_w, pkv = self.attn(hidden_states, **kwargs)

        pkv, v_new, _, keep_idx = self._maybe_prune(pkv)
        if self.debug and keep_idx is not None:
            klen = unpack_kv(pkv, self.layer_idx)[0].shape[-2]
            print(f"[L{self.layer_idx}]   max(keep_idx)={keep_idx.max().item()}  "
                  f"K_current={klen}")

        if keep_idx is None or attn_w is None or self.keep_mask is None:
            return ctx, attn_w       # early exit as before

        # 1. drop the columns that correspond to pruned keys
        attn_w = attn_w[..., keep_idx]                    # (B,H,Q,K_new)

        # 2. renormalise (keep_mask is already trimmed to K_new)
        attn_w = attn_w / (attn_w.sum(-1, keepdim=True) + 1e-6)

        # 3. choose the matching value tensor
        if v_new is None:
            _, v_use = unpack_kv(pkv, self.layer_idx)
        else:
            k_use, _ = unpack_kv(pkv, self.layer_idx)
            k_use = k_use[..., keep_idx, :]            # same slice as V
            v_use = v_new

        if self.debug:
            print(f"[L{self.layer_idx}] ⌁ masked-fwd  shapes")
            # (B,H,Q,K_new)
            print(f"    • attn_w   : {tuple(attn_w.shape)}")
            # (B,H,K_new,D)
            print(f"    • K_use    : {tuple(k_use.shape)}")
            # (B,H,K_new,D)")
            print(f"    • V_use    : {tuple(v_use.shape)}")
            # (K_new,)                                # (B,Hkv,K_new,D)
            print(f"    • keep_idx : {tuple(keep_idx.shape)}")

        v_all = repeat_kv(v_use, self.attn.num_key_value_groups)
        new_ctx = torch.matmul(attn_w, v_all)
        B, H, Q, D = new_ctx.shape                     # NB: H before Q!
        new_ctx = (
            new_ctx.permute(0, 2, 1, 3)               # (B, Q, H, D)
            .contiguous()
            .view(B, Q, H * D)                # (B, Q, 4096)
        )
        out = self.attn.o_proj(new_ctx)

        return out, attn_w


class ScratchpadTracker:
    """
    Detects scratch-pad regions in the token stream, skips them in the
    output, and instructs wrapped attention layers to prune their KV cache.
    """

    # OPEN_RE = re.compile(r"Scratchpad\s*:\s*$",               re.I)
    # CLOSE_RE = re.compile(r"Corrected\s*Words\s*:\s*$",        re.I)
    # FINAL_RE = re.compile(r"Final\s*Corrected\s*Words\s*:\s*$", re.I)
    OPEN_RE = re.compile(r"^Scratchpad\s*:", re.I)
    CLOSE_RE = re.compile(r"^Intermediate\s*Result\s*:", re.I)
    FINAL_RE = re.compile(r"^Final\s*Answer\s*:", re.I)

    def __init__(self, tokenizer, attn_modules, tail_len: int = 120):
        self.tok = tokenizer
        self.mods = attn_modules
        self.tail_len = tail_len

        self.ids:        list[int] = []
        self.cur_pad:    list[int] = []
        self.in_pad:     bool = False
        self.text_buf:   str = ""
        self.prompt_len: int = 0
        self.debug:      bool = False  # toggle this to enable/disable debug

    def initialize_with_prompt(self, input_ids: List[int]):
        self.prompt_len = len(input_ids)
        self.ids = input_ids.copy()
        self.text_buf = self.tok.decode(input_ids, skip_special_tokens=False)
        print(f"[Init] Input Tokens: {self.prompt_len}")
        print(f"[Init] Ids in Prompt: {len(self.ids)}")

        for i, _ in enumerate(input_ids):
            for mod_idx, mod in enumerate(self.mods):
                if self.debug:
                    print(
                        f"[Init] Attn Module {mod_idx}: append_position(pos={i}, keep=True)")
                mod.append_position(keep=True)

    def _tail_norm(self) -> str:
        last_newline = self.text_buf.rfind("\n")
        if last_newline == -1:
            return self.text_buf.strip()
        return self.text_buf[last_newline:].strip()

    def step(self, token_id: int) -> bool:
        pos = len(self.ids)
        self.ids.append(token_id)

        tok_str = self.tok.decode([token_id], skip_special_tokens=False)
        self.text_buf += tok_str

        tail = self._tail_norm()
        is_open = bool(self.OPEN_RE.search(tail))
        is_close = bool(self.CLOSE_RE.search(
            tail) or self.FINAL_RE.search(tail))

        # Default: outside any pad, so keep
        keep_flag = True

        if not self.in_pad and is_open:
            #  ─── we see "Scratchpad:" and we’re not already in one → start a new pad
            print(f"🟡 OPEN scratch-pad @ {pos}")
            self.in_pad = True
            self.cur_pad = []            # start collecting positions
            keep_flag = False          # drop the "Scratchpad:" marker itself

        elif self.in_pad:
            if is_close:
                #  ─── we see "Intermediate Result:" or "Corrected Words:" → end pad
                print(f"🔴 CLOSE scratch-pad @ {pos}")
                self._commit_drop()     # prune everything we collected
                self.in_pad = False
                self.cur_pad = []
                keep_flag = True      # keep the closing marker
            else:
                #  ─── we’re inside a pad, so mark this token for pruning
                self.cur_pad.append(pos)
                keep_flag = False

        # Tell each attention wrapper whether to keep or drop this token
        for mod_idx, mod in enumerate(self.mods):
            if self.debug:
                print(
                    f"[Step {pos}] Attn Module {mod_idx}: append_position(keep={keep_flag})")
            mod.append_position(keep=keep_flag)

        return keep_flag

    # def step(self, token_id: int) -> bool:
    #     pos = len(self.ids)
    #     self.ids.append(token_id)

    #     tok_str = self.tok.decode([token_id], skip_special_tokens=False)
    #     self.text_buf += tok_str

    #     if "\n" in tok_str:
    #         self.line_start = pos + 1
    #     elif not hasattr(self, "line_start"):
    #         self.line_start = 0

    #     tail = self._tail_norm()
    #     keep_flag = True

    #     if (not self.in_pad) and self.OPEN_RE.search(tail):
    #         print(f"🟡  OPEN scratch-pad @ {pos}")
    #         hdr_pos_open = list(range(self.line_start, pos + 1))
    #         self.in_pad = True
    #         self.cur_pad = hdr_pos_open.copy()

    #         # print(
    #         #     f"[DEBUG] OPEN → cur_pad initialized: {self.cur_pad[:10]}{' …' if len(self.cur_pad) > 10 else ''}")
    #         keep_flag = False

    #     elif self.in_pad:
    #         if self.CLOSE_RE.search(tail) or self.FINAL_RE.search(tail):
    #             print(f"🔴  CLOSE pad @ {pos}")
    #             hdr_pos_close = set(range(self.line_start, pos + 1))
    #             self.cur_pad = [
    #                 p for p in self.cur_pad if p not in hdr_pos_close]

    #             # print(
    #             #     f"[DEBUG] CLOSE → Committing drop for {len(self.cur_pad)} tokens")
    #             # print(
    #             #     f"[DEBUG] cur_pad to commit: {self.cur_pad[:10]}{' …' if len(self.cur_pad) > 10 else ''}")
    #             self._commit_drop()
    #             self.in_pad = False
    #             self.cur_pad = []
    #             keep_flag = True
    #         else:
    #             self.cur_pad.append(pos)
    #             # print(f"[DEBUG] Append to cur_pad: {pos}")
    #             keep_flag = False  # <-- this is where keep=False is set

    #     for mod_idx, mod in enumerate(self.mods):
    #         if self.debug:
    #             print(
    #                 f"[Step {pos}] Attn Module {mod_idx}: append_position(keep={keep_flag})")
    #         mod.append_position(keep=keep_flag)

    #     return keep_flag

    def _commit_drop(self):
        if not self.cur_pad:
            print("⚠️  pad list empty – nothing to drop")
            return
        print(
            f"🧹  pruning {len(self.cur_pad)} tokens (first five: {self.cur_pad[:5]})")
        for mod_idx, attn in enumerate(self.mods):
            if self.debug:
                print(
                    f"[Commit Drop] Attn Module {mod_idx}: mark_positions_dropped({self.cur_pad})")
            attn.mark_positions_dropped(self.cur_pad)
