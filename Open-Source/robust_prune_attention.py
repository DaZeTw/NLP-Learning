# import ast
# import csv
# import difflib
# import itertools
# import json
# import random
# import re
# import statistics
# import textwrap
# import time
# from collections import Counter
# from pathlib import Path
# from typing import Dict, List, Tuple, Union

# import matplotlib.pyplot as plt
# import nltk
# import torch
# import torch.nn as nn

# # -----------------------------------------------------------------------------
# # Custom attention wrappers that remain unchanged from your original codebase.
# # -----------------------------------------------------------------------------
# from customize_attention import DynamicCache  # helper class for HARD cache ops
# from customize_attention import unpack_kv  # helper imported here for clarity
# from customize_attention import (
#     PrunedAttention,
#     ScratchpadPrunedAttention,
#     TopKPrunedAttention,
# )
# from nltk.tokenize import word_tokenize
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.models.mistral.modeling_mistral import repeat_kv

# # =============================================================================
# #  Global configuration (NO AGE‑DECAY, purely probabilistic)
# # =============================================================================
# PRUNE_CFG: dict = dict(
#     tail_len=120,    # re‑evaluate pruning every N generated tokens
#     prob_min=0.05,   # drop probability at layer 0
#     prob_max=0.90,   # drop probability at final layer
# )

# # =============================================================================
# #  Scratch‑Pad Tracker  — step‑level, *no* age handling
# # =============================================================================


# class ScratchpadTracker:
#     """Detects scratch‑pad spans and periodically notifies pruners."""

#     OPEN_RE = re.compile(r"^Scratchpad\s*:", re.I)
#     CLOSE_RE = re.compile(r"^Intermediate\s*Result\s*:\s*", re.I)
#     FINAL_RE = re.compile(r"^Final\s*Answer\s*:\s*", re.I)

#     def __init__(self, tokenizer, attn_modules, *, cfg: dict | None = None):
#         self.tok = tokenizer
#         self.mods = attn_modules
#         self.cfg = cfg or PRUNE_CFG

#         # runtime state --------------------------------------------------
#         self.ids: List[int] = []
#         self.text_buf: str = ""
#         self.prompt_len: int = 0
#         self.gen_step: int = 0

#         self.cur_pad: List[int] = []  # indices of the pad currently open
#         self.in_pad: bool = False

#         # ledger of *completed* pads, each {start, end}
#         self.pad_regions: List[dict] = []

#     # ------------------------------------------------------------------
#     def initialize_with_prompt(self, input_ids: List[int]):
#         self.prompt_len = len(input_ids)
#         self.ids = input_ids.copy()
#         self.text_buf = self.tok.decode(input_ids, skip_special_tokens=False)
#         self.gen_step = 0

#         for mod in self.mods:
#             for _ in input_ids:
#                 mod.append_position(keep=True)

#     # ------------------------------------------------------------------
#     def _tail_norm(self) -> str:
#         last_new = self.text_buf.rfind("\n")
#         return (self.text_buf[last_new + 1:] if last_new != -1 else self.text_buf).strip()

#     # ------------------------------------------------------------------
#     def step(self, token_id: int) -> bool:
#         self.gen_step += 1
#         pos = len(self.ids)
#         self.ids.append(token_id)

#         tok_str = self.tok.decode([token_id], skip_special_tokens=False)
#         self.text_buf += tok_str
#         tail = self._tail_norm()

#         is_open = bool(self.OPEN_RE.search(tail))
#         is_close = bool(self.CLOSE_RE.search(
#             tail) or self.FINAL_RE.search(tail))

#         keep_flag = True

#         # --------------------------------------------------------------
#         # scratch‑pad boundary tracking
#         # --------------------------------------------------------------
#         if is_open and not self.in_pad:
#             print(f"🟡 OPEN scratch-pad @ {pos}")
#             self.in_pad = True
#             self.cur_pad = [pos]
#             keep_flag = False

#         elif self.in_pad:
#             self.cur_pad.append(pos)
#             keep_flag = False
#             if is_close:
#                 print(f"🔴 CLOSE scratch-pad @ {pos}")
#                 self.pad_regions.append(
#                     dict(start=self.cur_pad[0], end=self.cur_pad[-1]))
#                 self.in_pad = False
#                 self.cur_pad = []
#                 keep_flag = True
#                 self._commit_pruning_decisions()

#         # notify pruners for this position
#         for mod in self.mods:
#             mod.append_position(keep=keep_flag)

#         # periodic evaluation
#         # if self.gen_step % self.cfg["tail_len"] == 0:
#         #     self._commit_pruning_decisions()

#         return keep_flag

#     # ------------------------------------------------------------------
#     def _commit_pruning_decisions(self):
#         if not self.pad_regions:
#             return
#         for mod in self.mods:
#             mod.evaluate_and_mark(self.pad_regions.copy())

# # =============================================================================
# #  Probabilistic Step‑Level Pruner (applies to *every* layer)
# # =============================================================================


# class SingleLayerScratchpadPruner(nn.Module):
#     """Wraps one attention layer and drops *whole* pad blocks with probability
#     that increases from `prob_min` at layer 0 to `prob_max` at top layer."""

#     def __init__(self, base_attn: nn.Module, *, layer_idx: int, total_layers: int, cfg: dict | None = None, debug: bool = False):
#         super().__init__()
#         self.attn = base_attn
#         self.layer_idx = layer_idx
#         self.total_layers = total_layers
#         self.cfg = cfg or PRUNE_CFG
#         self.debug = debug

#         self.keep_mask: torch.BoolTensor | None = None
#         self.to_drop: set[int] = set()
#         self.shift = 0

#     # ------------------------------------------------------------------
#     #  Streaming hooks
#     # ------------------------------------------------------------------
#     def append_position(self, *, keep: bool):
#         dev = self.attn.o_proj.weight.device
#         bit = torch.tensor([keep], dtype=torch.bool, device=dev)
#         self.keep_mask = bit if self.keep_mask is None else torch.cat(
#             [self.keep_mask, bit])

#     def evaluate_and_mark(self, ledger: List[dict]):
#         depth_ratio = self.layer_idx / (self.total_layers - 1)
#         drop_prob = self.cfg["prob_min"] + depth_ratio * \
#             (self.cfg["prob_max"] - self.cfg["prob_min"])

#         for rec in ledger:
#             if random.random() < drop_prob:
#                 # convert ABS ➔ REL indices
#                 rel_start = rec["start"] - self.shift
#                 rel_end = rec["end"] - self.shift
#                 if rel_end >= 0:                       # still inside current KV
#                     self.to_drop.update(range(max(0, rel_start), rel_end + 1))

#     # ------------------------------------------------------------------
#     #  KV compression (identical logic)
#     # ------------------------------------------------------------------

#     def _compress_kv(self, pkv, keep_idx):
#         k, v = unpack_kv(pkv, self.layer_idx)
#         k_new = k[:, :, keep_idx, :].contiguous()
#         v_new = v[:, :, keep_idx, :].contiguous()

#         hard = isinstance(pkv, DynamicCache)
#         if hard:
#             pkv.key_cache[self.layer_idx] = k_new
#             pkv.value_cache[self.layer_idx] = v_new
#             return pkv, v_new, True
#         else:
#             return (k_new, v_new), v_new, False

#     def _maybe_prune(self, pkv):
#         if not self.to_drop or self.keep_mask is None:
#             return pkv, None, False, None

#         rel = torch.tensor(sorted(self.to_drop), device=self.keep_mask.device)
#         rel = rel[rel < len(self.keep_mask)]
#         if rel.numel() == 0:
#             self.to_drop.clear()
#             return pkv, None, False, None

#         # flip mask bits
#         old_len = len(self.keep_mask)
#         self.keep_mask[rel] = False
#         keep_idx = self.keep_mask.nonzero(as_tuple=False).squeeze(-1)

#         # compress KV
#         pkv, v_new, hard = self._compress_kv(pkv, keep_idx)

#         # bookkeeping --------------------------------------------------
#         removed_now = old_len - len(keep_idx)
#         self.shift += removed_now          # <── accumulate total drop
#         self.keep_mask = self.keep_mask[keep_idx].clone()
#         self.to_drop.clear()
#         return pkv, v_new, hard, keep_idx

#     # ------------------------------------------------------------------
#     def forward(self, hidden_states, **kwargs):
#         kwargs["output_attentions"] = True
#         kwargs["use_cache"] = True
#         ctx, attn_w = self.attn(hidden_states, **kwargs)
#         pkv = kwargs.get("past_key_value", None)

#         pkv, v_new, _, keep_idx = self._maybe_prune(pkv)
#         if keep_idx is None or attn_w is None or self.keep_mask is None:
#             return ctx, attn_w

#         attn_w = attn_w[..., keep_idx]
#         attn_w = attn_w / (attn_w.sum(-1, keepdim=True) + 1e-6)

#         k_use, _ = unpack_kv(pkv, self.layer_idx)
#         k_use = k_use[..., keep_idx, :]
#         v_all = repeat_kv(v_new if v_new is not None else unpack_kv(
#             pkv, self.layer_idx)[1], self.attn.num_key_value_groups)
#         new_ctx = torch.matmul(attn_w, v_all)
#         B, H, Q, D = new_ctx.shape
#         new_ctx = new_ctx.permute(0, 2, 1, 3).contiguous().view(B, Q, H * D)
#         out = self.attn.o_proj(new_ctx)
#         return out, attn_w


import random
import re
from typing import Dict, List

import torch
import torch.nn as nn
from customize_attention import DynamicCache, unpack_kv
from transformers.models.mistral.modeling_mistral import repeat_kv

# ============================================================
# Config
# ============================================================
PRUNE_CFG = dict(
    prob_min=0.05,   # drop prob at layer 0  (shallow)
    prob_max=0.90    # drop prob at top layer (deep)
)

# ============================================================
# Probabilistic Single‑Layer Pruner  (shift‑aware)
# ============================================================


class SingleLayerScratchpadPruner(nn.Module):
    def __init__(self, base_attn: nn.Module, *, layer_idx: int, total_layers: int, cfg: dict | None = None, debug: bool = False):
        super().__init__()
        self.attn = base_attn
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.cfg = cfg or PRUNE_CFG
        self.debug = debug
        self.to_drop: list[tuple[int, int]] = []
        self.shift = 0

        # For age decay: persistent buffer of closed spans (only for early layers)
        self.span_buffer: List[dict] = []  # [{start, end, close_time}, ...]

    def append_position(self, *, keep: bool):
        # No-op: we do not track keep_mask anymore
        pass

    def evaluate_and_mark(self, ledger: List[dict], gen_step: int = None, age_threshold: int = 200):
        current_time = gen_step if gen_step is not None else 0

        if self.layer_idx < 10:
            # Age-decay logic for early layers
            for rec in ledger:
                rel_start = rec["start"] - self.shift
                rel_end = rec["end"] - self.shift

                span_with_time = {
                    "start": rel_start,
                    "end": rel_end,
                    "shift_at_add": self.shift,
                    "close_time": current_time
                }
                self.span_buffer.append(span_with_time)
                if self.debug:
                    print(
                        f"[L{self.layer_idx}] 🧠 Buffer span ({rel_start},{rel_end}) at shift={self.shift} time={current_time}")

            # Re-evaluate buffer
            spans_to_prune = []
            for span in self.span_buffer:
                age = current_time - span["close_time"]
                if age >= age_threshold:
                    rel_start = span["start"] + \
                        (self.shift - span["shift_at_add"])
                    rel_end = span["end"] + (self.shift - span["shift_at_add"])

                    if rel_end < 0 or rel_start >= rel_end:
                        continue  # span fully dropped already

                    self.to_drop.append((rel_start, rel_end))
                    spans_to_prune.append(span)
                    if self.debug:
                        print(
                            f"[AgeDecay L{self.layer_idx}] ✅ Drop span ({rel_start},{rel_end}) age={age}")

            # Clean up buffer
            self.span_buffer = [
                s for s in self.span_buffer if s not in spans_to_prune]

        else:
            # Depth-based probabilistic pruning for late layers
            depth_ratio = self.layer_idx / (self.total_layers - 1)
            drop_prob = self.cfg["prob_min"] + depth_ratio * \
                (self.cfg["prob_max"] - self.cfg["prob_min"])

            for rec in ledger:
                if random.random() < drop_prob:
                    rel_start = rec["start"] - self.shift
                    rel_end = rec["end"] - self.shift

                    if rel_end < 0 or rel_start >= rel_end:
                        continue

                    self.to_drop.append((rel_start, rel_end))
                    if self.debug:
                        print(
                            f"[DepthDrop L{self.layer_idx}] ✅ Drop span ({rel_start},{rel_end}) prob={drop_prob:.2f}")

    def _compress_kv(self, pkv, keep_idx):
        k, v = unpack_kv(pkv, self.layer_idx)
        print(
            f"[L{self.layer_idx}] 🔍 compress_kv: original KV shape = {k.shape}, {v.shape}")
        # print(f"[L{self.layer_idx}] 🔍 compress_kv: keep_idx = {keep_idx.tolist()}")

        k_new = k[:, :, keep_idx, :].contiguous()
        v_new = v[:, :, keep_idx, :].contiguous()
        print(
            f"[L{self.layer_idx}] ✅ compress_kv: new KV shape = {k_new.shape}, {v_new.shape}")

        hard = isinstance(pkv, DynamicCache)
        if hard:
            print(
                f"[L{self.layer_idx}] 🧠 Using DynamicCache → updating key_cache/value_cache directly")
            pkv.key_cache[self.layer_idx] = k_new
            pkv.value_cache[self.layer_idx] = v_new
            return pkv, v_new, True
        return (k_new, v_new), v_new, False

    def _maybe_prune(self, pkv):
        if not self.to_drop:
            return pkv, None, False, None

        k, v = unpack_kv(pkv, self.layer_idx)
        N = k.shape[2]
        drop_indices = set()
        for start, end in self.to_drop:
            if start >= N or end < 0:
                print(
                    f"[L{self.layer_idx}] ⚠️ Skipping invalid drop span: ({start},{end}) — KV size = {N}")
                continue
            drop_indices.update(range(start, end + 1))
            print(
                f"[L{self.layer_idx}] 📌 Request drop span: ({start},{end}) with shift={self.shift} on KV size={N}"
            )

        keep_indices = sorted(set(range(N)) - drop_indices)

        if not keep_indices:
            print(f"[L{self.layer_idx}] ⚠️ All tokens dropped — pruning aborted.")
            self.to_drop.clear()
            return pkv, None, False, None

        print(
            f"[L{self.layer_idx}] ✅ Total {len(drop_indices)} tokens to drop. {len(keep_indices)} will be kept.")

        keep_idx_tensor = torch.tensor(keep_indices, device=k.device)
        pkv, v_new, hard = self._compress_kv(pkv, keep_idx_tensor)

        removed_now = N - len(keep_indices)
        print(f"[L{self.layer_idx}] 🔧 Removed {removed_now} tokens, shift before={self.shift}, after={self.shift + removed_now}")

        self.shift += removed_now
        self.to_drop.clear()

        return pkv, v_new, hard, keep_idx_tensor

    # def forward(self, hidden_states, **kwargs):
    #     kwargs["output_attentions"] = True
    #     kwargs["use_cache"] = True
    #     ctx, attn_w = self.attn(hidden_states, **kwargs)
    #     pkv = kwargs.get("past_key_value", None)
    #     pkv, v_new, _, keep_idx = self._maybe_prune(pkv)
    #     if keep_idx is None or attn_w is None:
    #         return ctx, attn_w
    #     # Safe trim before slicing
    #     Lw = attn_w.shape[-1]
    #     k_full, v_full = unpack_kv(pkv, self.layer_idx)
    #     Nk = v_full.shape[2]
    #     max_idx = int(keep_idx.max())
    #     if max_idx >= Lw or max_idx >= Nk:
    #         if self.debug:
    #             print(
    #                 f"[L{self.layer_idx}] ⚠️ Trimming keep_idx {max_idx}>={{Lw,Nk}}=({Lw},{Nk})")
    #         keep_idx = keep_idx[(keep_idx < Lw) & (keep_idx < Nk)]
    #         if keep_idx.numel() == 0:
    #             return ctx, attn_w
    #     attn_w = attn_w[..., keep_idx]
    #     if attn_w is not None:
    #         if keep_idx.max().item() >= attn_w.shape[-1]:
    #             print(
    #                 f"[L{self.layer_idx}] ❌ attn_w.shape[-1]={attn_w.shape[-1]}, keep_idx.max={keep_idx.max().item()}")

    #     attn_w = attn_w / (attn_w.sum(-1, keepdim=True) + 1e-6)
    #     # slice values
    #     if v_new is not None:
    #         v_select = v_new
    #     else:
    #         v_select = v_full[..., keep_idx, :]
    #     v_all = repeat_kv(v_select, self.attn.num_key_value_groups)
    #     # compute new context
    #     new_ctx = torch.matmul(attn_w, v_all)
    #     B, H, Q, D = new_ctx.shape
    #     new_ctx = new_ctx.permute(0, 2, 1, 3).contiguous().view(B, Q, H*D)
    #     out = self.attn.o_proj(new_ctx)
    #     return out, attn_w

    # def forward(self, hidden_states, **kwargs):
    #     # Pre-attention prune
    #     pkv = kwargs.get('past_key_value', None)
    #     pkv, v_new, hard, keep_idx = self._maybe_prune(pkv)
    #     kwargs['past_key_value'] = pkv

    #     # Get the actual KV cache size after any pruning from previous layers
    #     if pkv is not None:
    #         if hard:  # DynamicCache
    #             actual_kv_len = pkv.key_cache[self.layer_idx].shape[2] if len(pkv.key_cache) > self.layer_idx else 0
    #         else:  # tuple cache
    #             k_cache, v_cache = pkv
    #             actual_kv_len = k_cache.shape[2] if k_cache is not None else 0
    #     else:
    #         actual_kv_len = 0

    #     # Adjust cache_position to match actual KV cache size
    #     if 'cache_position' in kwargs and kwargs['cache_position'] is not None:
    #         cp = kwargs['cache_position']
    #         # Calculate total shift needed to align with actual KV cache
    #         if actual_kv_len > 0:
    #             expected_len = cp.max().item() + 1  # Expected total length
    #             actual_shift = expected_len - actual_kv_len
    #             new_cp = cp - actual_shift
    #             kwargs['cache_position'] = new_cp
    #             if self.debug:
    #                 print(f"[L{self.layer_idx}] 🔄 Adjusted cache_position: expected_len={expected_len}, actual_kv_len={actual_kv_len}, shift={actual_shift}")

    #     # compute attention
    #     kwargs['output_attentions'] = True
    #     kwargs['use_cache'] = True
    #     ctx, attn_w = self.attn(hidden_states, **kwargs)

    #     if self.debug:
    #         print(
    #             f"[L{self.layer_idx}] 🛠 post-attn kv-shift={self.shift}, attn_w.shape={attn_w.shape}")
    #     return ctx, attn_w
    
    # def forward(self, hidden_states, **kwargs):
    # # Pre-attention prune
    #     pkv = kwargs.get('past_key_value', None)
    #     pkv, v_new, hard, keep_idx = self._maybe_prune(pkv)
    #     kwargs['past_key_value'] = pkv

    #     # Fix cache position to match pruned KV cache
    #     if pkv is not None:
    #         if hard:  # DynamicCache
    #             current_kv_len = pkv.key_cache[self.layer_idx].shape[2] if len(pkv.key_cache) > self.layer_idx else 0
    #         else:  # tuple cache
    #             current_kv_len = pkv[0].shape[2] if pkv[0] is not None else 0
            
    #         # Create cache position that matches current KV cache size
    #         if current_kv_len > 0:
    #             device = hidden_states.device
    #             new_cache_position = torch.arange(current_kv_len, device=device).unsqueeze(0)
    #             kwargs['cache_position'] = new_cache_position
        
    #     # Run attention
    #     kwargs['output_attentions'] = True
    #     kwargs['use_cache'] = True
    #     ctx, attn_w = self.attn(hidden_states, **kwargs)

    #     if self.debug:
    #         kv_size = pkv.key_cache[self.layer_idx].shape[2] if pkv and hard else "unknown"
    #         print(f"[L{self.layer_idx}] 🛠 post-attn kv-shift={self.shift}, attn_w.shape={attn_w.shape}, kv_size={kv_size}")
        
    #     return ctx, attn_w
    
    # def forward(self, hidden_states, **kwargs):
    # # Pre-attention prune
    #     pkv = kwargs.get('past_key_value', None)
    #     pkv, v_new, hard, keep_idx = self._maybe_prune(pkv)
    #     kwargs['past_key_value'] = pkv

    #     # DON'T modify cache_position - let it be the original token position
    #     # The attention mechanism will handle the pruned cache correctly
        
    #     # Run attention
    #     kwargs['output_attentions'] = True
    #     kwargs['use_cache'] = True
    #     ctx, attn_w = self.attn(hidden_states, **kwargs)

    #     return ctx, attn_w
    
    def forward(self, hidden_states, **kwargs):
        # Pre-attention prune
        pkv = kwargs.get('past_key_value', None)
        pkv, v_new, hard, keep_idx = self._maybe_prune(pkv)
        kwargs['past_key_value'] = pkv

        # CRITICAL FIX: Synchronize cache_position with actual KV cache size
        if pkv is not None and 'cache_position' in kwargs and kwargs['cache_position'] is not None:
            if hard:  # DynamicCache
                actual_kv_len = pkv.key_cache[self.layer_idx].shape[2] if len(pkv.key_cache) > self.layer_idx else 0
            else:  # tuple cache
                actual_kv_len = pkv[0].shape[2] if pkv[0] is not None else 0
            
            # For autoregressive generation, cache_position should be the LAST position in the cache
            if actual_kv_len > 0:
                device = kwargs['cache_position'].device
                # Set cache_position to the last position in the actual cache
                kwargs['cache_position'] = torch.tensor([actual_kv_len - 1], device=device)
                
                if self.debug:
                    print(f"[L{self.layer_idx}] 🔧 Fixed cache_position to [{actual_kv_len - 1}] for KV cache size {actual_kv_len}")
        
        # Run attention with synchronized parameters
        kwargs['output_attentions'] = True
        kwargs['use_cache'] = True
        ctx, attn_w = self.attn(hidden_states, **kwargs)

        if self.debug:
            kv_size = pkv.key_cache[self.layer_idx].shape[2] if pkv and hard else "unknown"
            print(f"[L{self.layer_idx}] 🛠 post-attn kv-shift={self.shift}, attn_w.shape={attn_w.shape}, kv_size={kv_size}")
        
        return ctx, attn_w
    
    # def forward(self, hidden_states, **kwargs):
    #     # Pre-attention prune
    #     pkv = kwargs.get('past_key_value', None)
    #     pkv, v_new, hard, keep_idx = self._maybe_prune(pkv)
    #     kwargs['past_key_value'] = pkv

    #     # Disable attention mask to avoid size mismatches
    #     kwargs['attention_mask'] = None
        
    #     # Run attention
    #     kwargs['output_attentions'] = True
    #     kwargs['use_cache'] = True
    #     ctx, attn_w = self.attn(hidden_states, **kwargs)

    #     if self.debug:
    #         kv_size = pkv.key_cache[self.layer_idx].shape[2] if pkv and hard else "unknown"
    #         print(f"[L{self.layer_idx}] 🛠 post-attn kv-shift={self.shift}, attn_w.shape={attn_w.shape}, kv_size={kv_size}")
        
    #     return ctx, attn_w

    def forward(self, hidden_states, **kwargs):
        # Prune the cache, but don't replace it in kwargs
        pkv = kwargs.get('past_key_value', None)
        self._maybe_prune(pkv)  # Modifies pkv in-place
        
        # Let attention use the (now pruned) cache normally
        kwargs['output_attentions'] = True
        kwargs['use_cache'] = True
        ctx, attn_w = self.attn(hidden_states, **kwargs)
        
        return ctx, attn_w
    
    
    def forward(self, hidden_states, **kwargs):
        """
        This is the corrected forward pass.
        1. It prunes the KV cache for the current layer *before* attention.
        2. It passes the pruned cache to the original attention mechanism.
        3. It does NOT modify `cache_position`, allowing the underlying
           attention code to handle mask slicing automatically, which prevents size mismatches.
        """
        if self.debug:
            # Log the state *before* pruning and attention
            pkv_pre = kwargs.get("past_key_value", None)
            if pkv_pre:
                pre_kv_len = unpack_kv(pkv_pre, self.layer_idx)[0].shape[2]
                print(f"[L{self.layer_idx}] ➡️ Pre-prune KV len: {pre_kv_len}, Current shift: {self.shift}")

        # Step 1: Prune this layer's KV cache before doing anything else.
        pkv = kwargs.get('past_key_value', None)
        pkv, _, _, _ = self._maybe_prune(pkv)
        kwargs['past_key_value'] = pkv

        # Step 2: Call the original attention mechanism.
        # We do not modify `cache_position` or `attention_mask`. The underlying
        # `eager_attention_forward` will correctly slice the mask to match the
        # (potentially smaller) size of our pruned `pkv`.
        ctx, attn_w = self.attn(hidden_states, **kwargs)

        if self.debug:
            # Log the state *after* attention
            pkv_post = kwargs.get("past_key_value", None)
            if pkv_post:
                post_kv_len = unpack_kv(pkv_post, self.layer_idx)[0].shape[2]
                attn_w_shape = attn_w.shape if attn_w is not None else "N/A"
                print(f"[L{self.layer_idx}] ⬅️ Post-attn KV len: {post_kv_len}, attn_w.shape: {attn_w_shape}")
        
        return ctx, attn_w
    # ============================================================
# ScratchpadTracker (unchanged except no tail_len logic)
# ============================================================


class ScratchpadTracker:

    OPEN_RE = re.compile(r"Scratchpad\s*:\s*$",               re.I)
    CLOSE_RE = re.compile(r"Corrected\s*Words\s*:\s*$",        re.I)
    FINAL_RE = re.compile(r"Final\s*Corrected\s*Words\s*:\s*$", re.I)
    # OPEN_RE = re.compile(r"^Scratchpad\s*:", re.I)
    # CLOSE_RE = re.compile(r"^Intermediate\s*Result\s*:", re.I)
    # FINAL_RE = re.compile(r"^Final\s*Answer\s*:", re.I)

    def __init__(self, tokenizer, attn_modules):
        self.tok = tokenizer
        self.mods = attn_modules
        self.ids: List[int] = []
        self.cur_pad: List[int] = []
        self.in_pad = False
        self.text_buf = ""

    # ------------------------------------------
    def initialize_with_prompt(self, input_ids: List[int]):
        self.ids = input_ids.copy()
        self.text_buf = self.tok.decode(input_ids, skip_special_tokens=False)

    # ------------------------------------------
    def _tail(self):
        last = self.text_buf.rfind("\n")
        return (self.text_buf[last+1:] if last != -1 else self.text_buf).strip()

    def step(self, token_id: int):
        pos = len(self.ids)
        self.ids.append(token_id)
        self.text_buf += self.tok.decode([token_id], skip_special_tokens=False)
        tail = self._tail()

        is_open = bool(self.OPEN_RE.search(tail))
        is_close = bool(self.CLOSE_RE.search(
            tail) or self.FINAL_RE.search(tail))

        if is_open and not self.in_pad:
            self.in_pad = True
            self.cur_pad = [pos]
        elif self.in_pad:
            self.cur_pad.append(pos)
            if is_close:
                # send span to every layer wrapper, with gen_step for age decay
                print(
                    f"🚩 Add span to Pruner: ({self.cur_pad[0]}, {self.cur_pad[-1]})")
                for mod in self.mods:
                    if hasattr(mod, "evaluate_and_mark"):
                        mod.evaluate_and_mark(
                            [{"start": self.cur_pad[0], "end": self.cur_pad[-1]}], gen_step=len(self.ids), age_threshold=20)
                self.in_pad = False
                self.cur_pad = []

        # Always return True (all tokens are kept for output purposes)
        return True


# BUGGGG
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1176])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1176])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1176])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1176])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1176])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1176])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1176])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1145])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1176])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1145])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1176])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1176])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1145])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1176])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1145])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1145])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1145])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1145])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1145])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1145])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1145])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1176])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1145])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1145])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1145])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1145])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1145])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1177])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1177])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1177])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1177])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1177])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1177])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1177])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1177])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1177])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1177])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1177])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1177])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1146])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1177])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1146])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1177])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1177])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1146])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1177])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1146])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1146])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1146])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1146])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1146])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1146])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1146])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1177])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1146])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1146])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1146])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1146])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1146])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1178])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1178])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1178])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1178])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1178])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1178])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1178])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1178])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1178])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1178])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1178])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1178])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1147])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1178])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1147])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1178])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1178])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1147])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1178])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1147])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1147])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1147])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1147])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1147])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1147])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1147])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1178])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1147])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1147])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1147])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1147])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1147])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1179])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1179])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1179])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1179])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1179])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1179])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1179])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1179])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1179])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1179])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1179])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1179])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1148])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1179])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1148])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1179])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1179])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1148])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1179])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1148])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1148])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1148])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1148])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1148])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1148])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1148])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1179])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1148])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1148])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1148])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1148])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1148])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1180])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1180])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1180])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1180])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1180])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1180])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1180])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1180])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1180])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1180])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1180])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1180])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1149])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1180])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1149])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1180])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1180])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1149])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1180])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1149])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1149])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1149])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1149])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1149])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1149])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1149])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1180])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1149])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1149])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1149])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1149])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1149])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1181])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1181])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1181])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1181])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1181])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1181])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1181])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1181])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1181])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1181])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1181])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1181])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1150])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1181])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1150])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1181])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1181])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1150])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1181])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1150])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1150])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1150])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1150])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1150])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1150])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1150])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1181])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1150])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1150])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1150])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1150])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1150])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1182])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1182])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1182])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1182])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1182])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1182])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1182])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1182])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1182])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1182])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1182])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1182])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1151])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1182])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1151])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1182])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1182])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1151])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1182])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1151])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1151])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1151])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1151])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1151])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1151])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1151])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1182])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1151])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1151])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1151])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1151])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1151])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1183])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1183])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1183])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1183])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1183])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1183])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1183])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1183])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1183])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1183])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1183])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1183])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1152])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1183])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1152])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1183])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1183])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1152])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1183])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1152])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1152])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1152])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1152])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1152])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1152])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1152])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1183])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1152])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1152])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1152])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1152])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1152])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1184])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1184])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1184])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1184])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1184])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1184])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1184])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1184])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1184])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1184])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1184])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1184])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1153])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1184])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1153])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1184])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1184])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1153])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1184])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1153])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1153])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1153])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1153])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1153])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1153])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1153])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1184])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1153])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1153])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1153])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1153])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1153])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1185])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1185])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1185])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1185])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1185])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1185])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1185])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1185])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1185])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1185])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1185])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1185])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1154])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1185])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1154])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1185])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1185])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1154])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1185])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1154])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1154])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1154])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1154])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1154])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1154])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1154])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1185])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1154])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1154])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1154])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1154])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1154])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1186])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1186])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1186])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1186])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1186])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1186])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1186])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1186])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1186])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1186])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1186])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1186])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1155])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1186])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1155])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1186])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1186])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1155])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1186])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1155])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1155])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1155])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1155])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1155])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1155])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1155])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1186])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1155])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1155])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1155])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1155])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1155])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1187])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1187])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1187])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1187])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1187])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1187])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1187])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1187])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1187])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1187])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1187])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1187])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1156])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1187])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1156])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1187])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1187])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1156])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1187])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1156])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1156])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1156])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1156])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1156])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1156])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1156])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1187])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1156])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1156])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1156])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1156])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1156])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1188])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1188])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1188])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1188])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1188])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1188])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1188])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1188])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1188])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1188])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1188])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1188])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1157])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1188])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1157])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1188])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1188])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1157])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1188])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1157])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1157])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1157])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1157])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1157])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1157])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1157])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1188])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1157])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1157])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1157])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1157])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1157])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1189])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1189])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1189])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1189])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1189])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1189])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1189])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1189])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1189])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1189])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1189])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1189])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1158])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1189])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1158])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1189])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1189])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1158])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1189])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1158])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1158])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1158])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1158])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1158])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1158])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1158])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1189])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1158])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1158])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1158])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1158])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1158])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1190])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1190])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1190])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1190])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1190])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1190])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1190])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1190])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1190])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1190])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1190])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1190])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1159])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1190])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1159])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1190])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1190])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1159])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1190])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1159])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1159])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1159])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1159])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1159])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1159])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1159])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1190])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1159])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1159])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1159])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1159])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1159])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1191])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1191])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1191])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1191])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1191])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1191])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1191])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1191])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1191])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1191])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1191])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1191])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1160])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1191])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1160])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1191])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1191])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1160])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1191])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1160])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1160])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1160])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1160])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1160])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1160])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1160])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1191])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1160])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1160])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1160])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1160])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1160])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1192])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1192])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1192])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1192])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1192])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1192])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1192])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1192])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1192])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1192])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1192])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1192])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1161])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1192])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1161])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1192])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1192])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1161])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1192])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1161])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1161])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1161])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1161])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1161])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1161])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1161])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1192])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1161])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1161])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1161])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1161])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1161])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1193])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1193])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1193])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1193])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1193])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1193])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1193])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1193])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1193])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1193])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1193])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1193])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1162])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1193])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1162])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1193])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1193])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1162])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1193])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1162])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1162])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1162])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1162])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1162])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1162])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1162])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1193])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1162])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1162])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1162])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1162])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1162])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1194])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1194])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1194])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1194])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1194])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1194])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1194])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1194])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1194])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1194])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1194])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1194])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1163])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1194])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1163])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1194])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1194])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1163])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1194])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1163])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1163])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1163])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1163])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1163])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1163])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1163])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1194])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1163])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1163])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1163])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1163])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1163])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1195])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1195])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1195])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1195])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1195])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1195])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1195])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1195])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1195])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1195])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1195])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1195])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1164])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1195])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1164])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1195])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1195])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1164])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1195])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1164])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1164])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1164])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1164])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1164])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1164])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1164])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1195])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1164])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1164])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1164])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1164])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1164])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1196])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1196])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1196])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1196])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1196])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1196])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1196])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1196])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1196])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1196])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1196])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1196])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1165])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1196])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1165])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1196])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1196])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1165])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1196])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1165])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1165])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1165])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1165])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1165])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1165])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1165])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1196])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1165])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1165])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1165])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1165])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1165])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1197])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1197])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1197])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1197])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1197])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1197])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1197])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1197])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1197])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1197])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1197])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1197])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1166])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1197])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1166])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1197])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1197])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1166])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1197])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1166])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1166])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1166])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1166])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1166])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1166])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1166])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1197])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1166])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1166])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1166])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1166])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1166])
# [L0] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1198])
# [L1] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1198])
# [L2] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1198])
# [L3] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1198])
# [L4] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1198])
# [L5] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1198])
# [L6] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1198])
# [L7] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1198])
# [L8] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1198])
# [L9] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1198])
# [L10] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1198])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1198])
# [L12] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1198])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L15] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1198])
# [L16] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1198])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1198])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L20] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L21] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L22] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L23] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L25] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1198])
# [L27] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L28] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L29] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L30] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L31] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1167])
# 🚩 Add span to Pruner: (1170, 1198)
# [L0] 🧠 Buffer span (1170,1198) at shift=0 time=1199
# [AgeDecay L0] ✅ Drop span (1128,1159) age=39
# [L1] 🧠 Buffer span (1170,1198) at shift=0 time=1199
# [AgeDecay L1] ✅ Drop span (1128,1159) age=39
# [L2] 🧠 Buffer span (1170,1198) at shift=0 time=1199
# [AgeDecay L2] ✅ Drop span (1128,1159) age=39
# [L3] 🧠 Buffer span (1170,1198) at shift=0 time=1199
# [AgeDecay L3] ✅ Drop span (1128,1159) age=39
# [L4] 🧠 Buffer span (1170,1198) at shift=0 time=1199
# [AgeDecay L4] ✅ Drop span (1128,1159) age=39
# [L5] 🧠 Buffer span (1170,1198) at shift=0 time=1199
# [AgeDecay L5] ✅ Drop span (1128,1159) age=39
# [L6] 🧠 Buffer span (1170,1198) at shift=0 time=1199
# [AgeDecay L6] ✅ Drop span (1128,1159) age=39
# [L7] 🧠 Buffer span (1170,1198) at shift=0 time=1199
# [AgeDecay L7] ✅ Drop span (1128,1159) age=39
# [L8] 🧠 Buffer span (1170,1198) at shift=0 time=1199
# [AgeDecay L8] ✅ Drop span (1128,1159) age=39
# [L9] 🧠 Buffer span (1170,1198) at shift=0 time=1199
# [AgeDecay L9] ✅ Drop span (1128,1159) age=39
# [DepthDrop L10] ✅ Drop span (1170,1198) prob=0.32
# [DepthDrop L12] ✅ Drop span (1139,1167) prob=0.38
# [DepthDrop L15] ✅ Drop span (1170,1198) prob=0.46
# [DepthDrop L16] ✅ Drop span (1170,1198) prob=0.49
# [DepthDrop L20] ✅ Drop span (1139,1167) prob=0.60
# [DepthDrop L21] ✅ Drop span (1139,1167) prob=0.63
# [DepthDrop L22] ✅ Drop span (1139,1167) prob=0.65
# [DepthDrop L23] ✅ Drop span (1139,1167) prob=0.68
# [DepthDrop L25] ✅ Drop span (1139,1167) prob=0.74
# [DepthDrop L27] ✅ Drop span (1139,1167) prob=0.79
# [DepthDrop L28] ✅ Drop span (1139,1167) prob=0.82
# [DepthDrop L29] ✅ Drop span (1139,1167) prob=0.85
# [DepthDrop L30] ✅ Drop span (1139,1167) prob=0.87
# [DepthDrop L31] ✅ Drop span (1139,1167) prob=0.90
# [L0] 📌 Request drop span: (1128,1159) with shift=0 on KV size=1198
# [L0] ✅ Total 32 tokens to drop. 1166 will be kept.
# [L0] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1198, 128]), torch.Size([1, 8, 1198, 128])
# [L0] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1166, 128]), torch.Size([1, 8, 1166, 128])
# [L0] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L0] 🔧 Removed 32 tokens, shift before=0, after=32
# [L0] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L1] 📌 Request drop span: (1128,1159) with shift=0 on KV size=1198
# [L1] ✅ Total 32 tokens to drop. 1166 will be kept.
# [L1] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1198, 128]), torch.Size([1, 8, 1198, 128])
# [L1] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1166, 128]), torch.Size([1, 8, 1166, 128])
# [L1] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L1] 🔧 Removed 32 tokens, shift before=0, after=32
# [L1] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L2] 📌 Request drop span: (1128,1159) with shift=0 on KV size=1198
# [L2] ✅ Total 32 tokens to drop. 1166 will be kept.
# [L2] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1198, 128]), torch.Size([1, 8, 1198, 128])
# [L2] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1166, 128]), torch.Size([1, 8, 1166, 128])
# [L2] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L2] 🔧 Removed 32 tokens, shift before=0, after=32
# [L2] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L3] 📌 Request drop span: (1128,1159) with shift=0 on KV size=1198
# [L3] ✅ Total 32 tokens to drop. 1166 will be kept.
# [L3] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1198, 128]), torch.Size([1, 8, 1198, 128])
# [L3] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1166, 128]), torch.Size([1, 8, 1166, 128])
# [L3] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L3] 🔧 Removed 32 tokens, shift before=0, after=32
# [L3] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L4] 📌 Request drop span: (1128,1159) with shift=0 on KV size=1198
# [L4] ✅ Total 32 tokens to drop. 1166 will be kept.
# [L4] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1198, 128]), torch.Size([1, 8, 1198, 128])
# [L4] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1166, 128]), torch.Size([1, 8, 1166, 128])
# [L4] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L4] 🔧 Removed 32 tokens, shift before=0, after=32
# [L4] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L5] 📌 Request drop span: (1128,1159) with shift=0 on KV size=1198
# [L5] ✅ Total 32 tokens to drop. 1166 will be kept.
# [L5] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1198, 128]), torch.Size([1, 8, 1198, 128])
# [L5] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1166, 128]), torch.Size([1, 8, 1166, 128])
# [L5] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L5] 🔧 Removed 32 tokens, shift before=0, after=32
# [L5] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L6] 📌 Request drop span: (1128,1159) with shift=0 on KV size=1198
# [L6] ✅ Total 32 tokens to drop. 1166 will be kept.
# [L6] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1198, 128]), torch.Size([1, 8, 1198, 128])
# [L6] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1166, 128]), torch.Size([1, 8, 1166, 128])
# [L6] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L6] 🔧 Removed 32 tokens, shift before=0, after=32
# [L6] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L7] 📌 Request drop span: (1128,1159) with shift=0 on KV size=1198
# [L7] ✅ Total 32 tokens to drop. 1166 will be kept.
# [L7] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1198, 128]), torch.Size([1, 8, 1198, 128])
# [L7] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1166, 128]), torch.Size([1, 8, 1166, 128])
# [L7] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L7] 🔧 Removed 32 tokens, shift before=0, after=32
# [L7] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L8] 📌 Request drop span: (1128,1159) with shift=0 on KV size=1198
# [L8] ✅ Total 32 tokens to drop. 1166 will be kept.
# [L8] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1198, 128]), torch.Size([1, 8, 1198, 128])
# [L8] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1166, 128]), torch.Size([1, 8, 1166, 128])
# [L8] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L8] 🔧 Removed 32 tokens, shift before=0, after=32
# [L8] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L9] 📌 Request drop span: (1128,1159) with shift=0 on KV size=1198
# [L9] ✅ Total 32 tokens to drop. 1166 will be kept.
# [L9] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1198, 128]), torch.Size([1, 8, 1198, 128])
# [L9] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1166, 128]), torch.Size([1, 8, 1166, 128])
# [L9] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L9] 🔧 Removed 32 tokens, shift before=0, after=32
# [L9] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1167])
# [L10] 📌 Request drop span: (1170,1198) with shift=0 on KV size=1198
# [L10] ✅ Total 29 tokens to drop. 1170 will be kept.
# [L10] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1198, 128]), torch.Size([1, 8, 1198, 128])
# [L10] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1170, 128]), torch.Size([1, 8, 1170, 128])
# [L10] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L10] 🔧 Removed 28 tokens, shift before=0, after=28
# [L10] 🛠 post-attn kv-shift=28, attn_w.shape=torch.Size([1, 32, 1, 1171])
# [L11] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1199])
# [L12] 📌 Request drop span: (1139,1167) with shift=31 on KV size=1167
# [L12] ✅ Total 29 tokens to drop. 1139 will be kept.
# [L12] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1167, 128]), torch.Size([1, 8, 1167, 128])
# [L12] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1139, 128]), torch.Size([1, 8, 1139, 128])
# [L12] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L12] 🔧 Removed 28 tokens, shift before=31, after=59
# [L12] 🛠 post-attn kv-shift=59, attn_w.shape=torch.Size([1, 32, 1, 1140])
# [L13] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1199])
# [L14] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1168])
# [L15] 📌 Request drop span: (1170,1198) with shift=0 on KV size=1198
# [L15] ✅ Total 29 tokens to drop. 1170 will be kept.
# [L15] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1198, 128]), torch.Size([1, 8, 1198, 128])
# [L15] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1170, 128]), torch.Size([1, 8, 1170, 128])
# [L15] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L15] 🔧 Removed 28 tokens, shift before=0, after=28
# [L15] 🛠 post-attn kv-shift=28, attn_w.shape=torch.Size([1, 32, 1, 1171])
# [L16] 📌 Request drop span: (1170,1198) with shift=0 on KV size=1198
# [L16] ✅ Total 29 tokens to drop. 1170 will be kept.
# [L16] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1198, 128]), torch.Size([1, 8, 1198, 128])
# [L16] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1170, 128]), torch.Size([1, 8, 1170, 128])
# [L16] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L16] 🔧 Removed 28 tokens, shift before=0, after=28
# [L16] 🛠 post-attn kv-shift=28, attn_w.shape=torch.Size([1, 32, 1, 1171])
# [L17] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1168])
# [L18] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1199])
# [L19] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1168])
# [L20] 📌 Request drop span: (1139,1167) with shift=31 on KV size=1167
# [L20] ✅ Total 29 tokens to drop. 1139 will be kept.
# [L20] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1167, 128]), torch.Size([1, 8, 1167, 128])
# [L20] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1139, 128]), torch.Size([1, 8, 1139, 128])
# [L20] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L20] 🔧 Removed 28 tokens, shift before=31, after=59
# [L20] 🛠 post-attn kv-shift=59, attn_w.shape=torch.Size([1, 32, 1, 1140])
# [L21] 📌 Request drop span: (1139,1167) with shift=31 on KV size=1167
# [L21] ✅ Total 29 tokens to drop. 1139 will be kept.
# [L21] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1167, 128]), torch.Size([1, 8, 1167, 128])
# [L21] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1139, 128]), torch.Size([1, 8, 1139, 128])
# [L21] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L21] 🔧 Removed 28 tokens, shift before=31, after=59
# [L21] 🛠 post-attn kv-shift=59, attn_w.shape=torch.Size([1, 32, 1, 1140])
# [L22] 📌 Request drop span: (1139,1167) with shift=31 on KV size=1167
# [L22] ✅ Total 29 tokens to drop. 1139 will be kept.
# [L22] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1167, 128]), torch.Size([1, 8, 1167, 128])
# [L22] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1139, 128]), torch.Size([1, 8, 1139, 128])
# [L22] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L22] 🔧 Removed 28 tokens, shift before=31, after=59
# [L22] 🛠 post-attn kv-shift=59, attn_w.shape=torch.Size([1, 32, 1, 1140])
# [L23] 📌 Request drop span: (1139,1167) with shift=31 on KV size=1167
# [L23] ✅ Total 29 tokens to drop. 1139 will be kept.
# [L23] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1167, 128]), torch.Size([1, 8, 1167, 128])
# [L23] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1139, 128]), torch.Size([1, 8, 1139, 128])
# [L23] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L23] 🔧 Removed 28 tokens, shift before=31, after=59
# [L23] 🛠 post-attn kv-shift=59, attn_w.shape=torch.Size([1, 32, 1, 1140])
# [L24] 🛠 post-attn kv-shift=31, attn_w.shape=torch.Size([1, 32, 1, 1168])
# [L25] 📌 Request drop span: (1139,1167) with shift=31 on KV size=1167
# [L25] ✅ Total 29 tokens to drop. 1139 will be kept.
# [L25] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1167, 128]), torch.Size([1, 8, 1167, 128])
# [L25] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1139, 128]), torch.Size([1, 8, 1139, 128])
# [L25] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L25] 🔧 Removed 28 tokens, shift before=31, after=59
# [L25] 🛠 post-attn kv-shift=59, attn_w.shape=torch.Size([1, 32, 1, 1140])
# [L26] 🛠 post-attn kv-shift=0, attn_w.shape=torch.Size([1, 32, 1, 1199])
# [L27] 📌 Request drop span: (1139,1167) with shift=31 on KV size=1167
# [L27] ✅ Total 29 tokens to drop. 1139 will be kept.
# [L27] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1167, 128]), torch.Size([1, 8, 1167, 128])
# [L27] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1139, 128]), torch.Size([1, 8, 1139, 128])
# [L27] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L27] 🔧 Removed 28 tokens, shift before=31, after=59
# [L27] 🛠 post-attn kv-shift=59, attn_w.shape=torch.Size([1, 32, 1, 1140])
# [L28] 📌 Request drop span: (1139,1167) with shift=31 on KV size=1167
# [L28] ✅ Total 29 tokens to drop. 1139 will be kept.
# [L28] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1167, 128]), torch.Size([1, 8, 1167, 128])
# [L28] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1139, 128]), torch.Size([1, 8, 1139, 128])
# [L28] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L28] 🔧 Removed 28 tokens, shift before=31, after=59
# [L28] 🛠 post-attn kv-shift=59, attn_w.shape=torch.Size([1, 32, 1, 1140])
# [L29] 📌 Request drop span: (1139,1167) with shift=31 on KV size=1167
# [L29] ✅ Total 29 tokens to drop. 1139 will be kept.
# [L29] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1167, 128]), torch.Size([1, 8, 1167, 128])
# [L29] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1139, 128]), torch.Size([1, 8, 1139, 128])
# [L29] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L29] 🔧 Removed 28 tokens, shift before=31, after=59
# [L29] 🛠 post-attn kv-shift=59, attn_w.shape=torch.Size([1, 32, 1, 1140])
# [L30] 📌 Request drop span: (1139,1167) with shift=31 on KV size=1167
# [L30] ✅ Total 29 tokens to drop. 1139 will be kept.
# [L30] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1167, 128]), torch.Size([1, 8, 1167, 128])
# [L30] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1139, 128]), torch.Size([1, 8, 1139, 128])
# [L30] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L30] 🔧 Removed 28 tokens, shift before=31, after=59
# [L30] 🛠 post-attn kv-shift=59, attn_w.shape=torch.Size([1, 32, 1, 1140])
# [L31] 📌 Request drop span: (1139,1167) with shift=31 on KV size=1167
# [L31] ✅ Total 29 tokens to drop. 1139 will be kept.
# [L31] 🔍 compress_kv: original KV shape = torch.Size([1, 8, 1167, 128]), torch.Size([1, 8, 1167, 128])
# [L31] ✅ compress_kv: new KV shape = torch.Size([1, 8, 1139, 128]), torch.Size([1, 8, 1139, 128])
# [L31] 🧠 Using DynamicCache → updating key_cache/value_cache directly
# [L31] 🔧 Removed 28 tokens, shift before=31, after=59
# [L31] 🛠 post-attn kv-shift=59, attn_w.shape=torch.Size([1, 32, 1, 1140])
# [L0] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1168])
# [L1] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1168])
# [L2] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1168])
# [L3] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1168])
# [L4] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1168])
# [L5] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1168])
# [L6] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1168])
# [L7] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1168])
# [L8] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1168])
# [L9] 🛠 post-attn kv-shift=32, attn_w.shape=torch.Size([1, 32, 1, 1168])
# Traceback (most recent call last):
#   File "/home/da530038/lang-pro/Open-Source/nous_mistral_robus_play.py", line 102, in <module>
#     out = model(
#           ^^^^^^
#   File "/home/da530038/lang-pro/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/da530038/lang-pro/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/da530038/lang-pro/.venv/lib/python3.12/site-packages/transformers/utils/generic.py", line 961, in wrapper
#     output = func(self, *args, **kwargs)
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/da530038/lang-pro/.venv/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py", line 434, in forward
#     outputs: BaseModelOutputWithPast = self.model(
#                                        ^^^^^^^^^^^
#   File "/home/da530038/lang-pro/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/da530038/lang-pro/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/da530038/lang-pro/.venv/lib/python3.12/site-packages/transformers/utils/generic.py", line 1069, in wrapper
#     outputs = func(self, *args, **kwargs)
#               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/da530038/lang-pro/.venv/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py", line 364, in forward
#     hidden_states = decoder_layer(
#                     ^^^^^^^^^^^^^^
#   File "/home/da530038/lang-pro/.venv/lib/python3.12/site-packages/transformers/modeling_layers.py", line 94, in __call__
#     return super().__call__(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/da530038/lang-pro/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/da530038/lang-pro/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/da530038/lang-pro/.venv/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py", line 228, in forward
#     hidden_states, _ = self.self_attn(
#                        ^^^^^^^^^^^^^^^
#   File "/home/da530038/lang-pro/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/da530038/lang-pro/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/da530038/lang-pro/Open-Source/robust_prune_attention.py", line 471, in forward
#     ctx, attn_w = self.attn(hidden_states, **kwargs)
#                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/da530038/lang-pro/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/da530038/lang-pro/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/da530038/lang-pro/.venv/lib/python3.12/site-packages/transformers/utils/generic.py", line 1037, in wrapped_forward
#     output = orig_forward(*args, **kwargs)
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/da530038/lang-pro/.venv/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py", line 167, in forward
#     attn_output, attn_weights = attention_interface(
#                                 ^^^^^^^^^^^^^^^^^^^^
#   File "/home/da530038/lang-pro/.venv/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py", line 112, in eager_attention_forward
#     attn_weights = attn_weights + causal_mask
#                    ~~~~~~~~~~~~~^~~~~~~~~~~~~
# RuntimeError: The size of tensor a (1172) must match the size of tensor b (1168) at non-singleton dimension 3
