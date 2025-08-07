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

    def evaluate_and_mark(self, ledger: List[dict], gen_step: int = None, age_threshold: int = 20):
        current_time = gen_step if gen_step is not None else 0
        
        if self.layer_idx < 10:
            # Age decay logic for early layers
            # 1. Add new spans to buffer with close timestamps
            for rec in ledger:
                span_with_time = {
                    "start": rec["start"],
                    "end": rec["end"],
                    "close_time": current_time
                }
                self.span_buffer.append(span_with_time)
                if self.debug:
                    print(f"[L{self.layer_idx}] Added span ({rec['start']},{rec['end']}) to buffer at time {current_time}")
            
            # 2. Check ALL spans in buffer for age-based pruning
            spans_to_prune = []
            for span in self.span_buffer:
                age = current_time - span["close_time"]
                if age >= age_threshold:
                    rel_start = max(0, span["start"] - self.shift)
                    rel_end = span["end"] - self.shift
                    if rel_end >= 0:  # Still within current KV cache
                        self.to_drop.append((rel_start, rel_end))
                        spans_to_prune.append(span)
                        if self.debug:
                            print(f"[AgeDecay L{self.layer_idx}] dropping span ({rel_start},{rel_end}) age={age}")
            
            # 3. Clean up buffer: remove spans that are pruned or shifted out
            self.span_buffer = [s for s in self.span_buffer 
                              if s not in spans_to_prune and s["end"] >= self.shift]
            
        else:
            # Depth ratio drop logic for later layers (immediate decision)
            depth_ratio = self.layer_idx / (self.total_layers - 1)
            drop_prob = self.cfg["prob_min"] + depth_ratio * (self.cfg["prob_max"] - self.cfg["prob_min"])
            
            for rec in ledger:
                if random.random() < drop_prob:
                    rel_start = max(0, rec["start"] - self.shift)
                    rel_end = rec["end"] - self.shift
                    if rel_end >= 0:
                        self.to_drop.append((rel_start, rel_end))
                        if self.debug:
                            print(f"[DepthDrop L{self.layer_idx}] drop span ({rel_start},{rel_end}) prob={drop_prob:.2f}")

    def _compress_kv(self, pkv, keep_idx):
        k, v = unpack_kv(pkv, self.layer_idx)
        k_new = k[:, :, keep_idx, :].contiguous()
        v_new = v[:, :, keep_idx, :].contiguous()
        hard = isinstance(pkv, DynamicCache)
        if hard:
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
            drop_indices.update(range(start, end + 1))
        keep_indices = sorted(set(range(N)) - drop_indices)
        if not keep_indices:
            self.to_drop.clear()
            return pkv, None, False, None
        pkv, v_new, hard = self._compress_kv(pkv, torch.tensor(keep_indices, device=k.device))
        removed_now = N - len(keep_indices)
        self.shift += removed_now
        self.to_drop.clear()
        return pkv, v_new, hard, torch.tensor(keep_indices, device=k.device)

    def forward(self, hidden_states, **kwargs):
        kwargs["output_attentions"] = True
        kwargs["use_cache"] = True
        ctx, attn_w = self.attn(hidden_states, **kwargs)
        pkv = kwargs.get("past_key_value", None)
        pkv, v_new, _, keep_idx = self._maybe_prune(pkv)
        if keep_idx is None or attn_w is None:
            return ctx, attn_w
        attn_w = attn_w[..., keep_idx]
        attn_w = attn_w / (attn_w.sum(-1, keepdim=True) + 1e-6)
        k_use, _ = unpack_kv(pkv, self.layer_idx)
        k_use = k_use[..., keep_idx, :]
        v_all = repeat_kv(v_new if v_new is not None else unpack_kv(
            pkv, self.layer_idx)[1], self.attn.num_key_value_groups)
        new_ctx = torch.matmul(attn_w, v_all)
        B, H, Q, D = new_ctx.shape
        new_ctx = new_ctx.permute(0, 2, 1, 3).contiguous().view(B, Q, H * D)
        out = self.attn.o_proj(new_ctx)
        return out, attn_w

# ============================================================
# ScratchpadTracker (unchanged except no tail_len logic)
# ============================================================


class ScratchpadTracker:
    OPEN_RE = re.compile(r"^Scratchpad\s*:", re.I)
    CLOSE_RE = re.compile(r"^Intermediate\s*Result\s*:", re.I)
    FINAL_RE = re.compile(r"^Final\s*Answer\s*:", re.I)

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
        is_close = bool(self.CLOSE_RE.search(tail) or self.FINAL_RE.search(tail))

        if is_open and not self.in_pad:
            self.in_pad = True
            self.cur_pad = [pos]
        elif self.in_pad:
            self.cur_pad.append(pos)
            if is_close:
                # send span to every layer wrapper, with gen_step for age decay
                for mod in self.mods:
                    if hasattr(mod, "evaluate_and_mark"):
                        mod.evaluate_and_mark([{"start": self.cur_pad[0], "end": self.cur_pad[-1]}], gen_step=len(self.ids), age_threshold=20)
                self.in_pad = False
                self.cur_pad = []

        # Always return True (all tokens are kept for output purposes)
        return True
