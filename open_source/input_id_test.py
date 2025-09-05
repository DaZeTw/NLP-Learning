import ast
import csv
import difflib
import itertools
import json
import os
import random
import re
import statistics
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nltk
import torch
import torch.nn
import torch.nn as nn
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mistral.modeling_mistral import repeat_kv


def customize_model(
    model_path: str, device: str = "auto", dtype=torch.float16
) -> Tuple[nn.Module, AutoTokenizer]:

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        device_map=device,
        torch_dtype=dtype,
        local_files_only=True,
        trust_remote_code=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )

    return model, tokenizer


def ends_with_pattern(
    seq: List[int], pattern: List[int], gen_start: int
) -> Optional[int]:
    """
    Check if the sequence ends with the given pattern,
    and make sure the match is in the generated region (not in prompt).

    Returns:
        Length of pattern if match found after gen_start, else None.
    """
    if len(seq) < len(pattern) + gen_start:
        return None
    if seq[-len(pattern) :] == pattern and (len(seq) - len(pattern)) >= gen_start:
        return len(pattern)
    return None


def run_inference(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    full_prompt: str,
    max_new_tokens: int = 600,
    scratchpad_token: str = "[SCRATCHPAD]",
    return_token: str = "[RETURN]",
    final_answer_token: str = "[Final Answer]",
    additional_guidance: str = "This is an intermediate result and yous must to use this result to continue the reasoning to get the final answer. Don't try to recalculate this result, just use it to continue the reasoning. Facts you must to remember, don't recalculate this: ",
    enable_pruning: bool = True,
    verbose: bool = False,
) -> Tuple[str, str]:

    sp_patterns = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(scratchpad_token))
    ret_patterns = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(return_token))
    fn_patterns = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(final_answer_token)
    )
    additional_guidance_patterns = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(additional_guidance)
    )

    sp_patterns[0] = 28792
    ret_patterns[0] = 28792
    fn_patterns[0] = 28792
    print(f"Scratchpad patterns: {sp_patterns}")
    print(f"Return patterns: {ret_patterns}")
    print(f"Final Answer patterns: {fn_patterns}")
    # Encode prompt and mark the boundary so we NEVER prune inside the prompt
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(
        model.device
    )  # (1, T)

    additional_guidance_patterns = tokenizer(
        additional_guidance, return_tensors="pt"
    ).input_ids.to(input_ids.device)
    gen_start = input_ids.size(1)
    raw_ids = input_ids.tolist()[0].copy()

    last_sp_start = None
    ret_start = None
    have_return_after_last_sp = False
    shift = 0
    right_post = True
    first_prune = True
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids=input_ids, use_cache=False).logits
            next_id = logits[:, -1, :].argmax(dim=-1)  # greedy
            token = next_id.item()

            raw_ids.append(token)
            input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)
            token_list = input_ids[0].tolist()

            if enable_pruning:
                # Print token ID
                print(token_list[-1], end=", ", flush=True)

                # Detect [RETURN]
                ret_len = ends_with_pattern(token_list, ret_patterns, gen_start)
                if ret_len:
                    have_return_after_last_sp = True
                    ret_start = len(token_list) - ret_len
                    if verbose:
                        print(f"\n🟥 Found [RETURN] at index {ret_start}")

                # Detect [SCRATCHPAD]
                sp_len = ends_with_pattern(token_list, sp_patterns, gen_start)
                if sp_len:
                    sp_start = len(token_list) - sp_len
                    right_post = ~right_post
                    if verbose:
                        print(f"\n🟩 Found [SCRATCHPAD] at index {sp_start}")

                    if (
                        last_sp_start is not None
                        and have_return_after_last_sp
                        and right_post
                    ):
                        if first_prune:
                            input_ids = torch.cat(
                                [
                                    input_ids[:, :last_sp_start],
                                    additional_guidance_patterns,
                                    input_ids[:, ret_start + len(ret_patterns) :],
                                ],
                                dim=1,
                            )
                            shift = (
                                ret_start
                                + len(ret_patterns)
                                - additional_guidance_patterns.size(1)
                                - last_sp_start
                            )
                            first_prune = False
                            if verbose:
                                print(
                                    f"🔁 Pruned tokens from {last_sp_start} to {ret_start}"
                                )
                                print(
                                    f"🆗 Insert additional guidance tokens at {last_sp_start}"
                                )
                        else:
                            input_ids = torch.cat(
                                [
                                    input_ids[:, :last_sp_start],
                                    input_ids[:, ret_start + len(ret_patterns) :],
                                ],
                                dim=1,
                            )
                            shift = ret_start + len(ret_patterns) - last_sp_start
                            if verbose:
                                print(
                                    f"🔁 Pruned tokens from {last_sp_start} to {ret_start}"
                                )

                        have_return_after_last_sp = False

                    # Always update this
                    last_sp_start = sp_start - shift

                fn_len = ends_with_pattern(token_list, fn_patterns, gen_start)
                if fn_len and last_sp_start is not None and have_return_after_last_sp:
                    print("\n🟦 Found [Final Answer], pruning and stopping generation.")
                    input_ids = torch.cat(
                        [
                            input_ids[:, :last_sp_start],
                            input_ids[:, ret_start + len(ret_patterns) :],
                        ],
                        dim=1,
                    )

            # Stop on EOS
            if token == tokenizer.eos_token_id:
                break

    origin_id = raw_ids[gen_start:]  # generated tokens only
    origin_text = tokenizer.decode(origin_id, skip_special_tokens=True)

    output_ids = input_ids[0, gen_start:].tolist()  # generated tokens only
    final_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    if verbose:
        print("\n──────── Generation Summary ────────")
        print(f"• Tokens generated (raw) : {len(raw_ids)}")
        if enable_pruning:
            print(f"• Tokens after pruning   : {input_ids.size(1)}")
            print(f"• Tokens in output_ids: {output_ids}")

    torch.cuda.empty_cache()
    return origin_text, final_text, len(origin_id), len(output_ids)


prompt_template = """You are a helpful, accurate assistant for solving **math word problems**.
---

### INSTRUCTIONS

1. **Scratchpad Stage**:
   • Begin each step with **[SCRATCHPAD]** on its own line.  
   • Write detailed, natural-language reasoning.  
   • End the step with **[RETURN]** on its own line.  
   • Immediately after **[RETURN]**, write  
     <value and what it represents of scratchpad step>  
     on the same line or the next line.

   Example of one step:

   [SCRATCHPAD]  
   I know each box holds 6 apples. There are 5 boxes, so 5 × 6 = 30 apples.  
   [RETURN]  
   There are 30 apples in total.

2. **Final Answer Stage**:
   After all scratchpad steps, write:  
   [Final Answer] <numeric answer only>

---
### FORMAT EXAMPLE

[SCRATCHPAD]  
The problem says Natalia sold 48 clips in April.  
In May, she sold **half as many clips** as she did in April.  
To find the number of clips she sold in May, I divide the April amount by 2:  
48 ÷ 2 = 24  
So, she sold 24 clips in May.  
[RETURN]  
Natalia sold 24 clips in May.

[SCRATCHPAD]  
Now I want to find the total number of clips Natalia sold in April and May combined.  
She sold 48 clips in April and 24 clips in May.  
So I add the two amounts together:  
48 + 24 = 72  
This gives the total number of clips she sold in both months.  
[RETURN]  
Natalia sold 72 clips in total over April and May.

[Final Answer] 
72

### PROBLEM

{question}
"""

prompt_template = """You are a helpful, accurate assistant for solving **math word problems**.

For **each problem**, follow *exactly* this two-stage format that uses the special
markers **[SCRATCHPAD]** and **[RETURN]** to indicate your reasoning steps and intermediate results:

---

### INSTRUCTIONS

Try to solve the problem step by step and break down the reasoning into smaller parts. 

1. **Scratchpad Stage** (repeat as many times as needed):
   • Begin each step with **[SCRATCHPAD]** on its own line.  
   • Write detailed, natural-language reasoning.  
   • End the step with **[RETURN]** on its own line.  
   • Immediately after **[RETURN]**, write  
     Intermediate Result: <value and what it represents>  
     on the same line or the next line.

   Example of one step:

   [SCRATCHPAD]  
   I know each box holds 6 apples. There are 5 boxes, so 5 × 6 = 30 apples.  
   [RETURN]  
   Intermediate Result: There are 30 apples in total.

2. **Final Answer Stage**:
   After all scratchpad steps, write:  
   [Final Answer]
   <numeric answer only>

---

### FORMAT EXAMPLE

[SCRATCHPAD]  
In May, she sold half as many clips as in April, so 48 ÷ 2 = 24.  
[RETURN]  
Intermediate Result: Natalia sold 24 clips in May.

[SCRATCHPAD]  
In total, she sold 48 + 24 clips in April and May.  
[RETURN]  
Intermediate Result: Natalia sold 72 clips in total over April and May.

[Final Answer] 
72

---

### PROBLEM

{question}

"""


prompt_template = """You are a helpful, accurate assistant for solving **math word problems**.

For **each problem**, follow *exactly* this two-stage format that uses the special
markers **[SCRATCHPAD]** and **[RETURN]** to indicate your reasoning steps and intermediate results. 

### INSTRUCTIONS

Try to solve the problem step by step and break down the reasoning into smaller parts and each part should have the pattern below.

1. **Scratchpad Stage** (repeat as many times as needed):
   • Begin each step with **[SCRATCHPAD]** on its own line.  
   • Write detailed, natural-language reasoning.  
   • End the step with **[RETURN]** on its own line.  
   • Immediately after **[RETURN]**, write  
     <value and what it represents>  
     on the same line or the next line.

   Example of one step:

   [SCRATCHPAD]  
   I know each box holds 6 apples. There are 5 boxes, so 5 × 6 = 30 apples.  
   [RETURN]  
   There are 30 apples in total.

2. **Final Answer Stage**:
   After all scratchpad steps, write:  
   [Final Answer]
   <numeric answer only>

Remember the structure above by giving a [SCRATCHPAD] before each reasoning step and a [RETURN] after each reasoning step. Not create multiple [SCRATCHPAD]s before a [RETURN] or multiple [RETURN]s after a [SCRATCHPAD]. Remember to go with **[SCRATCHPAD]** directly after the question without any other text.

### FORMAT EXAMPLE

[SCRATCHPAD]  
In May, she sold half as many clips as in April, so 48 ÷ 2 = 24.  
[RETURN]  
Natalia sold 24 clips in May.

[SCRATCHPAD]  
In total, she sold 48 + 24 clips in April and May.  
[RETURN]  
Natalia sold 72 clips in total over April and May.

[Final Answer] 
72

### PROBLEM

{question}
"""
# Remember to go with scratchpad directly after the question without any other text.

question = """Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?"""
# question = """Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?"""
prompt = prompt_template.format(question=question)

local_path = "/home/da530038/lang-pro/open_source/mistral_model"

model, tokenizer = customize_model(local_path)

# noprun_out, raw_out = run_inference(
#     model, tokenizer, prompt, enable_pruning=False)

# print("\n\n──────── No Pruning ────────")
# print("Raw output:", raw_out)
# print("No pruning output:", noprun_out)

origin_text, final_text, origin_len, final_id = run_inference(
    model, tokenizer, prompt, enable_pruning=True, verbose=True
)

print("\n\n──────── Original Output ────────\n")
print(origin_text)

print("\n\n──────── With Pruning ────────\n")
print(final_text)

print(f"\n• Tokens generated (raw) : {origin_len}")
print(f"• Tokens after pruning   : {final_id}")
