import ast
import csv
import difflib
import itertools
import json
import random
import re
import statistics
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union

import nltk
import torch
import torch.nn
import torch.nn as nn
from customize_attention import (
    PrunedAttention,
    ScratchpadPrunedAttention,
    ScratchpadTracker,
    SingleLayerScratchpadPruner,
    TopKPrunedAttention,
)
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from transformers import LlamaTokenizer, MistralForCausalLM


def customize_model(model_path: str,
                    model_type: str = "pruned",
                    last_k_layers: int = 10,
                    device: str = "auto",
                    dtype=torch.float16) -> Tuple[nn.Module, LlamaTokenizer, Union[ScratchpadTracker, None]]:
    """
    Load and customize the Mistral model with optional pruning.

    Parameters:
    - model_path: Local path to the model weights.
    - model_type: "baseline" or "pruned".
    - last_k_layers: How many final layers to prune (only if model_type == "pruned").
    - device: Device map to use ("auto" recommended).
    - dtype: Torch dtype (e.g., torch.float16).

    Returns:
    - model: The HuggingFace model object (potentially pruned).
    - tokenizer: The associated tokenizer.
    - tracker: A ScratchpadTracker if pruned; None otherwise.
    """

    model = MistralForCausalLM.from_pretrained(model_path,
                                               torch_dtype=torch.float16,
                                               device_map="auto",
                                               attn_implementation="eager")

    tokenizer = LlamaTokenizer.from_pretrained(
        model_path, trust_remote_code=True)

    if model_type == "baseline":
        return model, tokenizer, None

    elif model_type == "pruned":
        wrappers = []
        total_layers = len(model.model.layers)
        start_layer = max(0, total_layers - last_k_layers)

        for i in range(start_layer, total_layers):
            blk = model.model.layers[i]
            wrapper = SingleLayerScratchpadPruner(
                blk.self_attn,
                layer_idx=i,
                debug=True
            )
            blk.self_attn = wrapper
            wrappers.append(wrapper)

        tracker = ScratchpadTracker(tokenizer, wrappers)
        return model, tokenizer, tracker

    else:
        raise ValueError(
            f"Invalid model_type: {model_type}. Choose 'baseline' or 'pruned'.")


def run_inference(model: nn.Module,
                  tokenizer: LlamaTokenizer,
                  full_prompt: str,
                  tracker: Union[ScratchpadTracker, None] = None,
                  max_new_tokens: int = 3000,
                  verbose: bool = False) -> Tuple[str, str, Tuple]:
    """
    Run autoregressive inference on a full prompt and return:
    - the pruned output (generated_text)
    - the full generated output (origin_text)
    - the final KV cache after generation

    Parameters:
    - model: The language model (with or without pruning)
    - tokenizer: Corresponding tokenizer
    - full_prompt: Text prompt to feed the model
    - tracker: Optional ScratchpadTracker to track dropped tokens
    - max_new_tokens: Max number of tokens to generate
    - verbose: If True, prints token statistics and KV sizes

    Returns:
    - generated_text: Final decoded text with pruning (if tracker is used)
    - origin_text: Raw output including scratchpad
    - kv_cache: The final key/value cache (tuple of tuples)
    """

    input_ids = tokenizer(
        full_prompt, return_tensors="pt").input_ids.to(model.device)

    if tracker:
        tracker.initialize_with_prompt(input_ids[0].tolist())

    kv_cache = None
    all_ids = []
    kept_ids = []

    with torch.no_grad():
        # First step: prompt input
        out = model(input_ids=input_ids, use_cache=True,
                    output_attentions=False)
        kv_cache = out.past_key_values
        next_id = out.logits[:, -1].argmax(-1)
        all_ids.append(next_id.item())

        if tracker and tracker.step(next_id.item()):
            kept_ids.append(next_id.item())

        input_ids = next_id.unsqueeze(0)

        # Loop through generations
        for step in range(max_new_tokens):
            out = model(input_ids=input_ids,
                        past_key_values=kv_cache, use_cache=True)
            logits, kv_cache = out.logits, out.past_key_values
            next_id = logits[:, -1].argmax(-1)

            all_ids.append(next_id.item())
            if tracker and tracker.step(next_id.item()):
                kept_ids.append(next_id.item())

            if next_id.item() == tokenizer.eos_token_id:
                break

            input_ids = next_id.unsqueeze(0)

    # Decode outputs
    origin_text = tokenizer.decode(all_ids, skip_special_tokens=True)
    if tracker:
        generated_text = tokenizer.decode(kept_ids, skip_special_tokens=True)
    else:
        generated_text = origin_text

    if verbose:
        print("\n──────── Generation Summary ────────")
        print(f"• Tokens generated     : {len(all_ids)}")
        print(
            f"• Tokens in final text : {len(kept_ids) if tracker else len(all_ids)}")
        if tracker:
            print(f"• Output compression   : {len(kept_ids)/len(all_ids):.2%}")

        # Inspect KV cache token lengths per layer
        kv_lengths = [layer[0].shape[-2] for layer in kv_cache]
        print("\n──────── KV Cache Lengths Per Layer ────────")
        for i, l in enumerate(kv_lengths):
            print(f"• Layer {i:2}: {l} tokens")

    return generated_text, origin_text, kv_cache


def extract_inference_outputs(
    full_text: str,
    kv_cache
) -> Tuple[str, List[str], float]:
    """
    Extracts the final answer, scratchpad lines, and compression ratio from the output.
    If kv_cache is None or invalid, returns compression_ratio = -1.

    Parameters:
    - full_text: The complete output text from which the information needs to be extracted.
    - kv_cache: The key-value cache containing model layers' tensor shapes for calculating the compression ratio.

    Returns:
    - A tuple containing:
        1. final_answer (str): The final answer extracted from the text.
        2. scratchpad_lines (List[str]): A list of lines containing "scratchpad" or "intermediate result".
        3. compression_ratio (float): The compression ratio of the key-value cache. Returns -1 if kv_cache is invalid.
    """

    # 1. Extract the final answer from the text (looks for the pattern "Final Answer: <text>")
    match = re.search(r"Final\s*Answer\s*:\s*(.*)", full_text, re.I)
    # Use regex to extract the final answer
    final_answer = match.group(1).strip() if match else ""

    # 2. Extract scratchpad-related lines
    scratchpad_lines = []
    current_block = []
    in_block = False

    for line in full_text.splitlines():
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        if line.lower().startswith("final answer:"):
            if current_block:  # Save previous block if exists
                scratchpad_lines.append("\n".join(current_block))
            current_block = []  # Reset block, ignore Final Answer content
            in_block = False
            continue
        if line.lower().startswith("scratchpad:") or line.lower().startswith("intermediate result:"):
            if current_block:  # Save previous block if exists
                scratchpad_lines.append("\n".join(current_block))
            current_block = [line]  # Start new block
            in_block = True
        elif in_block:
            current_block.append(line)  # Add content to current block

    # Append the last block if exists
    if current_block:
        scratchpad_lines.append("\n".join(current_block))

    # 3. Calculate compression ratio based on kv_cache:
    # Compression ratio is calculated as 1 - (min length / max length) if kv_cache is valid

    # Extract KV lengths
    kv_lengths = []
    for i, layer in enumerate(kv_cache):
        try:
            length = layer[0].shape[-2]
            if length > 0:
                kv_lengths.append(length)
            print(f"Layer {i}: {length} tokens")
        except (AttributeError, IndexError) as e:
            print(f"Error accessing shape for layer {i}: {e}")
            return final_answer, scratchpad_lines, -1

    if not kv_lengths:
        print("Error: No valid KV lengths found")
        return final_answer, scratchpad_lines, -1

    kv_min = min(kv_lengths)
    kv_max = max(kv_lengths)
    print(f"KV lengths: min={kv_min}, max={kv_max}")

    # Calculate compression ratio
    compression_ratio = 1 - (kv_min / kv_max) if kv_max > 0 else 0.0
    print(
        f"Compression ratio calculated: {compression_ratio:.4f} ({compression_ratio*100:.2f}%)")

    return final_answer, scratchpad_lines, compression_ratio


prompt_template = """You are a highly accurate and logical assistant that solves math word problems using fully **explicit**, **step-by-step reasoning**.

Your solution must follow a strict structure with **no skipped steps**, **no combined logic**, and **no assumptions**. Every operation, transformation, and interpretation must be **fully spelled out**.

---

**Stage 1: Explicit Reasoning Steps**

You must alternate between the following two blocks:

1. `Scratchpad:`
   - Perform **one and only one small step** of reasoning.
   - Start from understanding the problem, identifying knowns/unknowns, parsing units, or restating parts of the question.
   - For math steps, **only do one operation per step** (e.g., only a subtraction, only converting a phrase into math, only combining two values).
   - Always explain **what you're doing** and **why**, using natural language and logic.
   - Avoid shortcuts, summarization, or grouping steps together.

2. `Intermediate Result:`
   - State the **numerical result or symbolic outcome** of the current step only.
   - Include a **short, clear explanation** of what this result means in context.
   - Do not mention any previous intermediate values — only focus on this current step.

Repeat this `Scratchpad:` → `Intermediate Result:` loop for **every tiny piece of logic**, until the final answer can be obtained.

---

**Stage 2: Final Answer**

After all reasoning steps are complete:

- Write `Final Answer:` followed by **only the final number** (no explanation, no units, no punctuation).
- Ensure this number is traceable from the previous `Intermediate Result:` steps.

---

**Strict Guidelines:**

- **Never combine steps**. If two actions must happen, split them into two `Scratchpad:` → `Intermediate Result:` pairs.
- **Never skip conversions**, derivations, or interpretations. Everything must be shown.
- **Never summarize or reference prior steps**. Each step must stand alone with its own logic and outcome.
- Keep formatting **exactly**: alternate `Scratchpad:` → `Intermediate Result:` → … → `Final Answer:`.

---

Now, solve the following math word problem step by step, strictly following the format and rules above:

{question}
"""


question = "A bar of steel weighs twice the mass of a bar of tin. If a steel bar also weighs 20 kgs more than a copper bar and a copper bar weighs 90 kgs, calculate the total weight of a container with 20 bars of each type of metal."
local_path = "/home/da530038/lang-pro/Open-Source/nous_hermes_2_mistral"
model, tokenizer, tracker = customize_model(
    local_path, model_type="pruned", last_k_layers=8)
full_prompt = prompt_template.format(question=question)

pruned_text, full_output, kv_cache = run_inference(
    model, tokenizer, full_prompt, tracker=tracker, verbose=True)

print("\n--- PRUNED TEXT ---\n", pruned_text)
print("\n--- FULL OUTPUT ---\n", full_output)
final_answer, scratchpad, compression = extract_inference_outputs(
    full_output, kv_cache)

print("📌 Final Answer:", final_answer)
print("🧠 Scratchpad Reasoning:")
print("\n".join(scratchpad))
print(f"📉 Compression Ratio (1 - min/max): {compression:.2%}")


def evaluate_gsm8k_examples(
    data_path: str,
    output_path: str,
    model_path: str,
    prompt_template: str,
    last_k_layers: int = 6,
    max_examples: int = 100
):
    """
    Run GSM8K-style reasoning on both baseline and pruned models, saving reasoning traces,
    answers, compression ratios, and inference times.

    Parameters:
    - data_path: Path to input JSON file (with "question" and "final_answer" fields)
    - output_path: Path to save the output JSON results
    - model_path: HuggingFace-compatible model path
    - prompt_template: Template with {question} to format prompt
    - last_k_layers: Number of layers to prune in the pruned model
    - max_examples: Number of examples to evaluate
    """

    # Load data
    with open(data_path, "r") as f:
        data = json.load(f)

    examples = random.sample(data, min(max_examples, len(data)))
    results = []

    total_baseline_time = 0.0
    total_pruned_time = 0.0

    for idx, ex in enumerate(tqdm(examples, desc="Evaluating")):
        # Load models
        baseline_model, tokenizer, _ = customize_model(
            model_path, model_type="baseline")
        pruned_model, _, tracker = customize_model(
            model_path, model_type="pruned", last_k_layers=last_k_layers)

        question = ex["question"]
        answer = ex["final_answer"]
        prompt = prompt_template.format(question=question)

        # Baseline inference
        start = time.time()
        base_out, base_full, _ = run_inference(
            baseline_model, tokenizer, prompt, tracker=None, verbose=False)
        end = time.time()
        base_time = end - start
        total_baseline_time += base_time

        base_answer, base_scratchpad, _ = extract_inference_outputs(
            base_full, kv_cache=[(None, None)])

        # Pruned inference
        start = time.time()
        pruned_out, pruned_full, pruned_kv = run_inference(
            pruned_model, tokenizer, prompt, tracker=tracker, verbose=False)
        end = time.time()
        pruned_time = end - start
        total_pruned_time += pruned_time

        pruned_answer, pruned_scratchpad, compression = extract_inference_outputs(
            pruned_full, pruned_kv)

        results.append({
            "id": idx,
            "question": question,
            "final_answer": answer,
            "baseline": {
                "final_answer": base_answer,
                "scratchpad": base_scratchpad,
                "output": base_out,
                "inference_time_sec": round(base_time, 3)
            },
            "pruned": {
                "final_answer": pruned_answer,
                "scratchpad": pruned_scratchpad,
                "compression_ratio": compression,
                "output": pruned_out,
                "inference_time_sec": round(pruned_time, 3)
            }
        })

        del baseline_model
        del pruned_model
        torch.cuda.empty_cache()

    print(f"Total Baseline Inference Time: {total_baseline_time:.3f} seconds")
    print(f"Total Pruned Inference Time: {total_pruned_time:.3f} seconds")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved results to {output_path}")


evaluate_gsm8k_examples(
    data_path="/home/da530038/lang-pro/Benchmarks/gsm8k_socratic.json",
    output_path="/home/da530038/lang-pro/Benchmarks/output/nous_mistral_gsm8k_eval_results.json",
    model_path="/home/da530038/lang-pro/Open-Source/nous_hermes_2_mistral",
    prompt_template=prompt_template,
    last_k_layers=6,
    max_examples=100
)
