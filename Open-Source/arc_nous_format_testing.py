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

    model = MistralForCausalLM.from_pretrained(local_path,
                                               torch_dtype=torch.float16,
                                               device_map="auto",
                                               attn_implementation="eager")

    tokenizer = LlamaTokenizer.from_pretrained(
        local_path, trust_remote_code=True)

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
                total_layers=total_layers,
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
    Extract final answer, scratchpad lines, and compression ratio from output.
    If kv_cache is None or invalid, returns compression_ratio = -1.
    """

    # 1. Extract final answer
    match = re.search(r"Final\s*Answer\s*:\s*(\d+)", full_text, re.I)
    final_answer = match.group(1) if match else ""

    # 2. Extract scratchpad-related lines
    scratchpad_lines = []
    for line in full_text.splitlines():
        line = line.strip()
        if line.lower().startswith("scratchpad:") or line.lower().startswith("intermediate result:"):
            scratchpad_lines.append(line)

    # 3. Compression ratio: 1 - (min / max), only if valid kv_cache
    if (
        isinstance(kv_cache, (list, tuple)) and
        all(layer and hasattr(layer[0], "shape") for layer in kv_cache)
    ):
        kv_lengths = [layer[0].shape[-2] for layer in kv_cache]
        kv_min = min(kv_lengths)
        kv_max = max(kv_lengths)
        compression_ratio = 1 - (kv_min / kv_max) if kv_max > 0 else 0.0
    else:
        compression_ratio = -1  # signal not applicable

    return final_answer, scratchpad_lines, compression_ratio


prompt_template = """You are a helpful and intelligent assistant designed to solve challenging science questions from standardized tests.

For each question, follow these steps:

1. **Step-by-Step Choice Evaluation**:
   - Analyze the question carefully.
   - For **each answer choice**, provide a `Scratchpad:` explaining whether it is correct or incorrect, based on **scientific reasoning** and **real-world knowledge**.
   - Follow each scratchpad with an `Intermediate Result:` that clearly states whether the choice is **valid** or **eliminated**, and why.

2. **Final Answer Selection**:
   - After evaluating all choices, conclude with `Final Answer:` followed by the full text of the best choice.

---

Question:
{question}

Choices:
{choices}

---

### Format Example:

Question:
George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?

Choices:
- dry palms
- wet palms
- palms covered with oil
- palms covered with lotion

Scratchpad: Rubbing dry palms creates more friction, which leads to more heat generation.

Intermediate Result: dry palms is a strong candidate due to higher friction.

Scratchpad: Wet palms reduce friction because of moisture, generating less heat.

Intermediate Result: wet palms is eliminated due to lower friction.

Scratchpad: Oil creates a slippery surface, reducing friction further.

Intermediate Result: palms covered with oil is eliminated.

Scratchpad: Lotion also reduces friction and acts as a lubricant.

Intermediate Result: palms covered with lotion is eliminated.

Final Answer:
dry palms
"""


question = """Which of the following statements best explains why magnets usually stick to a refrigerator door?"""
choices = """The refrigerator door is smooth.,
      The refrigerator door contains iron.,
      The refrigerator door is a good conductor.,
      The refrigerator door has electric wires in it."""
local_path = "/home/da530038/lang-pro/Open-Source/nous_hermes_2_mistral"
model, tokenizer, tracker = customize_model(
    local_path, model_type="pruned", last_k_layers=8)
full_prompt = prompt_template.format(question=question, choices=choices)

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


def evaluate_arc_examples(
    data_path: str,
    output_path: str,
    model_path: str,
    prompt_template: str,
    last_k_layers: int = 6,
    max_examples: int = 100
):
    """
    Evaluate multiple-choice science questions (AI2 ARC-style) on baseline and pruned models.
    Logs final answer, scratchpad reasoning, compression stats, and inference time for each model.

    Parameters:
    - data_path: Path to input JSON with "question", "choices", and "correct_choice"
    - output_path: Path to save evaluation results as JSON
    - model_path: HuggingFace-compatible local or remote model path
    - prompt_template: Prompt with placeholders: {question}, {choices}
    - last_k_layers: Number of layers to prune in the pruned model
    - max_examples: Number of random examples to evaluate
    """
    with open(data_path, "r") as f:
        data = json.load(f)

    examples = random.sample(data, min(max_examples, len(data)))
    results = []

    total_baseline_time = 0.0
    total_pruned_time = 0.0

    for idx, ex in enumerate(tqdm(examples, desc="Evaluating")):
        # Reload models to prevent KV state interference
        baseline_model, tokenizer, _ = customize_model(
            model_path, model_type="baseline")
        pruned_model, _, tracker = customize_model(
            model_path, model_type="pruned", last_k_layers=last_k_layers)

        # Format input prompt
        question_text = ex["question"]
        choice_text = "\n".join([f"- {c}" for c in ex["choices"]])
        gold_answer = ex["correct_choice"]
        prompt = prompt_template.format(
            question=question_text, choices=choice_text)

        # Baseline inference
        start = time.time()
        base_out, base_full, _ = run_inference(
            baseline_model, tokenizer, prompt, tracker=None, verbose=False)
        base_time = time.time() - start
        total_baseline_time += base_time

        base_answer, base_scratchpad, _ = extract_inference_outputs(
            base_full, kv_cache=[(None, None)])  # no pruning

        # Pruned inference
        start = time.time()
        pruned_out, pruned_full, pruned_kv = run_inference(
            pruned_model, tokenizer, prompt, tracker=tracker, verbose=False)
        pruned_time = time.time() - start
        total_pruned_time += pruned_time

        pruned_answer, pruned_scratchpad, compression = extract_inference_outputs(
            pruned_full, pruned_kv)

        results.append({
            "id": idx,
            "question": question_text,
            "choices": ex["choices"],
            "gold": gold_answer,
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

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved results to {output_path}")


# evaluate_arc_examples(
#     data_path="/home/da530038/lang-pro/Benchmarks/ai2_arc_arc-challenge.json",
#     output_path="/home/da530038/lang-pro/Benchmarks/output/arc_eval_results.json",
#     model_path="/home/da530038/lang-pro/Open-Source/mistral_model",
#     prompt_template=prompt_template,
#     last_k_layers=6,
#     max_examples=100
# )
