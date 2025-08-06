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


prompt_template = """<|im_start|>system
You are a helpful and accurate assistant for solving multiple-choice language understanding questions.

Your task is divided into two stages:

1. **Scratchpad Stage**: Analyze the original sentence and evaluate each choice step by step.
   - Use `Scratchpad:` for the sentence context to provide a comprehensive analysis, including:
     - **Syntactic structure**: Identify the sentence's grammatical components (e.g., subject, verb, object) and how they shape the required word or phrase.
     - **Semantic meaning**: Clarify the sentence's core meaning, intent, and any implied relationships or goals.
     - **Contextual cues**: Consider tone, register (e.g., formal/informal), and pragmatic implications (e.g., cultural or situational context).
     - **Potential ambiguities**: Note any unclear elements that might affect interpretation.
   - For each choice, use a separate `Scratchpad:` to evaluate its fit, addressing:
     - **Syntactic fit**: Assess whether the word or phrase matches the required part of speech and grammatical role in the sentence.
     - **Semantic fit**: Evaluate how the meaning aligns with the sentence's intent and context.
     - **Contextual appropriateness**: Analyze tone, register, collocations (common word pairings), and connotations (positive, negative, or neutral).
     - **Pragmatic suitability**: Consider real-world plausibility, cultural nuances, or situational relevance.
     - **Comparison to sentence requirements**: Explain why the choice satisfies or fails to meet the sentence's needs, referencing specific linguistic or contextual evidence.
   - Use `Intermediate Result:` after each choice to summarize whether the choice is plausible or eliminated, providing a concise justification that ties directly to the Scratchpad analysis, highlighting key reasons for the decision.

2. **Final Answer Stage**: After analyzing all choices, select the best-fitting word or phrase from the provided list. Output **only** the correct word or phrase in the `Final Answer:` section, with no punctuation or additional text.

### RULES:
- Do NOT modify the original sentence.
- Provide one `Scratchpad:` for the sentence context analysis before evaluating choices.
- Evaluate each choice in a separate `Scratchpad:` with detailed reasoning as specified above.
- In each `Intermediate Result:`, clearly state whether the choice is plausible or eliminated, with a brief justification referencing the Scratchpad analysis.
- In `Final Answer:`, output only the chosen word or phrase (e.g., `ignore`).

Question:
{question}

Choices:
{choices}
<|im_end|>
<|im_start|>user
"""


question = """Sammy wanted to go to where the people were.  Where might he go?"""
choices = """race track,
      populated areas,
      the desert,
      apartment,
        roadblock"""
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


def evaluate_commonqa_examples(
    data_path: str,
    output_path: str,
    model_path: str,
    prompt_template: str,
    last_k_layers: int = 6,
    max_examples: int = 100
):
    """
    Run evaluation on multiple-choice questions (e.g., CommonQA-style) for both baseline and pruned models,
    saving reasoning traces, answers, compression ratios, and inference time.

    Parameters:
    - data_path: Path to input JSON with "question", "choices", and "correct_choice"
    - output_path: Path to write the evaluation results
    - model_path: HuggingFace-compatible model path
    - prompt_template: Template with {question} and {choices} to format prompt
    - last_k_layers: Number of layers to apply pruning on for the pruned model
    - max_examples: Number of examples to evaluate
    """

    with open(data_path, "r") as f:
        data = json.load(f)

    examples = random.sample(data, min(max_examples, len(data)))
    results = []

    total_baseline_time = 0.0
    total_pruned_time = 0.0

    for idx, ex in enumerate(tqdm(examples, desc="Evaluating")):
        # Load models separately to avoid cross-KV interference
        baseline_model, tokenizer, _ = customize_model(
            model_path, model_type="baseline")
        pruned_model, _, tracker = customize_model(
            model_path, model_type="pruned", last_k_layers=last_k_layers)

        # Format question
        question_text = ex["question"]
        choice_text = "\n".join([f"- {c}" for c in ex["choices"]])
        gold_answer = ex["correct_choice"]
        prompt = prompt_template.format(
            question=question_text, choices=choice_text)

        # Baseline inference
        start = time.time()
        base_out, base_full, _ = run_inference(
            baseline_model, tokenizer, prompt, tracker=None, verbose=False)
        end = time.time()
        base_time = end - start
        total_baseline_time += base_time

        base_answer, base_scratchpad, _ = extract_inference_outputs(
            base_full, kv_cache=[(None, None)])  # dummy for baseline

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


evaluate_commonqa_examples(
    data_path="/home/da530038/lang-pro/Benchmarks/commonsense_qa.json",
    output_path="/home/da530038/lang-pro/Benchmarks/output/nous_mistral_commonqa_eval_results.json",
    model_path="/home/da530038/lang-pro/Open-Source/nous_hermes_2_mistral",
    prompt_template=prompt_template,
    last_k_layers=6,
    max_examples=100
)
