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
You are a meticulous assistant trained to answer multi-hop questions by reasoning across multiple documents, emulating detailed human-like analysis.<|im_end|>
<|im_start|>user
Your task follows three strict rules:

1. **Block-by-Block Reasoning**
   - You **must** go to examine **every block of text** in the provided context **in the exact order given**, without skipping any, as a human would systematically analyze each document.
   - For each block:
     - Begin with `Scratchpad:`  
       - Evaluate relevance to the question, citing specific details (e.g., names, nationalities, events) that align or conflict with the goal. 
       - If relevant, extract facts, explaining their role in answering the question, including context (e.g., what the fact implies), reliability (e.g., explicit vs. implied), and limitations (e.g., missing details).  
       - If irrelevant, justify why, citing specific mismatches (e.g., wrong person, topic, or timeframe).  
     - Follow with `Intermediate Result:`  
       - Summarize extracted facts and their role, or state “No useful info from this block” with a brief justification tied to the Scratchpad.

2. **Final Answer**
   - After analyzing all blocks, provide one line:  
     `Final Answer: <short factual answer>`  
   - The answer must be concise (e.g., “Yes”, “No”, or “Not found”).

3. **Format Strictness**
   - Include a `Scratchpad:` and `Intermediate Result:` pair for **every block of text** in the context, with no exceptions.  
   - Place `Final Answer:` as a single line at the end, with no extra text or sections.  
   - Follow the structure exactly, with detailed reasoning for each block.

---

### Context:
{context}

### Question:
{question}
<|im_end|>
<|im_start|>user
"""
question = """Were Scott Derrickson and Ed Wood of the same nationality?"""
context_string = """
Adam Collis: Adam Collis is an American filmmaker and actor. He attended the Duke University from 1986 to 1990 and the University of California, Los Angeles from 2007 to 2010. He also studied cinema at the University of Southern California from 1991 to 1997. Collis first work was the assistant director for the Scott Derrickson's short "Love in the Ruins" (1995). In 1998, he played "Crankshaft" in Eric Koyanagi's "Hundred Percent".

Ed Wood(film): Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood. The film concerns the period in Wood's life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau. Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones, Lisa Marie, and Bill Murray are among the supporting cast.

Tyler Bates: Tyler Bates(born June 5, 1965) is an American musician, music producer, and composer for films, television, and video games. Much of his work is in the action and horror film genres, with films like "Dawn of the Dead, 300, Sucker Punch," and "John Wick." He has collaborated with directors like Zack Snyder, Rob Zombie, Neil Marshall, William Friedkin, Scott Derrickson, and James Gunn. With Gunn, he has scored every one of the director's films
including "Guardians of the Galaxy", which became one of the highest grossing domestic movies of 2014, and its 2017 sequel. In addition, he is also the lead guitarist of the American rock band Marilyn Manson, and produced its albums "The Pale Emperor" and "Heaven Upside Down".

Doctor Strange(2016 film): Doctor Strange is a 2016 American superhero film based on the Marvel Comics character of the same name, produced by Marvel Studios and distributed by Walt Disney Studios Motion Pictures. It is the fourteenth film of the Marvel Cinematic Universe(MCU). The film was directed by Scott Derrickson, who wrote it with Jon Spaihts and C. Robert Cargill, and stars Benedict Cumberbatch as Stephen Strange, along with Chiwetel Ejiofor, Rachel McAdams, Benedict Wong, Michael Stuhlbarg, Benjamin Bratt, Scott Adkins, Mads Mikkelsen, and Tilda Swinton. In "Doctor Strange", surgeon Strange learns the mystic arts after a career-ending car accident.

Hellraiser: Inferno: Hellraiser: Inferno(also known as Hellraiser V: Inferno) is a 2000 American horror film. It is the fifth installment in the "Hellraiser" series and the first "Hellraiser" film to go straight-to-DVD. It was directed by Scott Derrickson and released on October 3, 2000. The film concerns a corrupt detective who discovers Lemarchand's box at a crime scene. The film's reviews were mixed.

Sinister(film): Sinister is a 2012 supernatural horror film directed by Scott Derrickson and written by Derrickson and C. Robert Cargill. It stars Ethan Hawke as fictional true-crime writer Ellison Oswalt who discovers a box of home movies in his attic that puts his family in danger.

Deliver Us from Evil(2014 film): Deliver Us from Evil is a 2014 American supernatural horror film directed by Scott Derrickson and produced by Jerry Bruckheimer. The film is officially based on a 2001 non-fiction book entitled "Beware the Night" by Ralph Sarchie and Lisa Collier Cool, and its marketing campaign highlighted that it was "inspired by actual accounts". The film stars Eric Bana, Édgar Ramírez, Sean Harris, Olivia Munn, and Joel McHale in the main roles and was released on July 2, 2014.

Woodson, Arkansas: Woodson is a census-designated place(CDP) in Pulaski County, Arkansas, in the United States. Its population was 403 at the 2010 census. It is part of the Little Rock–North Little Rock–Conway Metropolitan Statistical Area. Woodson and its accompanying Woodson Lake and Wood Hollow are the namesake for Ed Wood Sr., a prominent plantation owner, trader, and businessman at the turn of the 20th century. Woodson is adjacent to the Wood Plantation, the largest of the plantations owned by Ed Wood Sr.

Conrad Brooks: Conrad Brooks(born Conrad Biedrzycki on January 3, 1931 in Baltimore, Maryland) is an American actor. He moved to Hollywood, California in 1948 to pursue a career in acting. He got his start in movies appearing in Ed Wood films such as "Plan 9 from Outer Space", "Glen or Glenda", and "Jail Bait." He took a break from acting during the 1960s and 1970s but due to the ongoing interest in the films of Ed Wood, he reemerged in the 1980s and has become a prolific actor. He also has since gone on to write, produce and direct several films.

The Exorcism of Emily Rose: The Exorcism of Emily Rose is a 2005 American legal drama horror film directed by Scott Derrickson and starring Laura Linney and Tom Wilkinson. The film is loosely based on the story of Anneliese Michel and follows a self-proclaimed agnostic who acts as defense counsel(Linney) representing a parish priest(Wilkinson), accused by the state of negligent homicide after he performed an exorcism.
"""

local_path = "/home/da530038/lang-pro/Open-Source/nous_hermes_2_mistral"
model, tokenizer, tracker = customize_model(
    local_path, model_type="pruned", last_k_layers=8)
full_prompt = prompt_template.format(question=question, context=context_string)

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


def evaluate_hotpotqa_examples(
    data_path: str,
    output_path: str,
    model_path: str,
    prompt_template: str,
    last_k_layers: int = 6,
    max_examples: int = 100
):
    """
    Run evaluation on HotpotQA-style data with long contexts and reasoning,
    saving reasoning traces, answers, compression ratios, and inference time.

    Parameters:
    - data_path: Path to input JSON with "question", "answer", and "context"
    - output_path: Path to write the evaluation results
    - model_path: HuggingFace-compatible model path
    - prompt_template: Template with {question} and {context} to format prompt
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

        # Format context
        context_text = ""
        for title, sentences in ex["context"]:
            paragraph = " ".join(sentences)
            context_text += f"{title}: {paragraph}\n"
        context_text = context_text.strip()

        question_text = ex["question"]
        gold_answer = ex["answer"]
        prompt = prompt_template.format(
            question=question_text, context=context_text)

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
            "context": context_text,
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


evaluate_hotpotqa_examples(
    data_path="/home/da530038/lang-pro/Benchmarks/hotpot_dev_fullwiki_v1.json",
    output_path="/home/da530038/lang-pro/Benchmarks/output/nous_mistral_hotpotqa_eval_results.json",
    model_path="/home/da530038/lang-pro/Open-Source/nous_hermes_2_mistral",
    prompt_template=prompt_template,
    last_k_layers=6,
    max_examples=100
)
