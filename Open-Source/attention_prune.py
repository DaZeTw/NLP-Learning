import ast
import csv
import difflib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import nltk
import torch
import torch.nn
import torch.nn as nn
from customize_attention import PrunedAttention, TopKPrunedAttention
from nltk.tokenize import word_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mistral.modeling_mistral import repeat_kv

# nltk.download('punkt_tab')


def customize_model(isPrune=False, attn_type='prune', thresh=0.1, prune_start=16, k_const=64):
    devices = "cuda" if torch.cuda.is_available() else "cpu"
    local_model_path = "/home/da530038/lang-pro/Open-Source/mistral_model"
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        output_attentions=True,
        output_hidden_states=True,
        attn_implementation="eager"
    )
    model.eval()
    if (isPrune):
        if (attn_type == 'prune'):
            print(f"Threshold:{thresh}")
            for idx, block in enumerate(model.model.layers):
                block.self_attn = PrunedAttention(block.self_attn,
                                                  thresh=thresh,     # ← your mask threshold
                                                  layer_idx=idx)   # ← MUST pass layer index
        elif (attn_type == 'topk'):
            print(f"Prune Start:{prune_start} with K = {k_const}")
            for idx, block in enumerate(model.model.layers):
                block.self_attn = TopKPrunedAttention(
                    block.self_attn,
                    layer_idx=idx,
                    prune_start_layer=prune_start,
                    k_keep=k_const,          # or k_schedule
                )

    return model, tokenizer


def apply_attention_pruning(model, attn_type='topk', thresh=0.1, prune_start=16, k_const=64):
    """
    Modify attention layers in-place (no reloading).
    """
    if attn_type == 'prune':
        print(f"Apply PruneAttention: thresh = {thresh}")
        for idx, block in enumerate(model.model.layers):
            block.self_attn = PrunedAttention(
                block.self_attn, thresh=thresh, layer_idx=idx)
    elif attn_type == 'topk':
        print(f"Apply TopKPrune: prune_start={prune_start}, k={k_const}")
        for idx, block in enumerate(model.model.layers):
            block.self_attn = TopKPrunedAttention(
                block.self_attn,
                layer_idx=idx,
                prune_start_layer=prune_start,
                k_keep=k_const
            )


def run_inference(prompt_template, input_text, model, tokenizer):
    devices = 'cuda' if torch.cuda.is_available() else 'cpu'
    full_prompt = prompt_template.format(text=input_text)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(devices)

    # Step 1: Generate new tokens
    gen_output = model.generate(
        **inputs,
        max_new_tokens=1000,
        return_dict_in_generate=True,
        output_attentions=True,
        output_hidden_states=False,
    )

    # Step 2: Combine input + generated tokens
    gen_tokens = gen_output.sequences  # Shape: (1, input_len + gen_len)

    # Step 3: Re-run full input through model to get attention
    with torch.no_grad():
        full_output = model(
            input_ids=gen_tokens,
            output_attentions=True,
            output_hidden_states=True
        )

    decoded_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(gen_tokens[0])
    attentions = full_output.attentions

    # Print the model's full generated output (input + generated tokens)
    # print("Model Output:\n")
    # print(tokenizer.decode(gen_tokens[0], skip_special_tokens=True))

    return decoded_text, attentions, tokens


def extract_corrections(llm_output: str) -> Tuple[str, List[str]]:
    """
    Extracts corrected text and list of corrected words from LLM output.
    Supports flexible formats, including malformed outputs.

    Returns:
        (corrected_text, corrected_words)
    """
    # 1. Extract the last corrected text block
    text_matches = list(re.finditer(
        r"Corrected Text:\s*(.*?)(?:\n+|$)",
        llm_output,
        re.DOTALL
    ))
    corrected_text = text_matches[-1].group(1).strip() if text_matches else ""

    # 2. Extract the last corrected words block
    words_match = None

    # Try standard list pattern
    matches = list(re.finditer(
        r"Corrected Words[:\s]*\n*(\[[^\]]*\])",
        llm_output,
        re.DOTALL
    ))
    if matches:
        words_match = matches[-1].group(1).strip()
    else:
        # Try fallback: raw list even without brackets (line-by-line)
        raw_lines = llm_output.strip().splitlines()
        for i in reversed(range(len(raw_lines))):
            if "Corrected Words" in raw_lines[i]:
                # Look at next non-empty line(s)
                for j in range(i + 1, len(raw_lines)):
                    candidate = raw_lines[j].strip()
                    if candidate:
                        words_match = candidate
                        break
                break

    # Parse word list
    corrected_words = []
    if words_match:
        try:
            # Clean common issues
            words_match = words_match.replace("Ellipsis", "'...'")
            corrected_words = ast.literal_eval(words_match)
            if not isinstance(corrected_words, list):
                corrected_words = []
        except Exception:
            corrected_words = []

    return corrected_text, corrected_words


def clean_ellipsis(obj):
    """Recursively replace Ellipsis (...) with a string or None for JSON dumping"""
    if obj is ...:
        return "..."  # or return None
    elif isinstance(obj, list):
        return [clean_ellipsis(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: clean_ellipsis(v) for k, v in obj.items()}
    else:
        return obj


def process_spelling_detection(input_file, output_file, model, tokenizer, prompt_template):
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    for i, item in enumerate(data):
        text = item["text"]
        essay_id = item.get("id", f"sample_{i}")
        # Get LLM response
        raw_output, _, _ = run_inference(
            prompt_template, text, model, tokenizer)

        # Extract corrected text and corrected words
        corrected_text, corrected_words = extract_corrections(raw_output)

        # Add prediction to data
        item["llm_corrected_text"] = corrected_text
        item["llm_corrections"] = corrected_words
        item["llm_count"] = len(corrected_words)

        # Logging
        print(f"Processed Essay ID: {essay_id}")
        print(f"Corrected Words: {corrected_words}")
        print("-" * 50)

        # Limit processing to 100 items
        if i >= 100:
            break

    # Save updated data
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(clean_ellipsis(data), file, indent=4, ensure_ascii=False)

    print(f"✅ Output saved to: {output_file}")


def calculate_metrics(input_json_path, output_csv_path):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []  # <-- this was missing in your original function

    for i, sample in enumerate(data):
        sample_id = sample.get("id", f"sample_{i}")
        file_name = input_json_path.split(
            "/")[-1]  # or manually assign if needed
        # number of corrections (ground truth)
        count = len(sample.get("edits", [[]])[0][1])
        correction_tokens = [edit[2]
                             for edit in sample.get("edits", [[]])[0][1]]

        text = sample.get("text", "")
        token_count = len(word_tokenize(text))

        # 1. LLM-corrected words (e.g., from direct output)
        llm_corrections = sample.get("llm_corrections", [])
        llm_counter = Counter(llm_corrections)
        tp = sum(min(llm_counter[word], correction_tokens.count(
            word)) for word in set(correction_tokens))
        fp = len(llm_corrections) - tp
        fn = len(correction_tokens) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        exact_match = int(llm_corrections == correction_tokens)

        # 2. Extracted corrections from corrected text
        corrected_text = sample.get("llm_corrected_text", "")
        llm_token_count = len(word_tokenize(corrected_text))
        extracted = extract_corrections_from_llm_text(
            text, corrected_text) if corrected_text else []
        ex_counter = Counter(extracted)
        ex_tp = sum(min(ex_counter[word], correction_tokens.count(
            word)) for word in set(correction_tokens))
        ex_fp = len(extracted) - ex_tp
        ex_fn = len(correction_tokens) - ex_tp

        ex_precision = ex_tp / (ex_tp + ex_fp) if (ex_tp + ex_fp) > 0 else 0
        ex_recall = ex_tp / (ex_tp + ex_fn) if (ex_tp + ex_fn) > 0 else 0
        ex_exact_match = int(extracted == correction_tokens)

        rows.append({
            "sample_id": sample_id,
            "file": file_name,
            "token_count": token_count,
            "llm_token_count": llm_token_count,
            "expected_count": len(correction_tokens),

            "llm_count": len(llm_corrections),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "exact_match": exact_match,

            "ex_llm_count": len(extracted),
            "ex_tp": ex_tp,
            "ex_fp": ex_fp,
            "ex_fn": ex_fn,
            "ex_precision": round(ex_precision, 4),
            "ex_recall": round(ex_recall, 4),
            "ex_exact_match": ex_exact_match
        })

    rows.sort(key=lambda x: x['token_count'])

    # === Save to CSV ===
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Evaluation complete. CSV saved to: {output_csv_path}")


def summarize_metrics_by_length_bins(csv_path, bins=None):
    df = pd.read_csv(csv_path)

    # === Overall metrics ===
    total_tp = df["tp"].sum()
    total_fp = df["fp"].sum()
    total_fn = df["fn"].sum()
    total_exact = df["exact_match"].sum()
    total_samples = len(df)

    precision = total_tp / \
        (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / \
        (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision + recall) > 0 else 0
    accuracy = total_exact / total_samples if total_samples > 0 else 0

    overall_metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "total_samples": total_samples
    }

    # === Bin metrics ===
    if bins is None:
        bins = [0, 50, 100, 200, 500, np.inf]
    df["length_bin"] = pd.cut(df["token_count"], bins=bins)

    bin_metrics = []
    for bin_range, group in df.groupby("length_bin"):
        bin_tp = group["tp"].sum()
        bin_fp = group["fp"].sum()
        bin_fn = group["fn"].sum()
        bin_exact = group["exact_match"].sum()
        bin_total = len(group)

        bin_precision = bin_tp / \
            (bin_tp + bin_fp) if (bin_tp + bin_fp) > 0 else 0
        bin_recall = bin_tp / (bin_tp + bin_fn) if (bin_tp + bin_fn) > 0 else 0
        bin_f1 = 2 * bin_precision * bin_recall / \
            (bin_precision + bin_recall) if (bin_precision + bin_recall) > 0 else 0
        bin_accuracy = bin_exact / bin_total if bin_total > 0 else 0

        bin_metrics.append({
            "bin_range": str(bin_range),
            "samples": bin_total,
            "precision": round(bin_precision, 4),
            "recall": round(bin_recall, 4),
            "f1": round(bin_f1, 4),
            "accuracy": round(bin_accuracy, 4)
        })

    return overall_metrics, bin_metrics

    bin_labels = [m["bin_range"] for m in bin_metrics]
    precisions = [m["precision"] for m in bin_metrics]
    recalls = [m["recall"] for m in bin_metrics]
    f1s = [m["f1"] for m in bin_metrics]
    accuracies = [m["accuracy"] for m in bin_metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(bin_labels, precisions, marker='o', label='Precision')
    plt.plot(bin_labels, recalls, marker='s', label='Recall')
    plt.plot(bin_labels, f1s, marker='^', label='F1 Score')
    plt.plot(bin_labels, accuracies, marker='x', label='Accuracy')

    plt.title("Performance Metrics by Token Length Bin")
    plt.xlabel("Token Length Bin")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

    bin_labels = [m["bin_range"] for m in bin_metrics]
    precisions = [m["precision"] for m in bin_metrics]
    recalls = [m["recall"] for m in bin_metrics]
    f1s = [m["f1"] for m in bin_metrics]
    accuracies = [m["accuracy"] for m in bin_metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(bin_labels, precisions, marker='o', label='Precision')
    plt.plot(bin_labels, recalls, marker='s', label='Recall')
    plt.plot(bin_labels, f1s, marker='^', label='F1 Score')
    plt.plot(bin_labels, accuracies, marker='x', label='Accuracy')

    plt.title("Performance Metrics by Token Length Bin")
    plt.xlabel("Token Length Bin")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show


def extract_corrections_from_llm_text(original_text: str, corrected_text: str) -> List[str]:
    corrected_text = corrected_text.strip().replace('\\"', '"')
    if corrected_text.startswith('"') and corrected_text.endswith('"'):
        corrected_text = corrected_text[1:-1].strip()

    orig_tokens = word_tokenize(original_text)
    corr_tokens = word_tokenize(corrected_text)

    corrections = []
    sm = difflib.SequenceMatcher(None, orig_tokens, corr_tokens)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            # Only collect new words in the corrected version (substitutions/insertions)
            corrections.extend(corr_tokens[j1:j2])
    return corrections


def apply_human_edits_to_text(original_text, edits):
    edits = sorted(edits, key=lambda e: e[0], reverse=True)
    for start, end, replacement in edits:
        if replacement is None:
            replacement = ""
        original_text = original_text[:start] + \
            replacement + original_text[end:]
    return original_text


def _word_spans(text: str, words: List[str]) -> List[Tuple[int, int]]:
    spans = []
    for word in words:
        if not isinstance(word, str):
            continue
        word_clean = word.strip()
        if not word_clean:
            continue
        for match in re.finditer(r'\b{}\b'.format(re.escape(word_clean)), text, re.IGNORECASE):
            spans.append((match.start(), match.end()))
            break  # Only use the first match
    return spans


def _match_spans(human_spans, llm_spans):
    used, tp = set(), 0
    for h_start, h_end in human_spans:
        if h_start == -1:
            continue
        for idx, (l_start, l_end) in enumerate(llm_spans):  # FIXED: Added enumerate here
            if idx in used or l_start == -1:
                continue
            overlap = max(0, min(l_end, h_end) - max(l_start, h_start))
            total = max(l_end, h_end) - min(l_start, h_start)
            if total and (overlap/total >= .5 or l_start <= h_start <= h_end <= l_end):
                tp += 1
                used.add(idx)
                break
    return tp


def evaluate_sample(original_text: str,
                    corrected_text: str,
                    llm_corrections: List[str],
                    human_edits: List[Tuple[int, int, str]],
                    sample_id: str,
                    file_name: str) -> Dict:
    human_tokens = [r for _, _, r in human_edits]
    human_spans = [(s, e) for s, e, _ in human_edits]
    exp_counter = Counter(human_tokens)
    exp_total = len(human_tokens)

    true_text = apply_human_edits_to_text(original_text, human_edits)

    llm_counter = Counter(llm_corrections)
    llm_spans = _word_spans(true_text, llm_corrections)

    tp = _match_spans(human_spans, llm_spans)
    fp = len(llm_corrections) - tp
    fn = exp_total - tp
    prec = tp / (tp+fp) if tp+fp else 0
    rec = tp / (tp+fn) if tp+fn else 0
    match_flag = int(llm_counter == exp_counter)

    extracted = extract_corrections_from_llm_text(
        original_text, corrected_text)
    ex_counter = Counter(extracted)
    ex_spans = _word_spans(true_text, extracted)

    ex_tp = _match_spans(human_spans, ex_spans)
    ex_fp = len(extracted) - ex_tp
    ex_fn = exp_total - ex_tp
    ex_prec = ex_tp / (ex_tp+ex_fp) if ex_tp+ex_fp else 0
    ex_rec = ex_tp / (ex_tp+ex_fn) if ex_tp+ex_fn else 0
    ex_match = int(ex_counter == exp_counter)

    token_count = len(word_tokenize(true_text))
    llm_token_count = len(word_tokenize(corrected_text))
    print("Human spans:", human_spans)
    print("LLM spans:", llm_spans)
    print("LLM_ex spans:", ex_spans)

    return {
        "sample_id": sample_id,
        "file": file_name,
        "token_count": token_count,
        "llm_token_count": llm_token_count,
        "expected_count":  exp_total,

        "llm_count":   len(llm_corrections),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "match":     match_flag,

        "ex_llm_count": len(extracted),
        "ex_tp": ex_tp, "ex_fp": ex_fp, "ex_fn": ex_fn,
        "ex_precision": round(ex_prec, 4),
        "ex_recall":    round(ex_rec, 4),
        "ex_match":     ex_match,
    }


def eval_dataset_to_csv(json_path: str, output_csv: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    file_name = Path(json_path).name

    for idx, sample in enumerate(data):
        # === Parse human edits (format: [0, [[start, end, replacement], ...]]) ===
        human_edits = sample.get("edits", [[]])[0][1]
        # human_edits = []

        # if isinstance(edits_field, list) and len(edits_field) == 2 and isinstance(edits_field[1], list):
        #     human_edits = [
        #         (edit[0], edit[1], edit[2])
        #         for edit in edits_field[1]
        #         if isinstance(edit, list) and len(edit) == 3
        #     ]

        # Fallback if no edits found (can be adapted to your use case)
        if not human_edits:
            print("Can't get human_edits")

        rows.append(
            evaluate_sample(
                original_text=sample["text"],
                corrected_text=sample.get("llm_corrected_text", ""),
                llm_corrections=sample.get("llm_corrections", []),
                human_edits=human_edits,
                sample_id=f"{file_name}::sample_{idx}",
                file_name=file_name,
            )
        )

    if not rows:
        print("⛔ No samples found!")
        return

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Metrics saved to: {output_csv}")


prompt_template = """You are a human-like assistant designed for **spelling correction only**.

You must:
1. Fix **only spelling mistakes** in the input.
2. Return the corrected text.
3. Return a list of corrected words — only words that were changed due to spelling.

### STRICT RULES:
- Fix **only spelling mistakes**
- Do NOT include any examples.
- Do NOT list all words from the corrected text
- Do NOT include phrases — list only **single corrected words**.
- Do NOT include words that were already correct.

### Now, correct the following:
{text}

### Output Format:
Corrected Text:
<your corrected version here>

Corrected Words:
["word1", "word2", "word1", ...]
"""


def model_testing(thresh, attn_type, prune_start, k_const):
    text = """My name is Sarah. I am 17 years od. I am looking forawrd to joining you in this year's summer camps. I love children, and I enjoy looking after them. Also, I organized many sports activities before at my school. In additxion to tat, I enjoy coloking. My family think that my cooking is amazing. I hope thst you give my the chance to ioin you. Thanks"""
    human_edits = [
        [
            32,
            35,
            "old."
        ],
        [
            49,
            56,
            "forward"
        ],
        [
            215,
            224,
            "addition"
        ],
        [
            228,
            232,
            "that,"
        ],
        [
            241,
            250,
            "cooking."
        ],
        [
            302,
            306,
            "that"
        ],
        [
            333,
            337,
            "join"
        ]
    ]
    model, tokenizer = customize_model(
        thresh=thresh, attn_type=attn_type, isPrune=True, prune_start=prune_start, k_const=k_const)
    decoded_text, _, _ = run_inference(prompt_template, text, model, tokenizer)
    print("\nModel Output")
    print(decoded_text)
    corrected_text, corrected_words = extract_corrections(decoded_text)
    print(f"Corrected Text:\n {corrected_text}")
    print(f"Corrected Words\n {corrected_words}")
    word_extracted = extract_corrections_from_llm_text(text, corrected_text)
    print(f"Extracted words:\n{word_extracted}")

    sample_evaluation = evaluate_sample(
        text, corrected_text, corrected_words, human_edits, '12', 'haha')
    print(sample_evaluation)


def model_evaluation(input_file, threshs, params):
    # for thresh in threshs:
    #     model, tokenizer = customize_model(
    #         thresh=thresh, attn_type='prune', isPrune=True)
    #     json_out = f"results/correction_report_prune_{thresh}.json"
    #     csv_out = f"results/correction_csv_prune_{thresh}.csv"
    #     process_spelling_detection(
    #         input_file, json_out, model, tokenizer, prompt_template)
    #     eval_dataset_to_csv(json_out, csv_out)

    for param in params:
        model, tokenizer = customize_model(
            thresh=0, attn_type='topk',
            prune_start=param[0], k_const=param[1], isPrune=True)
        json_out = f"results/correction_report_topk_{param[0]}-{param[1]}.json"
        csv_out = f"results/correction_csv_topk_{param[0]}-{param[1]}.csv"
        process_spelling_detection(
            input_file, json_out, model, tokenizer, prompt_template)
        eval_dataset_to_csv(json_out, csv_out)


# threshs = [0.1, 0.01, 0.003, 0.001, 0.0005, 0.0001]\
# threshs = [0.005, 0.003, 0.002, 0.001, 0.0008]
# for thresh in threshs:
#     model_testing(thresh=thresh)
threshs = [
    #    0.01,
    #    0.001,
    #    0.0008,
    #    0.0005,
    0.0001]
params = [
    # (16, 64),
    # (22, 64),
    # (28, 64),
    # (16, 128),
    (22, 128),
    (28, 128)]
input_file = "/home/da530038/lang-pro/Open-Source/non_repeated_test.json"
model_evaluation(input_file, threshs=threshs, params=params)
# for param in params:
#     model_testing(thresh=0, attn_type='topk',
#                   prune_start=param[0], k_const=param[1])


# calculate_metrics('correction_report.json', 'correction_report.csv')
# csv_path = "correction_report.csv"
# overall, bin_metrics = summarize_metrics_by_length_bins(csv_path)
# print("Overall Metrics:", overall)
# plot_bin_metrics(bin_metrics)
