import torch
from robust_prune_attention import ScratchpadTracker, SingleLayerScratchpadPruner
from transformers import LlamaTokenizer, MistralForCausalLM

# Load model and tokenizer
local_path = "/home/da530038/lang-pro/Open-Source/nous_hermes_2_mistral"
tokenizer = LlamaTokenizer.from_pretrained(local_path, trust_remote_code=True)
model = MistralForCausalLM.from_pretrained(local_path,
                                           torch_dtype=torch.float16,
                                           device_map="auto",
                                           attn_implementation="eager")

# Attach custom wrappers to all layers
wrappers = []
total_layers = len(model.model.layers)
for i in range(total_layers):
    blk = model.model.layers[i]
    wrapper = SingleLayerScratchpadPruner(
        blk.self_attn, layer_idx=i, total_layers=total_layers, debug=True)
    blk.self_attn = wrapper
    wrappers.append(wrapper)

# Initialize tracker
tracker = ScratchpadTracker(tokenizer, wrappers)

# Prompt
prompt_template = """<|im_start|>system
You are a human-like assistant designed for spelling correction only.<|im_end|>
<|im_start|>user
Your task is divided into two stages for each sentence:
1. Scratchpad Stage: Think step-by-step and identify only the spelling errors in that sentence.
2. Output Stage: Provide a list of corrected words — only words that were changed due to spelling mistakes in that sentence.

STRICT RULES:
- Fix only spelling mistakes — do NOT correct grammar or phrasing.
- Go sentence by sentence — do NOT merge corrections across multiple sentences.
- For each sentence:
  - First provide a scratchpad reasoning block.
  - Then, provide the corrected words for that sentence only.
- The list must include only single corrected words. No phrases. No words that were already correct.
- If a misspelled word appears multiple times in a sentence, include it multiple times.
- Do NOT include the corrected version of the sentence.
- If no spelling mistakes are found in a sentence, return an empty list.

FINAL STEP (IMPORTANT):
After finishing all sentences, combine all corrected words from each sentence into one final list.
This list must include all corrected words, preserving duplicates where they occurred.

### Format Example:

Scratchpad:
Sentence 1: I am 17 years od.
"od" is a misspelling of "old".
Corrected Words: ["old"]

Scratchpad:
Sentence 2: I enjoy coloking and bakng.
"coloking" is a misspelling of "cooking"
"bakng" is a misspelling of "baking".
Corrected Words: ["cooking", "baking"]

Final Corrected Words:
["old", "cooking", "baking"]

Now, correct the following text sentence by sentence:

{text}
<|im_end|>
<|im_start|>assistant
"""

text = """About mye! Hey, my name is Mathias. I nwas bolrn in Danderyds hospital but lived in Stockholm. When I was four, I mved to Danderyd. I have a big brother. He's 14 and goes on Friberga as well. His name is Ulric and he's in 8f2. My favorite activity is football. It's great fun to play it with all my friends. I aslo like to play floorball but it isn\u2019t as fun as fodotball. I also kike to lpay video games. I paly with my rfiends. My favorite games are FIFA, NHL, GTAV and some others. I have lived in stonckholm avnd in two houses in Danderyd, but no other places. I don\u2019t know which is the best place I have been to jbut New York was cool, but I actually like F\u00e5r\u00f6 most. It is an island next to Gotland and I hsve been going there since I was 0 years. We always rent a littel cottage from a shdep fahrmer and now we know his family. We always go to the beach, which is very jice, and when it\u2019s sunny and warm it is better than the Mediterranean. I dont know what I\u2019m proud of, but when I score a nice gosl, I am proud ecause it feels goo,d especialyl when it\u2019s an important goal that maye will change the game. I would lke to be the best football player in the wobrld, a Youtuber, or take over a big company. It is often hard to get up in the morning. I aleays think jst one more minute rand then I fall asleep. I asekd my parents and they said ambitious, curious and considerate. I don\u2019t know if I an agree wjth tht but I listened to them. :) I don\u2019t know what I like people to know aobut me. That I danced for two years, but I stopped because I didn\u2019t have time. I woyld ilke to play a World Cup game and I also want to wni Championsions League. I don\u2019t have a favourite movie. I lgike mny but the movaies I like tjhe mot are clmedies and action adventure. I don\u2019t read many books. I like a lot of sonvs, bhut I think it\u2019s Let\u2019s Do It Again by J Boog because I like the singer's voice, i\u2019s a song and it\u2019s easy to sing too. I also like Stolen Dance by Milky Chance because it\u2019s a sojg. I don\u2019t only liie music, I also like pop music and house music. I like to watch South Park. It's really funny and I have amlost seen ll 17 seaosns. I also like to watch YouTube videos every day. I like a lot of food, hut is good and pizza. My favourite restaurant must have tood mwat and be really nice. My favourite emmories must be every summer on F\u00e5r\u00f6."""
prompt = prompt_template.format(text=text)

# Tokenize full prompt and mark for tracker
input_ids_full = tokenizer(
    prompt, return_tensors="pt").input_ids.to(model.device)
prompt_ids = input_ids_full[0].tolist()
tracker.initialize_with_prompt(prompt_ids)

kv_cache = None
kept_ids = []
all_ids = []
max_new_tokens = 2000

with torch.no_grad():
    out = model(
        input_ids=input_ids_full,
        use_cache=True,
        output_attentions=True
    )
    kv_cache = out.past_key_values
    next_id = out.logits[:, -1].argmax(-1)
    all_ids.append(next_id.item())
    
    if tracker.step(next_id.item()):
        kept_ids.append(next_id.item())
    
    input_ids = next_id.unsqueeze(0)

    for step in range(max_new_tokens):
        out = model(
            input_ids=input_ids,
            past_key_values=kv_cache,
            use_cache=True,
            output_attentions=True,
        )

        logits, kv_cache = out.logits, out.past_key_values
        next_id = logits[:, -1].argmax(-1)
        all_ids.append(next_id.item())

        if tracker.step(next_id.item()):
            kept_ids.append(next_id.item())

        input_ids = next_id.unsqueeze(0)
        if next_id.item() == tokenizer.eos_token_id:
            break

# Decode results
generated_text = tokenizer.decode(kept_ids, skip_special_tokens=True)
origin_text = tokenizer.decode(all_ids, skip_special_tokens=True)

print("\n--- OUTPUT TEXT ---\n", generated_text)
print("\n--- ORIGIN TEXT ---\n", origin_text)
