# test_strong_prompt_infer.py
# -*- coding: utf-8 -*-
import os, json, ast, argparse
import torch
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding,GenerationConfig


#SYSTEM_PROMPT = "You are a helpful assistant." 

SYSTEM_PROMPT = (
    "You are a linguistics span extractor.\n"
    "Return ONLY array of strings (no extra text before/after).\n"
    "Every string MUST be a verbatim substring of the INPUT (character-for-character), "
    "preserving case, punctuation, and whitespace.\n"
    "Do NOT paraphrase, normalize, or reorder text. Do NOT add labels or explanations.\n"
    "No duplicates. Keep spans in left-to-right order as they appear in the INPUT.\n"
     
    """STRICT FORMAT:
- Valid FORMAT: [], ["span"], ["span A", "span B"]
- Invalid FORMAT: [{"text": "..." }], [{"start": 0, "end": 3}], ["span", []], true, null, numbers"""
    "If there is no valid span, return [] exactly.\n"
    "Never include any explanation or details in the output."
)

def build_user_text(ex):

    return f"Instruction: {ex['instruction']}\nInput: {ex['input']}"

def build_prompt(tokenizer, ex):
    user_text = build_user_text(ex)
    
    return (
        f"SYSTEM: {SYSTEM_PROMPT}\n"
        f"USER: {user_text}\n"
        f"ASSISTANT:"
    )

def preprocess(tokenizer, ex):
    prompt = build_prompt(tokenizer, ex)
    tok = tokenizer(prompt, padding=False, truncation=True, return_tensors=None)
    ex["meta_prompt"] = prompt
    ex["input_ids"] = tok["input_ids"]
    ex["attention_mask"] = tok["attention_mask"]
    return ex
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"], 
    #target_modules=["q_proj","k_proj","v_proj","o_proj"] 
)
def main():
    gen_config = GenerationConfig.from_pretrained("microsoft/phi-1_5")
    parser = argparse.ArgumentParser()
   
    parser.add_argument("--model_name", default="microsoft/phi-1_5")
    
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--dataset", default="haeunkim/sentence_tag_multitask_v2")
    parser.add_argument("--split",   default="test")
    parser.add_argument("--output_dir", default="phi_output/dpo_samplecurriculum_2targets")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_shards", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end",   type=int, default=10)
    parser.add_argument("--temperature",   type=int, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    print(args.adapter_path, args.output_dir)
    model_name=args.model_name
   
    
    
    if args.adapter_path:
      print("wrong")
      base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
      tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, use_fast=True)
      model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
      print("correct")
      tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
      model = AutoModelForCausalLM.from_pretrained(model_name,  
   torch_dtype=torch.float16,
   device_map=None)

      
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
   
    tokenizer.padding_side = "left"  
    model.config.use_cache = False 
    model.to("cuda")
    model.eval()

    # ????
    dataset = load_dataset(args.dataset, split='test')
    print(len(dataset))
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    for idx in trange(args.start, args.end, desc="Shard Processing"):
        raw_shard = dataset.shard(num_shards=args.num_shards, index=idx)

        tokenized_shard = raw_shard.map(
            lambda ex: preprocess(tokenizer, ex),
            remove_columns=[c for c in raw_shard.column_names if c != "output"]
        )

        prompt_shard = [build_prompt(tokenizer, ex) for ex in raw_shard]
        gold_shard   = [ex["output"] for ex in raw_shard]

        tokenized_shard.set_format(type="torch", columns=["input_ids", "attention_mask"])
        dataloader = DataLoader(tokenized_shard, batch_size=args.batch_size, collate_fn=collator)
        gen_config = GenerationConfig.from_pretrained(model_name)
        gen_config.temperature = args.temperature
        gen_config.do_sample = True
        pred_texts = []
        for batch in tqdm(dataloader, desc=f"Shard {idx}"):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            
            with torch.no_grad():
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    
                    num_beams=1,
                    generation_config=gen_config,

                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                new_tokens = gen[:, input_ids.size(1):]
            pred_batch = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            pred_texts.extend([t.strip() for t in pred_batch])

        df = pd.DataFrame({
            "prompt": prompt_shard,
            "pred": pred_texts,
            "gold": gold_shard
        })
        folder_name=f"{args.output_dir}_temp_{args.temperature}"
        os.makedirs(args.output_dir, exist_ok=True)
       
      
        out_path = f"{args.output_dir}/shard_{idx}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[Saved] {out_path}")

if __name__ == "__main__":
    main()