from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os, re, json, ast, argparse, torch 
from datasets import load_dataset ,concatenate_datasets
import torch.nn.functional as F
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
from peft import LoraConfig, PeftModel, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch
import torch.nn.functional as F
from trl import DPOTrainer
MODEL_NAME =  "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

class DPOContrastiveTrainer(DPOTrainer):
  
  def __init__(self,*args, beta: float = 0.1, contrastive_weight: float = 0.2, margin: float = 0.01, **kwargs):
    super().__init__(*args, **kwargs)
    self.beta = beta
    self.contrastive_weight = contrastive_weight
    self.triplet = torch.nn.TripletMarginLoss(margin=margin, reduction="mean")
    self._pad_id = (getattr(tokenizer, "pad_token_id", None)
                        or getattr(tokenizer, "eos_token_id", None) or 0)
    print(f'margin:{margin}')
  def _ensure_mask(self,ids, mask): #mask
    if mask is None:
        return torch.ones_like(ids, dtype=torch.long, device=ids.device)
    return mask
  
  def _pad_to_len(self,ids,mask,Lmax): # padding 
    
    B , L = ids.size()
    if L == Lmax:
      return ids, mask
    pad_ids =torch.full((B,Lmax -L),self._pad_id,dtype=ids.dtype,device=ids.device)
    pad_mask = torch.zeros((B, Lmax - L), dtype=mask.dtype, device=mask.device)
    return torch.cat([ids, pad_ids], dim=1), torch.cat([mask, pad_mask], dim=1)
  
  def compute_loss(self,model,inputs,return_outputs:bool=False,num_items_in_batch:int |None=None,**kwargs):
    
    try:
      base_loss, base_outputs= super().compute_loss(model ,inputs,return_outputs=True, num_items_in_batch=num_items_in_batch)
    except TypeError:
            base_loss, base_outputs = super().compute_loss(     
                model, inputs, return_outputs=True
            )
    a_ids = inputs.get("prompt_input_ids", inputs.get("input_ids", inputs["chosen_input_ids"]))
    a_ms  = self._ensure_mask(a_ids, inputs.get("prompt_attention_mask", inputs.get("attention_mask")))
    
    p_ids = inputs["chosen_input_ids"]
    p_ms  = self._ensure_mask(p_ids, inputs.get("chosen_attention_mask"))
    
    n_ids = inputs["rejected_input_ids"]
    n_ms  = self._ensure_mask(n_ids, inputs.get("rejected_attention_mask"))
    
    Lmax = max(a_ids.size(1),p_ids.size(1),n_ids.size(1))
    a_ids ,a_ms=self._pad_to_len(a_ids, a_ms, Lmax)
    p_ids ,p_ms =self._pad_to_len(p_ids, p_ms, Lmax)
    n_ids ,n_ms = self._pad_to_len(n_ids,n_ms,Lmax)
    
    all_ids = torch.cat([a_ids,p_ids,n_ids],dim=0)
    all_mask =torch.cat([a_ms,p_ms,n_ms],dim=0)
    
    out = model(input_ids=all_ids, 
                attention_mask=all_mask,
                output_hidden_states=True,
                return_dict=True)
    #(3b,l,h)->(3b,h)
    last =out.hidden_states[-1]
    #(3b,h)->(3b,h,1)
    m=all_mask.float().unsqueeze(-1)
    pooled = (last *m).sum(dim=1)/m.sum(dim=1).clamp_min(1.0)
    
    B = a_ids.size(0)
    a,p,n = torch.split(pooled, [B,B,B],dim=0) 
    a= F.normalize(a,p=2,dim=1)
    p=F.normalize(p,p=2,dim=1)
    n=F.normalize(n,p=2,dim=1)
    c_loss = self.triplet(a,p,n)
    total = base_loss +c_loss *self.contrastive_weight
    
    if return_outputs:
      
      out_dict=dict(base_outputs)
      out_dict['base_loss']=base_loss.detach()
      out_dict['contrastive_loss'] =c_loss.detach()
      return total, out_dict
    
    return total
    
              
  
    
  

SYSTEM_PROMPT = (
    "You are a linguistics span extractor.\n"
    "Return ONLY a JSON array of strings (no extra text before/after).\n"
    "Every string MUST be a verbatim substring of the INPUT (character-for-character), "
    "preserving case, punctuation, and whitespace.\n"
    "Do NOT paraphrase, normalize, or reorder text. Do NOT add labels or explanations.\n"
    "No duplicates. Keep spans in left-to-right order as they appear in the INPUT.\n"
    "If there is no valid span, return [] exactly.\n"
    "Never include any additional text in the output."
) 


peft_config = LoraConfig(
      r=8,
      lora_alpha=16,
      lora_dropout=0.1,
      bias="none",
      task_type=TaskType.CAUSAL_LM,
      target_modules=["q_proj", "v_proj"],
  )

def build_user_text(ex):
    if "prompt" in ex and ex["prompt"]:
        return ex["prompt"].strip()
    inst = (ex.get("instruction") or "").strip()
    inp  = (ex.get("input") or "").strip()
    return f"Instruction: {inst}\nInput: {inp}".strip()

def build_message(ex):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_text(ex)},
    ]
    return messages

def normalize_to_json(x):
    if isinstance(x, str):
        try:
            v = json.loads(x)
            if isinstance(v, list):
                return json.dumps([str(s).strip() for s in v], ensure_ascii=False)
        except: pass
        try:
            v = ast.literal_eval(x)
            if isinstance(v, list):
                return json.dumps([str(s).strip() for s in v], ensure_ascii=False)
        except:
            return json.dumps([x.strip()], ensure_ascii=False)
    elif isinstance(x, list):
        return json.dumps([str(s).strip() for s in x], ensure_ascii=False)
    return json.dumps([str(x).strip()], ensure_ascii=False)
    
def dpo_preprocess(examples):
  batch_size = len(examples[next(iter(examples))])
  prompts , chosens ,rejecteds = [], [] ,[]
   
  for i in range(batch_size):
    ex = {k: examples[k][i] for k in examples}
    message =build_user_text(ex)
    prompt = (
        f"SYSTEM: {SYSTEM_PROMPT}\n"
        f"USER: {message}\n"
        f"ASSISTANT:"
    )
    prompts.append(prompt)
    if 'chosen' in examples and examples['chosen'] is not None:
      chosens.append(normalize_to_json(examples['chosen'][i]) + tokenizer.eos_token)
    elif 'output' in examples and examples['output'] is not None:
      chosens.append(normalize_to_json(examples['output'][i]) + tokenizer.eos_token)
    rejecteds.append(normalize_to_json(examples['rejected'][i])+ tokenizer.eos_token)
  return {"prompt": prompts, "chosen": chosens, "rejected" :rejecteds}

def build_ref(prev_outdir=None):
  ref_base = AutoModelForCausalLM.from_pretrained(
  MODEL_NAME, torch_dtype= torch.float16)
  ref_base.config.use_cache=False
  if prev_outdir:
    ref = PeftModel.from_pretrained(ref_base , prev_outdir)
  else:
    ref= ref_base
  ref.eval()
  for p in ref.parameters():
    p.requires_grad_(False)
  return ref

def sft_preprocess(examples,max_seq_length=1024):
  batch_size= len(examples['input'])
  prompts ,targets = [] ,[]
  
  for i in range(batch_size):
    ex = {k: examples[k][i] for k in examples}
    message = build_user_text(ex)
    
    prompt_text = (
        f"SYSTEM: {SYSTEM_PROMPT}\n"
        f"USER: {message}\n"
        f"ASSISTANT:"
    )
    prompts.append(prompt_text)

    
    outs = examples.get("output", None)
    if 'chosen' in examples and examples['chosen'] is not None:
      out_obj = examples['chosen'][i]
    elif 'output' in examples and examples['output'] is not None:
      out_obj = examples['output'][i]
    else:
      out_obj = "[]"
    target_text = normalize_to_json(out_obj) + tokenizer.eos_token 
    targets.append(target_text)
  full_texts = [p + t for p, t in zip(prompts, targets)]

 
  enc = tokenizer(
      full_texts,
      padding="max_length",
      truncation=True,
      max_length=max_seq_length,
      add_special_tokens=False,
      return_tensors="pt",
  )
  input_ids = enc["input_ids"]              
  attn_mask = enc["attention_mask"]
  labels    = input_ids.clone()

  
  prompt_tok = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
        return_tensors=None,
    )
  prompt_len_list = [len(x) for x in prompt_tok["input_ids"]]

  B, L = input_ids.size()
  for i in range(B):
       
    labels[i][attn_mask[i] == 0] = -100
    
    p_len = min(prompt_len_list[i], L)
    labels[i][:p_len] = -100

  return {
      "input_ids": input_ids.tolist(),
      "attention_mask": attn_mask.tolist(),
      "labels": labels.tolist(),
  }
def get_latest_checkpoint(dirpath: str):
    if not os.path.isdir(dirpath):
        return None
    cand = []
    for name in os.listdir(dirpath):
        m = re.match(r"checkpoint-(\d+)$", name)
        if m:
            cand.append((int(m.group(1)), name))
    if not cand:
        return None
    cand.sort(reverse=True)
    return os.path.join(dirpath, cand[0][1]) 

     
def main():
  ap=argparse.ArgumentParser()
  ap.add_argument("--stage", required=True, choices=["sample_dpo","low","medium","high","only_dpo",])
  
  ap.add_argument("--output_dir", default="phi_models/phi_curriculum_dpo_sample/2targets_20P_strong")
  ap.add_argument("--epochs", type=int, default=1)
  ap.add_argument("--lr", type=float, default=5e-6)
  ap.add_argument("--beta", type=float, default=0.05)
  ap.add_argument("--sft_dir", type=str, default=None),

  args = ap.parse_args()
  outdir = f"{args.output_dir}/{args.stage}"
  
  
  
  sft_config = SFTConfig(
      output_dir=outdir,
      num_train_epochs=1,
      per_device_train_batch_size=2,
      gradient_accumulation_steps=1,
      report_to="wandb",
      learning_rate=2e-4,
      logging_steps=10,
      logging_dir="./logs",
      save_strategy="steps",
      save_steps=10,
      save_total_limit=3,
      bf16=False,    
      fp16=True, 
      optim="adamw_torch",
      packing=False,
      #deepspeed="configs/zero_stage_3.json",
      label_names=["labels"],
       save_safetensors=True
  
      
     
  )
  dpo_config = DPOConfig(
          output_dir=outdir,
          per_device_train_batch_size=4,          
          gradient_accumulation_steps=8,           
          num_train_epochs=3,
          learning_rate=args.lr,
          beta=args.beta,
          max_prompt_length=128,
          max_length=256,
          save_steps=200,
          save_total_limit=3,
          logging_steps=10,
          remove_unused_columns=False,
          deepspeed="configs/zero_stage_2.json",
          lr_scheduler_type="cosine",
          warmup_ratio=0.1,
          report_to="wandb",
          bf16=False,
          fp16=True,
            gradient_checkpointing=False,
           save_safetensors=True
      )
  
  latest_ckpt = get_latest_checkpoint(outdir)
  
  
  
  if args.stage =="sample_dpo":
    dataset_low = load_dataset("haeunkim/curriculum_learning_dpo_created_negative",split='train_low')
    dataset_medium = load_dataset("haeunkim/curriculum_learning_dpo_created_negative",split='train_medium')
    dataset_high = load_dataset("haeunkim/curriculum_learning_dpo_created_negative",split='train_high')
    dataset_total=concatenate_datasets([dataset_low,dataset_medium,dataset_high])
    n=int(len(dataset_total)*0.1)
    dataset = dataset_total.shuffle(seed=42).select(range(n))
    
    
    tokenized_dpo = dataset.map(
    lambda x: dpo_preprocess(x), 
    remove_columns=dataset.column_names,
    batched=True,
    batch_size=32,
    load_from_cache_file=True
)   
    latest_ckpt = args.sft_dir if args.sft_dir else f"{args.output_dir}/high"
    
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    )
    base = prepare_model_for_kbit_training(base)
    model =  PeftModel.from_pretrained(base, f"{args.output_dir}/high", is_trainable=True)
   
    base.config.use_cache = False
    
    trainer = DPOContrastiveTrainer(
        model=model,
        ref_model=build_ref(f"{args.output_dir}/high"),
        args=dpo_config,
        train_dataset=tokenized_dpo,
        eval_dataset=None,
        processing_class=tokenizer,  
        contrastive_weight=0.2  
                                        
    )
    
  
    trainer.train()
    trainer.model.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    return 
  elif args.stage=='only_dpo':
    raw = load_dataset("haeunkim/sentence_tag_dpo_train_v2",split='train')
    ds_dpo = raw.map(lambda ex: dpo_preprocess(ex), batched=True,
                     remove_columns=raw.column_names, load_from_cache_file=True)

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    )
    base = prepare_model_for_kbit_training(base)
    policy = get_peft_model(base, peft_config)
    ref    = build_ref(prev_outdir=None)
    policy.config.use_cache = False
    trainer = DPOContrastiveTrainer(
        model=policy,
        ref_model=ref,
        args=dpo_config,
        train_dataset=ds_dpo,
        eval_dataset=None,
        processing_class=tokenizer,  
        contrastive_weight=0.2 ,
        margin=1.0
                                        
    )
    trainer.train(resume_from_checkpoint=latest_ckpt if latest_ckpt else None)
    trainer.model.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    return  
  
  else:
    split_name = {"low":"train_low", "medium":"train_medium", "high":"train_high"}[args.stage]
    raw = load_dataset("haeunkim/curriculum_learning_dpo_created_negative", split=split_name)
    ds_sft = raw.map(lambda ex: sft_preprocess(ex), batched=True,
                     remove_columns=raw.column_names, load_from_cache_file=True)
    ds_sft.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    if args.stage=='low':
      n = int(len(ds_sft) * 0.4)
      ds_sft.shuffle(seed=42).select(range(n))
    elif args.stage=='medium':
      n = int(len(ds_sft) * 0.2)
      ds_sft.shuffle(seed=42).select(range(n))
    elif args.stage=='high':
      n = int(len(ds_sft) * 0.1)
      ds_sft.shuffle(seed=42).select(range(n))
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    )
    
    base = prepare_model_for_kbit_training(base)
    
   
    
    if args.stage == "low":
        
      sft_ckpt = args.sft_dir if args.sft_dir else f"{args.output_dir}/low"
      if os.path.isdir(sft_ckpt):
        print(sft_ckpt)
        policy = PeftModel.from_pretrained(base, sft_ckpt, is_trainable=True)
        
      else:
        print(f"&&**********{args.stage}****************")
        policy = get_peft_model(base, peft_config)
        
    else:
    
      prev_outdir = f"{args.output_dir}/{'low' if args.stage=='medium' else 'medium'}" 
      print(prev_outdir) 
      policy = PeftModel.from_pretrained(base, prev_outdir, is_trainable=True)
      
    policy.config.use_cache = False
    
    trainer=SFTTrainer(
    model=policy,
    args=sft_config,
    peft_config=peft_config,
    train_dataset=ds_sft,
    processing_class=tokenizer)
    trainer.train(resume_from_checkpoint=latest_ckpt if latest_ckpt else None)
    trainer.model.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    return
if __name__ == "__main__":
    main()

    
  
    
