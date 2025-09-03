from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os, re, json, ast, argparse, torch 
from datasets import load_dataset ,concatenate_datasets
import torch.nn.functional as F
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
from peft import LoraConfig, PeftModel, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch
import torch.nn as nn

from trl import DPOTrainer

parse=argparse.ArgumentParser()
parse.add_argument("--stage", required=True, choices=["sample_dpo","sft_lite","curriculum_lite","high","only_dpo",])
parse.add_argument("--model",required=True,choices=['phi','llama'])
parse.add_argument("--output_dir", default="phi_models/phi_curriculum_dpo_sample/2targets_20P_strong")
parse.add_argument("--epochs", type=int, default=1)
parse.add_argument("--lr", type=float, default=5e-6)
parse.add_argument("--beta", type=float, default=0.05)
parse.add_argument("--sft_dir", type=str, default=None)
parse.add_argument("--margin", type=float, default=1.0)
parse.add_argument("--contra_weight", type=float, default=0.2)
parse.add_argument("--two_layer",action="store_true")
parse.add_argument("--proj_dim", type=int, default=None)
parse.add_argument("--concat_forward",action="store_true")
args = parse.parse_args()
if args.model=='phi':  
  MODEL_NAME =  "microsoft/phi-1_5"
else:
  MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

class DPOContrastiveTrainer(DPOTrainer):
  
  def __init__(self,*args, beta: float = 0.1, contrastive_weight: float = 0.2, margin: float = 0.01,two_layer_head:bool =True, concat_forward:bool=False,project_dim:int |None=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.beta = beta
    self.contrastive_weight = contrastive_weight
    self.triplet = torch.nn.TripletMarginLoss(margin=margin, reduction="mean")
    self._pad_id = getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", 0))
    self.concat_forward = concat_forward
    self.two_layer_head = two_layer_head
    
    if project_dim is not None:
      self.attach_proj_head(self.model, dim=project_dim)
  
  def _ensure_mask(self,ids, mask): #mask
    if mask is None:
        return torch.ones_like(ids, dtype=torch.long, device=ids.device)
    return mask
  
  
  def attach_proj_head(self,model,dim=None):
    H = model.config.hidden_size
    D = dim or H
    if self.two_layer_head:
      
      head = nn.Sequential(
              nn.Linear(H,H),
              nn.ReLU(),
              nn.Linear(H,D))
    else:
      
      head = nn.Linear(H,D, bias=False)

    head.to(next(model.parameters()).device).to(dtype=next(model.parameters()).dtype)
    model.proj_head = head   
    
    return model    
    
  def _pad_to_len(self,ids,mask,Lmax): # padding 
    
    B , L = ids.size()
    if L == Lmax:
      return ids, mask
    pad_ids =torch.full((B,Lmax -L),self._pad_id,dtype=ids.dtype,device=ids.device)
    pad_mask = torch.zeros((B, Lmax - L), dtype=mask.dtype, device=mask.device)
    return torch.cat([ids, pad_ids], dim=1), torch.cat([mask, pad_mask], dim=1)
 
  def forward_pool(self,model,ids,mask,):
 
    out = model(input_ids=ids, 
                attention_mask=mask,
                output_hidden_states=True,
                return_dict=True)
    #(3b,l,h)->(3b,h) or (b,l,h)->(b,h)
    last =out.hidden_states[-1]
    #(3b,h)->(3b,h,1)or (b,h)->(b,h,1 or )
    m=mask.float().unsqueeze(-1)
    pooled = (last *m).sum(dim=1)/m.sum(dim=1).clamp_min(1e-9)
    z = model.proj_head(pooled) if hasattr(model, "proj_head") else pooled
    
    return F.normalize(z,p=2,dim=1) 

  def compute_loss(self,model,inputs,return_outputs:bool=False,num_items_in_batch:int |None=None,**kwargs):
    
    try:
      base_loss, base_outputs= super().compute_loss(model ,inputs,return_outputs=True, num_items_in_batch=num_items_in_batch)
    except TypeError:
            base_loss, base_outputs = super().compute_loss(     
                model, inputs, return_outputs=True
            )
    #ANCHOR
    a_ids = inputs.get("prompt_input_ids", inputs.get("input_ids", inputs["chosen_input_ids"]))
    a_ms  = self._ensure_mask(a_ids, inputs.get("prompt_attention_mask", inputs.get("attention_mask")))
    #POSITIVE
    p_ids = inputs["chosen_input_ids"]
    p_ms  = self._ensure_mask(p_ids, inputs.get("chosen_attention_mask"))
    #NEGATIVE
    n_ids = inputs["rejected_input_ids"]
    n_ms  = self._ensure_mask(n_ids, inputs.get("rejected_attention_mask"))
    
    Lmax = max(a_ids.size(1),p_ids.size(1),n_ids.size(1))
    a_ids ,a_ms=self._pad_to_len(a_ids, a_ms, Lmax)
    p_ids ,p_ms =self._pad_to_len(p_ids, p_ms, Lmax)
    n_ids ,n_ms = self._pad_to_len(n_ids,n_ms,Lmax)
    
    if self.concat_forward:
      all_ids = torch.cat([a_ids,p_ids,n_ids],dim=0)
      all_mask =torch.cat([a_ms,p_ms,n_ms],dim=0)
      z=self.forward_pool(model,all_ids,all_mask)
      B=a_ids.size(0)
      a,p,n = torch.split(z,[B,B,B],dim=0)
    
    else:
      a = self.forward_pool(model,a_ids,a_ms)
      p = self.forward_pool(model,p_ids,p_ms)
      n = self.forward_pool(model,n_ids,n_ms)
  
    c_loss = self.triplet(a,p,n)
    total = base_loss +c_loss *self.contrastive_weight
    
    if return_outputs:
      out_dict=dict(base_outputs)
      out_dict['base_loss']=base_loss.detach()
      out_dict['contrastive_loss'] =c_loss.detach()
      return total, out_dict
    return total
    
    
SYSTEM_PROMPT = (
    """You are a span extractor.
Reply with a JSON array of strings only.
Each string must be an exact substring of the INPUT.
Keep left-to-right order and remove duplicates.
If none, return []."""
) 


peft_config = LoraConfig(
      r=8,
      lora_alpha=16,
      lora_dropout=0.1,
      bias="none",
      task_type=TaskType.CAUSAL_LM,
      target_modules=["q_proj", "v_proj"],
      modules_to_save=["proj_head"],
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
  
  outdir = f"{args.output_dir}/{args.stage}"
  
  
  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
)
  sft_config = SFTConfig(
      output_dir=outdir,
      num_train_epochs=args.epochs,
      per_device_train_batch_size=4,
      gradient_accumulation_steps=4,
      report_to="wandb",
      learning_rate=2e-4,
      eval_strategy='steps',
      eval_steps=200,
      logging_steps=10,
      logging_dir="./logs",
      save_strategy="steps",
      save_steps=10,
      save_total_limit=3,
      bf16=False,    
      fp16=False, 
      optim="adamw_torch",
      packing=False,
      #deepspeed="configs/zero_stage_3.json",
      label_names=["labels"],
       save_safetensors=True
  
      
     
  )
  dpo_config = DPOConfig(
          output_dir=outdir,
          per_device_train_batch_size=2,          
          gradient_accumulation_steps=8,           
          num_train_epochs=args.epochs,
          eval_strategy='steps',
          eval_steps=200,
          learning_rate=args.lr,
          beta=args.beta,
          max_prompt_length=128,
          max_length=256,
          save_steps=200,
          save_total_limit=3,
          logging_steps=10,
          remove_unused_columns=False,
          #deepspeed="configs/zero_stage_2.json",
          deepspeed=None,
          lr_scheduler_type="cosine",
          warmup_ratio=0.1,
          report_to="wandb",
          bf16=False,
          fp16=False,
            gradient_checkpointing=False,
           save_safetensors=True
      )
  
  
  
 
  
  #after_sft
  if args.stage =="sft_lite":
    #after sft
    
    ds_low = load_dataset("haeunkim/final_dataset",split='train_low')
    ds_me = load_dataset("haeunkim/final_dataset",split='train_medium')
    ds_high = load_dataset("haeunkim/final_dataset",split='train_high')
    dataset_total=concatenate_datasets([ds_low,ds_me,ds_high])
    ds_eval = load_dataset("haeunkim/final_dataset",split='eval')
    n=int(len(dataset_total)*0.2)
    sft_lite = dataset_total.shuffle(seed=42).select(range(n))
    
    
    ds_sft = sft_lite.map(
    lambda x: sft_preprocess(x), 
    remove_columns=sft_lite.column_names,
    batched=True,
    batch_size=32,
    load_from_cache_file=True
)   
    ds_sft_eval = ds_eval.map(
    lambda x: sft_preprocess(x), 
    remove_columns=ds_eval.column_names,
    batched=True,
    batch_size=32,
    load_from_cache_file=True
)   
    latest_ckpt = get_latest_checkpoint(outdir)
    
    if args.model=='phi':
      base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    else:
      base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config ,torch_dtype=torch.float16)
    base = prepare_model_for_kbit_training(base)
    model =  get_peft_model(base, peft_config)
    model.config.use_cache = False
    
    trainer=SFTTrainer(
    model=model,
    args=sft_config,
    
    peft_config=None,
    train_dataset=ds_sft,
    eval_dataset=ds_sft_eval,
    processing_class=tokenizer)
    trainer.train(resume_from_checkpoint=latest_ckpt if latest_ckpt else None)
    trainer.model.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    
        
    return 
  elif args.stage =="curriculum_lite":
     #curriculum sft
    
    ds_low   = load_dataset("haeunkim/final_dataset",split='train_low')
    ds_me   = load_dataset("haeunkim/final_dataset",split='train_medium')
    ds_high = load_dataset("haeunkim/final_dataset",split='train_high')
    ds_eval= load_dataset("haeunkim/final_dataset",split='eval')
    n_low=int(len(ds_low)*0.4)
    n_me=int(len(ds_me)*0.2)
    n_high = int(len(ds_high)*0.1)
    ds_sft_eval = ds_eval.map(
            sft_preprocess,
            batched=True,
            remove_columns=ds_eval.column_names,
            load_from_cache_file=True,
        )
        
    ds_sft_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    stages = [
        ("low",    ds_low,  n_low,  None),                                 
        ("medium", ds_me,  n_me,  f"{args.output_dir}/curriculum_lite/low"),   
        ("high",   ds_high, n_high, f"{args.output_dir}/curriculum_lite/medium") 
    ]
    for level, raw_ds, n_take, prev_dir in stages:
        if n_take <= 0:
            print(f"[curriculum_lite] skip {level}: n=0")
            continue

        
        subset = raw_ds.shuffle(seed=42).select(range(min(n_take, len(raw_ds))))
        ds_sft = subset.map(
            sft_preprocess,
            batched=True,
            remove_columns=subset.column_names,
            load_from_cache_file=True,
        )
        
        ds_sft.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        if args.model=='phi':
          base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
        else:
           base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config ,torch_dtype=torch.float16)
        base = prepare_model_for_kbit_training(base)
        if prev_dir is None or not os.path.isdir(prev_dir):
            policy = get_peft_model(base, peft_config)
        else:
            policy = PeftModel.from_pretrained(base, prev_dir, is_trainable=True)
        policy.config.use_cache = False

        
        stage_outdir = os.path.join(args.output_dir, "curriculum_lite", level)
        os.makedirs(stage_outdir, exist_ok=True)
       
        latest_ckpt = get_latest_checkpoint(stage_outdir)
        is_high = (level == "high")
        eval_kwargs = (
        dict(eval_strategy="steps", eval_steps=200)
        if is_high else
        dict(eval_strategy="no")
)
        stage_cfg = SFTConfig(
            output_dir=stage_outdir,
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
            fp16=False,
            optim="adamw_torch",
            packing=False,
            label_names=["labels"],
            save_safetensors=True,
            **eval_kwargs,
        )

        # 7) ??
        
        trainer = SFTTrainer(
            model=policy,
            args=stage_cfg,
            peft_config=None,       
            train_dataset=ds_sft,
            eval_dataset=ds_sft_eval if level == "high" else None,
            processing_class=tokenizer,     
        )
        
        trainer.train(resume_from_checkpoint=latest_ckpt or None)

        # 8) ??
        trainer.model.save_pretrained(stage_outdir)
        tokenizer.save_pretrained(stage_outdir)

    return
  
  elif args.stage=='only_dpo':
  
    ds_low = load_dataset("haeunkim/final_dataset",split='train_low')
    ds_me = load_dataset("haeunkim/final_dataset",split='train_medium')
    ds_high = load_dataset("haeunkim/final_dataset",split='train_high')
    ds_eval = load_dataset("haeunkim/final_dataset",split='eval')
    raw=concatenate_datasets([ds_low,ds_me,ds_high])
    ds_dpo = raw.map(lambda ex: dpo_preprocess(ex), batched=True,
                     remove_columns=raw.column_names, load_from_cache_file=True)
    ds_dpo_eval=ds_eval.map(lambda ex: dpo_preprocess(ex), batched=True,
                     remove_columns=ds_eval.column_names, load_from_cache_file=True)
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
        eval_dataset=ds_dpo_eval,
        processing_class=tokenizer,  
        contrastive_weight=args.contra_weight ,
        project_dim=args.proj_dim,
        margin=args.margin,
        concat_forward=args.concat_forward,
        two_layer_head=args.two_layer)
        
                                        
    if args.two_layer:
      head_tag='mlp'
    else:
      head_tag='linear' 
    if args.concat_forward:
      concat_tag = "concat"
    else:
      concat_tag='separate'
    outdir = f"{args.output_dir}/{head_tag}_{concat_tag}/margin_{args.margin}_weight_{args.contra_weight}_epoch_{args.epochs}"
    os.makedirs(outdir, exist_ok=True)
    
    trainer.train()
    trainer.model.save_pretrained(outdir)
   
    tokenizer.save_pretrained(outdir)
    with open(os.path.join(outdir, "train_config.txt"), "w") as f:
      f.write(f"margin={args.margin}\n")
      f.write(f"contra_weight={args.contra_weight}\n")
      f.write(f"epochs={args.epochs}\n")
      f.write(f"learning_rate={args.lr}\n")
      f.write(f"beta={args.beta}\n")
    return  
   
  #after sft
  else:
    ds_low = load_dataset("haeunkim/curriculum_learning_dpo_negative",split='train_low')
    ds_me = load_dataset("haeunkim/curriculum_learning_dpo_negative",split='train_medium')
    ds_high = load_dataset("haeunkim/curriculum_learning_dpo_negative",split='train_high')
    raw=concatenate_datasets([ds_low,ds_me,ds_high])
    ds_dpo = raw.map(lambda ex: dpo_preprocess(ex), batched=True,
                     remove_columns=raw.column_names, load_from_cache_file=True)

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    )
    base = prepare_model_for_kbit_training(base)
    policy = get_peft_model(base, peft_config)
    ref    = build_ref(prev_outdir=None)
    policy.config.use_cache = False
    trainer = DPOTrainer(
        model=policy,
        ref_model=ref,
        args=dpo_config,
        train_dataset=ds_dpo,
        eval_dataset=None,
        processing_class=tokenizer,  
        contrastive_weight=args.contra_weight ,
        project_dim=args.proj_dim,
        margin=args.margin,
        concat_forward=args.concat_forward,
        two_layer_head=args.two_layer)
    trainer.train()
    trainer.model.save_pretrained(outdir)
   
    tokenizer.save_pretrained(outdir)   
if __name__ == "__main__":
    main()

    
  
    
#sft /sample dpo
#sft /curriculum dpo -sample dpo (10%) 
#curriculum sft /sample dpo
#without sft dpo 

