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
parse.add_argument("--stage", required=True, choices=["sample_dpo","sft_lite","curriculum_lite","only_dpo_contrastive","only_dpo","sft_dpo","dpo"])
parse.add_argument("--model",required=True,choices=['phi','llama'])
parse.add_argument("--output_dir", default="phi_models/phi_curriculum_dpo_sample/2targets_20P_strong")
parse.add_argument("--epochs", type=int, default=1)
parse.add_argument("--lr", type=float, default=1e-4)
parse.add_argument("--beta", type=float, default=0.05)
parse.add_argument("--sft_dir", type=str, default=None)
parse.add_argument("--margin", type=float, default=1.0)
parse.add_argument("--contra_weight", type=float, default=0.2)

parse.add_argument("--proj_dim", type=int, default=None)

args = parse.parse_args()
if args.model=='phi':  
  MODEL_NAME =  "microsoft/phi-1_5"
else:
  MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
)
class Modelwithhead(nn.Module):
    def __init__(self,base_model):
      super().__init__()
      self.model = base_model
      self.config=self.model.config
      self.warnings_issued = getattr(self.model, "warnings_issued", {})
    
    def forward(self, *args, compute_proj: bool = False, proj_detach: bool = False, **kwargs):
      need_hidden = compute_proj or kwargs.get("output_hidden_states", False)
      kwargs = dict(kwargs)
      kwargs["output_hidden_states"] = need_hidden
      kwargs["return_dict"] = True

      out = self.model(*args, **kwargs)

      if compute_proj:
        attn = kwargs.get("attention_mask", None)
        if attn is None:
          raise RuntimeError("attention_mask is required when compute_proj=True")
        last = out.hidden_states[-1]          # (B, L, H)
        m = attn.float().unsqueeze(-1)        # (B, L, 1)
        pooled = (last * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-9)  # (B, H)
        if proj_detach:
          print("detach")
          pooled = pooled.detach()
        z = self.model.proj_head(pooled)      # (B, D)
        
        setattr(out, "proj", z)  
      return out
    def generate(self, *args, **kwargs):
      return self.model.generate(*args, **kwargs)
    def save_pretrained(self, *args, **kwargs):
        return self.model.save_pretrained(*args, **kwargs)
  
    


class DPOContrastiveTrainer(DPOTrainer):
  
  def __init__(self,*args, beta: float = 0.1, contrastive_weight: float = 0.2, margin: float = 0.01, project_dim:int |None=None, **kwargs):
    model = kwargs.get("model", args[0] if args else None)
    if model is None:
        raise ValueError("model is required")

    
    self.attach_proj_head(model, dim=project_dim)

    kwargs["model"] = model
    super().__init__(*args, **kwargs)

    self.beta = beta
    self.contrastive_weight = contrastive_weight
    self.triplet = torch.nn.TripletMarginLoss(margin=margin, reduction="mean")
    self._pad_id = getattr(self.tokenizer, "pad_token_id", getattr(self.tokenizer, "eos_token_id", 0))
    self.ref = getattr(self, "ref_model", getattr(self, "ref", None))

  
  @staticmethod
  def ensure_mask(ids, mask): #mask
    if mask is None:
        return torch.ones_like(ids, dtype=torch.long, device=ids.device)
    return mask
  
  
  def attach_proj_head(self,model,dim=None):
    m = self._unwrap(model)
    if hasattr(m, "proj_head"):         
        return model
    H = m.config.hidden_size
    D = dim or H
    
      
    head = nn.Sequential(
              nn.Linear(H,H),
              nn.ReLU(),
              nn.Linear(H,D))
    
    p = next(m.parameters())
    head.to(p.device,dtype=p.dtype)
    m.add_module("proj_head", head)
    
    return model    
    
  def _pad_to_len(self,ids,mask,Lmax): # padding 
    
    B , L = ids.size()
    if L == Lmax:
      return ids, mask
    pad_ids =torch.full((B,Lmax -L),self._pad_id,dtype=ids.dtype,device=ids.device)
    pad_mask = torch.zeros((B, Lmax - L), dtype=mask.dtype, device=mask.device)
    return torch.cat([ids, pad_ids], dim=1), torch.cat([mask, pad_mask], dim=1)
 
  def forward_pool(self,model,ids,mask):
  
    out =model(input_ids=ids, 
                attention_mask=mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True)
    #(3b,l,h)->(3b,h) or (b,l,h)->(b,h)
    last =out.hidden_states[-1]
    #(3b,h)->(3b,h,1)or (b,h)->(b,h,1 or )
    m=mask.to(dtype=last.dtype).unsqueeze(-1)
    pooled = (last *m).sum(dim=1)/m.sum(dim=1).clamp_min(1e-9)
    
   #head = getattr(self.model, "proj_head", None)
    #z = head(pooled) if head is not None else pooled
    
    return pooled 
  def _unwrap(self, m):
    return m.module if hasattr(m, "module") else m

  def _get_head(self, model):
    m = self._unwrap(model)
    if hasattr(m,'proj_head'):
      return m.proj_head
    
    raise RuntimeError("proj_head not found on policy model")
  
  def compute_loss(self,model,inputs,return_outputs:bool=False,num_items_in_batch:int |None=None,**kwargs):
    
    try:
      base_loss, base_outputs= super().compute_loss(model ,inputs,return_outputs=True, num_items_in_batch=num_items_in_batch)
    except TypeError:
            base_loss, base_outputs = super().compute_loss(     
                model, inputs, return_outputs=True
            )
    #ANCHOR
    a_ids = inputs.get("prompt_input_ids", inputs.get("input_ids", inputs["chosen_input_ids"]))
    a_ms  = self.ensure_mask(a_ids, inputs.get("prompt_attention_mask", inputs.get("attention_mask")))
    #POSITIVE
    p_ids = inputs["chosen_input_ids"]
    p_ms  = self.ensure_mask(p_ids, inputs.get("chosen_attention_mask"))
    #NEGATIVE
    n_ids = inputs["rejected_input_ids"]
    n_ms  = self.ensure_mask(n_ids, inputs.get("rejected_attention_mask"))
    
    Lmax = max(a_ids.size(1),p_ids.size(1),n_ids.size(1))
    a_ids , a_ms = self._pad_to_len(a_ids, a_ms, Lmax)
    p_ids , p_ms = self._pad_to_len(p_ids, p_ms, Lmax)
    n_ids , n_ms = self._pad_to_len(n_ids,n_ms,Lmax)
    
     # dpo loss
    
    with torch.no_grad():
      a_feat = self.forward_pool(model, a_ids, a_ms)
      p_feat = self.forward_pool(model, p_ids, p_ms)
      n_feat = self.forward_pool(model, n_ids, n_ms)
    head = self._get_head(model) # contrastive loss
    head_dtype = next(head.parameters()).dtype
    a = F.normalize(head(a_feat.to(head_dtype)), p=2, dim=1)
    p = F.normalize(head(p_feat.to(head_dtype)), p=2, dim=1)
    n = F.normalize(head(n_feat.to(head_dtype)), p=2, dim=1)
    c_loss = self.triplet(a, p, n)
    
    total = base_loss + c_loss * self.contrastive_weight
    
    
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
      modules_to_save=["proj_head"] if args.stage=='only_dpo_contrastive' else None
  )

def build_user_text(ex):
    if "prompt" in ex and ex["prompt"]:
        return ex["prompt"].strip()
    inst = (ex.get("instruction") or "").strip()
    inp  = (ex.get("input") or "").strip()
    return f"Instruction: {inst}\nInput: {inp}".strip()



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
    print(target_text)
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
  
  
  
  sft_config = SFTConfig(
      output_dir=outdir,
      num_train_epochs=args.epochs,
      per_device_train_batch_size=4,
      gradient_accumulation_steps=4,
      report_to="wandb",
      learning_rate=2e-4,
      eval_strategy='steps',
      eval_steps=500,
      logging_steps=10,
      logging_dir="./logs",
      save_strategy="steps",
      save_steps=10,
      save_total_limit=2,
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
          per_device_train_batch_size=4 if args.model=='phi' else 1,          
          gradient_accumulation_steps=8,           
          num_train_epochs=args.epochs,
          eval_strategy='steps',
          eval_steps=500,
          learning_rate=args.lr,
          beta=args.beta,
          max_prompt_length=128,
          max_length=256,
          save_steps=10,
          save_total_limit=2,
          logging_steps=10,
          remove_unused_columns=False,
          #deepspeed="configs/zero_stage_2.json",
          #deepspeed="None",
          lr_scheduler_type="cosine",
          warmup_ratio=0.1,
          report_to="wandb",
          bf16=False,
          fp16=True ,
           gradient_checkpointing=False,
    ddp_find_unused_parameters=False,
    ddp_broadcast_buffers=False,
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
    n=int(len(dataset_total)*0.5)
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
      base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
      #base = prepare_model_for_kbit_training(base)
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
    n_low=int(len(ds_low)*0.7)
    n_me=int(len(ds_me)*0.5)
    n_high = int(len(ds_high)*0.3)
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
          base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
          #base = prepare_model_for_kbit_training(base)
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
            fp16=True,
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

    
        trainer.model.save_pretrained(stage_outdir)
        tokenizer.save_pretrained(stage_outdir)

    return
  
  elif args.stage=='only_dpo_contrastive':
    print("contrastive_dpo")
    ds_low = load_dataset("haeunkim/final_dataset",split='train_low')
    ds_me = load_dataset("haeunkim/final_dataset",split='train_medium')
    ds_high = load_dataset("haeunkim/final_dataset",split='train_high')
    ds_eval = load_dataset("haeunkim/final_dataset",split='eval')
    raw_dpo=concatenate_datasets([ds_low,ds_me,ds_high])
    #n=int(len(raw)*0.1)
    #raw_dpo = raw.shuffle(seed=42).select(range(n))
    
    ds_dpo = raw_dpo.map(lambda ex: dpo_preprocess(ex), batched=True,
                     remove_columns=raw_dpo.column_names, load_from_cache_file=True)
    ds_dpo_eval=ds_eval.map(lambda ex: dpo_preprocess(ex), batched=True,
                     remove_columns=ds_eval.column_names, load_from_cache_file=True)
    if args.model=='phi':
      base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    else:
      #base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
      base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config ,torch_dtype=torch.float16)
      base = prepare_model_for_kbit_training(base)
   
    if args.sft_dir:
      policy_base=PeftModel.from_pretrained(base, args.sft_dir, is_trainable=True)
    else:
      policy_base = get_peft_model(base, peft_config)
    policy_base.config.use_cache = False
    
    policy = Modelwithhead(policy_base)
    ref = build_ref(prev_outdir=None)
    dpo_config = DPOConfig(
          output_dir=outdir,
          per_device_train_batch_size=4 if args.model=='phi' else 1,          
          gradient_accumulation_steps=8,           
          num_train_epochs=args.epochs,
          eval_strategy='steps',
          eval_steps=500,
          learning_rate=args.lr,
          beta=args.beta,
          max_prompt_length=128,
          max_length=256,
          save_steps=10,
          save_total_limit=2,
          logging_steps=10,
          remove_unused_columns=False,
          #deepspeed="configs/zero_stage_2.json",
          #deepspeed="None",
          lr_scheduler_type="cosine",
          warmup_ratio=0.1,
          report_to="wandb",
          bf16=True,
          fp16=False ,
           gradient_checkpointing=False,
    ddp_find_unused_parameters=False,
    ddp_broadcast_buffers=False,
           save_safetensors=True
      )
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
        )
        
    
    outdir = f"{args.output_dir}/{args.stage}"
    os.makedirs(outdir, exist_ok=True)
    latest_ckpt = get_latest_checkpoint(f"{args.output_dir}/{args.stage}")
    trainer.train(resume_from_checkpoint=latest_ckpt or None)
    trainer.model.save_pretrained(outdir)
   
    tokenizer.save_pretrained(outdir)
   
    return  
  elif args.stage=='only_dpo':
    print("only_dpo")
    ds_low = load_dataset("haeunkim/final_dataset",split='train_low')
    ds_me = load_dataset("haeunkim/final_dataset",split='train_medium')
    ds_high = load_dataset("haeunkim/final_dataset",split='train_high')
    ds_eval = load_dataset("haeunkim/final_dataset",split='eval')
    raw_dpo=concatenate_datasets([ds_low,ds_me,ds_high])
    
    
    ds_dpo = raw_dpo.map(lambda ex: dpo_preprocess(ex), batched=True,
                     remove_columns=raw_dpo.column_names, load_from_cache_file=True)
    ds_dpo_eval=ds_eval.map(lambda ex: dpo_preprocess(ex), batched=True,
                     remove_columns=ds_eval.column_names, load_from_cache_file=True)
    if args.model=='phi':
      base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    else:
      base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
      #base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config ,torch_dtype=torch.float16)
      #base = prepare_model_for_kbit_training(base)
   
    
    policy_base = get_peft_model(base, peft_config)
    policy_base.config.use_cache = False
   
    ref = build_ref(prev_outdir=None)
    
    trainer = DPOTrainer(
        model=policy_base,
        ref_model=ref,
        args=dpo_config,
        train_dataset=ds_dpo,
        eval_dataset=ds_dpo_eval,
        processing_class=tokenizer,  
       )
        
                                        
   
    outdir = f"{args.output_dir}/{args.stage}"
    os.makedirs(outdir, exist_ok=True)
    latest_ckpt = get_latest_checkpoint(f"{args.output_dir}/{args.stage}")
    trainer.train(resume_from_checkpoint=latest_ckpt or None)
    trainer.model.save_pretrained(outdir)
   
    tokenizer.save_pretrained(outdir)
    
    return  
  
  elif args.stage=='sft_dpo':
    ds_low = load_dataset("haeunkim/final_dataset",split='train_low')
    ds_me = load_dataset("haeunkim/final_dataset",split='train_medium')
    ds_high = load_dataset("haeunkim/final_dataset",split='train_high')
    ds_eval = load_dataset("haeunkim/final_dataset",split='eval')
    
    n_total = len(ds_low) + len(ds_me) +len(ds_high)
    w_low, w_me, w_high = 0.4, 0.35, 0.25  
    n_target =int(n_total *0.2)
    n_low  = int(n_target * w_low)
    n_me   = int(n_target * w_me)
    n_high = int(n_target * w_high)
    
    ds_low_small  = ds_low.shuffle(seed=42).select(range(n_low))
    ds_me_small   = ds_me.shuffle(seed=42).select(range(n_me))
    ds_high_small = ds_high.shuffle(seed=42).select(range(n_high))
    ds_sft_eval=ds_eval.map(lambda ex: sft_preprocess(ex), batched=True,
                     remove_columns=ds_eval.column_names, load_from_cache_file=True)
    stages = [
    ("low",    ds_low_small,  n_low,  None),
    ("medium", ds_me_small,   n_me,   f"{args.output_dir}/{args.stage}/low"),
    ("high",   ds_high_small, n_high, f"{args.output_dir}/{args.stage}/medium")
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
          base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
          #base = prepare_model_for_kbit_training(base)
        if prev_dir is None or not os.path.isdir(prev_dir):
            policy = get_peft_model(base, peft_config)
        else:
            policy = PeftModel.from_pretrained(base, prev_dir, is_trainable=True)
        policy.config.use_cache = False
        try:
          policy.enable_input_require_grads()
          policy.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
        except Exception:
          pass
        
        stage_outdir = os.path.join(args.output_dir, args.stage, level)
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
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            report_to="wandb",
            learning_rate=2e-4,
            logging_steps=10,
            logging_dir="./logs",
            save_strategy="steps",
            save_steps=10,
            save_total_limit=2,
            bf16=False,
            fp16=True,
            optim="adamw_torch",
            packing=False,
            label_names=["labels"],
            save_safetensors=True,
            **eval_kwargs,
            gradient_checkpointing=False,
            ddp_find_unused_parameters=True,
            ddp_broadcast_buffers=False,      
            
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
    sft_final_dir=f"{args.output_dir}/{args.stage}/high"
    raw=concatenate_datasets([ds_low,ds_me,ds_high])
    n_ten=int(len(raw)*0.1)
    raw_dpo = raw.shuffle(seed=42).select(range(n_ten))
    ds_dpo = raw_dpo.map(lambda ex: dpo_preprocess(ex), batched=True,
                     remove_columns=raw.column_names, load_from_cache_file=True)
    ds_dpo_eval=ds_eval.map(lambda ex: dpo_preprocess(ex), batched=True,
                     remove_columns=ds_eval.column_names, load_from_cache_file=True)
    if args.model=='phi':
      base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    )
    else:
      base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,quantization_config=bnb_config
    )  
      base=prepare_model_for_kbit_training(base)
    policy_base =PeftModel.from_pretrained(base, sft_final_dir, is_trainable=True)
    policy_base.config.use_cache = False
    
    policy = Modelwithhead(policy_base)
    ref = build_ref(prev_outdir=None)
    
    dpo_config = DPOConfig(
          output_dir=outdir,
          per_device_train_batch_size=4 if args.model=='phi' else 1,          
          gradient_accumulation_steps=8,           
          num_train_epochs=args.epochs,
          eval_strategy='steps',
          eval_steps=500,
          learning_rate=args.lr,
          beta=args.beta,
          max_prompt_length=128,
          max_length=256,
          save_steps=10,
          save_total_limit=2,
          logging_steps=10,
          remove_unused_columns=False,
          #deepspeed="configs/zero_stage_2.json",
          #deepspeed="None",
          lr_scheduler_type="cosine",
          warmup_ratio=0.1,
          report_to="wandb",
          bf16=True,
          fp16=False ,
           gradient_checkpointing=False,
    ddp_find_unused_parameters=False,
    ddp_broadcast_buffers=False,
           save_safetensors=True,
           max_grad_norm=0.0
      )
  
  
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
       )
        
                                        
    
    outdir = os.path.join(args.output_dir, args.stage, "dpo_contrastive")
    os.makedirs(outdir, exist_ok=True)
    #latest_ckpt = get_latest_checkpoint(f"{args.output_dir}/{args.stage}")
    trainer.train()
    trainer.model.save_pretrained(outdir)
   
    tokenizer.save_pretrained(outdir)
    
      
    return  
  elif args.stage=='dpo':
    print('dpo')
    ds_low = load_dataset("haeunkim/final_dataset",split='train_low')
    ds_me = load_dataset("haeunkim/final_dataset",split='train_medium')
    ds_high = load_dataset("haeunkim/final_dataset",split='train_high')
    ds_eval = load_dataset("haeunkim/final_dataset",split='eval')
    if args.sft_dir:
      sft_final_dir=args.sft_dir
    print(sft_final_dir)
    raw=concatenate_datasets([ds_low,ds_me,ds_high])
    n_ten=int(len(raw)*0.1)
    raw_dpo = raw.shuffle(seed=42).select(range(n_ten))
    ds_dpo = raw_dpo.map(lambda ex: dpo_preprocess(ex), batched=True,
                     remove_columns=raw.column_names, load_from_cache_file=True)
    ds_dpo_eval=ds_eval.map(lambda ex: dpo_preprocess(ex), batched=True,
                     remove_columns=ds_eval.column_names, load_from_cache_file=True)
    
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    )
    policy =PeftModel.from_pretrained(base, sft_final_dir, is_trainable=True)
    policy.config.use_cache = False
    
    
    ref = build_ref(prev_outdir=None)
    
    trainer = DPOTrainer(
        model=policy,
        ref_model=ref,
        args=dpo_config,
        train_dataset=ds_dpo,
        eval_dataset=ds_dpo_eval,
        processing_class=tokenizer,  
        )
        
                                        
    
    outdir = os.path.join(args.output_dir, args.stage, "sft_simple_dpo")
    os.makedirs(outdir, exist_ok=True)
    #latest_ckpt = get_latest_checkpoint(f"{args.output_dir}/{args.stage}")
    trainer.train()
    trainer.model.save_pretrained(outdir)
   
    tokenizer.save_pretrained(outdir)
        
if __name__ == "__main__":
    main()

    
  
    


