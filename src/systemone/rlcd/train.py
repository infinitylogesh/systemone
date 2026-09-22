"""RLCD fine-tuning of an LLM's answer-label distribution, on top of TRL + PEFT.

The policy is the model's next-token logits restricted to a question's label
tokens, read at the answer slot (the last prompt position). Objectives:

  rlcd    Laya's loss: G noisy reported distributions q = softmax(z + eps), a strictly
          proper reward R(q, target) = log score + w_sph * spherical - w_rps * RPS,
          group-normalised advantage, REINFORCE through the Gaussian log-density,
          plus ce_weight * soft cross-entropy.
  proper  the same reward maximised directly by gradient (R is differentiable)
  ce      soft cross-entropy (the log score) alone

Why TRL's GRPOTrainer isn't used as-is: GRPO samples *token sequences* and scores
text, while RLCD's action is a continuous distribution over labels scored by a
proper scoring rule. A sampled label scored against the target rewards
argmax-style overconfidence (the expected reward sum_i p_i t_i is not proper). So
RLCD is a custom loss on TRL's SFTTrainer: TRL/transformers still provide the
training loop, PEFT/LoRA, gradient checkpointing, accelerate/DeepSpeed and logging.
"""

import argparse
import json
import math
import os

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

QTYPE = {"choice": 0, "score": 1, "noul": 2}


def proper_reward(q, target, is_score, mask, w_sph=0.75, w_rps=1.0, log_floor=-9.21):
    """Strictly proper scoring rule reward (higher is better); q, target: [..., B, K]."""
    q = q * mask
    log_score = (target * torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    k = mask.sum(-1).clamp(min=2).float()
    rps = (((torch.cumsum(q, -1) - torch.cumsum(target, -1)) ** 2) * mask).sum(-1) / (k - 1)
    return r - w_rps * rps * is_score


class Collator:
    """Left-pad so every row's answer slot is the last position."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        L = max(len(r["input_ids"]) for r in rows)
        K = max(len(r["label_ids"]) for r in rows)
        B = len(rows)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        att = torch.zeros((B, L), dtype=torch.long)
        lab = torch.zeros((B, K), dtype=torch.long)
        mask = torch.zeros((B, K), dtype=torch.bool)
        tgt = torch.zeros((B, K), dtype=torch.float32)
        for i, r in enumerate(rows):
            n, k = len(r["input_ids"]), len(r["label_ids"])
            ids[i, L - n:] = torch.tensor(r["input_ids"])
            att[i, L - n:] = 1
            lab[i, :k] = torch.tensor(r["label_ids"])
            mask[i, :k] = True
            tgt[i, :k] = torch.tensor(r["target"], dtype=torch.float32)
        qt = torch.tensor([QTYPE[r["qtype"]] for r in rows])
        return {"input_ids": ids, "attention_mask": att, "label_ids": lab, "label_mask": mask, "target": tgt, "qtype": qt}


class RLCDTrainer(SFTTrainer):
    def __init__(self, *a, objective="rlcd", group=4, sigma=(0.4, 0.1), ce_weight=1.0, w_sph=0.75, w_rps=1.0, **kw):
        super().__init__(*a, **kw)
        self.objective, self.group, self.sigma = objective, group, sigma
        self.ce_weight, self.w_sph, self.w_rps = ce_weight, w_sph, w_rps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], logits_to_keep=1, use_cache=False)
        logits = out.logits[:, -1, :].float()  # the answer slot
        mask = inputs["label_mask"]
        z = logits.gather(1, inputs["label_ids"]).masked_fill(~mask, -1e4)
        target = inputs["target"]
        is_score = (inputs["qtype"] == QTYPE["score"]).float()
        logq = torch.log_softmax(z, -1)
        ce = -(target * logq.masked_fill(~mask, 0)).sum(-1).mean()

        if self.objective == "ce":
            loss = ce
        elif self.objective == "proper":
            loss = -proper_reward(logq.exp(), target, is_score, mask, self.w_sph, self.w_rps).mean()
        else:  # rlcd
            progress = self.state.global_step / max(1, self.state.max_steps)
            s = self.sigma[0] + (self.sigma[1] - self.sigma[0]) * progress
            k = mask.sum(-1, keepdim=True).float()
            eps = torch.randn((self.group,) + z.shape, device=z.device) * s * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask  # zero-mean over the labels
            zs = z.detach().unsqueeze(0) + eps
            q = torch.softmax(zs.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), is_score, mask, self.w_sph, self.w_rps)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((zs - z.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * s * s)
            loss = -(adv * logp).mean() + self.ce_weight * ce
            if self.state.global_step % max(1, self.args.logging_steps) == 0:
                self.log({"reward": float(r.mean()), "sigma": s})
        self._metrics_ce = float(ce.detach())
        return (loss, out) if return_outputs else loss


def model_class(path):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    cfg = AutoConfig.from_pretrained(path)
    return AutoModelForImageTextToText if getattr(cfg, "vision_config", None) is not None else AutoModelForCausalLM


def main(argv=None):
    ap = argparse.ArgumentParser(prog="systemone rlcd train")
    ap.add_argument("--model", required=True, help="HF id or path of the base model (bf16 weights)")
    ap.add_argument("--data", required=True, help="records from `systemone rlcd prepare`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--objective", choices=["rlcd", "proper", "ce"], default="rlcd")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--sigma", type=float, nargs=2, default=(0.4, 0.1))
    ap.add_argument("--ce-weight", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    rows = [json.loads(line) for line in open(a.data)]
    rows = [r for r in rows if len(r["input_ids"]) <= a.max_len]
    if a.limit:
        rows = rows[: a.limit]
    ds = Dataset.from_list(rows)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = model_class(a.model).from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0},
                                                 attn_implementation="sdpa")
    # LoRA on the language model's projections only (not the vision tower)
    lora = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
                      if model_class(a.model).__name__.endswith("ImageTextToText")
                      else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    steps = math.ceil(len(rows) / (a.batch * a.accum) * a.epochs)
    cfg = SFTConfig(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=max(1, steps // 20),
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, save_strategy="no", report_to=[], remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True}, max_grad_norm=1.0, seed=0,
    )
    trainer = RLCDTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora,
                          data_collator=Collator(tok.pad_token_id), objective=a.objective, group=a.group,
                          sigma=tuple(a.sigma), ce_weight=a.ce_weight)
    print(f"RLCD: objective={a.objective} records={len(rows)} optimizer steps~{steps}", flush=True)
    trainer.train()
    trainer.model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    json.dump(vars(a), open(os.path.join(a.out, "rlcd_args.json"), "w"), indent=1)
    print(f"saved LoRA adapter to {a.out}", flush=True)


if __name__ == "__main__":
    main()
