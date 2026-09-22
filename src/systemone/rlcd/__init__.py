"""RLCD for LLMs: fine-tune the label distribution systemone reads at the answer slot.

RLCD (as in Laya's fine-tuning notebook) treats the model's label logits z as a
policy whose action is a *reported distribution* q = softmax(z + eps), eps ~ N(0, s^2),
rewards it with a strictly proper scoring rule against the gold distribution
(log score + spherical score - ranked probability score for ordinal questions),
normalises rewards within a group of G draws (GRPO-style), and adds a soft
cross-entropy term. Here the same loss is applied to an LLM's next-token
distribution over the answer labels, with LoRA.

  systemone rlcd prepare --upstream URL --out train.jsonl      # exact serving tokens
  systemone rlcd train --model google/gemma-4-31B-it --data train.jsonl --out adapter/
"""
