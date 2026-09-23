"""Gradient reference for the .NET trainer.

Builds the reference `DecisionModel` (`.reference/laya/common.py`) on a tiny random ModernBERT, runs
one padded batch through it in eval mode (no dropout) with autograd on, and backpropagates a fixed
linear function of the option logits. Writes a checkpoint directory the .NET side loads, plus the
logits and every parameter's gradient, so `TrainingGradientTests` can compare them tensor by tensor.

    python tools/dump_training_reference.py --out tests/Laya.Tests/Fixtures/training-tiny

Needs torch, transformers and safetensors. The checkpoint is saved in fp32, so the comparison measures
the backward pass rather than fp16 rounding.
"""
import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".reference"))

from laya.common import DecisionModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--checkpoint", help="instead: gradients of a real checkpoint on a few items (a subset of tensors)")
    ap.add_argument("--data", help="typed-decisions JSON lines, with --checkpoint")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.checkpoint:
        return dump_checkpoint(args)

    from transformers import ModernBertConfig, ModernBertModel

    cfg = ModernBertConfig(
        vocab_size=97, hidden_size=128, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
        global_attn_every_n_layers=3, local_attention=8, max_position_embeddings=256,
        attention_dropout=0.0, mlp_dropout=0.0, embedding_dropout=0.0,
        norm_bias=False, attention_bias=False, mlp_bias=False,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2,
    )
    encoder = ModernBertModel._from_config(cfg, attn_implementation="sdpa")
    model = DecisionModel(encoder, head_layers=2, n_act=2)

    # Every parameter gets a non-trivial value (LayerNorm weights included), so a wrong index or a
    # missing term shows up as a mismatch instead of hiding behind ones and zeros.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1 and "norm" in name:
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            else:
                p.copy_(0.08 * torch.randn_like(p))
    model.eval()

    lengths = [21, 13, 9]
    markers = [[3, 5, 8, 11], [2, 6], [1, 4, 6]]
    qtypes = [0, 1, 2]
    n, L, K = len(lengths), max(lengths), max(len(m) for m in markers)
    ids = torch.zeros((n, L), dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, K), dtype=torch.long)
    mmask = torch.zeros((n, K), dtype=torch.bool)
    coeff = torch.zeros((n, K))
    for i, (length, m) in enumerate(zip(lengths, markers)):
        ids[i, :length] = torch.randint(3, cfg.vocab_size, (length,))
        att[i, :length] = 1
        mpos[i, : len(m)] = torch.tensor(m)
        mmask[i, : len(m)] = True
        coeff[i, : len(m)] = torch.randn(len(m))

    logits, _ = model(ids, att, mpos, mmask, torch.tensor(qtypes))
    loss = (logits * coeff * mmask).sum()
    loss.backward()

    os.makedirs(os.path.join(args.out, "encoder"), exist_ok=True)
    state = {k: v.detach().float().contiguous() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(args.out, "model.safetensors"))
    cfg.save_pretrained(os.path.join(args.out, "encoder"))
    with open(os.path.join(args.out, "rl_agent_config.json"), "w") as f:
        json.dump({"encoder": "tiny", "head_layers": 2, "max_len": 64, "head_max_len": 32,
                   "act_costs": {"escalate": 0.5}, "temperature": [1.0, 1.0, 1.0]}, f, indent=2)

    grads = {name: p.grad.detach().float().contiguous() for name, p in model.named_parameters() if p.grad is not None}
    save_file(grads, os.path.join(args.out, "grads.safetensors"))
    with open(os.path.join(args.out, "batch.json"), "w") as f:
        json.dump({
            "items": [
                {"ids": ids[i, : lengths[i]].tolist(), "markers": markers[i], "qtype": qtypes[i],
                 "coeff": coeff[i, : len(markers[i])].tolist(), "logits": logits[i, : len(markers[i])].tolist()}
                for i in range(n)
            ],
            "loss": float(loss.detach()),
        }, f, indent=2)
    print("wrote", args.out, "with", len(grads), "gradients; loss", float(loss.detach()))
    dump_objective(args.out)


def dump_objective(out):
    """The notebook's RLCD loss on fixed logits, with the Gaussian draws written out so .NET can reuse them."""
    from laya.common import QTYPES, proper_reward

    ks, qtypes, G, sigma = [4, 2, 3], [0, 2, 1], 4, 0.3
    n, K = len(ks), max(ks)
    mask = torch.zeros((n, K), dtype=torch.bool)
    target = torch.zeros((n, K))
    logits = torch.full((n, K), -1e4)
    for i, k in enumerate(ks):
        mask[i, :k] = True
        target[i, :k] = torch.softmax(torch.randn(k) * 2, -1)
        logits[i, :k] = torch.randn(k) * 1.5
    logits.requires_grad_(True)
    draws = torch.randn((G, n, K))

    # --- verbatim from train_ddp.py, with torch.randn replaced by the recorded draws
    k = mask.sum(-1, keepdim=True).float()
    eps = draws * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), torch.tensor(qtypes), mask, w_sph=0.75, w_rps=1.0)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    loss = loss_rl + 1.0 * loss_ce
    # ---
    loss.backward()
    with open(os.path.join(out, "objective.json"), "w") as f:
        json.dump({
            "sigma": sigma, "qtypes": qtypes, "ks": ks,
            "logits": [logits[i, :ks[i]].tolist() for i in range(n)],
            "target": [target[i, :ks[i]].tolist() for i in range(n)],
            "draws": [[draws[g, i, :ks[i]].tolist() for i in range(n)] for g in range(G)],
            "reward_mean": float(r.mean()), "loss": float(loss), "loss_rl": float(loss_rl), "loss_ce": float(loss_ce),
            "grad": [logits.grad[i, :ks[i]].tolist() for i in range(n)],
        }, f, indent=2)



GRAD_SUBSET = [
    "encoder.embeddings.norm.weight", "encoder.layers.0.attn.Wqkv.weight", "encoder.layers.1.mlp.Wi.weight",
    "encoder.layers.14.attn.Wo.weight", "encoder.layers.27.attn.Wqkv.weight", "encoder.layers.27.mlp.Wo.weight",
    "encoder.final_norm.weight", "head.layers.0.self_attn.in_proj_weight", "head.layers.1.linear2.bias",
    "type_emb.weight", "scorer.1.weight", "scorer.3.bias",
]


def dump_checkpoint(args):
    """Gradients of a real checkpoint for two typed-decisions items, eval mode, fp32."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config
    from laya.common import QTYPES, build_model, build_sequence, collate_items

    _fix_tokenizer_config(args.checkpoint)
    tok = AutoTokenizer.from_pretrained(os.path.join(args.checkpoint, "tokenizer"))
    with open(os.path.join(args.checkpoint, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    model = build_model(cfg, encoder_dir=os.path.join(args.checkpoint, "encoder"))
    model.load_state_dict({k: v.float() for k, v in load_file(os.path.join(args.checkpoint, "model.safetensors")).items()})
    model.float().eval()

    with open(args.data) as f:
        row = json.loads(f.readline())
    state, questions = json.loads(row["state"]), json.loads(row["questions"])
    items, coeffs = [], []
    for qid in list(questions)[:2]:
        q = questions[qid]
        ids, markers = build_sequence(tok, state, {"t": q["type"], "ins": q["instructions"], "crit": q.get("criteria")},
                                      cfg["max_len"], cfg["head_max_len"])
        items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["type"]]})
        coeffs.append(torch.randn(len(markers)).tolist())

    n, L, K = len(items), max(len(i["ids"]) for i in items), max(len(i["markers"]) for i in items)
    ids = torch.full((n, L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, K), dtype=torch.long)
    mmask = torch.zeros((n, K), dtype=torch.bool)
    coeff = torch.zeros((n, K))
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        mpos[i, : len(it["markers"])] = torch.tensor(it["markers"])
        mmask[i, : len(it["markers"])] = True
        coeff[i, : len(it["markers"])] = torch.tensor(coeffs[i])
    logits, _ = model(ids, att, mpos, mmask, torch.tensor([it["qtype"] for it in items]))
    (logits * coeff * mmask).sum().backward()

    params = dict(model.named_parameters())
    os.makedirs(args.out, exist_ok=True)
    save_file({k: params[k].grad.float().contiguous() for k in GRAD_SUBSET}, os.path.join(args.out, "grads-subset.safetensors"))
    with open(os.path.join(args.out, "batch.json"), "w") as f:
        json.dump({"items": [dict(it, coeff=coeffs[i], logits=logits[i, : len(it["markers"])].tolist())
                             for i, it in enumerate(items)]}, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
