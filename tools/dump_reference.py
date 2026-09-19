#!/usr/bin/env python3
"""Dump PyTorch reference tensors for the .NET port to compare against.

Loads a laya checkpoint with the original Python implementation (kept verbatim under
``.reference/``), runs one state/question pair, and writes:

  * the tokenization of every question,
  * the hidden states after each encoder layer, after each decision-head layer, the option
    logits and the action logits,
  * the final answers.

The layout matches what ``laya dump-states`` writes, so ``tools/compare_dumps.py`` can diff the
two file for file.

    python tools/dump_reference.py --model-dir artifacts/models/english \
        --text "I was charged twice" --preset triage --out artifacts/dumps/torch.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".reference"))

from laya.agent import Agent            # noqa: E402
from laya.common import QTYPES, build_sequence, collate_items, render_options   # noqa: E402

PRESETS = ("triage", "email", "guard", "moderation", "router")


def load_preset(name):
    from laya import presets
    if name == "router":
        return presets.router_questions()
    return getattr(presets, f"{name}_questions")()


def to_record(name, tensor, sample):
    flat = tensor.detach().to(torch.float32).flatten()
    return {
        "name": name,
        "shape": list(tensor.shape),
        "count": int(flat.numel()),
        "mean": float(flat.mean()) if flat.numel() else 0.0,
        "max_abs": float(flat.abs().max()) if flat.numel() else 0.0,
        "values": [float(v) for v in flat[:sample]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", default=None)
    parser.add_argument("--state-file", default=None)
    parser.add_argument("--preset", default=None, choices=PRESETS)
    parser.add_argument("--questions-file", default=None)
    parser.add_argument("--out", default="artifacts/dumps/torch.json")
    parser.add_argument("--sample", type=int, default=64)
    args = parser.parse_args()

    if args.state_file:
        raw = open(args.state_file, encoding="utf-8").read()
        state = json.loads(raw) if args.state_file.endswith(".json") else raw
    else:
        state = args.text
    if state is None:
        parser.error("pass --text or --state-file")

    if args.preset:
        questions = load_preset(args.preset)
    elif args.questions_file:
        questions = json.load(open(args.questions_file, encoding="utf-8"))
    else:
        parser.error("pass --preset or --questions-file")

    agent = Agent(args.model_dir)
    agent.model.eval()
    model = agent.model
    tensors = []

    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)

    for qid, qdef in questions.items():
        q = Agent._to_internal(qdef)
        ids, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
        assert len(markers) == len(render_options(q)), qid

        tensors.append({
            "name": f"input_ids.{qid}", "shape": [len(ids)], "count": len(ids),
            "mean": sum(ids) / len(ids), "max_abs": float(max(ids)),
            "values": [float(v) for v in ids[: args.sample]],
        })
        tensors.append({
            "name": f"marker_pos.{qid}", "shape": [len(markers)], "count": len(markers),
            "mean": sum(markers) / max(1, len(markers)), "max_abs": float(max(markers) if markers else 0),
            "values": [float(v) for v in markers[: args.sample]],
        })

        input_ids = torch.tensor([ids], dtype=torch.long)
        attention = torch.ones_like(input_ids)
        marker_pos = torch.tensor([markers], dtype=torch.long)
        marker_mask = torch.ones_like(marker_pos, dtype=torch.bool)
        qtype = torch.tensor([QTYPES[q["t"]]], dtype=torch.long)

        captured = {}
        handles = []

        def hook(name):
            def fn(_module, _inputs, output):
                captured[name] = output[0] if isinstance(output, tuple) else output
            return fn

        encoder = model.encoder
        handles.append(encoder.embeddings.register_forward_hook(hook("encoder.embeddings")))
        for i, layer in enumerate(encoder.layers):
            handles.append(layer.register_forward_hook(hook(f"encoder.layers.{i}.output")))
        handles.append(encoder.final_norm.register_forward_hook(hook("encoder.last_hidden_state")))

        with torch.no_grad():
            h = encoder(input_ids=input_ids, attention_mask=attention).last_hidden_state
            for handle in handles:
                handle.remove()

            prefix = f"{qid}."
            for name in ["encoder.embeddings"] + [f"encoder.layers.{i}.output" for i in range(len(encoder.layers))] \
                    + ["encoder.last_hidden_state"]:
                if name in captured:
                    tensors.append(to_record(prefix + name, captured[name][0], args.sample))

            h = h + model.type_emb(qtype)[:, None, :]
            tensors.append(to_record(prefix + "head.input", h[0], args.sample))
            pad = ~attention.bool()
            for i, layer in enumerate(model.head.layers):
                h = layer(h, src_key_padding_mask=pad)
                tensors.append(to_record(prefix + f"head.layers.{i}.output", h[0], args.sample))

            idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
            m = torch.gather(h, 1, idx)
            logits = model.scorer(m).squeeze(-1).float()
            logits = logits.masked_fill(~marker_mask, -1e4)
            tensors.append(to_record(prefix + "logits", logits[0], args.sample))

            p = torch.softmax(logits.detach(), -1)
            k = marker_mask.sum(-1).clamp(min=2).float()
            ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
            top2 = p.topk(2, -1).values
            feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
            pooled = h[:, 0].float()
            act_logits = model.act_head(torch.cat([pooled, feats], -1))
            tensors.append(to_record(prefix + "act_logits", act_logits[0], args.sample))

    result = agent.system_one(state, questions)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "format": "laya-state-dump/1",
            # The inputs travel with the dump so a golden fixture cannot silently drift away from
            # the state and questions it was produced for.
            "input": {"state": state, "questions": questions},
            "tensors": tensors,
            "answers": result["answers"],
        }, f)
    print(f"wrote {len(tensors)} tensors to {args.out}")
    print(json.dumps(result["answers"], indent=2)[:2000])


if __name__ == "__main__":
    main()
