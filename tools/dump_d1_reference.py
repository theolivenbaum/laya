#!/usr/bin/env python3
"""Dump PyTorch reference tensors for the .NET port of LiquidAI's d1 decision model.

d1 is LFM2.5-VL with a System One readout: the state and a question are rendered as a chat prompt,
the causal LFM2 backbone runs once, and the answer is a softmax over the option tokens at the last
position. The model ships its own code (``modeling_d1.py``, ``hybrid.py``, ``prompt.py``…), which is
what this script runs: it is the behavioural specification for ``Laya.D1``.

For every case it records, in fp32:

  * the rendered prompt(s) and their token ids,
  * the embeddings, every decoder layer's output, the operator / MLP outputs inside the first
    conv layer and the first attention layer, and the final norm,
  * the log-probabilities at the answer slot (full vocabulary) and the option probabilities,
  * the answers ``system_one`` returns.

A single-question case runs the plain causal chain; a multi-question case runs the tree (the state
is the trunk, every question a branch), so both of the reference's code paths are pinned.

    python tools/dump_d1_reference.py --model-dir artifacts/models/d1-3B \
        --out artifacts/dumps/d1 --fixture tests/Laya.Tests/Fixtures/torch-d1.json

writes ``<out>/<case>.safetensors`` (every tensor, whole) and ``<out>/cases.json`` (the inputs,
prompts, ids and answers); ``--fixture`` additionally writes the sampled JSON the test suite reads.
``--threads 1 --time`` reports single-threaded latency, the number the .NET runtime is held to.
"""
import argparse
import json
import os
import sys
import time

import torch
from safetensors.torch import save_file

STATE = "I was charged twice this month, please refund one of them."

TEAM = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {
        "billing": "Charges, refunds, invoices",
        "technical": "App or site faults",
        "fraud": "Suspected unauthorised use",
    },
}
REFUND = {"type": "noul", "instructions": "Is the customer asking for a refund?"}
URGENCY = {
    "type": "score",
    "instructions": "How urgent is this?",
    "criteria": ["Can wait", "Today", "Blocking the customer now"],
}

CASES = {
    # one question: the plain causal chain (`_one_pass`)
    "single": (STATE, {"team": TEAM}),
    # several questions over one state: the tree (trunk + one branch per question)
    "tree": (STATE, {"refund": REFUND, "team": TEAM, "urgency": URGENCY}),
    # a JSON state, rendered with json.dumps(indent=2), a noul with criteria and an unlabelled choice
    "json": (
        {"subject": "Login broken", "body": "Since the update the app crashes when I open settings.\nPlease help!",
         "customer": {"tier": "gold", "seats": 12, "trial": False, "score": 4.5, "tags": ["ios", "beta"]}},
        {"crash": {"type": "noul", "instructions": "Does the app crash?",
                   "criteria": {"true": "The app stops or closes", "false": "It keeps running"}}},
    ),
    "ruling": (
        "Wir haben die Rechnung doppelt bezahlt. Bitte um Rückerstattung.",
        {"lang": {"type": "choice", "instructions": "Which language is this?",
                  "criteria": {"de": None, "en": None, "fr": "French", "es_mx": None}},
         "politeness": {"type": "score", "instructions": "How polite is it?",
                        "criteria": ["Rude", "Neutral", "Polite", "Very polite"]}},
    ),
}


def capture(model):
    """Forward hooks on the language model; returns (store, handles)."""
    lm = model.model.language_model
    store = {}
    handles = []

    def hook(name):
        def fn(_module, _inputs, output):
            value = (output[0] if isinstance(output, tuple) else output).detach().float().clone()
            if name == "embeddings" and name in store:
                # The tree embeds its trunk and its branches in two calls; keep them in row order.
                value = torch.cat((squeeze(store[name]), squeeze(value)))
            store[name] = value
        return fn

    handles.append(lm.embed_tokens.register_forward_hook(hook("embeddings")))
    first_conv = next(i for i, l in enumerate(lm.layers) if l.operator_name == "conv")
    first_attn = next(i for i, l in enumerate(lm.layers) if l.operator_name == "self_attn")
    for i, layer in enumerate(lm.layers):
        handles.append(layer.register_forward_hook(hook(f"layers.{i}.output")))
        if i in (first_conv, first_attn):
            handles.append(layer.operator_norm.register_forward_hook(hook(f"layers.{i}.operator_norm")))
            op = getattr(layer, layer.operator_name)
            handles.append(op.register_forward_hook(hook(f"layers.{i}.operator")))
            if layer.operator_name == "conv":
                handles.append(op.in_proj.register_forward_hook(hook(f"layers.{i}.conv.in_proj")))
            else:
                handles.append(op.q_layernorm.register_forward_hook(hook(f"layers.{i}.attn.q_norm")))
                handles.append(op.k_layernorm.register_forward_hook(hook(f"layers.{i}.attn.k_norm")))
                handles.append(op.v_proj.register_forward_hook(hook(f"layers.{i}.attn.v")))
            handles.append(layer.ffn_norm.register_forward_hook(hook(f"layers.{i}.ffn_norm")))
            handles.append(layer.feed_forward.register_forward_hook(hook(f"layers.{i}.feed_forward")))
    handles.append(lm.embedding_norm.register_forward_hook(hook("final_norm")))
    return store, handles


def squeeze(t):
    return t[0] if t.dim() >= 2 and t.shape[0] == 1 else t


def run_case(model, engine, state, questions):
    from_prompt = sys.modules[type(engine).__module__.replace("runner", "prompt")]
    qs = [from_prompt.as_question(q) for q in questions.values()]
    tok = engine.tokenizer

    prefix = from_prompt.prefix_text(tok, state, engine.bos, engine.state_style, engine.system)
    suffixes = [from_prompt.suffix_text(tok, q, engine.lead, engine.option_style) for q in qs]
    prompts = [prefix + s for s in suffixes]
    readout = [from_prompt.readout_ids(tok, q) for q in qs]
    # Exactly what `SystemOne.run` does: a lone question is the whole prompt encoded at once and
    # run as a chain; several are the prefix encoded once (the trunk) and each suffix on its own.
    if len(qs) == 1:
        trunk, rows = None, [tok.encode(prompts[0], add_special_tokens=False)]
    else:
        trunk = tok.encode(prefix, add_special_tokens=False)
        rows = [tok.encode(s, add_special_tokens=False) for s in suffixes]

    store, handles = capture(model)
    try:
        with torch.inference_mode():
            logz = engine._logz_ids(rows) if trunk is None else engine._tree_logz(trunk, rows)
    finally:
        for h in handles:
            h.remove()

    tensors = {k: squeeze(v).contiguous() for k, v in store.items()}
    probabilities = []
    for i, (q, z) in enumerate(zip(qs, logz)):
        tensors[f"logz.{i}"] = z.float().contiguous()
        probs = engine._readout(q, z)
        probabilities.append(probs)
        tensors[f"probabilities.{i}"] = torch.tensor(probs, dtype=torch.float32)

    result = model.system_one(state, questions)
    meta = {
        "state": state,
        "questions": questions,
        "prefix": prefix,
        "prompts": prompts,
        "ids": rows,
        "trunk": trunk,
        "readout_ids": readout,
        "probabilities": probabilities,
        "answers": result["answers"],
        "usage": result["usage"],
    }
    return tensors, meta


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


def strided_sample(tensor, sample):
    """A sample spread over the whole tensor (first rows, last rows) rather than its first values."""
    t = tensor.detach().float()
    if t.dim() == 2 and t.shape[0] > 2:
        return torch.cat((t[0, :sample // 2], t[-1, :sample // 2]))
    return t.flatten()[:sample]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", default="artifacts/dumps/d1")
    parser.add_argument("--fixture", default=None)
    parser.add_argument("--sample", type=int, default=64)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--time", action="store_true", help="time system_one per case (median of 3)")
    parser.add_argument("--cases", nargs="*", default=None)
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    from transformers import AutoModel

    # Loaded in bf16 (as stored) and widened to fp32 where the text path runs, so the reference
    # computes in fp32 as PyTorch on CPU does, without holding an fp32 copy of the vision tower.
    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True, dtype=torch.bfloat16)
    model.model.language_model.float()
    model.lm_head.float()
    assert model.lm_head.weight.data_ptr() == model.model.language_model.embed_tokens.weight.data_ptr()
    model.eval()
    engine = model.engine

    os.makedirs(args.out, exist_ok=True)
    all_meta = {}
    records = []
    names = args.cases or list(CASES)
    for name in names:
        state, questions = CASES[name]
        tensors, meta = run_case(model, engine, state, questions)
        save_file(tensors, os.path.join(args.out, f"{name}.safetensors"))
        if args.time:
            timings = []
            model.system_one(state, questions)
            for _ in range(3):
                start = time.perf_counter()
                model.system_one(state, questions)
                timings.append(time.perf_counter() - start)
            meta["seconds"] = sorted(timings)[1]
            print(f"{name}: {meta['seconds'] * 1000:.0f} ms (threads={torch.get_num_threads()}, "
                  f"tokens={meta['usage']['input_tokens']})")
        all_meta[name] = meta
        for key, value in tensors.items():
            record = to_record(f"{name}.{key}", value, args.sample)
            record["values"] = [float(v) for v in strided_sample(value, args.sample)]
            records.append(record)
        print(f"{name}: {len(tensors)} tensors; answers {json.dumps(meta['answers'])}")

    with open(os.path.join(args.out, "cases.json"), "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=1)

    if args.fixture:
        with open(args.fixture, "w", encoding="utf-8") as f:
            json.dump({"format": "laya-d1-dump/1", "sample": "first half of row 0, first half of the last row",
                       "cases": all_meta, "tensors": records}, f, ensure_ascii=False)
        print(f"wrote {len(records)} sampled tensors to {args.fixture}")


if __name__ == "__main__":
    main()
