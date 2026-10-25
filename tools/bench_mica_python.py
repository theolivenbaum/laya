#!/usr/bin/env python3
"""Time the reference Python implementation of Mica's forward pass on the CPU, for comparison with the C# port.

The reference is transformers' Qwen3.5 (the "plain" profile of Mica's runtime), timed on exactly the token ids a
Mica dump read (``artifacts/dumps/<name>/reference.json``), forward pass only — weight loading is excluded, and so
is the LM head, which Mica reads at a handful of label rows:

``bf16``  the whole text model resident in bf16, as one would run it on a CPU with 16 GB.
``fp32``  fp32 compute as the parity reference does it: one decoder layer resident at a time (4B fp32 parameters
          do not fit next to anything else), only the layer forwards are timed and summed.

    python tools/bench_mica_python.py --model-dir artifacts/models/mica --dumps artifacts/dumps --mode bf16

``--requests a.json b.json --mica-repo <checkout of github.com/akivet/Mica>`` times whole /v1/systemone requests
instead: every question row (multi-answer choices expanded, as Mica's contract does) rendered with Mica's own
``prompt_for`` and read in full, one forward per row — what the Python reference does without a cache.
"""
import argparse
import gc
import json
import os
import statistics
import time

import torch
from safetensors import safe_open
from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


def load_request_ids(paths, mica_repo, model_dir):
    import sys
    sys.path.insert(0, mica_repo)
    from mica import contract
    from mica.native import prompt_for
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    cases = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            request = json.load(f)
        rows = contract.rows_from_request(request)
        rows = [r for r in rows if not r.get("multi")] + [x for r in rows if r.get("multi") for x in contract.inclusion_rows(r)]
        name = os.path.splitext(os.path.basename(path))[0]
        for i, row in enumerate(rows):
            cases[f"{name}#{i}"] = tokenizer(prompt_for(row, tokenizer), add_special_tokens=False, return_tensors="pt").input_ids
    return cases


def load_ids(dumps):
    cases = {}
    for name in sorted(os.listdir(dumps)):
        path = os.path.join(dumps, name, "reference.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                cases[name] = torch.tensor([json.load(f)["input_ids"]])
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dumps", required=True)
    parser.add_argument("--mode", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--requests", nargs="*", help="time whole requests (one forward per question row) instead of dumps")
    parser.add_argument("--mica-repo", help="a checkout of github.com/akivet/Mica, for --requests")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    with open(os.path.join(args.model_dir, "config.json")) as f:
        config = Qwen3_5TextConfig(**json.load(f)["text_config"])
    config._attn_implementation = "sdpa"
    with open(os.path.join(args.model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    handles = {}

    def tensor(name, dtype):
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(args.model_dir, shard), framework="pt")
        return handles[shard].get_tensor(name).to(dtype)

    prefix = "model.language_model."
    cases = load_request_ids(args.requests, args.mica_repo, args.model_dir) if args.requests else load_ids(args.dumps)
    results = {}
    with torch.inference_mode():
        if args.mode == "bf16":
            torch.set_default_dtype(torch.bfloat16)          # build in bf16: an fp32 build alone is 17 GB
            model = qm.Qwen3_5TextModel(config).eval()
            torch.set_default_dtype(torch.float32)
            state = {k[len(prefix):]: tensor(k, torch.bfloat16) for k in weight_map if k.startswith(prefix)}
            model.load_state_dict(state, strict=False)        # the rotary inv_freq buffers are not in the checkpoint
            del state
            gc.collect()
            model(input_ids=next(iter(cases.values()))[:, :16])        # warm up
            for name, ids in cases.items():
                times = []
                for _ in range(args.repeat):
                    t = time.perf_counter()
                    model(input_ids=ids, use_cache=False)
                    times.append(time.perf_counter() - t)
                results[name] = (ids.shape[1], times)
                print(f"{name}: {ids.shape[1]} tokens, median {statistics.median(times):.2f} s", flush=True)
        else:
            embed = tensor(prefix + "embed_tokens.weight", torch.float32)
            totals = {name: [0.0] * args.repeat for name in cases}
            hidden = {name: torch.nn.functional.embedding(ids, embed) for name, ids in cases.items()}
            rotary = qm.Qwen3_5TextRotaryEmbedding(config)
            for i in range(config.num_hidden_layers):
                layer = qm.Qwen3_5DecoderLayer(config, i).eval()
                layer.load_state_dict({k[len(f"{prefix}layers.{i}."):]: tensor(k, torch.float32)
                                       for k in weight_map if k.startswith(f"{prefix}layers.{i}.")})
                full = config.layer_types[i] == "full_attention"
                for name, ids in cases.items():
                    n = ids.shape[1]
                    pos = torch.arange(n).view(1, 1, -1).expand(3, 1, -1)
                    emb = rotary(hidden[name], pos)
                    mask = torch.full((n, n), float("-inf")).triu(1)[None, None] if full else None
                    for r in range(args.repeat):
                        t = time.perf_counter()
                        out = layer(hidden[name], position_embeddings=emb, attention_mask=mask)
                        totals[name][r] += time.perf_counter() - t
                    hidden[name] = out
                del layer
                gc.collect()
            for name, ids in cases.items():
                results[name] = (ids.shape[1], totals[name])
                print(f"{name}: {ids.shape[1]} tokens, median {statistics.median(totals[name]):.2f} s", flush=True)

    out = os.path.join(args.dumps, f"python-{args.mode}{'-requests' if args.requests else ''}-timing.json")
    with open(out, "w") as f:
        json.dump({"mode": args.mode, "threads": args.threads, "torch": torch.__version__,
                   "cases": {k: {"tokens": n, "seconds": t, "median": statistics.median(t)} for k, (n, t) in results.items()}}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
