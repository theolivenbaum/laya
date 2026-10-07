#!/usr/bin/env python3
"""Time the PyTorch reference of d1 on CPU, so the .NET runtime has a number to be held to.

Runs ``model.system_one`` the way the model card does on CPU (fp32), warm, and prints the median of
``--iterations`` calls per case. ``--threads 1`` is the comparison CLAUDE.md asks for: a parallel
measurement hides a kernel problem behind memory bandwidth.

    python tools/bench_d1.py --model-dir artifacts/models/d1-3B --threads 1

The cases are the ones ``laya bench --model-dir artifacts/models/d1-3B --bench-d1`` runs.
"""
import argparse
import json
import os
import statistics
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def cases():
    with open(os.path.join(HERE, "d1_bench_cases.json"), encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--flush-denormal", action="store_true",
                        help="set flush-to-zero first; the reference does not, and d1's conv weights are a third subnormal")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.flush_denormal:
        torch.set_flush_denormal(True)

    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True, dtype=torch.bfloat16)
    model.model.language_model.float()
    model.lm_head.float()
    model.eval()

    for name, case in cases().items():
        if args.cases and name not in args.cases:
            continue
        state, questions = case["state"], case["questions"]
        result = model.system_one(state, questions)
        timings = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            model.system_one(state, questions)
            timings.append(time.perf_counter() - start)
        print(f"{name:12s} {len(questions)} question(s) {result['usage']['input_tokens']:5d} tokens  "
              f"median {statistics.median(timings) * 1000:8.1f} ms  (threads={torch.get_num_threads()})", flush=True)


if __name__ == "__main__":
    main()
