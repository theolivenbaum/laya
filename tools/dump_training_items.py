"""Training-item reference: the notebook's `build_training_item` over the first cases of a split.

    python tools/dump_training_items.py --model-dir artifacts/models/english \
        --data artifacts/data/LocalLLaMA_typed-decisions.all.train.jsonl --cases 10 \
        --out tests/Laya.Tests/Fixtures/training-items-english.json

`TrainingItemParityTests` builds the same items with `DecisionDataset.BuildItems` and compares token ids,
marker positions and targets.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".reference"))

from laya.common import QTYPES, build_sequence, render_options  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cases", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config

    _fix_tokenizer_config(args.model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))
    with open(os.path.join(args.model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    # --- verbatim from the notebook's cell 6
    def build_training_item(state, q, gold_q):
        t = q["type"]
        crit = q.get("criteria", {})
        if t == "choice":
            keys = list(crit.keys())
            target = [gold_q["probabilities"].get(k, 0.0) for k in keys]
        elif t == "noul":
            target = [gold_q["probabilities"].get("false", 0.5), gold_q["probabilities"].get("true", 0.5)]
        elif t == "score":
            n_levels = len(crit) if isinstance(crit, list) else 4
            target = [gold_q["probabilities"].get(str(i), 0.0) for i in range(n_levels)]
        s = sum(target)
        target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
        label = target.index(max(target))
        k = len(render_options({"t": t, "crit": crit}))
        seq, markers = build_sequence(tok, state, {"t": t, "ins": q["instructions"], "crit": crit},
                                      cfg["max_len"], cfg["head_max_len"])
        if len(markers) != k:
            return None
        return {"ids": seq, "markers": markers, "qtype": QTYPES[t], "target": target, "label": label}
    # ---

    items = []
    with open(args.data) as f:
        for n, line in enumerate(f):
            if n >= args.cases:
                break
            row = json.loads(line)
            state, questions, gold = (json.loads(row[k]) if isinstance(row[k], str) else row[k]
                                      for k in ("state", "questions", "gold"))
            for qid, q in questions.items():
                if qid in gold:
                    it = build_training_item(state, q, gold[qid])
                    if it:
                        it["case"], it["question"] = row["id"], qid
                        items.append(it)
    with open(args.out, "w") as f:
        json.dump({"max_len": cfg["max_len"], "head_max_len": cfg["head_max_len"], "items": items}, f)
    print("wrote", len(items), "items to", args.out)


if __name__ == "__main__":
    main()
