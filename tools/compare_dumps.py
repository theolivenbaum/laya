#!/usr/bin/env python3
"""Diff a PyTorch reference dump against a .NET dump, tensor by tensor.

    python tools/compare_dumps.py artifacts/dumps/torch.json artifacts/dumps/dotnet.json

Reports, for every tensor present in both files, the maximum absolute and relative difference over
the sampled values plus the difference in the whole-tensor mean and max magnitude. Exits non-zero
when anything exceeds the tolerance, so it can gate a build.
"""
import argparse
import json
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {t["name"]: t for t in data["tensors"]}, data.get("answers")


def compare_answers(reference, candidate, atol):
    """The dumps carry the final answers too; a match there is what actually matters."""
    failures = []
    for qid, expected in reference.items():
        got = candidate.get(qid)
        if got is None:
            failures.append(f"answers.{qid}: missing from the candidate")
            continue
        for key in ("type", "choice"):
            if expected.get(key) != got.get(key):
                failures.append(f"answers.{qid}.{key}: {expected.get(key)!r} vs {got.get(key)!r}")
        for key in ("score", "noul", "confidence"):
            a, b = expected.get(key), got.get(key)
            if a is None and b is None:
                continue
            if a is None or b is None or abs(a - b) > atol:
                failures.append(f"answers.{qid}.{key}: {a} vs {b}")
        for option, probability in (expected.get("probabilities") or {}).items():
            other = (got.get("probabilities") or {}).get(option)
            if other is None or abs(probability - other) > atol:
                failures.append(f"answers.{qid}.probabilities[{option}]: {probability} vs {other}")
    print(f"compared {len(reference)} answers")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    reference, reference_answers = load(args.reference)
    candidate, candidate_answers = load(args.candidate)

    shared = [n for n in reference if n in candidate]
    missing = [n for n in reference if n not in candidate]
    failures = []

    print(f"{'tensor':<48} {'shape':<14} {'max abs':>11} {'max rel':>11} {'mean Δ':>11}")
    for name in shared:
        a, b = reference[name], candidate[name]
        if a["shape"] != b["shape"]:
            failures.append(f"{name}: shape {a['shape']} vs {b['shape']}")
            continue

        n = min(len(a["values"]), len(b["values"]))
        max_abs = max_rel = 0.0
        for i in range(n):
            x, y = a["values"][i], b["values"][i]
            d = abs(x - y)
            max_abs = max(max_abs, d)
            max_rel = max(max_rel, d / max(abs(x), 1e-6))
        mean_delta = abs(a["mean"] - b["mean"])

        scale = max(abs(a["max_abs"]), 1e-6)
        ok = max_abs <= args.atol * max(1.0, scale) and (max_rel <= args.rtol or max_abs <= args.atol)
        flag = "" if ok else "   <-- FAIL"
        if not ok:
            failures.append(f"{name}: max_abs={max_abs:.6g} max_rel={max_rel:.6g}")
        if not args.quiet or not ok:
            print(f"{name:<48} {'x'.join(map(str, a['shape'])):<14} "
                  f"{max_abs:>11.3e} {max_rel:>11.3e} {mean_delta:>11.3e}{flag}")

    if reference_answers and candidate_answers:
        failures.extend(compare_answers(reference_answers, candidate_answers, args.atol))

    print()
    print(f"compared {len(shared)} tensors, {len(missing)} reference tensors missing from the candidate")
    if missing and not args.quiet:
        print("  missing:", ", ".join(missing[:10]))
    if failures:
        print(f"{len(failures)} tensor(s) outside tolerance:")
        for failure in failures[:20]:
            print("  " + failure)
        return 1
    print("all tensors within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
