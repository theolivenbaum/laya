# JevBench, run natively against laya's .NET `Agent`

[JevBench](https://github.com/theolivenbaum/jevbench) is Benchmark Heaven's benchmark for
"Jev-class" decision models: hand the model a piece of state and a bounded rubric, get back a
typed, calibrated answer. Laya was one of the systems measured in JevBench v1.2.2, through the
Python package (`jevbench/adapters/laya_local.py` calling `pip install laya`). This directory
reimplements that same benchmark natively in C#, calling this repository's own `Laya.Agent`
in-process — no Python, and no code from the jevbench repository is executed by this tool.

## What's here

```
data/                    copied from jevbench's public dataset (MIT; see JEVBENCH-LICENSE.txt)
  easy.jsonl              48 items -> the "easy" tier
  original.jsonl          72 items -> the "standard" tier's public half (96 total; 24 held out)
  hard.jsonl              111 items -> the "hard" tier's public half (220 total; 109 held out)
  manifest.json           dataset provenance + sha256 hashes, copied from jevbench/datasets/manifest.json
  baseline/laya.json              jevbench's own published Laya row (v1.2.2, python `laya` 0.3.3)
  baseline/laya-per-task.json     …and its public-item-only per-task outcomes, for this tool to diff against
Dataset/                 task record model + JSONL loader (port of jevbench/tasks.py)
                          and the JevQuestion -> Laya.Runtime.Question builder (port of
                          jevbench/adapters/base.py:build_question)
Scoring/                 ports of jevbench/scoring.py, metrics.py and composite_v13.py
Runner.cs                runs one task through Agent.Predict — the port of adapters/laya_local.py
Reporting/                tier/calibration/speed/cost aggregation + baseline diffing + JSON output
Program.cs                `run` command
```

**Note on the "judge" tier: zero public items exist for it.** Its two source cohorts (`router`,
`judge`, 146 decisions in total) are entirely held out — see `data/manifest.json`. This tool can
therefore only ever score `easy` (48/48 public), `standard` (72/96 public — `original.jsonl`
only, `heldout` is private) and `hard` (111/220 public). 231 of the 534 full v1.2 decisions.

## Running it

```bash
dotnet build src/Laya.Cli -c Release
dotnet run --project src/Laya.Cli -c Release -- download --model english --cache artifacts/models-cache

dotnet run --project benchmarks/JevBench -c Release -- run \
    --model-dir artifacts/models-cache/models.curiosity.ai/english/main --threads 4
```

Output lands in `artifacts/jevbench/` (gitignored, like every other downloaded/generated
artifact in this repo): `laya-dotnet.json` (tier accuracy, calibration, speed, cost, the axes),
`laya-dotnet-per-task.json` (one outcome letter + latency per item, same shape as
`data/baseline/laya-per-task.json`) and `comparison-vs-baseline.json` (a diff against the
published Python-package run — which items flipped outcome).

`--limit N` runs only the first N items of each tier, for a fast smoke test; `--tiers
easy,hard` restricts which tiers run at all.

## What matches the Python harness exactly, and what doesn't

- **Task loading, question building and scoring are a line-for-line port**: `tasks.py`'s
  `Task.validate`, `scoring.py`'s `validate_probs`/`argmax_label`/`score_task`, `metrics.py`'s
  Brier/ECE/percentile/paraphrase-consistency, and the calibration/intelligence/speed/cost
  formulas in `composite_v13.py`.
- **The question sent to the model is byte-for-byte the same shape** `adapters/base.py:build_question`
  builds. JevBench's frozen task files are written with `sort_keys=True`
  (`jevbench/tasks.py:Task.to_json`), so a `criteria` object's key order in the public JSONL *is*
  the order the original scored run presented options in — there's no separate "declared order"
  to recover. `Dataset/QuestionBuilder.cs` reads that order straight off the JSON, the same way
  it was read at scoring time originally.
- **`laya_local.py`'s answer mapping is reproduced exactly** in `Runner.cs`: `noul` ->
  `{"yes": p, "no": 1-p}`; `choice`/`score` -> the option-marker probabilities, used as-is.
- **Chance (for chance-corrected Intelligence) is *not* the published constant.** JevBench's
  `TIER_OPTION_COUNTS` in `composite_v13.py` is an option-count histogram over all 534 items in
  the full v1.2 set, most of which are private. This tool only ever sees the 231 public items it
  actually runs, so `Scoring/Composite.cs` computes chance directly from those (mean of `1/options`
  over the items in each tier it scored). Same formula, different — necessarily smaller and
  public-only — item set. See the doc comment on `Composite` for the reasoning.
- **Cost has no measured tariff**: local weights aren't billed. Like the original row, this
  reports an *estimate* (hosted-provider list price for an encoder of the same size class,
  $0.01/M input tokens) against this run's own measured mean input tokens — not a real bill.
- **Speed is whatever machine this runs on**, self-hosted CPU, with the same ×2 + 0.15s
  "approximate production load" adjustment JevBench applies to every self-hosted/demo row. It is
  not comparable across machines; it *is* comparable run-to-run on the same machine, which is the
  point (catching a latency regression in the port).
- **The composite `jevbench_score` this tool prints is not the ranked JevBench Score.** It's the
  same geometric-mean formula, computed over a public-only item set with recomputed chance and an
  estimated (not measured) cost. Useful for tracking laya's own public-item numbers release to
  release; not for comparing against `RESULTS-v1.2.md`'s leaderboard.

## Why the baseline is here

`data/baseline/laya.json` and `laya-per-task.json` are copied straight from
[jevbench's `results/v1.2/additions/`](https://github.com/theolivenbaum/jevbench/tree/main/results/v1.2/additions) —
the row jevbench published after running the *Python* `laya` package (0.3.3) against all 534
v1.2 decisions (including the 303 this tool cannot see). `laya-per-task.json`'s `public_tasks`
block happens to be restricted to exactly the 231 items published here, which is what makes an
apples-to-apples per-item diff possible without needing the private items at all: every task id
this tool runs has a corresponding baseline outcome letter (`c`/`w`/`f`/`n`) to compare against.

## Licensing

The copied dataset items (`data/*.jsonl`) are MIT, per each record's own
`provenance.license` field and `data/JEVBENCH-LICENSE.txt` (jevbench's own `LICENSE`, copied
here for attribution — Copyright (c) 2026 Florian Standhartinger and contributors). They are
data, not code from the jevbench repository; nothing under `jevbench/` is imported, vendored or
executed by this tool.
