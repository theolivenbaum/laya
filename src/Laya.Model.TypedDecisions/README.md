# Laya.Model.TypedDecisions

The typed-decisions checkpoint for [Laya](https://github.com/theolivenbaum/laya) — the non-autoregressive System 1
decision engine for .NET. Tuned for typed decisions over longer states.

```csharp
using Laya;
using Laya.Runtime;

// First call downloads the weights (843 MB) into ~/.cache/laya; later calls reuse them.
using var agent = LayaTypedDecisions.Load();

var result = agent.SystemOne("We were billed twice for March, please refund it.", Presets.Triage());
Console.WriteLine(result["intent"].Choice);
```

| | |
|---|---|
| Encoder | `answerdotai/ModernBERT-large` |
| Parameters | 421M |
| Max tokens | 1024 |
| Weights | 843 MB, downloaded from <https://models.curiosity.ai/laya/typed-decisions/model.safetensors> |

Everything except the weights — `rl_agent_config.json`, `encoder/config.json` and the
tokenizer — is embedded in this package, so the only thing fetched at runtime is
`model.safetensors`. It lands in `$LAYA_HOME`, `$HF_HOME/laya` or `~/.cache/laya`.
Point `LAYA_MODEL_BASE_URL` at a mirror to fetch it from somewhere else, or call
`LayaTypedDecisions.Prepare()` during deployment to warm the cache ahead of first use.

Pulls in the [`Laya`](https://www.nuget.org/packages/Laya) engine package as a dependency.

## Links

* Source, docs and issues: <https://github.com/theolivenbaum/laya>
* Model card: <https://huggingface.co/convaiinnovations/laya>

Apache-2.0. (c) Copyright 2026 Curiosity GmbH - all rights reserved. Laya itself is Copyright (c) 2025 Convai Innovations.
See `LICENSE` and `NOTICE` in this package; the weights carry the licence of the model card above.
