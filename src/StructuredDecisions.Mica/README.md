# StructuredDecisions.Mica

[Mica v0.5 4B](https://huggingface.co/sky7350/Mica-v0.5-4B) on the CPU, in managed code: a Qwen3.5-4B fine-tune that
answers typed questions about a state — yes/no, choice (including multi-answer), score (including "cannot grade"),
with optional abstention — with a probability for every allowed answer, read from one prefill. Nothing is generated.

```csharp
using StructuredDecisions;
using StructuredDecisions.Mica;

var mica = MicaCheckpoint.Load();     // downloads sky7350/Mica-v0.5-4B once (~9.3 GB), 7.8 GiB resident

// Mica's own /v1/systemone JSON …
var response = mica.SystemOne("""{"state": "The door is locked.",
  "questions": {"open": {"type": "noul", "instructions": "Is the door open?"}}}""");

// … or the shared IDecisionModel contract.
var result = mica.SystemOne("The door is locked.", new QuestionSet().Add("open", Question.Noul("Is the door open?")));
```

Validated layer by layer against transformers' Qwen3.5 implementation and against Mica's own contract fixtures.
Mica is by akivet ([github.com/akivet/Mica](https://github.com/akivet/Mica)), Apache-2.0, a fine-tune of
Qwen/Qwen3.5-4B (Apache-2.0).
