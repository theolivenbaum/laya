# StructuredDecisions.Base

The shared base of the StructuredDecisions model families ([`StructuredDecisions.Laya`](https://www.nuget.org/packages/StructuredDecisions.Laya),
[`StructuredDecisions.Mica`](https://www.nuget.org/packages/StructuredDecisions.Mica)). You rarely reference it on its own.

- **The contract**: `IDecisionModel.SystemOne(state, questions)`; `Question` (`Noul`, `Choice`, `Score`, plus
  `AllowingMultiple()`, `WithCannotGrade()`, `WithAbstainThreshold()`), `QuestionSet`, `Answer`, `DecisionResult`.
- **The runtime both models run on**: panel-packed SIMD GEMM in fp32 (`PackedMatrix`) and bf16 (`PackedMatrixBf16`),
  Hugging Face `tokenizer.json` (byte-level and SentencePiece BPE, managed NFC), safetensors read/write and sharded
  checkpoints, resumable downloads from the Hugging Face hub and the model host, per-stage timing and state dumps.

```csharp
using StructuredDecisions;

IDecisionModel model = /* StructuredDecisions.Laya.Agent, StructuredDecisions.Mica.MicaJudge, … */;
var result = model.SystemOne("The door is locked.", new QuestionSet().Add("open", Question.Noul("Is the door open?")));
double p = result["open"].Noul!.Value;
```
