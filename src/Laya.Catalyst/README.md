# Laya.Catalyst

A statistical language identifier for Laya's `Router`, backed by
[Catalyst](https://github.com/curiosity-ai/catalyst)'s language detector (53 languages, model embedded in
the Catalyst package — nothing is downloaded).

```csharp
using Laya;
using Laya.Catalyst;

var router = new Router { LanguageClassifier = await CatalystLanguageClassifier.CreateAsync() };
router.Route("Saya ditagih dua kali untuk langganan saya bulan ini").Model;   // "multilingual"
```

Laya's built-in detector is exact about script and names a Latin-script language only on stopword
evidence. Text it cannot identify that carries no non-English letters — plain-ASCII Indonesian, Swahili,
Tagalog, Turkish with its letters stripped — is otherwise treated as English and sent to the checkpoint
that cannot read it. The classifier is consulted for exactly that case: it can move an undecided state to
the multilingual checkpoint, never an identified one away from it, and it is not asked about text shorter
than `MinimumWords` or trusted below `MinimumProbability`.
