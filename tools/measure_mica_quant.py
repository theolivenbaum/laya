"""How much precision do Mica's published GGUF quantizations lose against the fp32 path?

Runs every rendered prompt of the contract fixtures (``Fixtures/mica-contract-cases.json``) through the
real checkpoint once per variant, one decoder layer at a time in fp32 with transformers' own
``Qwen3_5DecoderLayer`` (as ``dump_mica_reference.py checkpoint`` does), and compares each variant's label
probabilities with the unquantized run:

``fp32``    the bf16 checkpoint computed in fp32 - what ``Qwen35Model`` reproduces (<= 5e-6 on probabilities)
``<q>-w``   every matrix of a GGUF file dequantized exactly, computed in fp32: the weight error alone
``<q>-wa``  the same, and each matrix multiply's input quantized the way ggml's CPU backend does before
            its integer dot product (``vec_dot_type``: Q8_0 for Q8_0 weights, Q8_K for the k-quants and
            IQ4_XS), which is what llama.cpp / TensorSharp's GGML backend actually compute

The GGUF matrices come from Mica's own files on the hub (``mica-v0.5-4b-Q8_0.gguf``,
``mica-v0.5-4b-IQ4_XS.gguf``), dequantized with gguf-py. llama.cpp's converter reorders the Gated
DeltaNet's value heads (ggml broadcasts key heads by tiling, transformers by ``repeat_interleave``), so
those matrices are matched back to the checkpoint's head order block by block, and every matrix's
relative error against the bf16 weights is printed as a check. Norms, the conv kernel, ``A_log`` and
``dt_bias`` are F32 in the GGUF (identical values, reordered/offset) and are taken from the checkpoint.
Not emulated: llama.cpp's f16 KV cache, and its accumulation order (integer dot products are exact).

    python tools/measure_mica_quant.py --model-dir artifacts/models/mica \
        --gguf q8_0=artifacts/models/mica-gguf/mica-v0.5-4b-Q8_0.gguf \
        --gguf iq4_xs=artifacts/models/mica-gguf/mica-v0.5-4b-IQ4_XS.gguf \
        --out artifacts/dumps/quant.json
"""

import argparse
import gc
import json
import math
import os
import time

import numpy as np
import torch
from gguf import GGUFReader, GGMLQuantizationType as Q, quants
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "tests", "StructuredDecisions.Tests", "Fixtures", "mica-contract-cases.json")
CODEBOOK = os.path.join(HERE, "..", "src", "StructuredDecisions.Mica", "codebook_255.json")
PREFIX = "model.language_model."

# transformers module -> llama.cpp tensor, and the block size its rows (or columns) may have been reordered in
LINEARS = {
    "linear_attn.in_proj_qkv": ("attn_qkv", 0, 128),
    "linear_attn.in_proj_z": ("attn_gate", 0, 128),
    "linear_attn.in_proj_b": ("ssm_beta", 0, 1),
    "linear_attn.in_proj_a": ("ssm_alpha", 0, 1),
    "linear_attn.out_proj": ("ssm_out", 1, 128),
    "self_attn.q_proj": ("attn_q", 0, None),
    "self_attn.k_proj": ("attn_k", 0, None),
    "self_attn.v_proj": ("attn_v", 0, None),
    "self_attn.o_proj": ("attn_output", 0, None),
    "mlp.gate_proj": ("ffn_gate", 0, None),
    "mlp.up_proj": ("ffn_up", 0, None),
    "mlp.down_proj": ("ffn_down", 0, None),
}

Q8_0_INPUT = {Q.Q4_0, Q.Q5_0, Q.Q8_0, Q.IQ4_NL}
Q8_1_INPUT = {Q.Q4_1, Q.Q5_1}


def quantize_q8_0(x):
    """ggml quantize_row_q8_0 (x86): 32-blocks, d = amax/127 stored fp16, q = round-half-even(x * 127/amax)."""
    shape = x.shape
    b = x.reshape(-1, 32)
    amax = b.abs().amax(-1, keepdim=True)
    inv = torch.where(amax > 0, 127.0 / amax, torch.zeros_like(amax))
    q = torch.round(b * inv)
    d = (amax / 127.0).half().float()
    return (q * d).reshape(shape)


def quantize_q8_k(x):
    """ggml quantize_row_q8_K_ref: 256-blocks, iscale = -127/max (max: the signed value of largest |x|),
    q = min(127, nearest_int(iscale * x)), d = 1/iscale kept in fp32."""
    shape = x.shape
    b = x.reshape(-1, 256)
    idx = b.abs().argmax(-1, keepdim=True)
    mx = torch.gather(b, -1, idx)
    zero = mx == 0
    iscale = torch.where(zero, torch.zeros_like(mx), -127.0 / torch.where(zero, torch.ones_like(mx), mx))
    q = torch.clamp(torch.round(iscale * b), max=127)
    d = torch.where(zero, torch.zeros_like(mx), 1.0 / torch.where(zero, torch.ones_like(iscale), iscale))
    return (q * d).reshape(shape)


def input_quantizer(qtype):
    if qtype in (Q.F32,):
        return None
    if qtype == Q.F16:
        return lambda x: x.half().float()
    if qtype == Q.BF16:
        return lambda x: x.bfloat16().float()
    if qtype in Q8_0_INPUT:
        return quantize_q8_0
    if qtype in Q8_1_INPUT:
        raise NotImplementedError("Q8_1 activations (Q4_1/Q5_1 weights)")
    return quantize_q8_k        # Q2_K..Q6_K and the IQ types


def align(q, w, axis, block):
    """Reorder q's blocks (rows, or columns when axis=1) to match w's; returns q in w's order."""
    if block is None:
        return q
    qb = np.moveaxis(q, axis, 0).reshape(q.shape[axis] // block, -1).astype(np.float64)
    wb = np.moveaxis(w, axis, 0).reshape(w.shape[axis] // block, -1).astype(np.float64)
    dist = (qb ** 2).sum(1)[:, None] + (wb ** 2).sum(1)[None] - 2 * qb @ wb.T
    source = dist.argmin(0)                              # for each checkpoint block, the GGUF block holding it
    if len(set(source.tolist())) != len(source):
        raise RuntimeError("block matching is not a permutation")
    moved = np.moveaxis(q, axis, 0).reshape(len(source), block, -1)[source]
    return np.moveaxis(moved.reshape(np.moveaxis(q, axis, 0).shape), 0, axis)


class Gguf:
    def __init__(self, name, path):
        self.name = name
        self.reader = GGUFReader(path)
        self.tensors = {t.name: t for t in self.reader.tensors}

    def matrix(self, name):
        t = self.tensors[name]
        return quants.dequantize(t.data, t.tensor_type), t.tensor_type

    def rows(self, name, ids):
        t = self.tensors[name]
        return quants.dequantize(np.asarray(t.data[ids]), t.tensor_type), t.tensor_type


def load_prompts(tokenizer):
    with open(FIXTURES, encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    with open(CODEBOOK, encoding="utf-8") as f:
        codebook = json.load(f)
    items = []
    for case in cases:
        for r, (row, prompt) in enumerate(zip(case["rows"], case["prompts"])):
            labels = codebook["noul_labels"] if row["kind"] == "noul" else codebook["option_labels"][:len(row["candidates"])]
            ids = tokenizer(prompt, add_special_tokens=False).input_ids
            label_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in labels]
            items.append({"case": case["name"], "row": r, "kind": row["kind"], "ids": ids, "label_ids": label_ids})
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--gguf", action="append", default=[], help="name=path of a GGUF file")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="only the first N prompts")
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count())

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    items = load_prompts(tokenizer)
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} prompts, {sum(len(i['ids']) for i in items)} tokens", flush=True)

    with open(os.path.join(args.model_dir, "config.json")) as f:
        config = Qwen3_5TextConfig(**json.load(f)["text_config"])
    config._attn_implementation = "sdpa"     # with no mask: is_causal, never an n x n score matrix (prompts reach 7.9k tokens)
    with open(os.path.join(args.model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    handles = {}

    def tensor(name):
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(args.model_dir, shard), framework="pt")
        return handles[shard].get_tensor(name).float()

    ggufs = [Gguf(*spec.split("=", 1)) for spec in args.gguf]
    variants = ["fp32"] + [f"{g.name}-{m}" for g in ggufs for m in ("w", "wa")]
    by_name = {g.name: g for g in ggufs}

    # embeddings: the checkpoint's bf16 rows, or the GGUF's dequantized token_embd (ggml get_rows: exact)
    needed = sorted({t for i in items for t in i["ids"] + i["label_ids"]})
    where = {t: k for k, t in enumerate(needed)}
    embed = {"fp32": tensor(PREFIX + "embed_tokens.weight")[needed]}
    head_input = {"fp32": None}
    for g in ggufs:
        rows, qtype = g.rows("token_embd.weight", needed)
        e = torch.from_numpy(rows.astype(np.float32))
        ref = embed["fp32"]
        print(f"{g.name} token_embd {qtype.name}: relative error {float((e - ref).norm() / ref.norm()):.4f}", flush=True)
        for m in ("w", "wa"):
            embed[f"{g.name}-{m}"] = e
            head_input[f"{g.name}-{m}"] = input_quantizer(qtype) if m == "wa" else None

    rotary = qm.Qwen3_5TextRotaryEmbedding(config)
    hidden = {v: [] for v in variants}
    positions = []
    with torch.no_grad():
        for it in items:
            n = len(it["ids"])
            h0 = embed["fp32"][[where[t] for t in it["ids"]]][None]
            pos = torch.arange(n).view(1, 1, -1).expand(3, 1, -1)
            positions.append(rotary(h0, pos))
            for v in variants:
                hidden[v].append(embed[v][[where[t] for t in it["ids"]]][None].clone())

        weight_error = {g.name: {} for g in ggufs}
        start = time.time()
        for i in range(config.num_hidden_layers):
            kind = config.layer_types[i]
            state = {k[len(f"{PREFIX}layers.{i}."):]: tensor(k) for k in weight_map if k.startswith(f"{PREFIX}layers.{i}.")}
            matrices = {}                                   # (gguf, module) -> dequantized, aligned; shared by -w and -wa
            for g in ggufs:
                for module_name, (gguf_name, axis, block) in LINEARS.items():
                    key = f"{module_name}.weight"
                    if key in state:
                        w = state[key].numpy()
                        q, qtype = g.matrix(f"blk.{i}.{gguf_name}.weight")
                        q = align(q.astype(np.float32), w, axis, block)
                        err = float(np.linalg.norm(q - w) / np.linalg.norm(w))
                        weight_error[g.name].setdefault(f"{module_name}:{qtype.name}", []).append(err)
                        if err > 0.2:
                            raise RuntimeError(f"layer {i} {module_name}: relative error {err} - mapping is wrong")
                        matrices[g.name, module_name] = torch.from_numpy(q), qtype
            with torch.device("meta"):
                layer = qm.Qwen3_5DecoderLayer(config, i).eval()
            layer.load_state_dict({k: t.clone() for k, t in state.items()}, strict=True, assign=True)
            for v in variants:
                layer.load_state_dict(state, strict=True)      # back to the checkpoint's weights
                hooks = []
                if v != "fp32":
                    g, mode = by_name[v.rsplit("-", 1)[0]], v.rsplit("-", 1)[1]
                    for module_name in LINEARS:
                        if (g.name, module_name) not in matrices:
                            continue
                        q, qtype = matrices[g.name, module_name]
                        module = layer.get_submodule(module_name)
                        module.weight.copy_(q)
                        quantize = input_quantizer(qtype) if mode == "wa" else None
                        if quantize is not None:
                            hooks.append(module.register_forward_pre_hook(lambda _m, inp, f=quantize: (f(inp[0]),)))
                for p, it in enumerate(items):
                    out = layer(hidden[v][p], position_embeddings=positions[p], attention_mask=None)
                    hidden[v][p] = out[0] if isinstance(out, tuple) else out
                for hk in hooks:
                    hk.remove()
            del layer
            gc.collect()
            print(f"layer {i} ({kind}) done, {time.time() - start:.0f} s", flush=True)
            del state, matrices

        norm = qm.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.load_state_dict({"weight": tensor(PREFIX + "norm.weight")})
        probabilities = {v: [] for v in variants}
        for v in variants:
            for p, it in enumerate(items):
                last = norm(hidden[v][p])[0, -1]
                if head_input[v] is not None:
                    last = head_input[v](last)
                logits = last @ embed[v][[where[t] for t in it["label_ids"]]].T
                probabilities[v].append(torch.softmax(logits.double(), -1).tolist())

    def compare(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return {"max_abs": float(np.abs(a - b).max()),
                "kl": float(np.sum(a * (np.log(np.maximum(a, 1e-300)) - np.log(np.maximum(b, 1e-300))))),
                "same_top1": bool(a.argmax() == b.argmax())}

    summary = {}
    for v in variants:
        rows = [compare(probabilities["fp32"][p], probabilities[v][p])
                for p, it in enumerate(items)]
        d = np.array([r["max_abs"] for r in rows])
        kl = np.array([r["kl"] for r in rows])
        summary[v] = {"mean_max_abs": float(d.mean()), "median_max_abs": float(np.median(d)),
                      "p90_max_abs": float(np.quantile(d, 0.9)), "worst_max_abs": float(d.max()),
                      "mean_kl": float(kl.mean()), "worst_kl": float(kl.max()),
                      "top1_agreement": float(np.mean([r["same_top1"] for r in rows])), "rows": rows}
    print(f"\n{'variant':<12} {'mean|dp|':>9} {'median':>8} {'p90':>8} {'worst':>8} {'mean KL':>9} {'top-1':>7}")
    for v, s in summary.items():
        print(f"{v:<12} {s['mean_max_abs']:9.4f} {s['median_max_abs']:8.4f} {s['p90_max_abs']:8.4f} "
              f"{s['worst_max_abs']:8.4f} {s['mean_kl']:9.2e} {s['top1_agreement']:7.1%}")
    for g, errs in weight_error.items():
        print(f"\n{g} weight relative error (mean over layers):")
        for k, e in sorted(errs.items()):
            print(f"  {k:<36} {np.mean(e):.4f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"items": [{k: it[k] for k in ("case", "row", "kind")} | {"tokens": len(it["ids"])} for it in items],
                   "probabilities": probabilities, "summary": summary,
                   "weight_error": {g: {k: float(np.mean(e)) for k, e in errs.items()} for g, errs in weight_error.items()},
                   "torch": torch.__version__}, f, indent=1)


if __name__ == "__main__":
    main()
