#!/usr/bin/env python3
"""Reference activations of the Qwen3.5 decoder behind Mica, from transformers' own implementation.

Two modes:

``tiny``  builds a small random Qwen3.5 text model (both layer kinds, GQA, partial RoPE, a sequence longer
          than transformers' 64-token DeltaNet chunk) and writes its bf16 weights, the input ids
          and every layer's output, from an fp32 run and from an fp64 run of the same weights. Small enough to commit; ``Qwen35ParityTests`` checks the C# model
          against it on every test run.

``checkpoint``  runs the real Mica checkpoint over one rendered Mica prompt in fp32 on the CPU. 4B
          parameters do not fit in memory as fp32 next to anything else, so the model is evaluated one
          decoder layer at a time with transformers' Qwen3_5DecoderLayer modules: each layer's weights are
          loaded, run over the hidden states, recorded and freed. It also records the label-token logits and
          the probabilities Mica would report. ``laya mica-dump`` produces the same tensors from C#.

    python tools/dump_mica_reference.py tiny --out tests/Laya.Tests/Fixtures/qwen35-tiny
    python tools/dump_mica_reference.py checkpoint --model-dir artifacts/models/mica \
        --request request.json --out artifacts/dumps/mica-reference
"""
import argparse
import gc
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


def hooks(layers, norm, out, prefix=""):
    handles = []
    for i, layer in enumerate(layers):
        def layer_hook(_m, _inp, output, i=i):
            out[f"layers.{i}"] = (output[0] if isinstance(output, tuple) else output).detach().float().clone()
        handles.append(layer.register_forward_hook(layer_hook))
        mixer = layer.linear_attn if layer.block_type == "linear_attention" else layer.self_attn

        def mixer_hook(_m, _inp, output, i=i):
            out[f"layers.{i}.mixer"] = (output[0] if isinstance(output, tuple) else output).detach().float().clone()
        handles.append(mixer.register_forward_hook(mixer_hook))
    if norm is not None:
        handles.append(norm.register_forward_hook(lambda _m, _i, o: out.__setitem__("norm", o.detach().float().clone())))
    return handles


def tiny(args):
    torch.manual_seed(args.seed)
    config = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=160, num_hidden_layers=8,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=24,
        linear_conv_kernel_dim=4, layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
        rms_norm_eps=1e-6, tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    )
    config._attn_implementation = "eager"
    model = qm.Qwen3_5ForCausalLM(config).eval()
    with torch.no_grad():
        # Random init leaves the zero-centred norms at zero and the DeltaNet decays tiny; perturb everything so
        # every weight matters to the output.
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") and "linear_attn" not in name:
                p.normal_(0, 0.2)
            elif "linear_attn.norm.weight" in name:
                p.normal_(1, 0.2)
            elif name.endswith("A_log"):
                p.uniform_(-1.0, 1.5)
            elif name.endswith("dt_bias"):
                p.normal_(0, 0.5)
            elif "conv1d" in name:
                p.normal_(0, 0.4)
            else:
                p.normal_(0, 0.08)
        # The weights the C# side reads are bf16; the reference computes in fp32 from exactly those values.
        for p in model.parameters():
            p.copy_(p.to(torch.bfloat16).float())

    ids = torch.randint(0, config.vocab_size, (1, args.tokens), generator=torch.Generator().manual_seed(args.seed + 1))
    record = {}
    handles = hooks(model.model.layers, model.model.norm, record)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, -1].float()
    for h in handles:
        h.remove()

    # The same model in fp64: what both fp32 implementations are approximating. The C# side is judged by how
    # close it gets to this, next to how close transformers' own fp32 gets.
    exact = {}
    model64 = model.double()
    handles = hooks(model64.model.layers, model64.model.norm, exact)
    with torch.no_grad():
        exact["logits"] = model64(input_ids=ids).logits[:, -1].float()
    for h in handles:
        h.remove()
    model = model64.float()

    os.makedirs(args.out, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file({k: v.to(torch.bfloat16) for k, v in state.items()}, os.path.join(args.out, "model.safetensors"))
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"architectures": ["Qwen3_5ForCausalLM"], "text_config": config.to_dict(), "tie_word_embeddings": True}, f, indent=1)
    tensors = {k: v[0].contiguous() for k, v in record.items()}
    tensors["logits"] = logits.contiguous()                  # the last position's, which is all Mica reads
    tensors.update({"fp64." + k: v[0].float().contiguous() for k, v in exact.items()})
    save_file(tensors, os.path.join(args.out, "reference.safetensors"))
    with open(os.path.join(args.out, "reference.json"), "w") as f:
        json.dump({"input_ids": ids[0].tolist(), "torch": torch.__version__}, f)
    print(f"wrote tiny model and {len(tensors)} reference tensors to {args.out}")


def checkpoint(args):
    import sys
    sys.path.insert(0, args.mica_repo)
    from mica import contract, codebook          # the reference request contract
    from mica.native import prompt_for
    from transformers import AutoTokenizer

    torch.set_num_threads(os.cpu_count())
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    with open(args.request, encoding="utf-8") as f:
        request = json.load(f)
    rows = contract.rows_from_request(request)
    row = rows[args.row]
    prompt = prompt_for(row, tokenizer)
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    labels = list(codebook.NOUL_LABELS) if row["kind"] == "noul" else codebook.OPTION_LABELS[:len(row["candidates"])]
    label_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in labels]

    with open(os.path.join(args.model_dir, "config.json")) as f:
        config = Qwen3_5TextConfig(**json.load(f)["text_config"])
    config._attn_implementation = "eager"
    with open(os.path.join(args.model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    handles = {}

    def tensor(name):
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(args.model_dir, shard), framework="pt")
        return handles[shard].get_tensor(name).float()

    prefix = "model.language_model."
    record = {}
    with torch.no_grad():
        embed = tensor(prefix + "embed_tokens.weight")
        h = torch.nn.functional.embedding(ids, embed)
        record["embeddings"] = h[0].clone()
        rotary = qm.Qwen3_5TextRotaryEmbedding(config)
        position_ids = torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
        position_embeddings = rotary(h, position_ids)
        mask = torch.full((ids.shape[1], ids.shape[1]), float("-inf")).triu(1)[None, None]
        for i in range(config.num_hidden_layers):
            layer = qm.Qwen3_5DecoderLayer(config, i).eval()
            state = {k[len(f"{prefix}layers.{i}."):]: tensor(k) for k in weight_map if k.startswith(f"{prefix}layers.{i}.")}
            layer.load_state_dict(state, strict=True)
            captured = {}
            hs = hooks([layer], None, captured)
            kind = config.layer_types[i]
            h = layer(h, position_embeddings=position_embeddings,
                      attention_mask=mask if kind == "full_attention" else None)
            for x in hs:
                x.remove()
            record[f"layers.{i}"] = h[0].clone()
            record[f"layers.{i}.mixer"] = captured["layers.0.mixer"][0]
            del layer, state
            gc.collect()
            print(f"layer {i} ({kind}) done", flush=True)
        norm = qm.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.load_state_dict({"weight": tensor(prefix + "norm.weight")})
        normed = norm(h)
        record["norm"] = normed[0].clone()
        logits = normed[0, -1] @ embed[label_ids].T
    probabilities = torch.softmax(logits.double(), -1).tolist()

    os.makedirs(args.out, exist_ok=True)
    record["label_logits"] = logits.float()
    save_file({k: v.contiguous() for k, v in record.items()}, os.path.join(args.out, "reference.safetensors"))
    with open(os.path.join(args.out, "reference.json"), "w", encoding="utf-8") as f:
        json.dump({"request": request, "row": args.row, "prompt": prompt, "input_ids": ids[0].tolist(),
                   "labels": labels, "label_ids": label_ids, "probabilities": probabilities,
                   "torch": torch.__version__}, f, ensure_ascii=False, indent=1)
    print(f"{len(ids[0])} tokens; probabilities {[round(p, 4) for p in probabilities]}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("tiny")
    t.add_argument("--out", required=True)
    t.add_argument("--tokens", type=int, default=150)
    t.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("checkpoint")
    c.add_argument("--model-dir", required=True)
    c.add_argument("--mica-repo", required=True, help="a checkout of github.com/akivet/Mica (its mica/ package)")
    c.add_argument("--request", required=True, help="a /v1/systemone request as JSON")
    c.add_argument("--row", type=int, default=0, help="which question row of the request to run")
    c.add_argument("--out", required=True)
    args = parser.parse_args()
    tiny(args) if args.mode == "tiny" else checkpoint(args)


if __name__ == "__main__":
    main()
