#!/usr/bin/env python3
"""Dump PyTorch reference tensors for d1's image path (SigLIP2 tower, projector, LFM2 over the
scattered features), so ``Laya.D1`` can be checked layer by layer.

The pictures are synthetic and deterministic (gradients, shapes and noise drawn with Pillow) and are
written next to the dump as binary PPM files, so the .NET test reads exactly the pixels the reference
saw. Three sizes exercise the three branches of the processor: a small picture resized whole, a
large one cut into 512-pixel tiles plus a thumbnail, and one over ``VISION_MAX_PIXELS`` that
``cap_pixels`` first shrinks with Pillow.

    python tools/dump_d1_vision_reference.py --model-dir artifacts/models/d1-3B \
        --out artifacts/dumps/d1-vision --fixture tests/Laya.Tests/Fixtures/torch-d1-vision.json

For every tile it records the patches, the patch embeddings, every tower layer, the post norm and
the projected features; then the prompt ids, the language model's layers, the full-vocabulary
log-softmax at each answer slot, and the answers.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_d1_reference import capture, squeeze, strided_sample, to_record  # noqa: E402

CATS = {"type": "choice", "instructions": "What is the main shape in the picture?",
        "criteria": {"circle": "A circle", "square": "A square", "triangle": "A triangle"}}
RED = {"type": "noul", "instructions": "Is there anything red in the picture?"}
BRIGHT = {"type": "score", "instructions": "How bright is the picture?", "criteria": ["Dark", "Medium", "Bright"]}


def picture(width, height, seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack([(x * 255 // max(1, width - 1)), (y * 255 // max(1, height - 1)),
                     ((x + y) * 127 // max(1, width + height))], -1).astype(np.uint8)
    image = Image.fromarray(base, "RGB")
    draw = ImageDraw.Draw(image)
    draw.ellipse([width * 0.2, height * 0.2, width * 0.6, height * 0.7], fill=(220, 30, 40))
    draw.rectangle([width * 0.65, height * 0.1, width * 0.9, height * 0.45], fill=(20, 40, 200))
    noise = rng.integers(0, 24, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(np.clip(np.asarray(image, dtype=np.int16) + noise - 12, 0, 255).astype(np.uint8), "RGB")


CASES = {
    # a picture that fits: resized whole, the state is the picture alone, one question (the chain)
    "small": ((300, 200, 1), None, {"shape": CATS}),
    # a large picture: a 2 x 1 grid of tiles and a thumbnail, with text, two questions (the tree)
    "tiled": ((1100, 600, 2), "A photo sent by a customer.", {"red": RED, "shape": CATS}),
    # over a megapixel: Pillow shrinks it first, then tiles
    "capped": ((1400, 1000, 3), None, {"bright": BRIGHT}),
}


def write_ppm(path, image):
    with open(path, "wb") as f:
        f.write(f"P6\n{image.width} {image.height}\n255\n".encode())
        f.write(np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes())


def run_case(model, engine, state, questions, image):
    prompt = sys.modules[type(engine).__module__.replace("runner", "prompt")]
    runner = sys.modules[type(engine).__module__]
    qs = [prompt.as_question(q) for q in questions.values()]
    tok = engine.tokenizer
    pics = [runner.cap_pixels(image)]
    prefix = prompt.prefix_text(tok, state, engine.bos, engine.state_style, engine.system, engine._image_markup(1))
    suffixes = [prompt.suffix_text(tok, q, engine.lead, engine.option_style) for q in qs]

    tensors = {}
    tower = model.model.vision_tower.vision_model
    tower_store = {}
    handles = []

    def hook(name):
        def fn(_m, _i, output):
            tower_store[name] = (output[0] if isinstance(output, tuple) else output).detach().float().clone()
        return fn

    handles.append(tower.embeddings.register_forward_hook(hook("embeddings")))
    for i, layer in enumerate(tower.encoder.layers):
        handles.append(layer.register_forward_hook(hook(f"layers.{i}.output")))
    handles.append(tower.post_layernorm.register_forward_hook(hook("post_layernorm")))
    projected = []
    handles.append(model.model.multi_modal_projector.register_forward_hook(
        lambda _m, _i, out: projected.append(out.detach().float().reshape(-1, out.shape[-1]).clone())))

    store, lm_handles = capture(model)
    try:
        with torch.inference_mode():
            if len(qs) == 1:
                inputs = engine._image_inputs(prefix + suffixes[0], pics)
                ids = inputs["input_ids"][0].tolist()
                trunk, rows = None, [ids]
                row = engine._one_pass(**inputs, logits_to_keep=1).logits[0, -1].float()
                logz = [row - torch.logsumexp(row, dim=-1)]
            else:
                vision = engine._image_inputs(prefix, pics)
                inputs = dict(vision)
                trunk = vision.pop("input_ids")[0].tolist()
                vision.pop("attention_mask", None)
                rows = [tok.encode(s, add_special_tokens=False) for s in suffixes]
                logz = engine._tree_logz(trunk, rows, **vision)
    finally:
        for h in handles + lm_handles:
            h.remove()

    mask = inputs["pixel_attention_mask"]
    shapes = inputs["spatial_shapes"].tolist()
    pixels = inputs["pixel_values"]
    for t in range(pixels.shape[0]):
        valid = int(mask[t].sum())
        tensors[f"image0.tile{t}.patches"] = pixels[t, :valid].float().contiguous()
        tensors[f"image0.tile{t}.embeddings"] = tower_store["embeddings"][t, :valid].contiguous()
        for i in range(len(tower.encoder.layers)):
            tensors[f"image0.tile{t}.layers.{i}.output"] = tower_store[f"layers.{i}.output"][t, :valid].contiguous()
        tensors[f"image0.tile{t}.post_layernorm"] = tower_store["post_layernorm"][t, :valid].contiguous()
        tensors[f"image0.tile{t}.projected"] = projected[t].contiguous()

    tensors.update({k: squeeze(v).contiguous() for k, v in store.items()})
    probabilities = []
    for i, (q, z) in enumerate(zip(qs, logz)):
        tensors[f"logz.{i}"] = z.float().contiguous()
        probs = engine._readout(q, z)
        probabilities.append(probs)
        tensors[f"probabilities.{i}"] = torch.tensor(probs, dtype=torch.float32)

    result = model.system_one(state, questions, images=[image])
    meta = {
        "state": state, "questions": questions, "prefix": prefix, "suffixes": suffixes,
        "ids": rows, "trunk": trunk, "spatial_shapes": shapes,
        "capped_size": [pics[0].width, pics[0].height],
        "probabilities": probabilities, "answers": result["answers"], "usage": result["usage"],
    }
    return tensors, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", default="artifacts/dumps/d1-vision")
    parser.add_argument("--fixture", default=None)
    parser.add_argument("--images", default="tests/Laya.Tests/Fixtures/d1-images",
                        help="where the synthetic pictures are written (as PPM)")
    parser.add_argument("--sample", type=int, default=32)
    args = parser.parse_args()

    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True, dtype=torch.bfloat16)
    model.float()   # everything in fp32, as on CPU
    model.eval()
    engine = model.engine

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.images, exist_ok=True)
    all_meta, records = {}, []
    for name, ((w, h, seed), state, questions) in CASES.items():
        image = picture(w, h, seed)
        write_ppm(os.path.join(args.images, f"{name}.ppm"), image)
        tensors, meta = run_case(model, engine, state, questions, image)
        meta["image"] = f"{name}.ppm"
        save_file(tensors, os.path.join(args.out, f"{name}.safetensors"))
        all_meta[name] = meta
        for key, value in tensors.items():
            if ".layers." in key and not key.startswith("layers."):
                continue   # the fixture keeps the tower's ends, the full dump keeps every layer
            record = to_record(f"{name}.{key}", value, args.sample)
            record["values"] = [float(v) for v in strided_sample(value, args.sample)]
            records.append(record)
        print(f"{name}: {len(tensors)} tensors, shapes {meta['spatial_shapes']}; answers {json.dumps(meta['answers'])}")

    with open(os.path.join(args.out, "cases.json"), "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=1)
    if args.fixture:
        with open(args.fixture, "w", encoding="utf-8") as f:
            json.dump({"format": "laya-d1-vision-dump/1", "cases": all_meta, "tensors": records}, f, ensure_ascii=False)
        print(f"wrote {len(records)} sampled tensors to {args.fixture}")


if __name__ == "__main__":
    main()
