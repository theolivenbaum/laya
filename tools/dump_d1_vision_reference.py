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
    "capped": ((1060, 1000, 3), None, {"bright": BRIGHT}),
}


def write_ppm(path, image):
    with open(path, "wb") as f:
        f.write(f"P6\n{image.width} {image.height}\n255\n".encode())
        f.write(np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes())


def prompts(engine, state, questions):
    prompt = sys.modules[type(engine).__module__.replace("runner", "prompt")]
    qs = [prompt.as_question(q) for q in questions.values()]
    prefix = prompt.prefix_text(engine.tokenizer, state, engine.bos, engine.state_style, engine.system,
                                engine._image_markup(1))
    suffixes = [prompt.suffix_text(engine.tokenizer, q, engine.lead, engine.option_style) for q in qs]
    return qs, prefix, suffixes


def vision_case(model, engine, state, questions, image):
    """Phase one: the processor, the tower and the projector, every tile's tensors and the features."""
    runner = sys.modules[type(engine).__module__]
    qs, prefix, suffixes = prompts(engine, state, questions)
    pics = [runner.cap_pixels(image)]
    inputs = engine._image_inputs(prefix + suffixes[0] if len(qs) == 1 else prefix, pics)

    tower = getattr(model.model.vision_tower, "vision_model", model.model.vision_tower)
    store, handles = {}, []

    def hook(name):
        def fn(_m, _i, output):
            store[name] = (output[0] if isinstance(output, tuple) else output).detach().float().clone()
        return fn

    handles.append(tower.embeddings.register_forward_hook(hook("embeddings")))
    for i, layer in enumerate(tower.encoder.layers):
        handles.append(layer.register_forward_hook(hook(f"layers.{i}.output")))
    handles.append(tower.post_layernorm.register_forward_hook(hook("post_layernorm")))
    try:
        with torch.inference_mode():
            features = model.model.get_image_features(
                pixel_values=inputs["pixel_values"], spatial_shapes=inputs["spatial_shapes"],
                pixel_attention_mask=inputs["pixel_attention_mask"]).pooler_output
    finally:
        for h in handles:
            h.remove()

    tensors = {}
    mask, pixels = inputs["pixel_attention_mask"], inputs["pixel_values"]
    for t in range(pixels.shape[0]):
        valid = int(mask[t].sum())
        tensors[f"image0.tile{t}.patches"] = pixels[t, :valid].float().contiguous()
        tensors[f"image0.tile{t}.embeddings"] = store["embeddings"][t, :valid].contiguous()
        for i in range(len(tower.encoder.layers)):
            tensors[f"image0.tile{t}.layers.{i}.output"] = store[f"layers.{i}.output"][t, :valid].contiguous()
        tensors[f"image0.tile{t}.post_layernorm"] = store["post_layernorm"][t, :valid].contiguous()
        tensors[f"image0.tile{t}.projected"] = features[t].float().contiguous()
    meta = {"spatial_shapes": inputs["spatial_shapes"].tolist(), "capped_size": [pics[0].width, pics[0].height]}
    return tensors, [f.float().clone() for f in features], meta


def text_case(model, engine, state, questions, image, features):
    """Phase two: the language model in fp32 over the features phase one computed."""
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    runner = sys.modules[type(engine).__module__]
    qs, prefix, suffixes = prompts(engine, state, questions)
    tok = engine.tokenizer
    pics = [runner.cap_pixels(image)]
    model.model.get_image_features = lambda **_: BaseModelOutputWithPooling(pooler_output=list(features))

    store, handles = capture(model)
    try:
        with torch.inference_mode():
            if len(qs) == 1:
                inputs = engine._image_inputs(prefix + suffixes[0], pics)
                trunk, rows = None, [inputs["input_ids"][0].tolist()]
                row = engine._one_pass(**inputs, logits_to_keep=1).logits[0, -1].float()
                logz = [row - torch.logsumexp(row, dim=-1)]
            else:
                vision = engine._image_inputs(prefix, pics)
                trunk = vision.pop("input_ids")[0].tolist()
                vision.pop("attention_mask", None)
                rows = [tok.encode(s, add_special_tokens=False) for s in suffixes]
                logz = engine._tree_logz(trunk, rows, **vision)
    finally:
        for h in handles:
            h.remove()

    tensors = {k: squeeze(v).contiguous() for k, v in store.items()}
    probabilities = []
    for i, (q, z) in enumerate(zip(qs, logz)):
        tensors[f"logz.{i}"] = z.float().contiguous()
        probs = engine._readout(q, z)
        probabilities.append(probs)
        tensors[f"probabilities.{i}"] = torch.tensor(probs, dtype=torch.float32)

    result = model.system_one(state, questions, images=[image])
    meta = {"state": state, "questions": questions, "prefix": prefix, "suffixes": suffixes,
            "ids": rows, "trunk": trunk, "probabilities": probabilities,
            "answers": result["answers"], "usage": result["usage"]}
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

    # Everything computes in fp32, as on CPU, but the whole model in fp32 (12.5 GB) does not fit
    # beside its activations in a 13 GB container: phase one runs the tower and the projector in fp32
    # with the language model still bf16, phase two drops the tower, widens the language model, and
    # replays phase one's features through get_image_features.
    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True, dtype=torch.bfloat16)
    model.model.vision_tower.float()
    model.model.multi_modal_projector.float()
    model.eval()
    engine = model.engine

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.images, exist_ok=True)
    phase_one = {}
    for name, ((w, h, seed), state, questions) in CASES.items():
        image = picture(w, h, seed)
        write_ppm(os.path.join(args.images, f"{name}.ppm"), image)
        phase_one[name] = (image,) + vision_case(model, engine, state, questions, image)
        print(f"{name}: vision done, tiles {phase_one[name][3]['spatial_shapes']}", flush=True)

    model.model.vision_tower = None
    import gc
    gc.collect()
    model.model.language_model.float()
    model.lm_head.float()

    all_meta, records = {}, []
    for name, ((w, h, seed), state, questions) in CASES.items():
        image, tensors, features, vision_meta = phase_one.pop(name)
        lm_tensors, meta = text_case(model, engine, state, questions, image, features)
        tensors.update(lm_tensors)
        meta.update(vision_meta)
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
