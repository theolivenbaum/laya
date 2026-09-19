#!/usr/bin/env python3
"""Dump reference tokenizations so the .NET tokenizer can be checked without transformers.

Covers the shapes laya actually feeds the tokenizer — rendered option text, serialized JSON state,
instruction heads — plus the usual traps: whitespace runs that are added tokens, non-Latin scripts,
emoji and surrogate pairs, and the literal special-token strings.

    python tools/dump_tokenizations.py --model-dir artifacts/models/english \
        --out tests/Laya.Tests/Fixtures/tokenizer-english.json
"""
import argparse
import json
import os
import random

from transformers import AutoTokenizer

CASES = [
    "Hello world",
    "I was charged twice and nobody answers",
    "  leading and   multiple   spaces  ",
    "tabs\there",
    "newline\nhere",
    "Ünïcödé tëxt with açcents",
    "emoji 🙂🚀 test",
    "日本語のテキスト",
    "हिन्दी पाठ",
    "Mein Konto wurde zweimal belastet",
    "numbers 1234567890 and 3.14159",
    "punctuation!!! ??? ;;; ---",
    "CamelCaseWordsGlued",
    "snake_case_words",
    "a" * 300,
    " " * 30 + "x",
    "|||IP_ADDRESS|||",
    "choice question: What does the customer want in `message`?",
    " refund: money returned or a duplicate charge reversed",
    '{"subject": "Invoice", "body": "please pay"}',
    "level 0: calm and neutral",
    "false: no, the statement does not hold",
    "Ελληνικά κείμενο",
    "русский текст",
    "混合 mixed 语言 text",
    "▁weird metaspace",
    "trailing spaces   ",
    "\r\n\r\n",
    "",
]

ALPHABET = "abcdefgABCDEFG 0123456789 ,.!?;:'\"()[]{}\n\téü中\U0001F600_-"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--random", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))

    cases = list(CASES)
    random.seed(args.seed)
    for _ in range(args.random):
        cases.append("".join(random.choice(ALPHABET) for _ in range(random.randint(1, 80))))

    records = [{"text": c, "ids": tokenizer(c, add_special_tokens=False)["input_ids"]} for c in cases]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "format": "laya-tokenization/1",
            "special": {
                "cls": tokenizer.cls_token_id,
                "sep": tokenizer.sep_token_id,
                "mask": tokenizer.mask_token_id,
                "pad": tokenizer.pad_token_id,
            },
            "cases": records,
        }, f, ensure_ascii=False)
    print(f"wrote {len(records)} tokenizations to {args.out}")


if __name__ == "__main__":
    main()
