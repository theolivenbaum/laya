#!/usr/bin/env python3
"""Dump reference Qwen3.5 tokenizations for Mica, so the .NET tokenizer is checked id for id.

Mica's prompts go through the Qwen3.5 tokenizer: NFC, a Split pre-tokenizer on a regex that tokenizers
evaluates with Oniguruma over code points, then byte-level BPE. The corpus covers that regex's corners
(contractions, a leading non-letter before a word, digits one at a time, symbol runs that swallow line
breaks, whitespace runs that end in a newline or before a word), letters outside the BMP (which a UTF-16
regex engine gets wrong), the chat-control added tokens, and every rendered prompt of Mica's own
contract fixtures.

    python tools/dump_mica_tokenizations.py --model-dir artifacts/models/mica \
        --contract /path/to/akivet/mica/tests/fixtures/contract_cases.json \
        --out tests/StructuredDecisions.Tests/Fixtures/tokenizer-mica.json
"""
import argparse
import json
import os
import random

from tokenizers import Tokenizer

CASES = [
    "Hello world", "I'm sure they'll say we've done it, didn't they? I'D SAY 'RE 'VE 'LL 'Ll",
    "don’t curly apostrophe", "'s 'S 'ſ 'x", "  leading and   multiple   spaces  ", "tabs\there\t\t",
    "newline\nhere", "a \n b", "a  \n\n  b", "x \r\n\r\n y", "end with spaces   ", "   ", "\n", " \n ",
    "!!!\n\nnext", "?? \n", " .,;\n\r\n", "numbers 1234567890 and 3.14159", "١٢٣ ٤٥٦ Ⅻ ½ ²",
    "punctuation!!! ??? ;;; ---", "CamelCaseWordsGlued", "snake_case_words", "a" * 300, " " * 30 + "x",
    "Ünïcödé tëxt with açcents", "é combining", "́̂ marks first", " ́x",
    "emoji 🙂🚀 test 👨‍👩‍👧", "日本語のテキスト", "한국어 텍스트입니다", "中文测试，标点。", "हिन्दी पाठ",
    "العربية النص", "Ελληνικά", "русский текст", "𠀀𠀁 CJK ext B", "𝐀𝐁𝐂 math bold", "𐐀𐐁 deseret",
    "x𝐀y", "1𝐀", "!𝐀", " nbsp em　ideo", " line para", "\u0085nel",
    '{"subject": "Invoice", "body": "please pay"}', "<state>\n{\"a\": 1}\n</state>", "A) foo\nB) bar",
    "<|im_start|>user\nhi<|im_end|>\n", "<think>\n\n</think>\n\n", "<​|im_start|>", "<tool_call>x</tool_call>",
    "<|endoftext|><|vision_start|>", "trailing spaces   ", "\r\n\r\n", "\t\n\t", "",
]

ALPHABET = ("abcdefgABCDEFG 0123456789 ,.!?;:'\"()[]{}<>|\n\r\téüßñ中한́ 　"
            "\U0001F600\U00020000\U0001D400_-")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--contract", help="Mica's tests/fixtures/contract_cases.json: its prompts are added")
    parser.add_argument("--out", required=True)
    parser.add_argument("--random", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))

    cases = list(CASES)
    random.seed(args.seed)
    for _ in range(args.random):
        cases.append("".join(random.choice(ALPHABET) for _ in range(random.randint(1, 80))))
    if args.contract:
        with open(args.contract, encoding="utf-8") as f:
            for case in json.load(f)["cases"]:
                cases.extend(p for p in case["prompts"] if len(p) < 20000)

    records = [{"text": c, "ids": tokenizer.encode(c, add_special_tokens=False).ids} for c in cases]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"format": "laya-tokenization/1", "cases": records}, f, ensure_ascii=False)
    print(f"wrote {len(records)} tokenizations to {args.out}")


if __name__ == "__main__":
    main()
