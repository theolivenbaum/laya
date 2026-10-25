#!/usr/bin/env python3
"""Generate src/StructuredDecisions.Base/Tokenizers/UnicodeNfc.data, the tables behind Laya's managed NFC.

Laya builds with InvariantGlobalization, where string.Normalize does not compose non-ASCII text, so NFC is
implemented in managed code from these tables: every canonical decomposition, every non-zero canonical
combining class, and the canonical compositions NFC actually performs (primary composites only: composition
exclusions and singletons are left out by checking what unicodedata's own NFC produces).

    python tools/gen_unicode_nfc.py > src/StructuredDecisions.Base/Tokenizers/UnicodeNfc.data
"""
import sys
import unicodedata

HANGUL = range(0xAC00, 0xAC00 + 11172)

print(f"# unicode {unicodedata.unidata_version}")
for cp in range(0x110000):
    if cp in HANGUL or 0xD800 <= cp <= 0xDFFF:
        continue
    c = chr(cp)
    ccc = unicodedata.combining(c)
    if ccc:
        print(f"c {cp:X} {ccc}")
    d = unicodedata.decomposition(c)
    if d and not d.startswith("<"):
        parts = [int(x, 16) for x in d.split()]
        print("d %X %s" % (cp, " ".join("%X" % p for p in parts)))
        if len(parts) == 2 and unicodedata.normalize("NFC", chr(parts[0]) + chr(parts[1])) == c:
            print(f"p {cp:X}")
