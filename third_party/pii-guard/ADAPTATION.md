# PII Guard decoding policy

`seif/rubert_decoder.py` adapts the word alignment, window ownership and
punctuation-bridge policy of `src/pii_guard/ner/recognizer.py` from
https://github.com/redmadrobot-rnd/pii-guard at
`24230abb72949a9f85499244dd4f15a0ad0cdd9e` (Apache-2.0).

SEIF changes, 2026-09-23: a NumPy/TensorRT logits interface, one tokenization per
document, explicit integrity checks, real first-subword softmax confidence,
original Unicode offsets, and rejection of incomplete word coverage. The
Presidio engine, normalization, pseudonymization and optional transliteration
dependencies are not included in this adaptation.

The upstream LICENSE and NOTICE are included alongside this file. Their
dependency notices describe the upstream distribution, not dependencies added
to SEIF by this decoder.
