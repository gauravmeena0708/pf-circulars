# ONNX Cross-Encoder Quantization — Design

## Problem

`retriever.retrieve_relevant_chunks` re-ranks up to `TOP_N_INITIAL_RETRIEVAL` (default 20) hybrid candidates per query with a PyTorch `sentence_transformers.CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`). On the CPU-constrained free-tier Hugging Face Space this backlog targets, that's real per-query latency. TODO.md's optimization backlog asks for INT8 ONNX quantization, citing a "60-70% drop in re-ranking latency ... with identical ranking precision."

## What was actually measured (not assumed)

Before designing the pipeline, the export → quantize → inference flow was prototyped end-to-end against the real, installed environment and the real 74,317-passage corpus, since the TODO's numbers were unverified:

- **Environment**: neither `onnxruntime` nor `optimum` was installed. `cross-encoder/ms-marco-MiniLM-L-6-v2` is `BertForSequenceClassification` with `num_labels=1` (a regression-style single score, not a classifier) and `CrossEncoder.predict()`'s `activation_fn` is `Identity()` — it returns the raw logit, no sigmoid. This matters only for exact score-value parity; `retriever.py` only ever uses relative ordering (`.sort()`), never an absolute threshold, so the activation function doesn't affect ranking behavior either way.
- **Export + dynamic INT8 quantization** via `optimum.onnxruntime` (`ORTModelForSequenceClassification.from_pretrained(model_id, export=True)` + `ORTQuantizer` with `AutoQuantizationConfig.avx2(is_static=False, per_channel=False)`) took ~34s export + ~3s quantize, one-time, producing a 23MB `model_quantized.onnx`.
- **Runtime inference does not need `optimum`**: a raw `onnxruntime.InferenceSession` on the quantized file, fed by the existing `transformers.AutoTokenizer`, produces byte-identical scores to the `optimum`-wrapped version (verified: `[8.846713, -7.863151]` both ways). `optimum` is needed only by the one-off build script.
- **Quality**: across 5 real queries against the real corpus (20 candidates each, from the actual RRF-fused retrieval pipeline), Spearman rank correlation between PyTorch and ONNX-INT8 scores was 0.986-0.998. The **#1-ranked result never changed** in any query. The **top-5 *set* changed in 2 of 5 queries** — near-tie candidates at ranks 4-5 swapped. Small, but real and worth knowing about, not something to claim away as "identical."
- **Latency**: the first, naive measurement showed ONNX INT8 as *slower* than PyTorch (0.75x) — a cold-start/session-init artifact. After proper warmup, steady-state per-batch (20 pairs) was **72.5ms (ONNX INT8) vs. 97.9ms (PyTorch) — a 1.35x speedup, ~26% latency reduction**. Real, but well short of the TODO's claimed 60-70%.

**Decision, made explicitly by the user given these real numbers**: proceed. A genuine ~26% re-ranking speedup with a kill switch and a documented, small quality tradeoff (never affects the top result, occasionally reorders near-ties beyond it) is worth shipping. The TODO's specific "60-70%" and "identical ranking precision" claims should be corrected wherever cited going forward — they don't hold in this environment.

## Architecture

**Build-time (one-off, not part of the deployed app or its request path):**
`scripts/quantize_cross_encoder.py` — a standalone script using `optimum[onnxruntime]` to export `config.CROSS_ENCODER_MODEL_NAME` to ONNX and apply dynamic INT8 quantization, writing the result to `models/cross_encoder_onnx_int8/`. This is re-run only if the base cross-encoder model changes (rare) — analogous to the existing BM25-cache and FAISS-index backfill scripts already in this repo, which are also run occasionally by a developer, not on every deploy.

**Runtime:**
- New module `onnx_cross_encoder.py`: a small class exposing `predict(pairs) -> numpy.ndarray` — the exact interface `sentence_transformers.CrossEncoder.predict()` already provides for the two-argument list-of-pairs call pattern `retriever.py` uses. Backed by a raw `onnxruntime.InferenceSession` (no `optimum` at runtime) plus `transformers.AutoTokenizer.from_pretrained(config.CROSS_ENCODER_MODEL_NAME)`.
- Since the interface matches exactly, **`retriever.py` needs zero changes** — same pattern as the SQLite passage-store migration and the FAISS-metric-alignment plan earlier this session.
- `app.py`'s `load_cross_encoder_model`/`get_cross_encoder_model` gain a fallback chain: if `config.USE_ONNX_CROSS_ENCODER` is true and the quantized model file exists, load the ONNX wrapper; if the file is missing or loading raises, fall back to the existing PyTorch `CrossEncoder`; if that also fails, the existing "continue without re-ranking" behavior is unchanged. This mirrors `get_cross_encoder_model`'s existing try/except pattern, extended with one more fallback tier.
- New `config.USE_ONNX_CROSS_ENCODER` flag (default `True`, env-driven, same pattern as `CONFIDENCE_GATE_ENABLED`) — lets ops force PyTorch without a code change if quality complaints show up in practice.

## Storage

The quantized model (~23MB) is committed under a new `models/cross_encoder_onnx_int8/` directory, tracked via Git LFS — consistent with every other binary artifact in this repo (`vector_store/**`). Requires one new `.gitattributes` line: `models/** filter=lfs diff=lfs merge=lfs -text`.

## Dependencies

Added to `requirements.txt` (this repo's single shared file for both the deployed app and its build/ingestion tooling — no existing precedent for splitting dev-only deps into a separate file, so this follows the established convention rather than inventing a new one):
- `onnxruntime` — used at runtime by `onnx_cross_encoder.py`.
- `optimum[onnxruntime]` — used only by `scripts/quantize_cross_encoder.py`, never imported by `app.py`/`retriever.py`.

## Error Handling

- `onnx_cross_encoder.py`'s wrapper raises on genuine failures (missing file, corrupt ONNX graph, tokenizer mismatch) rather than swallowing them — the *caller* (`app.py`'s fallback chain) is what decides to degrade gracefully to PyTorch, matching the existing pattern where `get_cross_encoder_model` is the single place that catches load failures.
- No new failure mode for `retriever.py` — it receives whichever cross-encoder-like object the fallback chain resolved to, or `None`, exactly as today.

## Testing Strategy

- Unit tests for `onnx_cross_encoder.py` against the real, committed quantized model file (this repo's established convention throughout this session has been testing against real artifacts, not mocks, wherever the artifact is small and committed) — `predict()` returns scores in the right shape/order for a small batch, and matches a fixed reference score within a documented tolerance (guards against a future accidental re-export changing behavior unnoticed).
- A test for `app.py`'s fallback chain logic: ONNX file missing → PyTorch loaded; `USE_ONNX_CROSS_ENCODER=False` → PyTorch loaded even if the ONNX file exists.
- The quality/latency measurements in this spec are a one-time validation, not something to re-run in CI on every commit — no test asserts a specific Spearman correlation or latency number, since both are hardware- and environment-dependent and would make CI flaky for no real safety benefit. The kill switch is the actual production safety net.

## Scope Boundary (for the implementation plan)

This spec covers exactly: `scripts/quantize_cross_encoder.py`, `onnx_cross_encoder.py`, `app.py`'s cross-encoder loading fallback chain, the new `config.USE_ONNX_CROSS_ENCODER` flag, `requirements.txt`, `.gitattributes`, and generating + committing the actual quantized model artifact. It does not touch `retriever.py` or any ingestion script.
