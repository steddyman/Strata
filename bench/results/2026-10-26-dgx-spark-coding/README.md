# Coding speed on a DGX Spark, before and after the Spark speed settings (2026-10-26)

Measurement only. The numbers behind the README's "Writing code" infographic and the "Coding tasks" table in
[docs/DGX_SPARK.md](../../../docs/DGX_SPARK.md#coding-tasks). Every request, both rounds, the configurations and the engine
and server logs are in this folder. Home folders, the user and host names are replaced.

## What was compared

- **Machine:** one NVIDIA DGX Spark, GB10 (sm_121), 128 GB unified memory (121 GiB seen by Linux), DGX OS, kernel
  7.0.0-1019-nvidia, driver 580.178.04, CUDA 13.0.88.
- **Engine:** 0.1.51 (`main` at 1a0cc5b9, built on the Spark with `spark/build_engine.sh`), the same binary for every arm.
- **Models:** Qwen3.8-Flash-Next NVFP4 (Maximilian228's GPTQ pack) and DeepSeek-V4.1-Flash IST-DAS 1.75 bpw.
- **Configurations:** the setup-written configs on the Spark (`configs/`, single stream: `exp.py` removes `parallel`,
  `host` and `api_key` and serves on 127.0.0.1:8180). Every changed setting:

| Arm | Qwen NVFP4 | DeepSeek 1.75 bpw |
| --- | --- | --- |
| before | `STRATA_HC_Q8=0 STRATA_PLE_BATCH=0`, no `--mtp-q4` | `--dspark-conf 0.65` |
| after | the config as setup writes it on a Spark (`--mtp-q4 all`; the 8-bit hyper-connection read and batched PLE on by default on a GB10) | `--dspark-conf 0.4` |
| after + en vocab | after, plus `--mtp-draft-vocab data/draft_vocab_en.bin` | - |

## Method

`coding_bench.py` (in this folder) against the server: one warm-up request ("Say hello."), then five coding prompts, each
`max_tokens` 512, temperature 0, thinking off (`enable_thinking: false`), `cache_prompt: false`, a timestamp at the start
of each prompt so nothing is reused. Speeds are the server's `predicted_per_second` / `predicted_n` / `predicted_ms`. The
prompts:

1. `lru`: an LRU cache module with a TTL per entry, type hints, docstrings and an example
2. `dataclass`: a 12-field `Customer` dataclass with `to_dict()` and `from_dict()`
3. `tests`: pytest unit tests for a `parse_duration(s)` function, including edge cases and errors
4. `api`: a FastAPI todo app with Pydantic models, CRUD endpoints, paging, status codes and 404 handling
5. `ts`: a TypeScript conversion of an order-grouping function, with types and JSDoc

`run_coding.sh` ran the arms in this order, each on a freshly started server (`exp.py`): Qwen before, after, after + en
vocab; then the same again (round 2); then DeepSeek before, after, before, after. One round = one request per prompt.

## Results

Per-round totals (all generated tokens / all decode time of the round), tokens/s:

| Model | Arm | Round 1 | Round 2 | Average |
| --- | --- | ---: | ---: | ---: |
| Qwen NVFP4 | before | 57.08 | 57.92 | **57.50** |
| Qwen NVFP4 | after | 61.41 | 64.37 | **62.89** (+9.4%) |
| Qwen NVFP4 | after + en vocab | 63.07 | 63.16 | 63.11 |
| DeepSeek 1.75 bpw | before | 35.90 | 35.49 | **35.70** |
| DeepSeek 1.75 bpw | after | 38.39 | 36.66 | **37.53** (+5.1%) |

Per prompt (average of the two rounds; every request is in `raw/requests.tsv`):

| Prompt | Qwen before | Qwen after | DeepSeek before | DeepSeek after |
| --- | ---: | ---: | ---: | ---: |
| lru | 49.3 | 53.6 | 32.8 | 36.8 |
| dataclass | 63.1 | 67.0 | 36.6 | 37.3 |
| tests | 58.1 | 63.3 | 31.6 | 33.6 |
| api | 59.9 | 68.5 | 42.4 | 42.7 |
| ts | 59.3 | 64.3 | 37.4 | 38.8 |

## Variability and limits

- Two rounds per arm. For Qwen, every "after" round is above every "before" round (61.4 / 64.4 against 57.1 / 57.9). For
  DeepSeek the gain holds in both rounds (+7% and +3%), but the two "after" rounds differ by 1.7 tokens/s, so +5% is a
  soft figure; more rounds would narrow it.
- No request failed. Every coding request ran to the 512-token cap (the engine logs Qwen's `dataclass` answers as 511
  tokens in all six runs; every other request is 512).
- Temperature 0 with speculative decoding: the drafting settings (`--mtp-q4`, `--dspark-conf`, the draft vocabulary) do
  not change the answers. Qwen's 8-bit hyper-connection read does, at near-tie tokens (KL 0.030 against the 16-bit read,
  perplexity 3.936 against 3.937, `tools/quant_quality.py`), so the "before" and "after" Qwen texts differ slightly.
- One Spark. Other GB10 machines should be similar but were not measured.

## Files

- `README.md`: this report
- `coding_bench.py`: the prompts and the client
- `run_coding.sh`, `exp.py`: the run order and the per-run harness (starts the server from a config copy, runs the bench,
  stops it)
- `raw/requests.tsv`: every request (model, arm, round, prompt, prompt tokens, generated tokens, decode ms, tokens/s,
  drafts for Qwen), parsed from the engine logs
- `raw/run-console.txt`: the RESULT line of every run as printed
- `configs/`: the exact server config of every run
- `logs/`: the engine log (`exp-*.log`) and the server log (`exp-*.server.log`) of every run
