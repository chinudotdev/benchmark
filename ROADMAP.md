# Improvement Roadmap — gpu-benchmark

Prioritized improvements beyond bug fixes. Organized by impact on the tool's
core purpose: producing **accurate, trustworthy benchmark numbers**.

---

## Tier 1 — Benchmark Correctness

These affect the numbers the tool produces. If the numbers are wrong, nothing
else matters.

### 1. Warmup phase

The first few requests to a freshly loaded model are slower — cold KV cache,
JIT compilation, memory allocation settling. Those requests pollute the
benchmark.

A few warmup requests (discarded from measurement) before the timed run would
give much more stable numbers. This is standard in every serious benchmarking
tool.

**Effort:** Small — send N requests before the timed run, discard results.

### 2. TTFT (Time to First Token)

This is a **critical real-world metric** that's completely absent. For
streaming use cases, users care deeply about how long before the first token
appears, separate from overall throughput. Both should be measured and reported.

Currently the code only measures end-to-end latency — it can't distinguish
between a model that generates fast but starts slowly vs. one that starts fast
but generates slowly.

**Effort:** Medium — for streaming, capture timestamp on first `data:` chunk.
For non-streaming, less meaningful but could use `usage` timing fields if
vLLM exposes them.

### 3. Input token count is fictional

The prompt generation uses `1 token ≈ 4 chars` as a rough heuristic. But the
actual token count depends on the model's tokenizer. A 512-token target might
produce 300 or 700 actual tokens. The input length reported in results
(`"input_len": 512`) is misleading — it's a character approximation, not what
the model actually processed.

Two paths to fix this:
- **Quick:** Hit `/v1/completions` with the prompt and check
  `usage.prompt_tokens` in a calibration request before the benchmark.
- **Proper:** Use the model's tokenizer (available via the vLLM `/tokenize`
  endpoint) to generate prompts that are *exactly* N tokens.

**Effort:** Small (quick path) / Medium (proper path).

### 4. 429 rate-limit handling

The retry logic only retries on `5xx` and network errors. If vLLM returns
`429 Too Many Requests` (which it can under load), it's counted as a failure.
This silently inflates the failure count and deflates TPS.

**Effort:** Trivial — add `resp.status === 429` to the retry condition.

---

## Tier 2 — Reliability on Long Runs

An `--all` run can take hours. Small reliability gaps become real problems at
that scale.

### 5. Container teardown race condition

When running `--all`, each model does `stopServer` → `startServer` on the same
port. `docker rm -f` returns before the port is fully released. The next
`docker run` on the same container name can fail with "name already in use" or
the port check can see a ghost listener.

A short settle delay (1-2 seconds) after `docker rm -f` and before the next
`startServer` would prevent this.

**Effort:** Trivial — add `await Bun.sleep(2000)` after `stopServer` in the
model loop.

### 6. Pre-flight disk space check

Models are 10-40GB+. The tool checks if the model is already cached, but if
it's *not* cached, there's no check for available disk space. A download that
fails at 30/40GB because the disk is full wastes time and produces a confusing
error.

A simple check: `if model not cached and disk_available_gb < estimated_model_size → warn early`.

**Effort:** Small — `models.yaml` could add an optional `size_gb` field, or
fetch from HuggingFace API.

### 7. Exponential backoff on retries

The retry delay is a flat 2 seconds. If the server is genuinely overloaded
(which is likely during a benchmark at high concurrency), hammering it again
in 2 seconds may just get another failure. Exponential backoff (2s → 4s → 8s)
would give the server breathing room.

**Effort:** Small — replace `RETRY_DELAY_MS` with `RETRY_BASE_DELAY_MS * 2^(maxRetries - retries)`.

---

## Tier 3 — Architecture

### 8. `benchmark.ts` is a god module

It's ~860 lines handling dependency checking, Docker orchestration, model
downloading, port management, result writing, and summary printing. These
should be separate modules:

```
src/
  docker.ts       # Docker sudo detection, pull, run, rm, logs
  download.ts     # Model cache check + huggingface download
  deps.ts         # Dependency checking + auto-install
  results.ts      # Write result JSON + print summary
  benchmark.ts    # Orchestration only
```

This would also make it possible to unit test each concern in isolation.

**Effort:** Medium — mechanical refactoring, no logic changes.

### 9. No tests

For a benchmarking tool where accuracy matters, this is significant. At
minimum:

- `calculateCost()` — pure function, trivial to test
- `getQuantFlags()` — pure function, trivial to test
- `formatCost()` / `parseCost()` — edge cases with null/NaN
- `loadModels()` — valid YAML, invalid YAML, missing fields
- `BenchmarkMetrics` computation from known `RequestResult[]`
- Prompt generation determinism (seeded PRNG produces same output)

**Effort:** Medium — set up test runner, write tests for pure functions first.

### 10. Module-level mutable state

`activeContainer`, `isCleaningUp`, and `_dockerNeedsSudo` are module-level
mutable variables. They make the code harder to reason about and impossible to
test in isolation. Wrapping them in a class (e.g., `DockerManager`) or passing
them through a context object would be cleaner.

**Effort:** Medium — tied to the `benchmark.ts` split (#8).

---

## Tier 4 — Usability Gains

### 11. Run comparison

`summarize` shows one run at a time. Being able to diff two runs would be very
valuable:

```bash
gpu-benchmark compare results-a100/ results-h100/
```

This would show side-by-side TPS and cost for each model across the two runs.

**Effort:** Medium — new command, match models by ID, render side-by-side table.

### 12. Config file

Every option is a CLI flag. For repeated benchmarking with the same settings,
a `.gpu-benchmark.yaml` would be more ergonomic:

```yaml
gpu_rate: 2.00
gpu_count: 1
concurrency: 32
docker_image: vllm/vllm-openai:v0.8.1
stream: true
```

CLI flags would override config file values.

**Effort:** Small — read YAML, merge with CLI opts (commander supports this).

### 13. Confidence intervals

Running once gives a single number. Running N repetitions and reporting
mean ± stddev would give users confidence in the results. A `--repeat N`
flag, with the result JSON including `mean_tps`, `stddev_tps`, and `min/max`.

**Effort:** Medium — repeat the benchmark N times, aggregate metrics, extend
the result schema.

---

## Top 3 Picks (if time is limited)

| Priority | What | Why |
|----------|------|-----|
| 1 | **Warmup phase** | Single biggest impact on number accuracy. Easy to implement. |
| 2 | **TTFT measurement** | Fills a critical metric gap. Users need this for real-world decisions. |
| 3 | **Split `benchmark.ts`** | Everything above becomes easier when orchestration isn't 860 lines. |
