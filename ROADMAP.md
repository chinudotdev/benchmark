# Roadmap — GPU-Benchmark → Multi-Platform Inference Benchmarking Framework

> Comparative evaluation of Tenstorrent Wormhole accelerators, AMD accelerators, and traditional NVIDIA GPUs for AI inference workloads.

This roadmap maps what the current tool needs to become in order to satisfy the GPU.NET × Tenstorrent Inference Benchmarking Framework specification, extended to include AMD devices.

---

## Current State

The tool is a **single-platform GPU benchmark harness**:

- ✅ NVIDIA GPU detection, Docker orchestration, vLLM serving
- ✅ Concurrent request load generation with streaming support
- ✅ TPS, E2E latency (p50/p95/p99), cost-per-token calculation
- ✅ Seeded PRNG for deterministic prompts
- ✅ Result JSON persistence + summarize command
- ✅ Configurable retries, quantization, tensor parallelism

**What it does not yet do:** TTFT/TPOT measurement, SLA bands, quality gates, workload matrix sweeps, concurrency sweeps, multi-device scaling, cold-start recording, repeat runs, Tenstorrent support, AMD support, crossover comparison, or structured report generation.

---

## Phase 1 — Make Existing GPU Numbers Correct and Complete

> These affect the accuracy of every number the tool produces. If the numbers are wrong, nothing else matters.

### 1.1 Warmup phase

The first few requests to a freshly loaded model are slower — cold KV cache, JIT compilation, memory allocation settling. Those requests pollute the benchmark.

**What to build:** Send N warmup requests (discarded from measurement) before the timed run. Default N=5. The warmup request count, timing, and discard are recorded in the result JSON.

```
"warmup": {
  "requests": 5,
  "total_time_s": 12.3,
  "discarded": true
}
```

**Effort:** Small — send N requests before the timed run, discard results.

### 1.2 TTFT (Time to First Token)

Critical real-world metric that is completely absent. For streaming use cases, users care deeply about how long before the first token appears, separate from overall throughput.

**What to build:** In streaming mode, capture a high-resolution timestamp when the first `data:` chunk containing a token arrives. Report TTFT p50/p95/p99 as first-class metrics alongside E2E latency.

```
"ttft_s": { "p50": 0.12, "p95": 0.31, "p99": 0.48 },
```

For non-streaming mode, TTFT ≈ E2E latency (the server buffers everything). Log a note when TTFT is not meaningful.

**Effort:** Medium — timestamp capture on first SSE token chunk.

### 1.3 TPOT / ITL (Time Per Output Token / Inter-Token Latency)

The steady-state decode speed, measured as the gap between consecutive tokens during streaming.

**What to build:** In streaming mode, capture timestamps for each token chunk. Compute inter-arrival times. Report TPOT p50/p95/p99. Derive `tokens/sec/user = 1 / median_ITL`.

```
"tpot_s": { "p50": 0.008, "p95": 0.012, "p99": 0.025 },
"tokens_per_sec_per_user": 125.0,
```

**Effort:** Medium — per-chunk timing in the SSE reader.

### 1.4 SLA bands

The framework defines three named SLA bands (aligned with MLPerf Inference). Every latency number must be evaluated against them.

| Band | TTFT target (P99) | TPOT target (P99) | Use case |
|------|--------------------|--------------------|----------|
| Interactive | ≤ 500 ms | ≤ 30 ms | Chatbots, coding assistants, agent loops |
| Conversational | ≤ 2 s | ≤ 100 ms | General assistant / Q&A |
| Batch / Offline | No bound | No bound | Bulk processing |

**What to build:**

- Define the three bands as constants.
- After each run, classify each request against all three bands.
- Report throughput at each SLA (the "operating-point" number).
- A request "meets" a band if its TTFT ≤ target AND its TPOT ≤ target.
- `--sla-band <name>` flag to enforce a specific band and reject/fail requests that don't meet it.

```
"sla": {
  "interactive": { "met": 142, "total": 200, "throughput_tps": 2240.5 },
  "conversational": { "met": 198, "total": 200, "throughput_tps": 3150.2 },
  "batch": { "met": 200, "total": 200, "throughput_tps": 3200.5 }
}
```

**Effort:** Medium — metric classification + new CLI flag.

### 1.5 Goodput

Throughput counting only requests that met the SLA. Failed-SLA requests do not count.

**What to build:** For each SLA band, compute `goodput = (tokens from SLA-meeting requests) / total_time`. This is the headline number — not raw TPS.

**Effort:** Small — filter on top of SLA classification.

### 1.6 Cold-start / warmup time

Model load + kernel compilation time before the first useful request.

**What to build:** Record the wall-clock time from `docker run` to the first successful `/health` response. Report separately from steady-state metrics.

```
"cold_start_s": 47.3,
```

For Tenstorrent, kernel compilation can be multi-minute — this is even more critical.

**Effort:** Small — timestamp before container start, timestamp on health check, compute delta.

### 1.7 Concurrency sweep

The framework requires a throughput-vs-concurrency curve — "tokens/second as concurrent request count rises, until the SLA breaks."

**What to build:** `--concurrency-sweep <start>-<end>-<step>` flag. Runs the benchmark at each concurrency level, records throughput at SLA. Produces a curve. The knee of this curve is the recommended operating point.

```bash
gpu-benchmark run --model-id Qwen/Qwen3-8B --concurrency-sweep 1,2,4,8,16,32,64,128
```

Output: array of `{ concurrency, tps, goodput_interactive, goodput_conversational, ttft_p99, tpot_p99 }`.

**Effort:** Medium — loop over concurrency levels, aggregate results.

### 1.8 Sequence-length matrix sweep

The framework defines 5 sequence-length profiles. Currently `--input-len` and `--output-len` are single values.

**What to build:** `--seq-profile <name>` or `--seq-sweep` flag. Define the 5 profiles from the framework:

| Profile | Input tokens | Output tokens | Stresses |
|---------|-------------|--------------|----------|
| short-chat | 128 | 128 | Balanced |
| long-input-rag | 4,000 | 256 | Prefill-heavy |
| long-output-reasoning | 512 | 4,000 | Decode-heavy |
| very-long-context | 16,000 | 64 | KV-cache pressure |
| summarisation | 770 | 73 | MLPerf CNN/DailyMail |

`--seq-sweep` runs all 5 profiles for the given model.

**Effort:** Medium — named profiles, loop over them, aggregate results.

### 1.9 N ≥ 3 repeats with confidence intervals

Every reported figure should be the result of N≥3 runs; mean and variance both reported.

**What to build:** `--repeat N` flag. Run the entire benchmark N times. Result JSON includes:

```
"repeat": {
  "n": 3,
  "tps_mean": 3200.5, "tps_stddev": 45.2, "tps_min": 3150.1, "tps_max": 3255.8,
  "ttft_p99_mean": 0.48, "ttft_p99_stddev": 0.03
}
```

**Effort:** Medium — outer loop, aggregation, extended result schema.

### 1.10 Input token count accuracy

Prompt generation uses `1 token ≈ 4 chars`. The actual token count depends on the model's tokenizer.

**What to build:** Before the benchmark, send a calibration request to the vLLM `/tokenize` endpoint (or check `usage.prompt_tokens` from a warmup request). Report actual vs. target token count. Optionally, generate prompts that are exactly N tokens using the model's tokenizer.

```
"input_tokens_actual": { "target": 512, "mean": 498, "stddev": 42 },
```

**Effort:** Small (quick: calibration request) / Medium (proper: tokenizer-based generation).

---

## Phase 2 — Multi-Platform Software Stack

> One interface, three backends. Both NVIDIA, AMD, and Tenstorrent are driven through an identical OpenAI-compatible serving API so the load generator and quality gates are the same code. Only the serving backend and hardware differ.

### Architecture Overview

| Layer | NVIDIA GPU | AMD GPU | Tenstorrent |
|-------|-----------|---------|-------------|
| **Hardware** | NVIDIA A100/H100/L40S etc. | AMD Instinct MI300X/MI250X etc. | Wormhole n150 / n300 |
| **Low-level runtime** | CUDA / cuDNN | ROCm / MIOpen | TT-Metalium / TT-NN |
| **Model implementation** | Native PyTorch / HF | Native PyTorch / HF | tt-transformers / tt-metal demos |
| **Serving layer** | Upstream vLLM (or TensorRT-LLM) | vLLM ROCm fork | Tenstorrent vLLM fork |
| **Container runtime** | Docker + NVIDIA container toolkit | Docker + ROCm container toolkit | Docker or tt-inference-server |
| **API surface** | OpenAI-compatible HTTP | OpenAI-compatible HTTP | OpenAI-compatible HTTP |
| **Load generator** | Shared — same tool, same config | Shared — same tool, same config | Shared — same tool, same config |
| **Quality gate** | Shared — same harness, same datasets | Shared — same harness, same datasets | Shared — same harness, same datasets |

### 2.1 Platform abstraction

**What to build:** A `Platform` interface that encapsulates hardware detection, Docker/container orchestration, and serving backend configuration for each vendor.

```typescript
interface Platform {
  name: string;                    // "nvidia" | "amd" | "tenstorrent"
  detectHardware(): Promise<HardwareInfo>;
  getContainerConfig(model, opts): ContainerConfig;
  getDockerImage(opts): string;
  healthCheckUrl(port): string;
}
```

Implementations:
- `NvidiaPlatform` — current code (nvidia-smi, nvidia-container-toolkit, upstream vLLM)
- `AmdPlatform` — new (rocm-smi, ROCm container toolkit, vLLM ROCm fork)
- `TenstorrentPlatform` — new (TT hardware detection, tt-inference-server / TT vLLM fork)

`--platform <nvidia|amd|tenstorrent>` CLI flag (auto-detected if only one is present).

**Effort:** Medium — refactor existing NVIDIA code into a class, add two new implementations.

### 2.2 AMD GPU support

**What to build:**

- **Hardware detection:** `rocm-smi` for GPU name, VRAM, driver version. Parse output similar to `nvidia-smi`.
- **Container runtime:** ROCm container toolkit (passthrough via `--device /dev/kfd --device /dev/dri`). Detect if ROCm toolkit is installed.
- **Serving backend:** vLLM ROCm fork (docker image: `rocm/vllm-dev` or `vllm/vllm-openai:rocm`). Supports the same OpenAI-compatible API.
- **vLLM flags:** May need AMD-specific flags (e.g., `--dtype float16`, `--enforce-eager` for some MI-series GPUs).
- **Quantization:** AMD supports FP8 on MI300X, BF16 on MI250X. Update quantization validation to be platform-aware.
- **System info:** Extend `SystemInfo` to include ROCm version alongside CUDA version.

```typescript
interface AmdGpuInfo {
  name: string;           // e.g. "AMD Instinct MI300X"
  vramGb: number;
  rocmVersion: string;
  gfxArch: string;        // e.g. "gfx942"
}
```

**Effort:** Medium — mirror the NVIDIA path with ROCm equivalents.

### 2.3 Tenstorrent support

**What to build:**

- **Hardware detection:** Probe for Wormhole n150 / n300 via TT tools or PCIe device IDs.
- **Serving backend:** Integrate `tt-inference-server` or Tenstorrent's vLLM fork. Both expose an OpenAI-compatible HTTP endpoint.
- **Model bring-up:** Use `tt-transformers` model definitions. Record which TT-verified model list was used.
- **Cold-start handling:** TT kernel compilation can be multi-minute. Record this separately and exclude from steady-state metrics.
- **No Docker path (potentially):** TT may run natively or via `tt-inference-server` rather than Docker. The platform abstraction should handle this.

**Effort:** Large — new platform, likely requires TT SDK integration and testing on actual hardware.

### 2.4 GPU reference set

The framework requires comparison against three GPU tiers. Extend `models.yaml` or add a `platforms.yaml`:

| Tier | NVIDIA | AMD |
|------|--------|-----|
| Cost-segment | L40S-class | MI210-class |
| Mainstream datacenter | A100-class | MI250X-class |
| Flagship | H100/H200-class | MI300X-class |

**Effort:** Small — config + provisioning, not code.

### 2.5 Traffic profiles

The framework defines four traffic shapes. Only concurrent-fire currently exists.

| Profile | Shape | What to build |
|---------|-------|---------------|
| Single-stream | 1 request at a time, no batching | `--traffic single-stream` — concurrency=1, sequential |
| Interactive serving | Poisson request arrivals, concurrency swept, Interactive SLA enforced | `--traffic interactive` — Poisson inter-arrival times, concurrency sweep |
| High-concurrency serving | Saturating concurrent load, Conversational SLA enforced | `--traffic high-concurrency` — current behavior + SLA enforcement |
| Offline batch | Large fixed set, no latency bound, max throughput | `--traffic offline-batch` — maximize batch size, no SLA filter |

**Effort:** Medium — Poisson arrival generator + profile definitions.

---

## Phase 3 — Quality Gate & Comparison

> A faster token is worthless if it is a worse token.

### 3.1 lm-evaluation-harness integration

**What to build:** After each benchmark run, run `lm-evaluation-harness` against the serving endpoint on standard datasets. Record accuracy metrics. A platform's results are only valid if accuracy is within tolerance of the reference.

```
"quality": {
  "harness": "lm-evaluation-harness",
  "datasets": ["hellaswag", "arc_challenge", "truthfulqa_mc2"],
  "scores": { "hellaswag": 0.78, "arc_challenge": 0.52 },
  "reference_scores": { "hellaswag": 0.80, "arc_challenge": 0.53 },
  "passed": true
}
```

**Effort:** Medium — shell out to lm-evaluation-harness, parse output, compare to reference.

### 3.2 Numerical parity check

**What to build:** For a fixed prompt set with greedy decoding, compare output tokens (and optionally logits) against a reference implementation. Record token-level match rate.

```
"parity": {
  "prompts_tested": 50,
  "exact_match": 49,
  "match_rate": 0.98,
  "precision_used": "BF16"
}
```

**Effort:** Medium — fixed prompt set, greedy decode, token comparison.

### 3.3 Precision disclosure

**What to build:** Record the actual runtime data format (FP8, BF16, BlockFP8, INT8) used on each platform. Not just the quantization flag — the actual compute dtype reported by the serving backend.

```
"precision": {
  "compute_dtype": "BF16",
  "quantization": "none",
  "weight_dtype": "BF16"
}
```

**Effort:** Small — query the serving backend or infer from flags.

### 3.4 Crossover analysis

The central output of the comparison — for each model, a chart shading the workload space by which platform delivers higher throughput at the SLA.

**What to build:** Given results from all three platforms across the sequence-length matrix and concurrency sweep, generate:

- Throughput-at-SLA comparison tables (platform × model × seq-profile × SLA-band)
- Crossover maps: for each (model, seq-profile), plot throughput vs concurrency for each platform
- Highlight regions where each platform wins

Output format: JSON data + markdown tables (charts can be added later with a plotting library).

**Effort:** Large — data aggregation, comparison logic, report generation.

### 3.5 Multi-device scaling efficiency

**What to build:** For models that support tensor parallelism, sweep `--tensor-parallel-size` across available devices. Compute scaling efficiency = `throughput(N devices) / (N × throughput(1 device))`.

```
"scaling": {
  "devices": [1, 2, 4],
  "throughput_tps": [3200, 5900, 10800],
  "efficiency_pct": [100, 92, 84]
}
```

**Effort:** Medium — outer loop over device counts, efficiency calculation.

---

## Phase 4 — Reporting & Reproducibility

### 4.1 Structured benchmark report

**What to build:** A `gpu-benchmark report` command that produces the joint customer-facing document defined in the framework (Section 10.1):

- Executive summary — headline latency and throughput findings with conditions
- Methodology — condensed version of the framework spec
- Per-model results — Tier 1–3 metrics for each model, each SLA band, all platforms
- Crossover maps — where each platform wins
- Honest limitations — software-maturity gaps, unsupported features, deferred scope
- Reproducibility appendix — pinned commits, container digests, configs

Output: Markdown document suitable for PDF conversion.

**Effort:** Large — template + data aggregation.

### 4.2 Pinned version recording

**What to build:** Record the full software stack provenance per run:

```json
"versions": {
  "vllm": "v0.8.1",
  "container_digest": "sha256:abc123...",
  "cuda": "12.2",
  "rocm": "6.1.0",
  "tt_metal": "abc1234",
  "driver": "535.129.03",
  "benchmark_tool": "gpu-benchmark@abc1234"
}
```

**Effort:** Small — extend system info collection.

### 4.3 Run comparison command

**What to build:** `gpu-benchmark compare <dir1> <dir2> [dir3]` — side-by-side TPS, cost, latency for each model across result sets. Supports comparing NVIDIA vs AMD vs Tenstorrent results.

```
gpu-benchmark compare results-a100/ results-mi300x/ results-wormhole/
```

**Effort:** Medium — new command, match models by ID, render comparison table.

### 4.4 Reproducible artifact bundle

**What to build:** `gpu-benchmark export` — package everything needed to reproduce:

- The orchestration layer + pinned config
- All run configurations and pinned software versions
- Full results database (every run, every repeat, raw logs)
- Instructions to reproduce any published figure

Output: `.tar.gz` archive.

**Effort:** Medium — archive assembly + reproducibility instructions.

### 4.5 Config file support

**What to build:** `.gpu-benchmark.yaml` for repeated benchmarking with the same settings:

```yaml
platform: nvidia
gpu_rate: 2.00
gpu_count: 1
concurrency: 32
docker_image: vllm/vllm-openai:v0.8.1
stream: true
seq_sweep: true
repeat: 3
```

CLI flags override config file values.

**Effort:** Small — read YAML, merge with CLI opts.

---

## Phase 5 — Small Fixes & Reliability

These don't map to the framework spec directly but affect reliability on long multi-model runs.

### 5.1 Container teardown race condition

`docker rm -f` returns before the port is fully released. Next `docker run` can fail.

**Fix:** Add `await Bun.sleep(2000)` after `stopServer` in the model loop.

**Effort:** Trivial.

### 5.2 Pre-flight disk space check

No check for available disk space before downloading a 40GB model.

**Fix:** Add optional `size_gb` field to `models.yaml`. Warn if `disk_available_gb < size_gb * 1.1`.

**Effort:** Small.

### 5.3 Exponential backoff on retries

Flat 2s retry delay. Server under load benefits from backoff.

**Fix:** `delay = RETRY_BASE_DELAY_MS * 2^(maxRetries - retries)`.

**Effort:** Small.

### 5.4 429 rate-limit handling

Only retries on 5xx. vLLM can return 429 under load.

**Fix:** Add `resp.status === 429` to the retry condition.

**Effort:** Trivial.

### 5.5 Split `benchmark.ts`

~860-line god module. Split into:

```
src/
  docker.ts       # Docker sudo detection, pull, run, rm, logs
  download.ts     # Model cache check + huggingface download
  deps.ts         # Dependency checking + auto-install
  results.ts      # Write result JSON + print summary
  platforms/
    nvidia.ts     # NVIDIA platform implementation
    amd.ts        # AMD platform implementation
    tenstorrent.ts # Tenstorrent platform implementation
  benchmark.ts    # Orchestration only
```

**Effort:** Medium — mechanical refactoring.

### 5.6 Tests

For a benchmarking tool where accuracy matters, tests are essential:

- `calculateCost()` — pure function, trivial to test
- `getQuantFlags()` — pure function, trivial to test
- `formatCost()` / `parseCost()` — edge cases with null/NaN
- `loadModels()` — valid YAML, invalid YAML, missing fields
- `BenchmarkMetrics` computation from known `RequestResult[]`
- Prompt generation determinism (seeded PRNG produces same output)
- SLA classification logic
- Goodput computation

**Effort:** Medium.

---

## Implementation Priority

If time is limited, tackle in this order:

| Priority | What | Why | Phase |
|----------|------|-----|-------|
| 1 | **Warmup phase** | Single biggest impact on number accuracy. Easy. | 1 |
| 2 | **TTFT + TPOT measurement** | Fills the critical metric gap the framework requires. | 1 |
| 3 | **SLA bands + Goodput** | Without these, throughput numbers are meaningless in the framework context. | 1 |
| 4 | **Cold-start recording** | Trivial to add, critical for TT comparison later. | 1 |
| 5 | **Platform abstraction** | Required before adding AMD or Tenstorrent backends. | 2 |
| 6 | **AMD GPU support** | Extends addressable hardware. Shares most code with NVIDIA path. | 2 |
| 7 | **Concurrency sweep** | Required for the throughput-vs-concurrency curve. | 1 |
| 8 | **Sequence-length matrix** | Required for the full workload definition. | 1 |
| 9 | **Tenstorrent support** | Requires hardware access. Architecture is ready from step 5. | 2 |
| 10 | **Quality gate** | Makes all performance numbers valid. | 3 |
| 11 | **N ≥ 3 repeats** | Required for defensible numbers. | 1 |
| 12 | **Crossover analysis + report** | The final deliverable. Depends on everything above. | 4 |
