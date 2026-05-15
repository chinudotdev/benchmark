# Go Rewrite Plan — gpu-benchmark

Branch: `rewrite/go`

## Goal

Rewrite the Bun/TypeScript CLI as a Go binary that supports NVIDIA, AMD, and Tenstorrent platforms with the full metric suite (TTFT, TPOT, SLA bands, goodput, sweeps, quality gates).

## Principles

- **Single static binary** — `go build` produces one file, zero runtime deps
- **Platform interface** — NVIDIA/AMD/Tenstorrent behind a common `Platform` contract
- **OpenAI-compatible API only** — the load generator never cares what's behind the endpoint
- **Matrix-driven runs** — model × profile × seq-length × concurrency, not single configs
- **Preserve the old code** — the `src/` TypeScript code stays on `main` for reference until the Go version is feature-complete

## Dependencies (minimal)

| Library | Purpose |
|---------|---------|
| github.com/spf13/cobra | CLI framework |
| github.com/spf13/viper | Config file (.gpu-benchmark.yaml) |
| gopkg.in/yaml.v3 | Model registry parsing |
| github.com/fatih/color | Colored terminal output |

No HTTP frameworks, no ORM, no bloat. stdlib `net/http` for requests, `encoding/json` for serialization, `os/exec` for process management.

---

## Milestone 1 — Skeleton + Single-Platform NVIDIA Run

> Replicate what the current Bun tool does today, but in Go. No new features yet — just prove the architecture works.

**Deliverables:**
- [x] Go module + project structure
- [ ] CLI with `run`, `summarize`, `sysinfo` commands (cobra)
- [ ] `Platform` interface with `nvidia` implementation
- [ ] NVIDIA hardware detection (nvidia-smi parsing)
- [ ] Docker container lifecycle (pull, run, wait healthy, stop, rm)
- [ ] Load generator: concurrent requests to OpenAI-compatible endpoint
- [ ] Metric collection: E2E latency p50/p95/p99, output TPS, request success/fail
- [ ] Result JSON persistence (same shape as current output for backward compat)
- [ ] Cost estimate calculation
- [ ] `sysinfo` command
- [ ] `summarize` command (table, csv, json)
- [ ] Model registry (models.yaml)
- [ ] Build via Makefile
- [ ] SIGINT/SIGTERM cleanup of containers

**Test:** `./gpu-benchmark run --model-id Qwen/Qwen3-8B --gpu-rate 2.00` produces the same result as the Bun version.

---

## Milestone 2 — Correct Metrics (TTFT, TPOT, SLA, Goodput)

> Make the numbers the framework requires.

**Deliverables:**
- [ ] Streaming SSE parser with per-token timestamp capture
- [ ] TTFT measurement (time to first token chunk)
- [ ] TPOT / ITL measurement (inter-token latency distribution)
- [ ] Tokens/sec/user derivation
- [ ] SLA band definitions (Interactive, Conversational, Batch)
- [ ] Goodput computation (throughput at each SLA)
- [ ] Warmup phase (N discarded requests before measurement)
- [ ] Cold-start time recording (container start → first healthy response)
- [ ] Requests/sec (RPS) metric
- [ ] Extended result JSON with all new metric fields
- [ ] Updated `summarize` to show TTFT/TPOT/SLA breakdown

**Test:** A streaming run reports TTFT p99, TPOT p99, and goodput at all three SLA bands.

---

## Milestone 3 — Workload Matrix + Sweeps

> From single-config runs to full matrix execution.

**Deliverables:**
- [ ] Sequence-length profiles (5 profiles from framework spec)
- [ ] `--seq-sweep` flag to run all profiles
- [ ] Concurrency sweep (`--concurrency-sweep 1,2,4,8,16,32,64,128`)
- [ ] Traffic profiles (single-stream, interactive, high-concurrency, offline-batch)
- [ ] Fixed prompt dataset loader (replace random word generator)
- [ ] Token count verification via `/tokenize` endpoint
- [ ] `--repeat N` with mean ± stddev aggregation
- [ ] Matrix orchestration: model × seq-profile × concurrency × traffic-profile
- [ ] Matrix result storage (per-cell JSON files)

**Test:** `gpu-benchmark run --model-id Qwen/Qwen3-8B --seq-sweep --repeat 3` produces results for all 5 seq profiles with confidence intervals.

---

## Milestone 4 — Multi-Platform (AMD + Tenstorrent)

> Two more backends behind the shared interface.

**Deliverables:**
- [ ] `--platform <nvidia|amd|tenstorrent>` flag (auto-detect if only one present)
- [ ] AMD platform: `rocm-smi` detection, ROCm container toolkit, vLLM ROCm fork
- [ ] AMD-specific system info (ROCm version, gfx architecture)
- [ ] Tenstorrent platform: hardware detection, tt-inference-server integration
- [ ] Tenstorrent-specific system info (TT-Metalium version, Wormhole variant)
- [ ] Per-platform model configuration in models.yaml
- [ ] Platform-aware quantization validation
- [ ] Platform-aware Docker flags (NVIDIA: `--gpus`, AMD: `--device /dev/kfd`, TT: native or Docker)

**Test:** `gpu-benchmark run --platform amd --model-id ...` and `--platform tenstorrent` work end-to-end.

---

## Milestone 5 — Quality Gate + Comparison

> Validate that faster tokens aren't worse tokens.

**Deliverables:**
- [ ] Quality gate interface
- [ ] lm-evaluation-harness runner (subprocess, parse results)
- [ ] Greedy-decode parity check (fixed prompts, token-level comparison)
- [ ] Precision disclosure (query serving backend for actual compute dtype)
- [ ] Quality pass/fail in result JSON
- [ ] Multi-platform comparison command (`gpu-benchmark compare <dir1> <dir2> [dir3]`)
- [ ] Side-by-side TPS, cost, latency comparison table
- [ ] Crossover analysis: throughput-at-SLA across workload space

**Test:** Two result directories produce a comparison showing which platform wins at each SLA band.

---

## Milestone 6 — Reporting + Polish

> Customer-facing deliverables.

**Deliverables:**
- [ ] Config file support (`.gpu-benchmark.yaml`)
- [ ] Pinned version recording (container digest, CUDA/ROCm/TT-Metalium commit, driver)
- [ ] Structured Markdown report generation (`gpu-benchmark report`)
- [ ] Reproducible artifact bundle (`gpu-benchmark export`)
- [ ] Pre-flight disk space check
- [ ] Exponential backoff on retries
- [ ] 429 rate-limit retry handling
- [ ] Container teardown settle delay
- [ ] Cross-compilation via Makefile (linux/amd64, linux/arm64, darwin/arm64)
- [ ] Unit tests for: cost calculation, metric computation, SLA classification, prompt loading, YAML parsing

---

## Project Structure

```
gpu-benchmark/
├── cmd/
│   └── gpu-benchmark/
│       └── main.go
├── internal/
│   ├── platform/
│   │   ├── platform.go
│   │   ├── nvidia.go
│   │   ├── amd.go
│   │   └── tenstorrent.go
│   ├── benchmark/
│   │   ├── runner.go
│   │   ├── metrics.go
│   │   ├── request.go
│   │   └── sweep.go
│   ├── workload/
│   │   ├── profiles.go
│   │   ├── prompts.go
│   │   └── matrix.go
│   ├── quality/
│   │   ├── gate.go
│   │   └── eval.go
│   ├── report/
│   │   ├── result.go
│   │   ├── summarize.go
│   │   └── compare.go
│   ├── config/
│   │   └── config.go
│   └── sysinfo/
│       └── sysinfo.go
├── configs/
│   └── models.yaml
├── go.mod
├── go.sum
├── Makefile
├── PLAN.md
├── ROADMAP.md
└── README.md
```

## Current focus: Milestone 1

We scaffold the project structure, implement the `Platform` interface, get the NVIDIA path working end-to-end with Docker orchestration, load generation, metric collection, and result persistence.
