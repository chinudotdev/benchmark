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

**Status: ✅ DONE** (commit da42f5c + a306e28)

---

## Milestone 2 — Correct Metrics (TTFT, TPOT, SLA, Goodput)

> Make the numbers the framework requires.

**Status: ✅ DONE** (shipped with Milestone 1 — all items below were implemented in the initial Go scaffold)

**Deliverables:**
- [x] Streaming SSE parser with per-token timestamp capture
- [x] TTFT measurement (time to first token chunk)
- [x] TPOT / ITL measurement (inter-token latency distribution)
- [x] Tokens/sec/user derivation
- [x] SLA band definitions (Interactive, Conversational, Batch)
- [x] Goodput computation (throughput at each SLA)
- [x] Warmup phase (N discarded requests before measurement)
- [x] Cold-start time recording (container start → first healthy response)
- [x] Requests/sec (RPS) metric
- [x] Extended result JSON with all new metric fields
- [x] Updated `summarize` to show TTFT/TPOT/SLA breakdown

**Bug fixes applied (from bugs.md review):**
- [x] BUG 2: `break` in select now uses labeled break to exit for-loop
- [x] BUG 4: Cold start uses `time.Since(start)` instead of accumulated intervals
- [x] BUG 5: `useStreamOpts` is now passed to `sendRequest` and conditionally included in payload
- [x] BUG 6: Single signal handler that does both `cancel()` and `dmgr.Cleanup()`
- [x] BUG 7: Backoff formula simplified to `2^(attempt)` where attempt = cfg.Retries - retries
- [x] BUG 8: `printTable` now has nil check on `r.Metrics`
- [x] BUG 9: `truncate` now uses rune-based slicing for safe UTF-8
- [x] BUG 10: `FixCachePermissions` now probes write access before running sudo chown
- [x] BUG 12: `parseVRAM` uses proper rounding instead of integer truncation
- [x] BUG 13: `Summarize` now returns errors from `LoadResults` instead of swallowing them
- [x] BUG 14: Warmup uses separate prompt set so benchmark prompts aren't pre-cached
- [x] BUG 15: SSE scanner buffer increased to 1MB
- [x] Minor: Removed duplicate `configs/models.yaml`
- [x] Minor: `go vet` passes clean

**Test:** A streaming run reports TTFT p99, TPOT p99, and goodput at all three SLA bands.

---

## Milestone 3 — Workload Matrix + Sweeps

> From single-config runs to full matrix execution.

**Status: ✅ DONE**

**Deliverables:**
- [x] Sequence-length profiles (5 profiles from framework spec)
- [x] `--seq-sweep` flag to run all profiles
- [x] Concurrency sweep (`--concurrency-sweep 1,2,4,8,16,32,64,128`)
- [x] Traffic profiles (single-stream, interactive, high-concurrency, offline-batch)
- [x] Fixed prompt dataset loader (190 real-world prompts, embedded in binary)
- [x] Token count verification via `/tokenize` endpoint
- [x] `--repeat N` with mean ± stddev aggregation
- [x] Matrix orchestration: model × seq-profile × concurrency × traffic-profile
- [x] Matrix result storage (per-cell JSON files in `<model>_sweep/` subdirectory)

**New files:**
- `internal/benchmark/prompts.go` — PromptDataset with embedded prompt corpus
- `internal/benchmark/prompts.txt` — 190 real-world prompts (embedded via go:embed)
- `internal/benchmark/tokenize.go` — /tokenize endpoint verification
- `internal/benchmark/sweep.go` — RunConcurrencySweep, RunSeqSweep, AggregateSweep
- `internal/report/sweep.go` — per-cell JSON storage, load, sort
- Tests: prompts_test.go (8), sweep_test.go (6), sweep_test.go (4)

**Test:** `gpu-benchmark run --model-id Qwen/Qwen3-8B --seq-sweep --repeat 3` produces results for all 5 seq profiles with confidence intervals.

---

## Milestone 4 — Multi-Platform (AMD + Tenstorrent)

> Two more backends behind the shared interface.

**Status: ✅ DONE (implementation complete, pending hardware validation)**

**Deliverables:**
- [x] `--platform <nvidia|amd|tenstorrent>` flag (auto-detect if only one present)
- [x] AMD platform: `rocm-smi` detection with 4-strategy fallback
  - Strategy 1: rocm-smi (preferred) — GPU name, VRAM, driver, ROCm version
  - Strategy 2: /sys/class/kfd — KFD topology, mem_banks, GFX arch
  - Strategy 3: lspci — PCI device ID matching for AMD GPUs
  - Strategy 4: /sys/class/drm — DRI device enumeration
- [x] AMD-specific system info: ROCm version, GFX architecture (gfx942, gfx90a, etc.)
- [x] AMD container config: `--device /dev/kfd`, `--device /dev/dri`, `--group-add video,render`, `HIP_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`
- [x] AMD runtime checks: amdgpu driver, /dev/kfd access, user group membership
- [x] Tenstorrent platform: tt-smi detection with 4-strategy fallback
  - Strategy 1: tt-smi -j (JSON output, preferred)
  - Strategy 2: tt-smi text parsing (board/chip/firmware/DRAM)
  - Strategy 3: tt-topology --list
  - Strategy 4: /sys/class/tenstorrent sysfs scan
  - Strategy 5: lspci with Tenstorrent vendor ID (0x1e52)
- [x] Tenstorrent-specific system info: board type (n150/n300), Wormhole variant, firmware, DRAM
- [x] Tenstorrent container config: `--device /dev/tenstorrent`, hugepage mounts
- [x] Tenstorrent native serving: `StartNativeServer()` for non-Docker path
- [x] Per-platform model configuration in models.yaml (platform, docker_image, serving_backend fields)
- [x] Platform-aware model filtering in orchestrator
- [x] Platform-aware quantization validation (shared quantFlags)
- [x] GFX architecture detection: sysfs properties, PCI ID mapping, HSA version conversion
- [ ] Hardware validation on real AMD MI300X and Tenstorrent Wormhole

**Detection coverage:**

| AMD Device | rocm-smi | /sys/class/kfd | lspci |
|------------|----------|----------------|-------|
| MI300X (gfx942) | ✓ | ✓ | ✓ 1002:740C |
| MI300A (gfx942) | ✓ | ✓ | ✓ 1002:7408 |
| MI250X (gfx90a) | ✓ | ✓ | ✓ 1002:738C |
| RX 7900 XTX | ✓ | ✓ | ✓ 1002:7480 |

| TT Device | tt-smi -j | tt-smi text | tt-topology | sysfs | lspci |
|-----------|-----------|-------------|-------------|-------|-------|
| WH n300 | ✓ | ✓ | ✓ | ✓ | ✓ 1e52:1b01 |
| WH n150 | ✓ | ✓ | ✓ | ✓ | ✓ 1e52:1b02 |

**New test coverage:** 28 new tests for AMD + Tenstorrent backends

---

## Milestone 5 — Quality Gate + Comparison

> Validate that faster tokens aren't worse tokens.

**Status: ✅ DONE (no-hardware portion)**

**Deliverables:**
- [x] Quality gate interface (`internal/quality/gate.go`)
- [x] lm-evaluation-harness runner (subprocess, parse JSON results)
- [x] Precision disclosure (queries server + reads model config.json)
- [x] Quality pass/fail in result JSON with configurable thresholds
- [x] Multi-platform comparison command (`gpu-benchmark compare <dir1> <dir2>`)
- [x] Side-by-side TPS, cost, latency comparison table
- [x] Crossover analysis: throughput-at-SLA across workload space
- [x] SLA-band goodput comparison (`--sla` flag)
- [ ] Greedy-decode parity check (requires running server)
- [ ] Full end-to-end test with real lm-eval (requires GPU hardware)

**New files:**
- `internal/quality/gate.go` + `gate_test.go` (6 tests)
- `internal/report/compare.go` + `compare_test.go` (8 tests)

**New CLI commands:**
- `gpu-benchmark compare <dir1> <dir2> [--crossover] [--sla] [--format json]`
- `gpu-benchmark quality --model-id <id> --port 8000`

**Quality gate thresholds (defaults):**
- MMLU ≥ 55%
- HumanEval ≥ 35%
- GSM8K ≥ 45%
- Min 1 task must pass

---

## Milestone 6 — Reporting + Polish

> Customer-facing deliverables.

**Status: ✅ DONE**

**Deliverables:**
- [x] Config file support (`.gpu-benchmark.yaml`) — loads from cwd up to $HOME
- [x] Pinned version recording (container digest, driver versions via system_info.json)
- [x] Structured Markdown report generation (`gpu-benchmark report`)
- [x] Reproducible artifact bundle (`gpu-benchmark export` → tar.gz with manifest.json)
- [x] Pre-flight disk space check (`gpu-benchmark preflight`)
- [x] Exponential backoff on retries (done in M2 bug fixes)
- [x] 429 rate-limit retry handling with Retry-After header parsing
- [x] Container teardown settle delay (2s, done in M2)
- [x] Cross-compilation via Makefile (linux/amd64, linux/arm64, darwin/arm64)
- [x] Unit tests for: cost calculation, metric computation, SLA classification, prompt loading, YAML parsing

**New files:**
- `internal/config/config.go` — .gpu-benchmark.yaml loader with defaults and search path
- `internal/report/markdown.go` — structured Markdown report with system info, results, sweep tables
- `internal/report/export.go` — tar.gz artifact bundling with manifest
- `internal/sysinfo/preflight.go` — Docker, disk, GPU toolkit, Python, HF token checks

**New CLI commands:**
- `gpu-benchmark report --results-dir <dir>` → report.md
- `gpu-benchmark export --results-dir <dir>` → tar.gz
- `gpu-benchmark preflight --min-disk-gb 50` → pass/fail checks

**Tests:** 76 total (7 config, 4 export, 3 markdown)
All pass. go vet clean.

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

## Current focus: Hardware validation on real AMD MI300X and Tenstorrent Wormhole

All 6 milestones are code-complete. Remaining work is validating on real hardware.
See `bugs.md` for known issues to address before hardware testing.
