# gpu-benchmark

CLI tool for benchmarking LLM inference across GPU platforms (NVIDIA, AMD, Tenstorrent).

Single static Go binary — 6.5MB, zero runtime dependencies. Measures TTFT, TPOT, SLA compliance, goodput, and cost-per-token through full-matrix sweeps.

## Quick Start

### Build

```bash
make build
# Cross-compile for all platforms:
make cross-compile
```

### Run a benchmark

```bash
# Benchmark a single model on an A100
gpu-benchmark run --model-id Qwen/Qwen3-8B --gpu-rate 2.00 --stream

# Run all models from models.yaml
gpu-benchmark run --all

# Full sweep: all seq profiles × concurrency levels × 3 repeats
gpu-benchmark run --model-id Qwen/Qwen3-8B --seq-sweep \
  --concurrency-sweep 1,2,4,8,16,32,64,128 --repeat 3
```

### View results

```bash
# Cost comparison table
gpu-benchmark summarize

# Structured Markdown report
gpu-benchmark report

# Export reproducible artifact
gpu-benchmark export

# Compare two runs
gpu-benchmark compare results-a100 results-h100 --crossover --sla
```

### Pre-flight checks

```bash
gpu-benchmark preflight
```

## Commands

| Command | Description |
|---------|-------------|
| `run` | Run GPU benchmarks against one or all models |
| `summarize` | Print cost comparison table (table/csv/json) |
| `sysinfo` | Display system configuration |
| `report` | Generate Markdown report from results |
| `export` | Export results as reproducible tar.gz |
| `compare` | Compare results from two directories |
| `quality` | Run quality gate (lm-evaluation-harness) |
| `preflight` | Run pre-benchmark checks |

## Configuration

Create `.gpu-benchmark.yaml` in your project directory or home directory:

```yaml
gpu-rate: 2.00
concurrency: 32
stream: true
repeat: 3
# concurrency-sweep: "1,2,4,8,16,32,64,128"
# seq-sweep: false
# platform: nvidia
```

CLI flags always override config file values.

## Metrics

The tool measures:

| Metric | Description |
|--------|-------------|
| **Output TPS** | Total output tokens/second |
| **TTFT** | Time to first token (p50/p95/p99) |
| **TPOT** | Time per output token (p50/p95/p99) |
| **t/s/user** | Tokens per second per concurrent user |
| **RPS** | Requests per second |
| **Cold start** | Container start → first healthy response |
| **Cost/1M tok** | Estimated cost per 1M output tokens |
| **SLA compliance** | Goodput at Interactive/Conversational/Batch bands |
| **Quality** | lm-eval accuracy scores (MMLU, HumanEval, GSM8K) |

### SLA Bands

| Band | TTFT P99 | TPOT P99 | Use case |
|------|----------|----------|----------|
| Interactive | ≤500ms | ≤30ms | Real-time chat |
| Conversational | ≤2s | ≤100ms | Assistant apps |
| Batch | No limit | No limit | Offline processing |

### Sequence Profiles

| Profile | Input | Output | Description |
|---------|-------|--------|-------------|
| short-chat | 128 | 128 | Common chat traffic |
| long-input-rag | 4000 | 256 | Prefill-heavy retrieval |
| long-output-reasoning | 512 | 4000 | Decode-heavy generation |
| very-long-context | 16000 | 64 | KV-cache pressure test |
| summarisation | 770 | 73 | MLPerf CNN/DailyMail |

## Architecture

```
cmd/gpu-benchmark/       CLI (cobra)
internal/
├── platform/            Platform interface (NVIDIA/AMD/Tenstorrent)
├── benchmark/           Load generator, metrics, sweeps, prompts
├── workload/            Model registry, seq/traffic profiles
├── quality/             Quality gate (lm-evaluation-harness)
├── orchestrator/        Pipeline: detect → download → serve → bench → result
├── report/              JSON persistence, Markdown, CSV, compare, export
├── config/              .gpu-benchmark.yaml loading
├── docker/              Container lifecycle
├── download/            HuggingFace model cache
└── sysinfo/             System info collection, preflight checks
```

## Platform Support

| Platform | Status | Detection | Docker |
|----------|--------|-----------|--------|
| NVIDIA | ✅ Working | `nvidia-smi` | `--gpus all` + vLLM |
| AMD | 🔧 Stub | `rocm-smi` | `--device /dev/kfd` + vLLM ROCm |
| Tenstorrent | 🔧 Stub | `tt-smi` | tt-inference-server |

## Development

```bash
make build          # Build binary
make test           # Run tests
make cross-compile  # Build for linux/amd64, linux/arm64, darwin/arm64
make tidy           # Tidy modules
make bench          # Dry-run benchmark
```

## Dependencies

| Library | Purpose |
|---------|---------|
| github.com/spf13/cobra | CLI framework |
| gopkg.in/yaml.v3 | Model registry & config parsing |

No HTTP frameworks. stdlib `net/http` for requests, `os/exec` for Docker.

## License

MIT
