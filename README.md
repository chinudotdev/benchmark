# LLM GPU Benchmark

Benchmark any HuggingFace LLM on any GPU and get cost per 1M tokens.

Single-binary CLI — compiles with `bun build --compile`.

## Quick Start

### 1. Install dependencies

```bash
# Docker with NVIDIA container toolkit
apt-get install -y docker.io jq curl
# nvidia-container-toolkit for GPU passthrough
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### 2. Build or run

```bash
# Option A: Build a standalone binary (57MB)
bun install
bun run build
./gpu-benchmark run --help

# Option B: Run directly with Bun (no build step)
bun run src/index.ts run --help
```

### 3. Benchmark a single model

```bash
# A100 at $2.00/hr
./gpu-benchmark run \
  --model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --gpu-rate 2.00

# Qwen3-14B with custom params
./gpu-benchmark run \
  --model-id Qwen/Qwen3-14B \
  --gpu-rate 2.00 \
  --input-len 512 \
  --output-len 256

# Gated model with HF token
./gpu-benchmark run \
  --model-id meta-llama/Llama-3.3-70B-Instruct \
  --quant awq \
  --hf-token hf_xxxx \
  --gpu-rate 2.00

# Specific GPU on a multi-GPU machine
./gpu-benchmark run \
  --model-id Qwen/Qwen3-8B \
  --gpu-ids '"device=0"' \
  --gpu-rate 2.00
```

### 4. Run all models in registry

```bash
./gpu-benchmark run --all --gpu-rate 2.00
```

### 5. Preview commands without running

```bash
./gpu-benchmark run --model-id Qwen/Qwen3-8B --dry-run
```

### 6. Print cost comparison table

```bash
./gpu-benchmark summarize
./gpu-benchmark summarize --format csv
./gpu-benchmark summarize --format json
```

### 7. Show current system configuration

```bash
# Pretty print
./gpu-benchmark sysinfo

# JSON output
./gpu-benchmark sysinfo --json
```

## Output

Each model run produces a JSON in `./results/`:

```json
{
  "model_id": "Qwen/Qwen3-8B",
  "model_name": "Qwen3-8B",
  "quant": "none",
  "gpu": "NVIDIA A100-SXM4-80GB",
  "gpu_count": 1,
  "gpu_hourly_rate_usd": 2.00,
  "benchmark": {
    "num_prompts": 200,
    "input_len": 512,
    "output_len": 256,
    "max_model_len": 8192
  },
  "results": {
    "output_tokens_per_sec": 3200.5,
    "cost_per_1m_output_tokens_usd": 0.1736
  },
  "system": {
    "gpus": [{ "name": "NVIDIA A100-SXM4-80GB", "vram_gb": 80, "driver_version": "535.129.03", "cuda_version": "12.2" }],
    "cpu": { "model": "AMD EPYC 7763 64-Core Processor", "cores": 64 },
    "ram_gb": 512,
    "os": "Ubuntu 22.04.3 LTS",
    "kernel": "5.15.0-91-generic",
    "docker_version": "24.0.7",
    "vllm_image": "vllm/vllm-openai:latest",
    "disk_available_gb": 340,
    "collected_at": "2026-05-11T10:00:00.000Z"
  },
  "timestamp": "2026-05-11T10:00:00.000Z"
}
```

A standalone `system_info.json` is also written to the results directory for quick reference.

## CLI Reference

```
gpu-benchmark run [options]

  --model-id <id>        HuggingFace model ID
  --all                  Run all models in models.yaml
  --gpu-rate <rate>      GPU hourly rate in USD (default: 2.00)
  --gpu-count <n>        Number of GPUs (default: 1)
  --input-len <tokens>   Input token length (default: 512)
  --output-len <tokens>  Output token length (default: 256)
  --num-prompts <n>      Number of requests (default: 200)
  --max-model-len <n>    Max context length (default: 8192)
  --port <n>             vLLM server port (default: 8000)
  --concurrency <n>      Concurrent benchmark requests (default: 32)
  --results-dir <path>   Output directory (default: ./results)
  --models-yaml <path>   Model registry file (default: ./models.yaml)
  --quant <q>           Quantization for --model-id: none, int8, int4, awq, fp8
  --hf-token <token>    HuggingFace token for gated models (or set HF_TOKEN)
  --docker-image <img>  Docker image (default: vllm/vllm-openai:latest)
  --gpu-ids <ids>       GPU IDs for Docker (default: all). e.g. "device=0"
  --stream              Use streaming requests for benchmark
  --dry-run              Print commands without executing

gpu-benchmark summarize [options]

  --results-dir <path>   Results directory (default: ./results)
  --format <fmt>         Output: table, csv, json (default: table)

gpu-benchmark sysinfo [options]

  --docker-image <img>  Docker image to display (default: vllm/vllm-openai:latest)
  --json                Output as JSON instead of pretty table
```

## Adding a New Model

Edit `models.yaml`:

```yaml
- id: mistralai/Mistral-Large-2
  name: Mistral-Large-2
  quant: fp8
  min_vram_gb: 30
  tp: 1
  extra_flags: ""
```

Then run `./gpu-benchmark run --all`.

## Cost Formula

```
cost_per_1M = (gpu_hourly_rate × gpu_count) / TPS / 3600 × 1,000,000
```

## A100 Notes

- A100 does NOT have FP8 tensor cores — use `int8` or `awq` quantization instead
- 70B models need AWQ/INT4 to fit in 80GB
- For multi-GPU runs, pass `--gpu-count 2` to adjust cost math

## Project Structure

```
src/
├── index.ts          CLI entry point (commander)
├── benchmark.ts      Main orchestration (Docker, GPU, download, bench, cost)
├── bench-runner.ts   HTTP benchmark client (concurrent fetch)
├── sysinfo.ts        System info collection (GPU, CPU, RAM, OS, Docker)
├── summarize.ts      Results table/CSV/JSON
├── models.ts         YAML model registry parser
├── semaphore.ts      Async concurrency limiter
├── log.ts            Colored logging
└── types.ts          Shared TypeScript interfaces
```

## Migration from Python/Shell

This project was migrated from Python + Bash to TypeScript + Bun. The old files (`run_benchmark.py`, `summarize.py`, `llm_benchmark.sh`) have been removed. Improvements over the original:

- **Single binary** — `bun build --compile` produces a standalone executable
- **Zero external Python deps** — `fetch` is native in Bun
- **Type-safe** — full TypeScript with strict mode
- **SIGINT cleanup** — Docker containers are stopped on Ctrl+C
- **Safe JSON** — `JSON.stringify` instead of shell heredoc construction
- **Safe Docker commands** — array arguments instead of string eval
