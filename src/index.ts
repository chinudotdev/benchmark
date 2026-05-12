#!/usr/bin/env bun
// ── CLI entry point ──────────────────────────────────────────────────────────
//
// gpu-benchmark — LLM GPU inference benchmark + cost calculator
//
// Usage:
//   gpu-benchmark run --model-id Qwen/Qwen3-8B --gpu-rate 2.00
//   gpu-benchmark run --all --gpu-rate 2.00
//   gpu-benchmark summarize
//   gpu-benchmark summarize --format csv

import { Command } from "commander";
import { resolve } from "node:path";
import { runBenchmarkCommand } from "./benchmark";
import { runSummarize } from "./summarize";
import { runSysInfo } from "./sysinfo";

const program = new Command();

program
  .name("gpu-benchmark")
  .description("LLM GPU inference benchmark + cost calculator")
  .version("1.0.0");

// ── run command ──────────────────────────────────────────────────────────────

program
  .command("run")
  .description("Run GPU benchmarks against one or all models")
  .option("--model-id <id>", "HuggingFace model ID to benchmark")
  .option("--all", "Run all models in models.yaml")
  .option("--gpu-rate <rate>", "Hourly GPU cost in USD", "2.00")
  .option("--gpu-count <n>", "Number of GPUs", "1")
  .option("--input-len <tokens>", "Input token length for benchmark", "512")
  .option("--output-len <tokens>", "Output token length for benchmark", "256")
  .option("--num-prompts <n>", "Number of prompts to send", "200")
  .option("--max-model-len <n>", "Max context length to load", "8192")
  .option("--port <n>", "Port for vLLM server", "8000")
  .option("--results-dir <path>", "Directory to write result JSONs", "./results")
  .option("--concurrency <n>", "Concurrent benchmark requests", "32")
  .option("--models-yaml <path>", "Path to models.yaml registry", "./models.yaml")
  .option("--quant <q>", "Quantization for single model: none, int8, int4, awq, fp8")
  .option("--hf-token <token>", "HuggingFace Hub token for gated models (or set HF_TOKEN env var)")
  .option("--docker-image <image>", "Docker image to use (default: vllm/vllm-openai:latest)")
  .option("--gpu-ids <ids>", "GPU IDs for Docker (default: all). e.g. \"device=0\" or \"0,1\"")
  .option("--stream", "Use streaming requests for benchmark (default: false)")
  .option("--force", "Re-run benchmarks even if results already exist")
  .option("--dry-run", "Print commands without executing")
  .action(async (opts) => {
    // Parse and validate numeric options
    const gpuRate = parseFloat(opts.gpuRate);
    const gpuCount = parseInt(opts.gpuCount, 10);
    const inputLen = parseInt(opts.inputLen, 10);
    const outputLen = parseInt(opts.outputLen, 10);
    const numPrompts = parseInt(opts.numPrompts, 10);
    const maxModelLen = parseInt(opts.maxModelLen, 10);
    const port = parseInt(opts.port, 10);
    const concurrency = parseInt(opts.concurrency, 10);

    const numericVals = { gpuRate, gpuCount, inputLen, outputLen, numPrompts, maxModelLen, port, concurrency };
    for (const [key, val] of Object.entries(numericVals)) {
      if (isNaN(val)) {
        console.error(`Error: invalid number for ${key}`);
        process.exit(1);
      }
    }

    await runBenchmarkCommand({
      modelId: opts.modelId,
      all: opts.all === true,
      gpuRate,
      gpuCount,
      inputLen,
      outputLen,
      numPrompts,
      maxModelLen,
      port,
      concurrency,
      resultsDir: resolve(opts.resultsDir),
      dryRun: opts.dryRun === true,
      modelsYaml: resolve(opts.modelsYaml),
      quant: opts.quant,
      hfToken: opts.hfToken ?? process.env.HF_TOKEN ?? process.env.HUGGING_FACE_HUB_TOKEN,
      dockerImage: opts.dockerImage,
      gpuIds: opts.gpuIds,
      stream: opts.stream === true,
      force: opts.force === true,
    });
  });

// ── summarize command ────────────────────────────────────────────────────────

program
  .command("summarize")
  .description("Print cost comparison table from benchmark results")
  .option("--results-dir <path>", "Directory with result JSONs", "./results")
  .option("--format <fmt>", "Output format: table, csv, json", "table")
  .action((opts) => {
    runSummarize(resolve(opts.resultsDir), opts.format);
  });

// ── sysinfo command ──────────────────────────────────────────────────────────

program
  .command("sysinfo")
  .description("Display current system configuration (GPU, CPU, RAM, OS, Docker)")
  .option("--docker-image <image>", "Docker image to display (default: vllm/vllm-openai:latest)")
  .option("--json", "Output as JSON instead of pretty table")
  .action(async (opts) => {
    await runSysInfo({
      dockerImage: opts.dockerImage,
      json: opts.json === true,
    });
  });

// ── Parse ────────────────────────────────────────────────────────────────────

program.parse();
