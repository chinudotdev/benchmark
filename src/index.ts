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
  .option("--dry-run", "Print commands without executing")
  .action(async (opts) => {
    await runBenchmarkCommand({
      modelId: opts.modelId,
      all: opts.all === true,
      gpuRate: parseFloat(opts.gpuRate),
      gpuCount: parseInt(opts.gpuCount, 10),
      inputLen: parseInt(opts.inputLen, 10),
      outputLen: parseInt(opts.outputLen, 10),
      numPrompts: parseInt(opts.numPrompts, 10),
      maxModelLen: parseInt(opts.maxModelLen, 10),
      port: parseInt(opts.port, 10),
      resultsDir: resolve(opts.resultsDir),
      concurrency: parseInt(opts.concurrency, 10),
      dryRun: opts.dryRun === true,
      modelsYaml: resolve(opts.modelsYaml),
      quant: opts.quant,
      hfToken: opts.hfToken ?? process.env.HF_TOKEN ?? process.env.HUGGING_FACE_HUB_TOKEN,
      dockerImage: opts.dockerImage,
      gpuIds: opts.gpuIds,
      stream: opts.stream === true,
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

// ── Parse ────────────────────────────────────────────────────────────────────

program.parse();
