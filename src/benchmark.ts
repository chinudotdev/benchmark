// ── Main benchmark orchestration ─────────────────────────────────────────────
//
// Replaces llm_benchmark.sh — Docker orchestration, model management,
// benchmarking, cost calculation, result persistence.

import { join, resolve, dirname } from "node:path";
import { existsSync, mkdirSync, writeFileSync, readFileSync } from "node:fs";
import type {
  BenchmarkOptions,
  BenchmarkResult,
  BenchmarkMetrics,
  GpuInfo,
  ModelEntry,
  SystemInfo,
} from "./types";
import { log, success, warn, error, header, bold, green } from "./log";
import { runBenchmark, printMetrics } from "./bench-runner";
import { loadModels } from "./models";
import { collectSystemInfo, writeSystemInfo, printSystemInfo } from "./sysinfo";

// ── Constants ───────────────────────────────────────────────────────────────

const VALID_QUANTS = ["none", "int8", "int4", "awq", "fp8"] as const;
const COST_UNAVAILABLE = -1;
const DOCKER_IMAGE_DEFAULT = "vllm/vllm-openai:latest";

// ── System info collection ───────────────────────────────────────────────────



// ── Quantization validation ─────────────────────────────────────────────────

function validateQuant(quant: string): void {
  if (quant && !VALID_QUANTS.includes(quant as typeof VALID_QUANTS[number])) {
    error(`Invalid quant '${quant}'. Must be one of: ${VALID_QUANTS.join(", ")}`);
    process.exit(1);
  }
}

// ── Cost calculation ─────────────────────────────────────────────────────────

function calculateCost(
  tps: number,
  gpuRate: number,
  gpuCount: number,
): number {
  if (tps <= 0) return COST_UNAVAILABLE;
  const totalHourly = gpuRate * gpuCount;
  return (totalHourly / tps / 3600) * 1_000_000;
}

// ── Spawn helper ─────────────────────────────────────────────────────────────

async function spawn(
  cmd: string[],
  opts?: { stdout?: "pipe" | "inherit" | "ignore"; stderr?: "pipe" | "inherit" | "ignore" },
): Promise<{ stdout: string; exitCode: number }> {
  const proc = Bun.spawn(cmd, {
    stdout: opts?.stdout ?? "pipe",
    stderr: opts?.stderr ?? "pipe",
  });
  const stdout = opts?.stdout === "pipe"
    ? await new Response(proc.stdout).text()
    : "";
  const exitCode = await proc.exited;
  return { stdout: stdout.trim(), exitCode };
}

// ── Cleanup state ────────────────────────────────────────────────────────────

// Tracks the currently running container for cleanup.
// Models run sequentially so only one container is active at a time.
let activeContainer: string | null = null;
let isCleaningUp = false;

async function cleanup(): Promise<void> {
  if (isCleaningUp) return;
  isCleaningUp = true;
  if (activeContainer) {
    log("Cleaning up container...");
    await stopServer(activeContainer);
    activeContainer = null;
  }
}

process.on("SIGINT", async () => {
  process.stdout.write("\n");
  await cleanup();
  process.exit(130);
});

process.on("SIGTERM", async () => {
  await cleanup();
  process.exit(143);
});

// ── Dependency definitions ──────────────────────────────────────────────────

const DEPS: Record<string, { aptPackage: string | null; note?: string }> = {
  "docker":     { aptPackage: "docker.io", note: "Also needs nvidia-container-toolkit for GPU support" },
  "nvidia-smi":  { aptPackage: null, note: "Requires NVIDIA drivers (not available via apt)" },
};

// ── Dependency checks ────────────────────────────────────────────────────────

function checkCommand(name: string): boolean {
  try {
    // 'command -v' is a shell builtin — must run via sh -c
    const result = Bun.spawnSync(["sh", "-c", `command -v ${name}`], {
      stdout: "ignore",
      stderr: "ignore",
    });
    return result.exitCode === 0;
  } catch {
    return false;
  }
}

async function checkDependencies(): Promise<GpuInfo> {
  header("Checking dependencies");

  const missing: string[] = [];
  if (!checkCommand("docker")) missing.push("docker");
  if (!checkCommand("nvidia-smi")) missing.push("nvidia-smi");
  if (!checkCommand("python3")) {
    // python3 + huggingface-cli are only needed for model downloading,
    // not for the benchmark itself. Warn but don't block.
    warn("python3 not found — model downloading won't work (benchmarking already-cached models is fine)");
  }
  // jq and curl are not required — we use native fetch and JSON.parse

  if (missing.length > 0) {
    error(`Missing dependencies: ${missing.join(", ")}`);

    const aptPackages = missing
      .map((m) => DEPS[m]?.aptPackage)
      .filter(Boolean) as string[];

    if (missing.includes("nvidia-smi")) {
      console.log();
      console.log("  nvidia-smi requires NVIDIA drivers + nvidia-container-toolkit.");
      console.log("  Install drivers:  https://docs.nvidia.com/cuda/cuda-installation-guide-linux");
      console.log("  Install container: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide");
      console.log();
    }

    if (aptPackages.length > 0) {
      const installCmd = `sudo apt-get install -y ${aptPackages.join(" ")}`;
      console.log(`  Install command:  ${installCmd}`);
      console.log();

      // Prompt to auto-install
      process.stdout.write(`  Install now? [y/N] `);
      const answer = await new Promise<string>((resolve) => {
        process.stdin.once("data", (data) => resolve(data.toString().trim().toLowerCase()));
      });

      if (answer === "y" || answer === "yes") {
        log("Updating package lists...");
        const updateProc = Bun.spawn(["sudo", "apt-get", "update", "-y"], {
          stdout: "pipe",
          stderr: "inherit",
        });
        const updateExit = await updateProc.exited;
        if (updateExit !== 0) {
          warn("apt-get update failed (will try install anyway)");
        }

        log(`Installing: ${aptPackages.join(", ")}`);
        const proc = Bun.spawn(["sudo", "apt-get", "install", "-y", ...aptPackages], {
          stdout: "inherit",
          stderr: "inherit",
        });
        const exitCode = await proc.exited;
        if (exitCode !== 0) {
          error("Installation failed. Please install manually.");
          process.exit(1);
        }
        success("Dependencies installed.");

        // Re-check — if nvidia-smi was missing and still is, it needs a driver install
        const stillMissing = missing.filter((m) => !checkCommand(m));
        if (stillMissing.length > 0) {
          error(`Still missing: ${stillMissing.join(", ")} — may require a reboot or manual driver installation`);
          process.exit(1);
        }
      } else {
        process.exit(1);
      }
    } else {
      // Only nvidia-smi missing (no apt package for it)
      process.exit(1);
    }
  }

  // GPU detection
  const { stdout: gpuInfo, exitCode } = await spawn([
    "nvidia-smi",
    "--query-gpu=name,memory.total",
    "--format=csv,noheader",
  ]);

  if (exitCode !== 0 || !gpuInfo) {
    error("No GPUs detected by nvidia-smi");
    process.exit(1);
  }

  success("GPUs detected:");
  for (const line of gpuInfo.split("\n")) {
    if (line.trim()) console.log(`    • ${line.trim()}`);
  }

  // Parse first GPU VRAM
  const firstLine = gpuInfo.split("\n")[0]!;
  const vramMatch = firstLine.match(/(\d+)\s*MiB/i);
  const vramGb = vramMatch ? Math.round(parseInt(vramMatch[1]!) / 1024) : 0;

  const gpuName = firstLine.split(",")[0]!.trim();

  log(`Available VRAM: ${vramGb}GB (first GPU)`);
  log(`GPU: ${gpuName}`);

  return { name: gpuName, vramGb, fullInfo: gpuInfo };
}

// ── Docker image ─────────────────────────────────────────────────────────────

function getDockerImage(customImage?: string): string {
  return customImage || DOCKER_IMAGE_DEFAULT;
}

async function pullDockerImage(imageOverride: string | undefined, dryRun: boolean): Promise<void> {
  const image = getDockerImage(imageOverride);
  log(`Pulling Docker image: ${image}`);
  if (dryRun) {
    console.log(`  [DRY RUN] docker pull ${image}`);
    return;
  }
  const proc = Bun.spawn(["docker", "pull", image], {
    stdout: "inherit",
    stderr: "inherit",
  });
  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    error(`docker pull failed (exit ${exitCode})`);
    process.exit(1);
  }
  success(`Image ready: ${image}`);
}

// ── Download model ───────────────────────────────────────────────────────────

async function downloadModel(modelId: string, hfToken: string | undefined, dryRun: boolean): Promise<void> {
  const cacheDir = join(
    process.env.HOME ?? "/root",
    ".cache/huggingface",
  );
  const modelSlug = modelId.replace("/", "--");
  const modelCache = join(cacheDir, `hub/models--${modelSlug}`);

  log(`Checking model cache for: ${modelId}`);

  if (dryRun) {
    console.log(`  [DRY RUN] huggingface-cli download ${modelId}`);
    return;
  }

  if (existsSync(modelCache)) {
    success(`Model already cached: ${modelId}`);
    return;
  }

  log(`Downloading ${modelId} (this may take a while)...`);

  // Ensure huggingface_hub is available
  const pipProc = Bun.spawn(
    ["python3", "-m", "pip", "install", "-q", "huggingface_hub"],
    { stdout: "ignore", stderr: "ignore" },
  );
  await pipProc.exited;

  const env: Record<string, string> = { ...process.env as Record<string, string> };
  if (hfToken) env.HUGGING_FACE_HUB_TOKEN = hfToken;

  const proc = Bun.spawn(
    ["huggingface-cli", "download", modelId, "--resume-download", "--quiet"],
    { stdout: "inherit", stderr: "inherit", env },
  );
  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    warn(`Download exited with code ${exitCode} — model may still download partially`);
  } else {
    success(`Download complete: ${modelId}`);
  }
}

// ── Quantization flags ───────────────────────────────────────────────────────

function getQuantFlags(quant: string): string[] {
  switch (quant) {
    case "int8":
      return ["--quantization", "bitsandbytes"];
    case "int4":
      return ["--quantization", "bitsandbytes", "--load-in-4bit"];
    case "awq":
      return ["--quantization", "awq"];
    case "fp8":
      return ["--quantization", "fp8"];
    case "none":
    case "":
      return [];
    default:
      warn(`Unknown quant '${quant}', skipping flag`);
      return [];
  }
}

// ── Split extra flags ────────────────────────────────────────────────────────

function splitFlags(flags: string): string[] {
  return flags.trim().split(/\s+/).filter(Boolean);
}

// ── Port availability check ─────────────────────────────────────────────────

async function isPortInUse(port: number): Promise<boolean> {
  try {
    const resp = await fetch(`http://localhost:${port}/health`, {
      signal: AbortSignal.timeout(1000),
    });
    // If we get any response, something is listening
    return true;
  } catch {
    // Connection refused means port is free
    return false;
  }
}

// ── Start vLLM server ───────────────────────────────────────────────────────

async function startServer(
  modelId: string,
  quant: string,
  tp: number,
  extraFlags: string,
  maxModelLen: number,
  port: number,
  opts: BenchmarkOptions,
): Promise<string> {
  // Pre-check port availability
  if (!opts.dryRun) {
    const inUse = await isPortInUse(port);
    if (inUse) {
      error(`Port ${port} is already in use. Stop the process using it or specify a different --port.`);
      process.exit(1);
    }
  }

  const image = getDockerImage(opts.dockerImage);
  const containerName = `vllm_bench_${port}`;
  const hfCache = join(process.env.HOME ?? "/root", ".cache/huggingface");

  const quantFlags = getQuantFlags(quant);
  const extra = splitFlags(extraFlags);

  const gpuFlag = opts.gpuIds ?? "all";

  const args = [
    "docker", "run", "-d",
    "--gpus", gpuFlag,
    "--ipc", "host",
    "--name", containerName,
    "-p", `${port}:8000`,
    "-v", `${hfCache}:/root/.cache/huggingface`,
  ];

  // Pass HF token to container if provided
  if (opts.hfToken) {
    args.push("-e", `HUGGING_FACE_HUB_TOKEN=${opts.hfToken}`);
  }

  args.push(
    image,
    modelId,
    "--trust-remote-code",
    "--tensor-parallel-size", String(tp),
    "--max-model-len", String(maxModelLen),
    "--gpu-memory-utilization", "0.90",
    "--disable-log-requests",
    "--port", "8000",
    ...quantFlags,
    ...extra,
  );

  log("Starting vLLM server...");
  log(`  Model:  ${modelId}`);
  log(`  Quant:  ${quant || "none"}`);
  log(`  TP:     ${tp}`);
  log(`  Port:   ${port}`);
  log(`  Image:  ${image}`);
  log(`  GPUs:   ${gpuFlag}`);

  if (opts.dryRun) {
    console.log(`\n  [DRY RUN] docker run command:`);
    console.log(`  ${args.join(" ")}\n`);
    return containerName;
  }

  // Kill any existing container
  await stopServer(containerName);

  const proc = Bun.spawn(args, { stdout: "pipe", stderr: "pipe" });
  const exitCode = await proc.exited;

  if (exitCode !== 0) {
    const stderr = await new Response(proc.stderr).text();
    error(`Failed to start container: ${stderr.trim()}`);
    throw new Error("Docker start failed");
  }

  activeContainer = containerName;
  success(`Container started: ${containerName}`);
  return containerName;
}

// ── Wait for server ──────────────────────────────────────────────────────────

async function waitForServer(
  port: number,
  dryRun: boolean,
): Promise<boolean> {
  const maxWait = 600; // 10 minutes
  const interval = 10;
  let elapsed = 0;

  log(`Waiting for server to be ready (max ${maxWait}s)...`);

  if (dryRun) {
    console.log(`  [DRY RUN] waiting for http://localhost:${port}/health`);
    return true;
  }

  while (elapsed < maxWait) {
    try {
      const resp = await fetch(`http://localhost:${port}/health`, {
        signal: AbortSignal.timeout(5000),
      });
      if (resp.ok) {
        success(`Server is ready! (${elapsed}s)`);
        return true;
      }
    } catch {
      // Not ready yet
    }

    // Check if container crashed
    const { stdout: status } = await spawn([
      "docker", "inspect", `vllm_bench_${port}`,
      "--format={{.State.Status}}",
    ]);

    if (status === "exited") {
      error("Container exited unexpectedly. Check logs:");
      console.log(`  docker logs vllm_bench_${port}`);
      await stopServer(`vllm_bench_${port}`);
      return false;
    }

    log(`  Not ready yet (${elapsed}s) — retrying in ${interval}s...`);
    await Bun.sleep(interval * 1000);
    elapsed += interval;
  }

  error(`Server did not become ready within ${maxWait}s`);
  console.log(`  docker logs vllm_bench_${port}`);
  await stopServer(`vllm_bench_${port}`);
  return false;
}

// ── Stop server ──────────────────────────────────────────────────────────────

async function stopServer(containerName: string): Promise<void> {
  try {
    const proc = Bun.spawn(["docker", "rm", "-f", containerName], {
      stdout: "ignore",
      stderr: "ignore",
    });
    await proc.exited;
  } catch {
    // ignore — container may not exist
  }
  if (activeContainer === containerName) {
    activeContainer = null;
  }
}

// ── Write result JSON ────────────────────────────────────────────────────────

function writeResult(
  model: ModelEntry,
  opts: BenchmarkOptions,
  gpuInfo: GpuInfo,
  metrics: BenchmarkMetrics,
  cost: number,
  resultFile: string,
  systemInfo: SystemInfo,
): void {
  mkdirSync(dirname(resultFile), { recursive: true });

  const result: BenchmarkResult = {
    model_id: model.id,
    model_name: model.name,
    quant: model.quant,
    gpu: gpuInfo.name,
    gpu_count: opts.gpuCount,
    gpu_hourly_rate_usd: opts.gpuRate,
    benchmark: {
      num_prompts: opts.numPrompts,
      input_len: opts.inputLen,
      output_len: opts.outputLen,
      max_model_len: opts.maxModelLen,
    },
    results: {
      output_tokens_per_sec: metrics.output_tps,
      cost_per_1m_output_tokens_usd: isNaN(cost) ? COST_UNAVAILABLE : Math.round(cost * 10000) / 10000,
    },
    system: systemInfo,
    timestamp: new Date().toISOString(),
  };

  writeFileSync(resultFile, JSON.stringify(result, null, 2) + "\n");
  success(`Result written: ${resultFile}`);
}

// ── Print summary ────────────────────────────────────────────────────────────

function printSummary(resultsDir: string): void {
  header("Benchmark Summary");

  if (!existsSync(resultsDir)) {
    warn(`No results found in ${resultsDir}`);
    return;
  }

  const files = Array.from(new Bun.Glob("*.json").scanSync({ cwd: resultsDir }))
    .filter((f) => f !== "system_info.json");
  if (files.length === 0) {
    warn(`No results found in ${resultsDir}`);
    return;
  }

  // Show system info if available
  const sysInfoPath = join(resultsDir, "system_info.json");
  if (existsSync(sysInfoPath)) {
    try {
      const si: SystemInfo = JSON.parse(readFileSync(sysInfoPath, "utf-8"));
      const gpuStr = si.gpus.map((g) => `${g.name} (${g.vram_gb}GB)`).join(", ");
      console.log(`  ${bold("System:")} ${si.cpu.model} (${si.cpu.cores}c) | ${gpuStr} | ${si.ram_gb}GB RAM | ${si.os}`);
      console.log(`  ${bold("Image:")} ${si.vllm_image} | Docker ${si.docker_version} | Kernel ${si.kernel}`);
      console.log();
    } catch {
      // Non-critical — skip if unreadable
    }
  }

  const nameW = 40, quantW = 12, tpsW = 18, costW = 20;
  const sep = "─".repeat(nameW) + "  " + "─".repeat(quantW) + "  " +
    "─".repeat(tpsW) + "  " + "─".repeat(costW);

  console.log(`  ${bold("Model").padEnd(nameW)}  ${bold("Quant").padEnd(quantW)}  ${bold("TPS (tok/s)").padEnd(tpsW)}  ${bold("Cost/1M tokens")}`);
  console.log(`  ${sep}`);

  for (const file of files) {
    const fullPath = join(resultsDir, file);
    try {
      const data: BenchmarkResult = JSON.parse(
        readFileSync(fullPath, "utf-8"),
      );
      const name = data.model_name.padEnd(nameW);
      const quant = data.quant.padEnd(quantW);
      const tps = String(data.results.output_tokens_per_sec).padEnd(tpsW);
      const costVal = data.results.cost_per_1m_output_tokens_usd;
      const cost = (costVal === null || costVal === undefined || costVal === COST_UNAVAILABLE || isNaN(costVal))
        ? "N/A" : `$${costVal}`;
      console.log(`  ${name}  ${quant}  ${tps}  ${cost}`);
    } catch {
      warn(`Could not parse: ${file}`);
    }
  }

  console.log(`  ${sep}`);
  console.log();
  log(`Full JSON results in: ${resultsDir}/`);
}

// ── Benchmark a single model ─────────────────────────────────────────────────

async function benchmarkModel(
  model: ModelEntry,
  opts: BenchmarkOptions,
  gpuInfo: GpuInfo,
  systemInfo: SystemInfo,
): Promise<void> {
  header(`Benchmarking: ${model.name}`);

  // VRAM check
  if (gpuInfo.vramGb > 0 && model.min_vram_gb > 0 && gpuInfo.vramGb < model.min_vram_gb && !opts.dryRun) {
    warn(`Skipping ${model.name} — requires ${model.min_vram_gb}GB, GPU has ${gpuInfo.vramGb}GB`);
    return;
  }

  const safeName = model.name.replace(/[/\\s]/g, "_");
  const resultFile = resolve(join(opts.resultsDir, `${safeName}.json`));

  // Check if already benchmarked
  if (existsSync(resultFile) && !opts.dryRun) {
    warn(`Result already exists: ${resultFile} (delete to re-run)`);
    return;
  }

  // Download model weights
  await downloadModel(model.id, opts.hfToken, opts.dryRun);

  // Start vLLM server
  const containerName = await startServer(
    model.id,
    model.quant,
    model.tp,
    model.extra_flags,
    opts.maxModelLen,
    opts.port,
    opts,
  );

  // Wait for server
  const ready = await waitForServer(opts.port, opts.dryRun);
  if (!ready) return;

  // Run benchmark
  log("Running benchmark...");
  log(`  Prompts:    ${opts.numPrompts}`);
  log(`  Input len:  ${opts.inputLen} tokens`);
  log(`  Output len: ${opts.outputLen} tokens`);

  if (opts.dryRun) {
    console.log(`  [DRY RUN] benchmark would run with concurrency=${opts.concurrency}`);
    console.log(`  [DRY RUN] results would be written to: ${resultFile}`);
    await stopServer(containerName);
    return;
  }

  const metrics = await runBenchmark({
    model: model.id,
    host: "localhost",
    port: opts.port,
    numPrompts: opts.numPrompts,
    inputLen: opts.inputLen,
    outputLen: opts.outputLen,
    concurrency: opts.concurrency,
    stream: opts.stream ?? false,
    retries: 2,
  });

  printMetrics(metrics);

  // Calculate cost
  const cost = calculateCost(metrics.output_tps, opts.gpuRate, opts.gpuCount);
  const costStr = (isNaN(cost) || cost === COST_UNAVAILABLE) ? "N/A" : `$${cost.toFixed(4)}`;

  // Write result
  writeResult(model, opts, gpuInfo, metrics, cost, resultFile, systemInfo);

  // Print inline results
  console.log();
  console.log(`  ${bold(`Results for ${model.name}:`)}`);
  console.log(`  TPS:              ${green(`${metrics.output_tps} tokens/sec`)}`);
  console.log(`  Cost per 1M tok:  ${green(costStr)}`);
  console.log();

  // Stop server
  await stopServer(containerName);
}

// ── Main entry ───────────────────────────────────────────────────────────────

export async function runBenchmarkCommand(opts: BenchmarkOptions): Promise<void> {
  header("LLM GPU Benchmark");
  log(`GPU rate:   $${opts.gpuRate}/hr × ${opts.gpuCount} GPU(s)`);
  log(`Results:    ${opts.resultsDir}`);
  if (opts.dryRun) warn("DRY RUN mode — no commands will execute");

  const gpuInfo = await checkDependencies();

  // Collect full system info (once for the whole run)
  const systemInfo = await collectSystemInfo(opts.dockerImage);

  // Validate quant if provided
  if (opts.quant) validateQuant(opts.quant);

  await pullDockerImage(opts.dockerImage, opts.dryRun);

  mkdirSync(opts.resultsDir, { recursive: true });

  // Write standalone system_info.json
  if (!opts.dryRun) writeSystemInfo(opts.resultsDir, systemInfo);

  if (opts.all) {
    // Run all models from registry
    const models = loadModels(resolve(opts.modelsYaml));
    log(`Found ${models.length} models in registry`);

    for (const model of models) {
      await benchmarkModel(model, opts, gpuInfo, systemInfo);
    }
  } else if (opts.modelId) {
    // Run single model (quant defaults to "none" unless overridden via --quant)
    await benchmarkModel(
      {
        id: opts.modelId,
        name: opts.modelId.split("/").pop() ?? opts.modelId,
        quant: opts.quant ?? "none",
        min_vram_gb: 0,
        tp: 1,
        extra_flags: "",
      },
      opts,
      gpuInfo,
      systemInfo,
    );
  } else {
    error("Provide --model-id <id> or --all");
    process.exit(1);
  }

  printSummary(opts.resultsDir);
}
