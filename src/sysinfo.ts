// ── System info collection ───────────────────────────────────────────────────
//
// Collects GPU, CPU, RAM, OS, kernel, Docker, and disk info.
// Used by the `sysinfo` command and embedded in benchmark results.

import { join } from "node:path";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import type { SystemInfo, GpuDetails } from "./types";
import { log, success, warn, error, header, bold } from "./log";

// ── Helpers ──────────────────────────────────────────────────────────────────

async function readFileSafe(path: string): Promise<string> {
  try {
    return await Bun.file(path).text();
  } catch {
    return "";
  }
}

async function spawn(
  cmd: string[],
): Promise<{ stdout: string; exitCode: number }> {
  try {
    const proc = Bun.spawn(cmd, {
      stdout: "pipe",
      stderr: "pipe",
    });
    const stdout = await new Response(proc.stdout).text();
    const exitCode = await proc.exited;
    return { stdout: stdout.trim(), exitCode };
  } catch {
    return { stdout: "", exitCode: -1 };
  }
}

// ── Collect ──────────────────────────────────────────────────────────────────

export async function collectSystemInfo(dockerImage?: string): Promise<SystemInfo> {
  const image = dockerImage || "vllm/vllm-openai:latest";

  // ── GPUs ──
  const gpus: GpuDetails[] = [];
  const { stdout: gpuQuery } = await spawn([
    "nvidia-smi",
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader",
  ]);

  // CUDA version from nvidia-smi header
  let cudaVersion = "unknown";
  const { stdout: smiHeader } = await spawn(["nvidia-smi"]);
  const cudaMatch = smiHeader.match(/CUDA Version:\s*([\d.]+)/);
  if (cudaMatch) cudaVersion = cudaMatch[1]!;

  if (gpuQuery) {
    for (const line of gpuQuery.split("\n")) {
      if (!line.trim()) continue;
      const parts = line.split(",").map((s) => s.trim());
      const name = parts[0] ?? "unknown";
      const vramStr = parts[1] ?? "";
      const vramMatch = vramStr.match(/(\d+)\s*MiB/i);
      const vramGb = vramMatch ? Math.round(parseInt(vramMatch[1]!) / 1024) : 0;
      const driverVersion = parts[2] ?? "unknown";
      gpus.push({ name, vram_gb: vramGb, driver_version: driverVersion, cuda_version: cudaVersion });
    }
  }

  // ── CPU ──
  let cpuModel = "unknown";
  let cpuCores = 0;
  const cpuInfo = await readFileSafe("/proc/cpuinfo");
  if (cpuInfo) {
    const modelMatch = cpuInfo.match(/model name\s*:\s*(.+)/);
    if (modelMatch) cpuModel = modelMatch[1]!.trim();
    cpuCores = cpuInfo.split("\n").filter((l) => l.startsWith("processor")).length;
  }
  // macOS fallback
  if (cpuModel === "unknown") {
    const { stdout: brand } = await spawn(["sysctl", "-n", "machdep.cpu.brand_string"]);
    if (brand) cpuModel = brand;
  }
  if (cpuCores === 0) {
    const { stdout: nproc } = await spawn(["nproc"]);
    cpuCores = parseInt(nproc) || 0;
  }
  if (cpuCores === 0) {
    const { stdout: ncpu } = await spawn(["sysctl", "-n", "hw.ncpu"]);
    cpuCores = parseInt(ncpu) || 0;
  }

  // ── RAM ──
  let ramGb = 0;
  const meminfo = await readFileSafe("/proc/meminfo");
  if (meminfo) {
    const memTotal = meminfo.match(/MemTotal:\s*(\d+)\s*kB/);
    if (memTotal) ramGb = Math.round(parseInt(memTotal[1]!) / 1024 / 1024);
  }
  // macOS fallback
  if (ramGb === 0) {
    const { stdout: memsize } = await spawn(["sysctl", "-n", "hw.memsize"]);
    if (memsize) ramGb = Math.round(parseInt(memsize) / 1024 / 1024 / 1024);
  }

  // ── OS ──
  let osName = "unknown";
  const osRelease = await readFileSafe("/etc/os-release");
  if (osRelease) {
    const nameMatch = osRelease.match(/^PRETTY_NAME="(.+)"/m);
    if (nameMatch) osName = nameMatch[1]!;
    else {
      const nameM = osRelease.match(/^NAME="(.+)"/m);
      const verM = osRelease.match(/^VERSION="(.+)"/m);
      osName = [nameM?.[1], verM?.[1]].filter(Boolean).join(" ") || "unknown";
    }
  }
  // macOS fallback
  if (osName === "unknown") {
    const { stdout: swVer } = await spawn(["sw_vers"]);
    if (swVer) {
      const prod = swVer.match(/ProductName:\s*(.+)/);
      const ver = swVer.match(/ProductVersion:\s*(.+)/);
      osName = [prod?.[1], ver?.[1]].filter(Boolean).join(" ") || "unknown";
    }
  }

  // ── Kernel ──
  const { stdout: kernel } = await spawn(["uname", "-r"]);

  // ── Docker version ──
  let dockerVersion = "unknown";
  const { stdout: dockVer } = await spawn(["docker", "version", "--format", "{{.Server.Version}}"]);
  if (dockVer) dockerVersion = dockVer;

  // ── Disk ──
  let diskAvailGb = 0;
  const hfCacheDir = join(process.env.HOME ?? "/root", ".cache/huggingface");
  // df on the target dir, fall back to HOME if it doesn't exist yet
  const dfTarget = existsSync(hfCacheDir) ? hfCacheDir : (process.env.HOME ?? "/");
  const { stdout: dfOut } = await spawn(["df", "-g", dfTarget]);
  if (dfOut) {
    const dfLines = dfOut.split("\n");
    if (dfLines.length >= 2) {
      // Columns: Filesystem, 1G-blocks, Used, Available, Capacity, ...
      const parts = dfLines[1]!.trim().split(/\s+/);
      const avail = parts[3]; // Available column (in GB)
      if (avail) diskAvailGb = parseInt(avail) || 0;
    }
  }

  return {
    gpus,
    cpu: { model: cpuModel, cores: cpuCores },
    ram_gb: ramGb,
    os: osName,
    kernel: kernel || "unknown",
    docker_version: dockerVersion,
    vllm_image: image,
    disk_available_gb: diskAvailGb,
    collected_at: new Date().toISOString(),
  };
}

// ── Write to file ────────────────────────────────────────────────────────────

export function writeSystemInfo(resultsDir: string, info: SystemInfo): void {
  mkdirSync(resultsDir, { recursive: true });
  const file = join(resultsDir, "system_info.json");
  writeFileSync(file, JSON.stringify(info, null, 2) + "\n");
  success(`System info written: ${file}`);
}

// ── Pretty print ─────────────────────────────────────────────────────────────

export function printSystemInfo(info: SystemInfo): void {
  header("System Configuration");

  if (info.gpus.length === 0) {
    console.log(`  ${bold("GPU:")}      No NVIDIA GPUs detected`);
  }
  for (const g of info.gpus) {
    console.log(`  ${bold("GPU:")}      ${g.name} (${g.vram_gb}GB VRAM)`);
    console.log(`             Driver ${g.driver_version} | CUDA ${g.cuda_version}`);
  }

  console.log(`  ${bold("CPU:")}      ${info.cpu.model} (${info.cpu.cores} cores)`);
  console.log(`  ${bold("RAM:")}      ${info.ram_gb}GB`);
  console.log(`  ${bold("OS:")}       ${info.os}`);
  console.log(`  ${bold("Kernel:")}   ${info.kernel}`);
  console.log(`  ${bold("Docker:")}   ${info.docker_version}`);
  console.log(`  ${bold("vLLM:")}     ${info.vllm_image}`);
  console.log(`  ${bold("Disk:")}     ${info.disk_available_gb}GB available`);
  console.log();
}

// ── CLI entry ────────────────────────────────────────────────────────────────

export async function runSysInfo(opts: { dockerImage?: string; json?: boolean }): Promise<void> {
  const info = await collectSystemInfo(opts.dockerImage);

  if (opts.json) {
    console.log(JSON.stringify(info, null, 2));
  } else {
    printSystemInfo(info);
  }
}
