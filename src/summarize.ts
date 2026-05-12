// ── Results summarizer ───────────────────────────────────────────────────────
//
// Reads all benchmark result JSONs and prints a cost comparison table.

import { resolve, join } from "node:path";
import { existsSync } from "node:fs";
import { readdirSync, readFileSync } from "node:fs";
import type { BenchmarkResult, SystemInfo } from "./types";

function loadResults(resultsDir: string): BenchmarkResult[] {
  const absDir = resolve(resultsDir);
  if (!existsSync(absDir)) {
    console.error(`Results directory not found: ${absDir}`);
    return [];
  }

  const results: BenchmarkResult[] = [];
  const files = readdirSync(absDir)
    .filter((f) => f.endsWith(".json") && f !== "system_info.json")
    .sort();

  for (const file of files) {
    try {
      const data: BenchmarkResult = JSON.parse(
        readFileSync(join(absDir, file), "utf-8"),
      );
      results.push(data);
    } catch (e) {
      console.error(`  Warning: could not read ${file}: ${e}`);
    }
  }

  return results;
}

// ── Table format ─────────────────────────────────────────────────────────────

function formatTable(results: BenchmarkResult[], resultsDir: string): string {
  if (results.length === 0) return "No results found.";

  const absDir = resolve(resultsDir);

  interface Row {
    model: string;
    gpu: string;
    quant: string;
    gpus: string;
    rate: string;
    tps: string;
    cost_1m: string;
    ts: string;
  }

  const rows: Row[] = results.map((r) => ({
    model: r.model_name || r.model_id || "unknown",
    gpu: r.gpu || "unknown",
    quant: r.quant || "-",
    gpus: String(r.gpu_count ?? 1),
    rate: `$${(r.gpu_hourly_rate_usd ?? 0).toFixed(2)}/hr`,
    tps: String(r.results?.output_tokens_per_sec ?? "N/A"),
    cost_1m: formatCost(r.results?.cost_per_1m_output_tokens_usd),
    ts: (r.timestamp || "").slice(0, 10),
  }));

  const cols: (keyof Row)[] = ["model", "gpu", "quant", "gpus", "rate", "tps", "cost_1m", "ts"];
  const headers = ["Model", "GPU", "Quant", "GPUs", "GPU rate", "TPS", "$/1M tokens", "Date"];

  const widths = headers.map((h, i) =>
    Math.max(h.length, ...rows.map((r) => r[cols[i]!].length)),
  );

  const sep = widths.map((w) => "─".repeat(w)).join("  ");
  const headerRow = headers.map((h, i) => h.padEnd(widths[i]!)).join("  ");

  const lines: string[] = [
    "",
    "  GPU Benchmark Cost Summary",
  ];

  // Show system info header if available
  const sysInfoPath = join(absDir, "system_info.json");
  if (existsSync(sysInfoPath)) {
    try {
      const si: SystemInfo = JSON.parse(readFileSync(sysInfoPath, "utf-8"));
      const gpuStr = si.gpus.map((g) => `${g.name} (${g.vram_gb}GB)`).join(", ");
      lines.push(`  System: ${si.cpu.model} (${si.cpu.cores}c) | ${gpuStr} | ${si.ram_gb}GB RAM | ${si.os}`);
      lines.push(`  Image:  ${si.vllm_image} | Docker ${si.docker_version} | Kernel ${si.kernel}`);
      lines.push("");
    } catch {
      // Non-critical
    }
  }

  lines.push("  " + sep);
  lines.push("  " + headerRow);
  lines.push("  " + sep);

  // Sort by cost ascending
  const sorted = [...rows].sort((a, b) => {
    const va = parseCost(a.cost_1m);
    const vb = parseCost(b.cost_1m);
    return va - vb;
  });

  for (const row of sorted) {
    lines.push(
      "  " + cols.map((c, i) => row[c].padEnd(widths[i]!)).join("  "),
    );
  }

  lines.push("  " + sep);
  lines.push("");
  return lines.join("\n");
}

function formatCost(val: number | null | undefined): string {
  if (val === undefined || val === null) return "N/A";
  if (isNaN(val)) return "N/A";
  return `$${val.toFixed(4)}`;
}

function parseCost(s: string): number {
  const cleaned = s.replace(/[$,]/g, "").replace("N/A", "9999");
  const n = parseFloat(cleaned);
  return isNaN(n) ? 9999 : n;
}

// ── CSV format ───────────────────────────────────────────────────────────────

function formatCsv(results: BenchmarkResult[]): string {
  const lines = [
    "model_name,model_id,gpu,quant,gpu_count,gpu_rate_usd,tps,cost_per_1m_usd,timestamp",
  ];
  for (const r of results) {
    lines.push(
      [
        r.model_name ?? "",
        r.model_id ?? "",
        r.gpu ?? "",
        r.quant ?? "",
        r.gpu_count ?? 1,
        r.gpu_hourly_rate_usd ?? 0,
        r.results?.output_tokens_per_sec ?? "",
        r.results?.cost_per_1m_output_tokens_usd ?? "",
        r.timestamp ?? "",
      ].join(","),
    );
  }
  return lines.join("\n");
}

// ── Entry ────────────────────────────────────────────────────────────────────

export function runSummarize(
  resultsDir: string,
  format: "table" | "csv" | "json",
): void {
  const results = loadResults(resultsDir);
  if (results.length === 0) {
    console.log(`No results in ${resultsDir}`);
    return;
  }

  switch (format) {
    case "table":
      console.log(formatTable(results, resultsDir));
      break;
    case "csv":
      console.log(formatCsv(results));
      break;
    case "json":
      console.log(JSON.stringify(results, null, 2));
      break;
  }
}
