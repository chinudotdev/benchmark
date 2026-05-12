// ── HTTP benchmark client ────────────────────────────────────────────────────
//
// Sends concurrent requests to an OpenAI-compatible API and measures TPS.
// Uses Bun's native fetch — zero external dependencies.

import { Semaphore } from "./semaphore";
import type { BenchRunnerOptions, BenchmarkMetrics, RequestResult } from "./types";

// ── Prompt generation ────────────────────────────────────────────────────────

const WORDS = [
  "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
  "data", "model", "inference", "token", "compute", "benchmark",
  "gpu", "neural", "network", "transformer", "attention",
  "embedding", "weight",
];

// ── Seeded PRNG (deterministic benchmarks) ─────────────────────────────────

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 4294967296;
  };
}

function makePrompt(numTokens: number, rng: () => number): string {
  // Rough: 1 token ≈ 4 chars for English text
  const targetChars = numTokens * 4;
  const words: string[] = [];
  let totalLen = 0;
  while (totalLen < targetChars) {
    const w = WORDS[Math.floor(rng() * WORDS.length)]!;
    words.push(w);
    totalLen += w.length + 1;
  }
  return words.join(" ");
}

// ── Single request (with retries) ───────────────────────────────────────────

const MAX_RETRIES = 2;
const RETRY_DELAY_MS = 2000;

async function sendRequest(
  url: string,
  model: string,
  prompt: string,
  maxTokens: number,
  stream: boolean,
  retries: number = MAX_RETRIES,
  includeStreamOptions: boolean = true,
): Promise<RequestResult> {
  const payload = {
    model,
    messages: [{ role: "user", content: prompt }],
    max_tokens: maxTokens,
    temperature: 1.0,
    stream,
    ...(stream && includeStreamOptions ? { stream_options: { include_usage: true } } : {}),
  };

  const start = performance.now();
  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(300_000), // 5 min
    });

    const latencyS = (performance.now() - start) / 1000;

    if (!resp.ok) {
      const text = await resp.text();
      // Retry on server errors (5xx) and connection issues
      if (resp.status >= 500 && retries > 0) {
        await Bun.sleep(RETRY_DELAY_MS);
        return sendRequest(url, model, prompt, maxTokens, stream, retries - 1);
      }
      return {
        success: false,
        latencyS,
        outputTokens: 0,
        error: `HTTP ${resp.status}: ${text.slice(0, 200)}`,
      };
    }

    // Streaming response: count tokens from SSE chunks
    if (stream) {
      let outputTokens = 0;
      if (resp.body) {
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith("data: ")) continue;
            const data = trimmed.slice(6);
            if (data === "[DONE]") continue;
            try {
              const chunk = JSON.parse(data);
              if (chunk.usage?.completion_tokens) {
                outputTokens = chunk.usage.completion_tokens;
              }
            } catch {
              // Skip malformed chunks
            }
          }
        }
      }
      // If no usage from stream (should be rare with stream_options)
      if (outputTokens === 0) {
        outputTokens = maxTokens; // fallback
      }
      return { success: true, outputTokens, latencyS };
    }

    // Non-streaming response
    const data = (await resp.json()) as Record<string, unknown>;
    const usage = data?.usage as Record<string, number> | undefined;
    const outputTokens = usage?.completion_tokens ?? 0;

    return { success: true, outputTokens, latencyS };
  } catch (err) {
    // Retry on network errors
    if (retries > 0) {
      await Bun.sleep(RETRY_DELAY_MS);
      return sendRequest(url, model, prompt, maxTokens, stream, retries - 1);
    }
    return {
      success: false,
      outputTokens: 0,
      latencyS: (performance.now() - start) / 1000,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

// ── Probe stream_options support ────────────────────────────────────────────
//
// Sends a single tiny streaming request with stream_options. If the server
// rejects it (400), we know it's an older version and skip the parameter for
// the rest of the benchmark. One cheap probe avoids N failed requests.

async function probeStreamOptions(
  url: string,
  model: string,
): Promise<boolean> {
  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "hi" }],
        max_tokens: 1,
        stream: true,
        stream_options: { include_usage: true },
      }),
      signal: AbortSignal.timeout(30_000),
    });
    // 400 means the server doesn't recognize stream_options
    if (resp.status === 400) {
      return false;
    }
    // Consume the body to free the connection
    if (resp.body) {
      const reader = resp.body.getReader();
      while (true) {
        const { done } = await reader.read();
        if (done) break;
      }
    }
    // Any non-400 response (200, 429, 500, etc.) means the server accepted
    // the field — it's safe to include in subsequent requests.
    return true;
  } catch {
    // Network error during probe — be conservative, skip stream_options
    return false;
  }
}

// ── Async benchmark ──────────────────────────────────────────────────────────

export async function runBenchmark(opts: BenchRunnerOptions): Promise<BenchmarkMetrics> {
  const url = `http://${opts.host}:${opts.port}/v1/chat/completions`;
  const rng = seededRandom(42);
  const prompts = Array.from({ length: opts.numPrompts }, () =>
    makePrompt(opts.inputLen, rng),
  );

  // Probe stream_options support for streaming benchmarks
  let useStreamOptions = true;
  if (opts.stream) {
    useStreamOptions = await probeStreamOptions(url, opts.model);
    if (!useStreamOptions) {
      console.log("  ⚠ Server doesn't support stream_options — token counts may be estimated");
    }
  }

  console.log(`\n  Sending ${opts.numPrompts} requests (concurrency=${opts.concurrency}, stream=${opts.stream})...`);

  const semaphore = new Semaphore(opts.concurrency);
  const results: RequestResult[] = [];
  let completed = 0;

  const benchmarkStart = performance.now();

  const tasks = prompts.map(async (prompt) => {
    await semaphore.acquire();
    try {
      const result = await sendRequest(url, opts.model, prompt, opts.outputLen, opts.stream, opts.retries, useStreamOptions);
      results.push(result);
      completed++;
      if (completed % 20 === 0 || completed === opts.numPrompts) {
        const elapsed = (performance.now() - benchmarkStart) / 1000;
        process.stdout.write(
          `  Progress: ${completed}/${opts.numPrompts} (${elapsed.toFixed(1)}s)\r`,
        );
      }
      return result;
    } finally {
      semaphore.release();
    }
  });

  await Promise.all(tasks);
  const totalMs = performance.now() - benchmarkStart;
  const totalTimeS = totalMs / 1000;

  process.stdout.write("\n");

  const successful = results.filter((r) => r.success);
  const failed = results.filter((r) => !r.success);

  if (successful.length === 0) {
    const firstError = failed[0]?.error ?? "unknown";
    console.error(`  ERROR: All requests failed! First error: ${firstError}`);
    return {
      total_requests: opts.numPrompts,
      successful_requests: 0,
      failed_requests: failed.length,
      total_time_s: Math.round(totalTimeS * 100) / 100,
      total_output_tokens: 0,
      output_tps: 0,
      latency_p50_s: 0,
      latency_p95_s: 0,
      latency_p99_s: 0,
    };
  }

  const totalOutputTokens = successful.reduce((sum, r) => sum + r.outputTokens, 0);
  const outputTps = totalOutputTokens / totalTimeS;

  const latencies = successful
    .map((r) => r.latencyS)
    .sort((a, b) => a - b);
  const p50 = latencies[Math.floor(latencies.length * 0.50)]!;
  const p95 = latencies[Math.floor(latencies.length * 0.95)]!;
  const p99 = latencies[Math.floor(latencies.length * 0.99)]!;

  return {
    total_requests: opts.numPrompts,
    successful_requests: successful.length,
    failed_requests: failed.length,
    total_time_s: Math.round(totalTimeS * 100) / 100,
    total_output_tokens: totalOutputTokens,
    output_tps: Math.round(outputTps * 100) / 100,
    latency_p50_s: Math.round(p50 * 1000) / 1000,
    latency_p95_s: Math.round(p95 * 1000) / 1000,
    latency_p99_s: Math.round(p99 * 1000) / 1000,
  };
}

// ── Print results ────────────────────────────────────────────────────────────

export function printMetrics(metrics: BenchmarkMetrics): void {
  const line = "─".repeat(50);
  console.log(`\n${line}`);
  console.log("  Results");
  console.log(`${line}`);
  console.log(
    `  Requests:        ${metrics.successful_requests}/${metrics.total_requests} succeeded`,
  );
  console.log(`  Duration:        ${metrics.total_time_s}s`);
  console.log(`  Output tokens:   ${metrics.total_output_tokens}`);
  console.log(`  Output TPS:      ${metrics.output_tps} tok/s`);
  console.log(`  Latency p50:     ${metrics.latency_p50_s}s`);
  console.log(`  Latency p95:     ${metrics.latency_p95_s}s`);
  console.log(`  Latency p99:     ${metrics.latency_p99_s}s`);
  console.log(`${line}\n`);
}
