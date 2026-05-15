# Bug Report: `gpu-benchmark` Project Code Review

Comprehensive code review of the Go + TypeScript codebase. All existing Go tests pass, but several runtime bugs, race conditions, and logic errors were found.

---

## 🔴 Critical Bugs

### BUG 1 — Race condition in benchmark results collection (TypeScript)

**File:** `src/bench-runner.ts`, `runBenchmark()`

```typescript
results.push(result);
completed++;
```

Multiple concurrent async tasks push to the `results[]` array and increment `completed` without any synchronization. In JavaScript this is *technically safe* within a single microtask, but since `fetch` results resolve across multiple microtasks, this is a latent race — `results.length` and `completed` can desync. More critically, **results are not guaranteed to be in order**, meaning percentile calculations may be subtly wrong. The Go version correctly uses a mutex.

**Fix:** Use an indexed assignment instead:
```typescript
results[idx] = result;
```
where `idx` is captured from the loop index (already available via the outer `prompts.map`).

---

### BUG 2 — `break` in `select` doesn't break the `for` loop (Go)

**File:** `internal/benchmark/runner.go`, line ~30

```go
for i, prompt := range prompts {
    select {
    case <-ctx.Done():
        break  // <-- breaks the SELECT, not the FOR loop!
    default:
    }
    // ... continues processing
```

The `break` inside `select` breaks out of the `select` block, **not** the `for` loop. When context is cancelled, the loop continues submitting goroutines. This is a classic Go gotcha.

**Fix:** Use a labeled break:
```go
loop:
for i, prompt := range prompts {
    select {
    case <-ctx.Done():
        break loop
    default:
    }
```

---

### BUG 3 — `proc` used outside `try` block (TypeScript `startServer`)

**File:** `src/benchmark.ts`, `startServer()` function

```typescript
let exitCode: number;
try {
    const proc = Bun.spawn(args, { stdout: "pipe", stderr: "pipe" });
    exitCode = await proc.exited;
} finally {
    if (envFile) {
        try { unlinkSync(envFile); } catch { /* non-critical */ }
    }
}
if (exitCode !== 0) {
    const stderr = await new Response(proc.stderr).text();  // <-- proc is NOT in scope!
```

`proc` is declared inside the `try` block with `const`, but is referenced outside it. This will cause a `ReferenceError: proc is not defined` at runtime.

**Fix:** Declare `proc` before the `try` block:
```typescript
let proc: ReturnType<typeof Bun.spawn>;
try {
    proc = Bun.spawn(args, { ... });
    exitCode = await proc.exited;
} finally { ... }
```

---

## 🟠 High-Severity Bugs

### BUG 4 — Health check doesn't measure cold start correctly (Go)

**File:** `internal/docker/docker.go`, `WaitHealthy()`

```go
elapsed := time.Duration(0)
// ...
time.Sleep(interval)
elapsed += interval
```

`elapsed` starts at 0, so if the server is ready on the very first check (within the first 5s), `WaitHealthy` returns `elapsed = 0` — reporting a cold start of 0 seconds. It should measure the actual wall clock time.

**Fix:**
```go
start := time.Now()
// ... loop ...
return time.Since(start), nil
```

---

### BUG 5 — `useStreamOpts` result is computed but never used (Go)

**File:** `internal/benchmark/runner.go`, lines ~42-44

```go
if useStreamOpts || !cfg.Stream {
    // nothing to adjust
}
```

This is a no-op dead code branch. The `useStreamOpts` result from `probeStreamOptions` is computed but never actually used to adjust the request payload. If `useStreamOpts` is `false`, the code should skip sending `stream_options` in subsequent requests, but it always sends it inside `sendRequest` when `cfg.Stream` is true:

```go
if cfg.Stream {
    payload["stream_options"] = map[string]any{"include_usage": true}
}
```

**Fix:** Pass `useStreamOpts` to `sendRequest` and conditionally include `stream_options`.

---

### BUG 6 — Double signal handler in orchestrator creates race (Go)

**File:** `internal/orchestrator/orchestrator.go`

Two goroutines listen on the same `sigCh` channel:

```go
// First handler
go func() {
    sig := <-sigCh
    fmt.Printf("\nReceived %v, cleaning up...\n", sig)
    cancel()
}()

// Second handler (inside Run)
go func() {
    <-sigCh
    dmgr.Cleanup()
    os.Exit(130)
}()
```

Only one goroutine will receive the signal (since the channel has capacity 1). If the second goroutine gets it first, `cancel()` is never called and in-flight operations won't be gracefully stopped. If the first gets it, `dmgr.Cleanup()` is never called (relies on `defer` from a long-running function).

**Fix:** Use a single signal handler that does both `cancel()` and `dmgr.Cleanup()`.

---

### BUG 7 — Backoff formula uses `cfg.Retries` instead of remaining `retries` parameter (Go)

**File:** `internal/benchmark/runner.go`, `sendRequest()`

```go
backoff := time.Duration(math.Pow(2, float64(cfg.Retries-retries+1))) * time.Second
```

This uses `cfg.Retries` (the original value from config) minus `retries` (the current remaining count). So if `cfg.Retries=2` and `retries=1` (first retry), backoff = `2^(2-1+1)` = 4s. If `cfg.Retries=2` and `retries=0` (shouldn't reach here), it'd be `2^3` = 8s. The formula works, but it's fragile — if someone calls `sendRequest` from warmup with `retries=1` but a different `cfg.Retries`, the backoff is wrong. The simpler formula would be:

```go
attempt := cfg.Retries - retries
backoff := time.Duration(math.Pow(2, float64(attempt))) * time.Second
```

---

### BUG 8 — Nil pointer dereference in `printTable()` (Go)

**File:** `internal/report/summarize.go`, `printTable()`

```go
tpsStr := fmt.Sprintf("%.2f", r.Metrics.OutputTPS)
```

There's no nil check on `r.Metrics` in `printTable()`, unlike `printCSV()` which correctly checks:

```go
if r.Metrics != nil {
    tps = fmt.Sprintf("%.2f", r.Metrics.OutputTPS)
}
```

This will panic with a nil pointer dereference if `r.Metrics` is nil.

**Fix:** Add nil checks consistently in `printTable()` for all `r.Metrics` accesses.

---

## 🟡 Medium-Severity Bugs

### BUG 9 — `truncate` can break multi-byte characters (Go)

**File:** `internal/benchmark/runner.go`

```go
func truncate(s string, maxLen int) string {
    if len(s) <= maxLen {
        return s
    }
    return s[:maxLen]
}
```

`len(s)` and `s[:maxLen]` operate on bytes, not runes. If an error message contains multi-byte UTF-8 characters (e.g., Chinese text in model names), this will slice mid-character and produce invalid UTF-8.

**Fix:** Use `[]rune` conversion or `utf8.RuneCountInString`.

---

### BUG 10 — `FixCachePermissions` always runs `sudo chown` (Go)

**File:** `internal/download/download.go`

```go
func FixCachePermissions() error {
    // ...
    cmd := exec.Command("sudo", "chown", "-R", user+":"+user, cacheDir)
    return cmd.Run()
}
```

This unconditionally runs `sudo chown` even if the directory is already owned by the current user. On macOS or systems without `sudo` passwordless config, this will hang waiting for a password.

**Fix:** Check actual ownership before running `chown`.

---

### BUG 11 — `--warmup` flag missing from TypeScript CLI

**File:** `src/index.ts`

The Go CLI has `--warmup` (number of warmup requests to discard), but the TypeScript version doesn't expose this option. The `BenchRunnerOptions` type also lacks a `warmupReqs` field.

---

### BUG 12 — VRAM integer truncation causes inaccurate values (Go)

**File:** `internal/platform/nvidia.go`, `parseVRAM()`

```go
return mib / 1024
```

Integer division truncates. For example, 48589 MiB → 47 GB (loses ~489 MiB). The test confirms this. For GPUs with e.g. 48576 MiB, this reports 47GB instead of the expected ~48GB, which could incorrectly skip models that need 48GB.

**Fix:** Use rounding: `return (mib + 512) / 1024` or return a float.

---

### BUG 13 — `Summarize()` silently swallows errors from `LoadResults()` (Go)

**File:** `internal/report/summarize.go`

```go
results, err := LoadResults(dir)
if err != nil || len(results) == 0 {
    fmt.Printf("No results in %s\n", dir)
    return nil
}
```

If `LoadResults` returns an actual error (e.g., permission denied on directory), the error is silently swallowed and a misleading "No results" message is shown.

---

### BUG 14 — Warmup reuses the same prompts as the benchmark (Go)

**File:** `internal/benchmark/runner.go`

```go
prompts := generatePrompts(cfg.NumPrompts, cfg.InputLen)
if cfg.WarmupReqs > 0 {
    for i := 0; i < cfg.WarmupReqs; i++ {
        prompt := prompts[i%len(prompts)]
        sendRequest(ctx, url, cfg, prompt, 1)
    }
}
```

If `WarmupReqs > NumPrompts`, the warmup reuses prompts via modulo, but the benchmark loop then uses the *same* prompts again. This means the server cache is already populated with those prompts, potentially inflating throughput numbers. Warmup prompts should ideally be a separate set.

---

### BUG 15 — Streaming SSE parser doesn't handle edge cases (Go + TypeScript)

**File:** `internal/benchmark/runner.go`, `parseStreamingResponse()`

```go
for scanner.Scan() {
    line := scanner.Text()
    if len(line) < 6 || line[:6] != "data: " {
```

If a single SSE data chunk spans multiple lines (unlikely for OpenAI format but possible with some servers), or if the scanner buffer overflows (default 64KB), tokens will be missed. The TypeScript version has the same issue — it doesn't properly handle partial SSE events across buffer boundaries.

---

## 🔵 Minor Issues

1. **Duplicate `models.yaml`** — Both `./models.yaml` and `./configs/models.yaml` exist with identical content. The default `--models-yaml` flag points to `./models.yaml` but `configs/` has a copy — confusing.

2. **`go vet` would flag the `break` in select** — The Go code should pass through `go vet` and `staticcheck` for additional safety.

3. **No `--warmup` equivalent in TypeScript** — Feature parity gap between Go and TS implementations.

4. **`sysinfo.Collect()` can fail silently** — If Docker isn't installed, `collectDockerVersion` tries `sudo docker` which may prompt for a password.

5. **No streaming TTFT/TPOT metrics in TypeScript** — The TS `BenchRunnerOptions` and `BenchmarkMetrics` types don't include TTFT, TPOT, or streaming-related metrics, unlike the Go version.

---

## Summary

| Severity | Count |
|----------|-------|
| 🔴 Critical | 3 |
| 🟠 High | 5 |
| 🟡 Medium | 7 |
| 🔵 Minor | 5 |
| **Total** | **20** |
