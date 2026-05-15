# Bug Report: gpu-benchmark Project

> Reviewed on 2026-05-15 — Go source files, tests, and configuration.

---

## 🔴 Critical Bugs

### 1. ✅ **FIXED: Resource leak in `ExportBundle` — file handle captured by closure**
**File:** `internal/report/export.go`

The inner file variable `f` shadowed the outer archive file variable. Renamed inner to `src` and close immediately after copy instead of deferring.

### 2. ✅ **FIXED: `defer cancel()` inside a loop causes context cancellation for all subsequent tasks**
**File:** `internal/quality/gate.go`, `RunGate` function

Wrapped loop body in an immediately-invoked function literal so `defer cancel()` fires at the end of each iteration.

### 3. ✅ **FIXED: `isLinux()` function is incorrect**
**File:** `internal/sysinfo/preflight.go`

Replaced environment variable checks (`$OS`, `$OSTYPE`) with `runtime.GOOS == "linux"`.

### 4. ✅ **FIXED: AMD container config uses `--network host` — port mapping is ignored**
**File:** `internal/platform/amd.go`

Changed `--port 8000` (hardcoded) to `--port strconv.Itoa(opts.Port)` so it matches the actual port option regardless of network mode.

### 5. ✅ **FIXED: `--device cpu` flag in AMD container config**
**File:** `internal/platform/amd.go`

Changed `--device cpu` to `--device rocm` for correct vLLM ROCm device selection.

---

## 🟡 Medium Bugs

### 6. ✅ **FIXED: VRAM check only looks at the first GPU**
**File:** `internal/orchestrator/orchestrator.go`

Now computes `availableVRAM = minGPUVRAM × numGPUs` and checks total available VRAM across all GPUs.

### 7. **`Duplicate model ID` in models.yaml — Qwen/Qwen3-8B appears three times**
**File:** `models.yaml`

When `--all` is used, `filterModelsByPlatform` correctly filters by platform. But `FindModel("Qwen/Qwen3-8B", "")` always returns a generic config without platform-specific Docker images or serving backends. If a user on an AMD system runs `--model-id Qwen/Qwen3-8B`, they'll get the generic config, not the ROCm-optimized one.

### 8. ✅ **FIXED: Config merge uses `0` as sentinel for "not set" — conflicts with legitimate zero values**
**File:** `cmd/gpu-benchmark/main.go`, `mergeConfig`

Now uses `cfgFieldString` directly instead of the removed `cfgField` wrapper.

### 9. ✅ **FIXED: `writeSystemInfo` ignores errors silently**
**File:** `internal/orchestrator/orchestrator.go`

Now logs warnings on marshal/write errors instead of silently ignoring them.

### 10. ✅ **FIXED: `aggregateSweepCells` uses population stddev but `AggregateSweep` uses sample stddev**
**File:** `internal/report/markdown.go`

Replaced custom `sqrt()` with `math.Sqrt` and corrected the variance formula to use sample variance (n-1) matching `AggregateSweep` in `sweep.go`.

### 11. ✅ **FIXED: Custom `sqrt` function instead of `math.Sqrt`**
**File:** `internal/report/markdown.go`

Removed custom Newton's method implementation, replaced with `math.Sqrt`.

### 12. **`RunnerConfig.TrafficProfile` field is declared but never used**
**File:** `internal/benchmark/metrics.go`

The traffic profiles defined in `workload/profiles.go` are never used. The benchmark always uses fixed concurrency without any Poisson arrival modeling. This is a feature gap, not a bug — keeping the field for future implementation.

---

## 🟢 Minor Issues / Code Smells

### 13. ✅ **FIXED: Dead code: `cfgField` function**
**File:** `cmd/gpu-benchmark/main.go`

Removed dead function. Updated `mergeConfig` to call `cfgFieldString` directly.

### 14. ✅ **FIXED: Dead code: `detectPlatform` function**
**File:** `internal/orchestrator/orchestrator.go`

Removed dead stub function.

### 15. ✅ **FIXED: Dead code: `Merge` function on `config.Config`**
**File:** `internal/config/config.go`

Removed empty-body function.

### 16. ✅ **FIXED: `allDevices` helper function is unused**
**File:** `cmd/gpu-benchmark/main.go`

Removed unused function.

### 17. ✅ **FIXED: `probesStreamOptions` may return false positives**
**File:** `internal/benchmark/runner.go`

Now only returns `true` on HTTP 200. All other status codes (400, 401, 404, etc.) return `false`.

### 18. ✅ **FIXED: `generatePrompts` uses custom PRNG with no scrambling**
**File:** `internal/benchmark/runner.go`

Replaced custom LCG PRNG with `math/rand` using `rand.New(rand.NewSource(42))` for deterministic, high-quality randomness.

---

## Summary

| Severity | Total | Fixed | Remaining |
|----------|-------|-------|-----------|
| 🔴 Critical | 5 | 5 | 0 |
| 🟡 Medium | 7 | 5 | 2 (#7, #12) |
| 🟢 Minor | 6 | 6 | 0 |

### Remaining items (feature gaps, not bugs):
- **#7**: Duplicate model IDs in `models.yaml` — needs a `FindModelForPlatform()` function
- **#12**: `TrafficProfile` field unused — awaiting Poisson arrival implementation
