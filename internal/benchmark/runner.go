package benchmark

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// Run executes the benchmark load generation against the target endpoint.
func Run(ctx context.Context, cfg RunnerConfig) (*Metrics, error) {
	return RunWithDataset(ctx, cfg, nil)
}

// RunWithDataset executes the benchmark with an optional prompt dataset.
// If dataset is nil, it falls back to synthetic prompt generation.
func RunWithDataset(ctx context.Context, cfg RunnerConfig, dataset *PromptDataset) (*Metrics, error) {
	url := fmt.Sprintf("http://%s:%d/v1/chat/completions", cfg.Host, cfg.Port)

	// Generate prompts
	var prompts []string
	if dataset != nil {
		prompts = dataset.GeneratePrompts(cfg.NumPrompts, cfg.InputLen)
	} else {
		prompts = generatePrompts(cfg.NumPrompts, cfg.InputLen)
	}

	// Warmup phase — use separate prompts so server cache isn't pre-warmed
	// with the exact benchmark prompts.
	if cfg.WarmupReqs > 0 {
		var warmupPrompts []string
		if dataset != nil {
			warmupPrompts = dataset.GeneratePrompts(cfg.WarmupReqs, cfg.InputLen)
		} else {
			warmupPrompts = generatePrompts(cfg.WarmupReqs, cfg.InputLen)
		}
		log.Printf("  Warming up (%d requests)...", cfg.WarmupReqs)
		for i := 0; i < cfg.WarmupReqs; i++ {
			sendRequest(ctx, url, cfg, warmupPrompts[i], 1, true)
		}
		log.Printf("  Warmup complete.")
	}

	// Probe stream_options support for streaming mode
	useStreamOpts := true
	if cfg.Stream {
		useStreamOpts = probeStreamOptions(ctx, url, cfg.Model)
		if !useStreamOpts {
			log.Println("  ⚠ Server doesn't support stream_options — token counts may be estimated")
		}
	}

	// Run benchmark
	log.Printf("  Sending %d requests (concurrency=%d, stream=%v)...", cfg.NumPrompts, cfg.Concurrency, cfg.Stream)

	var (
		results   []RequestResult
		resultsMu sync.Mutex
		sem       = make(chan struct{}, cfg.Concurrency)
		completed atomic.Int64
		start     = time.Now()
		wg        sync.WaitGroup
	)

	loop:
	for i, prompt := range prompts {
		select {
		case <-ctx.Done():
			break loop
		default:
		}

		sem <- struct{}{} // acquire
		wg.Add(1)

		go func(idx int, p string) {
			defer wg.Done()
			defer func() { <-sem }() // release

			result := sendRequest(ctx, url, cfg, p, cfg.Retries, useStreamOpts)

			resultsMu.Lock()
			results = append(results, result)
			resultsMu.Unlock()

			done := completed.Add(1)
			if done%20 == 0 || done == int64(cfg.NumPrompts) {
				elapsed := time.Since(start).Seconds()
				fmt.Printf("  Progress: %d/%d (%.1fs)\r", done, cfg.NumPrompts, elapsed)
			}
		}(i, prompt)
	}

	wg.Wait()
	fmt.Println() // newline after progress

	totalTime := time.Since(start)
	return aggregateResults(results, totalTime), nil
}

// sendRequest sends a single benchmark request with retries.
func sendRequest(ctx context.Context, url string, cfg RunnerConfig, prompt string, retries int, useStreamOpts bool) RequestResult {
	payload := map[string]any{
		"model":       cfg.Model,
		"messages":    []map[string]string{{"role": "user", "content": prompt}},
		"max_tokens":  cfg.OutputLen,
		"temperature": 1.0,
		"stream":      cfg.Stream,
	}
	if cfg.Stream && useStreamOpts {
		payload["stream_options"] = map[string]any{"include_usage": true}
	}

	body, _ := json.Marshal(payload)
	start := time.Now()

	reqCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	req, err := httpNewRequestWithContext(reqCtx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return RequestResult{Error: err.Error(), E2ELatency: time.Since(start)}
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClientDo(req)
	if err != nil {
		if retries > 0 {
			time.Sleep(2 * time.Second)
			return sendRequest(ctx, url, cfg, prompt, retries-1, useStreamOpts)
		}
		return RequestResult{Error: err.Error(), E2ELatency: time.Since(start)}
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 500 || resp.StatusCode == 429 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		if retries > 0 {
			var backoff time.Duration
			if resp.StatusCode == 429 {
				// Honor Retry-After header if present
				backoff = parseRetryAfter(resp.Header.Get("Retry-After"))
				if backoff == 0 {
					backoff = 5 * time.Second // default 429 backoff
				}
			} else {
				attempt := cfg.Retries - retries
				backoff = time.Duration(math.Pow(2, float64(attempt))) * time.Second
				if backoff > 8*time.Second {
					backoff = 8 * time.Second
				}
			}
			time.Sleep(backoff)
			return sendRequest(ctx, url, cfg, prompt, retries-1, useStreamOpts)
		}
		return RequestResult{
			E2ELatency: time.Since(start),
			Error:      fmt.Sprintf("HTTP %d: %s", resp.StatusCode, truncate(string(bodyBytes), 200)),
		}
	}

	if resp.StatusCode >= 400 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return RequestResult{
			E2ELatency: time.Since(start),
			Error:      fmt.Sprintf("HTTP %d: %s", resp.StatusCode, truncate(string(bodyBytes), 200)),
		}
	}

	// Parse response
	if cfg.Stream {
		return parseStreamingResponse(resp.Body, cfg.OutputLen, start)
	}
	return parseNonStreamingResponse(resp.Body, start)
}

// parseStreamingResponse reads SSE chunks, capturing per-token timestamps.
func parseStreamingResponse(body io.ReadCloser, maxTokens int, start time.Time) RequestResult {
	var (
		outputTokens   int
		tokenTimestamps []time.Time
		firstToken     time.Time
		gotFirst       bool
			scanner        = bufio.NewScanner(body)
	)
	// Increase scanner buffer to handle large SSE chunks
	scanner.Buffer(nil, 1024*1024) // 1MB buffer

	for scanner.Scan() {
		line := scanner.Text()
		if len(line) < 6 || line[:6] != "data: " {
			continue
		}
		data := line[6:]
		if data == "[DONE]" {
			break
		}

		var chunk map[string]any
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		// Check for usage in final chunk
		if usage, ok := chunk["usage"].(map[string]any); ok {
			if ct, ok := usage["completion_tokens"].(float64); ok {
				outputTokens = int(ct)
			}
		}

		// Check for token content (indicates a new token arrived)
		choices, _ := chunk["choices"].([]any)
		if len(choices) > 0 {
			choice, _ := choices[0].(map[string]any)
			delta, _ := choice["delta"].(map[string]any)
			if _, hasContent := delta["content"]; hasContent {
				now := time.Now()
				if !gotFirst {
					firstToken = now
					gotFirst = true
				}
				tokenTimestamps = append(tokenTimestamps, now)
			}
		}
	}

	result := RequestResult{
		Success:     true,
		OutputTokens: outputTokens,
		E2ELatency:  time.Since(start),
		TokenCount:  len(tokenTimestamps),
	}

	if gotFirst {
		result.TTFT = firstToken.Sub(start)
	}

	if outputTokens == 0 {
		outputTokens = maxTokens // fallback
	}
	result.OutputTokens = outputTokens

	// Compute TPOT (median inter-token latency)
	if len(tokenTimestamps) >= 2 {
		intervals := make([]time.Duration, len(tokenTimestamps)-1)
		for i := 1; i < len(tokenTimestamps); i++ {
			intervals[i-1] = tokenTimestamps[i].Sub(tokenTimestamps[i-1])
		}
		result.TPOT = medianDuration(intervals)
	}

	return result
}

// parseNonStreamingResponse reads a standard JSON completion response.
func parseNonStreamingResponse(body io.ReadCloser, start time.Time) RequestResult {
	data, err := io.ReadAll(body)
	if err != nil {
		return RequestResult{Error: err.Error(), E2ELatency: time.Since(start)}
	}

	var resp map[string]any
	if err := json.Unmarshal(data, &resp); err != nil {
		return RequestResult{Error: err.Error(), E2ELatency: time.Since(start)}
	}

	outputTokens := 0
	if usage, ok := resp["usage"].(map[string]any); ok {
		if ct, ok := usage["completion_tokens"].(float64); ok {
			outputTokens = int(ct)
		}
	}

	return RequestResult{
		Success:      true,
		OutputTokens: outputTokens,
		E2ELatency:   time.Since(start),
	}
}

// aggregateResults computes all metrics from individual request results.
func aggregateResults(results []RequestResult, totalTime time.Duration) *Metrics {
	m := &Metrics{
		TotalRequests: len(results),
		TotalTimeS:    totalTime.Seconds(),
		SLA:           make(map[SLABand]*SLAResult),
	}

	var successful []RequestResult
	for _, r := range results {
		if r.Success {
			successful = append(successful, r)
			m.TotalOutputTokens += int64(r.OutputTokens)
		} else {
			m.FailedReqs++
		}
	}
	m.SuccessfulReqs = len(successful)

	if len(successful) == 0 {
		return m
	}

	m.OutputTPS = float64(m.TotalOutputTokens) / totalTime.Seconds()
	m.RPS = float64(m.SuccessfulReqs) / totalTime.Seconds()

	// E2E latency percentiles
	e2eLatencies := make([]time.Duration, len(successful))
	for i, r := range successful {
		e2eLatencies[i] = r.E2ELatency
	}
	sort.Slice(e2eLatencies, func(i, j int) bool { return e2eLatencies[i] < e2eLatencies[j] })
	m.LatencyP50 = percentile(e2eLatencies, 0.50)
	m.LatencyP95 = percentile(e2eLatencies, 0.95)
	m.LatencyP99 = percentile(e2eLatencies, 0.99)

	// TTFT percentiles (streaming results only)
	var ttftVals []time.Duration
	for _, r := range successful {
		if r.TTFT > 0 {
			ttftVals = append(ttftVals, r.TTFT)
		}
	}
	if len(ttftVals) > 0 {
		sort.Slice(ttftVals, func(i, j int) bool { return ttftVals[i] < ttftVals[j] })
		m.TTFTP50 = percentile(ttftVals, 0.50)
		m.TTFTP95 = percentile(ttftVals, 0.95)
		m.TTFTP99 = percentile(ttftVals, 0.99)
	}

	// TPOT percentiles
	var tpotVals []time.Duration
	for _, r := range successful {
		if r.TPOT > 0 {
			tpotVals = append(tpotVals, r.TPOT)
		}
	}
	if len(tpotVals) > 0 {
		sort.Slice(tpotVals, func(i, j int) bool { return tpotVals[i] < tpotVals[j] })
		m.TPOTP50 = percentile(tpotVals, 0.50)
		m.TPOTP95 = percentile(tpotVals, 0.95)
		m.TPOTP99 = percentile(tpotVals, 0.99)

		// Tokens/sec/user = 1 / median_ITL
		m.TokensPerSecPerUser = 1.0 / m.TPOTP50.Seconds()
	}

	// SLA compliance
	for band, def := range SLADefs {
		sla := &SLAResult{Band: band, TotalCount: m.SuccessfulReqs}
		var slaTokens int64
		for _, r := range successful {
			if meetsSLA(r, def) {
				sla.MetCount++
				slaTokens += int64(r.OutputTokens)
			}
		}
		sla.GoodputTPS = float64(slaTokens) / totalTime.Seconds()
		m.SLA[band] = sla
	}

	return m
}

// meetsSLA checks if a request result satisfies the given SLA definition.
func meetsSLA(r RequestResult, def SLADef) bool {
	if def.TTFTTarget > 0 && r.TTFT > def.TTFTTarget {
		return false
	}
	if def.TPOTTarget > 0 && r.TPOT > def.TPOTTarget {
		return false
	}
	return true
}

// percentile returns the value at the given percentile from a sorted slice.
func percentile(sorted []time.Duration, p float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(p*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// medianDuration returns the median of a duration slice.
func medianDuration(d []time.Duration) time.Duration {
	if len(d) == 0 {
		return 0
	}
	sorted := make([]time.Duration, len(d))
	copy(sorted, d)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	mid := len(sorted) / 2
	if len(sorted)%2 == 0 {
		return (sorted[mid-1] + sorted[mid]) / 2
	}
	return sorted[mid]
}

// truncate shortens a string to maxLen runes, preserving valid UTF-8.
func truncate(s string, maxLen int) string {
	runes := []rune(s)
	if len(runes) <= maxLen {
		return s
	}
	return string(runes[:maxLen])
}

// probeStreamOptions sends a tiny streaming request with stream_options
// to check if the server supports it.
func probeStreamOptions(ctx context.Context, url, model string) bool {
	payload := map[string]any{
		"model":       model,
		"messages":    []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens":  1,
		"stream":      true,
		"stream_options": map[string]any{"include_usage": true},
	}
	body, _ := json.Marshal(payload)

	reqCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	req, err := httpNewRequestWithContext(reqCtx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return false
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClientDo(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		// Drain the body
		io.ReadAll(resp.Body)
		return false
	}

	return true
}

// generatePrompts creates deterministic synthetic prompts.
// This is a placeholder — Milestone 3 replaces this with a fixed dataset loader.
func generatePrompts(count, targetTokens int) []string {
	words := []string{
		"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
		"data", "model", "inference", "token", "compute", "benchmark",
		"gpu", "neural", "network", "transformer", "attention",
		"embedding", "weight",
	}

	rng := rand.New(rand.NewSource(42))
	prompts := make([]string, count)
	for i := 0; i < count; i++ {
		targetChars := targetTokens * 4
		var w []string
		totalLen := 0
		for totalLen < targetChars {
			word := words[rng.Intn(len(words))]
			w = append(w, word)
			totalLen += len(word) + 1
		}
		prompts[i] = joinWords(w)
	}
	return prompts
}

func joinWords(w []string) string {
	total := 0
	for _, s := range w {
		total += len(s) + 1
	}
	var buf bytes.Buffer
	buf.Grow(total)
	for i, s := range w {
		if i > 0 {
			buf.WriteByte(' ')
		}
		buf.WriteString(s)
	}
	return buf.String()
}

// parseRetryAfter parses the HTTP Retry-After header value.
// It handles both seconds ("120") and HTTP-date formats.
// Returns 0 if the value cannot be parsed.
func parseRetryAfter(value string) time.Duration {
	return parseRetryAfterHeader(value)
}
