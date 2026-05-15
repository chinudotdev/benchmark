package benchmark

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"time"
)

// TokenizeResponse is the response from the /tokenize endpoint.
type TokenizeResponse struct {
	Count int     `json:"count"`
	Tokens []string `json:"tokens"`
}

// Tokenize calls the /tokenize endpoint to get the exact token count for a prompt.
func Tokenize(ctx context.Context, host string, port int, model string, text string) (int, error) {
	url := fmt.Sprintf("http://%s:%d/tokenize", host, port)

	payload := map[string]any{
		"model":  model,
		"prompt": text,
		"add_special_tokens": false,
	}
	body, _ := json.Marshal(payload)

	reqCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	req, err := httpNewRequestWithContext(reqCtx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return 0, fmt.Errorf("create tokenize request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClientDo(req)
	if err != nil {
		return 0, fmt.Errorf("tokenize request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		return 0, fmt.Errorf("tokenize returned HTTP %d: %s", resp.StatusCode, truncate(string(respBody), 200))
	}

	var result TokenizeResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return 0, fmt.Errorf("decode tokenize response: %w", err)
	}

	return result.Count, nil
}

// TokenizeBatch verifies that a set of prompts approximately match the target token count.
// Returns the actual token counts for each prompt. Prompts that deviate more than
// tolerancePct percent from targetTokens are logged as warnings.
func TokenizeBatch(ctx context.Context, host string, port int, model string, prompts []string, targetTokens int, tolerancePct float64) ([]int, error) {
	counts := make([]int, len(prompts))
	warnCount := 0

	for i, p := range prompts {
		count, err := Tokenize(ctx, host, port, model, p)
		if err != nil {
			return nil, fmt.Errorf("tokenize prompt %d: %w", i, err)
		}
		counts[i] = count

		if targetTokens > 0 {
			deviation := float64(count-targetTokens) / float64(targetTokens) * 100
			if deviation < 0 {
				deviation = -deviation
			}
			if deviation > tolerancePct {
				warnCount++
			}
		}
	}

	if warnCount > 0 {
		// Summarize rather than log each one
		var sum int
		for _, c := range counts {
			sum += c
		}
		avg := float64(sum) / float64(len(counts))
		fmt.Printf("  ⚠ Tokenize check: %d/%d prompts deviate >%.0f%% from target %d (avg actual: %.0f)\n",
			warnCount, len(prompts), tolerancePct, targetTokens, avg)
	}

	return counts, nil
}
