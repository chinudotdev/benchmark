package benchmark

import (
	"context"
	"io"
	"net/http"
	"strconv"
	"time"
)

// Thin wrappers around net/http so the runner doesn't import
// net/http directly — makes testing with mocks easy later.

var httpClient = &http.Client{
	Timeout: 0, // per-request timeout handled via context
}

func httpNewRequestWithContext(ctx context.Context, method, url string, body io.Reader) (*http.Request, error) {
	return http.NewRequestWithContext(ctx, method, url, body)
}

func httpClientDo(req *http.Request) (*http.Response, error) {
	return httpClient.Do(req)
}

// httpParseTime parses an HTTP date string.
func httpParseTime(value string) (time.Time, error) {
	return http.ParseTime(value)
}

// parseRetryAfterHeader parses the Retry-After header value as seconds.
func parseRetryAfterHeader(value string) time.Duration {
	if value == "" {
		return 0
	}
	if secs, err := strconv.Atoi(value); err == nil {
		return time.Duration(secs) * time.Second
	}
	if t, err := http.ParseTime(value); err == nil {
		d := time.Until(t)
		if d > 0 {
			return d
		}
	}
	return 0
}
