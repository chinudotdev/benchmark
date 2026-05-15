package benchmark

import (
	"context"
	"io"
	"net/http"
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
