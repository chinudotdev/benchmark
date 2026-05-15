package docker

import (
	"context"
	"net/http"
)

// Thin wrappers for testability.
func httpNewGetWithContext(ctx context.Context, url string) (*http.Request, error) {
	return http.NewRequestWithContext(ctx, "GET", url, nil)
}

func httpClientDo(req *http.Request) (*http.Response, error) {
	return http.DefaultClient.Do(req)
}
