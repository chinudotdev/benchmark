BINARY     := gpu-benchmark
VERSION    := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
LDFLAGS    := -ldflags "-s -w -X main.version=$(VERSION)"
GO         := go
GOFLAGS    := -trimpath

.PHONY: build run test clean install cross-compile

## build: Build the binary for the current platform
build:
	$(GO) build $(GOFLAGS) $(LDFLAGS) -o $(BINARY) ./cmd/gpu-benchmark

## run: Run directly without building
run:
	$(GO) run ./cmd/gpu-benchmark $(ARGS)

## test: Run tests
test:
	$(GO) test ./... -v -count=1

## clean: Remove built binary
clean:
	rm -f $(BINARY)

## install: Install to $GOPATH/bin
install:
	$(GO) install $(GOFLAGS) $(LDFLAGS) ./cmd/gpu-benchmark

## cross-compile: Build for all target platforms
cross-compile:
	GOOS=linux   GOARCH=amd64 $(GO) build $(GOFLAGS) $(LDFLAGS) -o dist/$(BINARY)-linux-amd64  ./cmd/gpu-benchmark
	GOOS=linux   GOARCH=arm64 $(GO) build $(GOFLAGS) $(LDFLAGS) -o dist/$(BINARY)-linux-arm64  ./cmd/gpu-benchmark
	GOOS=darwin  GOARCH=arm64 $(GO) build $(GOFLAGS) $(LDFLAGS) -o dist/$(BINARY)-darwin-arm64  ./cmd/gpu-benchmark

## tidy: Tidy go modules
tidy:
	$(GO) mod tidy

## bench: Quick benchmark run (dev helper)
bench:
	$(GO) run ./cmd/gpu-benchmark run --model-id Qwen/Qwen3-8B --gpu-rate 2.00 --dry-run

## help: Show this help
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## //'
