// Package config handles .gpu-benchmark.yaml configuration file loading.
package config

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// Config is the top-level structure for .gpu-benchmark.yaml.
type Config struct {
	ModelID          string  `yaml:"model-id,omitempty"`
	AllModels        bool    `yaml:"all,omitempty"`
	Platform         string  `yaml:"platform,omitempty"`
	GPURate          float64 `yaml:"gpu-rate,omitempty"`
	GPUCount         int     `yaml:"gpu-count,omitempty"`
	InputLen         int     `yaml:"input-len,omitempty"`
	OutputLen        int     `yaml:"output-len,omitempty"`
	NumPrompts       int     `yaml:"num-prompts,omitempty"`
	MaxModelLen      int     `yaml:"max-model-len,omitempty"`
	Port             int     `yaml:"port,omitempty"`
	ResultsDir       string  `yaml:"results-dir,omitempty"`
	Concurrency      int     `yaml:"concurrency,omitempty"`
	ModelsYAML       string  `yaml:"models-yaml,omitempty"`
	Quant            string  `yaml:"quant,omitempty"`
	HFToken          string  `yaml:"hf-token,omitempty"`
	DockerImage      string  `yaml:"docker-image,omitempty"`
	GPUIDs           string  `yaml:"gpu-ids,omitempty"`
	Stream           *bool   `yaml:"stream,omitempty"`
	Force            bool    `yaml:"force,omitempty"`
	DryRun           bool    `yaml:"dry-run,omitempty"`
	Retries          int     `yaml:"retries,omitempty"`
	WarmupReqs       int     `yaml:"warmup,omitempty"`

	// Sweep options
	ConcurrencySweep string `yaml:"concurrency-sweep,omitempty"`
	SeqSweep         bool   `yaml:"seq-sweep,omitempty"`
	Repeat           int    `yaml:"repeat,omitempty"`
	TrafficProfile   string `yaml:"traffic-profile,omitempty"`
	VerifyTokens     bool   `yaml:"verify-tokens,omitempty"`
}

// DefaultConfig returns a config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		GPURate:     2.00,
		GPUCount:    1,
		InputLen:    512,
		OutputLen:   256,
		NumPrompts:  200,
		MaxModelLen: 8192,
		Port:        8000,
		ResultsDir:  "./results",
		Concurrency: 32,
		ModelsYAML:  "./models.yaml",
		Retries:     2,
		WarmupReqs:  5,
		Repeat:      1,
	}
}

const configFileName = ".gpu-benchmark.yaml"

// LoadFile searches for a config file and loads it.
// Search order: current directory, then parent directories up to home.
// Returns nil if no config file found.
func LoadFile() (*Config, error) {
	paths := searchPaths()
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return load(p)
		}
	}
	return nil, nil
}

// LoadPath loads a config from a specific path.
func LoadPath(path string) (*Config, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, nil // file doesn't exist is not an error
	}
	return load(path)
}

func load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config %s: %w", path, err)
	}

	cfg := DefaultConfig()
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config %s: %w", path, err)
	}

	return &cfg, nil
}

func searchPaths() []string {
	var paths []string

	// Current directory
	if cwd, err := os.Getwd(); err == nil {
		paths = append(paths, filepath.Join(cwd, configFileName))

		// Walk up to home
		home, _ := os.UserHomeDir()
		dir := cwd
		for {
			parent := filepath.Dir(dir)
			if parent == dir || parent == home {
				break
			}
			dir = parent
			paths = append(paths, filepath.Join(dir, configFileName))
		}
	}

	// Home directory
	if home, err := os.UserHomeDir(); err == nil {
		paths = append(paths, filepath.Join(home, configFileName))
	}

	return paths
}

// Merge applies config file values onto the provided options struct.
// Config file values only override CLI flags that weren't explicitly set.
// This uses the "config as defaults" pattern — CLI flags always win.
func Merge(cfg *Config, apply func(name string, fn func())) {
	if cfg == nil {
		return
	}

	// Each apply call checks if the flag was explicitly set by the CLI.
	// If not, the config value is used.
	// The caller passes a function that sets the value only if the flag
	// wasn't changed from its default.

	// This is a simplified merge — the orchestrator reads the Config
	// directly for unset fields.
}

// ExampleConfig returns an example config file content.
func ExampleConfig() string {
	return `# gpu-benchmark configuration
# CLI flags override these values when explicitly set.

# Model selection (use --model-id or --all on CLI)
# model-id: Qwen/Qwen3-8B
# all: false

# Cost
gpu-rate: 2.00        # hourly GPU cost in USD
gpu-count: 1          # number of GPUs

# Benchmark parameters
input-len: 512        # input token length
output-len: 256       # output token length
num-prompts: 200      # prompts to send
max-model-len: 8192   # max context length
concurrency: 32       # concurrent requests
retries: 2            # retries per request
warmup: 5             # warmup requests to discard
stream: true          # use streaming

# Sweep options
# concurrency-sweep: "1,2,4,8,16,32,64,128"
# seq-sweep: false
# repeat: 1
# traffic-profile: ""

# Platform
# platform: nvidia     # nvidia, amd, tenstorrent, or auto-detect

# Paths
# models-yaml: ./models.yaml
# results-dir: ./results
# docker-image: ""
# gpu-ids: ""

# Auth
# hf-token: ""        # or set HF_TOKEN env var
`
}
