package config

import (
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.GPURate != 2.00 {
		t.Errorf("default GPURate = %f, want 2.00", cfg.GPURate)
	}
	if cfg.Concurrency != 32 {
		t.Errorf("default Concurrency = %d, want 32", cfg.Concurrency)
	}
	if cfg.InputLen != 512 {
		t.Errorf("default InputLen = %d, want 512", cfg.InputLen)
	}
}

func TestLoadFileNotFound(t *testing.T) {
	cfg, err := LoadFile()
	if err != nil {
		t.Fatalf("LoadFile returned error: %v", err)
	}
	if cfg != nil {
		t.Log("Found a config file in parent directories")
	}
}

func TestLoadPath(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".gpu-benchmark.yaml")

	content := `
gpu-rate: 3.50
concurrency: 64
stream: true
repeat: 3
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadPath(path)
	if err != nil {
		t.Fatalf("LoadPath error: %v", err)
	}
	if cfg == nil {
		t.Fatal("config is nil")
	}
	if cfg.GPURate != 3.50 {
		t.Errorf("GPURate = %f, want 3.50", cfg.GPURate)
	}
	if cfg.Concurrency != 64 {
		t.Errorf("Concurrency = %d, want 64", cfg.Concurrency)
	}
	if cfg.Stream == nil || *cfg.Stream != true {
		t.Error("Stream should be true")
	}
	if cfg.Repeat != 3 {
		t.Errorf("Repeat = %d, want 3", cfg.Repeat)
	}
}

func TestLoadPathDefaults(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".gpu-benchmark.yaml")

	content := `gpu-rate: 5.00
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadPath(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.GPURate != 5.00 {
		t.Errorf("GPURate = %f, want 5.00", cfg.GPURate)
	}
	if cfg.Concurrency != 32 {
		t.Errorf("Concurrency = %d, want 32 (default)", cfg.Concurrency)
	}
}

func TestLoadPathNotExist(t *testing.T) {
	cfg, err := LoadPath("/nonexistent/.gpu-benchmark.yaml")
	if err != nil {
		t.Fatalf("expected nil error for nonexistent file, got: %v", err)
	}
	if cfg != nil {
		t.Error("expected nil config for nonexistent file")
	}
}

func TestLoadPathInvalidYAML(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".gpu-benchmark.yaml")
	os.WriteFile(path, []byte("{{invalid yaml"), 0o644)

	_, err := LoadPath(path)
	if err == nil {
		t.Error("expected error for invalid YAML")
	}
}

func TestExampleConfigIsValidYAML(t *testing.T) {
	example := ExampleConfig()
	if example == "" {
		t.Error("ExampleConfig returned empty string")
	}
	cfg := DefaultConfig()
	if err := yaml.Unmarshal([]byte(example), &cfg); err != nil {
		t.Errorf("ExampleConfig is not valid YAML: %v", err)
	}
}
