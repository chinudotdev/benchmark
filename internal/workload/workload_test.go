package workload

import (
	"os"
	"path/filepath"
	"testing"
)

const testModelsYAML = `
models:
  - id: Qwen/Qwen3-8B
    name: Qwen3-8B
    quant: none
    min_vram_gb: 18
    tp: 1
    extra_flags: "--enable-prefix-caching"

  - id: meta-llama/Llama-3.3-70B-Instruct
    name: Llama-3.3-70B-AWQ
    quant: awq
    min_vram_gb: 42
    tp: 1
    extra_flags: ""
    platform: nvidia
    docker_image: "vllm/vllm-openai:latest"
    serving_backend: vllm
`

func writeTestYAML(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "models.yaml")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadModels(t *testing.T) {
	path := writeTestYAML(t, testModelsYAML)
	models, err := LoadModels(path)
	if err != nil {
		t.Fatalf("LoadModels() error: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}

	// First model
	if models[0].ID != "Qwen/Qwen3-8B" {
		t.Errorf("models[0].ID = %q", models[0].ID)
	}
	if models[0].Name != "Qwen3-8B" {
		t.Errorf("models[0].Name = %q", models[0].Name)
	}
	if models[0].Quant != "none" {
		t.Errorf("models[0].Quant = %q", models[0].Quant)
	}
	if models[0].MinVRAM_GB != 18 {
		t.Errorf("models[0].MinVRAM_GB = %d", models[0].MinVRAM_GB)
	}
	if models[0].TP != 1 {
		t.Errorf("models[0].TP = %d", models[0].TP)
	}

	// Second model
	if models[1].ID != "meta-llama/Llama-3.3-70B-Instruct" {
		t.Errorf("models[1].ID = %q", models[1].ID)
	}
	if models[1].Quant != "awq" {
		t.Errorf("models[1].Quant = %q", models[1].Quant)
	}
	// Platform-specific fields
	if models[1].Platform != "nvidia" {
		t.Errorf("models[1].Platform = %q, want nvidia", models[1].Platform)
	}
	if models[1].DockerImage != "vllm/vllm-openai:latest" {
		t.Errorf("models[1].DockerImage = %q", models[1].DockerImage)
	}
	if models[1].ServingBackend != "vllm" {
		t.Errorf("models[1].ServingBackend = %q", models[1].ServingBackend)
	}
}

func TestLoadModelsMissingID(t *testing.T) {
	yaml := `
models:
  - name: No ID Model
    quant: none
`
	path := writeTestYAML(t, yaml)
	_, err := LoadModels(path)
	if err == nil {
		t.Error("expected error for model missing ID")
	}
}

func TestLoadModelsDefaultValues(t *testing.T) {
	yaml := `
models:
  - id: test/model
`
	path := writeTestYAML(t, yaml)
	models, err := LoadModels(path)
	if err != nil {
		t.Fatalf("LoadModels() error: %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}
	m := models[0]
	// Name defaults to ID
	if m.Name != "test/model" {
		t.Errorf("Name = %q, want %q", m.Name, "test/model")
	}
	// TP defaults to 1
	if m.TP != 1 {
		t.Errorf("TP = %d, want 1", m.TP)
	}
	// Quant defaults to "none"
	if m.Quant != "none" {
		t.Errorf("Quant = %q, want %q", m.Quant, "none")
	}
}

func TestLoadModelsFileNotFound(t *testing.T) {
	_, err := LoadModels("/nonexistent/models.yaml")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestLoadModelsInvalidYAML(t *testing.T) {
	path := writeTestYAML(t, "{{invalid yaml::")
	_, err := LoadModels(path)
	if err == nil {
		t.Error("expected error for invalid YAML")
	}
}

func TestFindModel(t *testing.T) {
	m := FindModel("Qwen/Qwen3-8B", "fp8")
	if m.ID != "Qwen/Qwen3-8B" {
		t.Errorf("ID = %q", m.ID)
	}
	if m.Name != "Qwen3-8B" {
		t.Errorf("Name = %q", m.Name)
	}
	if m.Quant != "fp8" {
		t.Errorf("Quant = %q", m.Quant)
	}
	if m.TP != 1 {
		t.Errorf("TP = %d", m.TP)
	}
}

func TestFindModelDefaultQuant(t *testing.T) {
	m := FindModel("test/model", "")
	if m.Quant != "none" {
		t.Errorf("Quant = %q, want %q", m.Quant, "none")
	}
}

func TestFindModelNoSlash(t *testing.T) {
	m := FindModel("local-model", "")
	if m.Name != "local-model" {
		t.Errorf("Name = %q, want %q", m.Name, "local-model")
	}
}
