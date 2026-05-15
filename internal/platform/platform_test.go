package platform

import (
	"testing"
)

func TestParseVRAM(t *testing.T) {
	tests := []struct {
		input string
		want  int
	}{
		{"81920 MiB", 80},
		{"16384 MiB", 16},
		{"24576 MiB", 24},
		{"0 MiB", 0},
		{"unknown", 0},
		{"", 0},
		{"48589 MiB", 47},  // 48589/1024 = 47.45, rounds to 47
		{"49152 MiB", 48},  // 49152/1024 = 48.00, exact
		{"49600 MiB", 48},  // 49600/1024 = 48.44, rounds to 48
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := parseVRAM(tt.input)
			if got != tt.want {
				t.Errorf("parseVRAM(%q) = %d, want %d", tt.input, got, tt.want)
			}
		})
	}
}

func TestQuantFlags(t *testing.T) {
	tests := []struct {
		quant string
		want  []string
	}{
		{"none", nil},
		{"", nil},
		{"int8", []string{"--quantization", "bitsandbytes"}},
		{"int4", []string{"--quantization", "bitsandbytes", "--load-in-4bit"}},
		{"awq", []string{"--quantization", "awq"}},
		{"fp8", []string{"--quantization", "fp8"}},
		{"unknown", nil},
	}

	for _, tt := range tests {
		t.Run(tt.quant, func(t *testing.T) {
			got := quantFlags(tt.quant)
			if len(got) != len(tt.want) {
				t.Errorf("quantFlags(%q) = %v, want %v", tt.quant, got, tt.want)
				return
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("quantFlags(%q)[%d] = %q, want %q", tt.quant, i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestNVIDIAPlatformName(t *testing.T) {
	p := NewNVIDIAPlatform()
	if p.Name() != "nvidia" {
		t.Errorf("Name() = %q, want %q", p.Name(), "nvidia")
	}
}

func TestAMDPlatformName(t *testing.T) {
	p := NewAMDPlatform()
	if p.Name() != "amd" {
		t.Errorf("Name() = %q, want %q", p.Name(), "amd")
	}
}

func TestTenstorrentPlatformName(t *testing.T) {
	p := NewTenstorrentPlatform()
	if p.Name() != "tenstorrent" {
		t.Errorf("Name() = %q, want %q", p.Name(), "tenstorrent")
	}
}

func TestNVIDIAContainerConfig(t *testing.T) {
	p := NewNVIDIAPlatform()
	model := ModelConfig{
		ID:         "Qwen/Qwen3-8B",
		Name:       "Qwen3-8B",
		Quant:      "none",
		TP:         1,
		ExtraFlags: "--enable-prefix-caching",
	}
	opts := RunOptions{
		GPURate:     2.0,
		GPUCount:    1,
		MaxModelLen: 8192,
		Port:        8000,
		HFToken:     "hf_test_token",
		GPUIDs:      "all",
	}

	cfg := p.ContainerConfig(model, opts)
	if cfg == nil {
		t.Fatal("ContainerConfig returned nil")
	}
	if cfg.Image != "vllm/vllm-openai:latest" {
		t.Errorf("Image = %q, want default", cfg.Image)
	}
	if cfg.Name != "vllm_bench_8000" {
		t.Errorf("Name = %q", cfg.Name)
	}
	if cfg.Port != 8000 {
		t.Errorf("Port = %d, want 8000", cfg.Port)
	}
	if cfg.EnvVars["HUGGING_FACE_HUB_TOKEN"] != "hf_test_token" {
		t.Errorf("HF token not set in env vars")
	}
	// Should contain model ID and TP flag
	found := false
	for _, arg := range cfg.ExtraArgs {
		if arg == "Qwen/Qwen3-8B" {
			found = true
		}
	}
	if !found {
		t.Errorf("ExtraArgs missing model ID: %v", cfg.ExtraArgs)
	}
}

func TestNVIDIAGetDockerImageOverride(t *testing.T) {
	p := NewNVIDIAPlatform()
	if img := p.GetDockerImage("custom/image:v1"); img != "custom/image:v1" {
		t.Errorf("GetDockerImage override = %q, want %q", img, "custom/image:v1")
	}
	if img := p.GetDockerImage(""); img != "vllm/vllm-openai:latest" {
		t.Errorf("GetDockerImage default = %q, want %q", img, "vllm/vllm-openai:latest")
	}
}

func TestAMDContainerConfig(t *testing.T) {
	p := NewAMDPlatform()
	model := ModelConfig{
		ID:    "Qwen/Qwen3-8B",
		Name:  "Qwen3-8B",
		Quant: "none",
		TP:    1,
	}
	opts := RunOptions{
		MaxModelLen: 4096,
		Port:        8001,
	}

	cfg := p.ContainerConfig(model, opts)
	if cfg == nil {
		t.Fatal("ContainerConfig returned nil")
	}
	if cfg.Name != "vllm_bench_8001" {
		t.Errorf("Name = %q", cfg.Name)
	}
	// AMD should use --device flags, not --gpus
	hasKFD := false
	for i, f := range cfg.GPUFlags {
		if f == "/dev/kfd" && i > 0 && cfg.GPUFlags[i-1] == "--device" {
			hasKFD = true
		}
	}
	if !hasKFD {
		t.Errorf("GPUFlags missing /dev/kfd: %v", cfg.GPUFlags)
	}
}
