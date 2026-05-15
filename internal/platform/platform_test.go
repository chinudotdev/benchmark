package platform

import (
	"testing"
)

// ── AutoDetect + ProbeAll ──────────────────────────────────────────────────

func TestAutoDetectUnknown(t *testing.T) {
	_, _, err := AutoDetect(t.Context(), "bogus")
	if err == nil {
		t.Error("expected error for unknown platform")
	}
}

func TestAutoDetectExplicit(t *testing.T) {
	// Explicitly request nvidia — will fail on non-NVIDIA but returns platform
	// On systems without NVIDIA it should return error
	_, _, err := AutoDetect(t.Context(), "nvidia")
	if err != nil {
		t.Logf("nvidia not available (expected on non-GPU): %v", err)
	}
}

func TestAutoDetectEmpty(t *testing.T) {
	// Empty string triggers full probe — will fail if no GPU at all
	_, _, err := AutoDetect(t.Context(), "")
	if err != nil {
		t.Logf("no GPU detected (expected on dev machine): %v", err)
	}
}

func TestProbeAll(t *testing.T) {
	detected := ProbeAll(t.Context())
	// On a dev machine without GPUs, this should return empty
	if len(detected) > 0 {
		for _, d := range detected {
			t.Logf("Found: %s (%d devices)", d.Platform.Name(), len(d.Hardware.Devices))
		}
	} else {
		t.Log("No platforms detected (expected on non-GPU system)")
	}
}

func TestAutoDetectCaseInsensitive(t *testing.T) {
	_, _, err := AutoDetect(t.Context(), "NVIDIA")
	if err != nil {
		t.Logf("NVIDIA not available: %v", err)
	}
	_, _, err = AutoDetect(t.Context(), "AMD")
	if err != nil {
		t.Logf("AMD not available: %v", err)
	}
}

// ── AMD Platform ────────────────────────────────────────────────────────────

func TestAMDPlatformName(t *testing.T) {
	p := NewAMDPlatform()
	if p.Name() != "amd" {
		t.Errorf("Name() = %q, want %q", p.Name(), "amd")
	}
}

func TestAMDGetDockerImage(t *testing.T) {
	p := NewAMDPlatform()
	if img := p.GetDockerImage(""); img != "rocm/vllm:latest" {
		t.Errorf("default image = %q, want rocm/vllm:latest", img)
	}
	if img := p.GetDockerImage("custom/rocm:v2"); img != "custom/rocm:v2" {
		t.Errorf("override image = %q, want custom/rocm:v2", img)
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
		MaxModelLen: 8192,
		Port:        8001,
	}

	cfg := p.ContainerConfig(model, opts)
	if cfg == nil {
		t.Fatal("ContainerConfig returned nil")
	}
	if cfg.Name != "vllm_bench_8001" {
		t.Errorf("Name = %q, want vllm_bench_8001", cfg.Name)
	}
	if cfg.Image != "rocm/vllm:latest" {
		t.Errorf("Image = %q", cfg.Image)
	}
	if cfg.Port != 8001 {
		t.Errorf("Port = %d, want 8001", cfg.Port)
	}

	// AMD uses --device /dev/kfd and --device /dev/dri
	hasKFD := false
	hasDRI := false
	hasVideoGroup := false
	for i, f := range cfg.GPUFlags {
		if f == "/dev/kfd" && i > 0 && cfg.GPUFlags[i-1] == "--device" {
			hasKFD = true
		}
		if f == "/dev/dri" && i > 0 && cfg.GPUFlags[i-1] == "--device" {
			hasDRI = true
		}
		if f == "video" && i > 0 && cfg.GPUFlags[i-1] == "--group-add" {
			hasVideoGroup = true
		}
	}
	if !hasKFD {
		t.Error("missing --device /dev/kfd in GPUFlags")
	}
	if !hasDRI {
		t.Error("missing --device /dev/dri in GPUFlags")
	}
	if !hasVideoGroup {
		t.Error("missing --group-add video in GPUFlags")
	}
}

func TestAMDContainerConfigWithGPUs(t *testing.T) {
	p := NewAMDPlatform()
	p.hw = &HardwareInfo{
		Platform: "amd",
		Devices: []DeviceInfo{
			{Name: "MI300X", VRAM_GB: 192, GFXArch: "gfx942"},
		},
	}

	model := ModelConfig{ID: "test/model", Name: "test", TP: 1}
	opts := RunOptions{
		MaxModelLen: 4096,
		Port:        8000,
		GPUIDs:      "0,1",
		HFToken:     "hf_test",
	}

	cfg := p.ContainerConfig(model, opts)
	if cfg.EnvVars["HIP_VISIBLE_DEVICES"] != "0,1" {
		t.Errorf("HIP_VISIBLE_DEVICES = %q, want '0,1'", cfg.EnvVars["HIP_VISIBLE_DEVICES"])
	}
	if cfg.EnvVars["HUGGING_FACE_HUB_TOKEN"] != "hf_test" {
		t.Error("HF token not set")
	}
	if cfg.EnvVars["HSA_OVERRIDE_GFX_VERSION"] != "9.4.2" {
		t.Errorf("HSA_OVERRIDE_GFX_VERSION = %q, want '9.4.2'", cfg.EnvVars["HSA_OVERRIDE_GFX_VERSION"])
	}
}

func TestAMDContainerConfigQuant(t *testing.T) {
	p := NewAMDPlatform()
	model := ModelConfig{ID: "test/model", Name: "test", Quant: "awq", TP: 1}
	opts := RunOptions{Port: 8000}

	cfg := p.ContainerConfig(model, opts)

	foundAwq := false
	for _, arg := range cfg.ExtraArgs {
		if arg == "awq" {
			foundAwq = true
		}
	}
	if !foundAwq {
		t.Error("missing awq quantization flag in ExtraArgs")
	}
}

func TestAMDHealthEndpoint(t *testing.T) {
	p := NewAMDPlatform()
	if ep := p.HealthEndpoint(); ep != "/health" {
		t.Errorf("HealthEndpoint = %q, want /health", ep)
	}
}

func TestGfxToHSA(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"gfx942", "9.4.2"},
		{"gfx90a", "9.0.10"},
		{"gfx941", "9.4.1"},
		{"gfx1100", "1.1.0"},
		{"gfx900", "9.0.0"},
		{"invalid", "invalid"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := gfxToHSA(tt.input)
			if got != tt.want {
				t.Errorf("gfxToHSA(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestParseGFXVersion(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"0x90012", "gfx9012"},
		{"0x90000", "gfx900"},
		{"invalid", ""},
		{"", ""},
	}

	for _, tt := range tests {
		got := parseGFXVersion(tt.input)
		if got != tt.want {
			t.Errorf("parseGFXVersion(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestPCIIDToGFX(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"1002:740C", "gfx942"}, // MI300X
		{"1002:7408", "gfx942"}, // MI300A
		{"1002:738C", "gfx90a"}, // MI250X
		{"1002:7480", "gfx1100"}, // RX 7900 XTX
		{"1002:9999", ""},       // Unknown
	}

	for _, tt := range tests {
		got := pciIDToGFX(tt.input)
		if got != tt.want {
			t.Errorf("pciIDToGFX(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestAMDDetectNoHardware(t *testing.T) {
	p := NewAMDPlatform()
	// On a system without AMD GPUs, detection should fail
	_, err := p.DetectHardware(t.Context())
	if err == nil {
		t.Log("AMD GPU detected (running on AMD hardware)")
	} else {
		// Expected on non-AMD systems
		t.Logf("DetectHardware correctly returned error: %v", err)
	}
}

// ── Tenstorrent Platform ────────────────────────────────────────────────────

func TestTenstorrentPlatformName(t *testing.T) {
	p := NewTenstorrentPlatform()
	if p.Name() != "tenstorrent" {
		t.Errorf("Name() = %q, want %q", p.Name(), "tenstorrent")
	}
}

func TestTenstorrentGetDockerImage(t *testing.T) {
	p := NewTenstorrentPlatform()
	if img := p.GetDockerImage(""); img != "ghcr.io/tenstorrent/tt-inference-server:latest" {
		t.Errorf("default image = %q", img)
	}
	if img := p.GetDockerImage("custom/tt:v1"); img != "custom/tt:v1" {
		t.Errorf("override image = %q", img)
	}
}

func TestTenstorrentContainerConfig(t *testing.T) {
	p := NewTenstorrentPlatform()
	p.hw = &HardwareInfo{
		Platform: "tenstorrent",
		Devices: []DeviceInfo{
			{Name: "Tenstorrent N300", VRAM_GB: 12, PCIAddress: "0000:3b:00.0"},
		},
	}

	model := ModelConfig{
		ID:    "Qwen/Qwen3-8B",
		Name:  "Qwen3-8B",
		Quant: "none",
		TP:    1,
	}
	opts := RunOptions{
		MaxModelLen: 4096,
		Port:        8002,
		HFToken:     "hf_tt_test",
	}

	cfg := p.ContainerConfig(model, opts)
	if cfg == nil {
		t.Fatal("ContainerConfig returned nil")
	}
	if cfg.Name != "tt_bench_8002" {
		t.Errorf("Name = %q, want tt_bench_8002", cfg.Name)
	}
	if cfg.Port != 8002 {
		t.Errorf("Port = %d, want 8002", cfg.Port)
	}
	if cfg.EnvVars["HUGGING_FACE_HUB_TOKEN"] != "hf_tt_test" {
		t.Error("HF token not set")
	}

	// Should have TT device passthrough
	hasTTDevice := false
	for _, f := range cfg.GPUFlags {
		if f == "/dev/tenstorrent" {
			hasTTDevice = true
		}
	}
	if !hasTTDevice {
		t.Error("missing /dev/tenstorrent in GPUFlags")
	}
}

func TestTenstorrentContainerConfigNoHardware(t *testing.T) {
	p := NewTenstorrentPlatform()
	model := ModelConfig{ID: "test/model", Name: "test", TP: 1}
	opts := RunOptions{Port: 8000}

	cfg := p.ContainerConfig(model, opts)
	if cfg == nil {
		t.Fatal("ContainerConfig returned nil")
	}
	// Should still produce valid config even without hw info
}

func TestTenstorrentHealthEndpoint(t *testing.T) {
	p := NewTenstorrentPlatform()
	if ep := p.HealthEndpoint(); ep != "/health" {
		t.Errorf("HealthEndpoint = %q, want /health", ep)
	}
}

func TestTTModelFromPCI(t *testing.T) {
	tests := []struct {
		line string
		want string
	}{
		{"3b:00.0 Processing accelerators [0b40]: Tenstorrent Inc. Device [1e52:1b01]", "Wormhole n300"},
		{"3b:00.0 Device [0b40]: Tenstorrent Inc. Device [1e52:1b02]", "Wormhole n150"},
		{"unknown device [1e52:9999]", "Unknown"},
		{"no match here", "Unknown"},
	}

	for _, tt := range tests {
		got := ttModelFromPCI(tt.line)
		if got != tt.want {
			t.Errorf("ttModelFromPCI(%q) = %q, want %q", tt.line, got, tt.want)
		}
	}
}

func TestTTParseTtSmiJSON(t *testing.T) {
	p := NewTenstorrentPlatform()

	jsonOutput := `[{"board_type":"n300","chip_type":"wormhole","firmware_version":"0.1.0","pcie_bus_id":"0000:3b:00.0","dram":{"total":12,"unit":"GB"},"board_id":"test-001"}]`

	hw, err := p.parseTtSmiJSON(jsonOutput)
	if err != nil {
		t.Fatalf("parseTtSmiJSON error: %v", err)
	}
	if hw.Platform != "tenstorrent" {
		t.Errorf("Platform = %q", hw.Platform)
	}
	if len(hw.Devices) != 1 {
		t.Fatalf("expected 1 device, got %d", len(hw.Devices))
	}
	dev := hw.Devices[0]
	if dev.VRAM_GB != 12 {
		t.Errorf("VRAM_GB = %d, want 12", dev.VRAM_GB)
	}
	if dev.PCIAddress != "0000:3b:00.0" {
		t.Errorf("PCIAddress = %q", dev.PCIAddress)
	}
	if dev.DriverVersion != "0.1.0" {
		t.Errorf("DriverVersion = %q", dev.DriverVersion)
	}
}

func TestTTParseTtSmiJSONMB(t *testing.T) {
	p := NewTenstorrentPlatform()

	jsonOutput := `[{"board_type":"n150","chip_type":"wormhole","dram":{"total":12288,"unit":"MB"}}]`

	hw, err := p.parseTtSmiJSON(jsonOutput)
	if err != nil {
		t.Fatalf("parseTtSmiJSON error: %v", err)
	}
	if hw.Devices[0].VRAM_GB != 12 {
		t.Errorf("VRAM_GB = %d, want 12 (from MB)", hw.Devices[0].VRAM_GB)
	}
}

func TestTTParseTtSmiJSONEmpty(t *testing.T) {
	p := NewTenstorrentPlatform()
	_, err := p.parseTtSmiJSON("[]")
	if err == nil {
		t.Error("expected error for empty JSON array")
	}
}

func TestTTParseTtSmiJSONInvalid(t *testing.T) {
	p := NewTenstorrentPlatform()
	_, err := p.parseTtSmiJSON("not json")
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestTTParseTtTopology(t *testing.T) {
	p := NewTenstorrentPlatform()

	output := `Board: WH N300 (wormhole), PCIe: 0000:3b:00.0, DRAM: 12 GB
Board: WH N150 (wormhole), PCIe: 0000:3c:00.0, DRAM: 12 GB`

	hw, err := p.parseTtTopology(output)
	if err != nil {
		t.Fatalf("parseTtTopology error: %v", err)
	}
	if len(hw.Devices) != 2 {
		t.Fatalf("expected 2 devices, got %d", len(hw.Devices))
	}
	if hw.Devices[0].VRAM_GB != 12 {
		t.Errorf("Device 0 VRAM = %d, want 12", hw.Devices[0].VRAM_GB)
	}
	if hw.Devices[0].PCIAddress != "0000:3b:00.0" {
		t.Errorf("Device 0 PCI = %q", hw.Devices[0].PCIAddress)
	}
	if hw.Devices[1].PCIAddress != "0000:3c:00.0" {
		t.Errorf("Device 1 PCI = %q", hw.Devices[1].PCIAddress)
	}
}

func TestTTDetectNoHardware(t *testing.T) {
	p := NewTenstorrentPlatform()
	_, err := p.DetectHardware(t.Context())
	if err == nil {
		t.Log("Tenstorrent hardware detected")
	} else {
		t.Logf("DetectHardware correctly returned error: %v", err)
	}
}

func TestTTParseTtSmiText(t *testing.T) {
	p := NewTenstorrentPlatform()

	output := `Board: WH N300
Chip: wormhole
Firmware: 0.1.0
DRAM: 12 GB
PCIe: 0000:3b:00.0`

	hw, err := p.parseTtSmiText(output)
	if err != nil {
		t.Fatalf("parseTtSmiText error: %v", err)
	}
	if len(hw.Devices) == 0 {
		t.Fatal("expected at least 1 device")
	}

	dev := hw.Devices[0]
	if dev.VRAM_GB != 12 {
		t.Errorf("VRAM_GB = %d, want 12", dev.VRAM_GB)
	}
	if dev.DriverVersion != "0.1.0" {
		t.Errorf("DriverVersion = %q", dev.DriverVersion)
	}
	if dev.PCIAddress != "0000:3b:00.0" {
		t.Errorf("PCIAddress = %q", dev.PCIAddress)
	}
}

func TestTTIsVLLMBackend(t *testing.T) {
	p := NewTenstorrentPlatform()

	tests := []struct {
		flags string
		want  bool
	}{
		{"--vllm-backend", true},
		{"--serving vllm", true},
		{"", false},
		{"--other-flag", false},
	}

	for _, tt := range tests {
		model := ModelConfig{ExtraFlags: tt.flags}
		got := p.isVLLMBackend(model)
		if got != tt.want {
			t.Errorf("isVLLMBackend(%q) = %v, want %v", tt.flags, got, tt.want)
		}
	}
}
