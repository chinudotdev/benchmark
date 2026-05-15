// Package platform defines the interface for hardware backend detection
// and serving orchestration. Each vendor (NVIDIA, AMD, Tenstorrent)
// implements this interface so the benchmark runner is hardware-agnostic.
package platform

import (
	"context"
	"fmt"
	"log"
	"strings"
)

// Platform is the contract each hardware backend must implement.
type Platform interface {
	// Name returns the platform identifier: "nvidia", "amd", "tenstorrent".
	Name() string

	// DetectHardware probes the system for accelerators belonging to this
	// platform. Returns ErrHardwareNotFound if no devices are present.
	DetectHardware(ctx context.Context) (*HardwareInfo, error)

	// DetectOrInstallRuntime checks for the container/runtime toolkit
	// (e.g. nvidia-container-toolkit, ROCm toolkit) and attempts to
	// install it if missing and the caller consents.
	DetectOrInstallRuntime(ctx context.Context) error

	// GetDockerImage returns the default container image for this platform.
	GetDockerImage(override string) string

	// ContainerConfig returns the Docker run configuration for a given model
	// and benchmark options. Returns nil if this platform doesn't use Docker.
	ContainerConfig(model ModelConfig, opts RunOptions) *ContainerConfig

	// HealthEndpoint returns the relative URL path to probe for readiness.
	HealthEndpoint() string
}

// DetectionResult holds the outcome of auto-detecting all platforms.
type DetectionResult struct {
	Platforms []DetectedPlatform `json:"platforms"`
}

// DetectedPlatform is a single platform that was found on the system.
type DetectedPlatform struct {
	Platform Platform      `json:"-"`
	Hardware *HardwareInfo `json:"hardware"`
}

// AutoDetect probes all known platforms and returns results for every
// platform that has hardware present. If name is non-empty, it selects
// that specific platform instead of probing.
//
// When multiple platforms are found and name is "", AutoDetect picks
// the first one but logs a message telling the user to use --platform
// to disambiguate.
func AutoDetect(ctx context.Context, name string) (Platform, *HardwareInfo, error) {
	if name != "" {
		return selectPlatform(ctx, name)
	}
	return autoDetectAll(ctx)
}

// selectPlatform returns the named platform without probing others.
func selectPlatform(ctx context.Context, name string) (Platform, *HardwareInfo, error) {
	var plat Platform
	switch strings.ToLower(name) {
	case "nvidia":
		plat = NewNVIDIAPlatform()
	case "amd":
		plat = NewAMDPlatform()
	case "tenstorrent":
		plat = NewTenstorrentPlatform()
	default:
		return nil, nil, fmt.Errorf("unknown platform: %s (use nvidia, amd, or tenstorrent)", name)
	}

	hw, err := plat.DetectHardware(ctx)
	if err != nil {
		return nil, nil, fmt.Errorf("%s hardware not detected: %w", plat.Name(), err)
	}
	log.Printf("Using %s platform (%d device(s))", plat.Name(), len(hw.Devices))
	return plat, hw, nil
}

// autoDetectAll probes all platforms and returns the best match.
// Priority: nvidia > amd > tenstorrent (most common first).
// If multiple platforms are found, logs a warning.
func autoDetectAll(ctx context.Context) (Platform, *HardwareInfo, error) {
	candidates := []struct {
		name string
		new  func() Platform
	}{
		{"nvidia", func() Platform { return NewNVIDIAPlatform() }},
		{"amd", func() Platform { return NewAMDPlatform() }},
		{"tenstorrent", func() Platform { return NewTenstorrentPlatform() }},
	}

	var detected []DetectedPlatform

	for _, c := range candidates {
		p := c.new()
		hw, err := p.DetectHardware(ctx)
		if err != nil {
			continue
		}
		detected = append(detected, DetectedPlatform{
			Platform: p,
			Hardware: hw,
		})
	}

	if len(detected) == 0 {
		return nil, nil, fmt.Errorf("no supported accelerators detected (tried NVIDIA, AMD, Tenstorrent)")
	}

	if len(detected) > 1 {
		names := make([]string, len(detected))
		for i, d := range detected {
			names[i] = d.Platform.Name()
		}
		log.Printf("⚠ Multiple platforms detected: %s — using %s. Use --platform to override.",
			strings.Join(names, ", "), detected[0].Platform.Name())
	}

	choice := detected[0]
	hw := choice.Hardware

	// Count total devices across all detected platforms for reporting
	totalDevices := 0
	for _, d := range detected {
		totalDevices += len(d.Hardware.Devices)
	}

	log.Printf("Auto-detected: %s platform (%d device(s))", choice.Platform.Name(), len(hw.Devices))

	// If there's only one platform (the common case), also enrich the
	// result by setting the hardware reference for the chosen platform.
	return choice.Platform, hw, nil
}

// ProbeAll probes every platform and returns all that were found.
// Used by sysinfo to display all available hardware.
func ProbeAll(ctx context.Context) []DetectedPlatform {
	candidates := []struct {
		name string
		new  func() Platform
	}{
		{"nvidia", func() Platform { return NewNVIDIAPlatform() }},
		{"amd", func() Platform { return NewAMDPlatform() }},
		{"tenstorrent", func() Platform { return NewTenstorrentPlatform() }},
	}

	var detected []DetectedPlatform
	for _, c := range candidates {
		p := c.new()
		hw, err := p.DetectHardware(ctx)
		if err != nil {
			continue
		}
		detected = append(detected, DetectedPlatform{
			Platform: p,
			Hardware: hw,
		})
	}
	return detected
}

// HardwareInfo describes detected accelerators.
type HardwareInfo struct {
	Platform string       `json:"platform"`
	Devices  []DeviceInfo `json:"devices"`
}

// DeviceInfo describes a single accelerator.
type DeviceInfo struct {
	Name           string `json:"name"`
	VRAM_GB        int    `json:"vram_gb"`
	DriverVersion  string `json:"driver_version"`
	RuntimeVersion string `json:"runtime_version"` // CUDA, ROCm, or TT-Metalium version
	GFXArch        string `json:"gfx_arch,omitempty"` // AMD-specific: gfx942, etc.
	PCIAddress     string `json:"pci_address,omitempty"`
}

// ContainerConfig holds everything needed to launch a serving container.
type ContainerConfig struct {
	Image     string            `json:"image"`
	Name      string            `json:"name"`
	GPUFlags  []string          `json:"gpu_flags"`  // e.g. ["--gpus", "all"] or ["--device", "/dev/kfd"]
	EnvVars   map[string]string `json:"env_vars"`
	ExtraArgs []string          `json:"extra_args"` // additional args to the container entrypoint
	Port      int               `json:"port"`
}

// ModelConfig describes a model to benchmark.
type ModelConfig struct {
	ID              string `yaml:"id"               json:"id"`
	Name            string `yaml:"name"             json:"name"`
	Quant           string `yaml:"quant"            json:"quant"`
	MinVRAM_GB      int    `yaml:"min_vram_gb"      json:"min_vram_gb"`
	TP              int    `yaml:"tp"              json:"tp"`
	ExtraFlags      string `yaml:"extra_flags"      json:"extra_flags"`
	Platform        string `yaml:"platform"         json:"platform,omitempty"`         // restrict to platform
	DockerImage     string `yaml:"docker_image"     json:"docker_image,omitempty"`     // per-model image override
	ServingBackend  string `yaml:"serving_backend"  json:"serving_backend,omitempty"` // vllm, tt-inference-server, tgi
}

// RunOptions holds CLI options that influence how a model is run.
type RunOptions struct {
	GPURate      float64
	GPUCount     int
	MaxModelLen  int
	Port         int
	Quant        string
	DockerImage  string
	GPUIDs       string
	HFToken      string
	Stream       bool
	Force        bool
	DryRun       bool
}

// DryRunPlatform is a synthetic platform used when --dry-run is specified
// and no real hardware is available. It produces valid container configs
// for preview purposes.
type DryRunPlatform struct {
	name string
}

// NewDryRunPlatform returns a synthetic platform for dry-run mode.
func NewDryRunPlatform(name string) *DryRunPlatform {
	return &DryRunPlatform{name: name}
}

func (p *DryRunPlatform) Name() string { return p.name }

func (p *DryRunPlatform) DetectHardware(_ context.Context) (*HardwareInfo, error) {
	return &HardwareInfo{
		Platform: p.name,
		Devices:  []DeviceInfo{{Name: "Dry-Run GPU", VRAM_GB: 80}},
	}, nil
}

func (p *DryRunPlatform) DetectOrInstallRuntime(_ context.Context) error { return nil }

func (p *DryRunPlatform) GetDockerImage(override string) string {
	if override != "" {
		return override
	}
	switch p.name {
	case "amd":
		return "rocm/vllm:latest"
	case "tenstorrent":
		return "ghcr.io/tenstorrent/tt-inference-server:latest"
	default:
		return "vllm/vllm-openai:latest"
	}
}

func (p *DryRunPlatform) ContainerConfig(model ModelConfig, opts RunOptions) *ContainerConfig {
	img := p.GetDockerImage(opts.DockerImage)

	var gpuFlags []string
	switch p.name {
	case "amd":
		gpuFlags = []string{"--device", "/dev/kfd", "--device", "/dev/dri", "--group-add", "video"}
	case "tenstorrent":
		gpuFlags = []string{"--device", "/dev/tenstorrent"}
	default:
		gpuFlags = []string{"--gpus", "all"}
	}

	return &ContainerConfig{
		Image:     img,
		Name:      fmt.Sprintf("vllm_bench_%d", opts.Port),
		GPUFlags:  gpuFlags,
		ExtraArgs: []string{model.ID, "--port", "8000"},
		Port:      opts.Port,
	}
}

func (p *DryRunPlatform) HealthEndpoint() string { return "/health" }
