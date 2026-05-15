// Package platform defines the interface for hardware backend detection
// and serving orchestration. Each vendor (NVIDIA, AMD, Tenstorrent)
// implements this interface so the benchmark runner is hardware-agnostic.
package platform

import "context"

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
