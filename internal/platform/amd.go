package platform

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/chinudotdev/gpu-benchmark/internal/sysinfo"
)

// AMDPlatform implements Platform for AMD GPUs (ROCm).
// Stub for Milestone 4 — will be fleshed out with real hardware access.
type AMDPlatform struct {
	hw *HardwareInfo
}

func NewAMDPlatform() *AMDPlatform {
	return &AMDPlatform{}
}

func (p *AMDPlatform) Name() string { return "amd" }

func (p *AMDPlatform) DetectHardware(ctx context.Context) (*HardwareInfo, error) {
	// Check rocm-smi is available
	if _, err := sysinfo.LookPath("rocm-smi"); err != nil {
		return nil, fmt.Errorf("rocm-smi not found: %w (install ROCm)", err)
	}

	// Query GPU name, VRAM
	output, err := sysinfo.Exec(ctx, "rocm-smi",
		"--showproductname",
		"--csv")
	if err != nil {
		return nil, fmt.Errorf("rocm-smi query failed: %w", err)
	}

	var devices []DeviceInfo
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(strings.ToLower(line), "device") {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) < 2 {
			continue
		}
		name := strings.TrimSpace(parts[0])
		vramStr := ""
		if len(parts) > 1 {
			vramStr = strings.TrimSpace(parts[1])
		}
		vramGB := 0
		if v, err := strconv.Atoi(strings.TrimSuffix(vramStr, "M")); err == nil {
			vramGB = v / 1024
		}
		devices = append(devices, DeviceInfo{
			Name:    name,
			VRAM_GB: vramGB,
		})
	}

	if len(devices) == 0 {
		return nil, fmt.Errorf("no AMD GPUs detected by rocm-smi")
	}

	// Get ROCm version
	rocmVer := ""
	if ver, err := sysinfo.Exec(ctx, "rocm-smi", "--version"); err == nil {
		// Parse version from output like "rocm-smi version: 6.1.0"
		for _, line := range strings.Split(ver, "\n") {
			if strings.Contains(strings.ToLower(line), "version") {
				parts := strings.Fields(line)
				if len(parts) > 0 {
					rocmVer = parts[len(parts)-1]
				}
				break
			}
		}
	}
	for i := range devices {
		devices[i].RuntimeVersion = rocmVer
	}

	p.hw = &HardwareInfo{
		Platform: "amd",
		Devices:  devices,
	}
	return p.hw, nil
}

func (p *AMDPlatform) DetectOrInstallRuntime(ctx context.Context) error {
	// Check ROCm container toolkit / amdgpu driver
	if _, err := sysinfo.Exec(ctx, "rocm-smi", "--showid"); err != nil {
		return fmt.Errorf("ROCm runtime not available: %w", err)
	}
	return nil
}

func (p *AMDPlatform) GetDockerImage(override string) string {
	if override != "" {
		return override
	}
	return "rocm/vllm:latest"
}

func (p *AMDPlatform) ContainerConfig(model ModelConfig, opts RunOptions) *ContainerConfig {
	image := p.GetDockerImage(opts.DockerImage)
	containerName := fmt.Sprintf("vllm_bench_%d", opts.Port)

	extraArgs := []string{
		model.ID,
		"--trust-remote-code",
		"--tensor-parallel-size", strconv.Itoa(model.TP),
		"--max-model-len", strconv.Itoa(opts.MaxModelLen),
		"--gpu-memory-utilization", "0.90",
		"--port", "8000",
	}
	extraArgs = append(extraArgs, quantFlags(model.Quant)...)
	if model.ExtraFlags != "" {
		extraArgs = append(extraArgs, strings.Fields(model.ExtraFlags)...)
	}

	envVars := map[string]string{
		"HSA_OVERRIDE_GFX_VERSION": "9.4.2", // Common default for MI300X
	}
	if opts.HFToken != "" {
		envVars["HUGGING_FACE_HUB_TOKEN"] = opts.HFToken
	}

	// AMD uses --device flags instead of --gpus
	gpuFlags := []string{
		"--device", "/dev/kfd",
		"--device", "/dev/dri",
		"--group-add", "video",
	}
	if opts.GPUIDs != "" {
		// If specific GPUs are requested, set HIP_VISIBLE_DEVICES
		envVars["HIP_VISIBLE_DEVICES"] = opts.GPUIDs
	}

	return &ContainerConfig{
		Image:     image,
		Name:      containerName,
		GPUFlags:  gpuFlags,
		EnvVars:   envVars,
		ExtraArgs: extraArgs,
		Port:      opts.Port,
	}
}

func (p *AMDPlatform) HealthEndpoint() string {
	return "/health"
}
