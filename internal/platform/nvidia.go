package platform

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/chinudotdev/gpu-benchmark/internal/sysinfo"
)

// NVIDIAPlatform implements Platform for NVIDIA GPUs.
type NVIDIAPlatform struct {
	hw   *HardwareInfo
	sudo bool
}

func NewNVIDIAPlatform() *NVIDIAPlatform {
	return &NVIDIAPlatform{}
}

func (p *NVIDIAPlatform) Name() string { return "nvidia" }

func (p *NVIDIAPlatform) DetectHardware(ctx context.Context) (*HardwareInfo, error) {
	// Check nvidia-smi is available
	if _, err := sysinfo.LookPath("nvidia-smi"); err != nil {
		return nil, fmt.Errorf("nvidia-smi not found: %w (install NVIDIA drivers)", err)
	}

	// Query GPU name, memory, driver version
	output, err := sysinfo.Exec(ctx, "nvidia-smi",
		"--query-gpu=name,memory.total,driver_version,pci.bus_id",
		"--format=csv,noheader")
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi query failed: %w", err)
	}

	var devices []DeviceInfo
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) < 3 {
			continue
		}
		name := strings.TrimSpace(parts[0])
		vramStr := strings.TrimSpace(parts[1])
		driverVer := strings.TrimSpace(parts[2])
		pciAddr := ""
		if len(parts) > 3 {
			pciAddr = strings.TrimSpace(parts[3])
		}

		vramGB := parseVRAM(vramStr)

		devices = append(devices, DeviceInfo{
			Name:          name,
			VRAM_GB:       vramGB,
			DriverVersion: driverVer,
			PCIAddress:    pciAddr,
		})
	}

	if len(devices) == 0 {
		return nil, fmt.Errorf("no NVIDIA GPUs detected by nvidia-smi")
	}

	// Get CUDA version from nvidia-smi header
	cudaVer := ""
	if header, err := sysinfo.Exec(ctx, "nvidia-smi"); err == nil {
		re := regexp.MustCompile(`CUDA Version:\s*([\d.]+)`)
		if m := re.FindStringSubmatch(header); len(m) > 1 {
			cudaVer = m[1]
		}
	}
	for i := range devices {
		devices[i].RuntimeVersion = cudaVer
	}

	p.hw = &HardwareInfo{
		Platform: "nvidia",
		Devices:  devices,
	}
	return p.hw, nil
}

func (p *NVIDIAPlatform) DetectOrInstallRuntime(ctx context.Context) error {
	// Check nvidia-container-toolkit
	if _, err := sysinfo.Exec(ctx, "dpkg", "-s", "nvidia-container-toolkit"); err != nil {
		return fmt.Errorf("nvidia-container-toolkit is not installed — Docker cannot access GPUs.\n" +
			"Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide")
	}
	return nil
}

func (p *NVIDIAPlatform) GetDockerImage(override string) string {
	if override != "" {
		return override
	}
	return "vllm/vllm-openai:latest"
}

func (p *NVIDIAPlatform) ContainerConfig(model ModelConfig, opts RunOptions) *ContainerConfig {
	gpuFlag := opts.GPUIDs
	if gpuFlag == "" {
		gpuFlag = "all"
	}

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

	// Add quantization flags
	extraArgs = append(extraArgs, quantFlags(model.Quant)...)

	// Add model-specific extra flags
	if model.ExtraFlags != "" {
		extraArgs = append(extraArgs, strings.Fields(model.ExtraFlags)...)
	}

	envVars := map[string]string{}
	if opts.HFToken != "" {
		envVars["HUGGING_FACE_HUB_TOKEN"] = opts.HFToken
	}

	return &ContainerConfig{
		Image:     image,
		Name:      containerName,
		GPUFlags:  []string{"--gpus", gpuFlag},
		EnvVars:   envVars,
		ExtraArgs: extraArgs,
		Port:      opts.Port,
	}
}

func (p *NVIDIAPlatform) HealthEndpoint() string {
	return "/health"
}

// parseVRAM parses VRAM strings like "81920 MiB" into GB, rounded to
// the nearest integer to avoid truncation (e.g. 48589 MiB → 47 GB, 48576 MiB → 48 GB).
func parseVRAM(s string) int {
	re := regexp.MustCompile(`(\d+)\s*MiB`)
	m := re.FindStringSubmatch(s)
	if len(m) > 1 {
		mib, err := strconv.Atoi(m[1])
		if err == nil {
			return int(float64(mib)/1024 + 0.5) // round to nearest GB
		}
	}
	return 0
}

// quantFlags returns vLLM quantization CLI flags.
func quantFlags(quant string) []string {
	switch quant {
	case "int8":
		return []string{"--quantization", "bitsandbytes"}
	case "int4":
		return []string{"--quantization", "bitsandbytes", "--load-in-4bit"}
	case "awq":
		return []string{"--quantization", "awq"}
	case "fp8":
		return []string{"--quantization", "fp8"}
	default:
		return nil
	}
}
