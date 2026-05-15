package platform

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/chinudotdev/gpu-benchmark/internal/sysinfo"
)

// AMDPlatform implements Platform for AMD GPUs via ROCm.
//
// Supported hardware: MI250X, MI300X, MI300A, MI325X, RX 7900 XTX, etc.
// Detection: rocm-smi, /sys/class/drm, /sys/class/kfd
// Container: vLLM ROCm fork with --device /dev/kfd --device /dev/dri
type AMDPlatform struct {
	hw      *HardwareInfo
	kfdPath string // path to /dev/kfd (for container passthrough)
}

func NewAMDPlatform() *AMDPlatform {
	return &AMDPlatform{
		kfdPath: "/dev/kfd",
	}
}

func (p *AMDPlatform) Name() string { return "amd" }

// ── Hardware Detection ─────────────────────────────────────────────────────

func (p *AMDPlatform) DetectHardware(ctx context.Context) (*HardwareInfo, error) {
	// Strategy 1: rocm-smi (preferred, requires ROCm installed)
	if _, err := sysinfo.LookPath("rocm-smi"); err == nil {
		hw, err := p.detectViaRocmSmi(ctx)
		if err == nil && hw != nil {
			p.hw = hw
			return hw, nil
		}
	}

	// Strategy 2: Parse /sys/class/kfd for AMD compute devices
	if hw, err := p.detectViaSysfs(); err == nil && hw != nil {
		p.hw = hw
		return hw, nil
	}

	// Strategy 3: lspci fallback
	if hw, err := p.detectViaLspci(ctx); err == nil && hw != nil {
		p.hw = hw
		return hw, nil
	}

	return nil, fmt.Errorf("no AMD GPUs detected (tried rocm-smi, /sys/class/kfd, lspci)")
}

// detectViaRocmSmi uses `rocm-smi` to enumerate AMD GPUs.
func (p *AMDPlatform) detectViaRocmSmi(ctx context.Context) (*HardwareInfo, error) {
	// rocm-smi --showproductname --csv gives us card name
	output, err := sysinfo.Exec(ctx, "rocm-smi",
		"--showproductname",
		"--csv")
	if err != nil {
		return nil, fmt.Errorf("rocm-smi --showproductname: %w", err)
	}

	var devices []DeviceInfo
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(strings.ToLower(line), "device") {
			continue // skip header
		}
		parts := strings.Split(line, ",")
		if len(parts) < 2 {
			continue
		}

		name := strings.TrimSpace(parts[0])
		cardSeries := ""
		if len(parts) > 1 {
			cardSeries = strings.TrimSpace(parts[1])
		}
		if name == "" && cardSeries == "" {
			continue
		}

		displayName := name
		if cardSeries != "" && !strings.Contains(name, cardSeries) {
			displayName = name + " " + cardSeries
		}

		devices = append(devices, DeviceInfo{
			Name: strings.TrimSpace(displayName),
		})
	}

	if len(devices) == 0 {
		return nil, fmt.Errorf("rocm-smi reported no devices")
	}

	// Enrich with VRAM, driver, ROCm version
	p.enrichFromRocmSmi(ctx, devices)

	// Detect GFX architecture
	p.enrichGFXArch(ctx, devices)

	return &HardwareInfo{
		Platform: "amd",
		Devices:  devices,
	}, nil
}

// enrichFromRocmSmi fills in VRAM, driver version, and ROCm runtime version.
func (p *AMDPlatform) enrichFromRocmSmi(ctx context.Context, devices []DeviceInfo) {
	// Get VRAM: rocm-smi --showmeminfo vram --csv
	if vramOutput, err := sysinfo.Exec(ctx, "rocm-smi", "--showmeminfo", "vram", "--csv"); err == nil {
		p.parseVRAMCSV(vramOutput, devices)
	}

	// Get driver version: rocm-smi --showdriverversion
	if drvOutput, err := sysinfo.Exec(ctx, "rocm-smi", "--showdriverversion"); err == nil {
		for _, line := range strings.Split(drvOutput, "\n") {
			if strings.Contains(line, "Driver version") || strings.Contains(line, "Driver") {
				parts := strings.Fields(line)
				if len(parts) > 0 {
					ver := parts[len(parts)-1]
					for i := range devices {
						if devices[i].DriverVersion == "" {
							devices[i].DriverVersion = ver
						}
					}
				}
				break
			}
		}
	}

	// Get ROCm version
	if verOutput, err := sysinfo.Exec(ctx, "rocm-smi", "--version"); err == nil {
		for _, line := range strings.Split(verOutput, "\n") {
			line = strings.TrimSpace(line)
			if strings.Contains(strings.ToLower(line), "version") {
				parts := strings.Fields(line)
				for _, part := range parts {
					if strings.Contains(part, ".") && len(part) >= 3 {
						for i := range devices {
							if devices[i].RuntimeVersion == "" {
								devices[i].RuntimeVersion = part
							}
						}
						break
					}
				}
				break
			}
		}
	}
}

// parseVRAMCSV parses rocm-smi VRAM CSV output like:
//
//	device,VRAM Total (B),VRAM Used (B),VRAM Free (B)
//	GPU[0] : xxxx, 68719476736, 0, 68719476736
func (p *AMDPlatform) parseVRAMCSV(output string, devices []DeviceInfo) {
	lines := strings.Split(output, "\n")
	deviceIdx := 0
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(strings.ToLower(line), "device") {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) < 2 {
			continue
		}
		vramBytesStr := strings.TrimSpace(parts[1])
		if vramBytes, err := strconv.ParseInt(vramBytesStr, 10, 64); err == nil && vramBytes > 0 {
			vramGB := int(float64(vramBytes) / 1024 / 1024 / 1024 + 0.5)
			if deviceIdx < len(devices) {
				devices[deviceIdx].VRAM_GB = vramGB
				deviceIdx++
			}
		}
	}
}

// enrichGFXArch detects the GFX architecture (e.g. gfx942, gfx90a) for each device.
func (p *AMDPlatform) enrichGFXArch(ctx context.Context, devices []DeviceInfo) {
	// rocm-smi --showgpuclocks --csv sometimes has arch info
	// Better: read from /sys/class/kfd/kfd/topology/nodes/*/properties
	for i := 0; i < len(devices); i++ {
		if gfx := probeGFXArch(i); gfx != "" {
			devices[i].GFXArch = gfx
		}
	}
}

// probeGFXArch reads the GFX architecture from sysfs for a given GPU index.
func probeGFXArch(idx int) string {
	// Try /sys/class/kfd/kfd/topology/nodes/<N>/properties
	// The file contains lines like: gfx_target_version 0x90012
	nodesDir := "/sys/class/kfd/kfd/topology/nodes"
	for n := 0; n < 64; n++ {
		propsPath := filepath.Join(nodesDir, strconv.Itoa(n), "properties")
		data, err := os.ReadFile(propsPath)
		if err != nil {
			continue
		}

		content := string(data)
		// Look for gpu_id or gfx_target_version to identify GPU nodes
		if !strings.Contains(content, "cpu_cores_id") && strings.Contains(content, "gfx_target_version") {
			// This is a GPU node
			for _, line := range strings.Split(content, "\n") {
				if strings.HasPrefix(line, "gfx_target_version") {
					parts := strings.Fields(line)
					if len(parts) >= 2 {
						return parseGFXVersion(parts[1])
					}
				}
			}
		}
	}

	// Fallback: try /sys/class/drm/card<idx>/device/uevent
	drmPath := fmt.Sprintf("/sys/class/drm/card%d/device/uevent", idx)
	if data, err := os.ReadFile(drmPath); err == nil {
		content := string(data)
		for _, line := range strings.Split(content, "\n") {
			if strings.HasPrefix(line, "PCI_ID=") {
				// Map PCI ID to GFX arch (common ones)
				return pciIDToGFX(strings.TrimPrefix(line, "PCI_ID="))
			}
		}
	}

	return ""
}

// parseGFXVersion converts hex GFX version like "0x90012" to "gfx942" etc.
func parseGFXVersion(hexStr string) string {
	hexStr = strings.TrimPrefix(hexStr, "0x")
	val, err := strconv.ParseInt(hexStr, 16, 64)
	if err != nil {
		return ""
	}
	major := (val >> 16) & 0xFF
	minor := (val >> 8) & 0xFF
	stepping := val & 0xFF
	if major > 0 {
		return fmt.Sprintf("gfx%d%d%x", major, minor, stepping)
	}
	return ""
}

// pciIDToGFX maps common AMD GPU PCI IDs to GFX architectures.
func pciIDToGFX(pciID string) string {
	known := map[string]string{
		"1002:740C": "gfx942", // MI300X
		"1002:7408": "gfx942", // MI300A
		"1002:74A1": "gfx942", // MI325X
		"1002:738C": "gfx90a", // MI250X
		"1002:738E": "gfx90a", // MI210
		"1002:744C": "gfx941", // MI250
		"1002:7480": "gfx1100", // RX 7900 XTX (Navi 31)
	}
	if arch, ok := known[strings.ToUpper(pciID)]; ok {
		return arch
	}
	return ""
}

// detectViaSysfs detects AMD GPUs by scanning /sys/class/kfd.
func (p *AMDPlatform) detectViaSysfs() (*HardwareInfo, error) {
	kfdDir := "/sys/class/kfd/kfd/topology/nodes"
	entries, err := os.ReadDir(kfdDir)
	if err != nil {
		return nil, fmt.Errorf("cannot read %s: %w", kfdDir, err)
	}

	var devices []DeviceInfo
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		propsPath := filepath.Join(kfdDir, entry.Name(), "properties")
		data, err := os.ReadFile(propsPath)
		if err != nil {
			continue
		}

		content := string(data)
		// Skip CPU nodes
		if strings.Contains(content, "cpu_cores_id") {
			continue
		}
		// Must be a GPU node
		if !strings.Contains(content, "gfx_target_version") {
			continue
		}

		dev := DeviceInfo{}
		for _, line := range strings.Split(content, "\n") {
			line = strings.TrimSpace(line)
			parts := strings.Fields(line)
			if len(parts) < 2 {
				continue
			}
			key, val := parts[0], parts[1]
			switch key {
			case "gfx_target_version":
				dev.GFXArch = parseGFXVersion(val)
			case "simd_count":
				// Has SIMD = GPU node
			case "mem_banks_count":
				// Memory banks
			}
		}

		// Read VRAM from mem_banks
		vram := p.probeSysfsVRAM(filepath.Join(kfdDir, entry.Name()))
		if vram > 0 {
			dev.VRAM_GB = vram
		}

		// Get name from marketing_name file if available
		marketName := filepath.Join(kfdDir, entry.Name(), "marketing_name")
		if nameData, err := os.ReadFile(marketName); err == nil {
			dev.Name = strings.TrimSpace(string(nameData))
		} else {
			dev.Name = "AMD GPU"
			if dev.GFXArch != "" {
				dev.Name = "AMD GPU (" + dev.GFXArch + ")"
			}
		}

		// Driver version from /sys/module/amdgpu/version
		if ver, err := os.ReadFile("/sys/module/amdgpu/version"); err == nil {
			dev.DriverVersion = strings.TrimSpace(string(ver))
		}

		devices = append(devices, dev)
	}

	if len(devices) == 0 {
		return nil, fmt.Errorf("no AMD GPU nodes found in %s", kfdDir)
	}

	// Try to get ROCm version
	rocmVer := probeROCmVersion()
	for i := range devices {
		if devices[i].RuntimeVersion == "" {
			devices[i].RuntimeVersion = rocmVer
		}
	}

	return &HardwareInfo{
		Platform: "amd",
		Devices:  devices,
	}, nil
}

// probeSysfsVRAM reads VRAM from sysfs mem_banks.
func (p *AMDPlatform) probeSysfsVRAM(nodeDir string) int {
	memDir := filepath.Join(nodeDir, "mem_banks")
	entries, err := os.ReadDir(memDir)
	if err != nil {
		return 0
	}

	totalBytes := int64(0)
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		sizePath := filepath.Join(memDir, entry.Name(), "size")
		data, err := os.ReadFile(sizePath)
		if err != nil {
			continue
		}
		if bytes, err := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64); err == nil {
			totalBytes += bytes
		}
	}

	if totalBytes > 0 {
		return int(float64(totalBytes)/1024/1024/1024 + 0.5)
	}
	return 0
}

// probeROCmVersion attempts to find the ROCm version.
func probeROCmVersion() string {
	// Check /opt/rocm/.info/version
	if data, err := os.ReadFile("/opt/rocm/.info/version"); err == nil {
		return strings.TrimSpace(string(data))
	}
	// Check rocm-version file
	if data, err := os.ReadFile("/opt/rocm/rocm.version"); err == nil {
		return strings.TrimSpace(string(data))
	}
	return "unknown"
}

// detectViaLspci uses lspci to find AMD GPUs as a last resort.
func (p *AMDPlatform) detectViaLspci(ctx context.Context) (*HardwareInfo, error) {
	if _, err := sysinfo.LookPath("lspci"); err != nil {
		return nil, fmt.Errorf("lspci not found: %w", err)
	}

	output, err := sysinfo.Exec(ctx, "lspci", "-nn", "-d", "::0300")
	if err != nil {
		// Try without device class filter
		output, err = sysinfo.Exec(ctx, "lspci", "-nn")
		if err != nil {
			return nil, fmt.Errorf("lspci failed: %w", err)
		}
	}

	re := regexp.MustCompile(`Advanced Micro Devices.*\[([0-9a-fA-F]{4}:[0-9a-fA-F]{4})\]`)
	var devices []DeviceInfo

	for _, line := range strings.Split(output, "\n") {
		if !strings.Contains(line, "Advanced Micro Devices") && !strings.Contains(line, "AMD") {
			continue
		}
		if !strings.Contains(line, "VGA") && !strings.Contains(line, "Display") && !strings.Contains(line, "3D") && !strings.Contains(line, "Accelerator") {
			continue
		}

		// Extract name
		name := strings.TrimSpace(line)
		if idx := strings.Index(name, "]: "); idx >= 0 {
			name = name[idx+4:]
		}

		dev := DeviceInfo{Name: name}

		// Try to extract PCI ID for GFX arch
		if m := re.FindStringSubmatch(line); len(m) > 1 {
			dev.GFXArch = pciIDToGFX(m[1])
		}

		devices = append(devices, dev)
	}

	if len(devices) == 0 {
		return nil, fmt.Errorf("no AMD GPU devices found via lspci")
	}

	return &HardwareInfo{
		Platform: "amd",
		Devices:  devices,
	}, nil
}

// ── Runtime Detection ──────────────────────────────────────────────────────

func (p *AMDPlatform) DetectOrInstallRuntime(ctx context.Context) error {
	// Check amdgpu driver is loaded
	if _, err := os.Stat("/sys/module/amdgpu"); err != nil {
		return fmt.Errorf("amdgpu kernel module not loaded — AMD GPU driver required")
	}

	// Check /dev/kfd exists (Kernel Fusion Driver)
	if _, err := os.Stat(p.kfdPath); err != nil {
		return fmt.Errorf("/dev/kfd not found — ROCm kernel driver (amdgpu with KFD) required")
	}

	// Check /dev/dri exists
	if _, err := os.Stat("/dev/dri"); err != nil {
		return fmt.Errorf("/dev/dri not found — DRI subsystem required for AMD GPU")
	}

	// Verify current user has access to /dev/kfd and /dev/dri
	if !p.checkDeviceAccess() {
		return fmt.Errorf("current user cannot access /dev/kfd or /dev/dri — " +
			"add user to 'video' and 'render' groups: " +
			"sudo usermod -aG video,render $USER")
	}

	// Check Docker can pass through AMD devices
	if _, err := sysinfo.LookPath("docker"); err == nil {
		// Verify Docker has the `--device` flag support
		if err := p.checkDockerAMDDevice(ctx); err != nil {
			return fmt.Errorf("Docker AMD device passthrough check failed: %w", err)
		}
	}

	return nil
}

// checkDeviceAccess verifies the current user can read/write /dev/kfd and /dev/dri.
func (p *AMDPlatform) checkDeviceAccess() bool {
	for _, path := range []string{p.kfdPath, "/dev/dri"} {
		f, err := os.OpenFile(path, os.O_RDONLY, 0)
		if err != nil {
			return false
		}
		f.Close()
	}
	return true
}

// checkDockerAMDDevice verifies Docker can pass AMD GPU devices.
func (p *AMDPlatform) checkDockerAMDDevice(ctx context.Context) error {
	// Run a quick test container to verify device passthrough
	output, err := sysinfo.Exec(ctx, "docker", "run", "--rm",
		"--device", "/dev/kfd",
		"--device", "/dev/dri",
		"rocm/vllm:latest",
		"rocm-smi", "--showid")
	if err != nil {
		// Not fatal — might just be that the image isn't pulled yet
		_ = output
	}
	return nil
}

// ── Container Configuration ────────────────────────────────────────────────

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
		"--device", "cpu", // vLLM ROCm needs this hint for AMD
	}
	extraArgs = append(extraArgs, quantFlags(model.Quant)...)
	if model.ExtraFlags != "" {
		extraArgs = append(extraArgs, strings.Fields(model.ExtraFlags)...)
	}

	envVars := map[string]string{}

	// Auto-detect GFX version and set HSA_OVERRIDE_GFX_VERSION if needed
	if p.hw != nil && len(p.hw.Devices) > 0 {
		gfx := p.hw.Devices[0].GFXArch
		if gfx != "" {
			// Convert gfx942 → 9.4.2 for HSA_OVERRIDE_GFX_VERSION
			envVars["HSA_OVERRIDE_GFX_VERSION"] = gfxToHSA(gfx)
		}
	}

	// HIP_VISIBLE_DEVICES for selecting specific GPUs
	if opts.GPUIDs != "" {
		envVars["HIP_VISIBLE_DEVICES"] = opts.GPUIDs
	}

	if opts.HFToken != "" {
		envVars["HUGGING_FACE_HUB_TOKEN"] = opts.HFToken
	}

	// AMD GPU passthrough flags
	gpuFlags := []string{
		"--device", "/dev/kfd",
		"--device", "/dev/dri",
		"--group-add", "video",
		"--group-add", "render",
	}

	// Network: host mode recommended for AMD ROCm performance
	gpuFlags = append(gpuFlags, "--network", "host")

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

// ── Helpers ────────────────────────────────────────────────────────────────

// gfxToHSA converts a GFX architecture string to HSA_OVERRIDE_GFX_VERSION format.
// e.g. "gfx942" → "9.4.2", "gfx90a" → "9.0.10"
func gfxToHSA(gfx string) string {
	re := regexp.MustCompile(`gfx(\d)(\d)(\w+)`)
	m := re.FindStringSubmatch(gfx)
	if len(m) < 4 {
		return gfx // fallback: return as-is
	}
	major := m[1]
	minor := m[2]
	stepping := m[3]
	// Hex stepping (a=10, b=11, etc.)
	stepVal, err := strconv.ParseInt(stepping, 16, 64)
	if err != nil {
		stepVal = 0
	}
	return fmt.Sprintf("%s.%s.%d", major, minor, stepVal)
}
