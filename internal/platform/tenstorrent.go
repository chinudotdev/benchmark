package platform

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/chinudotdev/gpu-benchmark/internal/sysinfo"
)

// TenstorrentPlatform implements Platform for Tenstorrent Wormhole accelerators.
//
// Supported hardware: Wormhole n150, Wormhole n300, Grayskull (legacy)
// Detection: tt-smi, /sys/class/tenstorrent, tt-topology, lspci
// Serving: tt-inference-server or vLLM with TT backend
type TenstorrentPlatform struct {
	hw          *HardwareInfo
	ttSmiPath   string
	ttTopoPath  string
	useDocker   bool // whether to use Docker or native tt-inference-server
}

func NewTenstorrentPlatform() *TenstorrentPlatform {
	return &TenstorrentPlatform{}
}

func (p *TenstorrentPlatform) Name() string { return "tenstorrent" }

// ── Hardware Detection ─────────────────────────────────────────────────────

func (p *TenstorrentPlatform) DetectHardware(ctx context.Context) (*HardwareInfo, error) {
	// Strategy 1: tt-smi (official Tenstorrent management tool)
	if path, err := sysinfo.LookPath("tt-smi"); err == nil {
		p.ttSmiPath = path
		hw, err := p.detectViaTtSmi(ctx)
		if err == nil && hw != nil {
			p.hw = hw
			return hw, nil
		}
	}

	// Strategy 2: tt-topology (device enumeration)
	if path, err := sysinfo.LookPath("tt-topology"); err == nil {
		p.ttTopoPath = path
		hw, err := p.detectViaTtTopology(ctx)
		if err == nil && hw != nil {
			p.hw = hw
			return hw, nil
		}
	}

	// Strategy 3: /sys/class/tenstorrent
	if hw, err := p.detectViaSysfs(); err == nil && hw != nil {
		p.hw = hw
		return hw, nil
	}

	// Strategy 4: lspci fallback (PCI device ID based)
	if hw, err := p.detectViaLspci(ctx); err == nil && hw != nil {
		p.hw = hw
		return hw, nil
	}

	return nil, fmt.Errorf("no Tenstorrent devices detected (tried tt-smi, tt-topology, /sys/class/tenstorrent, lspci)")
}

// detectViaTtSmi uses `tt-smi` to enumerate Tenstorrent devices.
//
// Output format varies by version. We try JSON first, then fall back to text parsing.
func (p *TenstorrentPlatform) detectViaTtSmi(ctx context.Context) (*HardwareInfo, error) {
	// Try JSON output first (tt-smi >= 1.3)
	if output, err := sysinfo.Exec(ctx, "tt-smi", "-j"); err == nil {
		hw, err := p.parseTtSmiJSON(output)
		if err == nil && hw != nil {
			return hw, nil
		}
	}

	// Fall back to text parsing
	output, err := sysinfo.Exec(ctx, "tt-smi", "-s")
	if err != nil {
		output, err = sysinfo.Exec(ctx, "tt-smi")
		if err != nil {
			return nil, fmt.Errorf("tt-smi failed: %w", err)
		}
	}

	return p.parseTtSmiText(output)
}

// parseTtSmiJSON parses JSON output from tt-smi -j.
//
// Expected format:
//
//	[
//	  {
//	    "board_type": "n300",
//	    "chip_type": "wormhole",
//	    "firmware_version": "0.1.0",
//	    "pcie_bus_id": "0000:3b:00.0",
//	    "dram": {"total": 12, "unit": "GB"},
//	    "board_id": "..."
//	  }
//	]
func (p *TenstorrentPlatform) parseTtSmiJSON(output string) (*HardwareInfo, error) {
	var devices []struct {
		BoardType      string `json:"board_type"`
		ChipType       string `json:"chip_type"`
		FirmwareVersion string `json:"firmware_version"`
		PCIeBusID      string `json:"pcie_bus_id"`
		DRAM           struct {
			Total int    `json:"total"`
			Unit  string `json:"unit"`
		} `json:"dram"`
		BoardID string `json:"board_id"`
	}

	if err := json.Unmarshal([]byte(output), &devices); err != nil {
		return nil, fmt.Errorf("parse tt-smi JSON: %w", err)
	}

	if len(devices) == 0 {
		return nil, fmt.Errorf("tt-smi JSON reported no devices")
	}

	var hwDevices []DeviceInfo
	for _, d := range devices {
		name := d.BoardType
		if name == "" {
			name = d.ChipType
		}
		if name == "" {
			name = "Tenstorrent Device"
		}

		vramGB := d.DRAM.Total
		if strings.EqualFold(d.DRAM.Unit, "MB") {
			vramGB = vramGB / 1024
		}

		hwDevices = append(hwDevices, DeviceInfo{
			Name:           "Tenstorrent " + strings.ToUpper(name),
			VRAM_GB:        vramGB,
			DriverVersion:  d.FirmwareVersion,
			RuntimeVersion: d.ChipType,
			PCIAddress:     d.PCIeBusID,
		})
	}

	return &HardwareInfo{
		Platform: "tenstorrent",
		Devices:  hwDevices,
	}, nil
}

// parseTtSmiText parses text output from tt-smi.
//
// Example:
//
//	╭──────────────────────┬──────────────────────────────────────────╮
//	│                      │  Board: WH N300                          │
//	│   ┌─────┐            │  Chip: wormhole                          │
//	│   │     │            │  Firmware: 0.1.0                         │
//	│   │     │            │  DRAM: 12 GB                             │
//	│   └─────┘            │  PCIe: 0000:3b:00.0                      │
//	│                      │                                          │
func (p *TenstorrentPlatform) parseTtSmiText(output string) (*HardwareInfo, error) {
	var devices []DeviceInfo

	// Split into sections by device (double newline or board separator)
	// Simple line-based parsing
	boardRe := regexp.MustCompile(`Board:\s*(.+)`)
	chipRe := regexp.MustCompile(`Chip:\s*(.+)`)
	fwRe := regexp.MustCompile(`Firmware:\s*(.+)`)
	dramRe := regexp.MustCompile(`DRAM:\s*(\d+)\s*(GB|MB|MiB)`)
	pcieRe := regexp.MustCompile(`PCIe?:\s*([0-9a-fA-F:.]+)`)

	// Try to extract all devices from the text
	lines := strings.Split(output, "\n")
	var currentDev *DeviceInfo

	for _, line := range lines {
		line = strings.TrimSpace(strings.Trim(line, "│"))

		// New device section
		if m := boardRe.FindStringSubmatch(line); len(m) > 1 {
			if currentDev != nil {
				devices = append(devices, *currentDev)
			}
			boardName := strings.TrimSpace(m[1])
			currentDev = &DeviceInfo{
				Name: "Tenstorrent " + strings.ToUpper(boardName),
			}
		}

		if currentDev == nil {
			// Might be first device without explicit "Board:" prefix
			if m := chipRe.FindStringSubmatch(line); len(m) > 1 {
				currentDev = &DeviceInfo{
					RuntimeVersion: strings.TrimSpace(m[1]),
				}
			}
			continue
		}

		if m := chipRe.FindStringSubmatch(line); len(m) > 1 {
			currentDev.RuntimeVersion = strings.TrimSpace(m[1])
		}
		if m := fwRe.FindStringSubmatch(line); len(m) > 1 {
			currentDev.DriverVersion = strings.TrimSpace(m[1])
		}
		if m := dramRe.FindStringSubmatch(line); len(m) > 1 {
			val, _ := strconv.Atoi(m[1])
			if strings.EqualFold(m[2], "MB") || strings.EqualFold(m[2], "MiB") {
				currentDev.VRAM_GB = val / 1024
			} else {
				currentDev.VRAM_GB = val
			}
		}
		if m := pcieRe.FindStringSubmatch(line); len(m) > 1 {
			currentDev.PCIAddress = m[1]
		}
	}

	if currentDev != nil {
		devices = append(devices, *currentDev)
	}

	if len(devices) == 0 {
		return nil, fmt.Errorf("could not parse any devices from tt-smi output")
	}

	// Fill in missing names
	for i := range devices {
		if devices[i].Name == "" {
			devices[i].Name = "Tenstorrent Device"
			if devices[i].RuntimeVersion != "" {
				devices[i].Name = "Tenstorrent " + strings.Title(devices[i].RuntimeVersion)
			}
		}
	}

	return &HardwareInfo{
		Platform: "tenstorrent",
		Devices:  devices,
	}, nil
}

// detectViaTtTopology uses `tt-topology` to enumerate devices.
func (p *TenstorrentPlatform) detectViaTtTopology(ctx context.Context) (*HardwareInfo, error) {
	output, err := sysinfo.Exec(ctx, "tt-topology", "--list")
	if err != nil {
		return nil, fmt.Errorf("tt-topology --list: %w", err)
	}

	return p.parseTtTopology(output)
}

func (p *TenstorrentPlatform) parseTtTopology(output string) (*HardwareInfo, error) {
	var devices []DeviceInfo

	// tt-topology output:
	//   Board: WH N300 (wormhole), PCIe: 0000:3b:00.0, DRAM: 12 GB
	boardRe := regexp.MustCompile(`Board:\s*(.+?)\s*\((\w+)\)\s*,\s*PCIe?:\s*([0-9a-fA-F:.]+)\s*,\s*DRAM:\s*(\d+)\s*(GB|MB)`)

	for _, line := range strings.Split(output, "\n") {
		m := boardRe.FindStringSubmatch(line)
		if len(m) < 5 {
			continue
		}

		boardName := strings.TrimSpace(m[1])
		chipType := strings.TrimSpace(m[2])
		pciAddr := m[3]
		dram, _ := strconv.Atoi(m[4])
		unit := m[5]

		vramGB := dram
		if strings.EqualFold(unit, "MB") {
			vramGB = dram / 1024
		}

		devices = append(devices, DeviceInfo{
			Name:           "Tenstorrent " + boardName,
			VRAM_GB:        vramGB,
			RuntimeVersion: chipType,
			PCIAddress:     pciAddr,
		})
	}

	if len(devices) == 0 {
		return nil, fmt.Errorf("no Tenstorrent devices found in tt-topology output")
	}

	return &HardwareInfo{
		Platform: "tenstorrent",
		Devices:  devices,
	}, nil
}

// detectViaSysfs scans /sys/class/tenstorrent for device nodes.
func (p *TenstorrentPlatform) detectViaSysfs() (*HardwareInfo, error) {
	ttDir := "/sys/class/tenstorrent"
	entries, err := os.ReadDir(ttDir)
	if err != nil {
		return nil, fmt.Errorf("cannot read %s: %w", ttDir, err)
	}

	var devices []DeviceInfo
	for _, entry := range entries {
		if !entry.IsDir() && !strings.HasPrefix(entry.Name(), "tt") {
			continue
		}

		devDir := filepath.Join(ttDir, entry.Name())
		dev := DeviceInfo{}

		// Read device type
		if data, err := os.ReadFile(filepath.Join(devDir, "device_type")); err == nil {
			dev.RuntimeVersion = strings.TrimSpace(string(data))
		}

		// Read board type
		if data, err := os.ReadFile(filepath.Join(devDir, "board_type")); err == nil {
			boardType := strings.TrimSpace(string(data))
			dev.Name = "Tenstorrent " + strings.ToUpper(boardType)
		}

		// Read firmware version
		if data, err := os.ReadFile(filepath.Join(devDir, "fw_version")); err == nil {
			dev.DriverVersion = strings.TrimSpace(string(data))
		}

		// Read DRAM size
		if data, err := os.ReadFile(filepath.Join(devDir, "dram_size")); err == nil {
			if val, err := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64); err == nil {
				dev.VRAM_GB = int(val / 1024 / 1024 / 1024)
				if dev.VRAM_GB == 0 && val > 0 {
					// Might be in GB already
					dev.VRAM_GB = int(val)
				}
			}
		}

		// Read PCIe address from symlink
		if link, err := os.Readlink(filepath.Join(devDir, "device")); err == nil {
			pciAddr := filepath.Base(link)
			if pciAddr != "" {
				dev.PCIAddress = pciAddr
			}
		}

		if dev.Name == "" {
			dev.Name = "Tenstorrent Device"
		}

		devices = append(devices, dev)
	}

	if len(devices) == 0 {
		return nil, fmt.Errorf("no Tenstorrent devices found in %s", ttDir)
	}

	return &HardwareInfo{
		Platform: "tenstorrent",
		Devices:  devices,
	}, nil
}

// detectViaLspci uses lspci to find Tenstorrent PCI devices.
//
// Tenstorrent PCI vendor ID: 0x1e52 (formerly 0x0a12)
func (p *TenstorrentPlatform) detectViaLspci(ctx context.Context) (*HardwareInfo, error) {
	if _, err := sysinfo.LookPath("lspci"); err != nil {
		return nil, fmt.Errorf("lspci not found: %w", err)
	}

	output, err := sysinfo.Exec(ctx, "lspci", "-nn", "-d", "1e52:")
	if err != nil {
		// Try legacy vendor ID
		output, err = sysinfo.Exec(ctx, "lspci", "-nn", "-d", "0a12:")
		if err != nil {
			return nil, fmt.Errorf("no Tenstorrent PCI devices found: %w", err)
		}
	}

	var devices []DeviceInfo
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Parse: "3b:00.0 Processing accelerators [0b40]: Tenstorrent Inc. Device [1e52:1b01]"
		pciAddr := ""
		name := "Tenstorrent Device"

		// Extract PCI address
		if idx := strings.Index(line, " "); idx > 0 {
			pciAddr = line[:idx]
		}

		// Extract device name after ": "
		if idx := strings.LastIndex(line, ": "); idx >= 0 {
			name = strings.TrimSpace(line[idx+2:])
		}

		// Determine model from PCI device ID
		model := ttModelFromPCI(line)

		devices = append(devices, DeviceInfo{
			Name:       name + " (" + model + ")",
			PCIAddress: pciAddr,
		})
	}

	if len(devices) == 0 {
		return nil, fmt.Errorf("no Tenstorrent PCI devices found via lspci")
	}

	return &HardwareInfo{
		Platform: "tenstorrent",
		Devices:  devices,
	}, nil
}

// ttModelFromPCI determines the Tenstorrent board model from the PCI device line.
func ttModelFromPCI(line string) string {
	// Known device IDs
	deviceIDs := map[string]string{
		"1b01": "Wormhole n300",
		"1b02": "Wormhole n150",
		"1406": "Grayskull e150",
		"1408": "Grayskull e75",
	}

	re := regexp.MustCompile(`\[1e52:([0-9a-fA-F]{4})\]`)
	if m := re.FindStringSubmatch(line); len(m) > 1 {
		if model, ok := deviceIDs[strings.ToLower(m[1])]; ok {
			return model
		}
	}

	// Try legacy vendor ID
	reLegacy := regexp.MustCompile(`\[0a12:([0-9a-fA-F]{4})\]`)
	if m := reLegacy.FindStringSubmatch(line); len(m) > 1 {
		if model, ok := deviceIDs[strings.ToLower(m[1])]; ok {
			return model
		}
	}

	return "Unknown"
}

// ── Runtime Detection ──────────────────────────────────────────────────────

func (p *TenstorrentPlatform) DetectOrInstallRuntime(ctx context.Context) error {
	// Check for tt-smi or tt-topology
	if _, err := sysinfo.LookPath("tt-smi"); err != nil {
		if _, err := sysinfo.LookPath("tt-topology"); err != nil {
			return fmt.Errorf("neither tt-smi nor tt-topology found — " +
				"install Tenstorrent software stack: " +
				"https://github.com/tenstorrent/tt-smi")
		}
	}

	// Check kernel module
	if _, err := os.Stat("/sys/class/tenstorrent"); err != nil {
		// Try loading the module
		if err := loadTTKernelModule(ctx); err != nil {
			return fmt.Errorf("Tenstorrent kernel module not loaded: %w", err)
		}
	}

	// Verify hugepages (recommended for TT performance)
	if err := checkHugepages(); err != nil {
		// Warning only, not fatal
		fmt.Printf("  ⚠ Warning: %v\n", err)
	}

	// Determine if Docker or native
	p.useDocker = p.detectDockerAvailable(ctx)

	return nil
}

// loadTTKernelModule attempts to load the Tenstorrent kernel module.
func loadTTKernelModule(ctx context.Context) error {
	cmd := exec.CommandContext(ctx, "modprobe", "tenstorrent")
	return cmd.Run()
}

// checkHugepages verifies that hugepages are configured.
func checkHugepages() error {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return nil // can't check, skip
	}

	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "HugePages_Total:") {
			parts := strings.Fields(line)
			if len(parts) >= 2 {
				if total, err := strconv.Atoi(parts[1]); err == nil && total > 0 {
					return nil // hugepages configured
				}
			}
		}
	}

	return fmt.Errorf("hugepages not configured — recommended for Tenstorrent performance")
}

// detectDockerAvailable checks if Docker is available for TT container usage.
func (p *TenstorrentPlatform) detectDockerAvailable(ctx context.Context) bool {
	if _, err := sysinfo.LookPath("docker"); err != nil {
		return false
	}
	_, err := sysinfo.Exec(ctx, "docker", "info")
	return err == nil
}

// ── Container Configuration ────────────────────────────────────────────────

func (p *TenstorrentPlatform) GetDockerImage(override string) string {
	if override != "" {
		return override
	}
	return "ghcr.io/tenstorrent/tt-inference-server:latest"
}

func (p *TenstorrentPlatform) ContainerConfig(model ModelConfig, opts RunOptions) *ContainerConfig {
	image := p.GetDockerImage(opts.DockerImage)
	containerName := fmt.Sprintf("tt_bench_%d", opts.Port)

	envVars := map[string]string{
		"MODEL_ID": model.ID,
		"PORT":     strconv.Itoa(8000), // internal port
	}

	if opts.HFToken != "" {
		envVars["HUGGING_FACE_HUB_TOKEN"] = opts.HFToken
	}

	// TT device passthrough flags
	gpuFlags := []string{}

	if p.hw != nil {
		// Pass through specific TT PCI devices
		for _, dev := range p.hw.Devices {
			if dev.PCIAddress != "" {
				gpuFlags = append(gpuFlags,
					"--device", fmt.Sprintf("/dev/tenstorrent/%s", dev.PCIAddress))
			}
		}
	}

	// Generic TT device passthrough
	gpuFlags = append(gpuFlags,
		"--device", "/dev/tenstorrent",
		"--volume", "/dev/hugepages:/dev/hugepages",
	)

	// Hugepage mount for performance
	gpuFlags = append(gpuFlags,
		"--shm-size", "1g",
	)

	// Build extra args based on serving backend
	extraArgs := []string{}

	// Detect if we're using tt-inference-server (native) or vLLM TT backend
	if p.isVLLMBackend(model) {
		// vLLM with TT backend
		extraArgs = append(extraArgs,
			model.ID,
			"--trust-remote-code",
			"--max-model-len", strconv.Itoa(opts.MaxModelLen),
			"--port", "8000",
		)
	} else {
		// tt-inference-server native
		extraArgs = append(extraArgs,
			"--model", model.ID,
			"--max-model-len", strconv.Itoa(opts.MaxModelLen),
			"--port", "8000",
		)
	}

	extraArgs = append(extraArgs, quantFlags(model.Quant)...)
	if model.ExtraFlags != "" {
		extraArgs = append(extraArgs, strings.Fields(model.ExtraFlags)...)
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

// isVLLMBackend checks if the model should use vLLM TT backend.
func (p *TenstorrentPlatform) isVLLMBackend(model ModelConfig) bool {
	// Models with extra_flags containing "vllm" use vLLM
	return strings.Contains(strings.ToLower(model.ExtraFlags), "vllm") ||
		strings.Contains(strings.ToLower(model.ExtraFlags), "serving")
}

func (p *TenstorrentPlatform) HealthEndpoint() string {
	return "/health"
}

// ── Native Serving (non-Docker) ────────────────────────────────────────────

// StartNativeServer starts tt-inference-server natively (without Docker).
// This is the preferred path when Docker isn't available.
func (p *TenstorrentPlatform) StartNativeServer(ctx context.Context, model ModelConfig, opts RunOptions) (int, error) {
	port := opts.Port
	if port == 0 {
		port = 8000
	}

	args := []string{
		"--model", model.ID,
		"--port", strconv.Itoa(port),
		"--max-model-len", strconv.Itoa(opts.MaxModelLen),
	}
	args = append(args, quantFlags(model.Quant)...)
	if model.ExtraFlags != "" {
		args = append(args, strings.Fields(model.ExtraFlags)...)
	}

	// Look for tt-inference-server binary
	serverBin := "tt-inference-server"
	if _, err := sysinfo.LookPath(serverBin); err != nil {
		return 0, fmt.Errorf("tt-inference-server not found in PATH: %w", err)
	}

	cmd := exec.CommandContext(ctx, serverBin, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if opts.HFToken != "" {
		cmd.Env = append(os.Environ(), "HUGGING_FACE_HUB_TOKEN="+opts.HFToken)
	}

	if err := cmd.Start(); err != nil {
		return 0, fmt.Errorf("start tt-inference-server: %w", err)
	}

	// Wait for healthy
	if err := p.waitForHealthy(ctx, port, 10*time.Minute); err != nil {
		cmd.Process.Kill()
		return 0, err
	}

	return port, nil
}

// waitForHealthy polls the health endpoint until it responds OK.
func (p *TenstorrentPlatform) waitForHealthy(ctx context.Context, port int, maxWait time.Duration) error {
	url := fmt.Sprintf("http://localhost:%d%s", port, p.HealthEndpoint())
	deadline := time.Now().Add(maxWait)

	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		cmd := exec.CommandContext(ctx, "curl", "-sf", url)
		if err := cmd.Run(); err == nil {
			return nil
		}
		time.Sleep(3 * time.Second)
	}

	return fmt.Errorf("server did not become healthy within %v", maxWait)
}
