package platform

import (
	"context"
	"fmt"

	"github.com/chinudotdev/gpu-benchmark/internal/sysinfo"
)

// TenstorrentPlatform implements Platform for Tenstorrent Wormhole accelerators.
// Stub for Milestone 4 — will be fleshed out with real hardware access.
type TenstorrentPlatform struct {
	hw *HardwareInfo
}

func NewTenstorrentPlatform() *TenstorrentPlatform {
	return &TenstorrentPlatform{}
}

func (p *TenstorrentPlatform) Name() string { return "tenstorrent" }

func (p *TenstorrentPlatform) DetectHardware(ctx context.Context) (*HardwareInfo, error) {
	// TODO: Probe for Wormhole n150/n300 via TT tools or PCIe device IDs
	// For now, detect if tt-smi or similar tooling is present
	if _, err := sysinfo.LookPath("tt-smi"); err != nil {
		return nil, fmt.Errorf("tt-smi not found: %w (Tenstorrent tools not installed)", err)
	}

	// TODO: Parse tt-smi output for device names, memory, firmware version
	return nil, fmt.Errorf("Tenstorrent hardware detection not yet implemented — pending hardware access")
}

func (p *TenstorrentPlatform) DetectOrInstallRuntime(ctx context.Context) error {
	return fmt.Errorf("Tenstorrent runtime detection not yet implemented — pending hardware access")
}

func (p *TenstorrentPlatform) GetDockerImage(override string) string {
	if override != "" {
		return override
	}
	// TT may use tt-inference-server natively rather than Docker
	return ""
}

func (p *TenstorrentPlatform) ContainerConfig(model ModelConfig, opts RunOptions) *ContainerConfig {
	// TODO: tt-inference-server may not use Docker at all.
	// This will return nil and the orchestration layer will use a native path.
	return nil
}

func (p *TenstorrentPlatform) HealthEndpoint() string {
	return "/health"
}
