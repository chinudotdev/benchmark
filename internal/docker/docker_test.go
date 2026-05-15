package docker

import (
	"context"
	"testing"
)

func TestNewManager(t *testing.T) {
	ctx := context.Background()
	mgr, err := NewManager(ctx)
	if err != nil {
		t.Fatalf("NewManager error: %v", err)
	}
	if mgr == nil {
		t.Error("manager is nil")
	}
}

func TestContainerStatusNonexistent(t *testing.T) {
	ctx := context.Background()
	mgr, _ := NewManager(ctx)
	status := mgr.ContainerStatus(ctx, "nonexistent_container_xyz_12345")
	if status != "" {
		t.Errorf("expected empty status for nonexistent container, got %q", status)
	}
}

func TestStopNonexistent(t *testing.T) {
	mgr, _ := NewManager(context.Background())
	// Should not error on nonexistent container
	err := mgr.Stop("nonexistent_container_xyz_12345")
	if err != nil {
		t.Errorf("Stop should not error on nonexistent container: %v", err)
	}
}

func TestCleanupNoActive(t *testing.T) {
	mgr, _ := NewManager(context.Background())
	// Should not panic with no active container
	mgr.Cleanup()
}

func TestActiveContainer(t *testing.T) {
	mgr, _ := NewManager(context.Background())
	if mgr.ActiveContainer() != "" {
		t.Error("expected empty active container initially")
	}
}
