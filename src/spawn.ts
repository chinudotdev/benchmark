// ── Shared spawn helper ──────────────────────────────────────────────────────
//
// Avoids duplication across modules. Reads both stdout and stderr to prevent
// pipe-buffer deadlocks on long-running processes.

export interface SpawnResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}

export async function spawn(
  cmd: string[],
  opts?: { stdout?: "pipe" | "inherit" | "ignore"; stderr?: "pipe" | "inherit" | "ignore" },
): Promise<SpawnResult> {
  try {
    const proc = Bun.spawn(cmd, {
      stdout: opts?.stdout ?? "pipe",
      stderr: opts?.stderr ?? "pipe",
    });
    const stdout = (opts?.stdout ?? "pipe") === "pipe"
      ? await new Response(proc.stdout).text()
      : "";
    const stderr = (opts?.stderr ?? "pipe") === "pipe"
      ? await new Response(proc.stderr).text()
      : "";
    const exitCode = await proc.exited;
    return { stdout: stdout.trim(), stderr: stderr.trim(), exitCode };
  } catch {
    return { stdout: "", stderr: "", exitCode: -1 };
  }
}
