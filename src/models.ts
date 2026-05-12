// ── YAML model registry parser ──────────────────────────────────────────────

import { parse as parseYaml } from "yaml";
import { readFileSync } from "node:fs";
import type { ModelEntry } from "./types";

export function loadModels(yamlPath: string): ModelEntry[] {
  let content: string;
  try {
    content = readFileSync(yamlPath, "utf-8");
  } catch {
    throw new Error(`Models file not found: ${yamlPath}`);
  }

  const data = parseYaml(content);

  if (!data?.models || !Array.isArray(data.models)) {
    throw new Error("Invalid models.yaml: expected a 'models' array");
  }

  return data.models.map((m: Record<string, unknown>, i: number) => {
    if (!m.id || typeof m.id !== "string") {
      throw new Error(`Invalid models.yaml: entry ${i + 1} missing required 'id' field`);
    }
    return {
      id: m.id,
      name: String(m.name ?? m.id),
      quant: String(m.quant ?? "none"),
      min_vram_gb: Number(m.min_vram_gb ?? 0),
      tp: Number(m.tp ?? 1),
      extra_flags: String(m.extra_flags ?? ""),
    };
  });
}
