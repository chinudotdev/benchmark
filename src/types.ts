// ── Shared types for gpu-benchmark ──────────────────────────────────────────

export interface ModelEntry {
  id: string;
  name: string;
  quant: string;
  min_vram_gb: number;
  tp: number;
  extra_flags: string;
}

export interface BenchmarkOptions {
  modelId?: string;
  all: boolean;
  gpuRate: number;
  gpuCount: number;
  inputLen: number;
  outputLen: number;
  numPrompts: number;
  maxModelLen: number;
  port: number;
  resultsDir: string;
  concurrency: number;
  dryRun: boolean;
  modelsYaml: string;
  quant?: string;
  hfToken?: string;
  dockerImage?: string;
  gpuIds?: string;
  stream?: boolean;
  force?: boolean;
}

export interface GpuInfo {
  name: string;
  vramGb: number;
  fullInfo: string;
}

export interface RequestResult {
  success: boolean;
  outputTokens: number;
  latencyS: number;
  error?: string;
}

export interface BenchmarkMetrics {
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  total_time_s: number;
  total_output_tokens: number;
  output_tps: number;
  latency_p50_s: number;
  latency_p95_s: number;
  latency_p99_s: number;
}

export interface GpuDetails {
  name: string;
  vram_gb: number;
  driver_version: string;
  cuda_version: string;
}

export interface CpuDetails {
  model: string;
  cores: number;
}

export interface SystemInfo {
  gpus: GpuDetails[];
  cpu: CpuDetails;
  ram_gb: number;
  os: string;
  kernel: string;
  docker_version: string;
  vllm_image: string;
  disk_available_gb: number; // available space on HF cache mount
  collected_at: string;
}

export interface BenchmarkResult {
  model_id: string;
  model_name: string;
  quant: string;
  gpu: string;
  gpu_count: number;
  gpu_hourly_rate_usd: number;
  benchmark: {
    num_prompts: number;
    input_len: number;
    output_len: number;
    max_model_len: number;
  };
  results: {
    output_tokens_per_sec: number;
    cost_per_1m_output_tokens_usd: number | null; // null when cost is unavailable (TPS was 0)
  };
  system: SystemInfo;
  timestamp: string;
}

export interface BenchRunnerOptions {
  model: string;
  host: string;
  port: number;
  numPrompts: number;
  inputLen: number;
  outputLen: number;
  concurrency: number;
  stream: boolean;
  retries: number;
}
