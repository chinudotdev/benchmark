# TODO — Minor Improvements

Low-priority items for future updates.

## 1. HF token visibility in process list

The Docker `-e HUGGING_FACE_HUB_TOKEN=hf_xxx` flag is visible in `ps aux`.

**Fix:** Use `--env-file` or Docker secrets instead.

---

## 2. Streaming token count fallback is coarse

When `usage.completion_tokens` isn't present in the SSE stream, the fallback is `outputTokens = maxTokens` (always the max). This overestimates TPS.

**Fix:** Add `stream_options: { include_usage: true }` to the streaming payload — vLLM includes `usage` in the final `[DONE]` chunk when this is set.

---

## 3. `retries` hardcoded at call site

`benchmarkModel` passes `retries: 2` directly.

**Fix:** Make it a CLI flag (`--retries`) or at least reference a shared named constant.

---

## 4. `isPortInUse` message could be more helpful

If a previous vLLM bench container is still running on the port, the error says "already in use" without suggesting it might be a stale bench container.

**Fix:** Check if a `vllm_bench_*` container exists on that port and suggest `docker rm -f vllm_bench_<port>`.

---

## 5. `COST_UNAVAILABLE` sentinel as -1

Using `-1` works, but a valid cost could theoretically be negative (credits/discounts). More semantically correct to use `null`.

**Fix:** Write `null` to JSON when cost is unavailable, type `cost_per_1m_output_tokens_usd` as `number | null`, and update display code accordingly.
