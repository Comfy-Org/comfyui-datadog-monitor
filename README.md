# ComfyUI Datadog Monitor

Background extension that adds Datadog APM tracing and CUDA memory tracking to ComfyUI workflow execution. No UI nodes — runs entirely in the background.

## Features

 **Automatic Library Instrumentation**: Uses `ddtrace.auto` to automatically trace HTTP requests, subprocess calls, asyncio operations, logging, and 100+ other Python libraries
 **Workflow Tracing**: Each `execute_async` call is wrapped in a Datadog APM span with prompt ID and job ID tags
 **CUDA Memory Snapshots**: Captures `torch.cuda.memory_stats()` before and after each workflow execution
 **CUDA Allocation Tracking**: Uses `torch.cuda.memory._snapshot()` to identify top VRAM allocations with stack traces
 **OOM Diagnostics**: Memory snapshots are captured in a `finally` block, so they're available even when workflows fail with OOM errors
 **MPS Support**: Basic memory tracking for Apple Silicon (allocated + driver memory)
 **Runtime Metrics**: Enables `ddtrace` runtime metrics collection
 **Zero Configuration**: Works automatically when installed — no nodes to add
 **Background Only**: No UI nodes, runs entirely in the background

## What Gets Traced

### Automatic (via `ddtrace.auto`)

The extension enables `ddtrace.auto` which uses import hooks to automatically instrument supported libraries. For ComfyUI, the most relevant auto-instrumented libraries are:

 **`requests` / `httpx`**: Model downloads, API calls to external services
 **`asyncio`**: Async span correlation across the event loop
 **`subprocess`**: Any external process spawns
 **`logging`**: Trace ID injection into log lines for correlation
 **`sqlite3`**: Database operations (if used)
 **`aiohttp`**: Async HTTP operations

### Manual (workflow-level)

This extension also monkey-patches `PromptExecutor.execute_async` to wrap each workflow execution in a Datadog span. Each span includes:

 `workflow.prompt_id`: The prompt ID being executed
 `job.id`: The job ID from extra_data (if present)
 `memory.pytorch.allocated_bytes.{before,after}`: CUDA memory allocated
 `memory.pytorch.reserved_bytes.{before,after}`: CUDA memory reserved
 `memory.pytorch.num_ooms.{before,after}`: OOM count from PyTorch stats
 `memory.pytorch.cuda_mb.after`: Total CUDA allocation in MB
 `memory.pytorch.largest_mb.after`: Largest single CUDA allocation in MB
 `error` / `error.type`: Set on workflow exceptions

## Installation

1. Install in your ComfyUI custom_nodes directory:
```bash
cd custom_nodes
git clone https://github.com/Comfy-Org/comfyui-datadog-monitor
cd comfyui-datadog-monitor
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export DD_ENV=production
export DD_SERVICE=comfyui
export DD_AGENT_HOST=localhost  # Your Datadog agent host

# Optional: enable detailed CUDA memory tracking
export PYTORCH_MEMORY_TRACKING=true
```

3. Restart ComfyUI — tracing starts automatically.

## How It Works

When ComfyUI loads this extension:
1. Calls `import ddtrace.auto` to enable automatic instrumentation of all supported libraries (uses import hooks, so it works even when libraries are already imported)
2. Configures `ddtrace` with service/env tags and enables runtime metrics
3. If `PYTORCH_MEMORY_TRACKING=true`, enables `torch.cuda.memory._record_memory_history()` for detailed allocation tracking
4. Monkey-patches `PromptExecutor.execute_async` to wrap workflow execution in a Datadog span with memory snapshots

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DD_ENV` | `production` | Datadog environment tag |
| `DD_SERVICE` | `comfyui` | Datadog service name |
| `DD_AGENT_HOST` | `localhost` | Datadog agent hostname |
| `PYTORCH_MEMORY_TRACKING` | `false` | Enable CUDA memory snapshots and allocation tracking |
| `DD_PROFILING_ENABLED` | `true` | Enable ddtrace CPU/memory profiler (~2% CPU overhead) |

Standard `ddtrace` environment variables (e.g. `DD_TRACE_SAMPLE_RATE`, `DD_LOGS_INJECTION`, `DD_TRACE_<LIBRARY>_ENABLED`) are also respected. You can disable auto-instrumentation for specific libraries with e.g. `DD_TRACE_REQUESTS_ENABLED=false`.

## OOM Debugging

When `PYTORCH_MEMORY_TRACKING=true`, the extension captures memory state in a `finally` block — so even when a workflow fails with a CUDA OOM error, you get:

- Memory stats at the point of failure (allocated, reserved, OOM count)
- Top CUDA VRAM allocations with stack traces showing where memory was allocated

Look for spans with `error=True` and `error.type=OutOfMemoryError` in Datadog APM.

## Performance Impact

 **Auto-instrumentation overhead**: Negligible per-call wrapping of library functions. For a GPU-bound ML inference workload, this is immeasurable.
 **Tracing overhead**: One additional span per workflow execution, plus child spans from auto-instrumented libraries.
 **Memory tracking** (when enabled): `torch.cuda.memory_stats()` and `torch.cuda.memory._snapshot()` are called twice per workflow (before/after). These are O(segments) operations on the CUDA allocator's internal data — no Python object scanning.
 ⏱ **CPU profiling** (enabled by default): Sampling-based, ~2% CPU overhead. No impact on GPU/CUDA kernel execution. Disable with `DD_PROFILING_ENABLED=false`.

## Troubleshooting

**DDTrace fails to start**: Check if Datadog agent is running and accessible at `DD_AGENT_HOST`.

**No data in Datadog**: Verify `DD_AGENT_HOST` points to your Datadog agent.

**Import error**: Make sure `ddtrace` is installed: `pip install ddtrace`

**No memory metrics**: Set `PYTORCH_MEMORY_TRACKING=true` and ensure CUDA is available.

## License

MIT
