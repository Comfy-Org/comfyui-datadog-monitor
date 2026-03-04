# ComfyUI Datadog Monitor

Background extension that adds Datadog APM tracing and CUDA memory tracking to ComfyUI workflow execution. No UI nodes — runs entirely in the background.

## Features

 **Automatic Library Instrumentation**: Uses `ddtrace.auto` to automatically trace HTTP requests, subprocess calls, asyncio operations, logging, and 100+ other Python libraries
 **Workflow Tracing**: Each `execute_async` call is wrapped in a Datadog APM span with prompt ID and job ID tags
 **Per-Node Tracing**: Each node execution gets a child span under the workflow span, tagged with `class_type`, node ID, and optional memory metrics
 **CUDA Memory Snapshots**: Captures `torch.cuda.memory_stats()` before and after each workflow execution (opt-in)
 **CUDA Allocation Tracking**: Uses `torch.cuda.memory._snapshot()` to identify top VRAM allocations with stack traces (opt-in)
 **Per-Node Memory Tracking**: Cheap VRAM/RAM snapshots per node using `torch.cuda.memory_allocated()` and `psutil.virtual_memory()` (~0.1ms each)
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

This extension monkey-patches `PromptExecutor.execute_async` to wrap each workflow execution in a Datadog span. Each span includes:

 `workflow.prompt_id`: The prompt ID being executed
 `job.id`: The job ID from extra_data (if present)
 `memory.pytorch.allocated_bytes.{before,after}`: CUDA memory allocated (when `PYTORCH_MEMORY_TRACKING=true`)
 `memory.pytorch.reserved_bytes.{before,after}`: CUDA memory reserved (when `PYTORCH_MEMORY_TRACKING=true`)
 `memory.pytorch.num_ooms.{before,after}`: OOM count from PyTorch stats (when `PYTORCH_MEMORY_TRACKING=true`)
 `memory.pytorch.cuda_mb.after`: Total CUDA allocation in MB (when `PYTORCH_MEMORY_TRACKING=true`)
 `memory.pytorch.largest_mb.after`: Largest single CUDA allocation in MB (when `PYTORCH_MEMORY_TRACKING=true`)
 `error` / `error.type`: Set on workflow exceptions

### Manual (per-node)

Each node execution is wrapped in a child span (`comfyui.node.execute`) under the workflow span. This is enabled by default and requires no configuration. Each node span includes:

 `node.id`: The unique node ID within the workflow
 `node.class_type`: The ComfyUI node type (e.g. `KSampler`, `VAEDecode`, `CLIPTextEncode`)
 `workflow.prompt_id`: The prompt ID (same as parent workflow span)
 `node.cached`: `true` if the node result was served from cache
 `node.result`: The execution result status (e.g. `SUCCESS`, `FAILURE`)
 `error` / `error.type` / `error.message`: Set on node exceptions
 `memory.vram_allocated_bytes.{before,after}`: CUDA VRAM allocated (cheap, ~0.1ms)
 `memory.vram_reserved_bytes.{before,after}`: CUDA VRAM reserved (cheap, ~0.1ms)
 `memory.ram_available_bytes.{before,after}`: System RAM available (cheap, ~0.1ms)

### Trace Structure

```
comfyui.workflow.execute  (resource: workflow#<prompt_id>)
├── comfyui.node.execute  (resource: CLIPTextEncode)
├── comfyui.node.execute  (resource: KSampler)        ← longest span = bottleneck
├── comfyui.node.execute  (resource: VAEDecode)
├── comfyui.node.execute  (resource: SaveImage)
└── comfyui.node.execute  (resource: LoadCheckpoint)   ← node.cached=true
```

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

# Optional: enable detailed CUDA memory tracking (workflow-level, expensive)
export PYTORCH_MEMORY_TRACKING=true
```

3. Restart ComfyUI — tracing starts automatically.

## How It Works

When ComfyUI loads this extension:
1. Calls `import ddtrace.auto` to enable automatic instrumentation of all supported libraries (uses import hooks, so it works even when libraries are already imported)
2. Configures `ddtrace` with service/env tags and enables runtime metrics
3. If `PYTORCH_MEMORY_TRACKING=true`, enables `torch.cuda.memory._record_memory_history()` for detailed allocation tracking
4. Monkey-patches `PromptExecutor.execute_async` to wrap workflow execution in a Datadog span with memory snapshots
5. Monkey-patches the module-level `execution.execute()` function to wrap each node execution in a child span with cheap memory snapshots

Child span nesting is handled automatically by ddtrace's context propagation — no explicit parent linking is needed.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DD_ENV` | `production` | Datadog environment tag |
| `DD_SERVICE` | `comfyui` | Datadog service name |
| `DD_AGENT_HOST` | `localhost` | Datadog agent hostname |
| `PYTORCH_MEMORY_TRACKING` | `false` | Enable detailed CUDA memory snapshots and allocation tracking (workflow-level, uses expensive `memory_stats()` and `memory._snapshot()`) |
| `NODE_TRACING_ENABLED` | `true` | Enable per-node child spans under the workflow span |
| `NODE_MEMORY_TRACKING` | `true` | Enable cheap VRAM/RAM snapshots on each node span (only applies when `NODE_TRACING_ENABLED=true`) |
| `DD_PROFILING_ENABLED` | `true` | Enable ddtrace CPU/memory profiler (~2% CPU overhead) |

Standard `ddtrace` environment variables (e.g. `DD_TRACE_SAMPLE_RATE`, `DD_LOGS_INJECTION`, `DD_TRACE_<LIBRARY>_ENABLED`) are also respected. You can disable auto-instrumentation for specific libraries with e.g. `DD_TRACE_REQUESTS_ENABLED=false`.

## OOM Debugging

When `PYTORCH_MEMORY_TRACKING=true`, the extension captures memory state in a `finally` block — so even when a workflow fails with a CUDA OOM error, you get:

- Memory stats at the point of failure (allocated, reserved, OOM count)
- Top CUDA VRAM allocations with stack traces showing where memory was allocated

Look for spans with `error=True` and `error.type=OutOfMemoryError` in Datadog APM.

## Performance Impact

 **Auto-instrumentation overhead**: Negligible per-call wrapping of library functions. For a GPU-bound ML inference workload, this is immeasurable.
 **Workflow tracing**: One additional span per workflow execution, plus child spans from auto-instrumented libraries.
 **Per-node tracing** (enabled by default): One span per node execution. Uses `torch.cuda.memory_allocated()` and `psutil.virtual_memory()` for memory snapshots — both are ~0.1ms with no GPU synchronization. For a typical 10-node workflow, this adds ~2ms total overhead.
 **Detailed memory tracking** (opt-in): `torch.cuda.memory_stats()` (~5-10ms) and `torch.cuda.memory._snapshot()` (~50-200ms) are called twice per workflow (before/after). These are O(segments) operations on the CUDA allocator's internal data — no Python object scanning.
 ⏱ **CPU profiling** (enabled by default): Sampling-based, ~2% CPU overhead. No impact on GPU/CUDA kernel execution. Disable with `DD_PROFILING_ENABLED=false`.

## Troubleshooting

**DDTrace fails to start**: Check if Datadog agent is running and accessible at `DD_AGENT_HOST`.

**No data in Datadog**: Verify `DD_AGENT_HOST` points to your Datadog agent.

**Import error**: Make sure `ddtrace` is installed: `pip install ddtrace`

**No memory metrics**: Set `PYTORCH_MEMORY_TRACKING=true` and ensure CUDA is available. For per-node memory metrics, ensure `NODE_MEMORY_TRACKING=true` (default).

**Too many spans**: If per-node spans create too much volume, disable with `NODE_TRACING_ENABLED=false`. Or keep tracing but disable memory snapshots with `NODE_MEMORY_TRACKING=false`.

**No per-node spans**: Verify `NODE_TRACING_ENABLED` is not set to `false`. Check startup logs for "Per-node execution instrumented".

## License

MIT
