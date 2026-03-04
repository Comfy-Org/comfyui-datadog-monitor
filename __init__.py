"""
ComfyUI Datadog Monitor
Adds Datadog APM tracing and CUDA memory tracking to ComfyUI workflow execution.
Includes per-node tracing spans with class_type and optional memory metrics.
Controlled via environment variables (see README).
"""

import os
import functools
import inspect
import logging


# Enable CPU profiling by default (sampling-based, ~2% overhead, no GPU impact).
# Can be disabled with DD_PROFILING_ENABLED=false if needed.
os.environ.setdefault('DD_PROFILING_ENABLED', 'true')

# Enable automatic instrumentation for all supported libraries (requests, asyncio,
# subprocess, logging, etc.) via ddtrace.auto. This uses import hooks so it works
# even though libraries may already be imported by the time this extension loads.
# See: https://ddtrace.readthedocs.io/en/stable/installation_quickstart.html
try:
    import ddtrace.auto  # noqa: F401
    from ddtrace import tracer, config
    from ddtrace.runtime import RuntimeMetrics
    from ddtrace.trace import Context as DDContext
    RuntimeMetrics.enable()
    DDTRACE_AVAILABLE = True
except ImportError:
    print("⚠️ DDTrace not available - install with: pip install ddtrace")
    DDTRACE_AVAILABLE = False

# Configure module-specific logger (don't touch root logger)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# PyTorch Memory Tracking Configuration (workflow-level, expensive — opt-in)
PYTORCH_MEMORY_TRACKING_ENABLED = os.getenv('PYTORCH_MEMORY_TRACKING', '').lower() == 'true'

# Per-node tracing — enabled by default when ddtrace is available.
# Creates a child span for each node execution under the workflow span.
# Disable with NODE_TRACING_ENABLED=false if the extra spans are unwanted.
NODE_TRACING_ENABLED = os.getenv('NODE_TRACING_ENABLED', 'true').lower() != 'false'

# Per-node memory capture — cheap VRAM/RAM snapshots on every node span.
# Enabled by default. Uses torch.cuda.memory_allocated() (~0.1ms) and
# psutil.virtual_memory() (~0.1ms), NOT the expensive memory_stats() call.
# Disable with NODE_MEMORY_TRACKING=false if even the minimal overhead is unwanted.
NODE_MEMORY_TRACKING_ENABLED = os.getenv('NODE_MEMORY_TRACKING', 'true').lower() != 'false'

if PYTORCH_MEMORY_TRACKING_ENABLED:
    print("🧠 PyTorch memory tracking enabled")

def enable_pytorch_memory_tracking():
    """Enable PyTorch's native memory tracking"""
    if not PYTORCH_MEMORY_TRACKING_ENABLED:
        return False

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.memory._record_memory_history(enabled=True)
            print("🧠 PyTorch CUDA memory history tracking enabled")
            return True
        elif torch.backends.mps.is_available():
            print("🧠 PyTorch MPS device detected - basic memory tracking available")
            return True
    except Exception as e:
        logger.debug(f"Could not enable PyTorch memory tracking: {e}")
    return False

def capture_pytorch_memory_snapshot(span, stage=""):
    """Capture PyTorch memory stats and send to DataDog"""
    if not PYTORCH_MEMORY_TRACKING_ENABLED:
        return

    try:
        import torch

        if torch.cuda.is_available():
            stats = torch.cuda.memory_stats()
            for key, value in {
                f'memory.pytorch.allocated_bytes.{stage}': stats.get('allocated_bytes.all.current', 0),
                f'memory.pytorch.reserved_bytes.{stage}': stats.get('reserved_bytes.all.current', 0),
                f'memory.pytorch.num_ooms.{stage}': stats.get('num_ooms', 0),
            }.items():
                span.set_metric(key, value)

            if stage == "after" and stats.get('num_ooms', 0) > 0:
                summary = torch.cuda.memory_summary()
                logger.info(f"PyTorch CUDA Memory Summary:\n{summary.split(chr(10))[:10]}")

        elif torch.backends.mps.is_available():
            allocated = torch.mps.current_allocated_memory()
            driver_allocated = torch.mps.driver_allocated_memory()
            span.set_metric(f'memory.pytorch.mps_allocated_bytes.{stage}', allocated)
            span.set_metric(f'memory.pytorch.mps_driver_allocated_bytes.{stage}', driver_allocated)

    except Exception as e:
        logger.error(f"Could not capture PyTorch memory snapshot: {e}")

def capture_node_memory_snapshot(span, stage):
    """Capture cheap VRAM/RAM metrics for a per-node span.

    Uses only lightweight calls (~0.1ms each):
    - torch.cuda.memory_allocated() / memory_reserved() — no GPU sync required
    - psutil.virtual_memory().available — kernel-level, very fast

    Does NOT call the expensive torch.cuda.memory_stats() or memory._snapshot().
    """
    if not NODE_MEMORY_TRACKING_ENABLED:
        return

    try:
        import torch

        if torch.cuda.is_available():
            span.set_metric(f'memory.vram_allocated_bytes.{stage}', torch.cuda.memory_allocated())
            span.set_metric(f'memory.vram_reserved_bytes.{stage}', torch.cuda.memory_reserved())
        elif torch.backends.mps.is_available():
            span.set_metric(f'memory.mps_allocated_bytes.{stage}', torch.mps.current_allocated_memory())
    except Exception:
        pass  # torch not available or no GPU — skip silently

    try:
        import psutil
        span.set_metric(f'memory.ram_available_bytes.{stage}', psutil.virtual_memory().available)
    except Exception:
        pass

def log_top_memory_allocations(span, prompt_id, stage="after", top_n=5):
    """Extract and log top N CUDA VRAM allocations from PyTorch memory snapshot"""
    if not PYTORCH_MEMORY_TRACKING_ENABLED:
        return

    try:
        import torch

        # Track CUDA (VRAM) allocations with stack traces
        cuda_allocations = []
        if torch.cuda.is_available():
            snapshot = torch.cuda.memory._snapshot()
            if snapshot and 'segments' in snapshot:
                for segment in snapshot.get('segments', []):
                    for block in segment.get('blocks', []):
                        if block.get('state') == 'active_allocated':
                            size_bytes = block.get('size', 0)
                            frames = block.get('frames', [])

                            stack_trace = []
                            for frame in frames[:5]:
                                filename = frame.get('filename', 'unknown')
                                line = frame.get('line', 0)
                                name = frame.get('name', 'unknown')
                                # Shorten paths
                                if 'site-packages' in filename:
                                    filename = '...' + filename.split('site-packages')[-1]
                                elif 'comfyui' in filename.lower():
                                    filename = '...' + filename.split('comfyui')[-1]
                                stack_trace.append(f"{filename}:{line} in {name}")

                            cuda_allocations.append({
                                'size_mb': size_bytes / 1024 / 1024,
                                'size_bytes': size_bytes,
                                'location': 'cuda',
                                'stack_trace': stack_trace
                            })

        if not cuda_allocations:
            return

        top_allocations = sorted(cuda_allocations, key=lambda x: x['size_bytes'], reverse=True)[:top_n]

        # Calculate totals
        total_cuda = sum(a['size_bytes'] for a in cuda_allocations) / 1024 / 1024

        # Log summary
        logger.info(f"Top {top_n} CUDA allocations ({total_cuda:.1f} MB total):")
        for i, alloc in enumerate(top_allocations, 1):
            logger.info(f"  #{i}: {alloc['size_mb']:.1f} MB")
            for frame in alloc['stack_trace'][:3]:
                logger.info(f"      {frame}")

        # Tag DataDog span
        if span:
            span.set_metric(f'memory.pytorch.num_cuda_allocations.{stage}', len(cuda_allocations))
            span.set_metric(f'memory.pytorch.cuda_mb.{stage}', total_cuda)

            if top_allocations:
                largest = top_allocations[0]
                span.set_metric(f'memory.pytorch.largest_mb.{stage}', largest['size_mb'])
                if largest['stack_trace']:
                    span.set_tag(f'memory.pytorch.largest_info.{stage}', largest['stack_trace'][0])

        # Log structured summary
        summary_parts = [
            f"prompt_id={prompt_id}",
            f"stage={stage}",
            f"cuda={total_cuda:.1f}MB"
        ]
        for i, alloc in enumerate(top_allocations, 1):
            info = alloc['stack_trace'][0] if alloc['stack_trace'] else 'unknown'
            summary_parts.append(f"top{i}=cuda:{alloc['size_mb']:.1f}MB:{info}")

        logger.info(f"pytorch_memory_allocations: {' '.join(summary_parts)}")

    except Exception as e:
        logger.error(f"Could not log top memory allocations: {e}")

def _extract_param(sig, args, kwargs, name, default=None):
    """Extract a named parameter from *args/**kwargs using a pre-computed signature.

    Uses inspect.signature binding to resolve positional and keyword arguments
    by name, regardless of position. Returns `default` if the parameter doesn't
    exist in the signature or wasn't provided — making wrappers resilient to
    upstream signature changes.

    Called once per invocation with ~0 overhead (dict lookup, no reflection).
    The `sig` object should be obtained once at patch time via inspect.signature().
    """
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound.arguments.get(name, default)
    except TypeError:
        # Signature mismatch (e.g. upstream added required params we don't know about).
        # Fall back gracefully rather than crashing the wrapper.
        return default


# Global state
_patched = False

def _configure_ddtrace():
    """Configure DDTrace settings"""
    if not DDTRACE_AVAILABLE:
        return False

    try:
        if hasattr(tracer, '_writer') and tracer._writer.status.name == 'STOPPED':
            tracer._writer.start()

        service = os.getenv('DD_SERVICE', 'comfyui')
        env = os.getenv('DD_ENV', 'production')

        tracer.set_tags({'service': service, 'env': env})

        print(f"📊 DDTrace configured: {service} ({env})")
        return True
    except Exception as e:
        print(f"⚠️ Could not configure DDTrace: {e}")
        return False

def monkey_patch_comfyui():
    """Patch ComfyUI workflow and node execution to add tracing and memory tracking"""
    global _patched

    if _patched:
        return

    _patched = True  # Don't retry regardless of outcome

    if not DDTRACE_AVAILABLE:
        logger.info("DDTrace not available, skipping instrumentation")
        return

    try:
        import execution

        print("🔧 Instrumenting ComfyUI...")

        # Patch workflow execution
        if hasattr(execution, 'PromptExecutor'):
            PromptExecutor = execution.PromptExecutor

            if hasattr(PromptExecutor, 'execute_async'):
                original_execute_async = PromptExecutor.execute_async
                _ea_sig = inspect.signature(original_execute_async)

                @functools.wraps(original_execute_async)
                async def traced_execute_async(self, *args, **kwargs):
                    """Traced version of workflow execution with PyTorch memory tracking.

                    Uses *args/**kwargs to forward all parameters transparently,
                    making this wrapper resilient to upstream signature changes.
                    Parameters we need are extracted by name via inspect.signature binding.
                    """
                    # Extract the parameters we need by name (position-independent)
                    prompt_id = _extract_param(_ea_sig, (self, *args), kwargs, 'prompt_id')
                    extra_data = _extract_param(_ea_sig, (self, *args), kwargs, 'extra_data', {})

                    # Activate parent trace context from inference service (if present).
                    # This makes comfyui.workflow.execute a child span of the inference service's span.
                    # Falls back silently to a root span on any error (missing keys, wrong types, etc.)
                    try:
                        dd_trace_id = extra_data.get('dd_trace_id') if extra_data else None
                        dd_parent_id = extra_data.get('dd_parent_id') if extra_data else None
                        dd_sampling_priority = extra_data.get('dd_sampling_priority') if extra_data else None
                        if dd_trace_id and dd_parent_id:
                            parent_ctx = DDContext(
                                trace_id=int(dd_trace_id),
                                span_id=int(dd_parent_id),
                                sampling_priority=int(dd_sampling_priority) if dd_sampling_priority else None,
                            )
                            tracer.context_provider.activate(parent_ctx)
                    except Exception:
                        pass  # Fall back to root span on any error

                    with tracer.trace(
                        "comfyui.workflow.execute",
                        service="comfyui",
                        resource=f"workflow#{prompt_id}"
                    ) as span:
                        job_id = extra_data.get('job_id') if extra_data else None

                        span.set_tags({
                            'workflow.prompt_id': prompt_id,
                            'job.id': job_id,
                        })

                        # Capture PyTorch memory before workflow
                        capture_pytorch_memory_snapshot(span, "before")

                        try:
                            result = await original_execute_async(self, *args, **kwargs)
                            return result

                        except Exception as e:
                            span.set_tag('error', True)
                            span.set_tag('error.type', type(e).__name__)
                            raise

                        finally:
                            # Always capture memory state — especially valuable on OOM
                            capture_pytorch_memory_snapshot(span, "after")
                            log_top_memory_allocations(span, prompt_id, stage="after", top_n=5)

                PromptExecutor.execute_async = traced_execute_async
                print("   ✅ Workflow execution instrumented")
            else:
                print("   ⚠️ PromptExecutor.execute_async not found, skipping workflow instrumentation")
        else:
            print("   ⚠️ PromptExecutor not found, skipping workflow instrumentation")

        # Patch per-node execution
        if NODE_TRACING_ENABLED and hasattr(execution, 'execute'):
            original_execute = execution.execute
            _ex_sig = inspect.signature(original_execute)

            @functools.wraps(original_execute)
            async def traced_execute(*args, **kwargs):
                """Traced version of per-node execution with memory tracking.

                Uses *args/**kwargs to forward all parameters transparently,
                making this wrapper resilient to upstream signature changes.
                Parameters we need are extracted by name via inspect.signature binding.

                Creates a child span under comfyui.workflow.execute for each node,
                tagged with class_type, node_id, and optional VRAM/RAM snapshots.
                Cached nodes get a short span tagged with node.cached=true.
                """
                # Extract only the parameters we need (position-independent)
                current_item = _extract_param(_ex_sig, args, kwargs, 'current_item')
                dynprompt = _extract_param(_ex_sig, args, kwargs, 'dynprompt')
                caches = _extract_param(_ex_sig, args, kwargs, 'caches')
                prompt_id = _extract_param(_ex_sig, args, kwargs, 'prompt_id')

                unique_id = current_item
                try:
                    class_type = dynprompt.get_node(unique_id).get('class_type', 'unknown')
                except Exception:
                    class_type = 'unknown'

                with tracer.trace(
                    "comfyui.node.execute",
                    service="comfyui",
                    resource=class_type,
                ) as span:
                    span.set_tag('node.id', str(unique_id))
                    span.set_tag('node.class_type', class_type)
                    span.set_tag('workflow.prompt_id', prompt_id)

                    # Check if this node is cached (will return early from original_execute)
                    is_cached = caches.outputs.get(unique_id) is not None if caches else False
                    if is_cached:
                        span.set_tag('node.cached', True)

                    # Capture cheap memory snapshot before node execution
                    capture_node_memory_snapshot(span, "before")

                    try:
                        result = await original_execute(*args, **kwargs)

                        # Tag the result status
                        if result and len(result) >= 1:
                            span.set_tag('node.result', result[0].name if hasattr(result[0], 'name') else str(result[0]))
                            if hasattr(result[0], 'name') and result[0].name == 'FAILURE':
                                span.set_tag('error', True)

                        return result

                    except Exception as e:
                        span.set_tag('error', True)
                        span.set_tag('error.type', type(e).__name__)
                        span.set_tag('error.message', str(e)[:500])
                        raise

                    finally:
                        # Capture cheap memory snapshot after node execution
                        capture_node_memory_snapshot(span, "after")

            execution.execute = traced_execute
            mem_status = "with memory tracking" if NODE_MEMORY_TRACKING_ENABLED else "without memory tracking"
            print(f"   ✅ Per-node execution instrumented ({mem_status})")
        elif not NODE_TRACING_ENABLED:
            print("   ℹ️ Per-node tracing disabled (NODE_TRACING_ENABLED=false)")
        else:
            print("   ⚠️ execution.execute not found, skipping per-node instrumentation")

        print("🎉 ComfyUI instrumentation complete!")

    except ImportError as e:
        logger.warning(f"Could not import execution module: {e}")
    except Exception as e:
        logger.error(f"Failed to instrument ComfyUI: {e}")

# Configure and patch on module import
if DDTRACE_AVAILABLE:
    _configure_ddtrace()
    enable_pytorch_memory_tracking()
    monkey_patch_comfyui()
else:
    print("⚠️ Skipping instrumentation - ddtrace not available")

# No UI nodes - this is a background-only extension
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
