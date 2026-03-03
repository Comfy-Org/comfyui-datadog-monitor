"""
ComfyUI Datadog Monitor
Adds Datadog APM tracing and CUDA memory tracking to ComfyUI workflow execution.
Controlled via environment variables (see README).
"""

import os
import functools
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
    try:
        from ddtrace.context import Context as DDContext
    except ImportError:
        from ddtrace._trace.context import Context as DDContext
    RuntimeMetrics.enable()
    DDTRACE_AVAILABLE = True
except ImportError:
    print("⚠️ DDTrace not available - install with: pip install ddtrace")
    DDTRACE_AVAILABLE = False

# Configure module-specific logger (don't touch root logger)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# PyTorch Memory Tracking Configuration
PYTORCH_MEMORY_TRACKING_ENABLED = os.getenv('PYTORCH_MEMORY_TRACKING', '').lower() == 'true'

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
        config.analytics_enabled = True

        print(f"📊 DDTrace configured: {service} ({env})")
        return True
    except Exception as e:
        print(f"⚠️ Could not configure DDTrace: {e}")
        return False

def monkey_patch_comfyui():
    """Patch ComfyUI workflow execution to add PyTorch memory tracking"""
    global _patched

    if _patched:
        return

    _patched = True  # Don't retry regardless of outcome

    if not DDTRACE_AVAILABLE:
        logger.info("DDTrace not available, skipping instrumentation")
        return

    try:
        import execution

        print("🔧 Instrumenting ComfyUI for PyTorch memory tracking...")

        # Patch workflow execution
        if hasattr(execution, 'PromptExecutor'):
            PromptExecutor = execution.PromptExecutor

            if hasattr(PromptExecutor, 'execute_async'):
                original_execute_async = PromptExecutor.execute_async

                @functools.wraps(original_execute_async)
                async def traced_execute_async(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
                    """Traced version of workflow execution with PyTorch memory tracking"""
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
                            result = await original_execute_async(self, prompt, prompt_id, extra_data, execute_outputs)
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
                print("   ✅ Workflow execution instrumented for PyTorch memory tracking")
            else:
                print("   ⚠️ PromptExecutor.execute_async not found, skipping instrumentation")
        else:
            print("   ⚠️ PromptExecutor not found, skipping instrumentation")

        print("🎉 ComfyUI PyTorch memory tracking enabled!")

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
