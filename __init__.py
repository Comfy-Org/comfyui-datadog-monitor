"""
ComfyUI PyTorch Memory Tracker
Tracks PyTorch CUDA memory allocations for OOM debugging.
Controlled via PYTORCH_MEMORY_TRACKING environment variable.
"""

import os
import functools
import logging

# Import tracer for DataDog APM integration
try:
    from ddtrace import tracer, config
    from ddtrace.runtime import RuntimeMetrics
    RuntimeMetrics.enable()
    DDTRACE_AVAILABLE = True
except ImportError:
    print("⚠️ DDTrace not available - install with: pip install ddtrace")
    DDTRACE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                f'pytorch.{stage}.allocated_bytes': stats.get('allocated_bytes.all.current', 0),
                f'pytorch.{stage}.reserved_bytes': stats.get('reserved_bytes.all.current', 0),
                f'pytorch.{stage}.num_ooms': stats.get('num_ooms', 0),
            }.items():
                span.set_metric(key, value)

            if stage == "after" and stats.get('num_ooms', 0) > 0:
                summary = torch.cuda.memory_summary()
                logger.info(f"PyTorch CUDA Memory Summary:\n{summary.split(chr(10))[:10]}")

        elif torch.backends.mps.is_available():
            allocated = torch.mps.current_allocated_memory()
            driver_allocated = torch.mps.driver_allocated_memory()
            span.set_metric(f'pytorch.{stage}.mps_allocated_bytes', allocated)
            span.set_metric(f'pytorch.{stage}.mps_driver_allocated_bytes', driver_allocated)

    except Exception as e:
        logger.error(f"Could not capture PyTorch memory snapshot: {e}")

def log_top_memory_allocations(span, prompt_id, stage="after", top_n=5):
    """Extract and log top N memory allocations from PyTorch (CUDA VRAM + CPU RAM)"""
    if not PYTORCH_MEMORY_TRACKING_ENABLED:
        return

    try:
        import torch
        import gc
        import traceback as tb

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

        # Track CPU (RAM) tensor allocations
        cpu_allocations = []
        gc.collect()  # Ensure we're looking at current state
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    # Only track CPU tensors (RAM, not VRAM)
                    if not obj.is_cuda and not (hasattr(obj, 'is_mps') and obj.is_mps):
                        size_bytes = obj.element_size() * obj.nelement()

                        # Get type info
                        dtype = str(obj.dtype).replace('torch.', '')
                        shape = 'x'.join(map(str, obj.shape)) if obj.shape else 'scalar'

                        cpu_allocations.append({
                            'size_mb': size_bytes / 1024 / 1024,
                            'size_bytes': size_bytes,
                            'location': 'cpu',
                            'dtype': dtype,
                            'shape': shape,
                            'stack_trace': [f"tensor({dtype}, shape={shape})"]
                        })
            except Exception:
                continue

        # Combine and sort all allocations
        all_allocations = cuda_allocations + cpu_allocations
        if not all_allocations:
            return

        top_allocations = sorted(all_allocations, key=lambda x: x['size_bytes'], reverse=True)[:top_n]

        # Calculate totals by location
        total_cuda = sum(a['size_bytes'] for a in cuda_allocations) / 1024 / 1024
        total_cpu = sum(a['size_bytes'] for a in cpu_allocations) / 1024 / 1024
        total_all = total_cuda + total_cpu

        # Log summary
        logger.info(f"🔍 Top {top_n} PyTorch allocations (CUDA: {total_cuda:.1f} MB, CPU: {total_cpu:.1f} MB, Total: {total_all:.1f} MB):")
        for i, alloc in enumerate(top_allocations, 1):
            location = alloc['location'].upper()
            logger.info(f"  #{i} [{location}]: {alloc['size_mb']:.1f} MB")
            for frame in alloc['stack_trace'][:3]:
                logger.info(f"      {frame}")

        # Tag DataDog span
        if span:
            span.set_metric(f'pytorch.{stage}.num_allocations', len(all_allocations))
            span.set_metric(f'pytorch.{stage}.num_cpu_tensors', len(cpu_allocations))
            span.set_metric(f'pytorch.{stage}.num_cuda_tensors', len(cuda_allocations))
            span.set_metric(f'pytorch.{stage}.total_allocated_mb', total_all)
            span.set_metric(f'pytorch.{stage}.cpu_allocated_mb', total_cpu)
            span.set_metric(f'pytorch.{stage}.cuda_allocated_mb', total_cuda)

            if top_allocations:
                largest = top_allocations[0]
                span.set_metric(f'pytorch.{stage}.largest_allocation_mb', largest['size_mb'])
                span.set_tag(f'pytorch.{stage}.largest_allocation_location', largest['location'])
                if largest['stack_trace']:
                    span.set_tag(f'pytorch.{stage}.largest_allocation_info', largest['stack_trace'][0])

        # Log structured summary
        summary_parts = [
            f"prompt_id={prompt_id}",
            f"stage={stage}",
            f"cpu={total_cpu:.1f}MB",
            f"cuda={total_cuda:.1f}MB",
            f"total={total_all:.1f}MB"
        ]
        for i, alloc in enumerate(top_allocations, 1):
            loc = alloc['location']
            info = alloc['stack_trace'][0] if alloc['stack_trace'] else 'unknown'
            summary_parts.append(f"top{i}={loc}:{alloc['size_mb']:.1f}MB:{info}")

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

                            # Capture PyTorch memory after workflow
                            capture_pytorch_memory_snapshot(span, "after")

                            # Log top memory allocations with stack traces
                            log_top_memory_allocations(span, prompt_id, stage="after", top_n=5)

                            return result

                        except Exception as e:
                            span.set_tag('error', True)
                            span.set_tag('error.type', type(e).__name__)
                            raise

                PromptExecutor.execute_async = traced_execute_async
                print("   ✅ Workflow execution instrumented for PyTorch memory tracking")

        _patched = True
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
