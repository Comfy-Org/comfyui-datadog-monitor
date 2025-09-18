"""
ComfyUI Datadog APM Tracer and Profiler
Automatically instruments ComfyUI with comprehensive APM tracing and profiling.
No UI nodes - runs entirely in the background via monkey patching.
"""

import os
import sys
import time
import psutil
import functools
import threading
from typing import Any, Dict, Optional
import logging

# MUST be imported as the very first thing for full instrumentation
try:
    import ddtrace.auto  # This enables ALL ddtrace features
    print("✅ DDTrace auto-instrumentation enabled (full tracing, profiling, ASM)")
    DDTRACE_AVAILABLE = True
except ImportError:
    print("⚠️ DDTrace not available - install with: pip install ddtrace")
    DDTRACE_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Could not enable DDTrace auto-instrumentation: {e}")
    DDTRACE_AVAILABLE = False

# Import tracer for manual instrumentation
if DDTRACE_AVAILABLE:
    from ddtrace import tracer, patch_all, config
    from ddtrace.runtime import RuntimeMetrics
    from ddtrace.profiling import Profiler

    # Enable runtime metrics collection (includes memory metrics)
    RuntimeMetrics.enable()

    # Enable memory and heap profiling via environment variables
    os.environ.setdefault('DD_PROFILING_MEMORY_ENABLED', 'true')
    os.environ.setdefault('DD_PROFILING_HEAP_ENABLED', 'true')

    # Enable profiling for detailed memory analysis
    try:
        profiler = Profiler(
            env=os.getenv('DD_ENV', 'production'),
            service=os.getenv('DD_SERVICE', 'comfyui'),
            version=os.getenv('DD_VERSION', '1.0.0'),
            tags={
                'component': 'comfyui',
                'profiler': 'ddtrace',
            }
        )
        profiler.start()

        # Log profiling configuration
        mem_enabled = os.getenv('DD_PROFILING_MEMORY_ENABLED', 'false').lower() == 'true'
        heap_enabled = os.getenv('DD_PROFILING_HEAP_ENABLED', 'false').lower() == 'true'
        print(f"🔬 DDTrace Profiler started (Memory: {mem_enabled}, Heap: {heap_enabled})")
    except Exception as e:
        print(f"⚠️ Could not start profiler: {e}")

    # Patch all available integrations
    patch_all()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state tracking
_patched = False
_execution_stats = {
    'workflows_executed': 0,
    'nodes_executed': 0,
    'total_execution_time': 0.0,
    'errors_tracked': 0
}

def _configure_ddtrace():
    """Configure DDTrace settings"""
    if not DDTRACE_AVAILABLE:
        return False

    try:
        # CRITICAL: Start the writer service if it's stopped
        if hasattr(tracer, '_writer') and tracer._writer.status.name == 'STOPPED':
            tracer._writer.start()
            print("🚀 DDTrace writer started")

        # Set service info if not already set by environment
        service = os.getenv('DD_SERVICE', 'comfyui')
        env = os.getenv('DD_ENV', 'production')
        version = os.getenv('DD_VERSION', '1.0.0')

        tracer.set_tags({
            'service': service,
            'env': env,
            'version': version
        })

        # Enable additional features via config
        config.analytics_enabled = True  # APM analytics

        # Log configuration
        agent_url = getattr(tracer._writer, 'agent_url', 'unknown') if hasattr(tracer, '_writer') else 'unknown'
        print(f"📊 DDTrace configuration applied")
        print(f"   Service: {service}, Env: {env}, Version: {version}")
        print(f"   Agent URL: {agent_url}")
        print(f"   Writer Status: {tracer._writer.status.name if hasattr(tracer, '_writer') else 'unknown'}")

        return True
    except Exception as e:
        print(f"⚠️ Could not configure DDTrace: {e}")
        return False

def monkey_patch_comfyui():
    """Monkey patch ComfyUI execution to add comprehensive tracing"""
    global _patched

    if _patched:
        logger.info("ComfyUI already instrumented, skipping")
        return

    if not DDTRACE_AVAILABLE:
        logger.info("DDTrace not available, skipping ComfyUI instrumentation")
        return

    try:
        # Import execution module
        import execution

        print("🔧 Instrumenting ComfyUI execution...")

        # Patch node execution
        if hasattr(execution, 'execute'):
            original_execute = execution.execute

            @functools.wraps(original_execute)
            async def traced_execute(server, dynprompt, caches, current_item, extra_data, executed, prompt_id, execution_list, pending_subgraph_results, pending_async_nodes):
                """Traced version of ComfyUI node execution"""
                global _execution_stats

                unique_id = current_item

                # Get node information safely
                try:
                    real_node_id = dynprompt.get_real_node_id(unique_id)
                    display_node_id = dynprompt.get_display_node_id(unique_id)
                    node_data = dynprompt.get_node(unique_id)
                    class_type = node_data.get('class_type', 'unknown')
                except:
                    real_node_id = unique_id
                    display_node_id = unique_id
                    class_type = 'unknown'

                # Create span for this node execution
                span_name = f"comfyui.node.execute"
                with tracer.trace(
                    span_name,
                    service="comfyui",
                    resource=f"{class_type}#{display_node_id}"
                ) as span:
                    # Add node metadata
                    span.set_tags({
                        'node.id': unique_id,
                        'node.display_id': display_node_id,
                        'node.real_id': real_node_id,
                        'node.class_type': class_type,
                        'prompt.id': prompt_id,
                        'job.id': extra_data.get('job_id') if extra_data else None,
                        'client.id': extra_data.get('client_id') if extra_data else None,
                    })

                    # Track comprehensive memory metrics before execution
                    process = psutil.Process()
                    mem_info_before = process.memory_info()
                    mem_before = mem_info_before.rss / 1024 / 1024  # MB
                    vms_before = mem_info_before.vms / 1024 / 1024  # Virtual memory

                    # Track system memory availability
                    sys_mem = psutil.virtual_memory()
                    sys_available_before = sys_mem.available / 1024 / 1024  # MB
                    sys_percent_before = sys_mem.percent

                    cpu_percent_before = process.cpu_percent(interval=0)

                    # Execute the node
                    start_time = time.time()
                    try:
                        result = await original_execute(
                            server, dynprompt, caches, current_item, extra_data,
                            executed, prompt_id, execution_list,
                            pending_subgraph_results, pending_async_nodes
                        )

                        # Track success
                        span.set_tag('node.status', 'success')
                        _execution_stats['nodes_executed'] += 1

                        return result

                    except Exception as e:
                        # Track error
                        span.set_tag('node.status', 'error')
                        span.set_tag('error', True)
                        span.set_tag('error.type', type(e).__name__)
                        span.set_tag('error.message', str(e))
                        _execution_stats['errors_tracked'] += 1
                        raise

                    finally:
                        # Track comprehensive memory and execution metrics
                        execution_time = time.time() - start_time
                        mem_info_after = process.memory_info()
                        mem_after = mem_info_after.rss / 1024 / 1024  # MB
                        vms_after = mem_info_after.vms / 1024 / 1024  # Virtual memory

                        # System memory after
                        sys_mem_after = psutil.virtual_memory()
                        sys_available_after = sys_mem_after.available / 1024 / 1024
                        sys_percent_after = sys_mem_after.percent

                        cpu_percent_after = process.cpu_percent(interval=0)

                        span.set_metrics({
                            'node.execution_time_seconds': execution_time,
                            # Process memory metrics
                            'node.memory_rss_before_mb': mem_before,
                            'node.memory_rss_after_mb': mem_after,
                            'node.memory_rss_delta_mb': mem_after - mem_before,
                            'node.memory_vms_before_mb': vms_before,
                            'node.memory_vms_after_mb': vms_after,
                            'node.memory_vms_delta_mb': vms_after - vms_before,
                            # System memory metrics
                            'system.memory_available_before_mb': sys_available_before,
                            'system.memory_available_after_mb': sys_available_after,
                            'system.memory_available_delta_mb': sys_available_after - sys_available_before,
                            'system.memory_percent_before': sys_percent_before,
                            'system.memory_percent_after': sys_percent_after,
                            # CPU metrics
                            'node.cpu_percent': cpu_percent_after,
                        })

                        # Track GPU if available
                        try:
                            import torch
                            if torch.cuda.is_available():
                                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                                gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
                                span.set_metrics({
                                    'node.gpu_memory_allocated_mb': gpu_memory,
                                    'node.gpu_memory_reserved_mb': gpu_reserved,
                                })
                                if torch.cuda.device_count() > 0:
                                    span.set_tag('gpu.device_name', torch.cuda.get_device_name(0))
                        except:
                            pass

                        _execution_stats['total_execution_time'] += execution_time

            # Replace the original function
            execution.execute = traced_execute
            print("   ✅ Node execution instrumented")

        # Patch workflow execution
        if hasattr(execution, 'PromptExecutor'):
            PromptExecutor = execution.PromptExecutor

            if hasattr(PromptExecutor, 'execute_async'):
                original_execute_async = PromptExecutor.execute_async

                @functools.wraps(original_execute_async)
                async def traced_execute_async(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
                    """Traced version of workflow execution"""
                    global _execution_stats

                    # Create main workflow span
                    with tracer.trace(
                        "comfyui.workflow.execute",
                        service="comfyui",
                        resource=f"workflow#{prompt_id}"
                    ) as span:
                        # Add workflow metadata
                        job_id = extra_data.get('job_id') if extra_data else None
                        client_id = extra_data.get('client_id') if extra_data else None

                        span.set_tags({
                            'workflow.prompt_id': prompt_id,
                            'workflow.client_id': client_id,
                            'job.id': job_id,
                            'workflow.node_count': len(prompt) if prompt else 0,
                        })

                        # Track comprehensive workflow execution metrics
                        start_time = time.time()
                        process = psutil.Process()
                        mem_info_before = process.memory_info()
                        mem_before = mem_info_before.rss / 1024 / 1024  # MB
                        vms_before = mem_info_before.vms / 1024 / 1024  # Virtual memory

                        # Track system memory availability
                        sys_mem = psutil.virtual_memory()
                        sys_available_before = sys_mem.available / 1024 / 1024  # MB
                        sys_percent_before = sys_mem.percent

                        cpu_percent_before = process.cpu_percent(interval=0)

                        try:
                            result = await original_execute_async(self, prompt, prompt_id, extra_data, execute_outputs)
                            span.set_tag('workflow.status', 'success')
                            _execution_stats['workflows_executed'] += 1
                            return result

                        except Exception as e:
                            span.set_tag('workflow.status', 'error')
                            span.set_tag('error', True)
                            span.set_tag('error.type', type(e).__name__)
                            span.set_tag('error.message', str(e))
                            _execution_stats['errors_tracked'] += 1
                            raise

                        finally:
                            # Track comprehensive workflow metrics
                            execution_time = time.time() - start_time
                            mem_info_after = process.memory_info()
                            mem_after = mem_info_after.rss / 1024 / 1024  # MB
                            vms_after = mem_info_after.vms / 1024 / 1024  # Virtual memory

                            # System memory after
                            sys_mem_after = psutil.virtual_memory()
                            sys_available_after = sys_mem_after.available / 1024 / 1024
                            sys_percent_after = sys_mem_after.percent

                            cpu_percent_after = process.cpu_percent(interval=0)

                            span.set_metrics({
                                'workflow.execution_time_seconds': execution_time,
                                # Process memory metrics
                                'workflow.memory_rss_before_mb': mem_before,
                                'workflow.memory_rss_after_mb': mem_after,
                                'workflow.memory_rss_delta_mb': mem_after - mem_before,
                                'workflow.memory_vms_before_mb': vms_before,
                                'workflow.memory_vms_after_mb': vms_after,
                                'workflow.memory_vms_delta_mb': vms_after - vms_before,
                                # System memory metrics
                                'system.memory_available_before_mb': sys_available_before,
                                'system.memory_available_after_mb': sys_available_after,
                                'system.memory_available_delta_mb': sys_available_after - sys_available_before,
                                'system.memory_percent_before': sys_percent_before,
                                'system.memory_percent_after': sys_percent_after,
                                # CPU metrics
                                'workflow.cpu_percent': cpu_percent_after,
                            })

                            # Track GPU for workflow
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    span.set_metrics({
                                        'workflow.gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                                        'workflow.gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                                    })
                            except:
                                pass

                            _execution_stats['total_execution_time'] += execution_time

                # Replace the method
                PromptExecutor.execute_async = traced_execute_async
                print("   ✅ Workflow execution instrumented")

        # Also instrument model loading if available
        try:
            import comfy.model_management as model_management

            if hasattr(model_management, 'load_models_gpu'):
                original_load_models = model_management.load_models_gpu

                @functools.wraps(original_load_models)
                def traced_load_models(*args, **kwargs):
                    """Trace model loading operations"""
                    with tracer.trace(
                        "comfyui.model.load",
                        service="comfyui",
                        resource="load_models_gpu"
                    ) as span:
                        # Track model info if available
                        if args and hasattr(args[0], '__len__'):
                            span.set_tag('model.count', len(args[0]))

                        process = psutil.Process()
                        mem_before = process.memory_info().rss / 1024 / 1024

                        start_time = time.time()
                        try:
                            result = original_load_models(*args, **kwargs)
                            span.set_tag('model.load.status', 'success')
                            return result
                        except Exception as e:
                            span.set_tag('model.load.status', 'error')
                            span.set_tag('error', True)
                            span.set_tag('error.type', type(e).__name__)
                            raise
                        finally:
                            execution_time = time.time() - start_time
                            mem_after = process.memory_info().rss / 1024 / 1024

                            span.set_metrics({
                                'model.load.time_seconds': execution_time,
                                'model.load.memory_delta_mb': mem_after - mem_before,
                            })

                            # Track GPU memory after model load
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    span.set_metrics({
                                        'model.gpu_memory_after_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                                    })
                            except:
                                pass

                model_management.load_models_gpu = traced_load_models
                print("   ✅ Model loading instrumented")
        except Exception as e:
            logger.debug(f"Could not instrument model loading: {e}")

        _patched = True
        print("🎉 ComfyUI fully instrumented with DDTrace APM!")

        # Log stats periodically
        def log_stats():
            while True:
                time.sleep(60)  # Log every minute
                if _execution_stats['workflows_executed'] > 0:
                    logger.info(f"APM Stats: Workflows={_execution_stats['workflows_executed']}, "
                               f"Nodes={_execution_stats['nodes_executed']}, "
                               f"Total Time={_execution_stats['total_execution_time']:.2f}s, "
                               f"Errors={_execution_stats['errors_tracked']}")

        stats_thread = threading.Thread(target=log_stats, daemon=True)
        stats_thread.start()

    except ImportError as e:
        logger.warning(f"Could not import execution module: {e}")
    except Exception as e:
        logger.error(f"Failed to instrument ComfyUI: {e}")
        import traceback
        traceback.print_exc()

# Configure and patch on module import
if DDTRACE_AVAILABLE:
    _configured = _configure_ddtrace()
    monkey_patch_comfyui()
else:
    print("⚠️ Skipping instrumentation - ddtrace not available")

# No UI nodes - this is a background-only extension
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]