"""
ComfyUI Datadog Memory Profiler
Auto-starts DDTrace with full instrumentation when ComfyUI loads this custom node.
"""

import os
import sys

# MUST be imported as the very first thing for full instrumentation
try:
    import ddtrace.auto  # This enables ALL ddtrace features
    print("✅ DDTrace auto-instrumentation enabled (full tracing, profiling, ASM)")
except ImportError:
    print("⚠️ DDTrace not available - install with: pip install ddtrace")
except Exception as e:
    print(f"⚠️ Could not enable DDTrace auto-instrumentation: {e}")

# Additional profiler configuration (already started by ddtrace.auto)
def _configure_ddtrace():
    """Configure DDTrace settings"""
    try:
        from ddtrace import config, tracer
        
        # Set service info if not already set by environment
        if not os.getenv('DD_SERVICE'):
            tracer.set_tags({
                'service': 'comfyui-inference',
                'env': os.getenv('DD_ENV', 'production'),
                'version': os.getenv('DD_VERSION', '1.0.0')
            })
        
        # Enable additional features via config
        # These may already be set by environment variables
        config.analytics_enabled = True  # APM analytics
        
        print("📊 DDTrace configuration applied")
        return True
    except Exception as e:
        print(f"⚠️ Could not configure DDTrace: {e}")
        return False

# Configure on module import
_configured = _configure_ddtrace()

# No UI nodes - this is a background-only extension
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]