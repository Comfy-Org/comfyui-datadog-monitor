"""
ComfyUI Datadog Memory Profiler Node
Simple pass-through node that reports memory usage.
DDTrace profiling is already enabled via ddtrace.auto in __init__.py
"""

import os
import psutil
import json


class DatadogMemoryProfiler:
    """
    Simple ComfyUI node that reports memory usage.
    Acts as a pass-through node for workflow compatibility.
    DDTrace profiling is already active via ddtrace.auto.
    """
    
    def __init__(self):
        self.process = psutil.Process()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": ("*",),  # Pass-through any data
            }
        }
    
    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("output", "memory_info")
    FUNCTION = "profile_memory"
    CATEGORY = "monitoring"
    
    def profile_memory(self, input_data):
        """Pass-through with basic memory info"""
        mem_info = self.process.memory_info()
        rss_gb = mem_info.rss / (1024**3)
        
        memory_data = {
            "rss_gb": round(rss_gb, 2),
            "vms_gb": round(mem_info.vms / (1024**3), 2),
            "percent": round(self.process.memory_percent(), 1),
            "ddtrace": "auto-instrumented"
        }
        
        # Simple memory status
        print(f"📊 Memory: {memory_data['rss_gb']}GB RSS ({memory_data['percent']}%)")
        
        return (input_data, json.dumps(memory_data, indent=2))


# ComfyUI Node Registration
NODE_CLASS_MAPPINGS = {
    "DatadogMemoryProfiler": DatadogMemoryProfiler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DatadogMemoryProfiler": "Datadog Memory Profiler"
}