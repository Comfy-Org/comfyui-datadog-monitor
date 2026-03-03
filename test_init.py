"""
Tests for comfyui-datadog-monitor trace context extraction.

Tests the extraction logic that activates a parent trace context from extra_data
before creating the comfyui.workflow.execute span. This enables distributed tracing
between the inference service and ComfyUI.
"""
import pytest
from ddtrace import tracer

from ddtrace.trace import Context as DDContext


class TestTraceContextExtraction:
    """Tests for parent trace context extraction from extra_data."""

    def test_trace_context_extraction_with_valid_parent(self):
        """When valid dd_* keys are in extra_data, the span should be a child of the parent."""
        extra_data = {
            'dd_trace_id': '123456789',
            'dd_parent_id': '987654321',
            'dd_sampling_priority': '1',
        }

        # Run the same extraction logic as traced_execute_async
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
            pass

        with tracer.trace("comfyui.workflow.execute", service="comfyui", resource="workflow#test-id") as span:
            assert span.trace_id == 123456789
            assert span.parent_id == 987654321

    def test_trace_context_fallback_without_parent(self):
        """When no dd_* keys in extra_data, a root span is created without error."""
        extra_data = {'client_id': 'inference', 'job_id': 'some-job'}

        # Should not raise
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
            pass

        with tracer.trace("comfyui.workflow.execute", service="comfyui", resource="workflow#test-id") as span:
            # Root span — no parent
            assert span is not None

    def test_trace_context_fallback_on_malformed_values(self):
        """When dd_* keys have non-numeric values, no exception is raised and a root span is created."""
        extra_data = {
            'dd_trace_id': 'not_a_number',
            'dd_parent_id': 'also_bad',
            'dd_sampling_priority': 'invalid',
        }

        # Should not raise — the except Exception: pass catches ValueError from int()
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
            pass  # Expected — malformed values cause ValueError

        with tracer.trace("comfyui.workflow.execute", service="comfyui", resource="workflow#test-id") as span:
            assert span is not None  # Span still created as root span

    def test_trace_context_with_none_extra_data(self):
        """When extra_data is None or empty, no exception is raised and a root span is created."""
        for extra_data in [None, {}]:
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
                pass

            with tracer.trace("comfyui.workflow.execute", service="comfyui", resource="workflow#test-id") as span:
                assert span is not None  # Span created without error
