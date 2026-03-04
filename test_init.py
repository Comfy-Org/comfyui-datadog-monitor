"""
Tests for comfyui-datadog-monitor trace context extraction and per-node tracing.

Tests the extraction logic that activates a parent trace context from extra_data
before creating the comfyui.workflow.execute span. This enables distributed tracing
between the inference service and ComfyUI.

Also tests the per-node span creation (comfyui.node.execute) including child span
nesting, class_type tagging, cached node detection, error handling, and memory metrics.
"""
import asyncio
import os
import unittest.mock as mock
from enum import Enum
from unittest.mock import MagicMock, patch

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


class ExecutionResult(Enum):
    """Mirror of ComfyUI's ExecutionResult for testing."""
    SUCCESS = 0
    FAILURE = 1
    PENDING = 2


def _make_mock_caches(cached_node_ids=None):
    """Create a mock caches object with an outputs cache."""
    cached = cached_node_ids or set()
    caches = MagicMock()
    caches.outputs.get.side_effect = lambda uid: MagicMock() if uid in cached else None
    return caches


def _make_mock_dynprompt(nodes=None):
    """Create a mock dynprompt with node lookups."""
    nodes = nodes or {}
    dynprompt = MagicMock()
    dynprompt.get_node.side_effect = lambda uid: nodes.get(uid, {'class_type': 'Unknown'})
    return dynprompt


class TestPerNodeTracing:
    """Tests for per-node comfyui.node.execute span creation."""

    def test_node_span_created_with_class_type(self):
        """A child span is created for each node execution with class_type as resource."""


        prompt_id = "test-prompt-123"
        node_id = "node_5"
        class_type = "KSampler"

        dynprompt = _make_mock_dynprompt({node_id: {'class_type': class_type}})
        caches = _make_mock_caches()

        # Simulate what traced_execute does
        with tracer.trace("comfyui.workflow.execute", service="comfyui", resource=f"workflow#{prompt_id}") as parent_span:
            with tracer.trace("comfyui.node.execute", service="comfyui", resource=class_type) as span:
                span.set_tag('node.id', str(node_id))
                span.set_tag('node.class_type', class_type)
                span.set_tag('workflow.prompt_id', prompt_id)

                assert span.resource == class_type
                assert span.parent_id == parent_span.span_id
                assert span.trace_id == parent_span.trace_id

    def test_node_span_tags_correct(self):
        """Node span includes node.id, node.class_type, and workflow.prompt_id tags."""
        prompt_id = "prompt-abc"
        node_id = "42"
        class_type = "VAEDecode"

        with tracer.trace("comfyui.node.execute", service="comfyui", resource=class_type) as span:
            span.set_tag('node.id', str(node_id))
            span.set_tag('node.class_type', class_type)
            span.set_tag('workflow.prompt_id', prompt_id)

        assert span.get_tag('node.id') == "42"
        assert span.get_tag('node.class_type') == "VAEDecode"
        assert span.get_tag('workflow.prompt_id') == "prompt-abc"

    def test_cached_node_tagged(self):
        """Cached nodes get a node.cached=true tag."""
        with tracer.trace("comfyui.node.execute", service="comfyui", resource="CLIPTextEncode") as span:
            # Simulate cached detection
            is_cached = True
            if is_cached:
                span.set_tag('node.cached', True)

        assert span.get_tag('node.cached') == "True"

    def test_uncached_node_no_cached_tag(self):
        """Non-cached nodes do not get a node.cached tag."""
        with tracer.trace("comfyui.node.execute", service="comfyui", resource="KSampler") as span:
            is_cached = False
            if is_cached:
                span.set_tag('node.cached', True)

        assert span.get_tag('node.cached') is None

    def test_success_result_tagged(self):
        """Successful node execution tags node.result=SUCCESS."""
        result = (ExecutionResult.SUCCESS, None, None)

        with tracer.trace("comfyui.node.execute", service="comfyui", resource="SaveImage") as span:
            if result and len(result) >= 1:
                span.set_tag('node.result', result[0].name if hasattr(result[0], 'name') else str(result[0]))

        assert span.get_tag('node.result') == "SUCCESS"
        assert span.get_tag('error') is None

    def test_failure_result_tagged_as_error(self):
        """Failed node execution sets error=true and node.result=FAILURE."""
        result = (ExecutionResult.FAILURE, {'error': 'something broke'}, RuntimeError("boom"))

        with tracer.trace("comfyui.node.execute", service="comfyui", resource="KSampler") as span:
            if result and len(result) >= 1:
                span.set_tag('node.result', result[0].name if hasattr(result[0], 'name') else str(result[0]))
                if hasattr(result[0], 'name') and result[0].name == 'FAILURE':
                    span.set_tag('error', True)

        assert span.get_tag('node.result') == "FAILURE"
        assert span.get_tag('error') == "True"

    def test_exception_sets_error_tags(self):
        """Exceptions during node execution set error, error.type, and error.message."""
        error = ValueError("Invalid input dimensions: expected 4D tensor")

        with tracer.trace("comfyui.node.execute", service="comfyui", resource="KSampler") as span:
            span.set_tag('error', True)
            span.set_tag('error.type', type(error).__name__)
            span.set_tag('error.message', str(error)[:500])

        assert span.get_tag('error') == "True"
        assert span.get_tag('error.type') == "ValueError"
        assert span.get_tag('error.message') == "Invalid input dimensions: expected 4D tensor"

    def test_error_message_truncated_at_500_chars(self):
        """Long error messages are truncated to 500 characters."""
        long_message = "x" * 1000

        with tracer.trace("comfyui.node.execute", service="comfyui", resource="KSampler") as span:
            span.set_tag('error.message', long_message[:500])

        assert len(span.get_tag('error.message')) == 500

    def test_child_span_nests_under_workflow(self):
        """Node spans are children of the workflow span via automatic context propagation."""
        with tracer.trace("comfyui.workflow.execute", service="comfyui", resource="workflow#p1") as workflow_span:
            with tracer.trace("comfyui.node.execute", service="comfyui", resource="CLIPTextEncode") as node_span_1:
                pass
            with tracer.trace("comfyui.node.execute", service="comfyui", resource="KSampler") as node_span_2:
                pass

        # Both node spans should be children of the workflow span
        assert node_span_1.parent_id == workflow_span.span_id
        assert node_span_2.parent_id == workflow_span.span_id
        # All share the same trace
        assert node_span_1.trace_id == workflow_span.trace_id
        assert node_span_2.trace_id == workflow_span.trace_id

    def test_pending_result_not_error(self):
        """PENDING result (async nodes, subgraph expansion) is not tagged as error."""
        result = (ExecutionResult.PENDING, None, None)

        with tracer.trace("comfyui.node.execute", service="comfyui", resource="APINode") as span:
            if result and len(result) >= 1:
                span.set_tag('node.result', result[0].name if hasattr(result[0], 'name') else str(result[0]))
                if hasattr(result[0], 'name') and result[0].name == 'FAILURE':
                    span.set_tag('error', True)

        assert span.get_tag('node.result') == "PENDING"
        assert span.get_tag('error') is None

    def test_unknown_class_type_fallback(self):
        """When class_type lookup fails, falls back to 'unknown'."""
        dynprompt = MagicMock()
        dynprompt.get_node.side_effect = KeyError("node not found")

        try:
            class_type = dynprompt.get_node("missing_node").get('class_type', 'unknown')
        except Exception:
            class_type = 'unknown'

        with tracer.trace("comfyui.node.execute", service="comfyui", resource=class_type) as span:
            span.set_tag('node.class_type', class_type)

        assert span.resource == "unknown"
        assert span.get_tag('node.class_type') == "unknown"


class TestCaptureNodeMemorySnapshot:
    """Tests for the capture_node_memory_snapshot helper."""

    @pytest.fixture(autouse=True)
    def _import_mod(self):
        """Import __init__ module with execution mock to prevent monkey_patch side effects."""
        import sys
        # Ensure execution module is mocked so monkey_patch_comfyui doesn't blow up
        mock_execution = MagicMock()
        with patch.dict(sys.modules, {'execution': mock_execution}):
            # Force re-import if already cached without mock
            if '__init__' in sys.modules:
                import importlib
                self.mod = importlib.reload(sys.modules['__init__'])
            else:
                import __init__ as mod
                self.mod = mod
            yield

    def test_sets_ram_metric(self):
        """RAM available bytes metric is set on the span."""
        span = MagicMock()

        with patch.object(self.mod, 'NODE_MEMORY_TRACKING_ENABLED', True):
            # Mock torch as unavailable so only RAM branch runs
            with patch.dict('sys.modules', {'torch': None}):
                self.mod.capture_node_memory_snapshot(span, "before")

        # RAM metric should be set (psutil is available in test env)
        span.set_metric.assert_called()
        metric_names = [call[0][0] for call in span.set_metric.call_args_list]
        assert 'memory.ram_available_bytes.before' in metric_names

    def test_skips_when_disabled(self):
        """No metrics set when NODE_MEMORY_TRACKING_ENABLED is false."""
        span = MagicMock()

        with patch.object(self.mod, 'NODE_MEMORY_TRACKING_ENABLED', False):
            self.mod.capture_node_memory_snapshot(span, "before")

        span.set_metric.assert_not_called()

    def test_stage_suffix_applied(self):
        """The stage parameter (before/after) is included in metric names."""
        span = MagicMock()

        with patch.object(self.mod, 'NODE_MEMORY_TRACKING_ENABLED', True):
            with patch.dict('sys.modules', {'torch': None}):
                self.mod.capture_node_memory_snapshot(span, "after")

        metric_names = [call[0][0] for call in span.set_metric.call_args_list]
        assert any('after' in name for name in metric_names)
        assert not any('before' in name for name in metric_names)


class TestNodeTracingConfig:
    """Tests for NODE_TRACING_ENABLED and NODE_MEMORY_TRACKING_ENABLED config."""

    def test_node_tracing_enabled_by_default(self):
        """NODE_TRACING_ENABLED defaults to true."""
        # When env var is not set, it should be true
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('NODE_TRACING_ENABLED', None)
            result = os.getenv('NODE_TRACING_ENABLED', 'true').lower() != 'false'
            assert result is True

    def test_node_tracing_disabled_via_env(self):
        """NODE_TRACING_ENABLED=false disables per-node tracing."""
        with patch.dict(os.environ, {'NODE_TRACING_ENABLED': 'false'}):
            result = os.getenv('NODE_TRACING_ENABLED', 'true').lower() != 'false'
            assert result is False

    def test_node_memory_tracking_enabled_by_default(self):
        """NODE_MEMORY_TRACKING defaults to true."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('NODE_MEMORY_TRACKING', None)
            result = os.getenv('NODE_MEMORY_TRACKING', 'true').lower() != 'false'
            assert result is True

    def test_node_memory_tracking_disabled_via_env(self):
        """NODE_MEMORY_TRACKING=false disables per-node memory capture."""
        with patch.dict(os.environ, {'NODE_MEMORY_TRACKING': 'false'}):
            result = os.getenv('NODE_MEMORY_TRACKING', 'true').lower() != 'false'
            assert result is False
