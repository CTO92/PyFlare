"""Custom exporters for PyFlare."""

from typing import Any, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class PyFlareExporter(SpanExporter):
    """
    Custom exporter that adds PyFlare-specific processing.

    This exporter wraps the OTLP exporter and adds:
    - Automatic embedding extraction for drift detection
    - Cost calculation
    - Metric aggregation
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4317",
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Initialize the PyFlare exporter.

        Args:
            endpoint: PyFlare collector endpoint
            headers: Additional headers for authentication
        """
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        self._otlp_exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers=headers,
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans to PyFlare collector."""
        # Pre-process spans if needed
        for span in spans:
            self._process_span(span)

        # Forward to OTLP exporter
        return self._otlp_exporter.export(spans)

    def _process_span(self, span: ReadableSpan) -> None:
        """Process span before export."""
        # This is where we could:
        # - Extract embeddings for drift detection
        # - Calculate costs based on token usage
        # - Aggregate metrics
        pass

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self._otlp_exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush pending spans."""
        return self._otlp_exporter.force_flush(timeout_millis)


class ConsoleExporter(SpanExporter):
    """
    Debug exporter that prints spans to console.

    Useful for development and debugging.
    """

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans to console."""
        for span in spans:
            self._print_span(span)
        return SpanExportResult.SUCCESS

    def _print_span(self, span: ReadableSpan) -> None:
        """Print span information."""
        print(f"\n{'=' * 60}")
        print(f"Span: {span.name}")
        print(f"  Trace ID: {format(span.context.trace_id, '032x')}")
        print(f"  Span ID: {format(span.context.span_id, '016x')}")
        print(f"  Duration: {(span.end_time - span.start_time) / 1e6:.2f}ms")
        print(f"  Status: {span.status.status_code.name}")

        if span.attributes:
            print("  Attributes:")
            for key, value in span.attributes.items():
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                print(f"    {key}: {value_str}")

        print(f"{'=' * 60}\n")

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush."""
        return True
