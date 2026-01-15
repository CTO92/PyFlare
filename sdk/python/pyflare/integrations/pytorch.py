"""PyTorch integration for PyFlare."""

from typing import Any, Optional
import functools

from pyflare.sdk import PyFlare
from pyflare.types import InferenceType


def trace_model(
    model_id: Optional[str] = None,
    capture_input_shape: bool = True,
    capture_output_shape: bool = True,
) -> Any:
    """
    Decorator to trace PyTorch model forward pass.

    Example:
        from pyflare.integrations.pytorch import trace_model

        @trace_model(model_id="my-classifier")
        class MyModel(nn.Module):
            def forward(self, x):
                return self.layers(x)
    """

    def decorator(cls: type) -> type:
        original_forward = cls.forward

        @functools.wraps(original_forward)
        def traced_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
            pyflare = PyFlare.get_instance()
            if pyflare is None or not pyflare.enabled:
                return original_forward(self, *args, **kwargs)

            from opentelemetry.trace import SpanKind, Status, StatusCode

            tracer = pyflare.tracer
            span_name = f"pytorch.{cls.__name__}.forward"

            with tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL) as span:
                span.set_attribute(
                    "pyflare.inference.type", InferenceType.CUSTOM.value
                )

                if model_id:
                    span.set_attribute("pyflare.model.id", model_id)
                else:
                    span.set_attribute("pyflare.model.id", cls.__name__)

                # Capture input shape
                if capture_input_shape and args:
                    try:
                        import torch

                        if isinstance(args[0], torch.Tensor):
                            span.set_attribute(
                                "pyflare.input.shape", str(list(args[0].shape))
                            )
                    except Exception:
                        pass

                try:
                    result = original_forward(self, *args, **kwargs)

                    # Capture output shape
                    if capture_output_shape:
                        try:
                            import torch

                            if isinstance(result, torch.Tensor):
                                span.set_attribute(
                                    "pyflare.output.shape", str(list(result.shape))
                                )
                        except Exception:
                            pass

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        cls.forward = traced_forward
        return cls

    return decorator


class PyTorchInstrumentation:
    """
    Automatic instrumentation for PyTorch.

    Note: Due to PyTorch's dynamic nature, automatic instrumentation
    is limited. Use the @trace_model decorator for best results.
    """

    def __init__(self) -> None:
        self._instrumented = False

    def instrument(self) -> None:
        """Enable PyTorch instrumentation."""
        # PyTorch is harder to instrument automatically
        # Users should use @trace_model decorator
        self._instrumented = True

    def uninstrument(self) -> None:
        """Disable PyTorch instrumentation."""
        self._instrumented = False
