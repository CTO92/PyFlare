"""
PyFlare Security Module

Provides security utilities for the PyFlare SDK including:
- PII detection and scrubbing
- Data sanitization
- Secure defaults
"""

from pyflare.security.pii_scrubber import (
    PIIPattern,
    PIIScrubber,
    configure_scrubber,
    get_scrubber,
    scrub_data,
    scrub_text,
)

__all__ = [
    "PIIPattern",
    "PIIScrubber",
    "configure_scrubber",
    "get_scrubber",
    "scrub_data",
    "scrub_text",
]
