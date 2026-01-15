"""
PII Scrubber for PyFlare SDK

Provides utilities to detect and redact Personally Identifiable Information (PII)
from trace data before storage.

Security Features:
- Pattern-based detection for common PII types
- Configurable redaction patterns
- Support for custom patterns
- Preserves data structure while redacting sensitive content
"""

import re
import logging
from typing import Any, Dict, List, Optional, Pattern, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PIIPattern:
    """Defines a PII detection pattern."""
    name: str
    pattern: Pattern[str]
    replacement: str = "[REDACTED]"
    enabled: bool = True


# Default PII patterns
DEFAULT_PATTERNS: List[PIIPattern] = [
    # Email addresses
    PIIPattern(
        name="email",
        pattern=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        replacement="[EMAIL_REDACTED]"
    ),
    # Phone numbers (various formats)
    PIIPattern(
        name="phone",
        pattern=re.compile(r'\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
        replacement="[PHONE_REDACTED]"
    ),
    # Social Security Numbers
    PIIPattern(
        name="ssn",
        pattern=re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'),
        replacement="[SSN_REDACTED]"
    ),
    # Credit Card Numbers (basic pattern)
    PIIPattern(
        name="credit_card",
        pattern=re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
        replacement="[CARD_REDACTED]"
    ),
    # API Keys (common formats)
    PIIPattern(
        name="api_key",
        pattern=re.compile(r'\b(?:sk|pk|api|key|token|secret|password|auth)[_-]?[A-Za-z0-9]{20,}\b', re.IGNORECASE),
        replacement="[API_KEY_REDACTED]"
    ),
    # Bearer tokens
    PIIPattern(
        name="bearer_token",
        pattern=re.compile(r'Bearer\s+[A-Za-z0-9\-_]+\.?[A-Za-z0-9\-_]*\.?[A-Za-z0-9\-_]*', re.IGNORECASE),
        replacement="Bearer [TOKEN_REDACTED]"
    ),
    # AWS Access Keys
    PIIPattern(
        name="aws_access_key",
        pattern=re.compile(r'\b(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}\b'),
        replacement="[AWS_KEY_REDACTED]"
    ),
    # AWS Secret Keys
    PIIPattern(
        name="aws_secret_key",
        pattern=re.compile(r'\b[A-Za-z0-9/+=]{40}\b'),
        replacement="[AWS_SECRET_REDACTED]"
    ),
    # IP Addresses
    PIIPattern(
        name="ip_address",
        pattern=re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
        replacement="[IP_REDACTED]"
    ),
    # Passwords in common formats
    PIIPattern(
        name="password_field",
        pattern=re.compile(r'(?:password|passwd|pwd|secret|credential)["\s:=]+["\']?[^\s"\']+["\']?', re.IGNORECASE),
        replacement="[PASSWORD_REDACTED]"
    ),
    # URLs with credentials
    PIIPattern(
        name="url_with_creds",
        pattern=re.compile(r'(?:https?://)[^\s:]+:[^\s@]+@[^\s]+'),
        replacement="[URL_WITH_CREDS_REDACTED]"
    ),
]


class PIIScrubber:
    """
    Scrubs PII from text content.

    Example:
        scrubber = PIIScrubber()
        clean_text = scrubber.scrub("Contact me at user@example.com")
        # Returns: "Contact me at [EMAIL_REDACTED]"
    """

    def __init__(
        self,
        patterns: Optional[List[PIIPattern]] = None,
        enabled: bool = True,
        log_detections: bool = False,
    ) -> None:
        """
        Initialize the PII scrubber.

        Args:
            patterns: Custom patterns to use (defaults to DEFAULT_PATTERNS)
            enabled: Whether scrubbing is enabled
            log_detections: Whether to log when PII is detected
        """
        self._patterns = patterns or DEFAULT_PATTERNS.copy()
        self._enabled = enabled
        self._log_detections = log_detections
        self._detection_counts: Dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        """Whether scrubbing is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable scrubbing."""
        self._enabled = value

    def add_pattern(self, pattern: PIIPattern) -> None:
        """Add a custom pattern."""
        self._patterns.append(pattern)

    def remove_pattern(self, name: str) -> None:
        """Remove a pattern by name."""
        self._patterns = [p for p in self._patterns if p.name != name]

    def enable_pattern(self, name: str) -> None:
        """Enable a specific pattern."""
        for pattern in self._patterns:
            if pattern.name == name:
                pattern.enabled = True
                break

    def disable_pattern(self, name: str) -> None:
        """Disable a specific pattern."""
        for pattern in self._patterns:
            if pattern.name == name:
                pattern.enabled = False
                break

    def get_detection_stats(self) -> Dict[str, int]:
        """Get detection statistics."""
        return self._detection_counts.copy()

    def reset_stats(self) -> None:
        """Reset detection statistics."""
        self._detection_counts.clear()

    def scrub(self, text: str) -> str:
        """
        Scrub PII from text.

        Args:
            text: Text to scrub

        Returns:
            Scrubbed text with PII redacted
        """
        if not self._enabled or not text:
            return text

        result = text
        for pattern in self._patterns:
            if not pattern.enabled:
                continue

            matches = pattern.pattern.findall(result)
            if matches:
                if self._log_detections:
                    logger.warning(
                        f"PII detected: {pattern.name} ({len(matches)} occurrences)"
                    )
                self._detection_counts[pattern.name] = (
                    self._detection_counts.get(pattern.name, 0) + len(matches)
                )
                result = pattern.pattern.sub(pattern.replacement, result)

        return result

    def scrub_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively scrub PII from a dictionary.

        Args:
            data: Dictionary to scrub

        Returns:
            Scrubbed dictionary
        """
        if not self._enabled:
            return data

        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.scrub(value)
            elif isinstance(value, dict):
                result[key] = self.scrub_dict(value)
            elif isinstance(value, list):
                result[key] = self.scrub_list(value)
            else:
                result[key] = value
        return result

    def scrub_list(self, data: List[Any]) -> List[Any]:
        """
        Recursively scrub PII from a list.

        Args:
            data: List to scrub

        Returns:
            Scrubbed list
        """
        if not self._enabled:
            return data

        result = []
        for item in data:
            if isinstance(item, str):
                result.append(self.scrub(item))
            elif isinstance(item, dict):
                result.append(self.scrub_dict(item))
            elif isinstance(item, list):
                result.append(self.scrub_list(item))
            else:
                result.append(item)
        return result


# Global scrubber instance
_default_scrubber: Optional[PIIScrubber] = None


def get_scrubber() -> PIIScrubber:
    """Get the default PII scrubber instance."""
    global _default_scrubber
    if _default_scrubber is None:
        _default_scrubber = PIIScrubber()
    return _default_scrubber


def configure_scrubber(
    enabled: bool = True,
    log_detections: bool = False,
    custom_patterns: Optional[List[PIIPattern]] = None,
) -> PIIScrubber:
    """
    Configure the default PII scrubber.

    Args:
        enabled: Whether scrubbing is enabled
        log_detections: Whether to log detections
        custom_patterns: Additional custom patterns

    Returns:
        Configured scrubber instance
    """
    global _default_scrubber
    patterns = DEFAULT_PATTERNS.copy()
    if custom_patterns:
        patterns.extend(custom_patterns)

    _default_scrubber = PIIScrubber(
        patterns=patterns,
        enabled=enabled,
        log_detections=log_detections,
    )
    return _default_scrubber


def scrub_text(text: str) -> str:
    """
    Scrub PII from text using the default scrubber.

    Args:
        text: Text to scrub

    Returns:
        Scrubbed text
    """
    return get_scrubber().scrub(text)


def scrub_data(data: Any) -> Any:
    """
    Scrub PII from data (str, dict, or list) using the default scrubber.

    Args:
        data: Data to scrub

    Returns:
        Scrubbed data
    """
    scrubber = get_scrubber()
    if isinstance(data, str):
        return scrubber.scrub(data)
    elif isinstance(data, dict):
        return scrubber.scrub_dict(data)
    elif isinstance(data, list):
        return scrubber.scrub_list(data)
    return data
