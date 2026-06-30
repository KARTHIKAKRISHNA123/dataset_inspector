"""Exception hierarchy.

Rationale: inspecting 100+ messy datasets means *some files will be broken*. We must distinguish
errors we recover from (skip this file, keep going) from errors that mean our setup is wrong (stop).
Catching a specific base (`InspectorError`) lets each layer choose its fault policy instead of
blanket `except Exception` that would also hide real bugs.
"""

from __future__ import annotations


class InspectorError(Exception):
    """Base for all intentional errors."""


class ConfigError(InspectorError):
    """Bad/missing configuration. Fatal."""


class DiscoveryError(InspectorError):
    """Filesystem walk failed in a non-recoverable way. Usually fatal."""


class ReaderError(InspectorError):
    """A reader could not parse a file. Recoverable — record the error, skip the file."""


class UnsupportedFormatError(InspectorError):
    """No registered reader handles this file. Recoverable — metadata-only report."""


class CorruptFileError(ReaderError):
    """Truncated/garbled file or archive member. Recoverable — quarantine via report.error."""


class CheckpointError(InspectorError):
    """Resume state unreadable. Fatal (so we don't silently re-do or skip work incorrectly)."""
