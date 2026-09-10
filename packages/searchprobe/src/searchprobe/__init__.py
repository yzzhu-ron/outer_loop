"""Provider-independent diagnostics for paired search probes."""

from .audit import audit_jsonl, audit_records, canonical_query, validate_record

__all__ = ["audit_jsonl", "audit_records", "canonical_query", "validate_record"]
__version__ = "0.1.0"
