"""Schema detection + dataset-type classification from a bounded record sample."""

from .detect import SchemaSampler, classify_dataset

__all__ = ["SchemaSampler", "classify_dataset"]
