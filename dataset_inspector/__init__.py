"""dataset_inspector — a standalone, streaming inspector/profiler for heterogeneous datasets.

It *understands* datasets (metadata, preview, schema, classification, language hints, statistics)
without ever loading a whole file into RAM, and without preprocessing. It is independent of any
downstream tokenizer/training pipeline.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
