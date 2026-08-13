"""Typed record-ancestry auditing for multi-view feedback datasets."""

from .adapter_api import DatasetAdapter, ManifestAdapter, SourceSpec
from .canonical import Atom, extract_atoms, hash_value, normalize_text
from .graph import AncestryGraph, RecordRef

__all__ = [
    "AncestryGraph",
    "Atom",
    "DatasetAdapter",
    "ManifestAdapter",
    "RecordRef",
    "SourceSpec",
    "extract_atoms",
    "hash_value",
    "normalize_text",
]

__version__ = "0.2.0"
