"""
Dataset Fingerprint Module

Generates a complete, cryptographic fingerprint of a dataset
for provenance tracking and reproducibility.
"""

import hashlib
import platform
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


class DatasetFingerprint:
    """
    Generates a comprehensive fingerprint of a dataset including
    content hash, schema hash, and metadata.
    """

    VERSION = "2.0.0"

    def __init__(self, data: pd.DataFrame, source_path: str = "DataFrame"):
        self.data = data
        self.source_path = source_path

    def generate(self) -> dict:
        """Generate complete dataset fingerprint."""
        content_hash = self._content_hash()
        schema_hash = self._schema_hash()

        return {
            "content_sha256": content_hash,
            "schema_sha256": schema_hash,
            "source_path": self.source_path,
            "row_count": len(self.data),
            "column_count": len(self.data.columns),
            "columns": list(self.data.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            "total_cells": self.data.size,
            "missing_cells": int(self.data.isnull().sum().sum()),
            "memory_usage_bytes": int(self.data.memory_usage(deep=True).sum()),
            "memory_usage_mb": round(self.data.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "system": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "autopsy_version": self.VERSION,
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
            },
        }

    def _content_hash(self) -> str:
        """Generate SHA-256 hash of the dataset content."""
        hasher = hashlib.sha256()
        for col in sorted(self.data.columns):
            col_data = self.data[col].to_string().encode("utf-8", errors="replace")
            hasher.update(col_data)
        return hasher.hexdigest()

    def _schema_hash(self) -> str:
        """Generate SHA-256 hash of the schema (column names + dtypes)."""
        hasher = hashlib.sha256()
        schema_str = "|".join(
            f"{col}:{dtype}" for col, dtype in sorted(self.data.dtypes.items())
        )
        hasher.update(schema_str.encode("utf-8"))
        return hasher.hexdigest()
