"""
Metadata Extractor for document nodes.

Applies configurable regex patterns to filenames to extract structured
metadata, and optionally enriches it via database lookups.  This module
is framework-generic — the specific patterns and DB mappings are driven
entirely by the YAML configuration.
"""

import logging
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core.schema import TextNode

from rag_framework.config.metadata_config import MetadataExtractionConfig

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extracts and enriches metadata on document nodes.

    Lifecycle:
    1. Instantiate with a ``MetadataExtractionConfig``.
    2. Call :meth:`extract_and_enrich` on every node **after** basic
       ingestion metadata (``file_name``) has been set.
    """

    def __init__(self, config: MetadataExtractionConfig) -> None:
        self.config = config
        self._compiled_patterns: List[re.Pattern] = []
        self._db_lookup_cache: Dict[str, str] = {}

        if not config.enabled:
            return

        # Pre-compile regex patterns
        for pat_cfg in config.filename_patterns:
            if pat_cfg.pattern:
                self._compiled_patterns.append(re.compile(pat_cfg.pattern))

        # Pre-load DB enrichment mapping
        if config.db_enrichment.enabled:
            self._load_db_mapping()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_and_enrich(self, node: TextNode) -> None:
        """Apply filename patterns + DB enrichment to *node* in-place."""
        if not self.config.enabled:
            return

        file_name = node.metadata.get("file_name", "")
        if not file_name:
            return

        # Step 1: regex extraction
        for compiled in self._compiled_patterns:
            match = compiled.search(file_name)
            if match:
                for key, value in match.groupdict().items():
                    if value is not None:
                        node.metadata[key] = value

        # Step 2: DB enrichment
        enr = self.config.db_enrichment
        if enr.enabled and enr.source_field in node.metadata:
            source_value = node.metadata[enr.source_field]
            resolved = self._db_lookup_cache.get(source_value)
            if resolved:
                node.metadata[enr.target_field] = resolved

        # Step 3: visibility rules
        self._apply_visibility(node)

    def get_all_values(self, field: str, nodes: List[TextNode]) -> List[str]:
        """Return sorted unique values of *field* across *nodes*."""
        values = set()
        for node in nodes:
            val = node.metadata.get(field)
            if val:
                values.add(val)
        return sorted(values)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_db_mapping(self) -> None:
        """Load the enrichment mapping from SQLite."""
        enr = self.config.db_enrichment
        db_path = Path(enr.sqlite_path)
        if not db_path.exists():
            logger.warning("Metadata enrichment DB not found: %s — skipping", db_path)
            return

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            # Use parameterised identifiers is not possible for table/column
            # names, but this value comes from the trusted YAML config, not
            # user input, so it is safe.
            query = f"SELECT {enr.key_column}, {enr.value_column} " f"FROM {enr.table}"
            cursor.execute(query)
            for key, value in cursor.fetchall():
                self._db_lookup_cache[str(key)] = str(value)
            conn.close()
            logger.info(
                "Metadata enrichment loaded %d mappings from %s.%s",
                len(self._db_lookup_cache),
                enr.table,
                enr.value_column,
            )
        except Exception as exc:
            logger.warning("Failed to load metadata enrichment DB: %s", exc)

    def _apply_visibility(self, node: TextNode) -> None:
        """Configure which metadata keys are visible to LLM / embedding."""
        all_extracted = set()
        for compiled in self._compiled_patterns:
            all_extracted.update(compiled.groupindex.keys())
        enr = self.config.db_enrichment
        if enr.enabled and enr.target_field:
            all_extracted.add(enr.target_field)

        embed_visible = set(self.config.embed_visible_fields)
        llm_visible = set(self.config.llm_visible_fields)

        for key in all_extracted:
            if key not in node.metadata:
                continue
            # Hide from embedding if not explicitly visible
            if key not in embed_visible:
                if key not in node.excluded_embed_metadata_keys:
                    node.excluded_embed_metadata_keys.append(key)
            # Hide from LLM if not explicitly visible
            if key not in llm_visible:
                if key not in node.excluded_llm_metadata_keys:
                    node.excluded_llm_metadata_keys.append(key)
