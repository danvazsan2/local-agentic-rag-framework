"""
Metadata extraction and filtering configuration dataclasses.

Provides optional, configurable metadata extraction from filenames
and metadata-based pre-filtering during retrieval.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class FilenamePatternConfig:
    """A single regex pattern to extract metadata from filenames.

    The pattern must use named groups ``(?P<name>...)`` to define which
    metadata fields are extracted.  Example::

        pattern: "^(?P<doc_type>Programa|Proyecto)_(?P<code>\\d+)"
    """

    pattern: str = ""
    description: str = ""

    def __post_init__(self):
        if self.pattern:
            import re

            try:
                re.compile(self.pattern)
            except re.error as exc:
                raise ValueError(f"Invalid regex in filename_patterns: {exc}") from exc


@dataclass
class DbEnrichmentConfig:
    """Optional database lookup to enrich extracted metadata.

    For example, resolve a numeric subject code to a human-readable name
    by querying a SQLite table.
    """

    enabled: bool = False
    sqlite_path: str = ""
    source_field: str = ""
    table: str = ""
    key_column: str = ""
    value_column: str = ""
    target_field: str = ""

    def __post_init__(self):
        if self.enabled:
            if not all(
                [
                    self.sqlite_path,
                    self.source_field,
                    self.table,
                    self.key_column,
                    self.value_column,
                    self.target_field,
                ]
            ):
                raise ValueError(
                    "db_enrichment: all fields are required when enabled=true"
                )


@dataclass
class MetadataFilterConfig:
    """Configuration for metadata-based pre-filtering during retrieval.

    When enabled, the query preprocessor will attempt to detect values
    of ``match_field`` in the user's query and, if found, restrict
    retrieval to nodes whose metadata matches.
    """

    enabled: bool = False
    match_field: str = ""
    fuzzy_threshold: float = 0.65
    boost_field: str = ""
    boost_mapping: Dict[str, List[str]] = field(default_factory=dict)
    boost_factor: float = 1.5

    def __post_init__(self):
        if self.enabled and not self.match_field:
            raise ValueError(
                "metadata_filtering.match_field is required when enabled=true"
            )
        if not 0.0 <= self.fuzzy_threshold <= 1.0:
            raise ValueError("fuzzy_threshold must be between 0.0 and 1.0")
        if self.boost_factor < 1.0:
            raise ValueError("boost_factor must be >= 1.0")


@dataclass
class MetadataExtractionConfig:
    """Top-level configuration for metadata extraction.

    Controls:
    - ``filename_patterns``: regex patterns to extract metadata from filenames
    - ``db_enrichment``: optional DB lookup to enrich extracted metadata
    - ``embed_visible_fields``: metadata fields visible to the embedding model
    - ``llm_visible_fields``: metadata fields visible to the LLM
    - ``filtering``: optional metadata-based pre-filtering during retrieval
    """

    enabled: bool = False
    filename_patterns: List[FilenamePatternConfig] = field(default_factory=list)
    db_enrichment: DbEnrichmentConfig = field(default_factory=DbEnrichmentConfig)
    embed_visible_fields: List[str] = field(default_factory=list)
    llm_visible_fields: List[str] = field(default_factory=list)
    filtering: MetadataFilterConfig = field(default_factory=MetadataFilterConfig)
