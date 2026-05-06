"""
Query Preprocessor for metadata-based pre-filtering.

Analyses the user's query to detect references to known metadata values
(e.g. subject names) and produces metadata filters that the retriever
can use to narrow down the search space before semantic ranking.

This module is framework-generic — the fields and values are driven
by the ``MetadataFilterConfig`` in the YAML configuration.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set

from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

from rag_framework.config.metadata_config import MetadataFilterConfig

logger = logging.getLogger(__name__)

# Stopwords filtered out of token overlap (common Spanish + articles).
# Keep short but meaningful tokens like "i", "ii", "1", "2", etc.
_STOPWORDS: Set[str] = frozenset(
    {
        "a",
        "al",
        "ante",
        "bajo",
        "con",
        "contra",
        "de",
        "del",
        "desde",
        "durante",
        "el",
        "en",
        "entre",
        "hacia",
        "hasta",
        "la",
        "las",
        "lo",
        "los",
        "mediante",
        "para",
        "por",
        "segun",
        "sin",
        "sobre",
        "tras",
        "un",
        "una",
        "unas",
        "uno",
        "unos",
        "y",
        "e",
        "ni",
        "o",
        "u",
        "que",
        "se",
        "es",
        "esta",
        "son",
        "como",
    }
)

# Bidirectional Roman ↔ Arabic lookup for numerals I–X.
_ROMAN_TO_ARABIC = {
    "i": "1",
    "ii": "2",
    "iii": "3",
    "iv": "4",
    "v": "5",
    "vi": "6",
    "vii": "7",
    "viii": "8",
    "ix": "9",
    "x": "10",
}
_ARABIC_TO_ROMAN = {v: k for k, v in _ROMAN_TO_ARABIC.items()}


def _normalise(text: str) -> str:
    """Lower-case and strip diacritics for fuzzy matching."""
    text = text.lower()
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _tokenise(text: str) -> List[str]:
    """Extract alphanumeric tokens, stripping punctuation."""
    return re.findall(r"\w+", text)


def _significant_tokens(text: str) -> List[str]:
    """Return tokens after removing stopwords, with numeral normalisation.

    Each token is returned in **both** its original form and its
    numeral-equivalent (if applicable) so that "2" can match "ii".
    """
    tokens = []
    for t in _tokenise(text):
        if t in _STOPWORDS:
            continue
        tokens.append(t)
    return tokens


def _expand_numerals(tokens: List[str]) -> Set[str]:
    """Expand token list so that roman and arabic forms both appear."""
    expanded = set(tokens)
    for t in tokens:
        if t in _ROMAN_TO_ARABIC:
            expanded.add(_ROMAN_TO_ARABIC[t])
        elif t in _ARABIC_TO_ROMAN:
            expanded.add(_ARABIC_TO_ROMAN[t])
    return expanded


@dataclass
class PrefilterResult:
    """Result produced by :class:`QueryPreprocessor`."""

    matched_value: Optional[str] = None
    matched_field: Optional[str] = None
    similarity: float = 0.0
    metadata_filters: Optional[MetadataFilters] = None
    boost_field_value: Optional[str] = None


class QueryPreprocessor:
    """Detect metadata values in user queries and build retrieval filters.

    Typical usage::

        preprocessor = QueryPreprocessor(config, known_values)
        result = preprocessor.analyse(query)
        # pass result.metadata_filters to the retriever
    """

    def __init__(
        self,
        config: MetadataFilterConfig,
        known_values: Optional[List[str]] = None,
    ) -> None:
        self.config = config
        # Build normalised lookup: normalised_value → original_value
        self._value_map: Dict[str, str] = {}
        if known_values:
            for val in known_values:
                self._value_map[_normalise(val)] = val

    def analyse(self, query: str) -> PrefilterResult:
        """Analyse *query* and return a :class:`PrefilterResult`."""
        if not self.config.enabled or not self._value_map:
            return PrefilterResult()

        result = PrefilterResult()

        # 1. Try to match the query against known metadata values
        norm_query = _normalise(query)
        best_score = 0.0
        best_original = None

        # Score tier constants: substring matches must always beat
        # token-overlap and fuzzy matches, regardless of ratio.
        _SUBSTRING_BASE = 1.5  # guaranteed to beat any 0-1 score

        for norm_val, original in self._value_map.items():
            # Check if the known value appears as a substring in the query,
            # respecting word boundaries so "Software I" does NOT match
            # inside "Software II".
            idx = norm_query.find(norm_val)
            if idx >= 0:
                end = idx + len(norm_val)
                # Word-boundary check: char after the match must be
                # non-alphanumeric (space, punctuation, end of string).
                if end >= len(norm_query) or not norm_query[end].isalnum():
                    # Longer substring match → higher score within the tier.
                    score = _SUBSTRING_BASE + len(norm_val)
                    if score > best_score:
                        best_score = score
                        best_original = original
                    continue

            # Compute token overlap using stopword filtering and numeral
            # normalisation so "II" matches "2" and short but meaningful
            # tokens like "I", "II" are not discarded.
            val_tokens = _significant_tokens(norm_val)
            query_token_set = _expand_numerals(_significant_tokens(norm_query))
            n_matching = (
                sum(1 for w in val_tokens if w in query_token_set) if val_tokens else 0
            )

            # Token overlap score: robust against abbreviated names
            if len(val_tokens) >= 2 and n_matching >= 2:
                token_score = n_matching / len(val_tokens)
                if token_score > best_score:
                    best_score = token_score
                    best_original = original

            # Fuzzy: slide a window of len(norm_val) over the query
            val_len = len(norm_val)
            if val_len < 4:
                continue
            fuzzy_best = 0.0
            for i in range(max(1, len(norm_query) - val_len + 1)):
                window = norm_query[i : i + val_len + 5]
                ratio = SequenceMatcher(None, norm_val, window).ratio()
                fuzzy_best = max(fuzzy_best, ratio)

            # Penalize fuzzy when fewer than 2 significant tokens overlap
            if len(val_tokens) >= 2 and n_matching < 2:
                fuzzy_best *= 0.7

            if fuzzy_best > best_score:
                best_score = fuzzy_best
                best_original = original

        if best_original and best_score >= self.config.fuzzy_threshold:
            # Normalise internal score back to 0-1 for external consumers
            display_score = min(best_score, 1.0)
            result.matched_value = best_original
            result.matched_field = self.config.match_field
            result.similarity = display_score
            result.metadata_filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key=self.config.match_field,
                        value=best_original,
                        operator=FilterOperator.EQ,
                    )
                ]
            )
            logger.info(
                "Pre-filter: matched '%s' = '%s' (score=%.2f)",
                self.config.match_field,
                best_original,
                display_score,
            )

        # 2. Boost field (e.g. document_type based on query keywords)
        if self.config.boost_field and self.config.boost_mapping:
            for value, keywords in self.config.boost_mapping.items():
                for kw in keywords:
                    if _normalise(kw) in norm_query:
                        result.boost_field_value = value
                        logger.debug(
                            "Pre-filter boost: %s=%s (keyword '%s')",
                            self.config.boost_field,
                            value,
                            kw,
                        )
                        break
                if result.boost_field_value:
                    break

        return result

    def apply_boost(
        self,
        nodes: List[NodeWithScore],
        boost_field: str,
        boost_value: str,
        factor: float,
    ) -> List[NodeWithScore]:
        """Multiply score of nodes whose *boost_field* matches *boost_value*."""
        for node_ws in nodes:
            meta_val = node_ws.node.metadata.get(boost_field, "")
            if _normalise(meta_val) == _normalise(boost_value):
                node_ws.score = (node_ws.score or 0.0) * factor
        # Re-sort by score descending
        nodes.sort(key=lambda n: n.score or 0.0, reverse=True)
        return nodes
