"""
Document ingestion module with support for multiple document types.
"""

import re
from pathlib import Path
from typing import List, Optional

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from rag_framework.config.models import RAGConfig, ChunkingConfig


class DocumentIngestion:
    """
    Handles document loading and chunking.

    Supports multiple document formats via Docling:
    - PDF, DOCX, PPTX
    - HTML, MD, TXT
    - XLSX
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".md", ".txt", ".xlsx"}

    def __init__(self, config: RAGConfig):
        """
        Initialize document ingestion.

        Args:
            config: RAG configuration
        """
        self.config = config
        self.documents_dir = config.directories.documents_dir
        self.chunking_config = config.chunking

        # Optional metadata extraction
        self._metadata_extractor = None
        if hasattr(config, "metadata") and config.metadata.enabled:
            from rag_framework.core.metadata_extractor import MetadataExtractor

            self._metadata_extractor = MetadataExtractor(config.metadata)

    def get_document_files(self, documents_dir: Optional[str] = None) -> List[Path]:
        """
        Get all supported document files from the specified directory.

        Args:
            documents_dir: Directory to scan (uses config default if not specified)

        Returns:
            List of file paths
        """
        doc_dir = Path(documents_dir or self.documents_dir)

        if not doc_dir.exists():
            doc_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created documents directory: {doc_dir}")
            return []

        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(doc_dir.glob(f"*{ext}"))
            files.extend(doc_dir.glob(f"**/*{ext}"))  # Recursive

        return list(set(files))  # Remove duplicates

    def load_documents(self, file_paths: Optional[List[Path]] = None) -> List[Document]:
        """
        Load documents using Docling for supported formats, simple reader for others.

        Args:
            file_paths: List of file paths to load (auto-discovers if not specified)

        Returns:
            List of loaded documents
        """
        if file_paths is None:
            file_paths = self.get_document_files()

        if not file_paths:
            print("No documents found to process.")
            return []

        print(f"Loading {len(file_paths)} documents...")

        # Separate files by type - Docling doesn't support .txt, .json
        docling_formats = {".pdf", ".docx", ".pptx", ".html", ".md", ".xlsx", ".csv"}
        simple_formats = {".txt", ".json"}

        docling_files = [f for f in file_paths if f.suffix.lower() in docling_formats]
        simple_files = [f for f in file_paths if f.suffix.lower() in simple_formats]

        documents = []

        # Load with Docling for supported formats
        if docling_files:
            try:
                from llama_index.readers.docling import DoclingReader

                reader = DoclingReader()

                for file_path in docling_files:
                    try:
                        print(f"   Processing (Docling): {file_path.name}")
                        docs = reader.load_data(file_path=str(file_path))
                        # Ensure file_name metadata is set (Docling may not set it)
                        for doc in docs:
                            if "file_name" not in doc.metadata:
                                doc.metadata["file_name"] = file_path.name
                            if "file_path" not in doc.metadata:
                                doc.metadata["file_path"] = str(file_path)
                        documents.extend(docs)
                    except Exception as e:
                        print(f"   Error processing {file_path.name}: {e}")
                        # Try simple reader as fallback
                        simple_files.append(file_path)
            except ImportError:
                print("Docling not available, using simple reader for all files")
                simple_files.extend(docling_files)

        # Load with simple reader for .txt, .json and fallback files
        if simple_files:
            documents.extend(self._load_with_simple_reader(simple_files))

        print(f"Loaded {len(documents)} documents")
        return documents

    def _load_with_simple_reader(self, file_paths: List[Path]) -> List[Document]:
        """Fallback loader for when Docling is not available."""
        from llama_index.core import SimpleDirectoryReader

        documents = []
        for file_path in file_paths:
            try:
                print(f"   Processing: {file_path.name}")
                reader = SimpleDirectoryReader(input_files=[str(file_path)])
                docs = reader.load_data()
                documents.extend(docs)
            except Exception as e:
                print(f"   Error processing {file_path.name}: {e}")

        return documents

    def create_nodes(self, documents: List[Document]) -> List[TextNode]:
        """
        Create nodes (chunks) from documents.

        Uses a hybrid strategy:
        1. If ``use_semantic_chunking`` is enabled and DoclingNodeParser is
           available, split by document structure (sections, tables, etc.).
        2. Re-split any oversized nodes with ``SentenceSplitter`` so every
           chunk stays within the configured ``chunk_size``.
        3. Falls back to pure ``SentenceSplitter`` when the Docling parser
           is missing or semantic chunking is disabled.

        Args:
            documents: List of documents to chunk

        Returns:
            List of text nodes
        """
        if not documents:
            return []

        use_semantic = self.chunking_config.use_semantic_chunking

        if use_semantic:
            nodes = self._semantic_chunking(documents)
        else:
            nodes = self._fixed_size_chunking(documents)

        # Enrich metadata: add chunk_index and source context
        for idx, node in enumerate(nodes):
            node.metadata["chunk_index"] = idx
            self._enrich_node_metadata(node)
            # Apply configurable metadata extraction (filename patterns + DB)
            if self._metadata_extractor is not None:
                self._metadata_extractor.extract_and_enrich(node)

        # Verify metadata propagation to nodes
        nodes_with_meta = sum(1 for n in nodes if n.metadata.get("file_name"))
        if nodes_with_meta < len(nodes) and nodes:
            print(
                f"   WARNING: {len(nodes) - nodes_with_meta}/{len(nodes)} chunks "
                f"missing file_name metadata"
            )

        print(f"Created {len(nodes)} chunks")
        return nodes

    def _semantic_chunking(self, documents: List[Document]) -> List[TextNode]:
        """
        Hybrid semantic chunking: DoclingNodeParser + SentenceSplitter.

        Phase 1: DoclingNodeParser creates structure-aware chunks.
        Phase 2: SentenceSplitter re-splits oversized chunks.
        """
        try:
            from llama_index.node_parser.docling import DoclingNodeParser
        except ImportError:
            print(
                "   DoclingNodeParser not available, "
                "falling back to fixed-size chunking"
            )
            return self._fixed_size_chunking(documents)

        print(
            f"Semantic chunking (Docling + SentenceSplitter fallback, "
            f"max_size={self.chunking_config.chunk_size}, "
            f"oversized_factor={self.chunking_config.semantic_oversized_factor})..."
        )

        # Phase 1: Structure-aware chunking
        docling_parser = DoclingNodeParser()
        try:
            structural_nodes = docling_parser.get_nodes_from_documents(
                documents, show_progress=True
            )
        except Exception as e:
            print(f"   DoclingNodeParser failed: {e}")
            print("   Falling back to fixed-size chunking")
            return self._fixed_size_chunking(documents)

        print(f"   Phase 1 (Docling): {len(structural_nodes)} structural chunks")

        # Extract headings from structural nodes before re-split
        # so sub-nodes inherit the parent's heading
        for node in structural_nodes:
            heading = self._extract_heading_from_text(node.get_content())
            if heading:
                node.metadata["heading"] = heading

        # Phase 2: Re-split oversized nodes (threshold in tokens)
        max_allowed_tokens = int(
            self.chunking_config.chunk_size
            * self.chunking_config.semantic_oversized_factor
        )
        final_nodes = self._resplit_oversized_nodes(
            structural_nodes, max_allowed_tokens
        )

        resplit_count = len(final_nodes) - len(structural_nodes)
        if resplit_count > 0:
            print(
                f"   Phase 2 (re-split): {resplit_count} extra chunks "
                f"from oversized nodes"
            )

        print(f"   Final: {len(final_nodes)} chunks total")
        return final_nodes

    def _fixed_size_chunking(self, documents: List[Document]) -> List[TextNode]:
        """Original fixed-size chunking with SentenceSplitter."""
        print(
            f"Chunking with size={self.chunking_config.chunk_size}, "
            f"overlap={self.chunking_config.chunk_overlap}..."
        )

        splitter = SentenceSplitter(
            chunk_size=self.chunking_config.chunk_size,
            chunk_overlap=self.chunking_config.chunk_overlap,
        )

        nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
        return nodes

    def _resplit_oversized_nodes(
        self,
        nodes: List[TextNode],
        max_tokens: int,
    ) -> List[TextNode]:
        """
        Re-split nodes that exceed *max_tokens* using SentenceSplitter.

        Small/normal nodes pass through untouched.  Oversized nodes are
        converted back to Documents (preserving metadata) and re-chunked.

        The comparison uses estimated token count (not character count)
        to match the units used by SentenceSplitter internally.

        Args:
            nodes: Structural nodes from DoclingNodeParser
            max_tokens: Maximum allowed tokens per chunk

        Returns:
            List of nodes with oversized ones subdivided
        """
        splitter = SentenceSplitter(
            chunk_size=self.chunking_config.chunk_size,
            chunk_overlap=self.chunking_config.chunk_overlap,
        )

        final_nodes: List[TextNode] = []

        for node in nodes:
            content = node.get_content()
            token_count = self._estimate_tokens(content)
            if token_count > max_tokens:
                # Re-chunk preserving original metadata
                sub_doc = Document(text=content, metadata=dict(node.metadata))
                sub_nodes = splitter.get_nodes_from_documents([sub_doc])
                # Propagate metadata from parent node (including heading)
                for sub in sub_nodes:
                    sub.metadata.update(node.metadata)
                final_nodes.extend(sub_nodes)
            else:
                final_nodes.append(node)

        return final_nodes

    # =========================================================================
    # Metadata Enrichment
    # =========================================================================

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Estimate the token count of a text string.

        Uses tiktoken if available (accurate for most LLMs), otherwise
        falls back to a character-based heuristic (~4 chars per token).
        """
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except (ImportError, Exception):
            return len(text) // 4

    @staticmethod
    def _extract_heading_from_text(text: str) -> str:
        """
        Extract the primary heading from markdown-formatted text.

        Scans the first lines for markdown headings (``## Heading``)
        and returns the most specific one found.  Works generically
        with any document converted to markdown by Docling.

        Returns:
            Heading text, or empty string if none found.
        """
        lines = text.strip().split("\n")
        headings: List[str] = []
        for line in lines[:20]:
            line = line.strip()
            match = re.match(r"^#{1,6}\s+(.+)", line)
            if match:
                heading_text = match.group(1).strip()
                if heading_text:
                    headings.append(heading_text)
            elif line:
                # Allow blank lines between headings but stop at
                # the first non-heading, non-empty content line
                if headings:
                    break

        if not headings:
            return ""

        # Return the last heading (usually the most specific / contextual)
        return headings[-1] if len(headings) > 1 else headings[0]

    def _enrich_node_metadata(self, node: TextNode) -> None:
        """
        Enrich a node with structured metadata and configure visibility.

        Adds:
        - ``heading``: extracted from markdown headings in the text
        - ``source_context``: formatted string combining file + section

        Configures which metadata keys are visible to the LLM vs
        the embedding model so that:
        - The **LLM** sees ``source_context`` (clean, human-readable)
        - The **embedding** sees ``heading`` (helps semantic retrieval)
        """
        # Extract heading if not already present (e.g. from Docling)
        if "heading" not in node.metadata:
            heading = self._extract_heading_from_text(node.get_content())
            if heading:
                node.metadata["heading"] = heading

        # Build formatted source context for LLM
        file_name = node.metadata.get("file_name", "")
        heading = node.metadata.get("heading", "")

        parts = []
        if file_name:
            parts.append(f"Fuente: {file_name}")
        if heading:
            parts.append(f"Sección: {heading}")

        if parts:
            node.metadata["source_context"] = " | ".join(parts)

        # LLM sees: source_context (clean, formatted attribution)
        for key in ["file_path", "chunk_index", "file_name", "heading"]:
            if key not in node.excluded_llm_metadata_keys:
                node.excluded_llm_metadata_keys.append(key)

        # Embeddings see: heading (helps semantic retrieval)
        for key in ["file_path", "chunk_index", "source_context"]:
            if key not in node.excluded_embed_metadata_keys:
                node.excluded_embed_metadata_keys.append(key)

    def ingest(self, documents_dir: Optional[str] = None) -> List[TextNode]:
        """
        Full ingestion pipeline: load documents and create nodes.

        Args:
            documents_dir: Directory to process (uses config default if not specified)

        Returns:
            List of text nodes ready for indexing
        """
        print("=" * 50)
        print("DOCUMENT INGESTION")
        print("=" * 50)

        # Get files
        file_paths = self.get_document_files(documents_dir)

        if not file_paths:
            print(f"No documents found in: {documents_dir or self.documents_dir}")
            print("   Supported formats:", ", ".join(self.SUPPORTED_EXTENSIONS))
            return []

        print(f"Found {len(file_paths)} files:")
        for f in file_paths[:5]:
            print(f"   - {f.name}")
        if len(file_paths) > 5:
            print(f"   ... and {len(file_paths) - 5} more")

        # Load documents
        documents = self.load_documents(file_paths)

        # Create nodes
        nodes = self.create_nodes(documents)

        print("=" * 50)
        print(f"INGESTION COMPLETE: {len(nodes)} chunks created")
        print("=" * 50)

        return nodes
