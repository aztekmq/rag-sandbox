"""PDF ingestion utilities powered by Docling.

This module converts PDFs into clean text, applies chunking, and returns
metadata used by the vector store. Verbose logging is provided for
traceability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Tuple

from docling.document_converter import DocumentConverter
from docling.document_converter import PdfFormat
from docling.document_converter.converters import PdfiumTextExtractor
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types import Page, TextSegment

logger = logging.getLogger(__name__)


class PdfIngestor:
    """Convert PDFs into text chunks with metadata."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.converter = DocumentConverter(
            pdf_formats=[PdfFormat.PDF],
            pipeline_builder=lambda: StandardPdfPipeline(text_extractor=PdfiumTextExtractor()),
        )
        logger.info("Initialized PdfIngestor with chunk_size=%d overlap=%d", chunk_size, chunk_overlap)

    def convert_pdf(self, pdf_path: Path) -> Tuple[List[str], List[dict]]:
        """Convert a PDF into text chunks and metadata entries."""

        logger.info("Converting PDF: %s", pdf_path)
        document = self.converter.convert(pdf_path)
        text_segments = self._collect_segments(document.pages)
        chunks, metadata = self._chunk_segments(text_segments, pdf_path)
        logger.info("Converted %s into %d chunks", pdf_path.name, len(chunks))
        return chunks, metadata

    @staticmethod
    def _collect_segments(pages: Iterable[Page]) -> List[TextSegment]:
        segments: List[TextSegment] = []
        for page in pages:
            segments.extend(page.text_segments)
        logger.debug("Collected %d text segments from PDF", len(segments))
        return segments

    def _chunk_segments(self, segments: Iterable[TextSegment], pdf_path: Path) -> Tuple[List[str], List[dict]]:
        text = "\n".join(segment.text for segment in segments if segment.text)
        return self._chunk_text(text, pdf_path)

    def _chunk_text(self, text: str, pdf_path: Path) -> Tuple[List[str], List[dict]]:
        chunks: List[str] = []
        metadata: List[dict] = []

        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            metadata.append({
                "source": pdf_path.name,
                "start": start,
                "end": end,
            })
            start += self.chunk_size - self.chunk_overlap

        logger.debug("Chunked text into %d parts for %s", len(chunks), pdf_path.name)
        return chunks, metadata


def ingest_pdf_files(
    pdf_paths: Iterable[Path],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Tuple[List[str], List[dict]]:
    """Ingest multiple PDF files and return combined chunks and metadata."""

    ingestor = PdfIngestor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: List[str] = []
    all_metadata: List[dict] = []

    pdf_list = list(pdf_paths)
    for pdf_path in pdf_list:
        if not pdf_path.exists():
            logger.warning("PDF not found: %s", pdf_path)
            continue
        chunks, metadata = ingestor.convert_pdf(pdf_path)
        all_chunks.extend(chunks)
        all_metadata.extend(metadata)

    logger.info("Ingested %d PDFs into %d chunks", len(pdf_list), len(all_chunks))
    return all_chunks, all_metadata
