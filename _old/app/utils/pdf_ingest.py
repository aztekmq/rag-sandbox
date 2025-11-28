"""PDF ingestion utilities powered by PyPDFium.

This module converts PDFs into clean text, applies chunking, and returns
metadata used by the vector store. Verbose logging is provided for
traceability in line with international programming documentation standards.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import pypdfium2 as pdfium

logger = logging.getLogger(__name__)


class PdfIngestor:
    """Convert PDFs into text chunks with metadata."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            "Initialized PdfIngestor with chunk_size=%d overlap=%d", chunk_size, chunk_overlap
        )

    def convert_pdf(self, pdf_path: Path) -> Tuple[List[str], List[dict]]:
        """Convert a PDF into text chunks and metadata entries."""

        logger.info("Converting PDF: %s", pdf_path)
        text = self._extract_text(pdf_path)
        chunks, metadata = self._chunk_text(text, pdf_path)
        logger.info("Converted %s into %d chunks", pdf_path.name, len(chunks))
        return chunks, metadata

    def _extract_text(self, pdf_path: Path) -> str:
        """Extract text from a PDF using PyPDFium with detailed logging."""

        document = pdfium.PdfDocument(str(pdf_path))
        logger.debug("Opened PDF %s with %d pages", pdf_path.name, len(document))
        pages_text: List[str] = []

        for index, page in enumerate(document):
            logger.debug("Processing page %d of %s", index + 1, pdf_path.name)
            text_page = page.get_textpage()
            page_text = text_page.get_text_range()
            pages_text.append(page_text)
            text_page.close()
            page.close()

        full_text = "\n".join(pages_text)
        logger.debug("Extracted %d characters from %s", len(full_text), pdf_path.name)
        return full_text

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
