"""
PDF image extractor for the OmniBrain ingestion pipeline.

Provides validation and page-by-page image extraction from PDF files,
returning structured ImageExtractionResult objects with full metadata.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from ingestion.models import DocumentMetadata, ExtractedImage, ImageExtractionResult
from ingestion.pdf_text_extractor import validate_pdf


def extract_images(filepath: str | Path) -> ImageExtractionResult:
    """Extract embedded images from every page of a PDF and return structured data.

    This function validates the PDF, opens it, inspects each page for embedded
    images, extracts their bytes and metadata, and builds an ImageExtractionResult.

    Args:
        filepath: Path to the PDF file to parse for images.

    Returns:
        ImageExtractionResult containing document metadata and extracted images.

    Raises:
        PDFNotFoundError: If the file does not exist.
        InvalidFileTypeError: If the file extension is not `.pdf`.
        CorruptedPDFError: If PyMuPDF cannot open/parse the file.
    """
    path = validate_pdf(filepath)
    document = pymupdf.open(str(path))

    try:
        images: list[ExtractedImage] = []
        pages_with_content = 0
        total_pages = len(document)

        for page_index in range(total_pages):
            page = document[page_index]
            page_number = page_index + 1

            # Check if page has text content
            page_text = page.get_text("text").strip()
            if page_text:
                pages_with_content += 1

            # Get images on this page
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = document.extract_image(xref)

                if base_image:
                    image_bytes = base_image.get("image", b"")
                    image_ext = base_image.get("ext", "png")
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    cs_val = base_image.get("cs-name") or base_image.get("colorspace") or "unknown"
                    colorspace = str(cs_val)
                    size_bytes = len(image_bytes)

                    images.append(
                        ExtractedImage(
                            page_number=page_number,
                            image_index=img_index,
                            image_format=image_ext,
                            width=width,
                            height=height,
                            image_bytes=image_bytes,
                            size_bytes=size_bytes,
                            colorspace=colorspace,
                            xref=xref,
                        )
                    )

        metadata = DocumentMetadata(
            document_id=str(uuid.uuid4()),
            filename=path.name,
            total_pages=total_pages,
            content_type="application/pdf",
            created_at=datetime.now(timezone.utc).isoformat(),
            pages_with_content=pages_with_content,
            pages_without_content=total_pages - pages_with_content,
        )

        return ImageExtractionResult(metadata=metadata, images=images)

    finally:
        document.close()
