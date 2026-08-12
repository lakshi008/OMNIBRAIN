import pymupdf


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from every page of a PDF.

    Returns:
        A list containing one dictionary per page.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        pages.append(
            {
                "page": page_number,
                "text": text.strip(),
            }
        )

    document.close()

    return pages