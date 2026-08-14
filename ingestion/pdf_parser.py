import pymupdf
from pathlib import Path


def extract_text_and_images(
    pdf_path: str,
    image_output_dir: str = "data/processed/images",
) -> dict:
    """
    Extract text and embedded images from every page of a PDF.

    Returns:
        A dictionary containing document metadata and page data.
    """

    document = pymupdf.open(pdf_path)

    document_metadata = {
        "filename": Path(pdf_path).name,
        "total_pages": len(document),
    }

    image_output_path = Path(image_output_dir)
    image_output_path.mkdir(parents=True, exist_ok=True)

    pages = []

    for page_number, page in enumerate(document, start=1):

        # Extract text
        text = page.get_text("text")

        # Extract images
        images = []

        for image_index, image_info in enumerate(
            page.get_images(full=True)
        ):
            xref = image_info[0]

            image_data = document.extract_image(xref)

            image_bytes = image_data["image"]
            image_extension = image_data["ext"]

            image_filename = (
                f"page_{page_number}_img_{image_index}.{image_extension}"
            )

            image_path = image_output_path / image_filename

            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            images.append(
                {
                    "image_index": image_index,
                    "path": str(image_path),
                }
            )

        pages.append(
            {
                "page": page_number,
                "text": text.strip(),
                "images": images,
            }
        )

    document.close()

    return {
        "metadata": document_metadata,
        "pages": pages,
    }