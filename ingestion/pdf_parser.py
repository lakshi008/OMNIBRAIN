import pymupdf
from pathlib import Path


def extract_text_and_images(
    pdf_path: str,
    image_output_dir: str = "data/processed/images",
) -> dict:

    document = pymupdf.open(pdf_path)

    filename = Path(pdf_path).name
    document_id = Path(pdf_path).stem.replace(" ", "_")

    document_metadata = {
        "filename": filename,
        "document_id": document_id,
        "total_pages": len(document),
    }

    image_output_path = Path(image_output_dir)
    image_output_path.mkdir(parents=True, exist_ok=True)

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text")
        clean_text = text.strip()

        images = []

        for image_index, image_info in enumerate(
            page.get_images(full=True)
        ):
            xref = image_info[0]

            image_data = document.extract_image(xref)

            image_filename = (
                f"page_{page_number}_img_{image_index}."
                f"{image_data['ext']}"
            )

            image_path = image_output_path / image_filename

            with open(image_path, "wb") as image_file:
                image_file.write(image_data["image"])

            images.append(
                {
                    "image_index": image_index,
                    "path": str(image_path),
                }
            )

        pages.append(
            {
                "page_id": f"{document_id}_page_{page_number}",
                "page": page_number,
                "text": clean_text,
                "images": images,
                "character_count": len(clean_text),
                "word_count": len(clean_text.split()),
            }
        )

    document.close()

    return {
        "metadata": document_metadata,
        "pages": pages,
    }