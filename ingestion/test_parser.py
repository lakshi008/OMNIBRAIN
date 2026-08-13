from ingestion.pdf_parser import extract_text_and_images


PDF_PATH = "data/raw/FINAL Annual Report.pdf"


pages = extract_text_and_images(PDF_PATH)

print(f"Total pages: {len(pages)}")

total_images = sum(len(page["images"]) for page in pages)

print(f"Total images: {total_images}")


for page in pages[:5]:
    print(f"\n--- Page {page['page']} ---")
    print(f"Text characters: {len(page['text'])}")
    print(f"Images: {len(page['images'])}")

    for image in page["images"]:
        print(f"  → {image['path']}")