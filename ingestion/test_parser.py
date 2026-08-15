from ingestion.pdf_parser import extract_text_and_images


PDF_PATH = "data/raw/FINAL Annual Report.pdf"


# Process the PDF
document = extract_text_and_images(PDF_PATH)


# Document metadata
print("Document:", document["metadata"]["filename"])
print("Total pages:", document["metadata"]["total_pages"])


# Get pages
pages = document["pages"]


# Count images
total_images = sum(
    len(page["images"])
    for page in pages
)

print("Total images:", total_images)


# Display first 5 pages
for page in pages[:5]:

    print(f"\n--- Page {page['page']} ---")

    print(f"Characters: {page['character_count']}")

    print(f"Words: {page['word_count']}")

    print(f"Images: {len(page['images'])}")

    for image in page["images"]:
        print(f"  → {image['path']}")