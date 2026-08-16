from ingestion.pdf_parser import extract_text_and_images


PDF_PATH = "data/raw/FINAL Annual Report.pdf"

document = extract_text_and_images(PDF_PATH)

print("Document:", document["metadata"]["filename"])
print("Document ID:", document["metadata"]["document_id"])
print("Total pages:", document["metadata"]["total_pages"])

pages = document["pages"]

for page in pages[:5]:

    print(f"\n--- Page {page['page']} ---")

    print("Page ID:", page["page_id"])
    print("Characters:", page["character_count"])
    print("Words:", page["word_count"])
    print("Images:", len(page["images"]))