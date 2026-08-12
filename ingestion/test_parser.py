from ingestion.pdf_parser import extract_text_from_pdf


PDF_PATH = "data/raw/FINAL Annual Report.pdf"

pages = extract_text_from_pdf(PDF_PATH)

print(f"Total pages extracted: {len(pages)}")

for page in pages[:3]:
    print(f"\n--- Page {page['page']} ---")
    print(page["text"][:500])

