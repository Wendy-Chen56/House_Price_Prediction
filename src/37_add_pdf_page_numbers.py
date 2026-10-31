
from pathlib import Path
import fitz

# Project paths
project_dir = Path(__file__).resolve().parent.parent
report_dir = project_dir / "report"

input_pdf = report_dir / "final_report.pdf"
output_pdf = report_dir / "final_report_numbered.pdf"

# Open original PDF
doc = fitz.open(input_pdf)
total_pages = len(doc)

# Add page numbers
for page_index, page in enumerate(doc):
    page_number = page_index + 1

    # Position near the bottom of the page
    page_width = page.rect.width
    page_height = page.rect.height

    page.insert_textbox(
        fitz.Rect(
            0,
            page_height - 35,
            page_width,
            page_height - 15
        ),
        f"{page_number}",
        fontsize=10,
        fontname="helv",
        color=(0.3, 0.3, 0.3),
        align=fitz.TEXT_ALIGN_CENTER
    )

# Save as a separate file
doc.save(output_pdf)
doc.close()

print("PDF PAGE NUMBERS ADDED")
print("=" * 45)
print(f"Total pages: {total_pages}")
print(f"Output: {output_pdf}")
print("Numbered PDF saved successfully!")
