
from pathlib import Path
import markdown

# Project paths
project_dir = Path(__file__).resolve().parent.parent
report_dir = project_dir / "report"

markdown_path = report_dir / "final_report.md"
html_path = report_dir / "final_report.html"

# Read Markdown report
markdown_text = markdown_path.read_text(encoding="utf-8")

# Convert Markdown to HTML
html_content = markdown.markdown(
    markdown_text,
    extensions=["tables", "fenced_code", "toc"]
)

# Add styling for the final report
html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Prediction - Final Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 24px;
            color: #222;
            line-height: 1.7;
        }}
        h1, h2, h3 {{
            color: #243746;
        }}
        h1 {{
            text-align: center;
        }}
        h2 {{
            margin-top: 36px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 8px;
        }}
        img {{
            display: block;
            max-width: 100%;
            height: auto;
            margin: 20px auto;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ccc;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background: #f2f4f6;
        }}
        @media print {{
            body {{
                max-width: none;
                margin: 0;
            }}
            img, table {{
                break-inside: avoid;
            }}
            h2, h3 {{
                break-after: avoid;
            }}
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""

# Save HTML file
html_path.write_text(html_document, encoding="utf-8")

print("HTML REPORT GENERATED")
print("=" * 45)
print(f"Source: {markdown_path}")
print(f"Output: {html_path}")
print(f"Figures referenced: {markdown_text.count('![')}")
print("HTML report saved successfully!")
