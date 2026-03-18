from pathlib import Path

import markdown
from weasyprint import CSS, HTML


def build_pdf(md_path: Path, out_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")

    html_body = markdown.markdown(
        text,
        extensions=[
            "extra",
            "admonition",
            "toc",
            "tables",
            "fenced_code",
            "codehilite",
            "sane_lists",
        ],
        output_format="html5",
    )

    html = f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>GPAR Workshop</title>
</head>
<body>
  <main class=\"container\">{html_body}</main>
</body>
</html>
"""

    css = CSS(
        string="""
@page {
  size: A4;
  margin: 20mm 16mm 20mm 16mm;
}

body {
  font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: #1f2937;
  line-height: 1.55;
  font-size: 11pt;
  text-align: left;
  hyphens: none;
}

.container {
  max-width: 100%;
}

h1, h2, h3, h4 {
  color: #111827;
  line-height: 1.25;
  font-weight: 700;
  margin-top: 1.2em;
  margin-bottom: 0.45em;
  break-after: avoid;
}

h1 {
  font-size: 23pt;
  border-bottom: 2px solid #e5e7eb;
  padding-bottom: 8px;
}

h2 {
  font-size: 17pt;
  border-bottom: 1px solid #e5e7eb;
  padding-bottom: 4px;
}

h3 {
  font-size: 14pt;
}

h4 {
  font-size: 12pt;
  color: #1f2937;
}

p, li {
  margin: 0.45em 0;
  white-space: normal;
  overflow-wrap: normal;
  word-break: normal;
}

ul, ol {
  padding-left: 1.2em;
}

blockquote {
  border-left: 4px solid #d1d5db;
  margin: 0.8em 0;
  padding: 0.35em 1em;
  color: #374151;
  background: #f9fafb;
}

code {
  font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
  font-size: 9.4pt;
  background: #f3f4f6;
  padding: 0.1em 0.35em;
  border-radius: 4px;
}

pre {
  background: #0b1020;
  color: #e5e7eb;
  padding: 10px 12px;
  border-radius: 8px;
  overflow: auto;
  white-space: pre-wrap;
  word-wrap: break-word;
}

pre code {
  background: transparent;
  color: inherit;
  padding: 0;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.8em 0;
  font-size: 10pt;
}

th, td {
  border: 1px solid #e5e7eb;
  padding: 8px;
  vertical-align: top;
}

th {
  background: #f3f4f6;
  text-align: left;
}

img {
  max-width: 100%;
  height: auto;
  margin: 0.8em 0;
  border-radius: 8px;
}

a {
  color: #1d4ed8;
  text-decoration: none;
}

hr {
  border: 0;
  border-top: 1px solid #e5e7eb;
  margin: 1em 0;
}
"""
    )

    HTML(string=html, base_url=str(md_path.parent.resolve())).write_pdf(
        str(out_path), stylesheets=[css]
    )


if __name__ == "__main__":
    docs_dir = Path(__file__).resolve().parent
    markdown_file = docs_dir / "full_workshop.md"
    pdf_file = docs_dir / "full_workshop.pdf"
    build_pdf(markdown_file, pdf_file)
    print(pdf_file)