# pdf2md

Convert PDFs to Markdown with LaTeX math support and image extraction.

<p align="center">
  <a href="https://github.com/DEVLlN/pdf2md/actions/workflows/convert.yml">
    <img src="https://github.com/DEVLlN/pdf2md/actions/workflows/convert.yml/badge.svg" alt="Convert PDFs">
  </a>
</p>

## Features

- **Text & Scanned PDFs** - Handles both digital and scanned documents with OCR
- **Image Extraction** - Extracts embedded images and detects drawings/diagrams
- **LaTeX Math** - Preserves equations in Obsidian-compatible `$$` syntax
- **GitHub Action** - Automatically convert PDFs in your repos
- **Batch Processing** - Convert entire directories at once
- **Obsidian Ready** - Output works perfectly with Obsidian

## Quick Start

### Use as GitHub Action (Easiest)

Add this workflow to your repo at `.github/workflows/convert-pdfs.yml`:

```yaml
name: Convert PDFs to Markdown

on:
  push:
    paths:
      - '**.pdf'
  workflow_dispatch:

jobs:
  convert:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Convert PDFs
        uses: DEVLlN/pdf2md@master

      - name: Commit results
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add "*.md" "*_images/" 2>/dev/null || true
          git diff --staged --quiet || git commit -m "Convert PDFs to Markdown"
          git push
```

Now any PDF you push will automatically be converted to Markdown!

### Use Locally (CLI)

```bash
# Install dependencies
pip install pymupdf4llm pymupdf pdf2image pytesseract

# For OCR support, also install Tesseract:
# macOS: brew install tesseract
# Ubuntu: sudo apt install tesseract-ocr

# Download and run
curl -sL https://raw.githubusercontent.com/DEVLlN/pdf2md/master/pdf2md.py -o pdf2md.py
python pdf2md.py document.pdf
```

## Usage

### Single File

```bash
# Convert a PDF (output: document.md + document_images/)
python pdf2md.py document.pdf

# Specify output name
python pdf2md.py document.pdf -o notes.md
```

### Directory

```bash
# Convert all PDFs in a folder
python pdf2md.py ./pdfs/ -o ./markdown/

# Recursive (include subfolders)
python pdf2md.py ./pdfs/ -r -o ./markdown/
```

### Options

```
python pdf2md.py [input] [options]

Arguments:
  input              PDF file or directory to convert

Options:
  -o, --output       Output file or directory
  -r, --recursive    Process directories recursively
  --basic            Use basic conversion (faster, no OCR)
  --version          Show version
  -h, --help         Show help
```

## Output Format

Each PDF creates:
- A `.md` file with the text content
- A `_images/` folder with extracted images and drawings

```
document.pdf
    ↓
document.md
document_images/
  ├── document_p1_img0.png
  ├── document_p1_drawing0.png
  ├── document_p2_img0.png
  └── ...
```

Example markdown output:

```markdown
---
source: lecture-notes.pdf
converted_by: pdf2md
---

# Chapter 1: Introduction

The quadratic formula is:

$$
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$

For inline math like $E = mc^2$, we use single dollar signs.

![drawing from page 1](lecture-notes_images/lecture-notes_p1_drawing0.png)
```

## Workflows

### Goodnotes → GitHub → Obsidian

1. Export from Goodnotes as PDF
2. Push PDF to a GitHub repo with the Action enabled
3. Pull the auto-generated markdown into your Obsidian vault

### Local Batch Convert

```bash
# Convert all PDFs in a folder
python pdf2md.py ~/Documents/PDFs/ -r -o ~/Obsidian/Notes/
```

## How It Works

1. **Text Detection** - Checks if PDF has selectable text or is scanned
2. **OCR** - Uses Tesseract for scanned documents
3. **Image Extraction** - Extracts embedded images and renders drawing regions
4. **Markdown Generation** - Converts to clean, Obsidian-compatible markdown

## Requirements

- Python 3.9+
- Tesseract OCR (for scanned PDFs)
- ~100MB for dependencies

## Troubleshooting

### No text extracted from scanned PDF

Make sure Tesseract is installed:
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr

# Windows
choco install tesseract
```

### Images not displaying in Obsidian

Ensure the `_images` folder is in the same directory as the markdown file in your vault.

### Poor OCR quality on handwritten notes

Handwriting recognition is limited with Tesseract. For best results:
- Use clear, large handwriting
- Ensure good contrast in the PDF
- Consider using Goodnotes' built-in text export for handwritten content

## License

MIT License - feel free to modify and share!

## Contributing

Contributions welcome! Please open an issue or PR on GitHub.
