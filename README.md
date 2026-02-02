# pdf2md

Convert PDFs to Markdown with LaTeX math support. Perfect for:
- Goodnotes exports to Obsidian
- Academic papers and textbooks
- Handwritten notes with equations
- Any PDF with math content

## Features

- **Text & Scanned PDFs** - Handles both digital and scanned documents
- **LaTeX Math** - Converts equations to Obsidian-compatible `$$` syntax
- **Batch Processing** - Convert entire directories at once
- **Fast Fallback** - Basic mode for quick conversions
- **Obsidian Ready** - Output works perfectly with Obsidian

## Installation

```bash
pip install pdf2md
```

Or install from source:

```bash
git clone https://github.com/DEVLlN/pdf2md.git
cd pdf2md
pip install -e .
```

## Usage

### Single File

```bash
# Convert a PDF (output: document.md)
pdf2md document.pdf

# Specify output name
pdf2md document.pdf -o notes.md
```

### Directory

```bash
# Convert all PDFs in a folder
pdf2md ./pdfs/ -o ./markdown/

# Recursive (include subfolders)
pdf2md ./pdfs/ -r -o ./markdown/
```

### Options

```
pdf2md [input] [options]

Arguments:
  input              PDF file or directory to convert

Options:
  -o, --output       Output file or directory
  -r, --recursive    Process directories recursively
  --basic            Use basic conversion (faster, lower quality)
  --version          Show version
  -h, --help         Show help
```

## Output Format

The output is Obsidian-compatible Markdown with:

- YAML frontmatter with source info
- Headings preserved from document structure
- Math equations in `$$` blocks (display) and `$` (inline)
- Images extracted and embedded (when using Marker)

Example output:

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
```

## Workflows

### Goodnotes → Obsidian

1. Export from Goodnotes as PDF to Google Drive
2. Run `pdf2md` on the PDF
3. Move the `.md` file to your Obsidian vault

```bash
# One-liner
pdf2md ~/Google\ Drive/Goodnotes/*.pdf -o ~/Obsidian/Notes/
```

### Batch Convert Textbooks

```bash
pdf2md ./textbooks/ -r -o ./markdown/textbooks/
```

## How It Works

pdf2md uses [Marker](https://github.com/VikParuchuri/marker) under the hood, which:

1. Detects if PDF is text-based or scanned
2. Uses OCR for scanned documents
3. Identifies math regions
4. Converts to clean Markdown with LaTeX

For faster (but lower quality) conversion, use `--basic` mode which extracts text directly without OCR.

## Requirements

- Python 3.9+
- ~2GB disk space for Marker models (downloaded on first run)

## Troubleshooting

### "Marker is not installed"

```bash
pip install marker-pdf
```

### Slow first run

The first conversion downloads ML models (~2GB). Subsequent runs are faster.

### Poor math quality

- Ensure the PDF is high resolution
- Try without `--basic` flag
- For handwritten math, ensure writing is clear

### Memory issues

For large PDFs, Marker can use significant RAM. Try:
- Processing fewer pages at once
- Using `--basic` mode
- Closing other applications

## License

MIT License - feel free to modify and share!

## Contributing

Contributions welcome! Please open an issue or PR on GitHub.
