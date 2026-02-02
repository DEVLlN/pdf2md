#!/usr/bin/env python3
"""
pdf2md - Convert PDFs to Markdown with LaTeX math support.

Handles both text-based and scanned PDFs. Outputs Obsidian-compatible
Markdown with math blocks using $$ syntax.
"""

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Check for pymupdf4llm (best quality, handles math well)
try:
    import pymupdf4llm
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False

# Fallback imports for basic PDF processing
try:
    import pymupdf  # PyMuPDF/fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz as pymupdf
        PYMUPDF_AVAILABLE = True
    except ImportError:
        PYMUPDF_AVAILABLE = False


def has_text_content(pdf_path: str) -> bool:
    """Check if PDF has extractable text or is scanned/image-based."""
    if not PYMUPDF_AVAILABLE:
        return True  # Assume text if we can't check

    doc = pymupdf.open(pdf_path)
    total_text_len = 0

    for page in doc:
        text = page.get_text("text")
        total_text_len += len(text.strip())
        if total_text_len > 100:  # Found enough text
            doc.close()
            return True

    doc.close()
    return total_text_len > 50  # Minimal text threshold


def convert_with_ocr(pdf_path: str) -> str:
    """Convert scanned PDF using OCR."""
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF is not installed. Install with: pip install pymupdf")

    print(f"Converting {pdf_path} with OCR (scanned document detected)...")

    doc = pymupdf.open(pdf_path)
    text_parts = []

    for page_num, page in enumerate(doc, 1):
        print(f"  OCR page {page_num}/{len(doc)}...", end="\r")

        # Get page as high-res image for OCR
        # Use pymupdf's built-in OCR with tesseract
        try:
            # Try to use get_textpage with OCR
            tp = page.get_textpage(flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
            text = page.get_text("text", textpage=tp)

            if not text.strip():
                # Fallback: render to image and OCR with tesseract directly
                pix = page.get_pixmap(dpi=300)
                text = pix.pdfocr_tobytes()  # OCR via tesseract
                if isinstance(text, bytes):
                    # This returns a PDF, need different approach
                    # Use page.get_text with OCR flag
                    text = page.get_text("text", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
        except Exception:
            # Simpler fallback
            text = page.get_text("text")

        if text.strip():
            text_parts.append(f"## Page {page_num}\n\n{text.strip()}")

    print()  # Clear the progress line
    doc.close()

    if not text_parts:
        # Last resort: try ocrmypdf-style conversion
        return convert_with_tesseract_direct(pdf_path)

    markdown_text = "\n\n---\n\n".join(text_parts)
    return postprocess_markdown(markdown_text)


def convert_with_tesseract_direct(pdf_path: str) -> str:
    """Direct OCR using pdf2image and pytesseract as fallback."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        raise ImportError(
            "For scanned PDFs, install: pip install pdf2image pytesseract\n"
            "Also ensure tesseract is installed: brew install tesseract"
        )

    print(f"Converting {pdf_path} with direct Tesseract OCR...")

    images = convert_from_path(pdf_path, dpi=300)
    text_parts = []

    for i, image in enumerate(images, 1):
        print(f"  OCR page {i}/{len(images)}...", end="\r")
        text = pytesseract.image_to_string(image)
        if text.strip():
            text_parts.append(f"## Page {i}\n\n{text.strip()}")

    print()
    markdown_text = "\n\n---\n\n".join(text_parts)
    return postprocess_markdown(markdown_text)


def convert_with_pymupdf4llm(pdf_path: str, force_ocr: bool = False) -> str:
    """Convert PDF to Markdown using pymupdf4llm (good quality, fast)."""
    if not PYMUPDF4LLM_AVAILABLE:
        raise ImportError("pymupdf4llm is not installed. Install with: pip install pymupdf4llm")

    # Check if PDF needs OCR
    if force_ocr or not has_text_content(pdf_path):
        return convert_with_ocr(pdf_path)

    print(f"Converting {pdf_path} with pymupdf4llm...")

    # Convert to markdown
    markdown_text = pymupdf4llm.to_markdown(pdf_path)

    # If output is empty, try OCR
    if not markdown_text.strip():
        print("No text extracted, trying OCR...")
        return convert_with_ocr(pdf_path)

    # Post-process for Obsidian compatibility
    markdown_text = postprocess_markdown(markdown_text)

    return markdown_text


def convert_with_pymupdf(pdf_path: str) -> str:
    """Basic PDF to text conversion using PyMuPDF (fallback)."""
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF is not installed. Install with: pip install pymupdf")

    print(f"Converting {pdf_path} with PyMuPDF (basic mode)...")

    doc = pymupdf.open(pdf_path)
    text_parts = []

    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text")
        if text.strip():
            text_parts.append(f"## Page {page_num}\n\n{text}")

    doc.close()

    markdown_text = "\n\n---\n\n".join(text_parts)
    markdown_text = postprocess_markdown(markdown_text)

    return markdown_text


def postprocess_markdown(text: str) -> str:
    """Post-process markdown for Obsidian compatibility."""

    # Normalize math delimiters to Obsidian-style
    # Convert \[ \] to $$
    text = re.sub(r'\\\[', '$$', text)
    text = re.sub(r'\\\]', '$$', text)

    # Convert \( \) to $ for inline math
    text = re.sub(r'\\\(', '$', text)
    text = re.sub(r'\\\)', '$', text)

    # Ensure display math is on its own lines
    text = re.sub(r'([^\n])\$\$', r'\1\n$$', text)
    text = re.sub(r'\$\$([^\n])', r'$$\n\1', text)

    # Clean up excessive newlines
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    # Fix common OCR artifacts in math
    text = fix_math_artifacts(text)

    return text.strip()


def fix_math_artifacts(text: str) -> str:
    """Fix common OCR/conversion artifacts in math expressions."""

    # Common substitutions
    replacements = [
        # Fix multiplication
        (r'(\d)\s*[xX]\s*(\d)', r'\1 \\times \2'),
        # Fix arrows
        (r'->', r'\\rightarrow'),
        (r'<-', r'\\leftarrow'),
        # Fix common symbols
        (r'(?<!\$)>=(?!\$)', r'\\geq'),
        (r'(?<!\$)<=(?!\$)', r'\\leq'),
        (r'(?<!\$)!=(?!\$)', r'\\neq'),
        # Fix fractions that look like a/b in math context
        (r'\$([^$]+)\$', lambda m: fix_inline_math(m.group(1))),
    ]

    for pattern, replacement in replacements:
        if callable(replacement):
            text = re.sub(pattern, replacement, text)
        else:
            text = re.sub(pattern, replacement, text)

    return text


def fix_inline_math(math_content: str) -> str:
    """Fix inline math content."""
    # Don't modify if already has LaTeX commands
    if '\\' in math_content:
        return f'${math_content}$'

    # Convert simple fractions
    math_content = re.sub(r'(\w+)/(\w+)', r'\\frac{\1}{\2}', math_content)

    return f'${math_content}$'


def convert_pdf(
    pdf_path: str,
    output_path: Optional[str] = None,
    use_marker: bool = True,
    force_ocr: bool = False,
) -> str:
    """
    Convert a PDF to Markdown.

    Args:
        pdf_path: Path to the PDF file
        output_path: Optional output path (default: same name with .md extension)
        use_marker: Use Marker for conversion (better quality, slower)
        force_ocr: Force OCR even for text-based PDFs

    Returns:
        Path to the output Markdown file
    """
    pdf_path = os.path.abspath(pdf_path)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Determine output path
    if output_path is None:
        output_path = os.path.splitext(pdf_path)[0] + ".md"

    output_path = os.path.abspath(output_path)

    # Convert
    try:
        if use_marker and PYMUPDF4LLM_AVAILABLE:
            markdown_text = convert_with_pymupdf4llm(pdf_path)
        elif PYMUPDF_AVAILABLE:
            print("Note: Using basic conversion. Install pymupdf4llm for better results.")
            markdown_text = convert_with_pymupdf(pdf_path)
        else:
            raise ImportError(
                "No PDF conversion library available. "
                "Install with: pip install pymupdf4llm"
            )
    except Exception as e:
        print(f"Error during conversion: {e}")
        if PYMUPDF_AVAILABLE:
            print("Falling back to basic conversion...")
            markdown_text = convert_with_pymupdf(pdf_path)
        else:
            raise

    # Add source metadata
    pdf_name = os.path.basename(pdf_path)
    header = f"---\nsource: {pdf_name}\nconverted_by: pdf2md\n---\n\n"
    markdown_text = header + markdown_text

    # Write output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_text)

    print(f"Saved: {output_path}")
    return output_path


def convert_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    recursive: bool = False,
    use_marker: bool = True,
) -> list[str]:
    """
    Convert all PDFs in a directory.

    Args:
        input_dir: Directory containing PDFs
        output_dir: Output directory (default: same as input)
        recursive: Process subdirectories
        use_marker: Use Marker for conversion

    Returns:
        List of output file paths
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir) if output_dir else input_dir

    # Find PDFs
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = list(Path(input_dir).glob(pattern))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return []

    print(f"Found {len(pdf_files)} PDF(s) to convert")

    output_files = []
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")

        # Calculate relative path for output
        rel_path = pdf_path.relative_to(input_dir)
        out_path = Path(output_dir) / rel_path.with_suffix('.md')

        try:
            result = convert_pdf(
                str(pdf_path),
                str(out_path),
                use_marker=use_marker,
            )
            output_files.append(result)
        except Exception as e:
            print(f"  Error: {e}")

    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDFs to Markdown with LaTeX math support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pdf2md document.pdf                    # Convert single file
  pdf2md document.pdf -o notes.md        # Specify output name
  pdf2md ./pdfs/ -o ./markdown/          # Convert directory
  pdf2md ./pdfs/ -r                      # Recursive conversion
  pdf2md document.pdf --basic            # Use basic conversion (faster)
        """,
    )

    parser.add_argument(
        "input",
        help="PDF file or directory to convert",
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file or directory",
    )

    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process directories recursively",
    )

    parser.add_argument(
        "--basic",
        action="store_true",
        help="Use basic conversion (faster, lower quality)",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="pdf2md 0.1.0",
    )

    args = parser.parse_args()

    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    use_marker = not args.basic

    # Convert
    try:
        if os.path.isfile(args.input):
            convert_pdf(args.input, args.output, use_marker=use_marker)
        else:
            convert_directory(
                args.input,
                args.output,
                recursive=args.recursive,
                use_marker=use_marker,
            )
        print("\nDone!")
    except KeyboardInterrupt:
        print("\nCancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
