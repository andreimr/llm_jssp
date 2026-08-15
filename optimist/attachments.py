"""Loading user attachments into content blocks and workspace files.

Two kinds of attachment:
- Documents/images (.pdf, .png, ...) become model-visible content blocks.
- Data files (.csv, .json, ...) are copied into the session workspace where
  solver code reads them directly; the model sees a short preview, not the
  whole file, so instance size is no longer capped by the context window.
"""

from __future__ import annotations

import base64
from pathlib import Path

from optimist.messages import ContentBlock, ImageBlock, PdfBlock

IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

DATA_FILE_TYPES = {".csv", ".tsv", ".json", ".txt", ".md", ".dat"}

MAX_ATTACHMENT_BYTES = 30 * 1024 * 1024


class AttachmentError(ValueError):
    pass


def is_data_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in DATA_FILE_TYPES


def data_file_preview(path: Path, max_lines: int = 30, max_chars: int = 2500) -> str:
    """First lines of a data file, for showing the model its shape."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(unreadable: {exc})"
    lines = text.splitlines()
    preview = "\n".join(lines[:max_lines])
    if len(preview) > max_chars:
        preview = preview[:max_chars]
    suffix = ""
    if len(lines) > max_lines or len(preview) < len(text):
        suffix = f"\n... ({len(lines)} lines, {len(text)} characters total)"
    return preview + suffix


def extract_pdf_text(data: bytes, max_chars: int = 200_000) -> str:
    """Best-effort text extraction, used for providers without native PDF input."""
    try:
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = []
        total = 0
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(f"--- page {i + 1} ---\n{text}")
            total += len(text)
            if total > max_chars:
                pages.append(f"... [stopped after page {i + 1}]")
                break
        return "\n".join(pages)
    except Exception as exc:
        return f"(PDF text extraction failed: {exc})"


def load_attachment(path: str | Path) -> ContentBlock:
    """Load a PDF or image file into a content block."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise AttachmentError(f"No such file: {p}")
    if p.stat().st_size > MAX_ATTACHMENT_BYTES:
        raise AttachmentError(f"{p.name} is larger than {MAX_ATTACHMENT_BYTES // 1024**2} MB")
    suffix = p.suffix.lower()
    data = p.read_bytes()
    b64 = base64.standard_b64encode(data).decode("ascii")
    if suffix == ".pdf":
        return PdfBlock(data_b64=b64, name=p.name, extracted_text=extract_pdf_text(data))
    if suffix in IMAGE_TYPES:
        return ImageBlock(media_type=IMAGE_TYPES[suffix], data_b64=b64, name=p.name)
    raise AttachmentError(
        f"Unsupported attachment type {suffix!r} (supported: .pdf, {', '.join(IMAGE_TYPES)})"
    )
