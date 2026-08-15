"""Loading user attachments (PDFs and images) into content blocks."""

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

MAX_ATTACHMENT_BYTES = 30 * 1024 * 1024


class AttachmentError(ValueError):
    pass


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
