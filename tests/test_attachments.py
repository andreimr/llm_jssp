import base64

import pytest

from optimist.attachments import AttachmentError, load_attachment
from optimist.messages import ImageBlock, PdfBlock

# A valid 1x1 transparent PNG.
PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
)


def test_load_png(tmp_path):
    p = tmp_path / "pixel.png"
    p.write_bytes(PNG_1PX)
    block = load_attachment(p)
    assert isinstance(block, ImageBlock)
    assert block.media_type == "image/png"
    assert base64.b64decode(block.data_b64) == PNG_1PX
    assert block.name == "pixel.png"


def test_load_pdf(tmp_path):
    from pypdf import PdfWriter

    p = tmp_path / "empty.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with open(p, "wb") as fh:
        writer.write(fh)
    block = load_attachment(p)
    assert isinstance(block, PdfBlock)
    assert block.name == "empty.pdf"
    assert "page 1" in block.extracted_text


def test_missing_file():
    with pytest.raises(AttachmentError, match="No such file"):
        load_attachment("/nonexistent/nope.pdf")


def test_unsupported_extension(tmp_path):
    p = tmp_path / "data.xlsx"
    p.write_bytes(b"not really")
    with pytest.raises(AttachmentError, match="Unsupported"):
        load_attachment(p)
