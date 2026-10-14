"""Opt-in Linux/macOS LibreOffice conversions; no network or OpenAI calls.

Install LibreOffice Writer and Impress, then run with RUN_LIBREOFFICE_SMOKE=1.
An explicitly requested run fails when the converter is missing or unusable.
"""

import os
from pathlib import Path
import sys

from docx import Document
from pptx import Presentation
from pypdf import PdfReader
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_chatbot import office_preview


pytestmark = [
    pytest.mark.skipif(os.name == "nt", reason="Exercises the Linux/macOS LibreOffice converter"),
    pytest.mark.skipif(os.environ.get("RUN_LIBREOFFICE_SMOKE") != "1",
                       reason="Set RUN_LIBREOFFICE_SMOKE=1 to run real LibreOffice conversion"),
]


@pytest.mark.parametrize("extension", [".docx", ".pptx"])
def test_real_libreoffice_preview_preserves_source_and_reuses_cache(tmp_path, monkeypatch, extension):
    workspace = tmp_path / "Managed workspace ü"
    directory = workspace / "sources" / "smoke-document"
    directory.mkdir(parents=True)
    original = directory / f"original{extension}"
    markers = ["Preview smoke page one", "Preview smoke page two"]
    if extension == ".docx":
        document = Document()
        document.add_paragraph(markers[0])
        document.add_page_break()
        document.add_paragraph(markers[1])
        document.save(original)
    else:
        presentation = Presentation()
        for marker in markers:
            slide = presentation.slides.add_slide(presentation.slide_layouts[0])
            slide.shapes.title.text = marker
        presentation.save(original)

    original_bytes = original.read_bytes()
    original_mtime = original.stat().st_mtime_ns
    preview = office_preview.get_pdf_preview(original, data_dir=workspace)
    assert preview.parent == directory / ".preview"
    assert preview.is_file() and preview.stat().st_size > 0
    with preview.open("rb") as handle:
        pages = PdfReader(handle).pages
        assert len(pages) == 2
        for page, marker in zip(pages, markers):
            assert float(page.mediabox.width) > 0 and float(page.mediabox.height) > 0
            assert marker in " ".join(page.extract_text().split())
    preview_bytes = preview.read_bytes()
    preview_mtime = preview.stat().st_mtime_ns

    def unexpected_conversion(*args, **kwargs):
        pytest.fail("The second preview must reuse the cached PDF without launching LibreOffice")

    monkeypatch.setattr(office_preview, "_run_worker", unexpected_conversion)
    assert office_preview.get_pdf_preview(original, data_dir=workspace) == preview
    assert preview.read_bytes() == preview_bytes
    assert preview.stat().st_mtime_ns == preview_mtime
    assert original.read_bytes() == original_bytes
    assert original.stat().st_mtime_ns == original_mtime
    assert not list(preview.parent.glob(".converting-*"))
