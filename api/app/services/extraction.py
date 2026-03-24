import base64
import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber
import pypdfium2 as pdfium
from PIL import Image

from api.app.config import Settings


SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt", ".csv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class PreparedDocument:
    file_name: str
    mime_type: str
    extension: str
    extracted_text: str
    tables: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    used_ocr: bool = False
    ocr_inputs: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def validate_file(file_name: str) -> str:
    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension or 'unknown'}")
    return extension


def prepare_document(
    *,
    file_name: str,
    content_type: str | None,
    data: bytes,
    settings: Settings,
) -> PreparedDocument:
    extension = validate_file(file_name)
    mime_type = content_type or _guess_mime_type(extension)

    if extension in {".txt", ".csv"}:
        return _prepare_text_document(file_name, extension, mime_type, data)
    if extension in IMAGE_EXTENSIONS:
        return _prepare_image_document(file_name, extension, mime_type, data)
    return _prepare_pdf_document(file_name, extension, mime_type, data, settings)


def _prepare_text_document(file_name: str, extension: str, mime_type: str, data: bytes) -> PreparedDocument:
    decoded = _decode_bytes(data)
    tables: list[dict[str, Any]] = []

    if extension == ".csv":
        reader = csv.reader(io.StringIO(decoded))
        rows = [row for row in reader]
        if rows:
            headers = rows[0]
            table_rows = rows[1:] if len(rows) > 1 else []
            tables.append(
                {
                    "name": "csv_data",
                    "columns": headers,
                    "rows": table_rows,
                }
            )

    return PreparedDocument(
        file_name=file_name,
        mime_type=mime_type,
        extension=extension,
        extracted_text=decoded.strip(),
        tables=tables,
        stats={
            "pages": 1,
            "text_blocks": max(1, len([line for line in decoded.splitlines() if line.strip()])),
            "tables_found": len(tables),
            "characters": len(decoded),
        },
    )


def _prepare_image_document(file_name: str, extension: str, mime_type: str, data: bytes) -> PreparedDocument:
    image = Image.open(io.BytesIO(data))
    image.load()

    png_buffer = io.BytesIO()
    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGB")
    image.save(png_buffer, format="PNG")
    data_url = _to_data_url(png_buffer.getvalue(), "image/png")

    return PreparedDocument(
        file_name=file_name,
        mime_type=mime_type,
        extension=extension,
        extracted_text="",
        tables=[],
        used_ocr=True,
        ocr_inputs=[{"page_number": 1, "image_url": data_url}],
        stats={
            "pages": 1,
            "text_blocks": 0,
            "tables_found": 0,
            "characters": 0,
        },
    )


def _prepare_pdf_document(
    file_name: str,
    extension: str,
    mime_type: str,
    data: bytes,
    settings: Settings,
) -> PreparedDocument:
    warnings: list[str] = []
    tables: list[dict[str, Any]] = []
    page_texts: list[str] = []
    text_blocks = 0

    doc = pdfium.PdfDocument(data)
    page_count = len(doc)

    for page_index in range(page_count):
        page = doc[page_index]
        text_page = page.get_textpage()
        text = (text_page.get_text_bounded() or "").strip()
        if text:
            page_texts.append(f"[Page {page_index + 1}]\n{text}")

        line_count = len([line for line in text.splitlines() if line.strip()])
        if line_count == 0 and text:
            line_count = 1
        text_blocks += line_count

        text_page.close()
        page.close()

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page_index in range(len(pdf.pages)):
            page = pdf.pages[page_index]
            raw_tables = page.extract_tables() or []
            for table_index, raw_table in enumerate(raw_tables, start=1):
                rows = [
                    [(cell or "").strip() for cell in row]
                    for row in raw_table
                    if row and any((cell or "").strip() for cell in row)
                ]
                if not rows:
                    continue
                tables.append(
                    {
                        "name": f"page_{page_index}_table_{table_index}",
                        "columns": rows[0],
                        "rows": rows[1:] if len(rows) > 1 else [],
                    }
                )

    extracted_text = "\n\n".join(page_texts).strip()
    needs_ocr = len(extracted_text) < max(120, page_count * 40)
    ocr_inputs: list[dict[str, Any]] = []

    if needs_ocr:
        limited_pages = min(page_count, settings.max_ocr_pages)
        for page_index in range(limited_pages):
            page = doc[page_index]
            bitmap = page.render(scale=170 / 72)
            pil_image = bitmap.to_pil()
            png_buffer = io.BytesIO()
            pil_image.save(png_buffer, format="PNG")
            ocr_inputs.append(
                {
                    "page_number": page_index + 1,
                    "image_url": _to_data_url(png_buffer.getvalue(), "image/png"),
                }
            )
            bitmap.close()
            page.close()
        if page_count > limited_pages:
            warnings.append(
                f"OCR was limited to the first {limited_pages} pages to control token usage."
            )

    return PreparedDocument(
        file_name=file_name,
        mime_type=mime_type,
        extension=extension,
        extracted_text=extracted_text,
        tables=tables,
        warnings=warnings,
        used_ocr=needs_ocr,
        ocr_inputs=ocr_inputs,
        stats={
            "pages": page_count,
            "text_blocks": text_blocks,
            "tables_found": len(tables),
            "characters": len(extracted_text),
        },
    )


def _decode_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _to_data_url(binary_data: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(binary_data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _guess_mime_type(extension: str) -> str:
    return {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".txt": "text/plain",
        ".csv": "text/csv",
    }[extension]
