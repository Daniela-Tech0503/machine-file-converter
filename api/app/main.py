from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.app.config import get_settings
from api.app.models.schemas import (
    ExtractionStats,
    PageInfo,
    PageProcessResponse,
    ProcessResponse,
    Provider,
    SplitResponse,
)
from api.app.services.ai import convert_document, convert_page
from api.app.services.extraction import (
    IMAGE_EXTENSIONS,
    prepare_document,
    split_image_page,
    split_pdf_pages,
    split_text_page,
    validate_file,
)
from api.app.services.reporting import generate_technical_report

settings = get_settings()
app = FastAPI(title="Machine Reader Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/process", response_model=ProcessResponse)
async def process_file(
    file: UploadFile = File(...),
    provider: Provider = Form(...),
    instructions: str = Form(""),
) -> ProcessResponse:
    data = await file.read()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large. Limit is {settings.max_upload_mb} MB.",
        )

    try:
        prepared = prepare_document(
            file_name=file.filename,
            content_type=file.content_type,
            data=data,
            settings=settings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to read file: {exc}") from exc

    result, transport, trace = await convert_document(prepared, provider, settings, instructions=instructions)
    export_file_name = f"{Path(file.filename).stem}.machine-readable.json"
    report_file_name = f"{Path(file.filename).stem}.pipeline-report.md"

    extraction = ExtractionStats(
        file_name=prepared.file_name,
        mime_type=prepared.mime_type,
        extension=prepared.extension,
        pages=prepared.stats.get("pages", 1),
        text_blocks=prepared.stats.get("text_blocks", 0),
        tables_found=prepared.stats.get("tables_found", 0),
        characters=len(result.get("raw_text", prepared.extracted_text)),
        used_ocr=trace.ocr_succeeded,
        ocr_requested=trace.ocr_requested,
        ocr_attempted=trace.ocr_attempted,
        ocr_succeeded=trace.ocr_succeeded,
        warnings=result.get("warnings", prepared.warnings),
    )

    report_markdown = generate_technical_report(
        prepared=prepared,
        extraction=extraction,
        provider=provider,
        settings=settings,
        transport=transport,
        trace=trace,
        instructions=instructions,
    )

    provider_label = "DeepSeek" if provider is Provider.DEEPSEEK else "Gemini 2.5 Pro"
    return ProcessResponse(
        message=f"Converted {file.filename} with {provider_label}.",
        provider=provider,
        transport=transport,
        export_file_name=export_file_name,
        report_file_name=report_file_name,
        extraction=extraction,
        json_result=result,
        report_markdown=report_markdown,
    )


# ---------------------------------------------------------------------------
# Parallel pipeline — Step 1: split the document into pages
# ---------------------------------------------------------------------------

@app.post("/api/split-pages", response_model=SplitResponse)
async def split_pages(file: UploadFile = File(...)) -> SplitResponse:
    """Split a document into pages and return metadata so the frontend can build a FIFO queue."""
    data = await file.read()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large. Limit is {settings.max_upload_mb} MB.",
        )

    try:
        extension = validate_file(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        if extension == ".pdf":
            pages = split_pdf_pages(data, settings)
        elif extension in IMAGE_EXTENSIONS:
            pages = [split_image_page(data)]
        else:
            pages = [split_text_page(data, extension)]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to split file: {exc}") from exc

    page_infos = [
        PageInfo(
            page_number=p.page_number,
            has_text=bool(p.text.strip()),
            needs_ocr=p.needs_ocr,
            text_preview=p.text[:120],
        )
        for p in pages
    ]

    return SplitResponse(
        file_name=file.filename,
        extension=extension,
        total_pages=len(pages),
        pages=page_infos,
    )


# ---------------------------------------------------------------------------
# Parallel pipeline — Step 2: process one page (OCR only, no heavy AI)
# ---------------------------------------------------------------------------

@app.post("/api/process-page", response_model=PageProcessResponse)
async def process_single_page(
    file: UploadFile = File(...),
    page_number: int = Form(...),
    provider: Provider = Form(...),
) -> PageProcessResponse:
    """Extract text (+ OCR if needed) from a single page.

    The frontend calls this endpoint once per page, up to CONCURRENCY workers
    in parallel, so each call completes in 2–8 seconds — well within Vercel limits.
    """
    data = await file.read()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")

    try:
        extension = validate_file(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        if extension == ".pdf":
            all_pages = split_pdf_pages(data, settings)
            matched = [p for p in all_pages if p.page_number == page_number]
            if not matched:
                raise HTTPException(status_code=400, detail=f"Page {page_number} not found.")
            page = matched[0]
        elif extension in IMAGE_EXTENSIONS:
            page = split_image_page(data)
        else:
            page = split_text_page(data, extension)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to read page: {exc}") from exc

    result = await convert_page(
        page=page,
        provider=provider,
        settings=settings,
        file_name=file.filename,
    )
    return result
