from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.app.config import get_settings
from api.app.models.schemas import ExtractionStats, ProcessResponse, Provider
from api.app.services.ai import convert_document
from api.app.services.extraction import prepare_document
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

    result, transport, trace = await convert_document(prepared, provider, settings)
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
        used_ocr=prepared.used_ocr,
        warnings=result.get("warnings", prepared.warnings),
    )

    report_markdown = generate_technical_report(
        prepared=prepared,
        extraction=extraction,
        provider=provider,
        settings=settings,
        transport=transport,
        trace=trace,
    )

    provider_label = "DeepSeek" if provider is Provider.DEEPSEEK else "Gemini 3.0"
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
