from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Provider(str, Enum):
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"


class ExtractionStats(BaseModel):
    file_name: str
    mime_type: str
    extension: str
    pages: int = 1
    text_blocks: int = 0
    tables_found: int = 0
    characters: int = 0
    used_ocr: bool = False
    warnings: list[str] = Field(default_factory=list)


class ProcessResponse(BaseModel):
    message: str
    provider: Provider
    transport: str
    export_file_name: str
    report_file_name: str
    extraction: ExtractionStats
    json_result: dict[str, Any]
    report_markdown: str
