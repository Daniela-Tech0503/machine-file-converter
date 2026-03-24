import json
from typing import Any

import httpx

from api.app.config import Settings
from api.app.models.schemas import Provider
from api.app.services.extraction import PreparedDocument


CONVERSION_SCHEMA: dict[str, Any] = {
    "name": "machine_readable_document",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "document": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": ["string", "null"]},
                    "type": {"type": "string"},
                    "language": {"type": "string"},
                    "source_format": {"type": "string"},
                    "ocr_applied": {"type": "boolean"}
                },
                "required": ["title", "type", "language", "source_format", "ocr_applied"]
            },
            "summary": {"type": "string"},
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "label": {"type": "string"},
                        "value": {"type": "string"}
                    },
                    "required": ["label", "value"]
                }
            },
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "heading": {"type": "string"},
                        "text": {"type": "string"}
                    },
                    "required": ["heading", "text"]
                }
            },
            "tables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "rows": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    },
                    "required": ["name", "columns", "rows"]
                }
            },
            "raw_text": {"type": "string"},
            "warnings": {"type": "array", "items": {"type": "string"}},
            "antigravity_export": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "comparison_ready": {"type": "boolean"},
                    "format_version": {"type": "string"}
                },
                "required": ["comparison_ready", "format_version"]
            }
        },
        "required": [
            "document",
            "summary",
            "entities",
            "sections",
            "tables",
            "raw_text",
            "warnings",
            "antigravity_export"
        ]
    }
}

OCR_SCHEMA: dict[str, Any] = {
    "name": "ocr_result",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "pages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "page_number": {"type": "integer"},
                        "text": {"type": "string"}
                    },
                    "required": ["page_number", "text"]
                }
            },
            "warnings": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["pages", "warnings"]
    }
}


async def convert_document(
    prepared: PreparedDocument,
    provider: Provider,
    settings: Settings,
) -> tuple[dict[str, Any], str]:
    text_content = prepared.extracted_text
    combined_warnings = list(prepared.warnings)

    if prepared.ocr_inputs:
        ocr_result = await _run_ocr(prepared, settings)
        combined_warnings.extend(ocr_result.get("warnings", []))
        ocr_text = "\n\n".join(
            page["text"].strip() for page in ocr_result.get("pages", []) if page["text"].strip()
        ).strip()
        if ocr_text:
            text_content = "\n\n".join(part for part in [text_content, ocr_text] if part).strip()
        if not text_content:
            combined_warnings.append("OCR did not return readable text.")

    text_content = text_content.strip()

    if provider is Provider.DEEPSEEK:
        result = await _call_deepseek(prepared, text_content, combined_warnings, settings)
        transport = "deepseek_direct"
    else:
        result = await _call_openrouter(prepared, text_content, combined_warnings, settings)
        transport = "openrouter"

    return _normalize_result(result, prepared, text_content, combined_warnings), transport


async def _run_ocr(prepared: PreparedDocument, settings: Settings) -> dict[str, Any]:
    if not settings.openrouter_api_key:
        return {"pages": [], "warnings": ["OpenRouter key missing, OCR fallback skipped."]}

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Perform OCR on the provided pages and return strict json. "
                "Preserve the original wording as closely as possible. "
                "If a page is unreadable, return an empty string for that page and add a warning."
            ),
        }
    ]

    for image in prepared.ocr_inputs:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image["image_url"]},
            }
        )

    payload = {
        "model": settings.openrouter_ocr_model,
        "messages": [{"role": "user", "content": content}],
        "response_format": {
            "type": "json_schema",
            "json_schema": OCR_SCHEMA,
        },
        "temperature": 0,
    }

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "Machine Reader Chat",
                },
                json=payload,
            )
            response.raise_for_status()

        data = response.json()
        content_text = _coerce_message_content(data["choices"][0]["message"]["content"])
        result = json.loads(content_text)
    except Exception as exc:  # noqa: BLE001
        return {"pages": [], "warnings": [f"OCR request failed: {exc}"]}

    pages = result.get("pages", [])
    for index, page_number in enumerate(prepared.ocr_inputs):
        if index < len(pages):
            pages[index]["page_number"] = page_number["page_number"]
    return result


async def _call_deepseek(
    prepared: PreparedDocument,
    text_content: str,
    warnings: list[str],
    settings: Settings,
) -> dict[str, Any]:
    if not settings.deepseek_api_key:
        fallback = _build_local_fallback(prepared, text_content, warnings)
        fallback["warnings"].append("DeepSeek key missing, returned local fallback JSON.")
        return fallback

    prompt = _build_conversion_prompt(prepared, text_content, warnings)
    payload = {
        "model": settings.deepseek_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You convert extracted document content into strict json. "
                    "Always answer with valid json and include the word json in your reasoning target."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 1800,
    }

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.deepseek_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
        data = response.json()
        return json.loads(_coerce_message_content(data["choices"][0]["message"]["content"]))
    except Exception as exc:  # noqa: BLE001
        fallback = _build_local_fallback(prepared, text_content, warnings)
        fallback["warnings"].append(f"DeepSeek request failed: {exc}")
        return fallback


async def _call_openrouter(
    prepared: PreparedDocument,
    text_content: str,
    warnings: list[str],
    settings: Settings,
) -> dict[str, Any]:
    if not settings.openrouter_api_key:
        fallback = _build_local_fallback(prepared, text_content, warnings)
        fallback["warnings"].append("OpenRouter key missing, returned local fallback JSON.")
        return fallback

    payload = {
        "model": settings.openrouter_gemini_model,
        "messages": [
            {
                "role": "system",
                "content": "Convert extracted content into strict machine-readable JSON.",
            },
            {
                "role": "user",
                "content": _build_conversion_prompt(prepared, text_content, warnings),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": CONVERSION_SCHEMA,
        },
        "temperature": 0.1,
    }

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "Machine Reader Chat",
                },
                json=payload,
            )
            response.raise_for_status()
        data = response.json()
        return json.loads(_coerce_message_content(data["choices"][0]["message"]["content"]))
    except Exception as exc:  # noqa: BLE001
        fallback = _build_local_fallback(prepared, text_content, warnings)
        fallback["warnings"].append(f"OpenRouter request failed: {exc}")
        return fallback


def _build_conversion_prompt(
    prepared: PreparedDocument,
    text_content: str,
    warnings: list[str],
) -> str:
    context = {
        "file_name": prepared.file_name,
        "mime_type": prepared.mime_type,
        "extension": prepared.extension,
        "used_ocr": prepared.used_ocr,
        "warnings": warnings,
        "tables": prepared.tables[:10],
        "text": text_content[:14000],
        "required_json_shape": CONVERSION_SCHEMA["schema"],
    }
    return (
        "Return valid json only. Build a machine-readable export for Antigravity comparison. "
        "Use the provided required_json_shape exactly, keep unknown values conservative, and keep raw_text faithful to the extraction.\n\n"
        f"INPUT_JSON:\n{json.dumps(context, ensure_ascii=True)}"
    )


def _build_local_fallback(
    prepared: PreparedDocument,
    text_content: str,
    warnings: list[str],
) -> dict[str, Any]:
    section_text = text_content[:4000]
    return {
        "document": {
            "title": prepared.file_name,
            "type": prepared.extension.replace(".", "").upper(),
            "language": "unknown",
            "source_format": prepared.extension.replace(".", ""),
            "ocr_applied": prepared.used_ocr,
        },
        "summary": "Local fallback JSON generated from extracted content.",
        "entities": [],
        "sections": [
            {
                "heading": "Extracted Content",
                "text": section_text,
            }
        ] if section_text else [],
        "tables": [
            {
                "name": table.get("name", "table"),
                "columns": [str(cell) for cell in table.get("columns", [])],
                "rows": [
                    [str(cell) for cell in row]
                    for row in table.get("rows", [])
                ],
            }
            for table in prepared.tables[:10]
        ],
        "raw_text": text_content,
        "warnings": warnings,
        "antigravity_export": {
            "comparison_ready": True,
            "format_version": "1.0",
        },
    }


def _normalize_result(
    result: dict[str, Any],
    prepared: PreparedDocument,
    text_content: str,
    warnings: list[str],
) -> dict[str, Any]:
    fallback = _build_local_fallback(prepared, text_content, warnings)

    document = result.get("document") if isinstance(result.get("document"), dict) else {}
    antigravity_export = (
        result.get("antigravity_export")
        if isinstance(result.get("antigravity_export"), dict)
        else {}
    )

    normalized = {
        "document": {
            "title": document.get("title", fallback["document"]["title"]),
            "type": str(document.get("type", fallback["document"]["type"])),
            "language": str(document.get("language", fallback["document"]["language"])),
            "source_format": str(
                document.get("source_format", fallback["document"]["source_format"])
            ),
            "ocr_applied": bool(document.get("ocr_applied", fallback["document"]["ocr_applied"])),
        },
        "summary": str(result.get("summary", fallback["summary"])),
        "entities": _normalize_entities(result.get("entities")),
        "sections": _normalize_sections(result.get("sections"), fallback["sections"]),
        "tables": _normalize_tables(result.get("tables"), fallback["tables"]),
        "raw_text": str(result.get("raw_text", fallback["raw_text"])),
        "warnings": _merge_warnings(warnings, result.get("warnings")),
        "antigravity_export": {
            "comparison_ready": bool(
                antigravity_export.get(
                    "comparison_ready", fallback["antigravity_export"]["comparison_ready"]
                )
            ),
            "format_version": str(
                antigravity_export.get(
                    "format_version", fallback["antigravity_export"]["format_version"]
                )
            ),
        },
    }

    if not normalized["sections"] and normalized["raw_text"]:
        normalized["sections"] = fallback["sections"]

    return normalized


def _normalize_entities(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    entities: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        entity_value = item.get("value")
        if label is None or entity_value is None:
            continue
        entities.append({"label": str(label), "value": str(entity_value)})
    return entities


def _normalize_sections(value: Any, fallback: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return [
            {"heading": str(item["heading"]), "text": str(item["text"])}
            for item in fallback
        ]

    sections: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        heading = item.get("heading")
        text = item.get("text")
        if heading is None or text is None:
            continue
        sections.append({"heading": str(heading), "text": str(text)})
    return sections


def _normalize_tables(value: Any, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source = value if isinstance(value, list) else fallback
    tables: list[dict[str, Any]] = []
    for item in source:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "table"))
        columns = [str(column) for column in item.get("columns", []) if column is not None]
        rows: list[list[str]] = []
        raw_rows = item.get("rows", [])
        if isinstance(raw_rows, list):
            for row in raw_rows:
                if isinstance(row, list):
                    rows.append([str(cell) for cell in row if cell is not None])
        tables.append({"name": name, "columns": columns, "rows": rows})
    return tables


def _merge_warnings(existing: list[str], incoming: Any) -> list[str]:
    merged = [str(warning) for warning in existing]
    if isinstance(incoming, list):
        merged.extend(str(warning) for warning in incoming)

    unique: list[str] = []
    seen: set[str] = set()
    for warning in merged:
        if warning and warning not in seen:
            unique.append(warning)
            seen.add(warning)
    return unique


def _coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return "".join(text_parts)
    raise ValueError("Model response did not contain text content.")
