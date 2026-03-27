import json
from typing import Any

import httpx

from api.app.config import Settings
from api.app.models.schemas import PageProcessResponse, Provider
from api.app.services.extraction import PageData, PreparedDocument
from api.app.services.reporting import PipelineTrace


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

STALE_OPENROUTER_MODEL_ALIASES = {
    "google/gemini-3-pro-preview": "google/gemini-2.5-pro",
}


async def convert_document(
    prepared: PreparedDocument,
    provider: Provider,
    settings: Settings,
    instructions: str = "",
) -> tuple[dict[str, Any], str, PipelineTrace]:
    text_content = prepared.extracted_text
    combined_warnings = list(prepared.warnings)
    combined_warnings.extend(_settings_warnings(settings))
    user_instructions = instructions.strip() or None
    trace = PipelineTrace(
        frontend_provider=provider,
        frontend_label="DeepSeek" if provider is Provider.DEEPSEEK else "Gemini 2.5 Pro",
        user_instructions=user_instructions,
        local_processing_steps=_build_local_processing_steps(prepared),
        available_fallbacks=[
            "Skip OCR if OpenRouter key is missing",
            "Return local fallback JSON if DeepSeek fails or key is missing",
            "Return local fallback JSON if OpenRouter JSON call fails or key is missing",
        ],
        ocr_requested=bool(prepared.ocr_inputs),
    )

    if prepared.ocr_inputs:
        ocr_result = await _run_ocr(prepared, settings)
        trace.ocr_attempted = True
        trace.ocr_transport = "openrouter"
        trace.ocr_provider_name = "OpenRouter"
        trace.ocr_model_alias = settings.openrouter_ocr_model
        trace.ocr_reads_file = True
        trace.ocr_trigger_reason = _build_ocr_reason(prepared)
        if settings.openrouter_ocr_model not in trace.openrouter_aliases:
            trace.openrouter_aliases.append(settings.openrouter_ocr_model)
        combined_warnings.extend(ocr_result.get("warnings", []))
        ocr_text = "\n\n".join(
            page["text"].strip() for page in ocr_result.get("pages", []) if page["text"].strip()
        ).strip()
        if ocr_text:
            trace.ocr_succeeded = True
            trace.ocr_used = True
            text_content = "\n\n".join(part for part in [text_content, ocr_text] if part).strip()
        else:
            trace.ocr_failure_reason = _summarize_ocr_failure(ocr_result.get("warnings", []))
        if trace.ocr_attempted and not trace.ocr_succeeded and not text_content:
            combined_warnings.append("OCR did not return readable text.")
        if ocr_result.get("warnings"):
            trace.fallback_events.append("OCR warnings were returned by the OpenRouter OCR stage")

    text_content = text_content.strip()
    ocr_applied = trace.ocr_succeeded

    if provider is Provider.DEEPSEEK:
        result = await _call_deepseek(prepared, text_content, combined_warnings, settings, user_instructions, ocr_applied)
        transport = "deepseek_direct"
        trace.json_transport = transport
        trace.json_provider_name = "DeepSeek"
        trace.json_model_alias = settings.deepseek_model
        trace.json_reads_file = False
        trace.json_generated_by = f"{settings.deepseek_model} via DeepSeek direct API"
        trace.direct_api_calls.append("DeepSeek API: https://api.deepseek.com/chat/completions")
    else:
        result = await _call_openrouter(prepared, text_content, combined_warnings, settings, user_instructions, ocr_applied)
        transport = "openrouter"
        trace.json_transport = transport
        trace.json_provider_name = "OpenRouter"
        trace.json_model_alias = settings.openrouter_gemini_model
        trace.json_reads_file = False
        trace.json_generated_by = f"{settings.openrouter_gemini_model} via OpenRouter"
        trace.openrouter_aliases.append(settings.openrouter_gemini_model)

    normalized = _normalize_result(result, prepared, text_content, combined_warnings, ocr_applied)
    if _warnings_include_deepseek_failure(normalized["warnings"]):
        trace.fallback_events.append("DeepSeek JSON call failed and local fallback JSON was returned")
        trace.json_transport = "local_backend_fallback"
        trace.json_provider_name = "local_backend"
        trace.json_model_alias = None
        trace.json_generated_by = "Local backend fallback JSON generator"
    if _warnings_include_openrouter_failure(normalized["warnings"]):
        trace.fallback_events.append("OpenRouter JSON call failed and local fallback JSON was returned")
        trace.json_transport = "local_backend_fallback"
        trace.json_provider_name = "local_backend"
        trace.json_model_alias = None
        trace.json_generated_by = "Local backend fallback JSON generator"
    if any("OpenRouter key missing" in warning for warning in normalized["warnings"]):
        trace.fallback_events.append("OpenRouter was unavailable for at least one stage")
    if any("DeepSeek key missing" in warning for warning in normalized["warnings"]):
        trace.fallback_events.append("DeepSeek was unavailable and local fallback JSON was used")

    trace.openrouter_aliases = _dedupe(trace.openrouter_aliases)
    trace.direct_api_calls = _dedupe(trace.direct_api_calls)
    trace.fallback_events = _dedupe(trace.fallback_events)

    return normalized, transport, trace


async def convert_page(
    page: PageData,
    provider: Provider,
    settings: Settings,
    file_name: str = "",
    instructions: str = "",
) -> PageProcessResponse:
    """Process a single page through OCR (if needed) and return its text + tables.

    This is intentionally lightweight — no full JSON structuring.  The frontend
    collects all PageProcessResponse objects and sends them to a final
    /api/assemble call (or assembles them locally) to build the merged JSON.
    """
    warnings: list[str] = []
    text = page.text
    ocr_applied = False
    transport = "local"

    # OCR step — only when the page has an image and no readable text
    if page.needs_ocr and page.image_url:
        if not settings.openrouter_api_key:
            warnings.append("OpenRouter key missing, OCR skipped for this page.")
        else:
            ocr_result = await _run_ocr_single_image(
                image_url=page.image_url,
                page_number=page.page_number,
                settings=settings,
            )
            warnings.extend(ocr_result.get("warnings", []))
            ocr_text = ocr_result.get("text", "").strip()
            if ocr_text:
                text = ocr_text
                ocr_applied = True
                transport = "openrouter_ocr"
            else:
                warnings.append(f"OCR returned no text for page {page.page_number}.")

    return PageProcessResponse(
        page_number=page.page_number,
        text=text,
        tables=page.tables,
        warnings=warnings,
        ocr_applied=ocr_applied,
        transport=transport,
    )


async def _run_ocr_single_image(
    image_url: str,
    page_number: int,
    settings: Settings,
) -> dict[str, Any]:
    """Send a single page image to OpenRouter for OCR and return {text, warnings}."""
    payload = {
        "model": settings.openrouter_ocr_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Perform OCR on page {page_number}. "
                            "Return only the extracted text, preserving layout as much as possible. "
                            "Do not add commentary."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": 0,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
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
        text = _extract_message_content("OpenRouter OCR", data)
        return {"text": text, "warnings": []}
    except Exception as exc:  # noqa: BLE001
        return {
            "text": "",
            "warnings": [_format_external_error("OpenRouter OCR", exc, settings.openrouter_ocr_model)],
        }


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
        async with httpx.AsyncClient(timeout=15.0) as client:
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
        content_text = _extract_message_content("OpenRouter OCR", data)
        result = json.loads(content_text)
    except Exception as exc:  # noqa: BLE001
        return {"pages": [], "warnings": [_format_external_error("OpenRouter OCR", exc, settings.openrouter_ocr_model)]}

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
    user_instructions: str | None,
    ocr_applied: bool,
) -> dict[str, Any]:
    if not settings.deepseek_api_key:
        fallback = _build_local_fallback(prepared, text_content, warnings, ocr_applied)
        fallback["warnings"].append("DeepSeek key missing, returned local fallback JSON.")
        return fallback

    prompt = _build_deepseek_prompt(prepared, text_content, warnings, user_instructions, ocr_applied)
    payloads = [
        {
            "model": settings.deepseek_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You convert extracted document content into strict json. "
                        "Always answer with valid json matching the requested shape exactly. "
                        "Keep the response compact and do not repeat the full source text in multiple fields."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": 3200,
        },
        {
            "model": settings.deepseek_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return valid JSON only. Do not wrap the answer in markdown fences. "
                        "Do not add commentary before or after the JSON. "
                        "Keep section text concise and leave raw_text as an empty string because the backend stores it locally."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 4200,
        },
    ]

    errors: list[str] = []
    for attempt_index, payload in enumerate(payloads, start=1):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
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
            content = _extract_message_content("DeepSeek", data)
            return _parse_json_response(content)
        except Exception as exc:  # noqa: BLE001
            errors.append(_format_external_error("DeepSeek", exc, settings.deepseek_model))
            if attempt_index == 1:
                continue

    fallback = _build_local_fallback(prepared, text_content, warnings, ocr_applied)
    fallback["warnings"].append(
        "DeepSeek returned malformed JSON for this document, so the backend used local fallback JSON."
    )
    fallback["warnings"].extend(_dedupe(errors))
    return fallback


async def _call_openrouter(
    prepared: PreparedDocument,
    text_content: str,
    warnings: list[str],
    settings: Settings,
    user_instructions: str | None,
    ocr_applied: bool,
) -> dict[str, Any]:
    if not settings.openrouter_api_key:
        fallback = _build_local_fallback(prepared, text_content, warnings, ocr_applied)
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
                "content": _build_conversion_prompt(prepared, text_content, warnings, user_instructions, ocr_applied),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": CONVERSION_SCHEMA,
        },
        "temperature": 0.1,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
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
        return json.loads(_extract_message_content("OpenRouter", data))
    except Exception as exc:  # noqa: BLE001
        fallback = _build_local_fallback(prepared, text_content, warnings, ocr_applied)
        fallback["warnings"].append(_format_external_error("OpenRouter", exc, settings.openrouter_gemini_model))
        return fallback


def _build_conversion_prompt(
    prepared: PreparedDocument,
    text_content: str,
    warnings: list[str],
    user_instructions: str | None,
    ocr_applied: bool,
) -> str:
    context = {
        "file_name": prepared.file_name,
        "mime_type": prepared.mime_type,
        "extension": prepared.extension,
        "ocr_requested": prepared.used_ocr,
        "ocr_applied": ocr_applied,
        "user_instructions": user_instructions,
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


def _build_deepseek_prompt(
    prepared: PreparedDocument,
    text_content: str,
    warnings: list[str],
    user_instructions: str | None,
    ocr_applied: bool,
) -> str:
    context = {
        "file_name": prepared.file_name,
        "mime_type": prepared.mime_type,
        "extension": prepared.extension,
        "ocr_requested": prepared.used_ocr,
        "ocr_applied": ocr_applied,
        "user_instructions": user_instructions,
        "warnings": warnings,
        "tables": prepared.tables[:5],
        "text": text_content[:12000],
        "required_keys": [
            "document",
            "summary",
            "entities",
            "sections",
            "tables",
            "raw_text",
            "warnings",
            "antigravity_export",
        ],
    }
    return (
        "Return valid JSON only. Build a machine-readable export for Antigravity comparison. "
        "Preserve the meaning of the document, but keep the output compact. "
        "Use concise section texts, keep `tables` as an empty array when unsure, and set `raw_text` to an empty string because the backend will inject the extracted raw text locally.\n\n"
        f"INPUT_JSON:\n{json.dumps(context, ensure_ascii=True)}"
    )


def _build_local_fallback(
    prepared: PreparedDocument,
    text_content: str,
    warnings: list[str],
    ocr_applied: bool,
) -> dict[str, Any]:
    section_text = text_content[:4000]
    return {
        "document": {
            "title": prepared.file_name,
            "type": prepared.extension.replace(".", "").upper(),
            "language": "unknown",
            "source_format": prepared.extension.replace(".", ""),
            "ocr_applied": ocr_applied,
        },
        "summary": (
            "Local fallback JSON generated from extracted content."
            if section_text
            else "Document extraction failed due to OCR or model errors, resulting in no readable content."
        ),
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
            "comparison_ready": bool(section_text.strip()),
            "format_version": "1.0",
        },
    }


def _normalize_result(
    result: dict[str, Any],
    prepared: PreparedDocument,
    text_content: str,
    warnings: list[str],
    ocr_applied: bool,
) -> dict[str, Any]:
    fallback = _build_local_fallback(prepared, text_content, warnings, ocr_applied)
    fallback_document = fallback["document"]
    fallback_antigravity = fallback["antigravity_export"]

    document_candidate = result.get("document")
    document: dict[str, Any] = document_candidate if isinstance(document_candidate, dict) else {}

    antigravity_candidate = result.get("antigravity_export")
    antigravity_export: dict[str, Any] = (
        antigravity_candidate if isinstance(antigravity_candidate, dict) else {}
    )

    normalized = {
        "document": {
            "title": document.get("title", fallback_document["title"]),
            "type": str(document.get("type", fallback_document["type"])),
            "language": str(document.get("language", fallback_document["language"])),
            "source_format": str(document.get("source_format", fallback_document["source_format"])),
            "ocr_applied": bool(document.get("ocr_applied", fallback_document["ocr_applied"])),
        },
        "summary": str(result.get("summary", fallback["summary"])),
        "entities": _normalize_entities(result.get("entities")),
        "sections": _normalize_sections(result.get("sections"), fallback["sections"]),
        "tables": _normalize_tables(result.get("tables"), fallback["tables"]),
        "raw_text": str(result.get("raw_text", fallback["raw_text"])),
        "warnings": _merge_warnings(warnings, result.get("warnings")),
        "antigravity_export": {
            "comparison_ready": bool(
                antigravity_export.get("comparison_ready", fallback_antigravity["comparison_ready"])
            ),
            "format_version": str(antigravity_export.get("format_version", fallback_antigravity["format_version"])),
        },
    }

    if not normalized["sections"] and normalized["raw_text"]:
        normalized["sections"] = fallback["sections"]
    if not normalized["tables"] and fallback["tables"]:
        normalized["tables"] = fallback["tables"]
    if not normalized["raw_text"].strip():
        normalized["raw_text"] = fallback["raw_text"]
        normalized["antigravity_export"]["comparison_ready"] = False
    if not normalized["sections"] and normalized["raw_text"]:
        normalized["sections"] = fallback["sections"]
    if normalized["raw_text"].strip():
        normalized["antigravity_export"]["comparison_ready"] = True

    return normalized


def _format_external_error(provider_name: str, exc: Exception, model_alias: str | None = None) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        response_text = exc.response.text.lower()
        api_error_message = _extract_api_error_message(exc.response)
        if provider_name.startswith("OpenRouter") and "user not found" in response_text:
            return (
                "OpenRouter rejected the configured key for chat completions. "
                "Check that OPENROUTER_API_KEY is set to a standard OpenRouter completion key."
            )
        if provider_name.startswith("OpenRouter") and status_code == 404 and "no endpoints found" in response_text:
            suggested_alias = STALE_OPENROUTER_MODEL_ALIASES.get(model_alias or "")
            if model_alias and suggested_alias:
                return (
                    f"OpenRouter model alias `{model_alias}` has no available endpoints. "
                    f"Update OPENROUTER_GEMINI_MODEL to `{suggested_alias}`."
                )
            if model_alias:
                return f"OpenRouter model alias `{model_alias}` has no available endpoints."
        if status_code == 401:
            return f"{provider_name} authentication failed. Check the configured API key."
        if api_error_message:
            return f"{provider_name} request failed with HTTP {status_code}: {api_error_message}"
        return f"{provider_name} request failed with HTTP {status_code}."
    return f"{provider_name} request failed: {exc}"


def _extract_api_error_message(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None

    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return None

    message = error.get("message")
    return str(message) if message else None


def _extract_message_content(provider_name: str, payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ValueError(f"{provider_name} returned a non-JSON response payload.")

    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message") or "Unknown API error."
        raise ValueError(f"{provider_name} returned API error {code}: {message}")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"{provider_name} response did not include any choices.")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError(f"{provider_name} response contained an invalid choice payload.")

    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError(f"{provider_name} response did not include a valid message object.")

    content = message.get("content")
    if content is None:
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            raise ValueError(f"{provider_name} stopped before returning JSON content because the response hit the token limit.")
        raise ValueError(f"{provider_name} response did not include message content.")

    return _coerce_message_content(content)


def _parse_json_response(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        cleaned = _extract_json_object(content)
        return json.loads(cleaned)


def _extract_json_object(content: str) -> str:
    start = content.find("{")
    if start == -1:
        raise ValueError("Model response did not contain a valid JSON object.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : index + 1]

    raise ValueError("Model response did not contain a complete JSON object.")


def _settings_warnings(settings: Settings) -> list[str]:
    warnings: list[str] = []
    suggested_alias = STALE_OPENROUTER_MODEL_ALIASES.get(settings.openrouter_gemini_model)
    if suggested_alias:
        warnings.append(
            f"Configured OpenRouter JSON model `{settings.openrouter_gemini_model}` is stale; use `{suggested_alias}` instead."
        )
    return warnings


def _warnings_include_openrouter_failure(warnings: list[str]) -> bool:
    for warning in warnings:
        lowered = warning.lower()
        if lowered.startswith("openrouter ") and (
            "request failed" in lowered
            or "authentication failed" in lowered
            or "rejected the configured key" in lowered
            or "model alias" in lowered
        ):
            return True
    return False


def _warnings_include_deepseek_failure(warnings: list[str]) -> bool:
    for warning in warnings:
        lowered = warning.lower()
        if lowered.startswith("deepseek ") and (
            "request failed" in lowered or "authentication failed" in lowered
        ):
            return True
    return False


def _summarize_ocr_failure(warnings: list[str]) -> str | None:
    for warning in warnings:
        if "authentication failed" in warning.lower():
            return warning
    for warning in warnings:
        if warning:
            return warning
    return None


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


def _build_local_processing_steps(prepared: PreparedDocument) -> list[str]:
    base_steps = [
        "FastAPI receives the upload and reads the file bytes locally",
        f"The backend validates extension `{prepared.extension}` and normalizes metadata",
    ]

    if prepared.extension == ".txt":
        base_steps.append("The backend decodes the text locally without AI")
    elif prepared.extension == ".csv":
        base_steps.append("The backend parses CSV rows locally without AI")
    elif prepared.extension == ".pdf":
        base_steps.append("The backend extracts PDF text locally with pypdfium2")
        base_steps.append("The backend extracts PDF tables locally with pdfplumber")
        if prepared.used_ocr:
            base_steps.append("The backend renders PDF pages to images locally before OCR")
    else:
        base_steps.append("The backend opens and normalizes the image locally with Pillow")
        base_steps.append("The backend converts the image to PNG and base64 locally before OCR")

    base_steps.append("The backend normalizes the final JSON payload locally before responding")
    return base_steps


def _build_ocr_reason(prepared: PreparedDocument) -> str:
    if prepared.extension in {".png", ".jpg", ".jpeg", ".webp"}:
        return "The uploaded file is an image, so OCR is required to read text"
    if prepared.extension == ".pdf":
        return "The PDF had too little extractable local text, so OCR was triggered for scanned or image-heavy content"
    return "OCR not required"


def _dedupe(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique
