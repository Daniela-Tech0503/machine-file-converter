from dataclasses import dataclass, field

from api.app.config import Settings
from api.app.models.schemas import ExtractionStats, Provider
from api.app.services.extraction import PreparedDocument


@dataclass
class PipelineTrace:
    frontend_provider: Provider
    frontend_label: str
    local_processing_steps: list[str] = field(default_factory=list)
    available_fallbacks: list[str] = field(default_factory=list)
    fallback_events: list[str] = field(default_factory=list)
    openrouter_aliases: list[str] = field(default_factory=list)
    direct_api_calls: list[str] = field(default_factory=list)
    ocr_used: bool = False
    ocr_transport: str | None = None
    ocr_provider_name: str | None = None
    ocr_model_alias: str | None = None
    ocr_trigger_reason: str | None = None
    ocr_reads_file: bool = False
    json_transport: str = "local_backend"
    json_provider_name: str = "local_backend"
    json_model_alias: str | None = None
    json_reads_file: bool = False
    json_generated_by: str = "local fallback logic"
    frontend_choice_controls_json_model: bool = True


def generate_technical_report(
    *,
    prepared: PreparedDocument,
    extraction: ExtractionStats,
    provider: Provider,
    settings: Settings,
    transport: str,
    trace: PipelineTrace,
) -> str:
    file_type_summary = _file_type_summary(prepared.extension, extraction.used_ocr)
    selected_model_line = _selected_model_line(provider, settings)
    ocr_line = trace.ocr_model_alias if trace.ocr_model_alias else "No OCR model used in this run"
    openrouter_aliases = ", ".join(trace.openrouter_aliases) if trace.openrouter_aliases else "None"
    direct_calls = ", ".join(trace.direct_api_calls) if trace.direct_api_calls else "None"
    fallback_events = ", ".join(trace.fallback_events) if trace.fallback_events else "None triggered"
    fallback_options = ", ".join(trace.available_fallbacks) if trace.available_fallbacks else "None"
    local_steps = "\n".join(f"- {step}" for step in trace.local_processing_steps)

    report = f"""# Technical Pipeline Report

## Submission Summary

- File: `{prepared.file_name}`
- File type: `{prepared.extension}`
- Frontend provider selected by the user: `{trace.frontend_label}`
- Backend transport used for JSON structuring in this run: `{transport}`
- Model used to structure content into JSON: `{trace.json_model_alias or 'Local fallback, no external model'}`
- Model used for OCR in this run: `{ocr_line}`
- OpenRouter aliases used in this run: `{openrouter_aliases}`
- Direct external API calls used in this run: `{direct_calls}`
- Fallbacks triggered in this run: `{fallback_events}`

## Actual Step By Step Pipeline For This Submission

1. The frontend uploads the selected file to `POST /api/process` together with the user-selected provider value (`deepseek` or `gemini`).
2. FastAPI reads the file bytes locally, validates file size, validates extension, and decides which extraction path to use.
3. Local backend processing for this file type:
{local_steps}
4. OCR stage for this submission: `{_ocr_step_summary(trace)}`
5. JSON structuring stage for this submission: `{_json_step_summary(trace)}`
6. The backend normalizes the final JSON shape, merges warnings, and returns both the JSON payload and this markdown report.

## Technical Answers To The Required Questions

### Quando submeto um ficheiro, o que acontece passo a passo?

1. O frontend envia o ficheiro e a opcao de modelo para o backend FastAPI.
2. O backend le o ficheiro localmente e escolhe um pipeline consoante o tipo (`PDF`, imagem, `TXT`, `CSV`).
3. Se houver texto legivel localmente, ele e extraido sem AI primeiro.
4. Se for imagem ou PDF com pouco texto, o backend prepara imagens e chama OCR via modelo externo.
5. Depois do texto estar reunido, o backend chama o modelo de estruturacao para produzir JSON.
6. Se a chamada AI falhar ou faltar chave, existe fallback local para devolver JSON basico com avisos.

### O ficheiro é lido localmente ou enviado para um modelo?

- O ficheiro e sempre lido primeiro localmente pelo backend.
- O ficheiro inteiro nao e enviado diretamente para DeepSeek nem Gemini neste projeto.
- Quando OCR e necessario, o backend envia imagens derivadas do ficheiro ao modelo de OCR.
- Quando a estruturacao em JSON acontece, o backend envia texto extraido e contexto ao modelo de estruturacao.

### Se for PDF com texto, quem extrai o texto?

- PDFs com texto sao processados localmente no backend.
- O texto e extraido por `pypdfium2`.
- As tabelas sao extraidas localmente por `pdfplumber`.

### Se for imagem ou PDF digitalizado, quem faz OCR?

- O OCR e feito por um modelo chamado via OpenRouter.
- O alias configurado atualmente para OCR e `{settings.openrouter_ocr_model}`.
- Nao existe Gemini direto neste projeto; a chamada passa por OpenRouter.

### Que AI e usada para estruturar a informacao?

- Se o utilizador escolher `DeepSeek`, a estruturacao JSON usa `{settings.deepseek_model}` por chamada direta a `https://api.deepseek.com/chat/completions`.
- Se o utilizador escolher `Gemini 3.0`, a estruturacao JSON usa o alias OpenRouter `{settings.openrouter_gemini_model}`.

### Quem gera o JSON final?

- O JSON final e produzido pelo backend.
- Normalmente o backend pede ao modelo selecionado para devolver JSON estruturado.
- Depois o backend normaliza o shape final e aplica validacao/fallback local se necessario.

### Existe diferenca de pipeline entre PDF, imagem, TXT e CSV?

- Sim.
- `TXT`: leitura e decode local, depois AI apenas para estruturar JSON.
- `CSV`: parse local com `csv`, depois AI apenas para estruturar JSON.
- `PDF` com texto: extracao local com `pypdfium2` + `pdfplumber`, depois AI para JSON.
- `PDF` digitalizado: tentativa local primeiro, depois OCR via OpenRouter, depois AI para JSON.
- `PNG/JPG/JPEG/WEBP`: preparacao local da imagem, OCR via OpenRouter, depois AI para JSON.

### Existe alguma logica de fallback se um modelo falhar?

- Sim.
- Se a chave OpenRouter faltar, o OCR e ignorado e isso aparece nos warnings.
- Se DeepSeek falhar ou faltar chave, o backend devolve JSON local fallback.
- Se OpenRouter falhar ou faltar chave na etapa de JSON, o backend devolve JSON local fallback.
- Fallbacks atualmente disponiveis: `{fallback_options}`.

### O frontend deixa o utilizador escolher o modelo ou o modelo esta hardcoded?

- O frontend deixa o utilizador escolher entre `DeepSeek` e `Gemini 3.0`.
- Essa escolha afeta a etapa de estruturacao em JSON.
- O modelo de OCR nao e escolhido pelo utilizador; esta configurado no backend por variavel de ambiente.

### O modelo real usado no backend coincide com o que aparece na interface?

- Parcialmente.
- `DeepSeek` na interface corresponde ao alias backend `{settings.deepseek_model}`.
- `Gemini 3.0` na interface atualmente corresponde ao alias OpenRouter `{settings.openrouter_gemini_model}`.
- Ou seja, a interface mostra a familia do modelo, mas o alias real pode ser mais especifico e vir de variaveis de ambiente.

### Há variáveis de ambiente ou aliases de modelos que possam estar a esconder o modelo real?

- Sim.
- `DEEPSEEK_MODEL={settings.deepseek_model}`
- `OPENROUTER_GEMINI_MODEL={settings.openrouter_gemini_model}`
- `OPENROUTER_OCR_MODEL={settings.openrouter_ocr_model}`
- Estas variaveis controlam os aliases reais usados pelo backend.

### Há chamadas feitas por OpenRouter? Se sim, com que aliases?

- Sim, existe OpenRouter neste projeto.
- Alias de OCR configurado: `{settings.openrouter_ocr_model}`.
- Alias Gemini configurado para JSON: `{settings.openrouter_gemini_model}`.
- Aliases realmente usados nesta submissao: `{openrouter_aliases}`.

### Há chamadas diretas a APIs externas?

- Sim.
- DeepSeek pode ser chamado diretamente em `https://api.deepseek.com/chat/completions`.
- OpenRouter pode ser chamado diretamente em `https://openrouter.ai/api/v1/chat/completions`.
- Gemini direto nao esta implementado neste projeto atual.

### Há partes do processo feitas sem AI, localmente no backend?

- Sim, varias.
- Validacao do upload.
- Leitura do ficheiro.
- Decode de `TXT`.
- Parse de `CSV`.
- Extracao de texto de `PDF` com `pypdfium2`.
- Extracao de tabelas de `PDF` com `pdfplumber`.
- Preparacao de imagens com `Pillow`.
- Normalizacao final do JSON e fallback local.

## Model Mapping And Cost Evaluation Notes

- O pipeline atual pode usar mais do que um modelo: um para OCR e outro para JSON.
- O custo sobe quando OCR e necessario, porque isso adiciona uma chamada extra ao alias `{settings.openrouter_ocr_model}`.
- Para `TXT`, `CSV` e `PDF` com texto legivel, a maior parte do trabalho e local; a AI entra sobretudo na estruturacao final em JSON.
- Para imagens e PDFs digitalizados, ha uma etapa adicional de OCR antes da estruturacao.
- Isto significa que o modelo mais barato para manter extracao quase a 100% depende principalmente de duas perguntas:
  - o ficheiro precisa de OCR ou nao
  - o modelo de JSON escolhido pode ser mais barato sem degradar a estruturacao
- Nesta implementacao, o sitio certo para comparar custo/qualidade e:
  - OCR: `{settings.openrouter_ocr_model}`
  - JSON via DeepSeek: `{settings.deepseek_model}`
  - JSON via OpenRouter Gemini: `{settings.openrouter_gemini_model}`

## Current Run Quick View

- Current file pipeline summary: `{file_type_summary}`
- Frontend selected model label: `{trace.frontend_label}`
- Real backend JSON model for this run: `{selected_model_line}`
- OCR trigger reason: `{trace.ocr_trigger_reason or 'No OCR trigger in this run'}`
- Was the file text read locally first: `{_yes_no(_file_read_locally(prepared))}`
- Was content sent to a model for OCR: `{_yes_no(trace.ocr_used)}`
- Was JSON generated by an external AI model: `{_yes_no(trace.json_model_alias is not None)}`

## Warning Signals

- Warnings returned by the pipeline: `{', '.join(extraction.warnings) if extraction.warnings else 'None'}`
- If the model shown in the UI ever differs from the actual alias in production, inspect the environment variables listed above.
"""

    return report.strip() + "\n"


def _selected_model_line(provider: Provider, settings: Settings) -> str:
    if provider is Provider.DEEPSEEK:
        return settings.deepseek_model
    return settings.openrouter_gemini_model


def _file_type_summary(extension: str, used_ocr: bool) -> str:
    if extension == ".txt":
        return "TXT decoded locally, then structured into JSON by the selected JSON model."
    if extension == ".csv":
        return "CSV parsed locally, then structured into JSON by the selected JSON model."
    if extension == ".pdf" and not used_ocr:
        return "PDF text and tables extracted locally, then structured into JSON by the selected JSON model."
    if extension == ".pdf" and used_ocr:
        return "PDF first inspected locally, then OCR was added for image-heavy or scanned pages, then JSON structuring ran."
    return "Image normalized locally, OCR ran through OpenRouter, then JSON structuring ran."


def _ocr_step_summary(trace: PipelineTrace) -> str:
    if not trace.ocr_used:
        return "No OCR stage was needed."
    return (
        f"OCR was required and used provider `{trace.ocr_provider_name}` with alias `{trace.ocr_model_alias}` "
        f"via `{trace.ocr_transport}`."
    )


def _json_step_summary(trace: PipelineTrace) -> str:
    if trace.json_model_alias:
        return (
            f"JSON was structured by `{trace.json_model_alias}` using `{trace.json_provider_name}` "
            f"through `{trace.json_transport}`."
        )
    return "JSON was generated by local fallback logic in the backend because the external model path was unavailable or failed."


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _file_read_locally(prepared: PreparedDocument) -> bool:
    return prepared.extension in {".txt", ".csv", ".pdf", ".png", ".jpg", ".jpeg", ".webp"}
