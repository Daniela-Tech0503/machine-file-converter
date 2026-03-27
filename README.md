# Machine Reader Chat

Minimal chat-style web app for turning files into machine-readable JSON.

## What it does

- Uploads `PDF`, `PNG`, `JPG`, `JPEG`, `WEBP`, `TXT`, and `CSV`
- Extracts text locally when possible
- Falls back to OCR when needed
- Converts content into structured JSON for comparison workflows such as Antigravity
- Lets the user choose:
  - `DeepSeek` via the direct DeepSeek API
- `Gemini 2.5 Pro` via OpenRouter
- Shows the result inside a simple chat UI
- Supports copy and JSON export

## Architecture

- `frontend/` - Vite + React + Radix UI + shadcn-style components
- `api/` - FastAPI backend for upload, extraction, OCR fallback, and model calls
- `vercel.json` - static frontend build plus Python serverless API routing

## Model routing

- `DeepSeek` selection -> direct request to DeepSeek using `deepseek-chat`
- Any non-DeepSeek model -> OpenRouter
- Current Gemini option -> `google/gemini-2.5-pro`
- OCR helper for image-heavy content -> `google/gemini-3-flash-preview` through OpenRouter

The OpenRouter aliases were chosen from the current OpenRouter model index during implementation.

## Local setup

1. Copy `.env.example` to `.env`
2. Fill in `DEEPSEEK_API_KEY` and `OPENROUTER_API_KEY`
3. Install frontend dependencies:

```bash
cd frontend
npm install
```

4. Create a Python environment and install backend dependencies:

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run locally

Backend:

```bash
.venv\Scripts\python.exe -m uvicorn api.app.main:app --reload
```

Frontend:

```bash
cd frontend
npm run dev
```

The frontend defaults to `http://127.0.0.1:8000/api` unless `VITE_API_BASE_URL` is set.

## API

### `GET /api/health`

Returns a simple status payload.

### `POST /api/process`

Multipart form fields:

- `file` - uploaded file
- `provider` - `deepseek` or `gemini`

Returns:

- extracted metadata
- OCR usage info
- final structured JSON
- export filename

## Vercel deployment

Set these environment variables in Vercel:

- `DEEPSEEK_API_KEY`
- `OPENROUTER_API_KEY`
- `DEEPSEEK_MODEL` optional
- `OPENROUTER_GEMINI_MODEL` optional
- `OPENROUTER_OCR_MODEL` optional
- `MAX_UPLOAD_MB` optional
- `MAX_OCR_PAGES` optional
- `ALLOWED_ORIGINS` optional

Deploy from the repository root. `vercel.json` builds the Vite app and routes `/api/*` to the FastAPI serverless entrypoint.

## Notes

- Local PDF extraction uses `pypdfium2` and `pdfplumber`
- OCR is AI-assisted so image-heavy files may consume model tokens
- If a provider is unavailable or a key is missing, the backend returns a safe local JSON fallback with warnings
