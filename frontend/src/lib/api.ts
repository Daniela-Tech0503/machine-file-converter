import type {
  PageProcessResponse,
  ProcessResponse,
  Provider,
  SplitResponse,
} from '@/lib/types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api'

// ---------------------------------------------------------------------------
// Legacy single-shot endpoint (kept for non-PDF / small files)
// ---------------------------------------------------------------------------

export async function processDocument(
  file: File,
  provider: Provider,
  instructions: string,
): Promise<ProcessResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('provider', provider)
  formData.append('instructions', instructions)

  const response = await fetch(`${API_BASE_URL}/process`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const rawBody = await response.text()
    let payload: { detail?: string } | null = null
    if (rawBody) {
      try {
        payload = JSON.parse(rawBody) as { detail?: string }
      } catch {
        payload = null
      }
    }
    throw new Error(payload?.detail ?? rawBody ?? `Failed to process file (${response.status}).`)
  }

  return (await response.json()) as ProcessResponse
}

// ---------------------------------------------------------------------------
// Parallel pipeline — Step 1: ask the backend to split the file into pages
// ---------------------------------------------------------------------------

export async function splitDocument(file: File): Promise<SplitResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/split-pages`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const rawBody = await response.text()
    let payload: { detail?: string } | null = null
    try {
      payload = JSON.parse(rawBody) as { detail?: string }
    } catch {
      payload = null
    }
    throw new Error(payload?.detail ?? rawBody ?? `Split failed (${response.status}).`)
  }

  return (await response.json()) as SplitResponse
}

// ---------------------------------------------------------------------------
// Parallel pipeline — Step 2: process a single page
// ---------------------------------------------------------------------------

export async function processPage(
  file: File,
  pageNumber: number,
  provider: Provider,
): Promise<PageProcessResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('page_number', String(pageNumber))
  formData.append('provider', provider)

  const response = await fetch(`${API_BASE_URL}/process-page`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const rawBody = await response.text()
    let payload: { detail?: string } | null = null
    try {
      payload = JSON.parse(rawBody) as { detail?: string }
    } catch {
      payload = null
    }
    throw new Error(
      payload?.detail ?? rawBody ?? `Page ${pageNumber} failed (${response.status}).`,
    )
  }

  return (await response.json()) as PageProcessResponse
}

// ---------------------------------------------------------------------------
// FIFO parallel queue
// ---------------------------------------------------------------------------

export interface QueueProgress {
  total: number
  completed: number
  failed: number
  currentPage: number | null
}

/**
 * Process all pages of a file using a FIFO queue with `concurrency` parallel
 * workers. Reports progress via `onProgress` after each page completes.
 * Returns pages sorted by page_number.
 */
export async function processAllPages(
  file: File,
  provider: Provider,
  splitResult: SplitResponse,
  concurrency = 3,
  onProgress?: (progress: QueueProgress) => void,
): Promise<PageProcessResponse[]> {
  const queue: number[] = splitResult.pages.map((p) => p.page_number) // FIFO order
  const results: PageProcessResponse[] = []
  let completed = 0
  let failed = 0

  async function worker(): Promise<void> {
    while (queue.length > 0) {
      const pageNumber = queue.shift()! // dequeue FIFO
      onProgress?.({ total: splitResult.total_pages, completed, failed, currentPage: pageNumber })

      try {
        const result = await processPage(file, pageNumber, provider)
        results.push(result)
      } catch (err) {
        failed++
        results.push({
          page_number: pageNumber,
          text: '',
          tables: [],
          warnings: [err instanceof Error ? err.message : `Page ${pageNumber} failed.`],
          ocr_applied: false,
          transport: 'error',
        })
      }

      completed++
      onProgress?.({ total: splitResult.total_pages, completed, failed, currentPage: null })
    }
  }

  // Launch `concurrency` workers in parallel — they all pull from the same FIFO queue
  await Promise.all(Array.from({ length: concurrency }, () => worker()))

  // Return pages in document order
  return results.sort((a, b) => a.page_number - b.page_number)
}

// ---------------------------------------------------------------------------
// Merge all processed pages into a single ProcessResponse-compatible object
// ---------------------------------------------------------------------------

export function mergePageResults(
  pages: PageProcessResponse[],
  file: File,
  provider: Provider,
  extension: string,
): ProcessResponse {
  const allText = pages.map((p) => `[Page ${p.page_number}]\n${p.text}`).join('\n\n').trim()
  const allTables = pages.flatMap((p) => p.tables)
  const allWarnings = pages.flatMap((p) => p.warnings)
  const ocrApplied = pages.some((p) => p.ocr_applied)
  const stem = file.name.replace(/\.[^.]+$/, '')

  const jsonResult = {
    document: {
      title: file.name,
      type: extension.replace('.', '').toUpperCase(),
      language: 'unknown',
      source_format: extension.replace('.', ''),
      ocr_applied: ocrApplied,
    },
    summary: `Parallel extraction of ${pages.length} page(s) from ${file.name}.`,
    entities: [],
    sections: pages
      .filter((p) => p.text.trim())
      .map((p) => ({ heading: `Page ${p.page_number}`, text: p.text.trim() })),
    tables: allTables,
    raw_text: allText,
    warnings: allWarnings,
    antigravity_export: {
      comparison_ready: Boolean(allText.trim()),
      format_version: '1.0',
    },
  }

  const transport = pages.some((p) => p.transport.startsWith('openrouter'))
    ? 'openrouter'
    : 'local'

  return {
    message: `Parallel conversion of ${file.name} complete (${pages.length} pages).`,
    provider,
    transport,
    export_file_name: `${stem}.machine-readable.json`,
    report_file_name: `${stem}.pipeline-report.md`,
    extraction: {
      file_name: file.name,
      mime_type: file.type || 'application/octet-stream',
      extension,
      pages: pages.length,
      text_blocks: pages.filter((p) => p.text.trim()).length,
      tables_found: allTables.length,
      characters: allText.length,
      used_ocr: ocrApplied,
      ocr_requested: pages.some((p) => p.ocr_applied || p.transport === 'openrouter_ocr'),
      ocr_attempted: pages.some((p) => p.transport === 'openrouter_ocr'),
      ocr_succeeded: ocrApplied,
      warnings: allWarnings,
    },
    json_result: jsonResult,
    report_markdown: _buildReport(pages, file, provider, transport, allWarnings),
  }
}

function _buildReport(
  pages: PageProcessResponse[],
  file: File,
  provider: Provider,
  transport: string,
  warnings: string[],
): string {
  const lines: string[] = [
    `# Pipeline Report — ${file.name}`,
    '',
    `**Provider:** ${provider}`,
    `**Transport:** ${transport}`,
    `**Total pages:** ${pages.length}`,
    `**OCR pages:** ${pages.filter((p) => p.ocr_applied).length}`,
    '',
    '## Pages',
    ...pages.map(
      (p) =>
        `- Page ${p.page_number}: ${p.text.length} chars, OCR: ${p.ocr_applied}, transport: ${p.transport}`,
    ),
  ]
  if (warnings.length) {
    lines.push('', '## Warnings', ...warnings.map((w) => `- ${w}`))
  }
  return lines.join('\n')
}
