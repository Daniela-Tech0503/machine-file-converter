export type Provider = 'deepseek' | 'gemini'

export interface ExtractionStats {
  file_name: string
  mime_type: string
  extension: string
  pages: number
  text_blocks: number
  tables_found: number
  characters: number
  used_ocr: boolean
  warnings: string[]
}

export interface ProcessResponse {
  message: string
  provider: Provider
  transport: string
  export_file_name: string
  report_file_name: string
  extraction: ExtractionStats
  json_result: Record<string, unknown>
  report_markdown: string
}

export interface ChatEntry {
  id: string
  role: 'user' | 'assistant'
  provider?: Provider
  fileName?: string
  response?: ProcessResponse
  text: string
}
