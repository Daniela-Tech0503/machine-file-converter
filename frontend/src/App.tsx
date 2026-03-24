import { useState } from 'react'
import { Bot, Copy, Download, FileUp, LoaderCircle, Sparkles } from 'lucide-react'

import { processDocument } from '@/lib/api'
import type { ChatEntry, ProcessResponse, Provider } from '@/lib/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

const ACCEPTED_TYPES = '.pdf,.png,.jpg,.jpeg,.webp,.txt,.csv'

function App() {
  const [selectedProvider, setSelectedProvider] = useState<Provider>('deepseek')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [messages, setMessages] = useState<ChatEntry[]>([])
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const canUseClipboard = typeof navigator !== 'undefined' && Boolean(navigator.clipboard)

  const latestResponse = [...messages].reverse().find((message) => message.role === 'assistant')?.response

  async function handleSubmit() {
    if (!selectedFile || isSubmitting) return

    setError(null)
    setIsSubmitting(true)

    const userMessage: ChatEntry = {
      id: crypto.randomUUID(),
      role: 'user',
      provider: selectedProvider,
      fileName: selectedFile.name,
      text: `Convert ${selectedFile.name} with ${getProviderLabel(selectedProvider)}.`,
    }

    setMessages((current) => [...current, userMessage])

    try {
      const response = await processDocument(selectedFile, selectedProvider)
      const assistantMessage: ChatEntry = {
        id: crypto.randomUUID(),
        role: 'assistant',
        response,
        text: response.message,
      }

      setMessages((current) => [...current, assistantMessage])
      handleExport(response)
      setSelectedFile(null)
    } catch (submissionError) {
      setError(
        submissionError instanceof Error ? submissionError.message : 'Unexpected processing error.',
      )
    } finally {
      setIsSubmitting(false)
    }
  }

  async function handleCopy(response: ProcessResponse) {
    if (!canUseClipboard) return
    await navigator.clipboard.writeText(JSON.stringify(response.json_result, null, 2))
  }

  function handleExport(response: ProcessResponse) {
    handleDirectExport(response)
  }

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-4 py-6 sm:px-6 lg:px-8">
      <div className="mb-6 flex items-center justify-between gap-4 rounded-full border border-white/60 bg-white/70 px-5 py-3 shadow-[0_12px_40px_rgba(27,31,35,0.06)] backdrop-blur">
        <div>
          <p className="text-xs uppercase tracking-[0.28em] text-[var(--muted-ink)]">Machine Reader</p>
          <h1 className="text-xl font-semibold tracking-[-0.03em] text-[var(--ink)] sm:text-2xl">
            Convert files into clean comparison JSON
          </h1>
        </div>
        <Badge>DeepSeek direct + OpenRouter fallback lane</Badge>
      </div>

      <div className="grid flex-1 gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <Card className="min-h-[72vh] overflow-hidden">
          <CardHeader className="border-b border-black/6 pb-4">
            <CardTitle className="flex items-center gap-2">
              <Bot className="h-5 w-5" />
              Conversion chat
            </CardTitle>
            <CardDescription>
              Upload one file, choose a model, and get a JSON file downloaded automatically.
            </CardDescription>
          </CardHeader>

          <CardContent className="flex h-[calc(72vh-88px)] flex-col gap-4 pt-6">
            <ScrollArea className="flex-1 pr-4">
              <div className="space-y-4 pb-4">
                {messages.length === 0 ? (
                  <EmptyState />
                ) : (
                  messages.map((message) => (
                    <MessageBubble
                      key={message.id}
                      message={message}
                      onCopy={handleCopy}
                      onExport={handleExport}
                      canCopy={canUseClipboard}
                    />
                  ))
                )}

                {isSubmitting ? (
                  <div className="inline-flex items-center gap-3 rounded-3xl bg-[var(--soft)] px-4 py-3 text-sm text-[var(--muted-ink)]">
                    <LoaderCircle className="h-4 w-4 animate-spin" />
                    Processing file, running extraction, OCR if needed, and JSON conversion...
                  </div>
                ) : null}
              </div>
            </ScrollArea>

            <div className="rounded-[28px] border border-black/8 bg-[var(--surface-strong)] p-4">
              <div className="grid gap-3 md:grid-cols-[1fr_220px]">
                <label className="flex min-h-24 cursor-pointer flex-col justify-center rounded-[22px] border border-dashed border-black/12 bg-white px-4 py-3 transition hover:border-black/25 hover:bg-white/90">
                  <span className="mb-1 flex items-center gap-2 text-sm font-medium text-[var(--ink)]">
                    <FileUp className="h-4 w-4" />
                    Upload file
                  </span>
                  <span className="text-sm text-[var(--muted-ink)]">
                    {selectedFile ? selectedFile.name : 'PDF, image, TXT, or CSV'}
                  </span>
                  <input
                    accept={ACCEPTED_TYPES}
                    className="hidden"
                    type="file"
                    onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
                  />
                </label>

                <div className="flex flex-col gap-3">
                  <Select
                    value={selectedProvider}
                    onValueChange={(value) => setSelectedProvider(value as Provider)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Choose a model" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="deepseek">DeepSeek</SelectItem>
                      <SelectItem value="gemini">Gemini 3.0 via OpenRouter</SelectItem>
                    </SelectContent>
                  </Select>

                  <Button className="w-full" onClick={handleSubmit} disabled={!selectedFile || isSubmitting}>
                    {isSubmitting ? (
                      <>
                        <LoaderCircle className="h-4 w-4 animate-spin" />
                        Converting...
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-4 w-4" />
                        Convert to JSON
                      </>
                    )}
                  </Button>
                </div>
              </div>

              {error ? <p className="mt-3 text-sm text-red-600">{error}</p> : null}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Current result</CardTitle>
            <CardDescription>Quick stats and direct JSON download for Antigravity comparison.</CardDescription>
          </CardHeader>
          <CardContent>
            {latestResponse ? <ResultPanel response={latestResponse} /> : <SidePlaceholder />}
          </CardContent>
        </Card>
      </div>
    </main>
  )
}

function EmptyState() {
  return (
    <div className="rounded-[28px] border border-dashed border-black/10 bg-white/55 p-6 text-sm leading-7 text-[var(--muted-ink)]">
      Drop in a file and run a single clean conversion flow.
      <div className="mt-4 flex flex-wrap gap-2">
        <Badge>PDF</Badge>
        <Badge>PNG</Badge>
        <Badge>JPG</Badge>
        <Badge>WEBP</Badge>
        <Badge>TXT</Badge>
        <Badge>CSV</Badge>
      </div>
    </div>
  )
}

function MessageBubble({
  message,
  onCopy,
  onExport,
  canCopy,
}: {
  message: ChatEntry
  onCopy: (response: ProcessResponse) => Promise<void>
  onExport: (response: ProcessResponse) => void
  canCopy: boolean
}) {
  if (message.role === 'user') {
    return (
      <div className="ml-auto max-w-[80%] rounded-[28px] bg-[var(--ink)] px-5 py-4 text-sm leading-7 text-white shadow-lg shadow-black/10">
        <p>{message.text}</p>
      </div>
    )
  }

  const response = message.response
  if (!response) return null

  return (
    <div className="max-w-[92%] rounded-[30px] bg-[var(--soft)] p-4 text-sm text-[var(--ink)]">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Badge>{getProviderLabel(response.provider)}</Badge>
        <Badge>{response.transport === 'deepseek_direct' ? 'Direct API' : 'OpenRouter'}</Badge>
        {response.extraction.used_ocr ? <Badge>OCR used</Badge> : <Badge>No OCR</Badge>}
      </div>

      <p className="mb-4 text-sm leading-7 text-[var(--ink)]">{message.text}</p>

      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="json">JSON</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <Stat label="Pages" value={response.extraction.pages} />
            <Stat label="Text blocks" value={response.extraction.text_blocks} />
            <Stat label="Tables" value={response.extraction.tables_found} />
            <Stat label="Characters" value={response.extraction.characters} />
          </div>
          {response.extraction.warnings.length > 0 ? (
            <div className="rounded-2xl bg-white/80 p-3 text-xs leading-6 text-[var(--muted-ink)]">
              {response.extraction.warnings.join(' ')}
            </div>
          ) : null}
        </TabsContent>

        <TabsContent value="json" className="space-y-3">
          <div className="flex flex-wrap gap-2">
            <Button variant="secondary" size="sm" onClick={() => void onCopy(response)} disabled={!canCopy}>
              <Copy className="h-4 w-4" />
              {canCopy ? 'Copy' : 'Copy unavailable'}
            </Button>
            <Button variant="secondary" size="sm" onClick={() => onExport(response)}>
              <Download className="h-4 w-4" />
              Export
            </Button>
          </div>
          <pre className="max-h-[420px] overflow-auto rounded-[22px] bg-[#171717] p-4 text-xs leading-6 text-[#f7f3ec]">
            {JSON.stringify(response.json_result, null, 2)}
          </pre>
        </TabsContent>
      </Tabs>
    </div>
  )
}

function ResultPanel({ response }: { response: ProcessResponse }) {
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <Badge>{response.extraction.file_name}</Badge>
        <Badge>{response.extraction.extension}</Badge>
        <Badge>{getProviderLabel(response.provider)}</Badge>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <Stat label="Pages" value={response.extraction.pages} />
        <Stat label="OCR" value={response.extraction.used_ocr ? 'Yes' : 'No'} />
        <Stat label="Blocks" value={response.extraction.text_blocks} />
        <Stat label="Tables" value={response.extraction.tables_found} />
      </div>

      <div className="rounded-[24px] bg-[var(--soft)] p-4">
        <p className="mb-2 text-xs uppercase tracking-[0.24em] text-[var(--muted-ink)]">Export</p>
        <p className="text-sm leading-7 text-[var(--ink)]">{response.export_file_name}</p>
        <div className="mt-3">
          <Button variant="secondary" size="sm" onClick={() => handleDirectExport(response)}>
            <Download className="h-4 w-4" />
            Download JSON file
          </Button>
        </div>
      </div>

      <pre className="max-h-[360px] overflow-auto rounded-[24px] bg-[#171717] p-4 text-xs leading-6 text-[#f7f3ec]">
        {JSON.stringify(response.json_result, null, 2)}
      </pre>
    </div>
  )
}

function SidePlaceholder() {
  return (
    <div className="rounded-[28px] bg-[var(--soft)] p-5 text-sm leading-7 text-[var(--muted-ink)]">
      Your latest converted JSON appears here for quick inspection and export.
    </div>
  )
}

function handleDirectExport(response: ProcessResponse) {
  const blob = new Blob([JSON.stringify(response.json_result, null, 2)], {
    type: 'application/json',
  })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = response.export_file_name
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-[22px] bg-white px-4 py-3 shadow-sm ring-1 ring-black/5">
      <p className="text-xs uppercase tracking-[0.22em] text-[var(--muted-ink)]">{label}</p>
      <p className="mt-2 text-lg font-semibold text-[var(--ink)]">{value}</p>
    </div>
  )
}

function getProviderLabel(provider: Provider) {
  return provider === 'deepseek' ? 'DeepSeek' : 'Gemini 3.0'
}


export default App
