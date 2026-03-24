import type { ProcessResponse, Provider } from '@/lib/types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api'

export async function processDocument(file: File, provider: Provider): Promise<ProcessResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('provider', provider)

  const response = await fetch(`${API_BASE_URL}/process`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null
    throw new Error(payload?.detail ?? 'Failed to process file.')
  }

  return (await response.json()) as ProcessResponse
}
