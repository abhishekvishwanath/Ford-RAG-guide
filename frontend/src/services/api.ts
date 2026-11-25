import type { ChatResponse, Health, SessionInfo, VehicleSelection } from '../types/api'

export class ApiError extends Error {
  status: number
  errorId?: string
  constructor(message: string, status: number, errorId?: string) {
    super(message)
    this.status = status
    this.errorId = errorId
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    })
  } catch {
    throw new ApiError('Cannot reach the workshop assistant server.', 0)
  }
  if (res.status === 204) return undefined as T
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    const err = body?.error
    const detail = Array.isArray(body?.detail) ? 'Please check your message and try again.' : undefined
    throw new ApiError(err?.message ?? detail ?? `Request failed (${res.status})`, res.status, err?.error_id)
  }
  return body as T
}

export const api = {
  health: () => request<Health>('/api/health'),
  chat: (message: string, sessionId: string | null, vehicle?: VehicleSelection) =>
    request<ChatResponse>('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message, session_id: sessionId, vehicle }),
    }),
  getSession: (id: string) => request<SessionInfo>(`/api/sessions/${encodeURIComponent(id)}`),
  deleteSession: (id: string) =>
    request<void>(`/api/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  updateContext: (id: string, vehicle: VehicleSelection) =>
    request<SessionInfo>(`/api/sessions/${encodeURIComponent(id)}/context`, {
      method: 'POST',
      body: JSON.stringify({ vehicle }),
    }),
}
