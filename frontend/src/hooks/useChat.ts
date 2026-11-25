import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../services/api'
import type {
  ChatMessage,
  ChatResponse,
  Readiness,
  ReadinessDecision,
  SlotView,
  VehicleSelection,
} from '../types/api'

const SESSION_KEY = 'mwa.session_id'

function loadSessionId(): string | null {
  try {
    return localStorage.getItem(SESSION_KEY)
  } catch {
    return null
  }
}

function saveSessionId(id: string | null) {
  try {
    if (id) localStorage.setItem(SESSION_KEY, id)
    else localStorage.removeItem(SESSION_KEY)
  } catch {
    /* storage unavailable: session lives only in memory */
  }
}

export function fromDecision(d: ReadinessDecision | null): Readiness | null {
  if (!d) return null
  return {
    status: d.status,
    score: d.score,
    missing_fields: [...(d.missing_hard ?? []), ...(d.missing_soft ?? [])],
    needs_confirmation: [...(d.needs_confirmation ?? []), ...(d.conflicts ?? [])],
    required_fields: d.required_slots ?? [],
    reason: d.reason,
  }
}

let counter = 0
const nextId = () => `m${Date.now()}_${counter++}`

export interface ChatState {
  sessionId: string | null
  messages: ChatMessage[]
  vehicle: Record<string, SlotView>
  repair: Record<string, SlotView>
  readiness: Readiness | null
  activeProcedure: string | null
  loading: boolean
  error: string | null
}

const empty: ChatState = {
  sessionId: null,
  messages: [],
  vehicle: {},
  repair: {},
  readiness: null,
  activeProcedure: null,
  loading: false,
  error: null,
}

export function useChat() {
  const [state, setState] = useState<ChatState>({ ...empty, sessionId: loadSessionId() })
  const lastMessage = useRef<string | null>(null)

  // restore an existing session after reload
  useEffect(() => {
    const id = state.sessionId
    if (!id || state.messages.length) return
    let cancelled = false
    api
      .getSession(id)
      .then((s) => {
        if (cancelled) return
        setState((prev) => ({
          ...prev,
          vehicle: s.vehicle_context,
          repair: s.repair_context,
          readiness: fromDecision(s.readiness),
          messages: s.messages.map((m) => ({ id: nextId(), role: m.role, content: m.content })),
        }))
      })
      .catch(() => {
        if (cancelled) return
        saveSessionId(null)
        setState((prev) => ({ ...prev, sessionId: null }))
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const apply = (prev: ChatState, r: ChatResponse): ChatState => ({
    ...prev,
    sessionId: r.session_id,
    vehicle: r.vehicle_context,
    repair: r.repair_context,
    readiness: r.readiness,
    activeProcedure: r.conversation_state.active_procedure,
  })

  const send = useCallback(
    async (text: string, vehicle?: VehicleSelection) => {
      const message = text.trim()
      if (!message || state.loading) return
      lastMessage.current = message
      setState((prev) => ({
        ...prev,
        loading: true,
        error: null,
        messages: [...prev.messages.filter((m) => !m.error), { id: nextId(), role: 'user', content: message }],
      }))
      try {
        const r = await api.chat(message, state.sessionId, vehicle)
        saveSessionId(r.session_id)
        setState((prev) => ({
          ...apply(prev, r),
          loading: false,
          messages: [...prev.messages, { id: nextId(), role: 'assistant', content: r.assistant_message, response: r }],
        }))
      } catch (e) {
        const msg = e instanceof ApiError ? e.message : 'Something went wrong.'
        setState((prev) => ({
          ...prev,
          loading: false,
          error: msg,
          messages: [...prev.messages, { id: nextId(), role: 'assistant', content: msg, error: true }],
        }))
      }
    },
    [state.loading, state.sessionId],
  )

  const retry = useCallback(() => {
    if (!lastMessage.current) return
    const text = lastMessage.current
    setState((prev) => {
      // drop the failed user turn + error bubble; send() re-adds the user turn
      const msgs = [...prev.messages]
      if (msgs.at(-1)?.error) msgs.pop()
      if (msgs.at(-1)?.role === 'user') msgs.pop()
      return { ...prev, messages: msgs }
    })
    void send(text)
  }, [send])

  const newSession = useCallback(async () => {
    const id = state.sessionId
    saveSessionId(null)
    setState({ ...empty })
    if (id) await api.deleteSession(id).catch(() => undefined)
  }, [state.sessionId])

  const updateVehicle = useCallback(
    async (vehicle: VehicleSelection) => {
      if (!state.sessionId) {
        // no session yet: the selection travels with the next chat message
        return vehicle
      }
      const s = await api.updateContext(state.sessionId, vehicle)
      setState((prev) => ({
        ...prev,
        vehicle: s.vehicle_context,
        repair: s.repair_context,
        activeProcedure: null,
        readiness: fromDecision(s.readiness) ?? prev.readiness,
      }))
      return undefined
    },
    [state.sessionId],
  )

  return { state, send, retry, newSession, updateVehicle }
}
