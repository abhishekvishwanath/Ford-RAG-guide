export type InteractionMode =
  | 'asking_context'
  | 'awaiting_confirmation'
  | 'answer'
  | 'safety_escalation'
  | 'out_of_scope'

export type SlotStatus = 'user_provided' | 'confirmed' | 'inferred' | 'user_unsure' | 'unknown'

export interface SlotView {
  value: string | number | string[] | null
  status: SlotStatus
  confidence: number
  basis?: string
  needs_confirmation_of?: unknown
}

export interface Readiness {
  status: string | null
  score: number
  missing_fields: string[]
  needs_confirmation: unknown[]
  required_fields: string[]
  reason: string
}

export interface Citation {
  source_id: string
  chunk_id: string
  document_title: string
  section: string | null
  procedure_title: string | null
  procedure_id: string
  subsection: string | null
  pdf_pages: string
  printed_pages: string | null
  figure_ids: string[]
  table_ids: string[]
}

export interface ImageRef {
  asset_id: string
  figure_id: string
  url: string
  caption: string
  figure_numbers: string[]
  page_number: number
  printed_page: string | null
  procedure_title: string | null
  section: string | null
  nearest_step: number | null
  association: string
  confidence: number
  source_id: string | null
}

export interface TableRef {
  table_id: string
  source_id: string
  header: string[]
  rows: string[][]
  page_number: number
}

export interface PendingQuestion {
  slots: string[]
  kind: 'information' | 'confirmation' | 'safety_ack'
  options: string[]
}

export interface ChatResponse {
  session_id: string
  assistant_message: string
  interaction_mode: InteractionMode
  readiness: Readiness
  pending_question: PendingQuestion | null
  vehicle_context: Record<string, SlotView>
  repair_context: Record<string, SlotView>
  citations: Citation[]
  retrieved_sources: Array<{
    source_id: string
    chunk_id: string
    procedure_title: string | null
    section: string | null
    subsection: string | null
    pages: string
    applicability: string
    expanded: boolean
  }>
  images: ImageRef[]
  tables: TableRef[]
  validation: { warnings: string[]; unsupported_values: string[] } | null
  conversation_state: {
    workflow_state: string
    turn: number
    intake_turns: number
    active_procedure: string | null
    trace: string[]
  }
}

export interface SessionInfo {
  session_id: string
  created_at: string
  expires_at: string | null
  vehicle_context: Record<string, SlotView>
  repair_context: Record<string, SlotView>
  readiness: ReadinessDecision | null
  workflow_state: string
  messages: Array<{ role: 'user' | 'assistant'; content: string; created_at: string }>
}

export interface ReadinessDecision {
  status: string
  score: number
  required_slots: string[]
  missing_hard: string[]
  missing_soft: string[]
  needs_confirmation: string[]
  conflicts: string[]
  reason: string
}

export interface Health {
  status: string
  index_ready: boolean
  model_years: number[]
  engines: string[]
  transmissions: string[]
  llm_configured: boolean
  llm_model: string
  manual: { chunks: number; procedures: number; figures: number; tables: number }
}

export interface VehicleSelection {
  model_year?: number
  engine?: string
  transmission?: 'automatic' | 'manual'
  body_style?: 'coupe' | 'convertible'
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  response?: ChatResponse
  error?: boolean
}
