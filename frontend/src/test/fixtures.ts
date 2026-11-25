import type { ChatResponse } from '../types/api'

export function response(over: Partial<ChatResponse> = {}): ChatResponse {
  return {
    session_id: 'sess1',
    assistant_message: 'Which model year is the Mustang?',
    interaction_mode: 'asking_context',
    readiness: { status: 'NEEDS_INFORMATION', score: 0.5, missing_fields: ['model_year'], needs_confirmation: [], required_fields: ['model_year', 'component'], reason: '' },
    pending_question: { slots: ['model_year'], kind: 'information', options: [] },
    vehicle_context: {},
    repair_context: { component: { value: 'water pump', status: 'user_provided', confidence: 0.9 } },
    citations: [],
    retrieved_sources: [],
    images: [],
    tables: [],
    validation: null,
    conversation_state: { workflow_state: 'asking_context', turn: 1, intake_turns: 1, active_procedure: null, trace: [] },
    ...over,
  }
}

export const answer = () =>
  response({
    interaction_mode: 'answer',
    assistant_message: '**Procedure**\n1. Drain the cooling system. [S1]\n2. Tighten to 20 Nm (177 lb-in). [S1]',
    readiness: { status: 'READY', score: 1, missing_fields: [], needs_confirmation: [], required_fields: ['model_year', 'engine'], reason: '' },
    pending_question: null,
    vehicle_context: {
      model_year: { value: 2014, status: 'user_provided', confidence: 0.95 },
      engine: { value: '5.0L', status: 'confirmed', confidence: 1 },
    },
    citations: [
      { source_id: 'S1', chunk_id: 'c1', document_title: 'Mustang 2014 Workshop Manual', section: '303-03A Engine Cooling', procedure_title: 'Coolant Pump — 5.0L', procedure_id: 'SEB33A09', subsection: 'Installation', pdf_pages: '1660-1661', printed_pages: 'Page 2 of 3 – Page 3 of 3', figure_ids: [], table_ids: [] },
    ],
    images: [
      { asset_id: 'img_0123456789abcdef0123', figure_id: 'fig1', url: '/api/manual-assets/img_0123456789abcdef0123', caption: 'Manual illustration: Coolant Pump — 5.0L, step 4; PDF page 1660', figure_numbers: ['N0123456'], page_number: 1660, printed_page: 'Page 2 of 3', procedure_title: 'Coolant Pump — 5.0L', section: '303-03A', nearest_step: 4, association: 'in_step', confidence: 0.9, source_id: 'S1' },
    ],
    conversation_state: { workflow_state: 'answered', turn: 3, intake_turns: 0, active_procedure: 'Coolant Pump — 5.0L', trace: [] },
  })
