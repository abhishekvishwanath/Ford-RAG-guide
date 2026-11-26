import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'
import { answer, response } from './test/fixtures'

function mockFetch(handlers: Record<string, (body?: unknown) => unknown>) {
  const calls: { url: string; body?: unknown }[] = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      const body = init?.body ? JSON.parse(String(init.body)) : undefined
      calls.push({ url, body })
      const key = Object.keys(handlers).find((k) => url.startsWith(k))
      const data = key ? handlers[key](body) : {}
      return new Response(JSON.stringify(data), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }),
  )
  return calls
}

const health = { status: 'ok', index_ready: true, model_years: [2014], engines: ['3.7L', '5.0L', '5.8L'], transmissions: ['automatic', 'manual'], llm_configured: true, llm_model: 'm', manual: { chunks: 1, procedures: 1, figures: 1, tables: 1 } }

beforeEach(() => {
  localStorage.clear()
  vi.unstubAllGlobals()
})

it('renders the product and starter prompts', async () => {
  mockFetch({ '/api/health': () => health })
  render(<App />)
  expect(screen.getByRole('heading', { name: 'Mustang Workshop Assistant' })).toBeInTheDocument()
  expect(await screen.findByText(/Answers only from the 2014 workshop manual/)).toBeInTheDocument()
  expect(screen.getAllByRole('button', { name: /2014 (Mustang|Shelby)/ })).toHaveLength(5)
  for (const kind of ['Procedure + diagrams', 'Specification table', 'Diagram lookup', 'Diagnostic code', 'Variant-specific'])
    expect(screen.getByText(kind)).toBeInTheDocument()
})

it('runs a clarification turn then an answer, keeping the session id', async () => {
  let turn = 0
  const calls = mockFetch({
    '/api/health': () => health,
    '/api/chat': () => (turn++ === 0 ? response() : answer()),
  })
  render(<App />)
  const box = screen.getByLabelText(/Describe the vehicle/)
  await userEvent.type(box, 'replace the water pump{Enter}')
  expect(await screen.findByText('Which model year is the Mustang?')).toBeInTheDocument()
  expect(screen.getByText('Collecting details')).toBeInTheDocument()
  await userEvent.type(box, '2014{Enter}')
  expect(await screen.findByText('From the workshop manual')).toBeInTheDocument()
  const chats = calls.filter((c) => c.url === '/api/chat')
  expect(chats[0].body).toMatchObject({ message: 'replace the water pump', session_id: null })
  expect(chats[1].body).toMatchObject({ message: '2014', session_id: 'sess1' })
  expect(localStorage.getItem('mwa.session_id')).toBe('sess1')
  expect(screen.getByText('Ready — manual search allowed')).toBeInTheDocument()
  expect(screen.getByText(/Current procedure/)).toHaveTextContent('Coolant Pump — 5.0L')
})

it('shows an error with retry when the server fails', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) =>
      url === '/api/health'
        ? new Response(JSON.stringify(health), { status: 200 })
        : new Response(JSON.stringify({ error: { message: 'The language model is currently unavailable.' } }), { status: 503 }),
    ),
  )
  render(<App />)
  await userEvent.type(screen.getByLabelText(/Describe the vehicle/), 'hello{Enter}')
  expect(await screen.findByText('The language model is currently unavailable.')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /Retry/ })).toBeInTheDocument()
})

it('vehicle panel edits re-check readiness through the API', async () => {
  const calls = mockFetch({
    '/api/health': () => health,
    '/api/chat': () => response(),
    '/api/sessions/sess1/context': () => ({
      session_id: 'sess1', created_at: '', expires_at: null, workflow_state: 'intake', messages: [],
      vehicle_context: { engine: { value: '5.0L', status: 'confirmed', confidence: 1 } }, repair_context: {},
      readiness: { status: 'READY', score: 1, required_slots: [], missing_hard: [], missing_soft: [], needs_confirmation: [], conflicts: [], reason: '' },
    }),
  })
  render(<App />)
  await userEvent.type(screen.getByLabelText(/Describe the vehicle/), 'water pump{Enter}')
  await screen.findByText('Which model year is the Mustang?')
  await userEvent.click(screen.getByRole('button', { name: /Edit/ }))
  await userEvent.selectOptions(screen.getByLabelText('Engine'), '5.0L')
  await userEvent.click(screen.getByRole('button', { name: /Save and re-check readiness/ }))
  await waitFor(() => expect(calls.some((c) => c.url === '/api/sessions/sess1/context')).toBe(true))
  expect(calls.find((c) => c.url === '/api/sessions/sess1/context')?.body).toMatchObject({ vehicle: { engine: '5.0L' } })
  expect(await screen.findByText('confirmed')).toBeInTheDocument()
})
