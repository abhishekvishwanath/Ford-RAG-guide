import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MessageBubble } from '../components/MessageBubble'
import { answer, response } from './fixtures'

const noop = () => undefined

it('labels clarification questions and offers option chips', async () => {
  const onOption = vi.fn()
  const r = response({ assistant_message: 'Which engine is fitted?', pending_question: { slots: ['engine'], kind: 'information', options: ['3.7L', '5.0L', '5.8L (GT500)'] } })
  render(<MessageBubble message={{ id: '1', role: 'assistant', content: r.assistant_message, response: r }} isLast onOption={onOption} onRetry={noop} />)
  expect(screen.getByText('Clarifying question')).toBeInTheDocument()
  expect(screen.queryByLabelText('Manual sources')).toBeNull() // no search results while collecting context
  await userEvent.click(screen.getByRole('button', { name: '5.8L (GT500)' }))
  expect(onOption).toHaveBeenCalledWith('5.8L')
})

it('renders a grounded answer with steps, diagrams, sources and page refs', async () => {
  const r = answer()
  render(<MessageBubble message={{ id: '2', role: 'assistant', content: r.assistant_message, response: r }} isLast onOption={noop} onRetry={noop} />)
  expect(screen.getByText('From the workshop manual')).toBeInTheDocument()
  const sources = screen.getByLabelText('Manual sources')
  expect(within(sources).getByText('Coolant Pump — 5.0L')).toBeInTheDocument()
  expect(sources).toHaveTextContent('PDF p. 1660-1661')
  expect(sources).toHaveTextContent('printed Page 2 of 3')
  const gallery = screen.getByLabelText('Manual diagrams')
  expect(within(gallery).getByRole('img')).toHaveAttribute('src', '/api/manual-assets/img_0123456789abcdef0123')
  expect(gallery).toHaveTextContent('Step 4')
  await userEvent.click(within(gallery).getByRole('button', { name: /Open diagram/ }))
  expect(screen.getByRole('dialog')).toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: 'Zoom in' }))
  expect(screen.getByText('125%')).toBeInTheDocument()
})

it('shows safety escalation distinctly', () => {
  const r = response({ interaction_mode: 'safety_escalation', assistant_message: '⚠️ Stop work now.', pending_question: null })
  render(<MessageBubble message={{ id: '3', role: 'assistant', content: r.assistant_message, response: r }} isLast onOption={noop} onRetry={noop} />)
  expect(screen.getByText('Safety first')).toBeInTheDocument()
})

it('offers retry on errors', async () => {
  const onRetry = vi.fn()
  render(<MessageBubble message={{ id: '4', role: 'assistant', content: 'Server unreachable', error: true }} isLast onOption={noop} onRetry={onRetry} />)
  await userEvent.click(screen.getByRole('button', { name: /Retry/ }))
  expect(onRetry).toHaveBeenCalled()
})
