import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Markdown } from '../components/Markdown'
import { admonitionKind, parseBlocks } from '../components/markdownParse'

it('keeps manual step numbers and nesting', () => {
  const blocks = parseBlocks('5. Remove the pulley. [S2]\n6. Remove the housing.\n   • Discard the seal.')
  expect(blocks[0]).toMatchObject({ kind: 'ol', items: [{ n: 5 }, { n: 6 }] })
  expect(blocks[1]).toMatchObject({ kind: 'ul', items: [{ depth: 1 }] })
})

it('renders warnings as alerts and notices as notes', () => {
  render(<Markdown text={'WARNING: Hot coolant can scald. [S1]\n\nNOTICE: Cover the belt.'} />)
  expect(screen.getByRole('alert')).toHaveTextContent('Hot coolant can scald')
  expect(screen.getByRole('note')).toHaveTextContent('Cover the belt')
  expect(admonitionKind('CAUTION: x')).toBe('danger')
  expect(admonitionKind('plain step')).toBeNull()
})

it('turns citation markers into clickable chips', async () => {
  const onCite = vi.fn()
  render(<Markdown text="Tighten to 25 Nm (18 lb-ft). [S1, S3]" onCite={onCite} />)
  await userEvent.click(screen.getByRole('button', { name: 'Show source S3' }))
  expect(onCite).toHaveBeenCalledWith('S3')
  expect(screen.getByRole('button', { name: 'Show source S1' })).toBeInTheDocument()
})

it('never injects HTML from model output', () => {
  const { container } = render(<Markdown text={'<img src=x onerror="alert(1)"> **bold**'} />)
  expect(container.querySelector('img')).toBeNull()
  expect(container.textContent).toContain('<img src=x')
  expect(container.querySelector('strong')?.textContent).toBe('bold')
})

it('renders pipe tables', () => {
  render(<Markdown text={'Description | Nm | lb-ft\nCaliper bolt | 115 | 85'} />)
  expect(screen.getByRole('table')).toHaveTextContent('Caliper bolt')
})
