/** Block parsing for the safe Markdown renderer (kept separate for fast refresh + unit tests). */

export type Block =
  | { kind: 'heading'; text: string; level: number }
  | { kind: 'para'; text: string }
  | { kind: 'ol'; items: { n: number; text: string; depth: number }[] }
  | { kind: 'ul'; items: { text: string; depth: number }[] }
  | { kind: 'table'; rows: string[][] }

const OL_RE = /^(\s*)(\d{1,3})\.\s+(.*)$/
const UL_RE = /^(\s*)[-*•]\s+(.*)$/
const H_RE = /^(#{1,4})\s+(.*)$/
const BOLD_LINE_RE = /^\*\*([^*]+)\*\*:?\s*$/

export function parseBlocks(src: string): Block[] {
  const blocks: Block[] = []
  const lines = src.replace(/\r/g, '').split('\n')
  let para: string[] = []
  const flush = () => {
    if (para.length) blocks.push({ kind: 'para', text: para.join(' ') })
    para = []
  }
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, '')
    if (!line.trim()) {
      flush()
      continue
    }
    if (line.includes(' | ') || /^\s*\|.*\|\s*$/.test(line)) {
      flush()
      if (/^\s*\|?\s*:?-{2,}/.test(line)) continue
      const cells = line.replace(/^\s*\|/, '').replace(/\|\s*$/, '').split('|').map((c) => c.trim())
      const last = blocks.at(-1)
      if (last?.kind === 'table') last.rows.push(cells)
      else blocks.push({ kind: 'table', rows: [cells] })
      continue
    }
    let m: RegExpMatchArray | null
    if ((m = line.match(H_RE))) {
      flush()
      blocks.push({ kind: 'heading', text: m[2], level: m[1].length })
    } else if ((m = line.match(BOLD_LINE_RE))) {
      flush()
      blocks.push({ kind: 'heading', text: m[1], level: 4 })
    } else if ((m = line.match(OL_RE))) {
      flush()
      const item = { n: Number(m[2]), text: m[3], depth: Math.floor(m[1].length / 2) }
      const last = blocks.at(-1)
      if (last?.kind === 'ol') last.items.push(item)
      else blocks.push({ kind: 'ol', items: [item] })
    } else if ((m = line.match(UL_RE))) {
      flush()
      const item = { text: m[2], depth: Math.min(3, Math.floor(m[1].length / 2)) }
      const last = blocks.at(-1)
      if (last?.kind === 'ul') last.items.push(item)
      else blocks.push({ kind: 'ul', items: [item] })
    } else {
      para.push(line.trim())
    }
  }
  flush()
  return blocks
}


export function admonitionKind(text: string): 'danger' | 'notice' | null {
  if (/\b(WARNING|CAUTION|DANGER)\b\s*:?/.test(text)) return 'danger'
  if (/\b(NOTICE|NOTE)\b\s*:/.test(text)) return 'notice'
  return null
}

