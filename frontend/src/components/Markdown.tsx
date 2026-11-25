import { Fragment, type ReactNode } from 'react'
import { TriangleAlert, Info } from 'lucide-react'
import { admonitionKind, parseBlocks } from './markdownParse'

/**
 * Minimal, safe Markdown renderer for assistant answers.
 * Produces React elements only (never injects HTML). Supports headings, bold-only lines,
 * numbered steps (manual numbering preserved), bullets with indentation, pipe tables,
 * inline **bold** / *italic* / `code`, and [S1] citation chips. Lines containing
 * WARNING/CAUTION/DANGER or NOTICE/NOTE are rendered as callouts.
 */

const INLINE_RE = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|_[^_\s][^_]*_|`[^`]+`|\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\])/g

function Inline({ text, onCite }: { text: string; onCite?: (id: string) => void }) {
  const parts = text.split(INLINE_RE).filter((p) => p !== '')
  return (
    <>
      {parts.map((p, i) => {
        if (p.startsWith('**') && p.endsWith('**')) return <strong key={i}>{p.slice(2, -2)}</strong>
        if (p.startsWith('`')) return <code key={i} className="rounded bg-slate-100 px-1 text-[0.9em]">{p.slice(1, -1)}</code>
        if (/^\[\s*S\d+/.test(p)) {
          const ids = p.replace(/[[\]\s]/g, '').split(/[,;]/)
          return (
            <span key={i} className="whitespace-nowrap">
              {ids.map((id) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => onCite?.(id)}
                  className="mx-0.5 inline-flex items-center rounded bg-sky-100 px-1.5 py-px align-baseline text-xs font-semibold text-sky-800 hover:bg-sky-200 focus:outline-none focus:ring-2 focus:ring-sky-500"
                  aria-label={`Show source ${id}`}
                >
                  {id}
                </button>
              ))}
            </span>
          )
        }
        if ((p.startsWith('*') && p.endsWith('*')) || (p.startsWith('_') && p.endsWith('_')))
          return <em key={i}>{p.slice(1, -1)}</em>
        return <Fragment key={i}>{p}</Fragment>
      })}
    </>
  )
}

function Callout({ kind, children }: { kind: 'danger' | 'notice'; children: ReactNode }) {
  const danger = kind === 'danger'
  return (
    <div
      role={danger ? 'alert' : 'note'}
      className={`my-1.5 flex gap-2 rounded-md border-l-4 px-3 py-2 ${
        danger ? 'border-red-600 bg-red-50 text-red-950' : 'border-amber-500 bg-amber-50 text-amber-950'
      }`}
    >
      {danger ? (
        <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0 text-red-600" aria-hidden />
      ) : (
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" aria-hidden />
      )}
      <div>{children}</div>
    </div>
  )
}

export function Markdown({ text, onCite }: { text: string; onCite?: (id: string) => void }) {
  const blocks = parseBlocks(text)
  return (
    <div className="space-y-2 leading-relaxed">
      {blocks.map((b, i) => {
        switch (b.kind) {
          case 'heading':
            return (
              <h4 key={i} className="pt-1 text-[0.95rem] font-semibold text-slate-900">
                <Inline text={b.text} onCite={onCite} />
              </h4>
            )
          case 'para': {
            const k = admonitionKind(b.text)
            const body = <Inline text={b.text} onCite={onCite} />
            return k ? <Callout key={i} kind={k}>{body}</Callout> : <p key={i}>{body}</p>
          }
          case 'ol':
            return (
              <ol key={i} className="space-y-1.5">
                {b.items.map((it, j) => {
                  const k = admonitionKind(it.text)
                  const body = <Inline text={it.text} onCite={onCite} />
                  return (
                    <li key={j} className="flex gap-2.5" style={{ marginLeft: it.depth * 20 }}>
                      <span className="mt-0.5 flex h-6 min-w-6 items-center justify-center rounded-full bg-slate-800 px-1.5 text-xs font-bold text-white">
                        {it.n}
                      </span>
                      <div className="flex-1 pt-0.5">{k ? <Callout kind={k}>{body}</Callout> : body}</div>
                    </li>
                  )
                })}
              </ol>
            )
          case 'ul':
            return (
              <ul key={i} className="space-y-1">
                {b.items.map((it, j) => {
                  const k = admonitionKind(it.text)
                  const body = <Inline text={it.text} onCite={onCite} />
                  return k ? (
                    <li key={j} style={{ marginLeft: it.depth * 20 }}>
                      <Callout kind={k}>{body}</Callout>
                    </li>
                  ) : (
                    <li key={j} className="flex gap-2" style={{ marginLeft: 8 + it.depth * 20 }}>
                      <span className="mt-2.5 h-1.5 w-1.5 shrink-0 rounded-full bg-slate-500" aria-hidden />
                      <span>{body}</span>
                    </li>
                  )
                })}
              </ul>
            )
          case 'table':
            return (
              <div key={i} className="overflow-x-auto">
                <table className="min-w-full border-collapse text-sm">
                  <tbody>
                    {b.rows.map((r, j) => (
                      <tr key={j} className={j === 0 ? 'bg-slate-100 font-semibold' : 'border-t border-slate-200'}>
                        {r.map((c, k) => (
                          <td key={k} className="px-2 py-1 align-top">
                            <Inline text={c} onCite={onCite} />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
        }
      })}
    </div>
  )
}
